"""Detection parameter sweep and L-curves (``openmeq diagnostics detect-sweep``, OpenMEQ-cl6).

Runs the ``openmeq detect`` coincidence STA/LTA over a grid of parameters on
a few sample days (or, for triggered archives, the recording windows of
``scripts/detect_windows.py``) and draws L-curves for choosing
``cfg.detection``. Nothing under ``2_detection/`` other than ``sweep/`` is
written; ``detections.csv`` is never touched.

Grid (``cfg.detection.sweep``)
------------------------------
The expensive part of detection is reading, filtering and the STA/LTA
characteristic function (CF); thresholds applied to a CF are cheap. So:

- one work item per (sample day, filter band): the day is read and
  preprocessed once (:func:`openmeq.detect.worker.preprocess_stream`, the
  detector's own path);
- per (``sta``, ``lta = sta * lta_factor``), each trace's CF is computed
  once (:func:`openmeq.detect.worker.characteristic_function`) and every
  (``trigger_on``, ``trigger_off``) pair is applied to it;
- per (``trigger_on``, ``trigger_off``), the coincidence candidates are
  built once and every ``min_stations`` value (with ``min_trace_coincidence
  = ceil(min_stations * trace_coincidence_per_station)``) is a cheap pass
  over them.

**Exactness.** :func:`fast_trigger_onset` and :func:`coincidence_candidates`
/ :func:`select_events` reproduce ObsPy's ``trigger_onset`` and
``coincidence_trigger`` exactly (``tests/test_detect_sweep.py`` compares
them), so a sweep point equals an ``openmeq detect`` run with those values
up to the dedup step: here triggers are kept when their on-time is inside
the sample day (or after ``edge_s`` in a recording window) and de-duplicated
within it by :func:`dedup_times` (the rule of
:func:`openmeq.detect.merge._cluster_triggers`).

:func:`fast_trigger_onset` only runs ObsPy's ``trigger_onset`` on the parts
of the CF that can trigger: the runs where the CF stays at or above the
lowest ``trigger_off`` and reaches the lowest ``trigger_on``, joined by a
sentinel sample below every off threshold. A trigger's on and off samples
always lie inside one such run, and the sentinel resets the on/off state
exactly as the CF falling below ``trigger_off`` does.

**Saturation.** A (``trigger_on``, ``trigger_off``) pair that yields more
than ``max_station_triggers`` single-trace triggers in one item is clearly
in the noise regime; it is recorded with ``saturated = True`` and no
detection count instead of running the (pure-Python) coincidence pass.

L-curves
--------
- **No reference (always):** detections per day against ``trigger_on``
  (log-log), one curve per (filter, sta, lta, trigger_off, min_stations).
  Noise triggers fall off steeply as ``trigger_on`` rises, then the count
  flattens out on the real events. The knee is the point farthest below the
  chord joining the curve's ends in normalised log-log space
  (:func:`knee_index`); its distance (``knee_strength``) ranks the curves --
  a sharp knee means noise and events are well separated. ``persist_frac``
  is the fraction of a point's detections still detected at the next higher
  ``trigger_on`` (within ``dedup_window_s / 2``).
- **Reference catalogue (``sweep.reference``):** recall against extra
  (unmatched) detections per day. A detection matches a reference event if
  ``det - ref`` is inside ``match_window_s`` (one-to-one, in time order).
  Reported per curve: the point closest to (recall 1, 0 extras) with extras
  normalised by the reference rate, and overall the best F1.

Outputs (``2_detection/sweep/``)
--------------------------------
``sweep.csv`` (one row per grid point, summed over the sample items),
``knees.csv`` (one row per curve), ``candidates.csv`` (best grid points),
``items.csv`` (the sample days/windows) and ``lcurve_*.png`` /
``recall_*.png`` (one figure per filter band and ``trigger_off``; panels
sta x lta, one line per ``min_stations``).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset

from openmeq.catalog import iso_z
from openmeq.config import Config, DetectionFilterConfig
from openmeq.detect.worker import (
    _COINCIDENCE_MAX_TRIGGER_LENGTH_S,
    _build_trace_ids,
    _cached_cfg,
    _cached_index,
    _read_window,
    characteristic_function,
    preprocess_stream,
)
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.paths import RunPaths

logger = get_logger("detect.sweep")

#: CF value joining the kept runs in :func:`fast_trigger_onset`; below every off threshold.
_SENTINEL = -1.0


# ---------------------------------------------------------------------------
# Triggering (exact replicas of ObsPy, made cheap to repeat)
# ---------------------------------------------------------------------------


def compress_cf(cft: np.ndarray, min_off: float, min_on: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the CF runs that can trigger (see module docstring).

    Returns ``(compressed, positions)``: the runs where ``cft >= min_off``
    whose maximum is ``>= min_on``, each followed by one :data:`_SENTINEL`
    sample, and each compressed sample's index in ``cft`` (``-1`` for a
    sentinel).
    """
    above = cft >= min_off
    if not above.any():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    peaks = np.maximum.reduceat(cft, starts)
    keep = peaks >= min_on
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    lengths = ends - starts + 1
    total = int(lengths.sum())
    positions = np.full(total, -1, dtype=np.int64)
    run_offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    # positions[run_offset + k] = start + k for k < run length; sentinel stays -1
    idx = np.arange(total) - np.repeat(run_offsets, lengths)
    is_data = idx < np.repeat(lengths - 1, lengths)
    positions[is_data] = (np.repeat(starts, lengths) + idx)[is_data]
    compressed = np.full(total, _SENTINEL, dtype=np.float64)
    compressed[is_data] = cft[positions[is_data]]
    return compressed, positions


def fast_trigger_onset(
    compressed: np.ndarray, positions: np.ndarray, thr_on: float, thr_off: float, max_len: int
) -> np.ndarray:
    """``trigger_onset(cft, thr_on, thr_off, max_len, max_len_delete=False)`` on a
    :func:`compress_cf` result; ``(n, 2)`` on/off sample indices into the full CF."""
    if len(compressed) == 0:
        return np.empty((0, 2), dtype=np.int64)
    picks = trigger_onset(compressed, thr_on, thr_off, max_len=max_len, max_len_delete=False)
    if len(picks) == 0:
        return np.empty((0, 2), dtype=np.int64)
    picks = np.asarray(picks, dtype=np.int64)
    # An "off" never lands on a sentinel (it is the last sample >= thr_off of a run).
    return positions[picks]


def coincidence_candidates(
    triggers: list[tuple[float, float, str]], weights: dict[str, float] | None
) -> list[tuple[float, float, tuple[str, ...], float]]:
    """ObsPy ``coincidence_trigger``'s event-building loop, before its thresholds.

    ``triggers`` are ``(on, off, trace_id)`` (epoch s), sorted. Returns one
    ``(on, off, trace_ids, coincidence_sum)`` per trigger, in order -- what
    ``coincidence_trigger`` builds for each trigger it pops, before testing
    ``thr_coincidence_sum`` and the shared-off-time subset rule
    (:func:`select_events` applies those).
    """
    out = []
    n = len(triggers)
    for i in range(n):
        on, off, tr_id = triggers[i]
        ids = [tr_id]
        seen = {tr_id}
        csum = weights[tr_id] if weights else 1.0
        for j in range(i + 1, n):
            tmp_on, tmp_off, tmp_id = triggers[j]
            if tmp_id in seen:
                continue
            if tmp_on > off:
                break
            ids.append(tmp_id)
            seen.add(tmp_id)
            csum += weights[tmp_id] if weights else 1.0
            off = max(off, tmp_off)
        out.append((on, off, tuple(ids), csum))
    return out


def select_events(
    candidates: list[tuple[float, float, tuple[str, ...], float]],
    thr_coincidence_sum: float,
    min_stations: int,
) -> list[tuple[float, float, int]]:
    """``coincidence_trigger``'s thresholds plus ``openmeq detect``'s station filter.

    Returns ``(time, coincidence_sum, n_stations)`` per surviving event.
    """
    events = []
    last_off = 0.0
    for on, off, ids, csum in candidates:
        if csum < thr_coincidence_sum:
            continue
        if off <= last_off:
            continue
        last_off = off
        n_sta = len({".".join(tid.split(".")[:2]) for tid in ids})
        if n_sta >= min_stations:
            events.append((on, csum, n_sta))
    return events


def dedup_times(events: list[tuple[float, float, int]], dedup_window_s: float) -> np.ndarray:
    """:func:`openmeq.detect.merge._cluster_triggers` on ``(time, coincidence_sum, ...)``
    tuples; returns each cluster's representative time, sorted."""
    events = sorted(events, key=lambda e: e[0])
    reps: list[float] = []
    best: tuple[float, float] | None = None  # (-csum, time) of the current cluster's representative
    for t, csum, *_ in events:
        if best is not None and t - best[1] <= dedup_window_s:
            best = min(best, (-csum, t))
        else:
            if best is not None:
                reps.append(best[1])
            best = (-csum, t)
    if best is not None:
        reps.append(best[1])
    return np.asarray(sorted(reps), dtype=np.float64)


def match_count(det: np.ndarray, ref: np.ndarray, window: tuple[float, float]) -> int:
    """One-to-one matches: each reference event (in time order) takes the
    earliest unused detection with ``det - ref`` in ``window``."""
    lo, hi = window
    j = 0
    n = 0
    for r in ref:
        while j < len(det) and det[j] - r < lo:
            j += 1
        if j < len(det) and det[j] - r <= hi:
            n += 1
            j += 1
    return n


def _persist_count(det: np.ndarray, det_next: np.ndarray, tol: float) -> int:
    """Detections in ``det`` with one in ``det_next`` within ``tol`` s."""
    if len(det) == 0 or len(det_next) == 0:
        return 0
    k = np.searchsorted(det_next, det)
    left = np.abs(det - det_next[np.clip(k - 1, 0, len(det_next) - 1)])
    right = np.abs(det_next[np.clip(k, 0, len(det_next) - 1)] - det)
    return int((np.minimum(left, right) <= tol).sum())


# ---------------------------------------------------------------------------
# Grid and work items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepGrid:
    filters: tuple[tuple[float, float], ...]
    sta: tuple[float, ...]
    lta_factor: tuple[float, ...]
    trigger_on: tuple[float, ...]
    trigger_off: tuple[float, ...]
    min_stations: tuple[int, ...]
    trace_coincidence_per_station: float
    max_station_triggers: int

    def thr_pairs(self) -> list[tuple[float, float]]:
        """(trigger_on, trigger_off) pairs with off < on."""
        return [(on, off) for off in self.trigger_off for on in self.trigger_on if off < on]

    def min_trace_coincidence(self, min_stations: int) -> int:
        return math.ceil(min_stations * self.trace_coincidence_per_station - 1e-9)


def default_trigger_on() -> list[float]:
    return [round(float(v), 2) for v in np.geomspace(2.5, 12.0, 15)]


def build_grid(cfg: Config, n_stations: int) -> SweepGrid:
    s = cfg.detection.sweep
    min_sta = s.min_stations or list(range(2, max(2, min(n_stations, 12)) + 1))
    return SweepGrid(
        filters=tuple((float(a), float(b)) for a, b in s.filters),
        sta=tuple(s.sta),
        lta_factor=tuple(s.lta_factor),
        trigger_on=tuple(sorted(s.trigger_on or default_trigger_on())),
        trigger_off=tuple(sorted(s.trigger_off)),
        min_stations=tuple(sorted(set(min_sta))),
        trace_coincidence_per_station=s.trace_coincidence_per_station,
        max_station_triggers=s.max_station_triggers,
    )


@dataclass(frozen=True)
class SweepItem:
    """One sample span; detections are counted for on-times in ``[t1 + edge_s, t2)``."""

    label: str
    t1: str
    t2: str
    pad_s: float
    edge_s: float

    @property
    def searched_s(self) -> float:
        return float(UTCDateTime(self.t2) - UTCDateTime(self.t1)) - self.edge_s


def _day_channel_counts(index: DataIndex, start, end, exclude: set[str]) -> pd.Series:
    """Channels with any data, per UTC day in ``[start, end]``."""
    counts: dict[str, set[str]] = defaultdict(set)
    first = UTCDateTime(start)
    last = UTCDateTime(end) + 86400
    for sid in index.seed_ids():
        if sid in exclude:
            continue
        for _path, a, b, _sr in index.channels[sid]:
            ta, tb = max(UTCDateTime(a), first), min(UTCDateTime(b), last)
            day = UTCDateTime(ta.date)
            while day < tb:
                counts[day.date.isoformat()].add(sid)
                day += 86400
    return pd.Series({d: len(v) for d, v in counts.items()}, dtype=int).sort_index()


def choose_days(counts: pd.Series, n_days: int) -> list[str]:
    """``n_days`` days spread evenly in time over those with >= 80% of the best channel count."""
    if counts.empty:
        return []
    good = counts[counts >= 0.8 * counts.max()].index.tolist()
    if len(good) <= n_days:
        return good
    picks = np.linspace(0, len(good) - 1, n_days).round().astype(int)
    return [good[i] for i in sorted(set(picks))]


def build_items(cfg: Config, index: DataIndex, max_lta: float) -> list[SweepItem]:
    s = cfg.detection.sweep
    if s.windows_csv is not None:
        windows = pd.read_csv(s.windows_csv)
        if s.max_windows is not None and len(windows) > s.max_windows:
            pick = np.linspace(0, len(windows) - 1, s.max_windows).round().astype(int)
            windows = windows.iloc[sorted(set(pick))]
        edge = 2.0 * max_lta if s.edge_s is None else s.edge_s
        items = [
            SweepItem(f"w{i:05d}", str(r.start), str(r.end), 0.0, edge)
            for i, r in enumerate(windows.itertuples())
            if float(UTCDateTime(r.end) - UTCDateTime(r.start)) > edge
        ]
        if len(items) < len(windows):
            logger.warning(
                "%d of %d recording window(s) are shorter than edge_s=%.1f s and are skipped",
                len(windows) - len(items),
                len(windows),
                edge,
            )
        return items

    days = [d.isoformat() for d in s.days]
    if not days:
        counts = _day_channel_counts(
            index, cfg.data.start, cfg.data.end, set(cfg.qc.exclude_channels)
        )
        days = choose_days(counts, s.n_days)
    pad = max(cfg.detection.day_overlap_s, 3.0 * max_lta)
    return [SweepItem(d, f"{d}T00:00:00Z", iso_z(UTCDateTime(d) + 86400), pad, 0.0) for d in days]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def sweep_item(task: tuple) -> list[dict]:
    """One (item, filter band) -> one row per (sta, lta, on, off, min_stations).

    Top-level and picklable (``openmeq.parallel.pmap``). ``task`` is
    ``(item, (freqmin, freqmax), grid, index_path, cfg_path, ref_times)``,
    ``ref_times`` being the reference event times (epoch s, sorted) inside
    the item or ``None``.
    """
    item, (fmin, fmax), grid, index_path, cfg_path, ref_times = task
    t0 = time.perf_counter()
    cfg = _cached_cfg(cfg_path)
    index = _cached_index(index_path)
    det = cfg.detection.model_copy(
        update={"filter": DetectionFilterConfig(type="bandpass", freqmin=fmin, freqmax=fmax)}
    )
    t1, t2 = UTCDateTime(item.t1), UTCDateTime(item.t2)
    lo, hi = (t1 + item.edge_s).timestamp, t2.timestamp
    stream = _read_window(index, cfg.qc.exclude_channels, t1, t2, pad=item.pad_s)
    stream = preprocess_stream(stream, det) if len(stream) else stream
    weights = _build_trace_ids(stream, det.station_weights)
    pairs = grid.thr_pairs()
    min_off = min(off for _, off in pairs)
    min_on = min(on for on, _ in pairs)
    sweep_cfg = cfg.detection.sweep
    tol = det.dedup_window_s / 2.0
    n_ref = 0 if ref_times is None else len(ref_times)

    rows: list[dict] = []
    for sta in grid.sta:
        for factor in grid.lta_factor:
            lta = sta * factor
            cf_cfg = SimpleNamespace(sta=sta, lta=lta)
            per_pair: dict[tuple[float, float], list] = {p: [] for p in pairs}
            for tr in stream:
                sr = tr.stats.sampling_rate
                cft = characteristic_function(tr, cf_cfg)
                comp, pos = compress_cf(cft, min_off, min_on)
                max_len = int(_COINCIDENCE_MAX_TRIGGER_LENGTH_S * sr + 0.5)
                start = tr.stats.starttime.timestamp
                for p in pairs:
                    for on_i, off_i in fast_trigger_onset(comp, pos, p[0], p[1], max_len):
                        per_pair[p].append((start + on_i / sr, start + off_i / sr, tr.id))
            # det times per (off, m) family in on order, for persistence
            family_times: dict[tuple[float, int], list[tuple[float, np.ndarray | None]]] = (
                defaultdict(list)
            )
            for (on, off), trigs in per_pair.items():
                base = {
                    "item": item.label,
                    "freqmin": fmin,
                    "freqmax": fmax,
                    "sta": sta,
                    "lta": lta,
                    "lta_factor": factor,
                    "trigger_on": on,
                    "trigger_off": off,
                    "n_station_triggers": len(trigs),
                }
                saturated = len(trigs) > grid.max_station_triggers
                cands = None
                if not saturated:
                    trigs.sort()
                    cands = coincidence_candidates(trigs, weights)
                for m in grid.min_stations:
                    mtc = grid.min_trace_coincidence(m)
                    row = {**base, "min_stations": m, "min_trace_coincidence": mtc}
                    if saturated:
                        times = None
                        row.update(saturated=True, n_det=np.nan, n_matched=np.nan)
                    else:
                        events = [e for e in select_events(cands, mtc, m) if lo <= e[0] < hi]
                        times = dedup_times(events, det.dedup_window_s)
                        row.update(saturated=False, n_det=len(times), n_matched=np.nan)
                        if ref_times is not None:
                            row["n_matched"] = match_count(
                                times, ref_times, tuple(sweep_cfg.match_window_s)
                            )
                    row["n_ref"] = n_ref
                    rows.append(row)
                    family_times[(off, m)].append((on, times, len(rows) - 1))
            for fam in family_times.values():
                fam.sort(key=lambda x: x[0])
                for (_, times, i), (_, nxt, _) in zip(
                    fam, fam[1:] + [(None, None, None)], strict=True
                ):
                    rows[i]["n_persist"] = (
                        _persist_count(times, nxt, tol)
                        if times is not None and nxt is not None
                        else np.nan
                    )
    logger.info(
        "sweep %s %g-%g Hz: %d trace(s), %d grid point(s) (%.1fs)",
        item.label,
        fmin,
        fmax,
        len(stream),
        len(rows),
        time.perf_counter() - t0,
    )
    return rows


# ---------------------------------------------------------------------------
# Aggregation and knees
# ---------------------------------------------------------------------------

_POINT_KEYS = [
    "freqmin",
    "freqmax",
    "sta",
    "lta_factor",
    "lta",
    "trigger_off",
    "min_stations",
    "min_trace_coincidence",
    "trigger_on",
]
_CURVE_KEYS = _POINT_KEYS[:-1]


def aggregate(rows: pd.DataFrame, items: list[SweepItem]) -> pd.DataFrame:
    """Sum the per-item rows into one row per grid point, with per-day rates."""
    days = sum(it.searched_s for it in items) / 86400.0
    g = rows.groupby(_POINT_KEYS, sort=True)
    out = g.agg(
        n_station_triggers=("n_station_triggers", "sum"),
        saturated=("saturated", "any"),
        n_det=("n_det", "sum"),
        n_matched=("n_matched", "sum"),
        n_persist=("n_persist", "sum"),
        n_ref=("n_ref", "sum"),
        persist_valid=("n_persist", lambda s: s.notna().all()),
    ).reset_index()
    out.loc[out["saturated"], ["n_det", "n_matched", "n_persist"]] = np.nan
    out.loc[~out["persist_valid"], "n_persist"] = np.nan
    out = out.drop(columns="persist_valid")
    out["sample_days"] = days
    out["det_per_day"] = out["n_det"] / days
    out["persist_frac"] = out["n_persist"] / out["n_det"].where(out["n_det"] > 0)
    if (out["n_ref"] > 0).any():
        out["recall"] = out["n_matched"] / out["n_ref"].where(out["n_ref"] > 0)
        out["extras_per_day"] = (out["n_det"] - out["n_matched"]) / days
        prec = out["n_matched"] / out["n_det"].where(out["n_det"] > 0)
        out["precision"] = prec
        out["f1"] = (
            2 * prec * out["recall"] / (prec + out["recall"]).where(prec + out["recall"] > 0)
        )
    else:
        out = out.drop(columns=["n_matched", "n_ref"])
    return out


def knee_index(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Index of the point farthest below the chord of a decreasing curve.

    ``x``/``y`` are already in the space the knee is judged in (log-log
    here); both are normalised to [0, 1] first. Returns ``(index, distance)``
    with ``index = -1`` when the curve has fewer than 3 points or no point
    lies below the chord.
    """
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return -1, 0.0
    xn = (x - x.min()) / np.ptp(x)
    yn = (y - y.min()) / np.ptp(y)
    chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / (xn[-1] - xn[0])
    below = chord - yn
    i = int(np.argmax(below))
    return (i, float(below[i])) if below[i] > 0 else (-1, 0.0)


def find_knees(points: pd.DataFrame) -> pd.DataFrame:
    """One row per curve: the no-reference knee and, with a reference, the
    point closest to (recall 1, 0 extras)."""
    has_ref = "recall" in points.columns
    rows = []
    for key, curve in points.groupby(_CURVE_KEYS, sort=True):
        curve = curve.sort_values("trigger_on")
        row = dict(zip(_CURVE_KEYS, key, strict=True))
        ok = curve[~curve["saturated"] & (curve["n_det"] > 0)]
        i, dist = knee_index(
            np.log10(ok["trigger_on"].to_numpy()), np.log10(ok["det_per_day"].to_numpy())
        )
        knee = ok.iloc[i] if i >= 0 else None
        row.update(
            n_points=len(ok),
            knee_trigger_on=np.nan if knee is None else knee["trigger_on"],
            knee_det_per_day=np.nan if knee is None else knee["det_per_day"],
            knee_persist_frac=np.nan if knee is None else knee["persist_frac"],
            knee_strength=dist,
        )
        if has_ref:
            r = curve[~curve["saturated"] & curve["recall"].notna()]
            ref_per_day = r["n_ref"].iloc[0] / r["sample_days"].iloc[0] if len(r) else np.nan
            if len(r) and ref_per_day > 0:
                d = np.hypot(1.0 - r["recall"], r["extras_per_day"] / ref_per_day)
                best = r.loc[d.idxmin()]
                row.update(
                    ref_trigger_on=best["trigger_on"],
                    ref_recall=best["recall"],
                    ref_extras_per_day=best["extras_per_day"],
                    ref_distance=float(d.min()),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def candidates(points: pd.DataFrame, knees: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Best grid points: by F1 with a reference, else the sharpest knees."""
    if "f1" in points.columns:
        best = points[~points["saturated"]].sort_values(["f1", "recall"], ascending=False).head(n)
        return best.assign(rank_by="f1")
    k = knees[knees["knee_strength"] > 0].sort_values("knee_strength", ascending=False).head(n)
    return k.assign(rank_by="knee_strength")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _panel_grid(points: pd.DataFrame, knees: pd.DataFrame, out_dir: Path, mode: str) -> list[Path]:
    import matplotlib.pyplot as plt

    from openmeq.plotting import save_fig, setup_style

    setup_style()
    written = []
    stas = sorted(points["sta"].unique())
    factors = sorted(points["lta_factor"].unique())
    ms = sorted(points["min_stations"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(i / max(1, len(ms) - 1)) for i, m in enumerate(ms)}
    for (fmin, fmax, off), sub in points.groupby(["freqmin", "freqmax", "trigger_off"]):
        fig, axes = plt.subplots(
            len(stas),
            len(factors),
            figsize=(3.6 * len(factors) + 1.2, 2.8 * len(stas) + 0.8),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for r, sta in enumerate(stas):
            for c, fac in enumerate(factors):
                ax = axes[r][c]
                panel = sub[(sub["sta"] == sta) & (sub["lta_factor"] == fac)]
                for m in ms:
                    cur = panel[(panel["min_stations"] == m) & ~panel["saturated"]].sort_values(
                        "trigger_on"
                    )
                    if mode == "lcurve":
                        cur = cur[cur["n_det"] > 0]
                        ax.plot(
                            cur["trigger_on"],
                            cur["det_per_day"],
                            "-o",
                            ms=2.5,
                            lw=1,
                            color=colors[m],
                            label=f"{m}",
                        )
                        k = knees[
                            (knees["freqmin"] == fmin)
                            & (knees["freqmax"] == fmax)
                            & (knees["trigger_off"] == off)
                            & (knees["sta"] == sta)
                            & (knees["lta_factor"] == fac)
                            & (knees["min_stations"] == m)
                        ]
                        if len(k) and np.isfinite(k["knee_trigger_on"].iloc[0]):
                            ax.plot(
                                k["knee_trigger_on"],
                                k["knee_det_per_day"],
                                "x",
                                color="k",
                                ms=6,
                                mew=1.5,
                            )
                    else:
                        ax.plot(
                            cur["extras_per_day"],
                            cur["recall"],
                            "-o",
                            ms=2.5,
                            lw=1,
                            color=colors[m],
                            label=f"{m}",
                        )
                    sat = panel[(panel["min_stations"] == m) & panel["saturated"]]
                    if mode == "lcurve" and len(sat):
                        ax.axvspan(
                            sat["trigger_on"].min(),
                            sat["trigger_on"].max(),
                            color="0.9",
                            lw=0,
                            zorder=0,
                        )
                if mode == "lcurve":
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                else:
                    ax.set_xscale("symlog", linthresh=1.0)
                    ax.set_ylim(-0.02, 1.02)
                ax.set_title(f"sta {sta:g} s, lta {sta * fac:g} s (x{fac:g})", fontsize=9)
                if r == len(stas) - 1:
                    ax.set_xlabel("trigger_on" if mode == "lcurve" else "extra detections / day")
                if c == 0:
                    ax.set_ylabel("detections / day" if mode == "lcurve" else "recall")
        axes[0][-1].legend(title="min_stations", fontsize=7, title_fontsize=7, loc="best")
        what = (
            "Detections vs trigger_on (x = knee; grey = saturated)"
            if mode == "lcurve"
            else "Recall vs extra detections"
        )
        fig.suptitle(f"{what} -- bandpass {fmin:g}-{fmax:g} Hz, trigger_off {off:g}")
        fig.tight_layout()
        path = out_dir / f"{mode}_{fmin:g}-{fmax:g}Hz_off{off:g}.png"
        save_fig(fig, path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_reference(path: Path, column: str) -> np.ndarray:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(
            f"reference catalogue {path} has no {column!r} column (have {list(df.columns)})"
        )
    times = pd.to_datetime(df[column], utc=True, format="ISO8601")
    return np.sort(times.astype("int64").to_numpy() / 1e9)


def run_sweep(cfg: Config, paths: RunPaths, config_path: str) -> Path:
    """Run the sweep and write ``2_detection/sweep/``; returns that directory."""
    s = cfg.detection.sweep
    if not paths.index_json.is_file():
        raise FileNotFoundError(f"{paths.index_json} not found -- run `openmeq qc` first")
    index = DataIndex.load(paths.index_json)
    exclude = set(cfg.qc.exclude_channels)
    n_stations = len(
        {".".join(sid.split(".")[:2]) for sid in index.seed_ids() if sid not in exclude}
    )
    grid = build_grid(cfg, n_stations)
    max_lta = max(grid.sta) * max(grid.lta_factor)
    items = build_items(cfg, index, max_lta)
    if not items:
        raise ValueError("detect-sweep: no sample days/windows with data")
    out_dir = paths.step_dir("detections") / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(it) | {"searched_s": it.searched_s} for it in items]).to_csv(
        out_dir / "items.csv", index=False
    )

    ref = load_reference(s.reference, s.reference_time_column) if s.reference else None

    def _ref_in(it: SweepItem):
        if ref is None:
            return None
        lo = (UTCDateTime(it.t1) + it.edge_s).timestamp
        hi = UTCDateTime(it.t2).timestamp
        return ref[(ref >= lo) & (ref < hi)]

    tasks = [
        (it, f, grid, str(paths.index_json), str(config_path), _ref_in(it))
        for it in items
        for f in grid.filters
    ]
    n_points = (
        len(grid.sta)
        * len(grid.lta_factor)
        * len(grid.thr_pairs())
        * len(grid.min_stations)
        * len(grid.filters)
    )
    logger.info(
        "detect-sweep: %d item(s) (%.2f day(s) searched) x %d filter(s); %d grid point(s); %s",
        len(items),
        sum(it.searched_s for it in items) / 86400,
        len(grid.filters),
        n_points,
        f"reference {s.reference} ({0 if ref is None else len(ref)} events)"
        if ref is not None
        else "no reference",
    )
    pcfg = SimpleNamespace(
        n_workers=s.n_workers, threads_per_worker=cfg.parallel.threads_per_worker
    )
    results = pmap(sweep_item, tasks, pcfg, desc="detect-sweep", ordered=False)
    rows = pd.DataFrame([r for res in results for r in res])
    rows.to_csv(out_dir / "sweep_items.csv", index=False)

    points = aggregate(rows, items)
    points.to_csv(out_dir / "sweep.csv", index=False)
    knees = find_knees(points)
    knees.to_csv(out_dir / "knees.csv", index=False)
    cands = candidates(points, knees)
    cands.to_csv(out_dir / "candidates.csv", index=False)

    _panel_grid(points, knees, out_dir, "lcurve")
    if "recall" in points.columns:
        _panel_grid(points, knees, out_dir, "recall")
    logger.info("detect-sweep: wrote %s", out_dir)
    return out_dir
