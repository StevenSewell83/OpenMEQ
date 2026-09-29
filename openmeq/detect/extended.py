"""Opt-in extended detection QAQC (``OpenMEQ-02o.2``): per-station trigger
table, station x hour trigger-rate heatmap, and characteristic-function
example panels.

Everything here runs only when ``cfg.diagnostics.extended_enabled("detect")``
is true (:func:`openmeq.detect.pipeline.run_detect` gates it); the always-on
baseline figures stay in :mod:`openmeq.detect.diagnostics`, untouched.

Outputs
-------
- ``detections/station_triggers.csv`` (:data:`STATION_TRIGGERS_COLUMNS`):
  every single-station STA/LTA trigger the detector saw, one row per
  ``(seed_id, on_time)``, as returned by
  :func:`openmeq.detect.worker.detect_window_with_station_triggers` -- not
  just the ones that took part in a coincidence detection. This is the raw
  material for "which station is noisy / dead / over-triggering" questions
  the merged ``detections.csv`` cannot answer.
- ``detections/diagnostics/extended/station_trigger_rate.png``: station
  (``NET.STA``) x hour heatmap of station-trigger counts
  (:func:`~openmeq.plotting.plot_heatmap`). For a run of at most
  :data:`ABSOLUTE_HOURS_MAX_DAYS` days, one column per absolute UTC hour; for
  a longer run (where that would mean thousands of columns) one column per
  hour-of-day, holding the mean triggers per hour across the run's days --
  which is also the view that exposes diurnal (cultural-noise) patterns.
  Every station in ``inventory/index.json`` gets a row, so a station that
  never triggered shows up as an explicit zero row rather than vanishing.
- ``detections/diagnostics/extended/cf_examples_<nn>.png`` (``nn`` = 01, 02,
  ...): up to ``cfg.diagnostics.n_examples`` detections, evenly sampled in
  time order, followed by :data:`N_QUIET_WINDOWS` randomly drawn windows
  with no detection (fixed RNG seed, so reruns draw the same windows). One
  row per station: left, the detection-filtered trace; right, its recursive
  STA/LTA ratio with ``trigger_on``/``trigger_off`` drawn as horizontal
  lines. The detection window (``time`` .. ``time + duration_s``) is shaded
  on both, and stations that took part in the detection are labelled in
  bold.

Characteristic-function fidelity
--------------------------------
The example panels reuse the detector's own code path rather than a
plotting-side reimplementation: the same flag-aware read
(:func:`openmeq.detect.worker._read_window`), the same
:func:`~openmeq.detect.worker.preprocess_stream` (merge/split/float32/fast
detrend/``cfg.detection.filter``) and the same
:func:`~openmeq.detect.worker.characteristic_function`. The one unavoidable
difference is the read span: the detector filters and runs STA/LTA over a
whole (padded) day, whereas an example reads only its display window plus
:func:`_warmup_s` seconds before it. The recursive STA/LTA forgets its
initial state with an ``lta``-second time constant and the bandpass
transient is similarly short, so after that warm-up the plotted ratio
matches the detector's to within plotting precision -- the warm-up portion
itself is read but never drawn.

Failure isolation
-----------------
Extended QAQC must never fail the detect step. Each figure (and each
individual example panel) is built under :func:`_safely`, which logs a
warning naming the figure and carries on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from openmeq.detect.diagnostics import _split_stations
from openmeq.detect.worker import _read_window, characteristic_function, preprocess_stream
from openmeq.log import get_logger
from openmeq.plotting import plot_heatmap, save_fig, setup_style

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("detect.extended")

#: ``detections/station_triggers.csv``'s exact column order.
STATION_TRIGGERS_COLUMNS = [
    "seed_id",
    "station",
    "on_time",
    "off_time",
    "duration_s",
    "peak_ratio",
]

#: Sub-directory of ``detections/diagnostics/`` holding the extended figures.
EXTENDED_SUBDIR = "extended"

#: Runs up to this many days get one heatmap column per absolute UTC hour;
#: longer runs fold onto hour-of-day (see the module docstring).
ABSOLUTE_HOURS_MAX_DAYS = 7

#: Number of random no-detection windows appended to the ``cf_examples`` set.
N_QUIET_WINDOWS = 2

#: Fixed seed for the quiet-window draw, so reruns are reproducible.
QUIET_WINDOW_SEED = 0

#: Seconds shown before a detection's ``time`` / after its end.
PRE_S = 10.0
POST_S = 20.0

#: Length of the "event" part of a quiet window (the analogue of a
#: detection's ``duration_s``), in seconds.
QUIET_DURATION_S = 5.0

#: At most this many station rows per ``cf_examples`` figure; triggering
#: stations are kept first when a network is larger than this.
MAX_STATION_ROWS = 30


def station_triggers_csv(paths: RunPaths) -> Path:
    """``detections/station_triggers.csv`` (next to ``detections.csv``)."""
    return Path(paths.detections_csv).parent / "station_triggers.csv"


def extended_dir(paths: RunPaths) -> Path:
    """``detections/diagnostics/extended/``."""
    return Path(paths.detect_diagnostics_dir) / EXTENDED_SUBDIR


def station_triggers_frame(per_window: list[list[dict]]) -> pd.DataFrame:
    """Flatten per-window station-trigger lists into one sorted table.

    Windows never share a station trigger (each keeps only on-times inside
    its unpadded span, see :mod:`openmeq.detect.worker`), so this is a plain
    concatenation, sorted by ``(on_time, seed_id)`` for a deterministic CSV.
    """
    rows = [row for window in per_window for row in window]
    df = pd.DataFrame(rows, columns=STATION_TRIGGERS_COLUMNS)
    if not df.empty:
        df = df.sort_values(["on_time", "seed_id"], kind="mergesort").reset_index(drop=True)
    return df


def write_station_triggers_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write :func:`station_triggers_frame`'s table (parent dir created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=STATION_TRIGGERS_COLUMNS)


def _safely(name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Run one extended figure; log (never raise) on failure."""
    try:
        func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
        logger.warning("detect extended QAQC: %s failed: %s", name, exc)


# ---------------------------------------------------------------------------
# station_trigger_rate.png
# ---------------------------------------------------------------------------


def _run_span(cfg: Config) -> tuple[UTCDateTime, UTCDateTime]:
    """``[cfg.data.start, cfg.data.end + 1 day)`` (``end`` is inclusive)."""
    return UTCDateTime(cfg.data.start), UTCDateTime(cfg.data.end) + 86400.0


def station_trigger_rate_matrix(
    station_triggers_df: pd.DataFrame,
    stations: list[str],
    span: tuple[UTCDateTime, UTCDateTime],
) -> tuple[pd.DataFrame, str]:
    """Station x hour trigger-count matrix for :func:`_station_trigger_rate`.

    Returns ``(matrix, x_description)``: rows are ``stations`` plus any
    station present only in ``station_triggers_df`` (sorted), columns are
    either absolute UTC hours (``"MM-DD HH"``) or hour-of-day ``0..23`` (mean
    triggers per hour), per the module docstring.
    """
    df = station_triggers_df
    all_stations = sorted(set(stations) | set(df["station"].astype(str) if len(df) else []))
    start, end = span
    n_days = max(1, int(round((end - start) / 86400.0)))

    on = pd.to_datetime(df["on_time"], utc=True, format="ISO8601") if len(df) else None
    start_ts = pd.Timestamp(start.datetime, tz="UTC")

    if n_days <= ABSOLUTE_HOURS_MAX_DAYS:
        hours = pd.date_range(start_ts, periods=n_days * 24, freq="1h")
        matrix = pd.DataFrame(0.0, index=all_stations, columns=hours)
        if on is not None:
            counts = pd.crosstab(df["station"].astype(str), on.dt.floor("h"))
            counts = counts.reindex(index=all_stations, columns=hours, fill_value=0)
            matrix = counts.astype(float)
        matrix.columns = [h.strftime("%m-%d %H") for h in hours]
        return matrix, "Hour (UTC)"

    matrix = pd.DataFrame(0.0, index=all_stations, columns=list(range(24)))
    if on is not None:
        counts = pd.crosstab(df["station"].astype(str), on.dt.hour)
        counts = counts.reindex(index=all_stations, columns=list(range(24)), fill_value=0)
        matrix = counts.astype(float) / n_days
    return matrix, f"Hour of day (UTC), mean over {n_days} days"


def _station_trigger_rate(
    station_triggers_df: pd.DataFrame,
    stations: list[str],
    span: tuple[UTCDateTime, UTCDateTime],
    path: Path,
) -> None:
    matrix, xlabel = station_trigger_rate_matrix(station_triggers_df, stations, span)
    vmax = float(matrix.to_numpy().max()) if matrix.size else 1.0
    plot_heatmap(
        matrix,
        path,
        xlabel=xlabel,
        ylabel="Station (NET.STA)",
        title="Single-station STA/LTA triggers per hour",
        cbar_label="Triggers per hour",
        vmin=0.0,
        vmax=max(vmax, 1.0),
    )


# ---------------------------------------------------------------------------
# cf_examples_<nn>.png
# ---------------------------------------------------------------------------


def _warmup_s(det_cfg) -> float:
    """Seconds read (but not drawn) before an example window -- see the
    module docstring's "Characteristic-function fidelity" note."""
    return max(10.0 * det_cfg.lta, 30.0)


def data_coverage(index, span: tuple[UTCDateTime, UTCDateTime]) -> list[tuple]:
    """Merged ``[start, end]`` intervals where *any* indexed channel has data,
    clipped to ``span`` -- where :func:`select_example_windows` may place a
    quiet window (a window drawn into a gap would only show "no data")."""
    raw = sorted(
        (UTCDateTime(start), UTCDateTime(end))
        for segments in index.channels.values()
        for _path, start, end, _sr in segments
    )
    merged: list[list[UTCDateTime]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    lo, hi = span
    return [(max(a, lo), min(b, hi)) for a, b in merged if min(b, hi) > max(a, lo)]


def select_example_windows(
    detections_df: pd.DataFrame,
    n_examples: int,
    span: tuple[UTCDateTime, UTCDateTime],
    *,
    coverage: list[tuple] | None = None,
    warmup_s: float = 0.0,
    n_quiet: int = N_QUIET_WINDOWS,
    margin_s: float = 10.0,
    seed: int = QUIET_WINDOW_SEED,
) -> list[dict]:
    """Choose the ``cf_examples`` windows: sampled detections + quiet windows.

    Detections are evenly sampled in time order (same ``linspace`` rule as
    :mod:`openmeq.pick.diagnostics`' example events). Quiet windows are drawn
    with a seeded RNG, uniformly over ``coverage`` (default: all of
    ``span``) so that the whole window, ``warmup_s`` lead-in included, lies
    inside one data interval, rejecting any whose displayed range comes
    within ``margin_s`` of a detection; fewer than ``n_quiet`` may be
    returned if there is no room (dense detections, very short segments).

    Returns
    -------
    list of dict
        ``{"kind": "detection"|"quiet", "label", "t_ref" (window start of
        the shaded span), "duration_s", "stations" (list of NET.STA),
        "trace_ids" (list)}``, detections first.
    """
    windows: list[dict] = []
    df = detections_df if detections_df is not None else pd.DataFrame()
    n = len(df)
    if n:
        df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
        take = min(n_examples, n)
        idx = sorted({int(round(i)) for i in np.linspace(0, n - 1, take)})
        for i in idx:
            row = df.iloc[i]
            windows.append(
                {
                    "kind": "detection",
                    "label": str(row.get("event_id", row["time"])),
                    "t_ref": UTCDateTime(str(row["time"])),
                    "duration_s": float(row.get("duration_s", 0.0) or 0.0),
                    "stations": _split_stations(row.get("stations")),
                    "trace_ids": _split_stations(row.get("trace_ids")),
                }
            )

    busy = []
    for _, row in df.iterrows():
        t = UTCDateTime(str(row["time"]))
        busy.append((t - margin_s, t + float(row.get("duration_s", 0.0) or 0.0) + margin_s))

    # Candidate t_ref ranges, one per coverage interval long enough to hold
    # a whole quiet window plus its warm-up.
    ranges = [
        (a + warmup_s + PRE_S, b - QUIET_DURATION_S - POST_S)
        for a, b in (coverage if coverage is not None else [span])
    ]
    ranges = [(lo, hi) for lo, hi in ranges if hi > lo]
    lengths = np.array([hi - lo for lo, hi in ranges], dtype=float)
    rng = np.random.default_rng(seed)
    quiet: list[UTCDateTime] = []
    attempts = 0
    while len(quiet) < n_quiet and ranges and attempts < 500:
        attempts += 1
        k = int(rng.choice(len(ranges), p=lengths / lengths.sum()))
        lo, hi = ranges[k]
        t_ref = lo + float(rng.uniform(0.0, hi - lo))
        w0, w1 = t_ref - PRE_S, t_ref + QUIET_DURATION_S + POST_S
        if any(w0 < b1 and b0 < w1 for b0, b1 in busy):
            continue
        quiet.append(t_ref)
        busy.append((w0, w1))
    for t_ref in sorted(quiet):
        windows.append(
            {
                "kind": "quiet",
                "label": f"quiet window {t_ref.strftime('%Y-%m-%dT%H:%M:%S')}Z",
                "t_ref": t_ref,
                "duration_s": QUIET_DURATION_S,
                "stations": [],
                "trace_ids": [],
            }
        )
    return windows


def _pick_channel(seed_ids: list[str], preferred: list[str]) -> str:
    """One seed id per station row: a triggering channel if any (Z first),
    else the station's Z channel, else its first channel (sorted)."""
    pref = sorted(s for s in seed_ids if s in set(preferred))
    pool = pref or sorted(seed_ids)
    return next((s for s in pool if s.endswith("Z")), pool[0])


def _plot_cf_example(window: dict, stream, det_cfg, path: Path) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    t_ref = window["t_ref"]
    x0, x1 = -PRE_S, window["duration_s"] + POST_S
    triggering = set(window["stations"])

    by_station: dict[str, list[str]] = {}
    for tr in stream:
        by_station.setdefault(f"{tr.stats.network}.{tr.stats.station}", []).append(tr.id)
    stations = sorted(by_station, key=lambda s: (s not in triggering, s))[:MAX_STATION_ROWS]
    stations.sort()
    n_rows = max(1, len(stations))

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12.0, max(2.5, 0.9 * n_rows + 1.2)), sharex=True, squeeze=False
    )
    if not stations:
        for ax in axes[0]:
            ax.text(0.5, 0.5, "no usable data", ha="center", va="center", transform=ax.transAxes)

    for row, station in enumerate(stations):
        ax_tr, ax_cf = axes[row, 0], axes[row, 1]
        seed_id = _pick_channel(by_station[station], window["trace_ids"])
        for tr in stream.select(id=seed_id):
            rel = tr.times() + (tr.stats.starttime - t_ref)
            keep = (rel >= x0) & (rel <= x1)
            if not keep.any():
                continue
            cft = characteristic_function(tr, det_cfg)
            ax_tr.plot(rel[keep], tr.data[keep], color="#0072B2", linewidth=0.5)
            ax_cf.plot(rel[keep], cft[keep], color="#222222", linewidth=0.7)
        ax_cf.axhline(
            det_cfg.trigger_on, color="#D55E00", linestyle="--", linewidth=0.9, label="trigger_on"
        )
        ax_cf.axhline(
            det_cfg.trigger_off, color="#009E73", linestyle=":", linewidth=0.9, label="trigger_off"
        )
        if row == 0:
            ax_cf.legend(loc="upper right", fontsize=6)
        for ax in (ax_tr, ax_cf):
            ax.axvspan(0.0, window["duration_s"], color="#E69F00", alpha=0.2, linewidth=0)
            ax.tick_params(labelsize=7)
        ax_tr.set_yticks([])
        ax_tr.set_ylabel(
            seed_id.split(".", 1)[1],
            rotation=0,
            ha="right",
            va="center",
            fontsize=7,
            fontweight="bold" if station in triggering else "normal",
        )

    axes[0, 0].set_title("Detection-filtered trace", fontsize=9)
    axes[0, 1].set_title(
        f"STA/LTA (sta={det_cfg.sta:g}s, lta={det_cfg.lta:g}s; "
        f"on={det_cfg.trigger_on:g}, off={det_cfg.trigger_off:g})",
        fontsize=9,
    )
    for ax in axes[-1]:
        ax.set_xlim(x0, x1)
        ax.set_xlabel(f"Seconds after {t_ref.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z")
    kind = "Detection" if window["kind"] == "detection" else "No detection"
    fig.suptitle(f"{kind}: {window['label']}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, path)


def _cf_examples(cfg: Config, paths: RunPaths, detections_df: pd.DataFrame, out_dir: Path) -> None:
    from openmeq.index import DataIndex

    det_cfg = cfg.detection
    index = DataIndex.load(paths.index_json)
    with open(paths.channel_flags_json, encoding="utf-8") as f:
        flags = json.load(f).get("flags", {})

    span = _run_span(cfg)
    warmup = _warmup_s(det_cfg)
    windows = select_example_windows(
        detections_df,
        cfg.diagnostics.n_examples,
        span,
        coverage=data_coverage(index, span),
        warmup_s=warmup,
        margin_s=det_cfg.dedup_window_s,
    )
    for i, window in enumerate(windows, start=1):
        t_ref = window["t_ref"]
        w0 = t_ref - PRE_S - warmup
        w1 = t_ref + window["duration_s"] + POST_S

        def _one(window=window, w0=w0, w1=w1, i=i) -> None:
            stream = preprocess_stream(_read_window(index, flags, w0, w1, pad=0.0), det_cfg)
            _plot_cf_example(window, stream, det_cfg, out_dir / f"cf_examples_{i:02d}.png")

        _safely(f"cf_examples_{i:02d} ({window['label']})", _one)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _index_stations(paths: RunPaths) -> list[str]:
    from openmeq.index import DataIndex

    try:
        seed_ids = DataIndex.load(paths.index_json).seed_ids()
    except Exception as exc:  # noqa: BLE001 - only used to add zero rows
        logger.warning("detect extended QAQC: could not read %s: %s", paths.index_json, exc)
        return []
    return sorted({".".join(s.split(".")[:2]) for s in seed_ids})


def make_detect_extended_diagnostics(
    cfg: Config,
    paths: RunPaths,
    detections_df: pd.DataFrame,
    station_triggers_df: pd.DataFrame,
) -> None:
    """Build every ``detections/diagnostics/extended/`` figure.

    The caller (:func:`openmeq.detect.pipeline.run_detect`) is responsible
    for the ``cfg.diagnostics.extended_enabled("detect")`` gate. Never
    raises: each figure is isolated by :func:`_safely`.

    Parameters
    ----------
    cfg : openmeq.config.Config
        ``cfg.detection`` (preprocessing/thresholds), ``cfg.data`` (run
        span) and ``cfg.diagnostics.n_examples`` are used.
    paths : openmeq.paths.RunPaths
        ``index_json``/``channel_flags_json`` (waveform reads) and
        ``detect_diagnostics_dir``.
    detections_df : pandas.DataFrame
        ``detections.csv``-shaped table, in-memory or re-read from disk.
    station_triggers_df : pandas.DataFrame
        :data:`STATION_TRIGGERS_COLUMNS` table (:func:`station_triggers_frame`).
    """
    out_dir = extended_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    _safely(
        "station_trigger_rate.png",
        lambda: _station_trigger_rate(
            station_triggers_df,
            _index_stations(paths),
            _run_span(cfg),
            out_dir / "station_trigger_rate.png",
        ),
    )
    _safely("cf_examples", _cf_examples, cfg, paths, detections_df, out_dir)
