"""``pick/diagnostics/extended/``: opt-in pick QAQC figures (``OpenMEQ-02o.4``).

Written only when ``cfg.diagnostics.extended_enabled("pick")`` is true, on
top of (never instead of) :mod:`openmeq.pick.diagnostics`'s always-on
baseline set, which is unchanged. Like the baseline, this module never
recomputes picking: it reads the already-built catalogue, the
``chunks_pn/`` waveforms, ``pick/phasenet_raw/result.h5`` (optional) and
``detections/detections.csv``. **Extended QAQC never fails the step**:
every figure (and every per-event zoom figure) is drawn inside its own
``try``/``except`` that logs a warning and moves on.

Outputs (``pick/diagnostics/extended/``)
-------------------------------------------
- ``pick_zoom_<event_id>.png``: for ``cfg.diagnostics.pick.pick_zoom.n_events``
  events (evenly sampled across the catalogue, same rule as the baseline
  ``example_<event_id>.png``), a grid with one row per station and a P and
  an S column. Each panel is windowed to the pick time ``+/- half_window_s``
  (x-axis in seconds relative to the pick) -- P panels show the ``Z``
  trace, S panels every horizontal present (``N``/``E`` or ``1``/``2``),
  each normalised to its own in-window peak. The pick is a vertical line
  labelled with its probability; the PhaseNet P or S softmax trace for that
  station is drawn on a twin y-axis (with the ``min_p_prob``/``min_s_prob``
  threshold dotted) when ``result.h5`` is present and its station columns
  can be trusted (below). A phase with no pick at that station is a blank
  panel. With ``pick_zoom.show_rejected``, dropped candidates are also drawn,
  grey and dashed, with their reason: ``picks.csv`` rows that
  :func:`~openmeq.pick.to_quakeml.build_catalog` dropped (``duplicate`` /
  ``unresolved channel``, via its ``rejected=`` side channel) and
  probability-trace peaks between :data:`REJECTED_PEAK_FLOOR` and the
  threshold (``< min_p_prob``) -- PhaseNet applies the threshold itself, so
  those never reach ``picks.csv`` and ``result.h5`` is the only record of
  them.
- ``wadati_prelocate.png``: S-P time vs P arrival time relative to that
  event's earliest P pick, one point per (event, station) with both a P and
  an S pick, all events pooled; a least-squares line and the implied
  ``Vp/Vs = 1 + slope`` are annotated. A pre-location sanity check: no
  origin time is needed, and a pooled slope far from ~0.7 (Vp/Vs ~1.7)
  flags mis-associated S picks.
- ``pick_offset_hist.png``: P and S pick times relative to their event's
  detection time (``detections/detections.csv``'s ``time``, matched by
  ``event_id`` -- the chunk/event id *is* the detection's id, M3.1), with
  the chunk window's edges (``-cfg.chunks.pre_s``/``+cfg.chunks.post_s``)
  marked. Picks piling up against an edge mean the window is too short.

``result.h5`` station columns
-------------------------------
The baseline example plot deliberately shows only the max-over-stations
probability, because PhaseNet's station order along ``nx`` is not part of
its documented contract. A per-station zoom needs the per-station trace,
so this module relies on the vendored PhaseNet's ``read_mseed``
(``station_keys = sorted(station_ids)``: columns are the file's distinct
``trace.id[:-1]`` values, sorted) -- **and verifies it per event**: the
column count must equal that sorted key count, and every catalogue pick
must read back its own ``probability`` (within
:data:`_PROB_CHECK_TOLERANCE`) from its mapped column at its sample index.
Any mismatch disables the probability overlay for that event (logged)
rather than risk drawing another station's trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import get_extra
from openmeq.log import get_logger
from openmeq.pick.diagnostics import _read_result_h5_probs, _sample_events
from openmeq.plotting import save_fig, setup_style

if TYPE_CHECKING:
    import matplotlib.figure
    from obspy import Catalog
    from obspy.core.event import Event

    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("pick.extended_diagnostics")

#: Sub-folder of ``pick/diagnostics/`` the extended figures go in.
EXTENDED_SUBDIR = "extended"

#: Lowest probability-trace peak drawn as a below-threshold rejected
#: candidate (``show_rejected``); smaller wiggles are noise, not candidates.
REJECTED_PEAK_FLOOR = 0.1

#: Max allowed ``|result.h5 value - pick probability|`` at a pick's own
#: sample when verifying the station-column mapping (module docstring);
#: ``picks.csv`` rounds ``phase_score`` to 3 decimals.
_PROB_CHECK_TOLERANCE = 0.05

_P_COLOR = "#0072B2"
_S_COLOR = "#D55E00"
_H_COLORS = ["#D55E00", "#009E73"]
_REJECT_COLOR = "#888888"


def _station_key(pick) -> str:
    """``NET.STA.LOC.BandInst`` (``picks.csv``'s ``station_id`` form) for a pick."""
    wid = pick.waveform_id
    return f"{wid.network_code}.{wid.station_code}.{wid.location_code}.{wid.channel_code[:-1]}"


# ---------------------------------------------------------------------------
# pick_zoom_<event_id>.png
# ---------------------------------------------------------------------------


def _prob_columns(
    stream: Stream, prob: np.ndarray | None, event: Event, begin: UTCDateTime, sr: float
) -> dict[str, int]:
    """``{station_key: result.h5 column}``, or ``{}`` if unverifiable (module docstring)."""
    if prob is None or prob.ndim != 3 or prob.shape[2] < 3:
        return {}
    keys = sorted({tr.id[:-1] for tr in stream})
    if len(keys) != prob.shape[1]:
        logger.warning(
            "result.h5 has %d station column(s) but chunks_pn has %d station(s) -- "
            "no probability overlay",
            prob.shape[1],
            len(keys),
        )
        return {}
    cols = {key: i for i, key in enumerate(keys)}
    for pick in event.picks:
        p = get_extra(pick, "probability")
        key = _station_key(pick)
        idx = int(round((pick.time - begin) * sr))
        if p is None or key not in cols or not 0 <= idx < prob.shape[0]:
            continue
        ch = 1 if pick.phase_hint == "P" else 2
        if abs(float(prob[idx, cols[key], ch]) - float(p)) > _PROB_CHECK_TOLERANCE:
            logger.warning(
                "result.h5 column for %s does not reproduce its %s pick probability "
                "(%.3f vs %.3f) -- no probability overlay for this event",
                key,
                pick.phase_hint,
                float(prob[idx, cols[key], ch]),
                float(p),
            )
            return {}
    return cols


def _panel_traces(stream: Stream, station_key: str, phase: str) -> list:
    candidates = sorted(
        (tr for tr in stream if tr.id[:-1] == station_key), key=lambda tr: tr.stats.channel
    )
    if phase == "P":
        return [tr for tr in candidates if tr.stats.channel.endswith("Z")]
    return [tr for tr in candidates if tr.stats.channel[-1] in "NE12"]


def _draw_zoom_panel(
    ax,
    traces: list,
    centre: UTCDateTime,
    half_window_s: float,
    phase: str,
    pick,
    rejected: list[dict],
    prob_trace: np.ndarray | None,
    begin: UTCDateTime,
    sr: float,
    threshold: float,
    show_rejected: bool,
) -> None:
    color = _P_COLOR if phase == "P" else _S_COLOR
    for i, tr in enumerate(traces):
        sub = tr.slice(centre - half_window_s, centre + half_window_s)
        if sub.stats.npts == 0:
            continue
        data = sub.data.astype(float)
        data = data - data.mean()
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak
        x = sub.times() + (sub.stats.starttime - centre)
        tr_color = _P_COLOR if phase == "P" else _H_COLORS[i % len(_H_COLORS)]
        ax.plot(x, data, color=tr_color, linewidth=0.6, label=tr.stats.channel[-1])
    if phase == "S" and len(traces) > 1:
        ax.legend(loc="lower left", fontsize=5, frameon=False, ncol=len(traces))
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([])

    if pick is not None:
        ax.axvline(pick.time - centre, color=color, linewidth=1.2)
        p = get_extra(pick, "probability")
        label = f"{phase} p={float(p):.2f}" if p is not None else phase
        ax.text(
            pick.time - centre,
            1.05,
            f" {label}",
            color=color,
            fontsize=6,
            va="top",
            ha="left",
        )

    ax2 = None
    if prob_trace is not None:
        ax2 = ax.twinx()
        x = (begin - centre) + np.arange(prob_trace.size) / sr
        in_win = (x >= -half_window_s) & (x <= half_window_s)
        ax2.plot(x[in_win], prob_trace[in_win], color=color, linewidth=0.8, alpha=0.5)
        ax2.axhline(threshold, color=color, linestyle=":", linewidth=0.7, alpha=0.7)
        ax2.set_ylim(0.0, 1.05)
        ax2.tick_params(axis="y", labelsize=5)
        ax2.grid(False)

    if show_rejected:
        marks = [(r["time"] - centre, r["probability"], r["reason"]) for r in rejected]
        if prob_trace is not None:
            from scipy.signal import find_peaks

            idx, props = find_peaks(
                prob_trace,
                height=(REJECTED_PEAK_FLOOR, threshold),
                distance=max(1, int(0.2 * sr)),
            )
            for i, h in zip(idx, props["peak_heights"], strict=True):
                marks.append(((begin - centre) + i / sr, float(h), f"< min_{phase.lower()}_prob"))
        for n, (x, p, reason) in enumerate(m for m in marks if abs(m[0]) <= half_window_s):
            ax.axvline(x, color=_REJECT_COLOR, linestyle="--", linewidth=0.9)
            ax.text(
                x,
                -1.05 + 0.22 * n,
                f" {p:.2f} {reason}",
                color=_REJECT_COLOR,
                fontsize=5,
                va="bottom",
                ha="left",
            )

    ax.set_xlim(-half_window_s, half_window_s)
    if ax2 is not None:
        ax2.set_xlim(-half_window_s, half_window_s)


def _build_pick_zoom_figure(
    event: Event,
    event_id: str,
    stream: Stream,
    prob: np.ndarray | None,
    cfg: Config,
    rejected: list[dict] | None = None,
) -> matplotlib.figure.Figure:
    """Build (not save) one event's ``pick_zoom`` grid figure (module docstring).

    Returned rather than saved so tests can inspect the axes; the caller
    passes it to :func:`~openmeq.plotting.save_fig`, which closes it.
    """
    setup_style()
    import matplotlib.pyplot as plt

    zoom = cfg.diagnostics.pick.pick_zoom
    half = float(zoom.half_window_s)
    show_rejected = bool(zoom.show_rejected)
    thresholds = {"P": cfg.phasenet.min_p_prob, "S": cfg.phasenet.min_s_prob}
    rejected = [r for r in (rejected or []) if r["event_id"] == event_id] if show_rejected else []

    kept: dict[tuple[str, str], object] = {}
    for pick in event.picks:
        if pick.phase_hint in ("P", "S"):
            kept[(_station_key(pick), pick.phase_hint)] = pick
    dropped: dict[tuple[str, str], list[dict]] = {}
    for r in rejected:
        dropped.setdefault((r["station_id"], r["phase_type"]), []).append(r)
    stations = sorted({key for key, _ in kept} | {key for key, _ in dropped})

    begin = min((tr.stats.starttime for tr in stream), default=UTCDateTime(0))
    sr = float(stream[0].stats.sampling_rate) if len(stream) else 100.0
    cols = _prob_columns(stream, prob, event, begin, sr)

    n_rows = max(1, len(stations))
    fig, axes = plt.subplots(n_rows, 2, figsize=(8.0, max(2.2, 1.5 * n_rows)), squeeze=False)
    for row, station_key in enumerate(stations):
        for col, phase in enumerate(("P", "S")):
            ax = axes[row, col]
            pick = kept.get((station_key, phase))
            rej = dropped.get((station_key, phase), [])
            if pick is None and not rej:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"no {phase} pick",
                    ha="center",
                    va="center",
                    color=_REJECT_COLOR,
                    fontsize=7,
                    transform=ax.transAxes,
                )
                continue
            if pick is not None:
                centre = pick.time
            else:
                centre = max(rej, key=lambda r: r["probability"])["time"]
            prob_trace = None
            if station_key in cols:
                prob_trace = np.asarray(prob[:, cols[station_key], 1 if phase == "P" else 2])
            _draw_zoom_panel(
                ax,
                _panel_traces(stream, station_key, phase),
                centre,
                half,
                phase,
                pick,
                rej,
                prob_trace,
                begin,
                sr,
                thresholds[phase],
                show_rejected,
            )
        label_ax = next((ax for ax in axes[row] if ax.axison), None)
        if label_ax is not None:
            label_ax.set_ylabel(
                station_key.split(".")[1], rotation=0, ha="right", va="center", fontsize=8
            )
        axes[row, 0].set_title("P (Z)" if row == 0 else "", fontsize=8)
        axes[row, 1].set_title("S (horizontals)" if row == 0 else "", fontsize=8)
    if not stations:
        for ax in axes.ravel():
            ax.set_axis_off()
        axes[0, 0].text(0.5, 0.5, "No picks", ha="center", va="center")
    # x tick labels only on each column's lowest drawn panel (a column's
    # bottom row may be a blank "no pick" panel).
    for col in range(2):
        drawn = [axes[row, col] for row in range(n_rows) if axes[row, col].axison]
        for ax in drawn:
            ax.tick_params(axis="x", labelsize=6, labelbottom=ax is drawn[-1])
        if drawn:
            drawn[-1].set_xlabel("Time rel. pick (s)", fontsize=7)

    overlay = "prob" if cols else "no prob"
    fig.suptitle(f"{event_id}: picks +/- {half:g} s ({overlay})", fontsize=9)
    fig.tight_layout()
    return fig


def _make_pick_zoom_plots(
    paths: RunPaths,
    catalog: Catalog,
    cfg: Config,
    out_dir: Path,
    rejected: list[dict] | None,
) -> None:
    result_h5_path = paths.phasenet_raw_dir / "result.h5"
    has_result_h5 = result_h5_path.is_file()
    for event in _sample_events(catalog, cfg.diagnostics.pick.pick_zoom.n_events):
        event_id = get_event_id(event)
        try:
            chunk_path = paths.chunk_pn_file(event_id)
            if not chunk_path.is_file():
                logger.warning("event %s: %s not found -- no pick_zoom plot", event_id, chunk_path)
                continue
            stream = read(str(chunk_path))
            prob = None
            if has_result_h5:
                prob = _read_result_h5_probs(result_h5_path, f"{event_id}/{event_id}.mseed")
            fig = _build_pick_zoom_figure(event, event_id, stream, prob, cfg, rejected)
            save_fig(fig, out_dir / f"pick_zoom_{event_id}.png")
        except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
            logger.warning("event %s: failed to draw pick_zoom plot: %s", event_id, exc)


# ---------------------------------------------------------------------------
# wadati_prelocate.png
# ---------------------------------------------------------------------------


def _wadati_points(catalog: Catalog) -> pd.DataFrame:
    """One ``(event_id, station, p_rel_s, s_minus_p_s)`` row per station with both P and S."""
    rows: list[dict] = []
    for event in catalog:
        p_times: dict[str, UTCDateTime] = {}
        s_times: dict[str, UTCDateTime] = {}
        for pick in event.picks:
            wid = pick.waveform_id
            sta = f"{wid.network_code}.{wid.station_code}"
            if pick.phase_hint == "P":
                p_times[sta] = pick.time
            elif pick.phase_hint == "S":
                s_times[sta] = pick.time
        if not p_times:
            continue
        t_first = min(p_times.values())
        for sta in sorted(p_times.keys() & s_times.keys()):
            rows.append(
                {
                    "event_id": get_event_id(event),
                    "station": sta,
                    "p_rel_s": p_times[sta] - t_first,
                    "s_minus_p_s": s_times[sta] - p_times[sta],
                }
            )
    return pd.DataFrame(rows, columns=["event_id", "station", "p_rel_s", "s_minus_p_s"])


def _plot_wadati(catalog: Catalog, path: Path) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    df = _wadati_points(catalog)
    fig, ax = plt.subplots()
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(df["p_rel_s"], df["s_minus_p_s"], s=14, alpha=0.7, color=_P_COLOR, lw=0)
        x = df["p_rel_s"].to_numpy(dtype=float)
        y = df["s_minus_p_s"].to_numpy(dtype=float)
        n_ev = df["event_id"].nunique()
        if len(df) >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, slope * xs + intercept, color=_S_COLOR, linewidth=1.2)
            text = f"Vp/Vs = 1 + slope = {1.0 + slope:.2f}\nn = {len(df)} ({n_ev} events)"
        else:
            text = f"too few points for a fit\nn = {len(df)} ({n_ev} events)"
        ax.text(0.03, 0.97, text, ha="left", va="top", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("P arrival rel. event's earliest P (s)")
    ax.set_ylabel("S - P (s)")
    ax.set_title("Wadati diagram (pre-location)")
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# pick_offset_hist.png
# ---------------------------------------------------------------------------


def _detection_times(paths: RunPaths) -> dict[str, UTCDateTime]:
    if not paths.detections_csv.is_file():
        logger.info("%s not found -- pick_offset_hist has no data", paths.detections_csv)
        return {}
    det = pd.read_csv(paths.detections_csv, usecols=["event_id", "time"])
    return {str(r.event_id): UTCDateTime(str(r.time)) for r in det.itertuples(index=False)}


def _plot_pick_offset_hist(paths: RunPaths, catalog: Catalog, cfg: Config, path: Path) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    det_times = _detection_times(paths)
    offsets: dict[str, list[float]] = {"P": [], "S": []}
    n_unmatched = 0
    for event in catalog:
        t_det = det_times.get(get_event_id(event))
        if t_det is None:
            n_unmatched += 1
            continue
        for pick in event.picks:
            if pick.phase_hint in offsets:
                offsets[pick.phase_hint].append(pick.time - t_det)
    if n_unmatched and det_times:
        logger.warning("pick_offset_hist: %d event(s) not in %s", n_unmatched, paths.detections_csv)

    pre_s, post_s = float(cfg.chunks.pre_s), float(cfg.chunks.post_s)
    fig, ax = plt.subplots()
    all_offsets = offsets["P"] + offsets["S"]
    if not all_offsets:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        lo = min(-pre_s, min(all_offsets))
        hi = max(post_s, max(all_offsets))
        bins = np.linspace(lo, hi, 61)
        for phase, color in (("P", _P_COLOR), ("S", _S_COLOR)):
            if offsets[phase]:
                ax.hist(
                    offsets[phase],
                    bins=bins,
                    color=color,
                    alpha=0.6,
                    label=f"{phase} (n={len(offsets[phase])})",
                )
    ax.axvline(-pre_s, color="#000000", linestyle="--", linewidth=1.0, label=f"-pre_s ({pre_s:g})")
    ax.axvline(
        post_s, color="#000000", linestyle="-.", linewidth=1.0, label=f"+post_s ({post_s:g})"
    )
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.8, label="detection")
    ax.legend(fontsize=7)
    ax.set_xlabel("Pick time rel. detection time (s)")
    ax.set_ylabel("Pick count")
    ax.set_title("Pick offset from detection")
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_pick_extended_diagnostics(
    paths: RunPaths,
    catalog: Catalog,
    cfg: Config,
    *,
    rejected: list[dict] | None = None,
) -> Path | None:
    """Write ``pick/diagnostics/extended/`` if ``cfg.diagnostics.extended_enabled("pick")``.

    Never raises for a figure failure (each is logged and skipped, module
    docstring).

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    catalog : obspy.Catalog
        :func:`openmeq.pick.to_quakeml.build_catalog`'s return value.
    cfg : openmeq.config.Config
        Gating (``cfg.diagnostics``), ``pick_zoom`` settings,
        ``cfg.phasenet`` thresholds and ``cfg.chunks`` window.
    rejected : list of dict, keyword-only, optional
        :func:`~openmeq.pick.to_quakeml.build_catalog`'s ``rejected=`` side
        channel, for ``pick_zoom.show_rejected``.

    Returns
    -------
    pathlib.Path or None
        The ``extended/`` directory, or ``None`` if disabled (nothing is
        written or created).
    """
    if not cfg.diagnostics.extended_enabled("pick"):
        return None
    out_dir = paths.diagnostics_dir("pick") / EXTENDED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("wadati_prelocate.png", lambda: _plot_wadati(catalog, out_dir / "wadati_prelocate.png")),
        (
            "pick_offset_hist.png",
            lambda: _plot_pick_offset_hist(paths, catalog, cfg, out_dir / "pick_offset_hist.png"),
        ),
        ("pick_zoom_*.png", lambda: _make_pick_zoom_plots(paths, catalog, cfg, out_dir, rejected)),
    ]
    for name, draw in figures:
        try:
            draw()
        except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
            logger.warning("pick extended diagnostics: failed to draw %s: %s", name, exc)

    logger.info("pick extended diagnostics -> %s", out_dir)
    return out_dir


__all__ = ["EXTENDED_SUBDIR", "REJECTED_PEAK_FLOOR", "make_pick_extended_diagnostics"]
