"""``pick/diagnostics/`` (M4.4, ``OpenMEQ-1m7.4``, PLAN.md §5's "Pick" row).

Works purely from the already-built :class:`obspy.Catalog`
(:func:`openmeq.pick.to_quakeml.build_catalog`'s return value) plus
``chunks_pn/``/``pick/phasenet_raw/result.h5`` on disk for the example
waveform plots -- never recomputes picking itself, matching every other
step's diagnostics module (see e.g. ``openmeq.detect.diagnostics``'s module
docstring for the same convention).

Outputs (``pick/diagnostics/``, ARCHITECTURE.md §3)
------------------------------------------------------
- ``p_pick_counts_per_station.png`` / ``s_pick_counts_per_station.png``:
  one bar per ``NET.STA``, the number of surviving P/S picks at that
  station across every event (:func:`~openmeq.plotting.plot_bar`).
- ``p_probability_hist.png`` / ``s_probability_hist.png``: histograms of
  every P/S pick's ``probability`` extra (:func:`~openmeq.plotting.plot_hist`).
- ``picks_per_event_hist.png``: histogram of total (P+S) pick count per
  event.
- ``example_<event_id>.png``: for a sample of ``min(10, n_events)`` events
  (evenly spaced across the catalogue's event order, for coverage across
  the whole run rather than always the first few), one stacked-subplot
  figure per event -- one row per station, that station's Z-component
  waveform with vertical P (blue)/S (red) pick-time markers, read from
  ``chunks_pn/<event_id>/<event_id>.mseed`` (never the raw
  ``chunks/`` chunk -- picks are attributed against the PhaseNet-ready
  stream, ARCHITECTURE.md §5).
- ``summary.csv``: ``metric,value`` rows (event/pick counts, mean
  probability by phase), matching every other step's diagnostics summary
  convention (e.g. ``openmeq.detect.diagnostics``, ``openmeq.chunk_diagnostics``).

Probability-trace overlay (``result.h5``, optional)
--------------------------------------------------------
``openmeq.pick.run_phasenet.run_phasenet`` always passes ``--save_prob``
(negligible overhead at this chunk size, ``docs/phasenet_interface.md`` §6),
so ``pick/phasenet_raw/result.h5`` is normally present -- but this module
still checks for it explicitly (rather than assuming it exists) so a
catalogue built from an externally-supplied ``picks.csv`` (no ``result.h5``
sibling) still gets a waveform-only example plot rather than raising.
When present, ``result.h5[f"{event_id}/{event_id}.mseed"]`` is a
``(nt, nx, 3)`` array (``docs/phasenet_interface.md`` §4's confirmed shape:
``nx`` is that file's own per-file station count, channel 0
noise/1 P/2 S). **This module does not attempt to map array index ``nx``
back to a specific station** -- PhaseNet's own internal per-file station
*order* along that axis is not part of the confirmed/documented contract
(only the station *count* is, ``docs/phasenet_interface.md`` §4.9), so any
per-station index correspondence here would be a guess. Instead, the
overlay plots the **max-over-stations** P and S softmax trace
(``prob[:, :, 1].max(axis=1)`` / ``prob[:, :, 2].max(axis=1)``) against
time -- an honest, always-correct summary of "how confident was the model,
anywhere in this chunk, at each instant", shown once per event beneath the
per-station waveform rows, rather than a per-station overlay that could
silently mislabel which station's probability trace is actually being shown.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import Stream, read

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import get_extra
from openmeq.log import get_logger
from openmeq.plotting import plot_bar, plot_hist, save_fig, setup_style

if TYPE_CHECKING:
    from obspy import Catalog
    from obspy.core.event import Event

    from openmeq.paths import RunPaths

logger = get_logger("pick.diagnostics")

#: ``pick/diagnostics/summary.csv``'s exact column order.
SUMMARY_COLUMNS = ["metric", "value"]

#: Max number of example waveform+probability figures to draw.
MAX_EXAMPLE_EVENTS = 10


# ---------------------------------------------------------------------------
# Flatten the catalogue into a per-pick table (internal helper)
# ---------------------------------------------------------------------------


def _pick_rows(catalog: Catalog) -> pd.DataFrame:
    rows: list[dict] = []
    for event in catalog:
        event_id = get_event_id(event)
        n_picks = len(event.picks)
        for pick in event.picks:
            wid = pick.waveform_id
            rows.append(
                {
                    "event_id": event_id,
                    "net_sta": f"{wid.network_code}.{wid.station_code}",
                    "channel": wid.channel_code,
                    "phase_hint": pick.phase_hint,
                    "probability": get_extra(pick, "probability"),
                    "n_picks_this_event": n_picks,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "net_sta",
            "channel",
            "phase_hint",
            "probability",
            "n_picks_this_event",
        ],
    )


# ---------------------------------------------------------------------------
# Aggregate plots
# ---------------------------------------------------------------------------


def _counts_per_station(df: pd.DataFrame, phase_hint: str, path: Path) -> None:
    if df.empty:
        plot_bar(
            [],
            [],
            path,
            xlabel="Station (NET.STA)",
            ylabel="Pick count",
            title=f"{phase_hint} picks per station",
        )
        return
    sub = df[df["phase_hint"] == phase_hint]
    counts = sub.groupby("net_sta").size().sort_index()
    plot_bar(
        list(counts.index),
        list(counts.to_numpy()),
        path,
        xlabel="Station (NET.STA)",
        ylabel="Pick count",
        title=f"{phase_hint} picks per station",
    )


def _probability_hist(df: pd.DataFrame, phase_hint: str, path: Path) -> None:
    sub = df[df["phase_hint"] == phase_hint] if not df.empty else df
    values = sub["probability"].to_numpy(dtype=float) if not sub.empty else []
    plot_hist(
        values,
        path,
        bins=20,
        xlabel="Probability",
        ylabel="Count",
        title=f"{phase_hint} pick probability",
    )


def _picks_per_event_hist(catalog: Catalog, path: Path) -> None:
    counts = [len(event.picks) for event in catalog]
    plot_hist(
        counts,
        path,
        bins=max(1, min(20, len(set(counts)) or 1)),
        xlabel="Picks per event",
        ylabel="Event count",
        title="Picks per event",
    )


def _write_summary(df: pd.DataFrame, catalog: Catalog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_events = len(catalog.events)
    if df.empty:
        rows = [
            {"metric": "n_events", "value": n_events},
            {"metric": "n_picks_total", "value": 0},
            {"metric": "n_p_picks", "value": 0},
            {"metric": "n_s_picks", "value": 0},
            {"metric": "mean_p_probability", "value": ""},
            {"metric": "mean_s_probability", "value": ""},
            {"metric": "n_stations", "value": 0},
        ]
    else:
        p = df[df["phase_hint"] == "P"]
        s = df[df["phase_hint"] == "S"]
        rows = [
            {"metric": "n_events", "value": n_events},
            {"metric": "n_picks_total", "value": int(len(df))},
            {"metric": "n_p_picks", "value": int(len(p))},
            {"metric": "n_s_picks", "value": int(len(s))},
            {
                "metric": "mean_p_probability",
                "value": float(p["probability"].mean()) if len(p) else "",
            },
            {
                "metric": "mean_s_probability",
                "value": float(s["probability"].mean()) if len(s) else "",
            },
            {"metric": "n_stations", "value": int(df["net_sta"].nunique())},
        ]
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Example waveform + probability plots
# ---------------------------------------------------------------------------


def _sample_events(catalog: Catalog, max_events: int) -> list[Event]:
    events = list(catalog)
    if len(events) <= max_events:
        return events
    idx = np.linspace(0, len(events) - 1, max_events)
    return [events[int(round(i))] for i in idx]


def _read_result_h5_probs(result_h5_path: Path, key: str) -> np.ndarray | None:
    try:
        import h5py
    except ImportError:  # pragma: no cover - h5py ships with tensorflow's own deps here
        return None
    try:
        with h5py.File(result_h5_path, "r") as f:
            if key not in f:
                return None
            return np.asarray(f[key])
    except Exception as exc:  # noqa: BLE001 - a bad/partial result.h5 must not abort diagnostics
        logger.warning("could not read %s from %s: %s", key, result_h5_path, exc)
        return None


def _plot_event_example(
    event: Event,
    event_id: str,
    stream: Stream,
    prob: np.ndarray | None,
    path: Path,
) -> None:
    setup_style()
    import matplotlib.pyplot as plt

    stations = sorted({tr.id[:-1] for tr in stream})
    n_rows = len(stations) + (1 if prob is not None else 0)
    if n_rows == 0:
        n_rows = 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(10.0, max(2.0, 1.4 * n_rows)), sharex=False)
    axes = np.atleast_1d(axes)

    picks_by_station: dict[str, list] = {}
    for pick in event.picks:
        wid = pick.waveform_id
        key = f"{wid.network_code}.{wid.station_code}.{wid.location_code}.{wid.channel_code[:-1]}"
        picks_by_station.setdefault(key, []).append(pick)

    for ax, station_key in zip(axes, stations, strict=False):
        candidates = [tr for tr in stream if tr.id[:-1] == station_key]
        z_tr = next((tr for tr in candidates if tr.id.endswith("Z")), None)
        if z_tr is None:
            z_tr = candidates[0] if candidates else None

        if z_tr is not None:
            times = z_tr.times("matplotlib")
            ax.plot(times, z_tr.data, color="#0072B2", linewidth=0.6)
            ax.xaxis_date()
        ax.set_ylabel(station_key.split(".")[1], rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])

        for pick in picks_by_station.get(station_key, []):
            color = "#0072B2" if pick.phase_hint == "P" else "#D55E00"
            try:
                x = pick.time.matplotlib_date
            except AttributeError:  # pragma: no cover - obspy always provides this
                continue
            ax.axvline(x, color=color, linestyle="--", linewidth=1.0)

    if prob is not None:
        ax = axes[-1]
        p_trace = prob[:, :, 1].max(axis=1) if prob.ndim == 3 and prob.shape[2] > 1 else None
        s_trace = prob[:, :, 2].max(axis=1) if prob.ndim == 3 and prob.shape[2] > 2 else None
        idx = np.arange(prob.shape[0])
        if p_trace is not None:
            ax.plot(
                idx, p_trace, color="#0072B2", linewidth=0.8, label="P prob (max over stations)"
            )
        if s_trace is not None:
            ax.plot(
                idx, s_trace, color="#D55E00", linewidth=0.8, label="S prob (max over stations)"
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("prob", fontsize=8)
        ax.legend(loc="upper right", fontsize=6)

    fig.suptitle(f"{event_id}: waveforms + picks" + (" + probability" if prob is not None else ""))
    fig.tight_layout()
    save_fig(fig, path)


def _make_example_plots(paths: RunPaths, catalog: Catalog, diagnostics_dir: Path) -> None:
    sample = _sample_events(catalog, MAX_EXAMPLE_EVENTS)
    result_h5_path = paths.phasenet_raw_dir / "result.h5"
    has_result_h5 = result_h5_path.is_file()

    for event in sample:
        event_id = get_event_id(event)
        chunk_path = paths.chunk_pn_file(event_id)
        if not chunk_path.is_file():
            logger.warning(
                "event %s: %s not found -- skipping its example waveform plot",
                event_id,
                chunk_path,
            )
            continue
        try:
            stream = read(str(chunk_path))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort diagnostics
            logger.warning("event %s: could not read %s: %s", event_id, chunk_path, exc)
            continue

        prob = None
        if has_result_h5:
            prob = _read_result_h5_probs(result_h5_path, f"{event_id}/{event_id}.mseed")

        out_path = diagnostics_dir / f"example_{event_id}.png"
        try:
            _plot_event_example(event, event_id, stream, prob, out_path)
        except Exception as exc:  # noqa: BLE001 - one bad plot must not abort the whole run
            logger.warning("event %s: failed to draw example plot: %s", event_id, exc)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_pick_diagnostics(paths: RunPaths, catalog: Catalog) -> Path:
    """Build every ``pick/diagnostics/`` output from an already-built catalogue.

    Never recomputes picking or conversion -- purely reads ``catalog``
    (:func:`openmeq.pick.to_quakeml.build_catalog`'s return value) plus
    ``chunks_pn/``/``pick/phasenet_raw/result.h5`` on disk for the example
    waveform plots (see module docstring). Robust to an empty catalogue (0
    events) and to an individual event with 0 picks -- neither raises.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    catalog : obspy.Catalog

    Returns
    -------
    pathlib.Path
        ``paths.diagnostics_dir("pick")``.
    """
    diagnostics_dir = paths.diagnostics_dir("pick")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    df = _pick_rows(catalog)

    _counts_per_station(df, "P", diagnostics_dir / "p_pick_counts_per_station.png")
    _counts_per_station(df, "S", diagnostics_dir / "s_pick_counts_per_station.png")
    _probability_hist(df, "P", diagnostics_dir / "p_probability_hist.png")
    _probability_hist(df, "S", diagnostics_dir / "s_probability_hist.png")
    _picks_per_event_hist(catalog, diagnostics_dir / "picks_per_event_hist.png")
    _write_summary(df, catalog, diagnostics_dir / "summary.csv")

    _make_example_plots(paths, catalog, diagnostics_dir)

    logger.info(
        "pick diagnostics: %d event(s), %d pick(s) -> %s",
        len(catalog.events),
        len(df),
        diagnostics_dir,
    )
    return diagnostics_dir


__all__ = ["MAX_EXAMPLE_EVENTS", "SUMMARY_COLUMNS", "make_pick_diagnostics"]
