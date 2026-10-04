"""``locate/diagnostics/`` (M6.5, ``OpenMEQ-xdf.5``, PLAN.md §5's "Locate" row).

:func:`make_locate_diagnostics` builds, **for each pass (A and B)**:

- P and S residual distributions -- all-station combined
  (``passX_residuals_<phase>.png``) and per-station small multiples
  (``passX_residuals_<phase>_per_station.png``), via
  :func:`openmeq.plotting.plot_hist` / a small local per-station grid helper.
- an event map + E-W/N-S depth-section figure
  (``passX_map_sections.png``, :func:`openmeq.plotting.plot_map_sections`).
- RMS/gap/error histograms (``passX_rms_hist.png``, ``passX_gap_hist.png``,
  ``passX_err_h_hist.png``, ``passX_err_z_hist.png``).
- pick removal counts by reason (``passX_removed_picks_by_reason.png``, from
  ``locate/removal_log.csv``) and event removal counts by reason
  (``passX_removed_events_by_reason.png``, from ``locate/summary.csv``'s
  rejected rows, attributed to pass A or pass B by which pass's own status
  is ``"rejected"``).

Plus ``locate/diagnostics/pass_comparison.pdf``: every figure type above
redrawn with pass A and pass B together on shared axes/bins/colour scale
(and the accepted events as a third map row), one figure per page --
:mod:`openmeq.locate.diagnostics_compare`.

Plus ``locate/diagnostics/summary.txt``: events in / accepted / rejected (by
reason) counts, and each pass's own event/pick counts and mean/median
residuals -- a flat ``key: value`` per line, parseable by any test or
operator without a special-purpose reader.

Plus, only when ``cfg.diagnostics.extended_enabled("locate")``, the opt-in
extended set under ``locate/diagnostics/extended/`` (residual vs distance,
station residual map + CSV, depth histogram vs grid/model) --
:mod:`openmeq.locate.diagnostics_extended`. The baseline set above is
unchanged either way, and an extended-QAQC failure only logs a warning.

Every ``groupby``/plot call here is guarded against an empty frame -- zero
rejections or zero removals is an expected, valid outcome on clean synthetic
data, not an error (this module never raises on an empty pass, event set or
residual set).

Works from :class:`openmeq.locate.locate_step.LocateRunResult` (the pass A/B
per-event diagnostics ``run_locate`` now returns, built entirely from plain
data -- see that module's docstring for why) plus ``locate/removal_log.csv``/
``locate/summary.csv`` already written to disk by ``run_locate`` -- this
module never recomputes a location itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import plot_bar, plot_hist, plot_map_sections, save_fig, setup_style

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.locate.locate_step import LocateRunResult, PassDiagnostics
    from openmeq.paths import RunPaths

logger = get_logger("locate.diagnostics")

#: Phases every pass is checked for -- pass A's own ``residuals``/
#: ``phase_by_seed_id`` naturally never contain "S" (pass A only ever
#: locates P picks, ``openmeq.locate.pass_a.run_pass_a``), so that half of
#: the per-pass-A output is always the "No data" placeholder -- expected,
#: not an error.
_PHASES = ("P", "S")

_PASS_DATAFRAME_COLUMNS = [
    "event_id",
    "ran",
    "status",
    "reason",
    "time",
    "lat",
    "lon",
    "depth_km",
    "rms",
    "gap",
    "err_h_km",
    "err_z_km",
    "n_iterations",
    "stabilized",
    "n_picks_removed",
]

_RESIDUAL_DATAFRAME_COLUMNS = ["event_id", "seed_id", "net_sta", "phase", "residual_s"]


# ---------------------------------------------------------------------------
# Flatten PassDiagnostics dicts into DataFrames (internal helpers)
# ---------------------------------------------------------------------------


def _pass_dataframe(diag_map: dict[str, PassDiagnostics]) -> pd.DataFrame:
    rows = [
        {
            "event_id": d.event_id,
            "ran": d.ran,
            "status": d.status,
            "reason": d.reason,
            "time": d.time,
            "lat": d.lat,
            "lon": d.lon,
            "depth_km": d.depth_km,
            "rms": d.rms,
            "gap": d.gap,
            "err_h_km": d.err_h_km,
            "err_z_km": d.err_z_km,
            "n_iterations": d.n_iterations,
            "stabilized": d.stabilized,
            "n_picks_removed": d.n_picks_removed,
        }
        for d in diag_map.values()
    ]
    return pd.DataFrame(rows, columns=_PASS_DATAFRAME_COLUMNS)


def _residual_dataframe(diag_map: dict[str, PassDiagnostics]) -> pd.DataFrame:
    rows = []
    for d in diag_map.values():
        for seed_id, residual in d.residuals.items():
            phase = d.phase_by_seed_id.get(seed_id)
            net_sta = ".".join(seed_id.split(".")[:2])
            rows.append(
                {
                    "event_id": d.event_id,
                    "seed_id": seed_id,
                    "net_sta": net_sta,
                    "phase": phase,
                    "residual_s": residual,
                }
            )
    return pd.DataFrame(rows, columns=_RESIDUAL_DATAFRAME_COLUMNS)


def _load_station_table(cfg: Config) -> pd.DataFrame | None:
    """Best-effort station table for the map overlay -- ``None`` (skips the
    overlay, ``plot_map_sections`` handles that gracefully) rather than
    raising, since diagnostics must never fail the whole ``openmeq locate``
    run over a station-metadata problem another step already reports on."""
    try:
        from openmeq.stations import station_table

        return station_table(cfg.stations)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never abort the run
        logger.warning("could not load station table for the map overlay: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-pass plots
# ---------------------------------------------------------------------------


def _plot_residual_hist(sub: pd.DataFrame, phase: str, pass_label: str, path: Path) -> None:
    values = (
        sub.loc[sub["phase"] == phase, "residual_s"].to_numpy(dtype=float) if not sub.empty else []
    )
    plot_hist(
        values,
        path,
        bins=30,
        xlabel="Time residual (s)",
        ylabel="Count",
        title=f"Pass {pass_label}: {phase} residuals (all stations)",
    )


def _plot_residuals_per_station(sub: pd.DataFrame, phase: str, pass_label: str, path: Path) -> None:
    """Small-multiples grid: one residual-histogram subplot per station."""
    setup_style()
    import matplotlib.pyplot as plt

    phase_sub = sub[sub["phase"] == phase] if not sub.empty else sub
    stations = sorted(phase_sub["net_sta"].unique()) if not phase_sub.empty else []

    if not stations:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Pass {pass_label}: {phase} residuals per station")
        save_fig(fig, path)
        return

    ncols = min(4, len(stations))
    nrows = -(-len(stations) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)
    for i, sta in enumerate(stations):
        ax = axes[i // ncols][i % ncols]
        vals = phase_sub.loc[phase_sub["net_sta"] == sta, "residual_s"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=15, color="#0072B2", edgecolor="white", linewidth=0.4)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8
            )
        ax.set_title(sta, fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(stations), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Pass {pass_label}: {phase} residuals per station")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_fig(fig, path)


def _plot_map(df: pd.DataFrame, stations: pd.DataFrame | None, pass_label: str, path: Path) -> None:
    map_df = df.dropna(subset=["lat", "lon", "depth_km"]) if not df.empty else df
    color_by = "rms" if "rms" in map_df.columns and map_df["rms"].notna().any() else None
    plot_map_sections(
        map_df,
        stations,
        path,
        color_by=color_by,
        cbar_label="RMS (s)",
        title=f"Pass {pass_label}: event locations",
    )


def _plot_scalar_hist(
    df: pd.DataFrame, column: str, xlabel: str, pass_label: str, path: Path
) -> None:
    values = (
        df[column].dropna().to_numpy(dtype=float) if not df.empty and column in df.columns else []
    )
    plot_hist(
        values,
        path,
        bins=20,
        xlabel=xlabel,
        ylabel="Event count",
        title=f"Pass {pass_label}: {xlabel}",
    )


def _pick_removal_counts_by_reason(removal_log: pd.DataFrame, pass_label: str) -> pd.Series:
    if removal_log.empty:
        return pd.Series(dtype="int64")
    sub = removal_log[removal_log["pass"] == pass_label]
    if sub.empty:
        return pd.Series(dtype="int64")
    return sub.groupby("reason").size().sort_index()


def _event_removal_counts_by_reason(
    summary_rows, pass_a_diag: dict[str, PassDiagnostics], pass_label: str
) -> pd.Series:
    """Rejected events attributed to ``pass_label`` ("A" if pass A itself
    rejected the event, else "B" -- pass B's own residual filtering or
    ``openmeq.locate.finalize``'s event-level criteria), grouped by
    ``summary.csv``'s own ``reason`` string."""
    counts: dict[str, int] = {}
    for row in summary_rows:
        if row.status != "rejected":
            continue
        pass_a = pass_a_diag.get(row.event_id)
        attributed_pass = "A" if (pass_a is not None and pass_a.status == "rejected") else "B"
        if attributed_pass != pass_label:
            continue
        reason = row.reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return pd.Series(counts).sort_index() if counts else pd.Series(dtype="int64")


def _plot_removal_counts(counts: pd.Series, pass_label: str, what: str, path: Path) -> None:
    plot_bar(
        list(counts.index),
        list(counts.to_numpy()) if not counts.empty else [],
        path,
        xlabel="Reason",
        ylabel="Count",
        title=f"Pass {pass_label}: {what} removed by reason",
    )


def _read_removal_log(path: Path) -> pd.DataFrame:
    columns = ["event_id", "pass", "iteration", "seed_id", "phase", "residual_s", "reason"]
    if not Path(path).is_file():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _make_one_pass_diagnostics(
    diag_dir: Path,
    pass_label: str,
    diag_map: dict[str, PassDiagnostics],
    residual_df: pd.DataFrame,
    removal_log: pd.DataFrame,
    summary_rows,
    pass_a_diag: dict[str, PassDiagnostics],
    stations: pd.DataFrame | None,
) -> None:
    df = _pass_dataframe(diag_map)

    for phase in _PHASES:
        _plot_residual_hist(
            residual_df, phase, pass_label, diag_dir / f"pass{pass_label}_residuals_{phase}.png"
        )
        _plot_residuals_per_station(
            residual_df,
            phase,
            pass_label,
            diag_dir / f"pass{pass_label}_residuals_{phase}_per_station.png",
        )

    _plot_map(df, stations, pass_label, diag_dir / f"pass{pass_label}_map_sections.png")

    _plot_scalar_hist(df, "rms", "RMS (s)", pass_label, diag_dir / f"pass{pass_label}_rms_hist.png")
    _plot_scalar_hist(
        df, "gap", "Azimuthal gap (deg)", pass_label, diag_dir / f"pass{pass_label}_gap_hist.png"
    )
    _plot_scalar_hist(
        df,
        "err_h_km",
        "Horizontal error (km)",
        pass_label,
        diag_dir / f"pass{pass_label}_err_h_hist.png",
    )
    _plot_scalar_hist(
        df,
        "err_z_km",
        "Depth error (km)",
        pass_label,
        diag_dir / f"pass{pass_label}_err_z_hist.png",
    )

    pick_counts = _pick_removal_counts_by_reason(removal_log, pass_label)
    _plot_removal_counts(
        pick_counts, pass_label, "picks", diag_dir / f"pass{pass_label}_removed_picks_by_reason.png"
    )
    event_counts = _event_removal_counts_by_reason(summary_rows, pass_a_diag, pass_label)
    _plot_removal_counts(
        event_counts,
        pass_label,
        "events",
        diag_dir / f"pass{pass_label}_removed_events_by_reason.png",
    )


# ---------------------------------------------------------------------------
# summary.txt
# ---------------------------------------------------------------------------


def _mean_median(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.median(finite))


def _write_summary_txt(path: Path, result: LocateRunResult, removal_log: pd.DataFrame) -> None:
    lines: list[str] = []

    n_in = len(result.summary_rows)
    n_accepted = sum(1 for r in result.summary_rows if r.status == "accepted")
    n_rejected = n_in - n_accepted
    lines.append(f"events_in: {n_in}")
    lines.append(f"events_accepted: {n_accepted}")
    lines.append(f"events_rejected: {n_rejected}")

    reason_counts: dict[str, int] = {}
    for row in result.summary_rows:
        if row.status != "rejected":
            continue
        reason = row.reason or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason in sorted(reason_counts):
        lines.append(f"rejected_reason_{reason}: {reason_counts[reason]}")

    for pass_label, diag_map in (("A", result.pass_a), ("B", result.pass_b)):
        ran = [d for d in diag_map.values() if d.ran]
        n_ok = sum(1 for d in ran if d.status == "ok")
        n_rej = sum(1 for d in ran if d.status == "rejected")
        n_picks_removed = sum(d.n_picks_removed for d in ran)
        lines.append(f"pass{pass_label}_n_events_run: {len(ran)}")
        lines.append(f"pass{pass_label}_n_ok: {n_ok}")
        lines.append(f"pass{pass_label}_n_rejected: {n_rej}")
        lines.append(f"pass{pass_label}_n_picks_removed: {n_picks_removed}")

        residual_df = _residual_dataframe(diag_map)
        for phase in _PHASES:
            values = (
                residual_df.loc[residual_df["phase"] == phase, "residual_s"].to_numpy(dtype=float)
                if not residual_df.empty
                else np.array([])
            )
            mean_r, median_r = _mean_median(values)
            mean_str = "" if mean_r is None else f"{mean_r:.6f}"
            median_str = "" if median_r is None else f"{median_r:.6f}"
            lines.append(f"pass{pass_label}_residual_{phase}_mean_s: {mean_str}")
            lines.append(f"pass{pass_label}_residual_{phase}_median_s: {median_str}")

    pick_total = 0 if removal_log.empty else int(len(removal_log))
    lines.append(f"total_picks_removed: {pick_total}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def make_locate_diagnostics(paths: RunPaths, cfg: Config, result: LocateRunResult) -> Path:
    """Build ``locate/diagnostics/`` from a :func:`openmeq.locate.locate_step.run_locate` result.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config
    result : openmeq.locate.locate_step.LocateRunResult
        ``run_locate``'s return value (events/summary/removal records plus
        the per-event pass A/B diagnostics records).

    Returns
    -------
    pathlib.Path
        ``paths.diagnostics_dir("locate")`` (the extended set, if enabled,
        is under its ``extended/`` subfolder).
    """
    diag_dir = paths.diagnostics_dir("locate")
    diag_dir.mkdir(parents=True, exist_ok=True)

    stations = _load_station_table(cfg)
    removal_log = _read_removal_log(paths.removal_log_csv)

    for pass_label, diag_map in (("A", result.pass_a), ("B", result.pass_b)):
        residual_df = _residual_dataframe(diag_map)
        _make_one_pass_diagnostics(
            diag_dir,
            pass_label,
            diag_map,
            residual_df,
            removal_log,
            result.summary_rows,
            result.pass_a,
            stations,
        )

    from openmeq.locate.diagnostics_compare import PASS_COMPARISON_PDF, make_pass_comparison_pdf

    make_pass_comparison_pdf(diag_dir / PASS_COMPARISON_PDF, result, removal_log, stations)

    _write_summary_txt(diag_dir / "summary.txt", result, removal_log)

    if cfg.diagnostics.extended_enabled("locate"):
        try:
            from openmeq.locate.diagnostics_extended import make_locate_extended_diagnostics

            make_locate_extended_diagnostics(paths, cfg, result, stations)
        except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
            logger.warning("extended locate diagnostics failed: %s", exc)

    return diag_dir


__all__ = ["make_locate_diagnostics"]
