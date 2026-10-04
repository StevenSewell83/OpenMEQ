"""``locate/diagnostics/pass_comparison.pdf``: pass A vs pass B on shared scales.

The per-pass PNGs (:mod:`openmeq.locate.diagnostics`) each pick their own
axis limits, bins and colour range, so pass A and pass B can't be compared
by flipping between them. This module draws every figure type once more
with both passes in the same figure and on the **same** limits/bins/colour
scale, one figure per page of a single PDF:

1. Map + E-W/N-S sections -- rows: pass A, pass B, accepted (the final
   catalogue, ``summary.csv`` status ``accepted``), coloured by RMS on one
   shared scale.
2. RMS / gap / horizontal / depth error histograms, A and B overlaid.
3. P and S residual histograms (all stations), A and B overlaid.
4-5. P and S residuals per station, A and B overlaid.
6. Picks and events removed, by reason, A and B side by side.

Pass A locates with P only, so it never has S residuals -- the S panels
show pass B alone (stated on the page). Event-removal reasons are grouped by
their prefix (``nlloc_failed:<message>`` -> ``nlloc_failed``), since the
NLLoc failure message embeds the event id.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import setup_style
from openmeq.plotting._maps import _plot_map_panel, _plot_section_panel
from openmeq.plotting._style import COLORBLIND_PALETTE, SEQUENTIAL_CMAP
from openmeq.plotting._util import auto_scale

if TYPE_CHECKING:
    from openmeq.locate.locate_step import LocateRunResult

logger = get_logger("locate.diagnostics_compare")

PASS_COMPARISON_PDF = "pass_comparison.pdf"

_COLORS = {"A": COLORBLIND_PALETTE[0], "B": COLORBLIND_PALETTE[1]}

_QUALITY_COLUMNS = (
    ("rms", "RMS (s)"),
    ("gap", "Azimuthal gap (deg)"),
    ("err_h_km", "Horizontal error (km)"),
    ("err_z_km", "Depth error (km)"),
)


def _shared_bins(arrays: list[np.ndarray], n: int) -> np.ndarray | None:
    finite = [a[np.isfinite(a)] for a in arrays]
    finite = [a for a in finite if a.size]
    if not finite:
        return None
    lo = min(float(a.min()) for a in finite)
    hi = max(float(a.max()) for a in finite)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n + 1)


def _overlay_hist(ax, data: dict[str, np.ndarray], bins: np.ndarray | None) -> None:
    """Step histograms of each pass on the same bins (or a "No data" note)."""
    if bins is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    for label, values in data.items():
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.4,
                color=_COLORS[label],
                label=f"Pass {label} (n={values.size})",
            )
    if ax.has_data():
        ax.legend(fontsize=8)


def _page_map(pdf, frames: dict[str, pd.DataFrame], stations: pd.DataFrame | None) -> None:
    import matplotlib.pyplot as plt

    located = {k: v.dropna(subset=["lat", "lon", "depth_km"]) for k, v in frames.items()}
    union = pd.concat([v for v in located.values() if not v.empty] or [pd.DataFrame()])
    rms_all = union["rms"].to_numpy(dtype=float) if "rms" in union else np.array([])
    vmin, vmax = auto_scale(rms_all, None, None)

    fig, axes = plt.subplots(len(located), 3, figsize=(15, 4.6 * len(located)), squeeze=False)
    mappable = None
    for row, (label, df) in enumerate(located.items()):
        colors = df["rms"].to_numpy(dtype=float) if not df.empty else None
        common = dict(color_values=colors, cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax)
        m = _plot_map_panel(
            axes[row][0],
            df,
            stations,
            lon_col="lon",
            lat_col="lat",
            station_lon_col="longitude",
            station_lat_col="latitude",
            marker_size=18.0,
            station_marker_size=50.0,
            **common,
        )
        mappable = mappable or m
        _plot_section_panel(
            axes[row][1],
            df,
            "lon",
            "depth_km",
            stations,
            "longitude",
            xlabel="Longitude (deg)",
            title="E-W section",
            marker_size=18.0,
            **common,
        )
        _plot_section_panel(
            axes[row][2],
            df,
            "lat",
            "depth_km",
            stations,
            "latitude",
            xlabel="Latitude (deg)",
            title="N-S section",
            marker_size=18.0,
            **common,
        )
        axes[row][0].set_title(f"{label} ({len(df)} events): map view")

    # Same limits in every row, per column (sections share the depth axis).
    for col in range(3):
        col_axes = [axes[r][col] for r in range(len(located))]
        xs = [ax.get_xlim() for ax in col_axes]
        ys = [ax.get_ylim() for ax in col_axes]
        x_lim = (min(x[0] for x in xs), max(x[1] for x in xs))
        if col == 0:
            y_lim = (min(y[0] for y in ys), max(y[1] for y in ys))
        else:  # inverted depth axes: (bottom=max depth, top=min depth)
            y_lim = (max(y[0] for y in ys), min(y[1] for y in ys))
        for ax in col_axes:
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    if mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), label="RMS (s)", shrink=0.6)
    fig.suptitle("Event locations: pass A vs pass B vs accepted (shared axes and RMS scale)")
    pdf.savefig(fig)
    plt.close(fig)


def _page_quality(pdf, frames: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (column, xlabel) in zip(axes.ravel(), _QUALITY_COLUMNS, strict=True):
        data = {k: frames[k][column].to_numpy(dtype=float) for k in ("A", "B")}
        _overlay_hist(ax, data, _shared_bins(list(data.values()), 30))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event count")
    fig.suptitle("Location quality: pass A vs pass B (same bins)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _page_residuals(pdf, residuals: dict[str, pd.DataFrame]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, phase in zip(axes, ("P", "S"), strict=True):
        data = {
            k: df.loc[df["phase"] == phase, "residual_s"].to_numpy(dtype=float)
            for k, df in residuals.items()
        }
        _overlay_hist(ax, data, _shared_bins(list(data.values()), 40))
        ax.set_xlabel("Time residual (s)")
        ax.set_ylabel("Count")
        only_b = " -- pass B only" if phase == "S" else ""
        ax.set_title(f"{phase} residuals (all stations){only_b}")
    fig.suptitle("Residuals: pass A vs pass B (same bins; pass A locates with P only)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def _page_residuals_per_station(pdf, residuals: dict[str, pd.DataFrame], phase: str) -> None:
    import matplotlib.pyplot as plt

    by_pass = {k: df[df["phase"] == phase] for k, df in residuals.items()}
    stations = sorted(set().union(*(set(df["net_sta"]) for df in by_pass.values())))
    title = f"{phase} residuals per station: pass A vs pass B (same bins per station)"
    if phase == "S":
        title += " -- pass A has no S"
    if not stations:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        pdf.savefig(fig)
        plt.close(fig)
        return

    ncols = min(4, len(stations))
    nrows = -(-len(stations) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    for i, sta in enumerate(stations):
        ax = axes[i // ncols][i % ncols]
        data = {
            k: df.loc[df["net_sta"] == sta, "residual_s"].to_numpy(dtype=float)
            for k, df in by_pass.items()
        }
        bins = _shared_bins(list(data.values()), 15)
        if bins is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        for label, values in data.items():
            values = values[np.isfinite(values)]
            if values.size and bins is not None:
                ax.hist(values, bins=bins, histtype="step", linewidth=1.2, color=_COLORS[label])
        counts = ", ".join(f"{k}:{np.isfinite(v).sum()}" for k, v in data.items())
        ax.set_title(f"{sta} ({counts})", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(stations), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [
        plt.Line2D([], [], color=_COLORS[k], linewidth=1.4, label=f"Pass {k}") for k in _COLORS
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _grouped_bars(ax, counts: dict[str, pd.Series], title: str) -> None:
    reasons = sorted(set().union(*(set(c.index) for c in counts.values())))
    if not reasons:
        ax.text(0.5, 0.5, "None removed", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    x = np.arange(len(reasons))
    width = 0.8 / len(counts)
    for i, (label, series) in enumerate(counts.items()):
        values = [int(series.get(r, 0)) for r in reasons]
        bars = ax.bar(x + (i - (len(counts) - 1) / 2) * width, values, width, color=_COLORS[label])
        bars.set_label(f"Pass {label} ({sum(values)})")
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _page_removals(pdf, removal_log: pd.DataFrame, event_counts: dict[str, pd.Series]) -> None:
    import matplotlib.pyplot as plt

    pick_counts = {}
    for label in ("A", "B"):
        sub = removal_log[removal_log["pass"] == label] if not removal_log.empty else removal_log
        pick_counts[label] = (
            (sub["phase"].astype(str) + ": " + sub["reason"].astype(str)).value_counts()
            if not sub.empty
            else pd.Series(dtype="int64")
        )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    _grouped_bars(axes[0], pick_counts, "Picks removed, by phase: reason")
    _grouped_bars(axes[1], event_counts, "Events rejected, by reason (pass that rejected them)")
    fig.suptitle("Removals: pass A vs pass B")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    pdf.savefig(fig)
    plt.close(fig)


def make_pass_comparison_pdf(
    path: Path,
    result: LocateRunResult,
    removal_log: pd.DataFrame,
    stations: pd.DataFrame | None,
) -> Path:
    """Write :data:`PASS_COMPARISON_PDF`-style multi-page A-vs-B comparison to ``path``."""
    from matplotlib.backends.backend_pdf import PdfPages

    from openmeq.locate.diagnostics import (
        _event_removal_counts_by_reason,
        _pass_dataframe,
        _residual_dataframe,
    )

    setup_style()
    import matplotlib.pyplot as plt

    frames = {"A": _pass_dataframe(result.pass_a), "B": _pass_dataframe(result.pass_b)}
    accepted_ids = {r.event_id for r in result.summary_rows if r.status == "accepted"}
    map_frames = {
        "Pass A": frames["A"],
        "Pass B": frames["B"],
        "Accepted": frames["B"][frames["B"]["event_id"].isin(accepted_ids)],
    }
    residuals = {"A": _residual_dataframe(result.pass_a), "B": _residual_dataframe(result.pass_b)}
    event_counts = {}
    for label in ("A", "B"):
        counts = _event_removal_counts_by_reason(result.summary_rows, result.pass_a, label)
        event_counts[label] = (
            counts.groupby(counts.index.str.split(":").str[0]).sum() if not counts.empty else counts
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        try:
            _page_map(pdf, map_frames, stations)
            _page_quality(pdf, frames)
            _page_residuals(pdf, residuals)
            for phase in ("P", "S"):
                _page_residuals_per_station(pdf, residuals, phase)
            _page_removals(pdf, removal_log, event_counts)
        finally:
            plt.close("all")
    logger.info("locate diagnostics: pass A vs B comparison -> %s", path)
    return path


__all__ = ["PASS_COMPARISON_PDF", "make_pass_comparison_pdf"]
