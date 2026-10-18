"""Row-grouped heatmap: one labelled row per channel/station, rows grouped by
instrument type with separators and group names (QC diagnostics)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ._save import save_fig
from ._style import setup_style

#: Background for cells with no value (missing data).
MISSING_GREY = "#d9d9d9"


def plot_grouped_heatmap(
    matrix: pd.DataFrame,
    path: str | Path,
    *,
    row_labels: Sequence[str] | None = None,
    row_groups: Sequence[str] | None = None,
    title: str | None = None,
    note: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    log: bool = False,
    cbar_label: str | None = None,
    cbar_ticks: dict[float, str] | None = None,
    categories: Sequence[tuple[float, str, str]] | None = None,
    annotate: Sequence[Sequence[str]] | None = None,
) -> None:
    """Draw ``matrix`` (rows x columns) as a heatmap with grouped, labelled rows.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Rows in display order (top to bottom). Columns that form a
        ``DatetimeIndex`` get a date axis; anything else is labelled
        categorically (e.g. ``YYYY-MM`` months). NaN cells show as grey.
    path : str or pathlib.Path
        Output PNG path.
    row_labels : sequence of str, optional
        One y tick label per row (default: ``matrix.index``).
    row_groups : sequence of str, optional
        One group name per row. Rows are expected to be contiguous per
        group; a white line separates groups and each group's name is
        written on the right-hand side.
    title, note : str, optional
        Figure title; a small note right-aligned above the axes.
    cmap, vmin, vmax : optional
        Continuous colour scale. ``vmin``/``vmax`` default to the 2nd/98th
        percentile of the finite values.
    log : bool, optional
        Logarithmic continuous colour scale (non-positive values are drawn
        as missing).
    cbar_label : str, optional
        Colour bar label.
    cbar_ticks : dict, optional
        ``{value: label}`` colour bar ticks (e.g. ratio labels on a
        log-ratio scale).
    categories : sequence of (value, colour, label), optional
        Categorical mode: cells hold one of the ``value``s, drawn in its
        ``colour`` and listed in a legend instead of a colour bar.
    annotate : 2-D sequence of str, optional
        Per-cell text, same shape as ``matrix`` (use for few columns).
    """
    setup_style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch

    labels = list(matrix.index if row_labels is None else row_labels)
    values = matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    is_dates = isinstance(matrix.columns, pd.DatetimeIndex)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 1.4 + 0.2 * max(n_rows, 1))))
    ax.set_facecolor(MISSING_GREY)
    ax.grid(False)
    x = matrix.columns if is_dates else np.arange(n_cols)
    y = np.arange(n_rows)

    kwargs: dict = {}
    if categories:
        cat_values = [c[0] for c in categories]
        bounds = [v - 0.5 for v in cat_values] + [cat_values[-1] + 0.5]
        kwargs = {
            "cmap": ListedColormap([c[1] for c in categories]),
            "norm": BoundaryNorm(bounds, len(categories)),
        }
    else:
        shown = values[np.isfinite(values) & ((values > 0) if log else True)]
        if shown.size:
            lo, hi = np.percentile(shown, [2, 98])
            lo = lo if vmin is None else vmin
            hi = hi if vmax is None else vmax
            if lo == hi:
                lo, hi = (lo / 2, hi * 2) if log else (lo - 1.0, hi + 1.0)
            if log:
                values = np.where(values > 0, values, np.nan)
                kwargs = {"cmap": cmap, "norm": LogNorm(lo, hi)}
            else:
                kwargs = {"cmap": cmap, "vmin": lo, "vmax": hi}

    if n_rows and n_cols and np.isfinite(values).any() and kwargs:
        im = ax.pcolormesh(x, y, values, shading="nearest", **kwargs)
        if categories:
            present = set(np.unique(values[np.isfinite(values)]))
            handles = [Patch(color=c, label=lab) for v, c, lab in categories if v in present]
            handles.append(Patch(color=MISSING_GREY, label="no data"))
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=8,
                frameon=False,
            )
        else:
            cbar = fig.colorbar(
                im, ax=ax, fraction=0.025, pad=0.07 if row_groups else 0.01, label=cbar_label
            )
            if cbar_ticks:
                cbar.set_ticks(list(cbar_ticks))
                cbar.set_ticklabels(list(cbar_ticks.values()))

    if annotate is not None:
        for i in range(n_rows):
            for j in range(n_cols):
                text = annotate[i][j]
                if text:
                    ax.text(j, i, text, ha="center", va="center", fontsize=7)

    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_ylim(n_rows - 0.5, -0.5)
    if is_dates:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        # offset_formats: show just the year under the axis, not the last month.
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(
                locator, offset_formats=["", "%Y", "%Y", "%Y", "%Y", "%Y-%b-%d"]
            )
        )
    else:
        ax.set_xticks(x, labels=[str(c) for c in matrix.columns])

    if row_groups is not None and n_rows:
        groups = np.asarray(list(row_groups), dtype=object)
        starts = [0, *list(np.flatnonzero(groups[1:] != groups[:-1]) + 1)]
        ends = [*starts[1:], n_rows]
        for b in starts[1:]:
            ax.axhline(b - 0.5, color="white", lw=2.0)
        trans = ax.get_yaxis_transform()
        for s, e in zip(starts, ends, strict=True):
            ax.text(
                1.01,
                (s + e - 1) / 2,
                str(groups[s]),
                transform=trans,
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
                rotation=90 if e - s > 3 else 0,
            )

    if title:
        ax.set_title(title, loc="left")
    if note:
        ax.text(1.0, 1.01, note, ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    save_fig(fig, path)
