"""Generic x/y scatter plot: station ML vs distance, ML-through-time, ...

Added by M8.3 (``OpenMEQ-625.3``, ``docs/PLAN.md`` §5's magnitude diagnostics
row). Distinct from :func:`~._timeseries.plot_timeseries`, which always
connects points with a line (the right shape for a daily-stat time series,
but wrong for an unordered/non-monotonic relationship like magnitude vs
distance, or a sparse per-event scatter where a connecting line would
visually imply an ordering/trend that isn't there) -- none of this
package's other existing primitives (``plot_hist``'s numeric binning,
``plot_bar``'s discrete categories, ``plot_boxplot``'s per-group
distribution) fit an unconnected x/y scatter either, so this is a genuinely
new shape (``docs/ARCHITECTURE.md`` §2's shared-module reuse rule: add only
if actually missing).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._save import save_fig
from ._style import setup_style


def plot_scatter(
    df: pd.DataFrame,
    path: str | Path,
    *,
    x: str,
    y: str,
    group_by: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    max_legend_entries: int = 15,
    s: float = 18.0,
    alpha: float = 0.8,
) -> None:
    """Draw an unconnected x/y scatter (one point per row) and save it.

    Parameters mirror :func:`~._timeseries.plot_timeseries` (``group_by``
    draws one colour per group, with the same legend-size cutoff), but
    never draws connecting lines between points.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form data with at least an ``x`` and a ``y`` column. May be
        empty (produces an empty axes with a "no data" annotation rather
        than raising).
    path : str or pathlib.Path
        Output PNG path.
    x, y : str
        Column names for the x- and y-axis values.
    group_by : str, optional
        Column name to colour points by (one colour per group). If
        omitted, every point is drawn in a single colour.
    xlabel, ylabel, title : str, optional
        Axis labels and title (default to the column names).
    max_legend_entries : int, optional
        A legend is only drawn if there are at most this many groups, to
        avoid an unreadable legend for e.g. hundreds of stations.
    s : float, optional
        Marker size (``matplotlib.axes.Axes.scatter``'s ``s``).
    alpha : float, optional
        Marker transparency, useful when points overlap.
    """
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    has_data = df is not None and not df.empty
    is_datetime_x = has_data and pd.api.types.is_datetime64_any_dtype(df[x])

    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif group_by is None:
        ax.scatter(df[x], df[y], s=s, alpha=alpha, color="#0072B2", edgecolor="none")
    else:
        groups = list(df.groupby(group_by, sort=True))
        show_legend = len(groups) <= max_legend_entries
        for name, sub in groups:
            ax.scatter(
                sub[x],
                sub[y],
                s=s,
                alpha=alpha,
                edgecolor="none",
                label=str(name) if show_legend else None,
            )
        if show_legend:
            ax.legend(loc="best", ncol=max(1, len(groups) // 8 + 1))

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if is_datetime_x:
        fig.autofmt_xdate()

    save_fig(fig, path)
