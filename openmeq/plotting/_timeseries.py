"""Time series: QC daily stats per channel, ML-through-time, etc."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._save import save_fig
from ._style import setup_style


def plot_timeseries(
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
    marker: str | None = "o",
) -> None:
    """Draw one or more time series (one line per ``group_by`` group).

    Used for QC daily-stat time series per channel (``group_by="seed_id"``)
    and for ML-through-time (no ``group_by``, one line).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form data with at least an ``x`` and a ``y`` column.
    path : str or pathlib.Path
        Output PNG path.
    x, y : str
        Column names for the x- and y-axis values.
    group_by : str, optional
        Column name to draw one line per group (e.g. per channel/station).
        If omitted, all rows are drawn as a single series.
    xlabel, ylabel, title : str, optional
        Axis labels and title (defaults to the column names).
    max_legend_entries : int, optional
        A legend is only drawn if there are at most this many groups, to
        avoid an unreadable legend when there are e.g. hundreds of
        channels; the lines are still all drawn.
    marker : str, optional
        Marker style for data points (``None`` for lines only).
    """
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    elif group_by is None:
        sub = df.sort_values(x)
        ax.plot(sub[x], sub[y], marker=marker, linewidth=1.2)
    else:
        groups = list(df.groupby(group_by, sort=True))
        show_legend = len(groups) <= max_legend_entries
        for name, sub in groups:
            sub = sub.sort_values(x)
            ax.plot(
                sub[x],
                sub[y],
                marker=marker,
                linewidth=1.0,
                label=str(name) if show_legend else None,
            )
        if show_legend:
            ax.legend(loc="best", ncol=max(1, len(groups) // 8 + 1))

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    fig.autofmt_xdate()

    save_fig(fig, path)
