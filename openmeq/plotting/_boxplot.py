"""Grouped box plots: QC monthly station box plots and rankings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._save import save_fig
from ._style import setup_style


def plot_boxplot(
    df: pd.DataFrame,
    path: str | Path,
    *,
    by: str,
    column: str,
    order: list | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    rotate_xticks: float = 45.0,
) -> None:
    """Draw a box plot of ``column`` grouped by ``by`` and save it.

    Used for QC monthly station box plots/rankings (e.g. one box per
    station of a daily-stat ratio-to-network-median).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form data with a grouping column ``by`` and a value column
        ``column``.
    path : str or pathlib.Path
        Output PNG path.
    by : str
        Column to group by (one box per distinct value).
    column : str
        Value column to summarize per group.
    order : list, optional
        Explicit group order (e.g. sorted by rank/median). Defaults to
        sorted group labels.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    rotate_xticks : float, optional
        Rotation (degrees) for group labels on the x-axis.
    """
    setup_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.4 * df[by].nunique()) if df is not None and not df.empty else 6.0, 5.0)
    )

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        groups = order if order is not None else sorted(df[by].dropna().unique())
        data = [df.loc[df[by] == g, column].dropna().to_numpy() for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups], showfliers=True)
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    ax.set_xlabel(xlabel or by)
    ax.set_ylabel(ylabel or column)
    if title:
        ax.set_title(title)

    save_fig(fig, path)
