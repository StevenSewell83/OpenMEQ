"""Categorical bar chart: counts (or any value) by discrete label.

Added by M2.3 (``OpenMEQ-ujx.3``) for detection's station-participation
diagnostic (PLAN.md §5's "Detect" row) -- none of the other primitives in
this package fit a discrete-category count: :func:`~._hist.plot_hist` bins a
*numeric* value, and :func:`~._boxplot.plot_boxplot` shows a distribution per
group, not a single count/value per group.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ._save import save_fig
from ._style import setup_style


def plot_bar(
    categories: Sequence[str],
    values: Sequence[float],
    path: str | Path,
    *,
    xlabel: str | None = None,
    ylabel: str = "Count",
    title: str | None = None,
    rotate_xticks: float = 45.0,
    color: str | None = None,
) -> None:
    """Draw a categorical bar chart (one bar per category) and save it.

    Used for counts-by-discrete-category diagnostics (e.g. detections per
    station) that :func:`~._hist.plot_hist`'s numeric binning can't
    represent -- each x position here is a caller-supplied label, not a bin
    edge.

    Parameters
    ----------
    categories : sequence of str
        One label per bar, in the order to draw them. This function never
        sorts or reorders -- sort ``categories``/``values`` together first
        if a specific order (e.g. by count, or alphabetical) is wanted.
    values : sequence of float
        One value per category, same length and order as ``categories``.
    path : str or pathlib.Path
        Output PNG path.
    xlabel, ylabel : str, optional
        Axis labels. ``ylabel`` defaults to ``"Count"``.
    title : str, optional
        Axes title.
    rotate_xticks : float, optional
        Rotation (degrees) for category labels on the x-axis (default 45,
        readable for typical ``NET.STA``-length labels without overlapping).
    color : str, optional
        Bar colour (defaults to the style's first cycle colour).
    """
    setup_style()
    import matplotlib.pyplot as plt

    categories = list(categories)
    values = list(values)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(categories)), 5.0))
    if not categories:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar(categories, values, color=color or "#0072B2", edgecolor="white", linewidth=0.5)
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    save_fig(fig, path)
