"""Generic histogram: residuals, CC distributions, magnitude-frequency, ..."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ._save import save_fig
from ._style import setup_style


def plot_hist(
    values,
    path: str | Path,
    *,
    bins: int | Sequence[float] = 30,
    xlabel: str | None = None,
    ylabel: str = "Count",
    title: str | None = None,
    vlines: Mapping[str, float] | Sequence[float] | None = None,
    color: str | None = None,
    log_y: bool = False,
) -> None:
    """Draw a histogram of ``values`` and save it to ``path``.

    Generic enough for pick residuals, CC-value distributions,
    magnitude-frequency (Gutenberg-Richter) plots, RMS/gap/error
    distributions, etc.

    Parameters
    ----------
    values : array-like
        Values to histogram. NaNs are dropped. May be empty (produces an
        empty axes with a "no data" annotation rather than raising).
    path : str or pathlib.Path
        Output PNG path.
    bins : int or sequence of float, optional
        Passed through to ``numpy``/``matplotlib`` histogram binning.
    xlabel, ylabel : str, optional
        Axis labels. Always include units in ``xlabel`` (e.g.
        ``"Residual (s)"``, ``"Distance (km)"``).
    title : str, optional
        Axes title.
    vlines : mapping or sequence of float, optional
        Vertical reference lines, e.g. a filter threshold. A mapping of
        ``{label: x}`` draws labelled lines with a legend; a plain sequence
        draws unlabelled lines.
    color : str, optional
        Bar colour (defaults to the style's first cycle colour).
    log_y : bool, optional
        Use a log-scaled y-axis (useful for long-tailed distributions).
    """
    setup_style()
    import matplotlib.pyplot as plt

    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots()
    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(arr, bins=bins, color=color or "#0072B2", edgecolor="white", linewidth=0.5)
        if log_y:
            ax.set_yscale("log")

    if vlines is not None:
        items = vlines.items() if isinstance(vlines, Mapping) else [(None, v) for v in vlines]
        for label, x in items:
            ax.axvline(x, color="#D55E00", linestyle="--", linewidth=1.2, label=label)
        if isinstance(vlines, Mapping) and len(vlines) > 0:
            ax.legend()

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    save_fig(fig, path)
