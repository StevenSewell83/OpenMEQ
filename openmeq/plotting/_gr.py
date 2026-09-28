"""Cumulative Gutenberg-Richter magnitude-frequency plot.

Added by M8.3 (``OpenMEQ-625.3``, ``docs/PLAN.md`` §5's magnitude
diagnostics row: "a Gutenberg-Richter plot"). Distinct from
:func:`~._hist.plot_hist`'s per-bin *count* histogram: a Gutenberg-Richter
plot needs the **cumulative** count of events with magnitude ``>= M``
(unbinned, one point per event, log10 y-axis), which no existing primitive
in this package produces -- a genuinely new shape, per
``docs/ARCHITECTURE.md`` §2's shared-module reuse rule (add only if
actually missing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._save import save_fig
from ._style import setup_style


def plot_gutenberg_richter(
    magnitudes,
    path: str | Path,
    *,
    xlabel: str = "Magnitude (ML)",
    ylabel: str = "N (>= M)",
    title: str = "Magnitude-frequency (Gutenberg-Richter)",
) -> None:
    """Draw a cumulative magnitude-frequency plot and save it.

    Each event contributes one point: for event ``i`` with magnitude
    ``m_i``, ``y_i`` is the count of events (across the whole run) with
    magnitude ``>= m_i`` -- an unbinned empirical complementary cumulative
    count, plotted on a log10 y-axis against magnitude on the x-axis, the
    standard Gutenberg-Richter presentation.

    Parameters
    ----------
    magnitudes : array-like
        One magnitude per event. NaNs are dropped. May be empty (produces
        an empty axes with a "no data" annotation rather than raising).
    path : str or pathlib.Path
        Output PNG path.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    """
    setup_style()
    import matplotlib.pyplot as plt

    arr = np.asarray(magnitudes, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    fig, ax = plt.subplots()
    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        sorted_desc = np.sort(arr)[::-1]
        counts = np.arange(1, sorted_desc.size + 1)
        ax.scatter(sorted_desc, counts, s=18, color="#0072B2", edgecolor="none")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    save_fig(fig, path)
