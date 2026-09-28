"""Matplotlib backend selection and shared visual style.

This module is imported first by :mod:`openmeq.plotting` (before any other
submodule) so the Agg backend is selected before ``matplotlib.pyplot`` is
imported anywhere else, regardless of import order elsewhere in the process.
This is what makes the package safe to use from a worker process spawned by
``openmeq.parallel.pmap``: each worker imports this package fresh, gets its
own Agg backend, and never touches a display.
"""

from __future__ import annotations

import matplotlib

# Force the non-interactive Agg backend. This must happen before
# ``matplotlib.pyplot`` is imported for the first time in this process.
# ``force=True`` makes it safe even if something upstream already selected a
# (different) backend; batch diagnostics code never needs a display.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

# Okabe-Ito colour-blind-safe qualitative palette, used everywhere a
# categorical colour cycle is needed (line series, station markers, ...).
COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Perceptually-uniform, colour-blind-safe sequential colormap used as the
# default for scatter/heatmap colour scales.
SEQUENTIAL_CMAP = "viridis"

DEFAULT_DPI = 150


def setup_style() -> None:
    """Select the Agg backend and apply the shared rcParams style.

    Safe (and cheap) to call more than once per process, and safe to call
    from multiple worker processes: it only ever touches process-local
    matplotlib state (``matplotlib.rcParams``), never anything shared
    across processes. Every public plotting function in this package calls
    this internally, so callers do not strictly need to call it themselves,
    but it is exposed so a worker initialiser or CLI entry point can call it
    once up front (e.g. before a batch of ``openmeq.parallel.pmap`` calls).
    """
    matplotlib.use("Agg", force=True)
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "figure.dpi": 100,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.prop_cycle": plt.cycler(color=COLORBLIND_PALETTE),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "legend.fontsize": 9,
            "legend.framealpha": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.markersize": 4,
            "image.cmap": SEQUENTIAL_CMAP,
        }
    )
