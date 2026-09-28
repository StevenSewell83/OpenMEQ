"""Shared map/section/histogram plotting functions for diagnostics.

Every step's ``<step>/diagnostics/`` output is built from the functions in
this package (see ``docs/ARCHITECTURE.md`` §2 and ``docs/PLAN.md`` §5), so
the same map, section, histogram, heatmap, time series and box plot code is
reused across steps instead of being reimplemented per step.

All functions here:

- use the non-interactive Agg backend (selected as soon as this package is
  imported, before ``matplotlib.pyplot`` is touched anywhere else), so they
  are safe to call from a headless CI runner or from a worker process
  spawned by ``openmeq.parallel.pmap``;
- take plain data (``pandas.DataFrame``/arrays/paths) and keyword styling
  options -- nothing here depends on ``openmeq.config``;
- always close the figure they create (via :func:`save_fig`), so a batch of
  thousands of per-event/per-channel diagnostic plots never leaks figures;
- accept a ``vmin``/``vmax`` colour-scale override but default to a
  percentile-based auto range, so the scale adapts to datasets of very
  different magnitude.

This module does not parallelize internally -- matplotlib figures are not
safely shared across processes -- so callers rendering many small PNGs
(e.g. per-event or per-channel diagnostics) should do so via
``openmeq.parallel.pmap``, with each call rendering and saving one figure in
its own worker process.
"""

from __future__ import annotations

# Import order matters: ``_style`` selects the Agg backend as its first
# action, before any other submodule (or anything else in this process) can
# import ``matplotlib.pyplot``.
from ._bar import plot_bar
from ._boxplot import plot_boxplot
from ._heatmap import plot_heatmap
from ._hist import plot_hist
from ._maps import plot_map_sections
from ._save import save_fig
from ._style import setup_style
from ._timeseries import plot_timeseries

__all__ = [
    "setup_style",
    "save_fig",
    "plot_map_sections",
    "plot_hist",
    "plot_heatmap",
    "plot_timeseries",
    "plot_boxplot",
    "plot_bar",
]
