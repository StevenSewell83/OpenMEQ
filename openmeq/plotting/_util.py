"""Small shared helpers used by several plot functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def auto_scale(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> tuple[float | None, float | None]:
    """Return a ``(vmin, vmax)`` colour-scale range.

    Explicit ``vmin``/``vmax`` are passed through unchanged. Otherwise the
    range is the ``low_pct``/``high_pct`` percentile of the finite values,
    so a colour scale never hard-codes an absolute range that would break on
    data very different in scale from the value it was tuned on (e.g. the
    synthetic test dataset vs. a real deployment). Returns ``(None, None)``
    if there is no finite data.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return vmin, vmax
    lo = float(np.percentile(finite, low_pct)) if vmin is None else vmin
    hi = float(np.percentile(finite, high_pct)) if vmax is None else vmax
    if lo == hi:
        hi = lo + 1e-9
    return lo, hi


def thin_ticks(labels: Sequence, max_ticks: int) -> tuple[list[int], list]:
    """Pick at most ``max_ticks`` evenly-spaced (index, label) pairs.

    Used to keep tick labels legible on heatmaps/timeseries with many
    columns (e.g. one column per day over a long deployment).
    """
    n = len(labels)
    if n <= max_ticks or max_ticks <= 0:
        return list(range(n)), list(labels)
    step = max(1, n // max_ticks)
    idx = list(range(0, n, step))
    return idx, [labels[i] for i in idx]
