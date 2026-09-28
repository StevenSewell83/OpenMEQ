"""2D heatmap: QC channel x day availability, and similar matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._save import save_fig
from ._style import SEQUENTIAL_CMAP, setup_style
from ._util import auto_scale, thin_ticks


def plot_heatmap(
    matrix_or_df,
    path: str | Path,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    x_labels: list | None = None,
    y_labels: list | None = None,
    max_x_ticks: int = 25,
    max_y_ticks: int = 40,
) -> None:
    """Draw a 2D heatmap (e.g. channel x day QC availability) and save it.

    Parameters
    ----------
    matrix_or_df : pandas.DataFrame or array-like
        2D data. A DataFrame's index/columns are used as row/column labels
        unless ``y_labels``/``x_labels`` are given explicitly.
    path : str or pathlib.Path
        Output PNG path.
    xlabel, ylabel, title : str, optional
        Axis labels and title.
    cmap : str, optional
        Colormap (default: the package's sequential colormap).
    vmin, vmax : float, optional
        Explicit colour-scale bounds. If not given, an auto range is used
        (2nd/98th percentile of the finite values) so the scale is not
        hard-coded to one dataset's magnitude.
    cbar_label : str, optional
        Colorbar label.
    x_labels, y_labels : list, optional
        Explicit tick labels, overriding a DataFrame's index/columns.
    max_x_ticks, max_y_ticks : int, optional
        Tick labels are thinned to at most this many so a heatmap with many
        columns (e.g. one per day over a long deployment) stays legible.
    """
    setup_style()
    import matplotlib.pyplot as plt

    if isinstance(matrix_or_df, pd.DataFrame):
        matrix = matrix_or_df.to_numpy(dtype=float)
        if y_labels is None:
            y_labels = list(matrix_or_df.index)
        if x_labels is None:
            x_labels = list(matrix_or_df.columns)
    else:
        matrix = np.asarray(matrix_or_df, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

    n_rows, n_cols = matrix.shape if matrix.size else (0, 0)
    if y_labels is None:
        y_labels = list(range(n_rows))
    if x_labels is None:
        x_labels = list(range(n_cols))

    vmin_auto, vmax_auto = auto_scale(matrix, vmin, vmax)

    fig, ax = plt.subplots(
        figsize=(max(6.0, min(0.35 * n_cols, 16.0)), max(4.0, min(0.3 * n_rows, 12.0)))
    )
    im = ax.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=vmin_auto, vmax=vmax_auto, interpolation="nearest"
    )

    x_idx, x_thinned = thin_ticks(x_labels, max_x_ticks)
    y_idx, y_thinned = thin_ticks(y_labels, max_y_ticks)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_thinned, rotation=45, ha="right")
    ax.set_yticks(y_idx)
    ax.set_yticklabels(y_thinned)

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.9)

    save_fig(fig, path)
