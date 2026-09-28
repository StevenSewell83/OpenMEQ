"""Figure saving with directory creation and guaranteed cleanup."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt

from ._style import DEFAULT_DPI, setup_style


def save_fig(fig: matplotlib.figure.Figure, path: str | Path, *, dpi: int = DEFAULT_DPI) -> None:
    """Save ``fig`` as a PNG at ``path`` and close it.

    Creates any missing parent directories (including deeply nested
    ``<step>/diagnostics/...`` paths, e.g. per-pass, per-event, per-iteration
    layouts), saves at a fixed dpi, and always closes the figure afterwards
    -- even if saving fails -- so a batch of hundreds/thousands of diagnostic
    plots never leaks open figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or pathlib.Path
        Output PNG path. Parent directories are created as needed.
    dpi : int, optional
        Resolution to save at (default: the package's standard dpi).
    """
    setup_style()
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
