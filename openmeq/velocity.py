"""1D layered velocity model file parsing.

Implements the ``read_layers`` side of the plain-text 1D layer-file format
fixed by ``docs/ARCHITECTURE.md`` §6a, used by NLLoc setup (``LAYER``
statements, M5), hypoDD 1D setup (M9), and the synthetic test dataset
generator (``tests/data/make_synthetic.py``, M0.8), so all three consume the
exact same on-disk format.

File format (ARCHITECTURE.md §6a)
----------------------------------
Whitespace-separated columns, one layer per line::

    # depth_top_km  vp_km_s  vs_km_s  [rho_g_cc]
    0.0   3.50  2.00  2.3
    1.0   4.50  2.60  2.5
    3.0   5.80  3.35  2.7

- A ``#`` starts a comment (rest of the line ignored); blank lines are
  skipped.
- ``depth_top_km`` is the depth to the *top* of each layer, measured below
  sea level (a negative value is above sea level), and must strictly
  increase from one line to the next -- the first layer applies from the
  top of the model down to the next line's depth, and the last layer
  extends to infinity.
- Each layer is constant-velocity (no gradients).
- ``rho_g_cc`` (density) is optional; layers that omit it default to
  ``2.7`` g/cc. A file may mix layers that specify it with layers that
  don't.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Default density (g/cc) used for a layer line that omits the column.
DEFAULT_RHO = 2.7


def read_layers(path: str | Path) -> pd.DataFrame:
    """Parse a 1D layer file into a table of layers.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the layer file (ARCHITECTURE.md §6a format, see module
        docstring).

    Returns
    -------
    pandas.DataFrame
        Columns ``depth_km`` (depth to the top of the layer, below sea
        level; negative = above sea level), ``vp`` (km/s), ``vs`` (km/s),
        ``rho`` (g/cc, defaulting to :data:`DEFAULT_RHO` where omitted).
        Rows are sorted ascending by ``depth_km`` (the file is required to
        already be in that order; this only guards against an
        out-of-order-but-otherwise-valid file).

    Raises
    ------
    ValueError
        If the file has no layers, a line doesn't have at least 3 columns,
        a column isn't numeric, or ``depth_km`` does not strictly increase
        from one layer to the next.
    """
    path = Path(path)
    rows: list[tuple[float, float, float, float]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{lineno}: expected at least 3 columns "
                    f"(depth_top_km vp_km_s vs_km_s [rho_g_cc]), got {len(parts)}: {raw_line!r}"
                )
            try:
                depth_km = float(parts[0])
                vp = float(parts[1])
                vs = float(parts[2])
                rho = float(parts[3]) if len(parts) >= 4 else DEFAULT_RHO
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-numeric column: {raw_line!r}") from exc
            rows.append((depth_km, vp, vs, rho))

    if not rows:
        raise ValueError(f"{path}: no layers found")

    rows.sort(key=lambda r: r[0])
    depths = [r[0] for r in rows]
    if any(b <= a for a, b in zip(depths, depths[1:], strict=False)):
        raise ValueError(f"{path}: layer depth_top_km values must strictly increase: {depths}")

    return pd.DataFrame(rows, columns=["depth_km", "vp", "vs", "rho"])
