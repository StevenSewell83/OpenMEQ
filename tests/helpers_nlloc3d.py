"""Shared test helper: hand-build a small NLLoc 3D model grid (``.hdr``/``.buf``
pair) directly, without running ``Vel2Grid3D``.

Built for M5.3 / OpenMEQ-mev.3 (``openmeq.nlloc_setup``'s 3D grid path,
``tests/test_nlloc_setup_3d.py``), and deliberately kept independent of
``openmeq`` production code so it's cheap to reuse elsewhere in the test
suite -- in particular M9.4 / OpenMEQ-6ld.4 (the NLLoc-3D -> SIMUL2000
converter for hypoDD) needs its own small 3D-grid input fixtures and should
import from here rather than re-deriving this format.

**Why hand-written, not ``Vel2Grid3D``**: ``Vel2Grid3D`` (docs at
``external/NonLinLoc/docs/programs/core.Vel2Grid3D.rst``) only reads
existing SimulPS/Simul2000/FDTomo-style external velocity-model files, and
picking a sub-format requires editing and recompiling ``Vel2Grid3D.c``'s
``#define`` switches -- there's no control-file-only way to hand it a tiny
synthetic model. Writing the ``.hdr``/``.buf`` pair directly, per the format
this module verifies against the installed NonLinLoc source below, is both
simpler and keeps the fixture arbitrarily small (this project never builds
3D grids for real use anyway -- PLAN.md §1, 3D grids are always
user-supplied -- so nothing here is a stand-in for missing production code).

Grid file format, verified against ``external/NonLinLoc/src/GridLib.c``
(the installed NonLinLoc v7.1.05 18Apr2025 build, docs/nlloc_interface.md
§1 -- no page in ``external/NonLinLoc/docs`` states this as a single worked
example, so every claim below is cited by file+line instead):

``<root>.<waveType>.mod.hdr`` -- a small text file:

- Line 1 (``GridLib.c:1019-1026``, ``WriteGrid3dHdr``)::

      numx numy numz  origx origy origz  dx dy dz gridType floatType

  whitespace-separated, origins/spacings in km. ``floatType`` is ``FLOAT``
  (4-byte) or ``DOUBLE`` (8-byte) depending on the ``GRID_FLOAT_TYPE_DOUBLE``
  compile flag -- **this project's installed build always writes/expects
  FLOAT** (the flag is commented out in
  ``external/NonLinLoc/src/CMakeLists.txt:75``; a mismatched ``floatType``
  makes ``Grid2Time``/``NLLoc`` refuse the grid outright, confirmed by the
  literal string in ``external/bin/Grid2Time``: ``"Global grid float type is
  set to FLOAT, but grid file type is not FLOAT"``).
- The next line is ``TRANSFORM  <type> ...`` (``GridLib.c:1037``) -- always
  present, even for ``TRANS NONE``. A velocity/slowness model grid (this
  helper's only output type) never has the extra source-position line that
  precedes it for a travel-time/angle grid (``GridLib.c:1032-1036``).

``<root>.<waveType>.mod.buf`` -- no header, just ``numx * numy * numz``
native-endian ("little" on this project's x86_64 dev machine) 4-byte floats,
**row-major (C order), z fastest-varying, then y, then x**: ``value[ix, iy,
iz]`` is at byte offset ``4 * (ix*numy*numz + iy*numz + iz)`` -- confirmed
from ``GridLib.c``'s own pointer arithmetic when it builds the in-memory 3D
array view (``GridLib.c:1321``: ``... pgrid->buffer + ix * numyz + iy *
pgrid->numz`` where ``numyz = numy*numz``) and from its disk-offset
computations (``GridLib.c:2160``: ``offset = sizeof(GRID_FLOAT_TYPE) * (ix *
(numy * numz))``; ``GridLib.c:2842``: ``offset = sizeof(GRID_FLOAT_TYPE) *
(ix * numyz + iy * pgrid->numz + iz)``). This is exactly a ``numpy.ndarray``
of shape ``(numx, numy, numz)`` in its default C order, so
``array.astype("<f4").tofile(path)`` round-trips this format precisely
(verified in this module's own tests, see ``if __name__`` note below --
actually verified in ``tests/test_nlloc_setup_3d.py`` by running the real
``Grid2Time``/``NLLoc`` binaries against a grid built this way).

**``SLOW_LEN`` grid values** (the ``gridType`` this helper always uses,
matching ``openmeq.nlloc_setup.build_1d_grid``'s own ``VGGRID`` line and
required by ``run_grid2time``'s Podvin & Lecomte finite-difference method,
``GT_PLFD``): ``value = dx / velocity_km_s`` (``Vel2Grid1.c:477-478``,
confirmed exactly: ``case GRID_SLOW_LEN: ... = grid->dx / vel;``), **using
``dx`` specifically, not the axis's own spacing** -- Podvin-Lecomte itself
requires a cubic grid (``dx == dy == dz``, ``Grid2Time1.c:541-545``:
``"ERROR: Podvin-Lecomte algorithm requires cubic grid, i.e. dx=dy=dz"``),
so this helper always builds one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "write_grid3d",
    "read_grid3d_buf",
    "slow_len_from_velocity",
    "build_small_3d_velocity_grid",
]


def write_grid3d(
    root: str,
    wave_type: str,
    values: np.ndarray,
    *,
    origin_km: tuple[float, float, float],
    spacing_km: float,
    grid_type: str = "SLOW_LEN",
    trans_line: str = "TRANSFORM  NONE",
    suffix: str = "mod",
) -> None:
    """Write ``<root>.<wave_type>.<suffix>.{hdr,buf}`` -- a velocity/slowness
    model grid pair (see module docstring for the exact format).

    Parameters
    ----------
    root : str
        File-root path (no extension), e.g. ``str(tmp_path / "model3d")``.
    wave_type : str
        ``"P"`` or ``"S"``.
    values : numpy.ndarray
        Shape ``(numx, numy, numz)``, any real dtype (cast to little-endian
        float32 on write, matching the installed NLLoc build's ``FLOAT``
        grids -- see module docstring). Use :func:`slow_len_from_velocity`
        to build a physically-meaningful ``SLOW_LEN`` array from a velocity
        field.
    origin_km : tuple[float, float, float]
        ``(origx, origy, origz)`` -- the grid's own ``(0,0,0)`` node, in km,
        in the coordinate frame ``trans_line`` defines.
    spacing_km : float
        Node spacing, used for all three axes (``dx == dy == dz``,
        cubic -- required by the Podvin-Lecomte method this grid will be
        run through, see module docstring).
    grid_type : str, default ``"SLOW_LEN"``
    trans_line : str, default ``"TRANSFORM  NONE"``
        The full ``TRANSFORM ...`` line (see
        ``openmeq.nlloc_setup.check_3d_grid_coverage``: for a grid meant to
        pass that check, this must be
        ``f"TRANSFORM  SIMPLE LatOrig {lat0}  LongOrig {lon0}  RotCW 0.000000"``,
        matching the project's own ``cfg.nlloc.grid.origin`` exactly).
    suffix : str, default ``"mod"``
        ``"mod"`` for a velocity/slowness model grid (the only kind this
        helper writes).
    """
    numx, numy, numz = values.shape
    origx, origy, origz = origin_km
    hdr_path = f"{root}.{wave_type}.{suffix}.hdr"
    buf_path = f"{root}.{wave_type}.{suffix}.buf"
    Path(hdr_path).parent.mkdir(parents=True, exist_ok=True)
    with open(hdr_path, "w") as f:
        f.write(
            f"{numx} {numy} {numz}  {origx} {origy} {origz}  "
            f"{spacing_km} {spacing_km} {spacing_km} {grid_type} FLOAT\n"
        )
        f.write(f"{trans_line}\n")
    values.astype("<f4").tofile(buf_path)


def read_grid3d_buf(
    root: str, wave_type: str, numx: int, numy: int, numz: int, *, suffix: str = "mod"
) -> np.ndarray:
    """Read back ``<root>.<wave_type>.<suffix>.buf`` as a ``(numx, numy,
    numz)`` array (the inverse of :func:`write_grid3d`'s ``values``
    argument) -- mainly for this module's own round-trip tests."""
    buf_path = f"{root}.{wave_type}.{suffix}.buf"
    return np.fromfile(buf_path, dtype="<f4").reshape(numx, numy, numz)


def slow_len_from_velocity(vel_km_s: np.ndarray, dx_km: float) -> np.ndarray:
    """``SLOW_LEN`` grid values (``dx / velocity``, ``Vel2Grid1.c:477-478``
    -- see module docstring) from a velocity (km/s) array."""
    return dx_km / vel_km_s


def build_small_3d_velocity_grid(
    root: str,
    *,
    lat0: float,
    lon0: float,
    x_range_km: tuple[float, float],
    y_range_km: tuple[float, float],
    z_range_km: tuple[float, float],
    spacing_km: float,
    vp0_km_s: float = 4.0,
    vs0_km_s: float = 2.3,
    vp_gradient_z_per_km: float = 0.3,
    vp_gradient_x_per_km: float = 0.08,
) -> tuple[int, int, int]:
    """Build a small, laterally-varying P+S 3D grid pair at
    ``<root>.{P,S}.mod.{hdr,buf}``.

    ``TRANS SIMPLE`` centered at ``(lat0, lon0)`` with no rotation --
    matching ``openmeq.nlloc_setup.write_control_common``'s own ``TRANS``
    line exactly (up to floating-point formatting, which
    ``check_3d_grid_coverage`` tolerates) so the pair validates cleanly
    against a project using that same ``cfg.nlloc.grid.origin``.

    Velocity increases linearly with depth (``z``) and, deliberately, with
    ``x`` (this project's own coverage-check/regression tests need a
    genuinely laterally-varying grid, i.e. ``numx > 2`` and P/S velocity
    that actually differs across x -- a toy gradient, not a physically
    calibrated model). Vp/Vs ratio is a fixed 1.73 scale of the base
    velocities (``vs0_km_s`` given separately since the ratio need not be
    exactly 1.73 at depth/lateral offsets, just plausible).

    Parameters
    ----------
    root : str
        File-root path (no extension).
    lat0, lon0 : float
        ``TRANS SIMPLE`` origin -- pass the same values as the project's
        own ``cfg.nlloc.grid.origin`` for the pair to validate against it.
    x_range_km, y_range_km, z_range_km : tuple[float, float]
        Inclusive ``(min, max)`` extent of the grid on each axis, in the
        ``TRANS SIMPLE`` frame (``z`` positive down, NLLoc's convention).
    spacing_km : float
        Node spacing (cubic grid, all three axes -- see module docstring).
    vp0_km_s, vs0_km_s : float
        Velocity at ``(x, z) = (x_range_km[0], z_range_km[0])``.
    vp_gradient_z_per_km, vp_gradient_x_per_km : float
        Vp increase per km of depth / per km of x (S applies the same
        fractional gradient, scaled by ``vs0_km_s / vp0_km_s``).

    Returns
    -------
    tuple[int, int, int]
        ``(numx, numy, numz)`` actually written (derived from the ranges
        and ``spacing_km``), for tests that need to assert against the
        resulting grid's own shape.
    """
    numx = int(round((x_range_km[1] - x_range_km[0]) / spacing_km)) + 1
    numy = int(round((y_range_km[1] - y_range_km[0]) / spacing_km)) + 1
    numz = int(round((z_range_km[1] - z_range_km[0]) / spacing_km)) + 1

    x = x_range_km[0] + spacing_km * np.arange(numx)
    z = z_range_km[0] + spacing_km * np.arange(numz)

    vp_xz = (
        vp0_km_s
        + vp_gradient_x_per_km * (x[:, None] - x_range_km[0])
        + vp_gradient_z_per_km * (z[None, :] - z_range_km[0])
    )
    vs_xz = vp_xz * (vs0_km_s / vp0_km_s)

    vp = np.broadcast_to(vp_xz[:, None, :], (numx, numy, numz))
    vs = np.broadcast_to(vs_xz[:, None, :], (numx, numy, numz))

    trans_line = f"TRANSFORM  SIMPLE LatOrig {lat0:.6f}  LongOrig {lon0:.6f}  RotCW 0.000000"
    write_grid3d(
        root,
        "P",
        slow_len_from_velocity(vp, spacing_km),
        origin_km=(x_range_km[0], y_range_km[0], z_range_km[0]),
        spacing_km=spacing_km,
        trans_line=trans_line,
    )
    write_grid3d(
        root,
        "S",
        slow_len_from_velocity(vs, spacing_km),
        origin_km=(x_range_km[0], y_range_km[0], z_range_km[0]),
        spacing_km=spacing_km,
        trans_line=trans_line,
    )
    return numx, numy, numz
