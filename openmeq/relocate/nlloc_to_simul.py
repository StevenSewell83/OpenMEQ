"""NLLoc 3D velocity grid (``.hdr``/``.buf``) -> SIMUL2000 ``vel3D.dat``
converter.

Plan key: M9.4 / OpenMEQ-6ld.4 (PLAN.md §7's ``--nlloc-to-simul`` flag).
Lets one 3D velocity model serve both NLLoc (M5.3's own ``.hdr``/``.buf``
grid) and hypoDD (M9.3's ``IMOD=9`` SIMUL2000 ``vel3D.dat``, written/read by
``openmeq.relocate.hypodd_files``).

**Ground truth, confirmed directly against source (not generic docs), per
the M9 epic's working agreement**:

- NLLoc's ``.hdr``/``.buf`` grammar: ``external/NonLinLoc/src/GridLib.c``
  (``WriteGrid3dHdr``/``ReadGrid3dHdr_grid_description`` write/read the
  header; ``AllocateGrid``'s ``ix * numy*numz + iy*numz + iz`` offset
  formula, also used directly by ``ReadGrid3dValue``'s per-value
  ``sizeof(GRID_FLOAT_TYPE) * (ix*numyz + iy*numz + iz)`` and by
  ``ReadAbsInterpGrid3d``'s node-coordinate convention, confirms x is the
  slowest-varying / z the fastest-varying array axis, C order). Header line
  grammar (``GridLib.c:2274``, mirrored exactly by :func:`read_nlloc_grid`)::

      numx numy numz  origx origy origz  dx dy dz gridType [floatType]

  followed by an optional ``TRANSFORM ...`` line (``GridLib.c:548-650``,
  read back at ``GridLib.c:2221-2226``) -- ``docs/nlloc_interface.md``'s
  ``TRANS`` statement, restated by NLLoc into its own ``TRANSFORM`` header
  line verbatim. ``GRID_FLOAT_TYPE`` is ``float`` (32-bit) unless NLLoc was
  built with ``GRID_FLOAT_TYPE_DOUBLE`` (``GridLib.c:95-98``) -- the
  bundled build (``external/bin``) is the default single-precision build,
  so :func:`read_nlloc_grid` requires ``floatType`` (when present) to say
  ``FLOAT``, not ``DOUBLE``.
- Grid-type -> velocity conversion: ``external/NonLinLoc/src/Vel2Grid3D.c``
  (``GRID_VELOCITY``/``GRID_VELOCITY_METERS``/``GRID_SLOW_LEN``/
  ``GRID_SLOW2_METERS`` cases, ``Vel2Grid3D.c:948-966``) plus
  ``GridLib.c:869-887``'s ``convert_grid_type`` char/enum table for the
  remaining ``GRID_SLOWNESS``/``GRID_VEL2``/``GRID_SLOW2`` types not
  produced by ``Vel2Grid3D`` itself but part of the same documented family
  (``docs/nlloc_interface.md``'s ``VGGRID`` ``gridType`` choices). Notably,
  ``GRID_SLOW_LEN`` (what ``docs/nlloc_interface.md`` confirms both the
  bundled sample and Podvin-Lecomte (``GT_PLFD``, what M5.3 uses) produce)
  stores ``dx / velocity`` at every node -- **using ``grid->dx``
  specifically, regardless of ``dy``/``dz``** (``Vel2Grid3D.c:959``), so
  this module's inversion does the same (and logs a warning if
  ``dx != dy`` or ``dx != dz``, since that's an unusual grid for this
  conversion to be strictly correct on).
- SIMUL2000 ``vel3D.dat``'s body grammar, node-value legality (``bld`` must
  be exactly ``1.0`` or ``0.1``, node coordinates should be exact multiples
  of ``bld``) and P/Vp-Vs-ratio block ordering/shape: ``external/HYPODD/
  src/hypoDD/get_vel3d.f`` (read line by line for this task; see
  :func:`nlloc_grid_to_vel3d`'s docstring for the exact citations) and
  ``external/HYPODD/examples/example1/vel3D.dat`` (the worked example this
  module's tests validate node values and file shape against).
- The ``ixkms``/``iykms``/``izkms`` = ``10000`` compiled map-size limit
  (``external/HYPODD/include/vel3d.inc``, used by ``get_vel3d.f``'s
  ``bldmap`` subroutine) is a *second*, independent compiled limit from
  ``maxnx``/``maxny``/``maxnz`` (the only one M9.3's
  ``openmeq.relocate.hypodd_limits`` parses/checks) -- it bounds the
  *physical span* of the node grid in units of ``bld`` km, not the node
  *count*. This module enforces it too (:func:`nlloc_grid_to_vel3d`), since
  a converted grid that silently violates it would abort hypoDD with
  ``bldmap``'s own fatal ``stop`` (``get_vel3d.f:279-285``) -- flagged as a
  follow-up gap in this module's own docstring and in the final report
  (``hypodd_limits.check_compiled_limits``/``VEL3D_INC_PARAMS`` could grow
  a fourth, map-span check against this same constant, parsed from
  ``vel3d.inc`` like the others, instead of this module's own hardcoded
  ``10000``).

**Resampling design** (PLAN.md §3's ``nlloc.grid.spacing_km`` default,
``0.05`` km, does not fit SIMUL2000's ``bld`` constraint directly --
ARCHITECTURE.md §6a/§6c and this module's own design note require a real
resampling step, not a reformat):

- :func:`nlloc_grid_to_vel3d` builds a *legal* (``bld``-aligned) SIMUL2000
  node grid at a caller-chosen ``node_spacing_km`` (default: ``bld``
  itself, the finest legal resolution), covering the NLLoc grid's full
  domain, then **linearly interpolates** the NLLoc grid onto those nodes
  (`scipy.interpolate.RegularGridInterpolator`, `method="linear"`) --
  matching what NLLoc's own `ReadAbsInterpGrid3d` (`GridLib.c:3038`) does
  when *it* reads a grid at an arbitrary point, so both programs interpret
  the same physical model consistently between two independent
  resamplings/gridding schemes.
- Two extra **"infinity" boundary nodes** are added past each axis's real
  domain (``examples/example1/vel3D.dat``'s own ``xn``/``yn``/``zn`` first
  and last entries, e.g. ``-800.0``/``800.0``, are exactly this pattern --
  a simulps/SIMUL2000 convention for "the model doesn't need to say
  anything beyond here, just don't let the ray tracer run off the edge of
  the array"). Because this module queries the interpolator with the query
  point **clamped into the NLLoc grid's own domain first** (constant/
  nearest-edge extrapolation, not linear), an infinity node's resampled
  velocity is automatically identical to its nearest real interior node's
  -- the same behaviour ``examples/example1/vel3D.dat``'s own (dataset-
  specific, hand-tuned) infinity nodes exhibit, achieved here generically
  for any NLLoc grid rather than copied from that one example.
- **Fails loudly, does not auto-decimate**, if the resampled node count
  exceeds the (caller-supplied, defaulting to the shipped
  ``vel3d.inc`` values) ``max_nx``/``max_ny``/``max_nz`` compiled limits --
  per the M9 epic design note and this task's explicit "out of scope"
  section, the fix is a coarser ``node_spacing_km``/``bld=1.0`` or a call
  to ``openmeq.relocate.hypodd_limits.recompile_hypodd``, both named in the
  raised error, never a silent truncation.
- **Coordinate frame**: SIMUL2000's ``vel3D.dat`` file itself carries no
  lat/lon/rotation -- that comes from hypoDD.inp's separate ``LAT_3D
  LON_3D ROT_3D`` line (``get_vel3d.f``'s ``lat_3d``/``lon_3d``/``rot_3d``
  arguments, already written by M9.3's `write_hypodd_inp_3d`), which must
  agree with whatever coordinate origin the NLLoc grid's own node
  coordinates are relative to (its ``TRANS`` statement, ARCHITECTURE.md
  §6d) or the two programs will silently locate events against
  inconsistently-registered models. :func:`nlloc_grid_to_vel3d` therefore
  takes `lat_3d`/`lon_3d`/`rot_3d` as explicit parameters (this task's
  originally proposed signature) and, whenever the source NLLoc grid's own
  ``TRANS`` is ``SIMPLE``/``SDC`` (the only kind ARCHITECTURE.md §6d's
  ``write_control_common`` ever emits), **validates them against it**
  rather than silently trusting the caller -- get_vel3d.f's ``rot_3d`` is
  documented "positive=anticlockwise" (``docs/ARCHITECTURE.md`` §6c) while
  NLLoc's ``TRANS``/``RotCW`` is "clockwise-from-Y-axis"
  (``docs/nlloc_interface.md`` §"Generic statements"), so the check negates
  one side rather than comparing the raw values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from openmeq.config import ConfigError as _ConfigError
from openmeq.log import get_logger
from openmeq.relocate.hypodd_files import read_vel3d_header

logger = get_logger("relocate.nlloc_to_simul")

#: ``external/HYPODD/include/vel3d.inc``'s shipped compiled node-count
#: limits (a recompile, M9.3's `openmeq.relocate.hypodd_limits`, may have
#: since changed these -- callers with a live `.inc` tree should read the
#: real values via `openmeq.relocate.hypodd_limits._parse_fortran_int_params`
#: and pass them in rather than relying on these shipped defaults).
DEFAULT_MAXNX = 25
DEFAULT_MAXNY = 25
DEFAULT_MAXNZ = 20

#: SIMUL2000's only two legal ``bld`` values (get_vel3d.f:69-74: any other
#: value is a fatal Fortran ``stop``).
BLD_LEGAL_VALUES = (1.0, 0.1)

#: ``external/HYPODD/include/vel3d.inc``'s ``ixkms``/``iykms``/``izkms``
#: compiled map-size limit, shared by all three axes: the total node-grid
#: span, in units of ``bld`` km, that ``get_vel3d.f``'s ``bldmap``
#: subroutine's index arrays can hold before a second, independent fatal
#: ``stop`` (get_vel3d.f:279-285, distinct from the ``maxnx``/``maxny``/
#: ``maxnz`` node-*count* limit `openmeq.relocate.hypodd_limits` already
#: checks). A margin is subtracted in `_check_map_span` since `bldmap`'s own
#: ``ixmax`` computation can be one or two units larger than the raw
#: ``(xn(nx)-xn(1))/bld`` span.
_IXKMS_IYKMS_IZKMS = 10000
_MAP_SPAN_MARGIN_BLD_UNITS = 50

#: Grid types NLLoc's ``VGGRID``/``Vel2Grid``/``Vel2Grid3D`` can produce
#: (``docs/nlloc_interface.md``'s ``VGGRID`` ``gridType`` choices;
#: ``external/NonLinLoc/src/GridLib.c:869-877``'s full char/enum table,
#: restricted to the velocity-model-shaped subset -- ``TIME``/``ANGLE``/
#: etc. grid types are a different NLLoc file family this module never
#: reads).
_KNOWN_GRID_TYPES = (
    "VELOCITY",
    "VELOCITY_METERS",
    "SLOWNESS",
    "SLOW_LEN",
    "VEL2",
    "SLOW2",
    "SLOW2_METERS",
)


@dataclass
class NLLocTransform:
    """An NLLoc grid ``.hdr``'s ``TRANSFORM`` line (``GridLib.c:548-650``
    writes it, mirroring ``docs/nlloc_interface.md``'s ``TRANS`` control
    statement almost verbatim).

    Attributes
    ----------
    kind : str
        ``"NONE"``, ``"GLOBAL"``, ``"SIMPLE"`` or ``"SDC"`` -- the four
        ``TRANS`` syntaxes ARCHITECTURE.md §6d's own control-file writer
        (`openmeq.nlloc_setup.write_control_common`) could ever have
        produced for an OpenMEQ-built grid (always ``SIMPLE`` in practice,
        per that function's docstring), plus ``NONE``/``GLOBAL`` for a
        hand-built or third-party grid. ``LAMBERT``/``TRANS_MERC``/
        ``AZIMUTHAL_EQUIDIST`` grids are rejected by `read_nlloc_grid` with
        a clear error rather than silently mis-parsed (out of scope: no
        OpenMEQ-written control file ever selects them, ARCHITECTURE.md
        §6d).
    lat_orig, lon_orig, rot_deg : float or None
        Only set for ``SIMPLE``/``SDC`` (``None`` for ``NONE``/``GLOBAL``,
        which carry no origin). `rot_deg` is NLLoc's own convention,
        clockwise-from-Y-axis positive (``docs/nlloc_interface.md``) --
        **not** SIMUL2000's anticlockwise-positive ``ROT_3D`` convention;
        see `nlloc_grid_to_vel3d`'s docstring for the sign flip this
        module applies when cross-checking the two.
    """

    kind: str
    lat_orig: float | None = None
    lon_orig: float | None = None
    rot_deg: float | None = None


@dataclass
class NLLocGrid:
    """A parsed NLLoc 3D velocity grid (`read_nlloc_grid`'s return value):
    the ``.hdr``'s grid geometry plus the ``.buf``'s node values, already
    converted to velocity (km/s) regardless of the grid's native
    `grid_type`.

    Attributes
    ----------
    nx, ny, nz : int
        Node counts (``.hdr``'s ``numx numy numz``).
    origin_x, origin_y, origin_z : float
        Coordinates (km, in the grid's own `transform`-relative Cartesian
        frame) of node index 0 on each axis (``.hdr``'s ``origx origy
        origz``).
    dx, dy, dz : float
        Node spacing (km) on each axis (``.hdr``'s ``dx dy dz``). Node
        index ``i`` on the x axis sits at ``origin_x + i * dx`` (and
        similarly for y/z) -- confirmed against
        `external/NonLinLoc/src/GridLib.c`'s own `ReadAbsInterpGrid3d`,
        which computes exactly this offset when interpolating a grid file
        at an arbitrary point.
    grid_type : str
        The ``.hdr``'s native grid-type token (e.g. ``"SLOW_LEN"``), kept
        for provenance/debugging -- `velocity_km_s` is already converted,
        this module never needs to re-derive it.
    transform : NLLocTransform
        The ``.hdr``'s ``TRANSFORM`` line, parsed.
    velocity_km_s : numpy.ndarray
        Shape ``(nx, ny, nz)``, already converted to velocity in km/s
        (`read_nlloc_grid`'s docstring documents each `grid_type`'s
        conversion formula).
    """

    nx: int
    ny: int
    nz: int
    origin_x: float
    origin_y: float
    origin_z: float
    dx: float
    dy: float
    dz: float
    grid_type: str
    transform: NLLocTransform
    velocity_km_s: np.ndarray

    def x_axis(self) -> np.ndarray:
        """Node x-coordinates (km), length `nx`."""
        return self.origin_x + self.dx * np.arange(self.nx)

    def y_axis(self) -> np.ndarray:
        """Node y-coordinates (km), length `ny`."""
        return self.origin_y + self.dy * np.arange(self.ny)

    def z_axis(self) -> np.ndarray:
        """Node z-coordinates (km, positive down), length `nz`."""
        return self.origin_z + self.dz * np.arange(self.nz)


# ---------------------------------------------------------------------------
# grid-type -> velocity (km/s) conversion
# ---------------------------------------------------------------------------


def _grid_values_to_velocity_km_s(
    values: np.ndarray, grid_type: str, dx: float, dy: float, dz: float, source: str
) -> np.ndarray:
    """Convert an NLLoc grid's raw node values to velocity (km/s), per
    `grid_type` (see this module's docstring for the source citations).
    """
    gt = grid_type.upper()
    values = values.astype(np.float64, copy=False)

    if gt == "VELOCITY":
        return values
    if gt == "VELOCITY_METERS":
        return values / 1000.0
    if gt == "SLOWNESS":
        return 1.0 / values
    if gt == "SLOW_LEN":
        if not (np.isclose(dx, dy) and np.isclose(dx, dz)):
            logger.warning(
                "%s: SLOW_LEN grid has non-cubic spacing (dx=%.6g, dy=%.6g, dz=%.6g); "
                "NLLoc's own Vel2Grid3D (Vel2Grid3D.c:959) always divides by dx alone when "
                "producing SLOW_LEN values, so this conversion does the same -- verify this "
                "is really the grid Vel2Grid3D wrote, not a hand-edited one.",
                source,
                dx,
                dy,
                dz,
            )
        return dx / values
    if gt == "VEL2":
        return np.sqrt(values)
    if gt == "SLOW2":
        return 1.0 / np.sqrt(values)
    if gt == "SLOW2_METERS":
        return 1.0e-3 / np.sqrt(values)

    raise _ConfigError(
        f"{source}: unrecognized/unsupported NLLoc grid type {grid_type!r}; known "
        f"velocity-model grid types: {_KNOWN_GRID_TYPES}"
    )


# ---------------------------------------------------------------------------
# .hdr TRANSFORM line parsing
# ---------------------------------------------------------------------------


def _parse_transform(line: str, source: str) -> NLLocTransform:
    tokens = line.split()
    if len(tokens) < 2 or tokens[0] != "TRANSFORM":
        raise _ConfigError(f"{source}: malformed TRANSFORM line: {line!r}")
    kind = tokens[1]

    if kind in ("NONE", "GLOBAL"):
        return NLLocTransform(kind=kind)

    if kind in ("SIMPLE", "SDC"):
        try:
            lat = float(tokens[tokens.index("LatOrig") + 1])
            lon = float(tokens[tokens.index("LongOrig") + 1])
            rot = float(tokens[tokens.index("RotCW") + 1])
        except (ValueError, IndexError) as exc:
            raise _ConfigError(
                f"{source}: malformed {kind} TRANSFORM line (expected 'TRANSFORM {kind} "
                f"LatOrig <val> LongOrig <val> RotCW <val>'): {line!r}"
            ) from exc
        return NLLocTransform(kind=kind, lat_orig=lat, lon_orig=lon, rot_deg=rot)

    raise _ConfigError(
        f"{source}: unsupported NLLoc TRANS type {kind!r} -- OpenMEQ's own control files "
        "only ever write TRANS SIMPLE (docs/ARCHITECTURE.md §6d's write_control_common), so "
        "only NONE/GLOBAL/SIMPLE/SDC are supported here"
    )


# ---------------------------------------------------------------------------
# read_nlloc_grid
# ---------------------------------------------------------------------------


def read_nlloc_grid(hdr_path: Path | str) -> NLLocGrid:
    """Parse an NLLoc 3D velocity grid's ``.hdr`` (text header) and ``.buf``
    (binary node values), converting the node values to velocity (km/s).

    The ``.buf`` file is assumed to sit alongside `hdr_path` with the same
    basename and a ``.buf`` extension (NLLoc's own
    ``WriteGrid3dBuf``/``OpenGrid3dFile`` convention,
    `external/NonLinLoc/src/GridLib.c`).

    Parameters
    ----------
    hdr_path : pathlib.Path or str
        The ``.hdr`` file, e.g. ``<root>.P.mod.hdr``.

    Returns
    -------
    NLLocGrid

    Raises
    ------
    openmeq.config.ConfigError
        If the header is malformed, the grid type is unrecognized, the
        grid was written in double precision (``GRID_FLOAT_TYPE_DOUBLE``,
        not the bundled single-precision build this module assumes), the
        ``.buf`` file is missing or the wrong size, or the ``TRANSFORM``
        line uses an unsupported ``TRANS`` type (see `NLLocTransform`).
    """
    hdr_path = Path(hdr_path)
    text = hdr_path.read_text()
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise _ConfigError(f"{hdr_path}: empty or missing NLLoc grid header")

    header_fields = lines[0].split()
    if len(header_fields) < 10:
        raise _ConfigError(
            f"{hdr_path}: header line must have at least 10 fields (numx numy numz origx "
            f"origy origz dx dy dz gridType), got {lines[0]!r}"
        )
    nx, ny, nz = int(header_fields[0]), int(header_fields[1]), int(header_fields[2])
    origin_x, origin_y, origin_z = (
        float(header_fields[3]),
        float(header_fields[4]),
        float(header_fields[5]),
    )
    dx, dy, dz = float(header_fields[6]), float(header_fields[7]), float(header_fields[8])
    grid_type = header_fields[9]

    if len(header_fields) >= 11 and header_fields[10].upper() == "DOUBLE":
        raise _ConfigError(
            f"{hdr_path}: grid was written in double precision (GRID_FLOAT_TYPE_DOUBLE); "
            "this module assumes the bundled single-precision (float32) NLLoc build "
            "(external/NonLinLoc/src/include/GridLib.h:95-98)"
        )

    transform = NLLocTransform(kind="NONE")
    for line in lines[1:]:
        if line.split()[:1] == ["TRANSFORM"]:
            transform = _parse_transform(line, str(hdr_path))
            break

    buf_path = hdr_path.with_suffix(".buf")
    if not buf_path.exists():
        raise _ConfigError(f"{hdr_path}: matching .buf file not found at {buf_path}")

    raw = np.fromfile(buf_path, dtype="<f4")
    expected = nx * ny * nz
    if raw.size != expected:
        raise _ConfigError(
            f"{buf_path}: expected {expected} float32 values ({nx}x{ny}x{nz} from the "
            f".hdr), got {raw.size}"
        )
    # x slowest-varying, z fastest-varying (C order) -- GridLib.c's own
    # `ix * numy*numz + iy*numz + iz` offset formula (see module docstring).
    raw = raw.reshape((nx, ny, nz))

    velocity = _grid_values_to_velocity_km_s(raw, grid_type, dx, dy, dz, str(hdr_path))

    return NLLocGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type=grid_type,
        transform=transform,
        velocity_km_s=velocity,
    )


# ---------------------------------------------------------------------------
# legal SIMUL2000 node-axis construction
# ---------------------------------------------------------------------------


def _bld_round(value: float, bld: float) -> float:
    return round(round(value / bld) * bld, 6)


def _build_axis_nodes(
    domain_min: float,
    domain_max: float,
    spacing_km: float,
    bld: float,
    infinity_km: float,
    axis_name: str,
) -> list[float]:
    """Build one axis's SIMUL2000-legal node coordinate list: an interior
    run of `bld`-aligned nodes at `spacing_km` spacing covering
    ``[domain_min, domain_max]``, plus one "infinity" node past each end
    (see this module's docstring).
    """
    if spacing_km <= 0:
        raise _ConfigError(f"{axis_name}: node_spacing_km must be positive, got {spacing_km!r}")

    ratio = spacing_km / bld
    if abs(ratio - round(ratio)) > 1e-6:
        raise _ConfigError(
            f"{axis_name}: node_spacing_km={spacing_km!r} is not a multiple of bld={bld!r} -- "
            "SIMUL2000 node coordinates must be exact multiples of bld (get_vel3d.f:64)"
        )

    x0 = _bld_round(domain_min, bld)
    if x0 > domain_min + 1e-9:
        x0 = _bld_round(x0 - bld, bld)

    coords = [x0]
    while coords[-1] < domain_max - 1e-9:
        coords.append(_bld_round(coords[-1] + spacing_km, bld))
    if len(coords) < 2:
        coords.append(_bld_round(coords[-1] + spacing_km, bld))

    infinity_lo = _bld_round(coords[0] - infinity_km, bld)
    infinity_hi = _bld_round(coords[-1] + infinity_km, bld)
    return [infinity_lo, *coords, infinity_hi]


def _check_axis_limit(n: int, max_n: int, axis_name: str) -> None:
    if n > max_n:
        raise _ConfigError(
            f"NLLoc-to-SIMUL2000 conversion: the resampled {axis_name} node count ({n}, "
            "including the 2 infinity boundary nodes) exceeds the compiled vel3d.inc limit "
            f"({max_n}). This is deliberately NOT auto-decimated -- a silent truncation would "
            "produce a wrong model. Either pass a coarser node_spacing_km (or bld=1.0) to "
            "reduce the node count, or call openmeq.relocate.hypodd_limits.recompile_hypodd "
            f"to raise max_{axis_name} in vel3d.inc and rebuild hypoDD/ph2dt."
        )


def _check_map_span(nodes: list[float], bld: float, axis_name: str) -> None:
    span_bld_units = (nodes[-1] - nodes[0]) / bld
    limit = _IXKMS_IYKMS_IZKMS - _MAP_SPAN_MARGIN_BLD_UNITS
    if span_bld_units > limit:
        raise _ConfigError(
            f"NLLoc-to-SIMUL2000 conversion: the resampled {axis_name} node span "
            f"({nodes[-1] - nodes[0]:.1f} km = {span_bld_units:.0f} bld units) exceeds "
            f"vel3d.inc's compiled ixkms/iykms/izkms map-size limit ({_IXKMS_IYKMS_IZKMS} bld "
            "units, get_vel3d.f's bldmap subroutine) -- pass a smaller infinity_km, or a "
            "larger bld, to reduce the map's physical span."
        )


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def _resample_velocity(
    grid: NLLocGrid, xs: list[float], ys: list[float], zs: list[float]
) -> np.ndarray:
    """Trilinearly interpolate `grid` onto node coordinates `xs`/`ys`/`zs`,
    clamping query points into the grid's own domain first (constant
    extrapolation past the domain edge -- see this module's docstring for
    why this also makes the "infinity" boundary nodes work for free).
    """
    x_axis, y_axis, z_axis = grid.x_axis(), grid.y_axis(), grid.z_axis()
    interp = RegularGridInterpolator(
        (x_axis, y_axis, z_axis),
        grid.velocity_km_s,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    xs_c = np.clip(np.asarray(xs, dtype=np.float64), x_axis[0], x_axis[-1])
    ys_c = np.clip(np.asarray(ys, dtype=np.float64), y_axis[0], y_axis[-1])
    zs_c = np.clip(np.asarray(zs, dtype=np.float64), z_axis[0], z_axis[-1])

    xx, yy, zz = np.meshgrid(xs_c, ys_c, zs_c, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    values = interp(points)
    return values.reshape(len(xs), len(ys), len(zs))


# ---------------------------------------------------------------------------
# coordinate-frame consistency check
# ---------------------------------------------------------------------------


def _check_coordinate_frame(
    transform: NLLocTransform, lat_3d: float, lon_3d: float, rot_3d: float, *, tol: float = 1e-6
) -> None:
    if transform.kind not in ("SIMPLE", "SDC"):
        return  # nothing to cross-check (NONE/GLOBAL carry no origin)

    expected_rot_3d = -transform.rot_deg  # NLLoc RotCW (CW+) vs SIMUL2000 ROT_3D (CCW+)
    mismatches = []
    if abs(transform.lat_orig - lat_3d) > tol:
        mismatches.append(f"lat_3d={lat_3d!r} vs NLLoc TRANS LatOrig={transform.lat_orig!r}")
    if abs(transform.lon_orig - lon_3d) > tol:
        mismatches.append(f"lon_3d={lon_3d!r} vs NLLoc TRANS LongOrig={transform.lon_orig!r}")
    if abs(expected_rot_3d - rot_3d) > tol:
        mismatches.append(
            f"rot_3d={rot_3d!r} vs NLLoc TRANS RotCW={transform.rot_deg!r} "
            f"(expected rot_3d={expected_rot_3d!r}, note the CW/anticlockwise sign flip)"
        )
    if mismatches:
        raise _ConfigError(
            "NLLoc-to-SIMUL2000 conversion: the supplied lat_3d/lon_3d/rot_3d do not match "
            "the source NLLoc grid's own TRANS origin -- the converted vel3D.dat's node "
            "coordinates are relative to the NLLoc grid's TRANS frame, so a mismatched "
            "LAT_3D/LON_3D/ROT_3D in hypoDD.inp would silently mis-register the whole model "
            "in hypoDD's own coordinate frame: " + "; ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# vel3D.dat writer
# ---------------------------------------------------------------------------


def _format_row(values: np.ndarray) -> str:
    return " ".join(f"{v:8.3f}" for v in values)


def _write_vel3d_dat(
    output_path: Path,
    bld: float,
    xn: list[float],
    yn: list[float],
    zn: list[float],
    vp: np.ndarray,
    ratio: np.ndarray | None,
) -> None:
    nx, ny, nz = len(xn), len(yn), len(zn)
    with open(output_path, "w") as fh:
        fh.write(f"{bld:g} {nx:d} {ny:d} {nz:d}  openmeq-generated (M9.4)\n")
        fh.write(" ".join(f"{v:.2f}" for v in xn) + "\n")
        fh.write(" ".join(f"{v:.2f}" for v in yn) + "\n")
        fh.write(" ".join(f"{v:.2f}" for v in zn) + "\n")
        # fixed-node list terminator, then linked-node list terminator
        # (get_vel3d.f:100-137, format '(3i3)') -- neither is supported by
        # this converter (out of scope), so both lists are empty.
        fh.write(f"{0:3d}{0:3d}{0:3d}\n")
        fh.write(f"{0:3d}{0:3d}{0:3d}\n")
        # P velocity block: nz layers, each ny lines of nx values
        # (get_vel3d.f:142-160's `do 37 k=1,nz` / `do 36 j=1,ny` loop).
        for k in range(nz):
            for j in range(ny):
                fh.write(_format_row(vp[:, j, k]) + "\n")
        if ratio is not None:
            # Vp/Vs ratio block, same (k, j) loop shape
            # (get_vel3d.f:166-174) -- hypoDD derives S velocity from this
            # ratio itself (get_vel3d.f:182), we never write S velocities
            # directly.
            for k in range(nz):
                for j in range(ny):
                    fh.write(_format_row(ratio[:, j, k]) + "\n")


def nlloc_grid_to_vel3d(
    nlloc_grid: NLLocGrid,
    output_path: Path | str,
    lat_3d: float,
    lon_3d: float,
    rot_3d: float,
    vp_vs_ratio: float | None = None,
    s_grid: NLLocGrid | None = None,
    *,
    bld: float = 0.1,
    node_spacing_km: float | None = None,
    infinity_km: float | None = None,
    max_nx: int = DEFAULT_MAXNX,
    max_ny: int = DEFAULT_MAXNY,
    max_nz: int = DEFAULT_MAXNZ,
) -> None:
    """Resample an NLLoc 3D velocity grid onto a SIMUL2000-legal node set
    and write it as a hypoDD ``vel3D.dat`` (``IMOD=9``).

    Every field/shape/ordering below is confirmed directly against
    ``external/HYPODD/src/hypoDD/get_vel3d.f`` (read line by line for this
    task) and ``external/HYPODD/examples/example1/vel3D.dat`` (the worked
    example this module's tests validate against) -- not generic SIMUL2000
    documentation:

    - Header line: ``bld nx ny nz`` (get_vel3d.f:68), where ``bld`` must be
      exactly ``1.0`` or ``0.1`` (get_vel3d.f:69-74's fatal ``stop``
      otherwise) and ``nx``/``ny``/``nz`` must not exceed the compiled
      ``vel3d.inc`` ``maxnx``/``maxny``/``maxnz`` (get_vel3d.f:76-77's
      fatal ``stop`` otherwise -- checked *before* writing, per this
      function's own `max_nx`/`max_ny`/`max_nz` parameters, defaulting to
      the shipped values; pass the real compiled ones from
      `openmeq.relocate.hypodd_limits` if they may have changed).
    - Node coordinate lines: ``xn(1..nx)``, ``yn(1..ny)``, ``zn(1..nz)``
      (get_vel3d.f:80-82), in the model's own Cartesian frame where
      ``(x,y,z)=(0,0,0)`` need not coincide with any node (get_vel3d.f's
      own comment, lines 60-64) -- this converter reuses the source NLLoc
      grid's own frame directly (its node coordinates, in km relative to
      its own ``TRANS`` origin), rather than re-deriving a new one, so
      `lat_3d`/`lon_3d`/`rot_3d` (hypoDD.inp's separate ``LAT_3D LON_3D
      ROT_3D`` line, `openmeq.relocate.hypodd_files.write_hypodd_inp_3d`)
      must describe that same frame -- validated against the NLLoc grid's
      own ``TRANS`` (when it is ``SIMPLE``/``SDC``) rather than blindly
      trusted, since a mismatch would silently mis-register the model.
    - Fixed-node and linked-node lists (get_vel3d.f:100-137, format
      ``(3i3)``): both left empty (a single ``  0  0  0`` terminator line
      each) -- this converter does not support fixing or linking specific
      nodes' velocities (out of scope).
    - P-velocity block: ``nz`` layers (outer loop), each ``ny`` lines
      (inner loop) of ``nx`` whitespace-separated values (get_vel3d.f:
      142-160's ``do 37 k=1,nz`` / ``do 36 j=1,ny`` / ``read (vel(i,j,k2),
      i=1,nx)``).
    - Vp/Vs-ratio block (**not** S velocities -- get_vel3d.f:166-174 reads
      a *ratio* and itself derives S velocity as ``vel(i,j,ks) =
      vel(i,j,k)/vpvs(i,j,k)`` at line 182): written only when
      `vp_vs_ratio` or `s_grid` is given, same ``(k, j)`` loop shape as the
      P block. Its *absence* is only legal if the ``hypoDD.inp`` ``IMOD=9``
      block's ``IPHA`` is ``1`` (P-only, ``iuses=1``,
      `openmeq.config.Simul2000RaytraceConfig.iphase`) -- with the config
      schema's own default (`iphase=2`), omitting this block would hit
      get_vel3d.f:171's read past EOF and its own fatal ``stop
      '>>>No S-velocities provided in 3D model file. Set IPHA=1.'``
      (get_vel3d.f:210). This function does not itself know the caller's
      chosen ``IPHA`` (that lives in `HypoddVelocityModelConfig`, a
      separate config), so it is the caller's responsibility to pass
      `vp_vs_ratio` or `s_grid` whenever ``IPHA != 1`` is configured.

    Parameters
    ----------
    nlloc_grid : NLLocGrid
        The source P-velocity grid, from `read_nlloc_grid`.
    output_path : pathlib.Path or str
        Where to write the ``vel3D.dat``-format file.
    lat_3d, lon_3d, rot_3d : float
        hypoDD.inp's ``LAT_3D LON_3D ROT_3D`` values this conversion is
        intended to be paired with (not written into `output_path` itself
        -- that's `write_hypodd_inp_3d`'s job -- but validated against
        `nlloc_grid`'s own ``TRANS`` origin, see above).
    vp_vs_ratio : float or None, optional
        A single constant Vp/Vs ratio applied at every node (the primary,
        always-supported path per this task's design note). Mutually
        exclusive with `s_grid`.
    s_grid : NLLocGrid or None, optional
        A co-located NLLoc S-velocity grid (same `read_nlloc_grid` shape);
        if given, a *per-node* Vp/Vs ratio is computed from `nlloc_grid`
        and `s_grid`, each independently resampled onto the same output
        node set. Mutually exclusive with `vp_vs_ratio`.
    bld : float, optional
        SIMUL2000 grid-unit factor, `1.0` or `0.1` km (default `0.1`, the
        finer of the two legal values). Every output node coordinate is
        rounded to the nearest multiple of `bld`.
    node_spacing_km : float or None, optional
        Interior node spacing (km), which must be a positive multiple of
        `bld`. Defaults to `bld` itself (the finest legal resolution) --
        for many real networks this will exceed `max_nx`/`max_ny`/
        `max_nz` and raise (per this function's "fail loudly, don't
        auto-decimate" design; see `_check_axis_limit`), in which case
        pass a coarser value here.
    infinity_km : float or None, optional
        Distance (km) each axis's two "infinity" boundary nodes sit past
        the real interior domain (see this module's docstring). Defaults
        to ``800.0 * bld`` -- for `bld=1.0` this matches
        ``examples/example1/vel3D.dat``'s own ``-800.0``/``800.0``
        convention exactly; scaling by `bld` keeps the resulting map span
        safely within `_IXKMS_IYKMS_IZKMS` regardless of which `bld` is
        chosen (see `_check_map_span`).
    max_nx, max_ny, max_nz : int, optional
        Compiled ``vel3d.inc`` node-count limits to check the resampled
        grid against before writing (defaults: the shipped values,
        `DEFAULT_MAXNX`/`DEFAULT_MAXNY`/`DEFAULT_MAXNZ`; pass the real
        currently-compiled values, e.g. via
        `openmeq.relocate.hypodd_limits`'s ``.inc`` parsing, if a previous
        recompile may have changed them).

    Raises
    ------
    openmeq.config.ConfigError
        If `bld` isn't legal, `vp_vs_ratio` and `s_grid` are both given,
        `node_spacing_km` isn't a positive multiple of `bld`, the
        resampled node count exceeds `max_nx`/`max_ny`/`max_nz`, the
        resampled map span exceeds the compiled ``ixkms``/``iykms``/
        ``izkms`` limit, `s_grid` produces a non-positive S velocity at
        some node, or the supplied `lat_3d`/`lon_3d`/`rot_3d` don't match
        `nlloc_grid`'s own ``TRANS`` origin (see above).
    """
    if bld not in BLD_LEGAL_VALUES:
        raise _ConfigError(
            f"bld must be one of {BLD_LEGAL_VALUES} (get_vel3d.f:69-74's fatal stop "
            f"otherwise), got {bld!r}"
        )
    if vp_vs_ratio is not None and s_grid is not None:
        raise _ConfigError("pass either vp_vs_ratio or s_grid, not both")

    _check_coordinate_frame(nlloc_grid.transform, lat_3d, lon_3d, rot_3d)

    spacing = node_spacing_km if node_spacing_km is not None else bld
    inf_km = infinity_km if infinity_km is not None else 800.0 * bld

    x_max = nlloc_grid.origin_x + (nlloc_grid.nx - 1) * nlloc_grid.dx
    y_max = nlloc_grid.origin_y + (nlloc_grid.ny - 1) * nlloc_grid.dy
    z_max = nlloc_grid.origin_z + (nlloc_grid.nz - 1) * nlloc_grid.dz

    xn = _build_axis_nodes(nlloc_grid.origin_x, x_max, spacing, bld, inf_km, "x")
    yn = _build_axis_nodes(nlloc_grid.origin_y, y_max, spacing, bld, inf_km, "y")
    zn = _build_axis_nodes(nlloc_grid.origin_z, z_max, spacing, bld, inf_km, "z")

    _check_axis_limit(len(xn), max_nx, "nx")
    _check_axis_limit(len(yn), max_ny, "ny")
    _check_axis_limit(len(zn), max_nz, "nz")
    _check_map_span(xn, bld, "x")
    _check_map_span(yn, bld, "y")
    _check_map_span(zn, bld, "z")

    vp = _resample_velocity(nlloc_grid, xn, yn, zn)

    ratio: np.ndarray | None = None
    if s_grid is not None:
        vs = _resample_velocity(s_grid, xn, yn, zn)
        if np.any(vs <= 0):
            raise _ConfigError(
                "s_grid produced a non-positive S velocity at some resampled node -- cannot "
                "compute a Vp/Vs ratio"
            )
        ratio = vp / vs
    elif vp_vs_ratio is not None:
        ratio = np.full_like(vp, float(vp_vs_ratio))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_vel3d_dat(output_path, bld, xn, yn, zn, vp, ratio)


def validate_vel3d_dat(output_path: Path | str, max_nx: int, max_ny: int, max_nz: int):
    """Convenience wrapper: `read_vel3d_header` the just-written file and
    re-check its node counts against `max_nx`/`max_ny`/`max_nz` -- a cheap,
    hand-written-parser-independent sanity check
    (`openmeq.relocate.hypodd_files.read_vel3d_header`/`Vel3DGrid`, per
    M9.3's note that M9.4 should validate its output this way), *before*
    the much stronger real-``get_vel3d.f`` acceptance test
    (``requires_hypodd``) this module's own test suite also runs.

    Returns
    -------
    openmeq.relocate.hypodd_files.Vel3DGrid

    Raises
    ------
    openmeq.config.ConfigError
        Propagated from `read_vel3d_header`, or raised here if the parsed
        node counts exceed `max_nx`/`max_ny`/`max_nz`.
    """
    grid = read_vel3d_header(output_path)
    _check_axis_limit(grid.nx, max_nx, "nx")
    _check_axis_limit(grid.ny, max_ny, "ny")
    _check_axis_limit(grid.nz, max_nz, "nz")
    return grid
