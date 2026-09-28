"""Predict a phase's arrival time at a station from a hypocenter, by reading
NLLoc's own per-station travel-time grid directly (M6.3 / OpenMEQ-xdf.3).

This resolves the one genuine gap PLAN.md §6 / ARCHITECTURE.md leave open:
neither document specifies *how* pass B is supposed to predict an S arrival
time from the pass-A origin in order to screen S picks
(``cfg.nlloc.filter.p_and_s.s_predicted_window_s``). Two mechanisms were
considered:

1. **Shell out to NLLoc/Time2EQ again per station.** Correct (it is exactly
   what NLLoc's own forward problem does), but far slower: pass B would pay
   a subprocess launch per (event, station, iteration) just to reject a
   handful of S picks *before* the real relocation -- and PLAN.md §4 makes
   speed an explicit design goal.
2. **Read the already-built travel-time grid's ``.buf`` array directly, at
   the nearest node to the pass-A hypocenter, in-process.** No subprocess,
   no extra I/O beyond a small ``.hdr`` text read and a 4-byte seek+read out
   of the ``.buf`` file. This is what this module does.

**Grid format and geometry, confirmed against the installed NonLinLoc
v7.1.05 source** (``external/NonLinLoc/src/GridLib.c``/``Grid2Time1.c``, the
same build docs/nlloc_interface.md's spike verifies -- not re-derived from
a doc page, there isn't one that states this as a single worked example):

- ``openmeq.nlloc_setup.build_1d_grid`` (the only velocity-model path this
  project currently builds/tests) always produces a **``GTMODE GRID2D``**
  travel-time grid per (station, phase): "2D, i.e. depth + radial-distance,
  radially symmetric per source" (docs/nlloc_interface.md's own
  ``Grid2Time`` statement notes). Concretely (``Grid2Time1.c`` around its
  "need only ix=0 sheet for 2D grids" comment, which sets ``numx`` to 1
  before writing): the saved ``<root>.<phase>.<STA>.time.{hdr,buf}`` has
  ``numx == 1``, and the two remaining axes are **not** x/y in the project's
  map frame -- axis "y" is *epicentral distance from the station* (km,
  ``origy`` is always 0.0) and axis "z" is depth (km, positive down, same
  frame as everywhere else in this project).
- NLLoc's own runtime lookup (``GridLib.c:GetEpiDist`` + ``ReadAbsInterpGrid2d``,
  called from ``NLLocLib.c`` for exactly this per-arrival predicted-travel-time
  purpose) computes the query point's distance from the *station*, not from
  the grid's ``(0,0)`` origin: ``distance = hypot(x_query - x_station,
  y_query - y_station)`` in the project's shared ``TRANS SIMPLE`` map frame
  (km, centered on ``cfg.nlloc.grid.origin`` -- ARCHITECTURE.md §6d). The
  station's own map-frame position is not re-derived from the station table
  here: NLLoc already writes it into the time grid's own ``.hdr`` file, as
  the second line (``label x y z``, only present for a travel-time/angle
  grid -- ``GridLib.c:WriteGrid3dHdr``), so it is read directly from there,
  keeping this function self-contained given only the grid files.
- The forward half of ``TRANS SIMPLE`` (geographic -> map-frame km,
  ``GridLib.c`` around its ``MAP_TRANS_SIMPLE`` branch of the
  latlon-to-rect conversion) is: ``x = dlon * 111.111 * cos(radians(lat))``,
  ``y = (lat - lat_orig) * 111.111`` (``dlon`` wrapped to [-180, 180], the
  project never rotates its grids -- ``RotCW`` is always ``0`` per
  ``write_control_common`` -- so the rotation terms drop out). This is
  re-implemented here (:func:`_geo_to_map_km`) rather than importing
  ``openmeq.nlloc_setup``'s private ``_parse_transform_simple`` /
  ``write_control_common`` internals, since this module only needs the
  forward direction and only needs it for a single point at a time; the
  ``TRANS SIMPLE`` origin itself is read from the *grid's own* ``TRANSFORM``
  line (via the already-public :func:`openmeq.nlloc_setup.read_grid3d_hdr`),
  not re-derived from ``cfg``, so this function only ever needs the grid
  files, ``station``, and the query point -- no ``cfg``/``paths`` beyond the
  file-root prefix.

**Nearest-neighbour, not NLLoc's own bilinear interpolation.** NLLoc itself
(``ReadAbsInterpGrid2d``) bilinearly interpolates the 4 surrounding nodes;
this function instead rounds to the single nearest node on each axis. This
is a deliberate simplification for a **pre-screen**, not the final location:
pass B's real accept/reject decision is always the residual computed by an
actual NLLoc relocation (:func:`openmeq.locate.nlloc_runner.locate_event`),
so a slightly coarser predicted time here only affects which S picks are
*offered* to that relocation, not the final answer. The resulting error is
bounded by the grid's own node spacing (``cfg.nlloc.grid.spacing_km``, the
same spacing ``openmeq.nlloc_setup.build_1d_grid`` uses for the travel-time
grid): at worst, the query point sits exactly halfway between two nodes on
both the distance and depth axes, giving a worst-case position error of
``spacing_km / sqrt(2)`` (Pythagoras on a half-cell diagonal) and thus a
worst-case *time* error of roughly ``spacing_km / (sqrt(2) * v)`` for a
local velocity ``v`` -- for this project's default 0.05 km spacing and a
typical S velocity of ~2-3 km/s, that is a few milliseconds, comfortably
below any sane ``s_predicted_window_s`` (default 0.3 s). A reviewer wanting
a tighter bound can compare ``spacing_km`` against ``s_predicted_window_s``
directly; if a future config ever used a much coarser grid, this
approximation would need revisiting.

**Nodes outside the grid's own extent are clamped**, not extrapolated: a
query index below 0 or above ``numy - 1`` / ``numz - 1`` is pinned to the
nearest valid node rather than raising, since a pass-A hypocenter is by
construction inside (or very near) the location search grid
(ARCHITECTURE.md §6d), which is itself always covered by the travel-time
grid (``openmeq.locate.nlloc_runner``'s own ``LOCGRID`` coverage check) --
clamping here just extends that same "nearest valid answer" behaviour to
the rare edge case of a query landing exactly on, or a hair outside, the
travel-time grid's own boundary.

**Only a radially-symmetric (``numx == 1``) time grid is supported.** This
is what this project's only currently-built velocity-model path (1D layered,
``openmeq.nlloc_setup.build_1d_grid``) always produces. A genuine 3D
velocity model (``cfg.nlloc.velocity_model.type == "3d"``) would make
``Grid2Time`` write a full ``GRID3D`` time grid (``numx`` unchanged, a true
3-D field, not radially symmetric), which this simple nearest-node lookup
does not attempt to index correctly -- :func:`predict_time` raises
``NotImplementedError`` rather than silently returning a wrong answer for
that case (flagged as a follow-up if pass B ever needs to run against a 3D
model).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from openmeq.nlloc_setup import NllocSetupError, read_grid3d_hdr

#: NLLoc's own km-per-degree-of-latitude constant for ``TRANS SIMPLE``
#: (``GridLib.c``'s ``c111``) -- a fixed value, not a real ellipsoid
#: (docs/nlloc_interface.md §4's ``TRANS SIMPLE`` note); reproduced here
#: exactly so this module's map-frame distance matches the one baked into
#: the travel-time grid it reads.
_C111_KM_PER_DEG = 111.111


def _geo_to_map_km(lat_deg: float, lon_deg: float, lat_orig_deg: float, lon_orig_deg: float):
    """Forward ``TRANS SIMPLE`` (geographic -> map-frame km), ``RotCW=0``.

    See the module docstring for the exact formula and its source citation.
    """
    dlon = lon_deg - lon_orig_deg
    if dlon > 180.0:
        dlon -= 360.0
    elif dlon < -180.0:
        dlon += 360.0
    x = dlon * _C111_KM_PER_DEG * math.cos(math.radians(lat_deg))
    y = (lat_deg - lat_orig_deg) * _C111_KM_PER_DEG
    return x, y


def _parse_simple_transform_origin(transform_line: str) -> tuple[float, float]:
    """``(lat_orig, long_orig)`` from a grid's own
    ``TRANSFORM  SIMPLE LatOrig <lat> LongOrig <lon> RotCW <rot>`` line.

    Raises ``NllocSetupError`` if the line isn't a ``SIMPLE`` transform --
    this project's grids always are (``write_control_common``,
    ARCHITECTURE.md §6d), so anything else means the grid wasn't built by
    this project's own setup path.
    """
    tokens = transform_line.split()
    if len(tokens) < 4 or tokens[0] != "TRANSFORM" or tokens[1] != "SIMPLE":
        raise NllocSetupError(f"expected a 'TRANSFORM SIMPLE ...' line, got {transform_line!r}")
    try:
        lat_orig = float(tokens[tokens.index("LatOrig") + 1])
        long_orig = float(tokens[tokens.index("LongOrig") + 1])
    except (ValueError, IndexError) as exc:
        raise NllocSetupError(
            f"could not parse LatOrig/LongOrig out of {transform_line!r}"
        ) from exc
    return lat_orig, long_orig


def _read_source_position_km(hdr_path: Path) -> tuple[float, float]:
    """``(x, y)`` map-frame km position of a travel-time grid's own source
    (the station, for a per-station time grid), from the ``.hdr``'s second
    line (``label x y z`` -- see module docstring, ``GridLib.c:WriteGrid3dHdr``).
    """
    try:
        lines = hdr_path.read_text().splitlines()
    except OSError as exc:
        raise NllocSetupError(f"cannot read travel-time grid header {hdr_path}: {exc}") from exc
    if len(lines) < 2:
        raise NllocSetupError(
            f"travel-time grid header {hdr_path} has no source-position line "
            "(expected 'label x y z' on line 2 -- is this a velocity/slowness "
            "model grid rather than a travel-time grid?)"
        )
    fields = lines[1].split()
    if len(fields) < 4:
        raise NllocSetupError(
            f"travel-time grid header {hdr_path} line 2 has {len(fields)} field(s), "
            f"expected 4 (label x y z): {lines[1]!r}"
        )
    try:
        src_x, src_y = float(fields[1]), float(fields[2])
    except ValueError as exc:
        raise NllocSetupError(f"{hdr_path} line 2 is malformed: {lines[1]!r}") from exc
    return src_x, src_y


def predict_time(
    grid_root: Path | str,
    phase: str,
    lat: float,
    lon: float,
    depth_km: float,
    station: str,
) -> float:
    """Predicted travel time (seconds) for ``phase`` from ``(lat, lon,
    depth_km)`` to ``station``, read from that station's own NLLoc
    travel-time grid at the nearest grid node (no interpolation -- see the
    module docstring for why that is adequate here, and its bounded error).

    Parameters
    ----------
    grid_root : pathlib.Path or str
        The shared ``GTFILES``/``VGOUT`` file-root prefix (e.g.
        ``paths.nlloc_grid_root``). The grid read is
        ``<grid_root>.<phase>.<station>.time.{hdr,buf}``.
    phase : str
        ``"P"`` or ``"S"``.
    lat, lon : float
        Query hypocenter position, decimal degrees.
    depth_km : float
        Query hypocenter depth, km, positive down (this project's Origin
        objects store depth in metres -- QuakeML's standard unit -- so a
        caller passing ``origin.depth`` must divide by 1000 first).
    station : str
        Bare station code (``NET.STA``'s ``STA`` part), matching the
        ``<STA>`` in the grid filename.

    Returns
    -------
    float
        Predicted travel time in seconds. Add this to the hypocenter's
        origin time to get the predicted arrival time.

    Raises
    ------
    openmeq.nlloc_setup.NllocSetupError
        If the grid files don't exist or don't parse as expected.
    NotImplementedError
        If the grid is a genuine 3D (``numx != 1``) time grid -- see the
        module docstring's "Only a radially-symmetric grid is supported" note.
    """
    grid_root = Path(grid_root)
    hdr_path = grid_root.parent / f"{grid_root.name}.{phase}.{station}.time.hdr"
    buf_path = grid_root.parent / f"{grid_root.name}.{phase}.{station}.time.buf"

    header = read_grid3d_hdr(hdr_path)
    if header.numx != 1:
        raise NotImplementedError(
            f"{hdr_path}: numx={header.numx} -- this is a genuine 3D (GRID3D) travel-time "
            "grid, not the radially-symmetric (GRID2D, numx=1) grid this nearest-node "
            "pre-screen lookup supports (see openmeq.locate.ttime_lookup's module "
            "docstring). Pass B's S-prediction screening does not yet support 3D models."
        )

    src_x, src_y = _read_source_position_km(hdr_path)
    lat_orig, long_orig = _parse_simple_transform_origin(header.transform_line)
    query_x, query_y = _geo_to_map_km(lat, lon, lat_orig, long_orig)
    distance_km = math.hypot(query_x - src_x, query_y - src_y)

    iy = round((distance_km - header.origy) / header.dy)
    iz = round((depth_km - header.origz) / header.dz)
    iy = min(max(iy, 0), header.numy - 1)
    iz = min(max(iz, 0), header.numz - 1)

    # Row-major, z fastest-varying, then y, then x (openmeq.nlloc_setup /
    # tests/helpers_nlloc3d.py's confirmed .buf layout); ix is always 0 here.
    offset = 4 * (iy * header.numz + iz)
    try:
        with open(buf_path, "rb") as f:
            f.seek(offset)
            raw = f.read(4)
    except OSError as exc:
        raise NllocSetupError(f"cannot read travel-time grid buffer {buf_path}: {exc}") from exc
    if len(raw) != 4:
        raise NllocSetupError(
            f"travel-time grid buffer {buf_path} is too short to read node (iy={iy}, iz={iz}) "
            f"at byte offset {offset}"
        )
    (value,) = struct.unpack("<f", raw)
    return float(value)
