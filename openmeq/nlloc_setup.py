"""NLLoc setup: control-file boilerplate, Vel2Grid (1D) / user-supplied grid
linking (3D), per-station Grid2Time, and the ObsPyck model link
(M5.2 / OpenMEQ-mev.2, extended by M5.3 / OpenMEQ-mev.3 for 3D model grids).

Implements PLAN.md §7's ``scripts/setup_nlloc.py`` for both
``nlloc.velocity_model.type`` cases: ``"1d"`` (:func:`build_1d_grid`, runs
``Vel2Grid`` against the project's layer file, M5.2) and ``"3d"``
(:func:`link_3d_grid`, copies/symlinks a user-supplied NLLoc grid in after
:func:`check_3d_grid_coverage` validates it, M5.3 -- PLAN.md §1: 3D grids are
always user-supplied, never built by this project). :func:`setup_nlloc`
dispatches between them; everything downstream (:func:`run_grid2time`,
:func:`link_obspyck_model`) is shared. This module is the reusable,
independently-testable piece; ``scripts/setup_nlloc.py`` is a thin ``click``
CLI wrapper around :func:`setup_nlloc`, and M6.1 (the per-event NLLoc runner,
OpenMEQ-xdf.1) imports :func:`write_control_common` directly rather than
re-deriving the shared ``CONTROL``/``TRANS`` boilerplate.

Every control-statement syntax claim here cites docs/nlloc_interface.md (the
M5.1 spike's verified reference); the ObsPyck filesystem layout cites
docs/obspyck_notes.md (the M7.1 spike). The 3D grid ``.hdr``/``.buf`` on-disk
format (no single doc page states it) is cited directly from the installed
``external/NonLinLoc/src/GridLib.c`` source (the same v7.1.05 build the rest
of this module already relies on) -- see :func:`read_grid3d_hdr`.

Directory layout produced (ARCHITECTURE.md §6d)
------------------------------------------------
::

    <output_dir>/nlloc/VELOCITY_MODELS/layer_time/layer.P.mod.{hdr,buf}
    <output_dir>/nlloc/VELOCITY_MODELS/layer_time/layer.S.mod.{hdr,buf}
    <output_dir>/nlloc/VELOCITY_MODELS/layer_time/layer.{P,S}.<STA>.time.{hdr,buf}
    <output_dir>/nlloc/VELOCITY_MODELS/layer_time/layer.{P,S}.<STA>.angle.{hdr,buf}
    <output_dir>/nlloc/manifest.json

(``openmeq.paths.RunPaths.nlloc_velocity_models_dir``/``.nlloc_grid_root``.)
This mirrors, one-to-one, the ``pluginpath/VELOCITY_MODELS/layer_time/``
structure ObsPyck's own generated NLLoc control file expects
(docs/obspyck_notes.md §5) -- ``link_obspyck_model`` symlinks
``nlloc_velocity_models_dir`` in as ``pluginpath/VELOCITY_MODELS`` directly,
so no separate copy/rename step is needed.

Grid coverage (the silent "0 events located" failure mode)
------------------------------------------------------------
docs/nlloc_interface.md's ``LOCGRID`` note: NLLoc rejects any observation
whose station's travel-time grid doesn't vertically span the location
grid's full z-range, and does this *silently* (exit code 0, "N events read,
0 events located"). Since M6.1's own future ``LOCGRID`` statement is built
from the same ``cfg.nlloc.grid`` this module already has, :func:`location_grid_bounds`
computes that eventual location grid's bounds once, here, as the single
place both this module (to size Vel2Grid/Grid2Time's shared grid generously
around it) and M6.1 (to write a *matching* ``LOCGRID`` statement) should use
-- see ARCHITECTURE.md §6d.
"""

from __future__ import annotations

import math
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
from obspy.geodetics import gps2dist_azimuth

from openmeq.config import ConfigError
from openmeq.external import bin_path, run
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap
from openmeq.velocity import read_layers

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

#: Manifest step name (``paths.manifest_json("nlloc")``, ``StepManifest(paths, "nlloc")``).
STEP_NAME = "nlloc"

#: Above-sea-level margin (km) applied to the location grid's own z origin
#: (ARCHITECTURE.md §6d), so every station's negative effective elevation
#: (ARCHITECTURE.md §6: ``elevation_m - depth_m``, boreholes are negative)
#: sits comfortably inside both the location grid and the travel-time grid
#: built here. Verified safe (docs/nlloc_interface.md §4/§5): Vel2Grid's
#: shallowest LAYER extends upward with no error, and GTSRCE's ``elev`` is
#: never range-checked by NLLoc, so there is no lower bound to respect
#: other than "wide enough for every configured station".
Z_MARGIN_ABOVE_SEA_LEVEL_KM = 2.0

#: Extra margin (km) added on top of the geometric coverage requirement
#: below, so a station/grid combination that is only marginally covered
#: doesn't trip NLLoc's exact-boundary rejection (docs/nlloc_interface.md
#: ``LOCGRID`` note). Applies only to the shared travel-time grid's own
#: radial reach (:func:`_required_radial_extent_km`) -- *not* to the
#: location grid's own bounds (see :data:`STATION_CLEARANCE_MARGIN_KM`):
#: applying a 5 km margin there would balloon the *search* volume itself
#: (observed, empirically, to let NLLoc's octree search wander into a
#: spurious, statistically-plausible-but-wrong shallow/negative-depth
#: solution on this project's own synthetic data -- keep this margin out
#: of ``location_grid_bounds``).
GRID_COVERAGE_MARGIN_KM = 5.0

#: Small margin (km) used only to make sure a station whose effective
#: elevation would otherwise sit exactly on the location grid's default
#: z-boundary is comfortably inside it -- deliberately much smaller than
#: :data:`GRID_COVERAGE_MARGIN_KM`, since this widens the actual *search*
#: grid (``LOCGRID``), not just the travel-time grid built around it.
STATION_CLEARANCE_MARGIN_KM = 0.5


class NllocSetupError(RuntimeError):
    """Raised when Vel2Grid/Grid2Time did not produce an expected output file."""


@dataclass(frozen=True)
class LocationGridBounds:
    """The eventual NLLoc ``LOCGRID`` statement's bounds, in km relative to
    ``cfg.nlloc.grid.origin`` (the ``TRANS SIMPLE`` reference point every
    control file this module writes shares, per :func:`write_control_common`).

    See :func:`location_grid_bounds` and ARCHITECTURE.md §6d. M6.1's own
    ``LOCGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType saveFlag``
    statement should be built from exactly these bounds (``xOrig=x_min_km``,
    ``yOrig=y_min_km``, ``zOrig=z_min_km``, ``xNum=ceil((x_max-x_min)/dx)+1``, ...)
    so the location grid this module's travel-time grids were sized for is
    the same one M6.1 actually asks NLLoc to search.
    """

    x_min_km: float
    x_max_km: float
    y_min_km: float
    y_max_km: float
    z_min_km: float
    z_max_km: float


def location_grid_bounds(cfg: Config, station_table: pd.DataFrame) -> LocationGridBounds:
    """The location-search grid's bounds (km, relative to ``cfg.nlloc.grid.origin``).

    Horizontal (x, y): centered on the origin, spanning ``+-size_km[0]/2``
    and ``+-size_km[1]/2`` (``cfg.nlloc.grid.size_km``).

    Vertical (z, NLLoc's positive-down convention): from
    ``-Z_MARGIN_ABOVE_SEA_LEVEL_KM`` down to ``size_km[2]`` by default,
    widened if needed so every station's own effective elevation
    (converted to NLLoc's ``z = -(elevation_m - depth_m)/1000`` convention,
    docs/nlloc_interface.md §5) sits inside ``[z_min_km, z_max_km]`` with
    only :data:`STATION_CLEARANCE_MARGIN_KM` to spare -- a deep borehole
    would otherwise sit outside the default margin. Deliberately a small
    margin: this is the actual search grid, and over-widening it (e.g. with
    :data:`GRID_COVERAGE_MARGIN_KM`, meant for the travel-time grid's own
    reach instead) invites NLLoc's octree search to wander into a spurious
    distant solution instead of the true, comparatively shallow one.

    Parameters
    ----------
    cfg : openmeq.config.Config
    station_table : pandas.DataFrame
        ``openmeq.stations.station_table``-shaped (``elevation_m``, ``depth_m``
        columns are used).

    Returns
    -------
    LocationGridBounds
    """
    size_x, size_y, size_z = cfg.nlloc.grid.size_km
    z_min = -Z_MARGIN_ABOVE_SEA_LEVEL_KM
    z_max = float(size_z)
    if len(station_table):
        eff_elev_km = (station_table["elevation_m"] - station_table["depth_m"]) / 1000.0
        station_z_km = -eff_elev_km  # NLLoc source z = zSrce - elev, zSrce=0 (§5)
        z_min = min(z_min, float(station_z_km.min()) - STATION_CLEARANCE_MARGIN_KM)
        z_max = max(z_max, float(station_z_km.max()) + STATION_CLEARANCE_MARGIN_KM)
    return LocationGridBounds(
        x_min_km=-size_x / 2,
        x_max_km=size_x / 2,
        y_min_km=-size_y / 2,
        y_max_km=size_y / 2,
        z_min_km=z_min,
        z_max_km=z_max,
    )


def _required_radial_extent_km(
    cfg: Config, station_table: pd.DataFrame, bounds: LocationGridBounds
) -> float:
    """Radial distance (km) the shared 2D velocity/travel-time grid must reach.

    Conservative bound: for any station and any point inside the location
    grid's horizontal box, the distance between them is at most that
    station's own distance from ``cfg.nlloc.grid.origin`` plus the box's
    half-diagonal -- so covering that sum for the *farthest* station (from
    the origin) covers every (station, location-grid-point) pair, without
    needing each station's individual distance to each of the box's four
    corners.
    """
    half_diag = 0.5 * math.hypot(
        bounds.x_max_km - bounds.x_min_km, bounds.y_max_km - bounds.y_min_km
    )
    lat0, lon0 = cfg.nlloc.grid.origin
    max_station_km = 0.0
    for row in station_table.itertuples(index=False):
        dist_m, _, _ = gps2dist_azimuth(row.latitude, row.longitude, lat0, lon0)
        max_station_km = max(max_station_km, dist_m / 1000.0)
    return max_station_km + half_diag + GRID_COVERAGE_MARGIN_KM


# ---------------------------------------------------------------------------
# write_control_common
# ---------------------------------------------------------------------------


def write_control_common(cfg: Config, paths: RunPaths) -> str:
    """The ``CONTROL``/``TRANS`` boilerplate shared by every Vel2Grid/Grid2Time/
    NLLoc control file this module (and M6.1) writes.

    Uses ``TRANS SIMPLE latOrig lonOrig 0.0`` (docs/nlloc_interface.md §4) --
    the geographic variant the M5.1 spike ran end to end
    (tests/test_nlloc_spike.py) and that ObsPyck's own generated control
    file also uses (docs/obspyck_notes.md §5) -- centered on
    ``cfg.nlloc.grid.origin`` with no rotation. ``TRANS`` (and any
    ``MAPLINE``) must be byte-identical across every control file used for
    one project (docs/nlloc_interface.md §4): every writer in this module
    goes through this one function so that always holds.

    ``paths`` is accepted (not currently used) to match the shape every
    other writer in this module takes, and in case a future TRANS variant
    needs a run-specific value.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths

    Returns
    -------
    str
    """
    lat0, lon0 = cfg.nlloc.grid.origin
    return f"CONTROL 1 54321\nTRANS SIMPLE {lat0} {lon0} 0.0\n"


# ---------------------------------------------------------------------------
# write_station_list
# ---------------------------------------------------------------------------


def write_station_list(station_table: pd.DataFrame) -> list[str]:
    """One ``GTSRCE`` line per unique ``NET.STA`` (ARCHITECTURE.md §6: location
    codes are ignored/collapsed when two rows share a site).

    ``label`` (the ``GTSRCE`` line's first field) is the bare station code,
    not ``NET.STA``: docs/obspyck_notes.md §5 confirms NLLoc/ObsPyck match
    observations to travel-time grids on the bare code alone, so this must
    be exactly what :func:`run_grid2time` names its output grids with.

    ``elev`` is the ARCHITECTURE.md §6 effective elevation
    (``elevation_m - depth_m``), converted metres -> km. docs/nlloc_interface.md
    §5 confirmed ``GTSRCE``'s ``elev`` field accepts negative values with no
    range check -- a borehole sensor's ``elev`` here is simply negative.

    Parameters
    ----------
    station_table : pandas.DataFrame
        ``openmeq.stations.station_table``-shaped (``network``, ``station``,
        ``latitude``, ``longitude``, ``elevation_m``, ``depth_m``).

    Returns
    -------
    list[str]
        One ``GTSRCE`` line per unique ``NET.STA``, in the table's row order.

    Raises
    ------
    openmeq.config.ConfigError
        If two different ``NET.STA`` rows share the same bare station code
        (across networks) -- named per the clashing pairs (M5.1's
        uniqueness finding: NLLoc/ObsPyck match on the bare code alone, so
        such a clash is not resolvable downstream).
    """
    grouped = station_table.drop_duplicates(subset=["network", "station"])

    by_code: dict[str, list[str]] = {}
    for row in grouped.itertuples(index=False):
        by_code.setdefault(row.station, []).append(f"{row.network}.{row.station}")
    clashes = {code: nets for code, nets in by_code.items() if len(nets) > 1}
    if clashes:
        detail = "; ".join(f"{code!r} used by {nets}" for code, nets in sorted(clashes.items()))
        raise ConfigError(
            "station code(s) shared across networks -- NLLoc/ObsPyck match travel-time "
            f"grids on the bare station code alone (docs/nlloc_interface.md §7): {detail}"
        )

    lines = []
    for row in grouped.itertuples(index=False):
        elev_km = (row.elevation_m - row.depth_m) / 1000.0
        lines.append(
            f"GTSRCE  {row.station}  LATLON  {row.latitude}  {row.longitude}  0.0  {elev_km}"
        )
    return lines


# ---------------------------------------------------------------------------
# build_1d_grid
# ---------------------------------------------------------------------------


def build_1d_grid(cfg: Config, paths: RunPaths, station_table: pd.DataFrame) -> None:
    """Parse the 1D layer file and run ``Vel2Grid`` to build the shared
    velocity grid every station's ``Grid2Time`` run reads.

    ``station_table`` is not in the original OpenMEQ-mev.2 spec's literal
    signature for this function, but is required to size the grid safely
    (see the "grid coverage" note below) -- this deviation is called out in
    the M5.2 report.

    File format (ARCHITECTURE.md §6a, ``openmeq.velocity.read_layers``):
    whitespace-separated ``depth_top_km vp_km_s vs_km_s [rho_g_cc]``, ``#``
    comments, strictly increasing depth. ``vs`` is a required column in
    this project's file format (unlike the bd issue's hedge, "S, if Vs
    given"), so an S grid is always built.

    Grid shape (docs/nlloc_interface.md §4): ``VGGRID xNum yNum zNum ...``
    with ``xNum=2``, ``xOrig=yOrig=0.0`` -- "for a 1D model use a 2D grid".
    ``yNum``/``zNum`` (the radial-distance and depth dimensions) are sized
    from :func:`location_grid_bounds`/:func:`_required_radial_extent_km` so
    every station's own (radially-symmetric) ``Grid2Time`` run can reach
    every point inside the eventual ``LOCGRID`` -- avoiding NLLoc's silent
    "0 events located" failure when a location grid outgrows its
    travel-time grid (docs/nlloc_interface.md, ``LOCGRID`` note).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    station_table : pandas.DataFrame

    Raises
    ------
    NllocSetupError
        If ``Vel2Grid`` did not produce the expected ``.P.mod``/``.S.mod``
        ``.hdr``/``.buf`` files.
    """
    layers = read_layers(cfg.nlloc.velocity_model.file)

    bounds = location_grid_bounds(cfg, station_table)
    y_extent_km = _required_radial_extent_km(cfg, station_table, bounds)
    z_span_km = bounds.z_max_km - bounds.z_min_km
    spacing = cfg.nlloc.grid.spacing_km

    y_num = max(2, math.ceil(y_extent_km / spacing) + 2)
    z_num = max(2, math.ceil(z_span_km / spacing) + 2)

    grid_dir = paths.nlloc_grid_root.parent
    grid_dir.mkdir(parents=True, exist_ok=True)
    root_name = paths.nlloc_grid_root.name

    layer_lines = "\n".join(
        f"LAYER  {r.depth_km}  {r.vp}  0  {r.vs}  0  {r.rho}  0"
        for r in layers.itertuples(index=False)
    )
    control = (
        write_control_common(cfg, paths) + "\n"
        f"VGOUT {root_name}\n"
        "VGTYPE P\n"
        "VGTYPE S\n"
        f"VGGRID  2 {y_num} {z_num}  0.0 0.0 {bounds.z_min_km}  "
        f"{spacing} {spacing} {spacing}  SLOW_LEN\n"
        "\n" + layer_lines + "\n"
    )
    ctrl_path = grid_dir / "vel2grid.in"
    ctrl_path.write_text(control)

    run([str(bin_path("Vel2Grid")), ctrl_path.name], cwd=grid_dir)

    missing = [
        str(grid_dir / f"{root_name}.{wave_type}.mod.{ext}")
        for wave_type in ("P", "S")
        for ext in ("hdr", "buf")
        if not (grid_dir / f"{root_name}.{wave_type}.mod.{ext}").exists()
    ]
    if missing:
        raise NllocSetupError(f"Vel2Grid did not produce expected file(s): {missing}")


# ---------------------------------------------------------------------------
# 3D grid coverage check + link_3d_grid (M5.3 / OpenMEQ-mev.3)
# ---------------------------------------------------------------------------

#: Grid ``chr_type`` values (GridLib.c's ``convert_grid_type`` table) that are
#: travel-time/take-off-angle grids, not a velocity/slowness model grid --
#: :func:`check_3d_grid_coverage` rejects a supplied root pointing at one of
#: these with a clear error rather than a confusing coverage-shortfall
#: message (the numbers would parse fine but mean something else entirely).
_NON_MODEL_GRID_TYPES = frozenset(
    {
        "TIME",
        "TIME2D",
        "ANGLE",
        "ANGLE2D",
        "INCLINATION",
        "INCLINATION2D",
        "PROB_DENSITY",
        "MISFIT",
        "LIKELIHOOD",
        "SSST_TIMECORR",
    }
)

#: Tolerance (degrees) for comparing a supplied 3D grid's own ``TRANSFORM``
#: origin against ``cfg.nlloc.grid.origin``. Not a "coverage margin" (design
#: note: coverage itself is checked with zero tolerance) -- purely to absorb
#: text round-tripping (the grid's ``.hdr`` was very likely written with
#: ``%lf`` -- 6 decimals, GridLib.c:593-596 -- while ``cfg.nlloc.grid.origin``
#: may have fewer). 1e-4 deg is ~11 m at the equator, far smaller than any
#: sane grid spacing.
_TRANS_ORIGIN_TOLERANCE_DEG = 1e-4


class Grid3DHeader(NamedTuple):
    """Parsed contents of an NLLoc 3D grid ``.hdr`` file.

    See :func:`read_grid3d_hdr` for the on-disk format this mirrors.
    ``orig{x,y,z}``/``d{x,y,z}`` are in km, in the coordinate frame the
    grid's own ``transform_line`` defines (**not** necessarily
    ``cfg.nlloc.grid.origin``'s frame -- :func:`check_3d_grid_coverage`
    checks the two match before treating them as interchangeable).
    """

    numx: int
    numy: int
    numz: int
    origx: float
    origy: float
    origz: float
    dx: float
    dy: float
    dz: float
    grid_type: str
    float_type: str
    transform_line: str


def read_grid3d_hdr(hdr_path: Path) -> Grid3DHeader:
    """Parse an NLLoc 3D grid ``.hdr`` text file.

    **Format** (not documented as a single worked example anywhere in
    ``external/NonLinLoc/docs``; read directly from the installed source,
    the same v7.1.05 build docs/nlloc_interface.md §1 confirms, and cited by
    file+line so this can be re-verified against a different NLLoc build):

    - Line 1 (``GridLib.c:1019-1025``, ``WriteGrid3dHdr``)::

          numx numy numz  origx origy origz  dx dy dz gridType floatType

      whitespace-separated (exact spacing not significant -- confirmed by
      the two separate ``fprintf`` calls that write it, GridLib.c:1019 and
      1024/1026). ``gridType`` is one of GridLib.c's ``convert_grid_type``
      table (``VELOCITY``, ``SLOWNESS``, ``SLOW_LEN``, ...). ``floatType``
      is ``FLOAT`` or ``DOUBLE`` (``GRID_FLOAT_TYPE_DOUBLE`` compile flag,
      GridLib.c:1023-1026) -- the installed build has this **commented out**
      (``external/NonLinLoc/src/CMakeLists.txt:75``), so every grid this
      project's own binaries write, and every grid they can read
      (GridLib.c's own runtime check refuses a mismatched ``floatType``,
      confirmed by the ``external/bin/Grid2Time`` string
      ``"Global grid float type is set to FLOAT, but grid file type is not
      FLOAT"``), is ``FLOAT`` (4-byte).
    - Line 2, **only** for a travel-time/angle grid (``GridLib.c:1032-1036``,
      ``psrce->label x y z``) -- never present for a velocity/slowness model
      grid (``<root>.{P,S}.mod.hdr``), which is all this function is used
      for in this module.
    - The next line is ``TRANSFORM  <type> ...`` (``GridLib.c:1037``,
      ``fprintf(fpio, "%s\\n", MapProjStr[0])`` -- always written, even for
      ``TRANS NONE``/``GLOBAL``) -- kept verbatim in
      :attr:`Grid3DHeader.transform_line` for
      :func:`check_3d_grid_coverage` to parse.

    Parameters
    ----------
    hdr_path : pathlib.Path

    Returns
    -------
    Grid3DHeader

    Raises
    ------
    openmeq.nlloc_setup.NllocSetupError
        If ``hdr_path`` doesn't exist or its first line doesn't parse as
        the format above.
    """
    try:
        lines = hdr_path.read_text().splitlines()
    except OSError as exc:
        raise NllocSetupError(f"cannot read 3D grid header {hdr_path}: {exc}") from exc
    if not lines:
        raise NllocSetupError(f"3D grid header {hdr_path} is empty")

    fields = lines[0].split()
    if len(fields) < 11:
        raise NllocSetupError(
            f"3D grid header {hdr_path} line 1 has {len(fields)} field(s), expected 11 "
            f"(numx numy numz origx origy origz dx dy dz gridType floatType): {lines[0]!r}"
        )
    try:
        numx, numy, numz = int(fields[0]), int(fields[1]), int(fields[2])
        origx, origy, origz = float(fields[3]), float(fields[4]), float(fields[5])
        dx, dy, dz = float(fields[6]), float(fields[7]), float(fields[8])
    except ValueError as exc:
        raise NllocSetupError(
            f"3D grid header {hdr_path} line 1 is malformed: {lines[0]!r}"
        ) from exc
    grid_type, float_type = fields[9], fields[10]

    transform_line = ""
    for line in lines[1:]:
        if line.startswith("TRANSFORM"):
            transform_line = line.strip()
            break

    return Grid3DHeader(
        numx=numx,
        numy=numy,
        numz=numz,
        origx=origx,
        origy=origy,
        origz=origz,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type=grid_type,
        float_type=float_type,
        transform_line=transform_line,
    )


def _parse_transform_simple(transform_line: str) -> tuple[float, float, float] | None:
    """``(lat_orig, long_orig, rot_cw)`` from a ``TRANSFORM SIMPLE ...`` line
    (``GridLib.c:593-596``: ``"TRANSFORM  %s LatOrig %lf  LongOrig %lf  RotCW
    %lf"``), or ``None`` if ``transform_line`` isn't a ``SIMPLE`` transform
    (including "isn't a TRANSFORM line at all", ``transform_line == ""``)."""
    tokens = transform_line.split()
    if len(tokens) < 2 or tokens[0] != "TRANSFORM" or tokens[1] != "SIMPLE":
        return None
    try:
        lat = float(tokens[tokens.index("LatOrig") + 1])
        lon = float(tokens[tokens.index("LongOrig") + 1])
        rot = float(tokens[tokens.index("RotCW") + 1])
    except (ValueError, IndexError):
        return None
    return (lat, lon, rot)


def check_3d_grid_coverage(hdr_path: Path, cfg: Config, bounds: LocationGridBounds) -> None:
    """Validate a user-supplied 3D NLLoc grid covers ``bounds`` (the eventual
    ``LOCGRID`` volume, :func:`location_grid_bounds`) before it's linked in.

    Two checks, both **fail-closed with no tolerance margin** (the design
    note: NLLoc's own behaviour for an out-of-grid travel-time query should
    not be relied upon). Empirically confirmed here (not just asserted)
    against ``external/NonLinLoc/src/NLLocLib.c:1904-1911``: for a genuine
    3D (``GRID_TIME``) travel-time grid, NLLoc rejects any observation whose
    station's time grid doesn't entirely contain the initial ``LOCGRID``
    (``IsGridInside``), printing ``"WARNING: initial location search grid
    not contained inside arrival time grid, rejecting observation: <STA>
    <PHASE>"`` -- the same *silent* (exit 0, message-level warning only,
    easy to miss in a batch run) "0 events located" failure mode
    docs/nlloc_interface.md's ``LOCGRID`` note already found for the 2D
    (1D-model-derived) case.

    1. **Transform match.** This project's every control file shares one
       ``TRANS SIMPLE latOrig lonOrig 0.0`` (:func:`write_control_common`,
       docs/nlloc_interface.md §4: ``TRANS``/``MAPLINE`` must be
       byte-identical project-wide). The supplied grid's own ``.hdr`` line
       self-describes the transform it was actually built with
       (:func:`read_grid3d_hdr`'s ``transform_line``) -- resolving the
       design note's "if the .hdr format does not self-describe its
       transform" hedge: it does, so rather than asking the user for a
       separate config field this validates the grid's own ``TRANSFORM``
       against ``cfg.nlloc.grid.origin`` directly. Only ``TRANS SIMPLE`` (no
       rotation) is supported, matching :func:`write_control_common`; any
       other transform type, or a ``SIMPLE`` one centered elsewhere, is
       rejected -- there is no general point-transform code in this project
       to convert between them, and guessing wrong would silently misplace
       every travel time.
    2. **Spatial coverage.** The grid's own extent (``[origx, origx + dx *
       (numx-1)]`` etc., in the now-confirmed-shared coordinate frame) must
       *contain* ``bounds`` on every axis.

    Parameters
    ----------
    hdr_path : pathlib.Path
        The supplied grid's ``.hdr`` file (one wave type, e.g.
        ``<root>.P.mod.hdr``).
    cfg : openmeq.config.Config
    bounds : LocationGridBounds
        From :func:`location_grid_bounds` -- the volume the eventual
        ``LOCGRID`` will span.

    Raises
    ------
    openmeq.config.ConfigError
        Naming the specific shortfall or mismatch found.
    """
    header = read_grid3d_hdr(hdr_path)

    if header.float_type != "FLOAT":
        raise ConfigError(
            f"{hdr_path}: grid float type is {header.float_type!r}, but the installed NLLoc "
            "build is FLOAT-only (external/NonLinLoc/src/CMakeLists.txt leaves "
            "GRID_FLOAT_TYPE_DOUBLE unset) -- Grid2Time/NLLoc will refuse to read this grid."
        )
    if header.grid_type in _NON_MODEL_GRID_TYPES:
        raise ConfigError(
            f"{hdr_path}: gridType is {header.grid_type!r}, which is a travel-time/angle/"
            "misfit grid, not a velocity or slowness model grid -- "
            "cfg.nlloc.velocity_model.file must point at the <root>.{P,S}.mod.{hdr,buf} "
            "grid pair Vel2Grid3D (or equivalent) produced, not a Grid2Time output."
        )

    lat0, lon0 = cfg.nlloc.grid.origin
    parsed = _parse_transform_simple(header.transform_line)
    if parsed is None:
        raise ConfigError(
            f"{hdr_path}: {header.transform_line or '(no TRANSFORM line found)'!r} is not a "
            "'TRANSFORM SIMPLE ...' transform -- this project's control files all use "
            f"'TRANS SIMPLE {lat0} {lon0} 0.0' (write_control_common, docs/nlloc_interface.md "
            "§4: TRANS must be byte-identical across every control file for one project), so "
            "a 3D grid built with a different transform (or a rotated/differently-centered "
            "SIMPLE one) cannot be used as-is. Rebuild the grid with a matching TRANS line."
        )
    grid_lat, grid_lon, grid_rot = parsed
    if (
        abs(grid_lat - lat0) > _TRANS_ORIGIN_TOLERANCE_DEG
        or abs(grid_lon - lon0) > _TRANS_ORIGIN_TOLERANCE_DEG
        or grid_rot != 0.0
    ):
        raise ConfigError(
            f"{hdr_path}: grid's own TRANSFORM (SIMPLE, LatOrig {grid_lat}, LongOrig "
            f"{grid_lon}, RotCW {grid_rot}) does not match cfg.nlloc.grid.origin "
            f"[{lat0}, {lon0}] (RotCW 0.0) -- every control file for one project must share "
            "byte-identical TRANS (docs/nlloc_interface.md §4). Either set cfg.nlloc.grid.origin "
            "to this grid's own LatOrig/LongOrig, or rebuild the grid with "
            f"'TRANS SIMPLE {lat0} {lon0} 0.0'."
        )

    x_min, x_max = header.origx, header.origx + header.dx * (header.numx - 1)
    y_min, y_max = header.origy, header.origy + header.dy * (header.numy - 1)
    z_min, z_max = header.origz, header.origz + header.dz * (header.numz - 1)
    eps = 1e-9

    shortfalls = []
    if x_min > bounds.x_min_km + eps or x_max < bounds.x_max_km - eps:
        shortfalls.append(
            f"x covers [{x_min:.3f}, {x_max:.3f}] km but location grid needs "
            f"[{bounds.x_min_km:.3f}, {bounds.x_max_km:.3f}] km"
        )
    if y_min > bounds.y_min_km + eps or y_max < bounds.y_max_km - eps:
        shortfalls.append(
            f"y covers [{y_min:.3f}, {y_max:.3f}] km but location grid needs "
            f"[{bounds.y_min_km:.3f}, {bounds.y_max_km:.3f}] km"
        )
    if z_min > bounds.z_min_km + eps or z_max < bounds.z_max_km - eps:
        shortfalls.append(
            f"z covers [{z_min:.3f}, {z_max:.3f}] km but location grid needs "
            f"[{bounds.z_min_km:.3f}, {bounds.z_max_km:.3f}] km"
        )
    if shortfalls:
        raise ConfigError(
            f"3D grid {hdr_path} does not cover the location grid: " + "; ".join(shortfalls)
        )


def link_3d_grid(cfg: Config, paths: RunPaths, station_table: pd.DataFrame) -> None:
    """3D counterpart of :func:`build_1d_grid`: instead of running
    ``Vel2Grid``, validate and link in the user-supplied 3D grid
    (``cfg.nlloc.velocity_model.file``, a root name -- PLAN.md §7: "3D: uses
    the supplied NLLoc model grid (.hdr/.buf) directly, after checking it
    covers the location grid").

    Copies (falling back from a symlink only if ``os.symlink`` raises
    ``OSError``, e.g. across filesystems -- same fallback
    :func:`link_obspyck_model` already uses) ``<file>.{P,S}.mod.{hdr,buf}``
    into ``paths.nlloc_grid_root``'s directory, renamed to the same
    ``<root>.{P,S}.mod.{hdr,buf}`` convention :func:`build_1d_grid`
    produces, so :func:`run_grid2time` needs no knowledge of which path
    produced its input.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    station_table : pandas.DataFrame

    Raises
    ------
    NllocSetupError
        If the source ``<file>.{P,S}.mod.{hdr,buf}`` files don't all exist.
    openmeq.config.ConfigError
        Via :func:`check_3d_grid_coverage`, for either wave type.
    """
    src_root = Path(cfg.nlloc.velocity_model.file)

    missing = [
        str(Path(f"{src_root}.{wave_type}.mod.{ext}"))
        for wave_type in ("P", "S")
        for ext in ("hdr", "buf")
        if not Path(f"{src_root}.{wave_type}.mod.{ext}").exists()
    ]
    if missing:
        raise NllocSetupError(f"supplied 3D grid missing expected file(s): {missing}")

    bounds = location_grid_bounds(cfg, station_table)
    for wave_type in ("P", "S"):
        check_3d_grid_coverage(Path(f"{src_root}.{wave_type}.mod.hdr"), cfg, bounds)

    grid_dir = paths.nlloc_grid_root.parent
    grid_dir.mkdir(parents=True, exist_ok=True)
    root_name = paths.nlloc_grid_root.name

    for wave_type in ("P", "S"):
        for ext in ("hdr", "buf"):
            src = Path(f"{src_root}.{wave_type}.mod.{ext}").resolve()
            dst = grid_dir / f"{root_name}.{wave_type}.mod.{ext}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# run_grid2time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Grid2TimeUnit:
    """One ``pmap`` work item: a single (station, phase) ``Grid2Time`` run.

    Self-contained and picklable (``openmeq.parallel.pmap``'s requirement
    for ``n_workers > 1``): carries everything :func:`_run_one_grid2time`
    needs rather than closing over shared state.
    """

    station: str
    lat: float
    lon: float
    elev_km: float
    phase: str
    control_common: str
    grid_dir: str
    root_name: str


def _grid_mode_for_model(grid_dir: Path, root_name: str, phase: str) -> str:
    """``GTMODE``'s ``gridMode`` field for this phase's own model grid
    (``<root>.<phase>.mod.hdr``, already built/linked by :func:`build_1d_grid`
    or :func:`link_3d_grid` before this is called): ``"GRID2D"`` for a
    1D-model-derived, radially-symmetric grid (``numx == 2``, exactly
    :func:`build_1d_grid`'s ``VGGRID`` convention), ``"GRID3D"`` for a
    genuine 3D grid (``numx != 2``).

    Verified against ``external/NonLinLoc/src/Grid2Time1.c:288-296``:
    ``GTMODE GRID2D`` hard-errors (``"ERROR: grid xNum must be 2 for
    gridMode GRID2D"``) unless the model grid's own ``numx`` is exactly 2,
    while ``GTMODE GRID3D`` accepts any ``numx`` (only a warning below 3) --
    so a single hardcoded ``GTMODE`` cannot serve both
    :func:`build_1d_grid`'s always-``numx=2`` output and a user-supplied,
    genuinely 3D (``numx>2``) grid; this reads the actual grid's own header
    rather than trusting which of the two setup paths produced it (M5.3 /
    OpenMEQ-mev.3's ``run_grid2time``-stays-shared requirement), so
    :func:`run_grid2time` itself needs no per-path branching.
    """
    hdr_path = grid_dir / f"{root_name}.{phase}.mod.hdr"
    return "GRID2D" if read_grid3d_hdr(hdr_path).numx == 2 else "GRID3D"


def _run_one_grid2time(unit: _Grid2TimeUnit) -> str:
    """Run ``Grid2Time`` for exactly one (station, phase); the ``pmap`` worker body.

    Writes its own uniquely-named control file (``grid2time_<STA>_<phase>.in``)
    into the shared ``grid_dir`` -- distinct filenames per unit mean workers
    never contend over the same file, even though they all read the same
    (read-only) velocity grid and write into the same output directory.
    """
    grid_dir = Path(unit.grid_dir)
    grid_mode = _grid_mode_for_model(grid_dir, unit.root_name, unit.phase)
    ctrl_text = (
        unit.control_common + "\n"
        f"GTFILES  {unit.root_name}  {unit.root_name}  {unit.phase}\n"
        f"GTMODE {grid_mode} ANGLES_NO\n"
        "\n"
        f"GTSRCE  {unit.station}  LATLON  {unit.lat}  {unit.lon}  0.0  {unit.elev_km}\n"
        "\n"
        "GT_PLFD  1.0e-3  0\n"
    )
    ctrl_path = grid_dir / f"grid2time_{unit.station}_{unit.phase}.in"
    ctrl_path.write_text(ctrl_text)
    run([str(bin_path("Grid2Time")), ctrl_path.name], cwd=grid_dir)
    return f"{unit.station}:{unit.phase}"


def run_grid2time(cfg: Config, paths: RunPaths, station_table: pd.DataFrame) -> None:
    """Run ``Grid2Time`` for every (station, phase) pair, in parallel.

    PLAN.md §7 / the design note: one (station, phase) pair is one unit of
    work (8 stations x 2 phases = 16 units on ``synthetic_project``); all
    units read the same, already-built (:func:`build_1d_grid`) or
    already-linked (:func:`link_3d_grid`, M5.3 / OpenMEQ-mev.3) velocity
    grid read-only, so they're safe to run concurrently via
    ``openmeq.parallel.pmap``. Either producer (a true, non-parallel
    dependency either way) must already have been run -- this function does
    not call :func:`build_1d_grid`/:func:`link_3d_grid` itself, and does not
    need to know which one produced its input: each unit reads its own
    ``GTMODE`` (``GRID2D``/``GRID3D``) straight from the model grid's own
    ``.hdr`` (:func:`_grid_mode_for_model`), so a 1D-derived (``numx=2``) and
    a genuinely-3D (``numx>2``) grid both just work.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    station_table : pandas.DataFrame

    Raises
    ------
    NllocSetupError
        If, after all units complete, any expected
        ``<root>.<phase>.<STA>.time.{hdr,buf}`` file is missing.
    """
    control_common = write_control_common(cfg, paths)
    grid_dir = paths.nlloc_grid_root.parent
    root_name = paths.nlloc_grid_root.name

    grouped = station_table.drop_duplicates(subset=["network", "station"])

    units = [
        _Grid2TimeUnit(
            station=row.station,
            lat=row.latitude,
            lon=row.longitude,
            elev_km=(row.elevation_m - row.depth_m) / 1000.0,
            phase=phase,
            control_common=control_common,
            grid_dir=str(grid_dir),
            root_name=root_name,
        )
        for row in grouped.itertuples(index=False)
        for phase in ("P", "S")
    ]

    pmap(_run_one_grid2time, units, cfg.parallel, desc="grid2time")

    missing = [
        str(grid_dir / f"{root_name}.{phase}.{row.station}.time.{ext}")
        for row in grouped.itertuples(index=False)
        for phase in ("P", "S")
        for ext in ("hdr", "buf")
        if not (grid_dir / f"{root_name}.{phase}.{row.station}.time.{ext}").exists()
    ]
    if missing:
        raise NllocSetupError(f"Grid2Time did not produce expected file(s): {missing}")


# ---------------------------------------------------------------------------
# link_obspyck_model
# ---------------------------------------------------------------------------


def _render_bool01(value: bool) -> str:
    """NLLoc renders a boolean control-statement field as a literal ``1``/``0``,
    not Python's ``True``/``False`` (docs/nlloc_interface.md §4, ``LOCSEARCH`` note)."""
    return "1" if value else "0"


def _obspyck_nlloc_control(cfg: Config, paths: RunPaths, station_table: pd.DataFrame) -> str:
    """The full NLLoc control file ObsPyck's own "do NLLoc" button runs.

    Mirrors the exact, empirically-verified-working control file in
    docs/obspyck_notes.md §5, parameterized from ``cfg.nlloc`` where a
    matching config field already exists (``locsearch``, ``locgau``,
    ``locgau2``, ``locqual2err_thresholds_s``); statements with no
    ``openmeq.config`` model yet (``LOCMETH``, ``LOCANGLES``, ``LOCPHSTAT``)
    use the same literal values the spike verified end-to-end
    (tests/test_nlloc_spike.py, docs/obspyck_notes.md §5) -- flagged as a
    follow-up for M6.1, which will need its own ``LOCMETH`` config section
    anyway (docs/nlloc_interface.md §8).

    ``LOCFILES``' travel-time-grid root is the relative path
    ``../VELOCITY_MODELS/layer_time/<root>``, reaching up from
    ``pluginpath/nlloc/`` (where this control file lives) to the
    symlinked ``pluginpath/VELOCITY_MODELS`` sibling
    (docs/obspyck_notes.md §5) -- *not* an absolute path, since ObsPyck
    symlinks ``VELOCITY_MODELS`` fresh into a new temp directory on every
    launch and the control file is copied alongside it each time.

    **Deviation from the docs/obspyck_notes.md §5 sample**: that sample used
    ``LOCANGLES ANGLES_YES 5``, but its travel-time grids were built with
    ``GTMODE ... ANGLES_YES`` (a separate, throwaway spike model). This
    module's own :func:`run_grid2time` always uses ``ANGLES_NO`` (whether
    ``GTMODE`` itself is ``GRID2D`` or ``GRID3D``, M5.3 / OpenMEQ-mev.3 --
    :func:`_grid_mode_for_model`), matching the M5.1 spike's own end-to-end
    example, the only take-off-angle behaviour actually verified so far --
    docs/nlloc_interface.md §4's ``LOCANGLES`` note flags per-arrival angles
    as an unverified M6.1 follow-up), so no angle grids exist alongside the
    travel-time grids this function's ``LOCFILES`` points at. ``LOCANGLES``
    here is kept consistent with that (``ANGLES_NO``) rather than requesting
    angle output the underlying grids don't carry.
    """
    bounds = location_grid_bounds(cfg, station_table)
    spacing = cfg.nlloc.grid.spacing_km
    x_num = max(2, math.ceil((bounds.x_max_km - bounds.x_min_km) / spacing) + 1)
    y_num = max(2, math.ceil((bounds.y_max_km - bounds.y_min_km) / spacing) + 1)
    z_num = max(2, math.ceil((bounds.z_max_km - bounds.z_min_km) / spacing) + 1)

    ls = cfg.nlloc.locsearch
    lg = cfg.nlloc.locgau
    lg2 = cfg.nlloc.locgau2
    locqual2err = " ".join(str(v) for v in cfg.nlloc.locqual2err_thresholds_s)
    root_name = paths.nlloc_grid_root.name

    return (
        write_control_common(cfg, paths) + "\n"
        "LOCSIG OpenMEQ (scripts/setup_nlloc.py, OpenMEQ-mev.2)\n"
        "LOCCOM ObsPyck model link -- generated, do not edit by hand\n"
        "\n"
        f"LOCFILES  nlloc.obs  NLLOC_OBS  ../VELOCITY_MODELS/layer_time/{root_name}  nlloc\n"
        "\n"
        "LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_HYPOINV_SUM\n"
        "\n"
        f"LOCSEARCH  OCT {ls.init_num_cells_x} {ls.init_num_cells_y} {ls.init_num_cells_z} "
        f"{ls.min_node_size} {ls.max_num_nodes} {ls.num_scatter} "
        f"{_render_bool01(ls.use_stations_density)} {_render_bool01(ls.stop_on_min_node_size)}\n"
        "\n"
        f"LOCGRID  {x_num} {y_num} {z_num}  "
        f"{bounds.x_min_km} {bounds.y_min_km} {bounds.z_min_km}  "
        f"{spacing} {spacing} {spacing}  PROB_DENSITY SAVE\n"
        "\n"
        "LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1.0 6 -1.0 1\n"
        "\n"
        f"LOCGAU {lg.sigma_time_s} {lg.corr_len_km}\n"
        f"LOCGAU2 {lg2.sigma_time_fraction} {lg2.sigma_time_min_s} {lg2.sigma_time_max_s}\n"
        "\n"
        "LOCPHASEID  P   P\n"
        "LOCPHASEID  S   S\n"
        "\n"
        f"LOCQUAL2ERR {locqual2err}\n"
        "\n"
        "LOCANGLES ANGLES_NO 5\n"
        "LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9\n"
    )


def link_obspyck_model(cfg: Config, paths: RunPaths, station_table: pd.DataFrame) -> None:
    """Create/refresh the filesystem link ObsPyck needs to relocate with this
    run's NLLoc model (docs/obspyck_notes.md).

    ``ObsPyck.util.setup_external_programs`` (inspected at ``obspyck/util.py``
    ~line 798) expects, inside ``config["base"]["pluginpath"]``
    (``cfg.nlloc.obspyck_pluginpath``):

    - ``<pluginpath>/nlloc/NLLoc__<platform.system()>__<platform.architecture()[0]>``
      -- the NLLoc executable, named for this machine (confirmed
      ``NLLoc__Linux__64bit`` on this project's dev machine,
      docs/obspyck_notes.md §1). Symlinked to ``external/bin/NLLoc``
      (falling back to a copy if symlinks aren't available, e.g. across
      filesystems).
    - ``<pluginpath>/nlloc/locate_BY.nlloc`` -- a control-file template
      (named ``BY`` because it's the combobox's default-selected item,
      docs/obspyck_notes.md §5's decision, needing zero UI interaction).
    - ``<pluginpath>/VELOCITY_MODELS`` -- symlinked (not copied) to this
      run's NLLoc model root (``paths.nlloc_velocity_models_dir``), which
      already contains ``layer_time/<root>.{P,S}.mod.*`` and
      ``layer_time/<root>.{P,S}.<STA>.time.*`` from
      :func:`build_1d_grid`/:func:`run_grid2time`.

    Does **not** create ``<pluginpath>/hyp_2000``/``<pluginpath>/focmec``
    (also required, empty, or ``ObsPyck.__init__`` raises
    ``FileNotFoundError`` before the GUI appears -- docs/obspyck_notes.md
    §4's pitfall) or write the ObsPyck ``.cfg`` ini itself -- both are part
    of the full generated pluginpath/config, M7.2 (OpenMEQ-c3l.2), which
    this task's issue explicitly scopes out ("this task only creates the
    filesystem link that M7.2's generated config will reference").

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    station_table : pandas.DataFrame
    """
    pluginpath = Path(cfg.nlloc.obspyck_pluginpath)
    nlloc_plugin_dir = pluginpath / "nlloc"
    nlloc_plugin_dir.mkdir(parents=True, exist_ok=True)

    exe_name = f"NLLoc__{platform.system()}__{platform.architecture()[0]}"
    exe_dst = nlloc_plugin_dir / exe_name
    exe_src = bin_path("NLLoc")
    if exe_dst.is_symlink() or exe_dst.exists():
        exe_dst.unlink()
    try:
        exe_dst.symlink_to(exe_src)
    except OSError:
        shutil.copy2(exe_src, exe_dst)
        exe_dst.chmod(exe_dst.stat().st_mode | 0o111)

    control = _obspyck_nlloc_control(cfg, paths, station_table)
    (nlloc_plugin_dir / "locate_BY.nlloc").write_text(control)

    vm_link = pluginpath / "VELOCITY_MODELS"
    if vm_link.is_symlink():
        vm_link.unlink()
    elif vm_link.is_dir():
        shutil.rmtree(vm_link)
    elif vm_link.exists():
        vm_link.unlink()
    vm_link.symlink_to(paths.nlloc_velocity_models_dir, target_is_directory=True)


# ---------------------------------------------------------------------------
# setup_nlloc -- top-level orchestration + skip-unchanged
# ---------------------------------------------------------------------------


def setup_nlloc(cfg: Config, paths: RunPaths, *, force: bool = False) -> bool:
    """Run the full NLLoc setup -- ``build_1d_grid`` (``cfg.nlloc.velocity_model.type
    == "1d"``) or ``link_3d_grid`` (``== "3d"``, M5.3 / OpenMEQ-mev.3) +
    ``run_grid2time`` + ``link_obspyck_model`` -- skipping if nothing
    relevant changed since the last run.

    Fingerprinted (``openmeq.manifest.fingerprint``) on ``cfg.nlloc``, the
    merged station table's contents, and the velocity model input's own
    size/mtime -- any change to the grid/filter config, the station list, or
    the velocity model input itself triggers a rebuild. For ``type == "1d"``
    that input is the single layer file (``cfg.nlloc.velocity_model.file``);
    for ``"3d"`` it's the four supplied ``<file>.{P,S}.mod.{hdr,buf}`` grid
    files (the ``file`` value itself is a root name, not a file that exists
    on its own, so fingerprinting it directly would never detect a changed
    grid).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Rebuild even if the manifest says nothing changed.

    Returns
    -------
    bool
        ``True`` if the setup was (re)built, ``False`` if skipped as
        unchanged.
    """
    from openmeq.stations import station_table as load_station_table

    table = load_station_table(cfg.stations)

    is_3d = cfg.nlloc.velocity_model.type == "3d"
    if is_3d:
        model_root = cfg.nlloc.velocity_model.file
        dep_paths = [
            Path(f"{model_root}.{wave_type}.mod.{ext}")
            for wave_type in ("P", "S")
            for ext in ("hdr", "buf")
        ]
    else:
        dep_paths = [Path(cfg.nlloc.velocity_model.file)]

    fp = fingerprint(cfg.nlloc, table.to_dict("records"), dep_paths)
    manifest = StepManifest(paths, STEP_NAME)
    key = "setup"
    if not force and manifest.is_done(key, fp):
        return False

    if is_3d:
        link_3d_grid(cfg, paths, table)
    else:
        build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)
    link_obspyck_model(cfg, paths, table)

    manifest.mark_done(key, fp)
    return True
