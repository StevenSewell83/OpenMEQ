"""Unit tests for M6.3 / OpenMEQ-xdf.3: ``openmeq.locate.ttime_lookup``.

Builds a small, hand-crafted constant-velocity travel-time grid directly
(the NLLoc ``.hdr``/``.buf`` format ``openmeq.locate.ttime_lookup``'s own
module docstring documents, confirmed against the installed NonLinLoc
source -- see that docstring for citations), rather than running the real
``Grid2Time`` binary: this module's own contract is entirely about reading
that file format correctly, so a hand-built grid with a known-analytic
travel-time field is a more direct test of exactly that.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from openmeq.locate.ttime_lookup import predict_time
from openmeq.nlloc_setup import NllocSetupError

#: Arbitrary TRANS SIMPLE origin -- matches the kind of value
#: cfg.nlloc.grid.origin holds (config/example.yaml).
_LAT_ORIG = 37.80
_LON_ORIG = -122.20


def _write_time_grid(
    root: Path,
    phase: str,
    station: str,
    *,
    velocity_km_s: float,
    numy: int,
    numz: int,
    origy: float = 0.0,
    origz: float = 0.0,
    spacing_km: float = 0.5,
    src_x_km: float = 0.0,
    src_y_km: float = 0.0,
    numx: int = 1,
    lat_orig: float = _LAT_ORIG,
    lon_orig: float = _LON_ORIG,
) -> None:
    """Hand-write ``<root>.<phase>.<station>.time.{hdr,buf}``: a radially
    symmetric (``numx=1``) constant-velocity travel-time grid, straight-ray
    travel time ``sqrt(distance_km**2 + depth_km**2) / velocity_km_s`` at
    every (distance, depth) node -- exactly the analytic field this test
    checks :func:`predict_time` against.

    Format verified against the installed NonLinLoc source
    (``GridLib.c:WriteGrid3dHdr``, see ``openmeq.locate.ttime_lookup``'s own
    module docstring for the full citation trail):

    - line 1: ``numx numy numz  origx origy origz  dx dy dz gridType floatType``
    - line 2 (travel-time grids only): ``label x y z`` (the source/station's
      own map-frame km position)
    - line 3: ``TRANSFORM ...``
    - ``.buf``: ``numx*numy*numz`` little-endian float32, row-major,
      z fastest-varying then y then x.
    """
    hdr_path = Path(f"{root}.{phase}.{station}.time.hdr")
    buf_path = Path(f"{root}.{phase}.{station}.time.buf")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    with open(hdr_path, "w") as f:
        f.write(
            f"{numx} {numy} {numz}  0.0 {origy} {origz}  "
            f"{spacing_km} {spacing_km} {spacing_km} TIME2D FLOAT\n"
        )
        f.write(f"{station} {src_x_km} {src_y_km} 0.0\n")
        f.write(
            f"TRANSFORM  SIMPLE LatOrig {lat_orig:.6f}  LongOrig {lon_orig:.6f}  RotCW 0.000000\n"
        )

    values = np.empty((numx, numy, numz), dtype="<f4")
    for iy in range(numy):
        distance_km = origy + iy * spacing_km
        for iz in range(numz):
            depth_km = origz + iz * spacing_km
            values[:, iy, iz] = math.hypot(distance_km, depth_km) / velocity_km_s
    values.tofile(buf_path)


def _geo_to_map_km_reference(lat: float, lon: float, lat_orig: float, lon_orig: float):
    """Independent (test-side) re-implementation of the forward TRANS SIMPLE
    transform, for building test inputs -- deliberately a fresh derivation
    from the same cited formula, not an import of the production helper."""
    dlon = lon - lon_orig
    x = dlon * 111.111 * math.cos(math.radians(lat))
    y = (lat - lat_orig) * 111.111
    return x, y


def test_predict_time_matches_analytic_travel_time_within_grid_tolerance(tmp_path):
    """A query point that does *not* land exactly on a grid node still
    matches the analytic straight-ray travel time to within the grid's own
    discretization tolerance (bounded by ``spacing_km``)."""
    velocity = 4.0
    spacing = 0.5
    root = tmp_path / "grid"
    # Station exactly at the TRANS SIMPLE origin (x=y=0), so a query at the
    # same longitude as the origin has an exactly-known map-frame position
    # (x=0 exactly, independent of any cos() rounding) -- see module docstring.
    _write_time_grid(
        root, "S", "STA1", velocity_km_s=velocity, numy=41, numz=21, spacing_km=spacing
    )

    query_lat = _LAT_ORIG + 8.2 / 111.111
    query_lon = _LON_ORIG
    query_depth_km = 4.1

    predicted = predict_time(root, "S", query_lat, query_lon, query_depth_km, "STA1")

    analytic = math.hypot(8.2, query_depth_km) / velocity
    # Worst-case error bound: half a grid cell on each axis -> a half-cell
    # diagonal in space, converted to time at this constant velocity (see
    # module docstring's error-bound derivation).
    tolerance_s = (spacing / math.sqrt(2)) / velocity
    assert abs(predicted - analytic) < tolerance_s
    assert predicted != pytest.approx(analytic, abs=1e-6)  # exercises real snapping, not a fluke


def test_predict_time_exact_on_node(tmp_path):
    """A query point that lands exactly on a grid node matches the analytic
    value almost exactly (float32 storage precision only)."""
    velocity = 3.0
    spacing = 0.25
    root = tmp_path / "grid"
    _write_time_grid(
        root, "P", "STA2", velocity_km_s=velocity, numy=61, numz=41, spacing_km=spacing
    )

    # 6.0 km north, 0 km east of the origin/station -> lands exactly on a node.
    query_lat = _LAT_ORIG + 6.0 / 111.111
    query_lon = _LON_ORIG
    query_depth_km = 2.0

    predicted = predict_time(root, "P", query_lat, query_lon, query_depth_km, "STA2")
    analytic = math.hypot(6.0, query_depth_km) / velocity
    assert predicted == pytest.approx(analytic, abs=1e-4)


def test_predict_time_distance_from_offset_station(tmp_path):
    """The station's own source position (not the grid's (0,0)) is what
    distance is measured from."""
    velocity = 5.0
    spacing = 0.5
    root = tmp_path / "grid"
    src_x, src_y = 3.0, -2.0
    _write_time_grid(
        root,
        "P",
        "STA3",
        velocity_km_s=velocity,
        numy=41,
        numz=21,
        spacing_km=spacing,
        src_x_km=src_x,
        src_y_km=src_y,
    )

    # Pick a query point 5 km (map-frame) east and 0 km north of the origin,
    # at a lat equal to lat_orig so y_query=0 exactly; independently compute
    # its true map-frame x via the reference transform to get the expected
    # station-relative distance without relying on ttime_lookup's own code.
    query_lat = _LAT_ORIG
    query_lon = _LON_ORIG + 5.0 / (111.111 * math.cos(math.radians(query_lat)))
    query_depth_km = 1.0

    query_x, query_y = _geo_to_map_km_reference(query_lat, query_lon, _LAT_ORIG, _LON_ORIG)
    expected_distance_km = math.hypot(query_x - src_x, query_y - src_y)

    predicted = predict_time(root, "P", query_lat, query_lon, query_depth_km, "STA3")
    analytic = math.hypot(expected_distance_km, query_depth_km) / velocity
    tolerance_s = (spacing / math.sqrt(2)) / velocity + 1e-6
    assert abs(predicted - analytic) < tolerance_s


def test_predict_time_clamps_out_of_range_query(tmp_path):
    """A query far outside the grid's own extent is clamped to the nearest
    valid node rather than raising."""
    velocity = 4.0
    spacing = 1.0
    root = tmp_path / "grid"
    _write_time_grid(
        root, "S", "STA4", velocity_km_s=velocity, numy=11, numz=11, spacing_km=spacing
    )

    # 500 km away and 1000 km deep -- absurdly outside the small test grid.
    query_lat = _LAT_ORIG + 500.0 / 111.111
    predicted = predict_time(root, "S", query_lat, _LON_ORIG, 1000.0, "STA4")

    # Clamped to the farthest/deepest node: distance=10, depth=10.
    expected_at_corner = math.hypot(10.0, 10.0) / velocity
    assert predicted == pytest.approx(expected_at_corner, abs=1e-4)


def _write_time_grid_3d(
    root: Path,
    phase: str,
    station: str,
    *,
    velocity_km_s: float,
    num: tuple[int, int, int],
    orig: tuple[float, float, float],
    spacing_km: float,
    src_km: tuple[float, float, float],
) -> None:
    """Hand-write a genuine 3D (``GRID3D``, ``numx > 1``) constant-velocity
    travel-time grid: x/y/z are the map frame, the station sits at
    ``src_km``, straight-ray time at every node."""
    hdr_path = Path(f"{root}.{phase}.{station}.time.hdr")
    buf_path = Path(f"{root}.{phase}.{station}.time.buf")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    numx, numy, numz = num
    with open(hdr_path, "w") as f:
        f.write(
            f"{numx} {numy} {numz}  {orig[0]} {orig[1]} {orig[2]}  "
            f"{spacing_km} {spacing_km} {spacing_km} TIME FLOAT\n"
        )
        f.write(f"{station} {src_km[0]} {src_km[1]} {src_km[2]}\n")
        f.write(
            f"TRANSFORM  SIMPLE LatOrig {_LAT_ORIG:.6f}  LongOrig {_LON_ORIG:.6f}  RotCW 0.000000\n"
        )
    x = orig[0] + spacing_km * np.arange(numx)
    y = orig[1] + spacing_km * np.arange(numy)
    z = orig[2] + spacing_km * np.arange(numz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist = np.sqrt((xx - src_km[0]) ** 2 + (yy - src_km[1]) ** 2 + (zz - src_km[2]) ** 2)
    (dist / velocity_km_s).astype("<f4").tofile(buf_path)


def test_predict_time_3d_grid_on_node(tmp_path):
    """A genuine 3D (``numx > 1``) time grid is indexed by the query's own
    map-frame x/y/depth; a query on a node returns that node's value exactly,
    and an asymmetric station position proves x and y aren't swapped."""
    root = tmp_path / "grid"
    velocity = 3.0
    spacing = 0.5
    src = (2.0, -1.5, -0.5)
    _write_time_grid_3d(
        root, "S", "STA5", velocity_km_s=velocity, num=(21, 17, 13),
        orig=(-5.0, -4.0, -1.0), spacing_km=spacing, src_km=src,
    )

    target_x, target_y, depth = -3.0, 2.5, 4.0
    lat = _LAT_ORIG + target_y / 111.111
    lon = _LON_ORIG + target_x / (111.111 * math.cos(math.radians(lat)))
    predicted = predict_time(root, "S", lat, lon, depth, "STA5")

    expected = math.dist((target_x, target_y, depth), src) / velocity
    assert predicted == pytest.approx(expected, abs=1e-5)


def test_predict_time_missing_grid_raises_nlloc_setup_error(tmp_path):
    root = tmp_path / "grid"
    with pytest.raises(NllocSetupError):
        predict_time(root, "P", _LAT_ORIG, _LON_ORIG, 1.0, "NOSTA")


def test_predict_time_rejects_non_simple_transform(tmp_path):
    """A ``TRANSFORM NONE``/other grid (never produced by this project's own
    setup path, ARCHITECTURE.md §6d) is rejected with a clear error rather
    than silently mis-projecting the query point."""
    root = tmp_path / "grid"
    hdr_path = Path(f"{root}.P.STA6.time.hdr")
    buf_path = Path(f"{root}.P.STA6.time.buf")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hdr_path, "w") as f:
        f.write("1 5 5  0.0 0.0 0.0  0.5 0.5 0.5 TIME2D FLOAT\n")
        f.write("STA6 0.0 0.0 0.0\n")
        f.write("TRANSFORM  NONE\n")
    np.zeros((1, 5, 5), dtype="<f4").tofile(buf_path)

    with pytest.raises(NllocSetupError):
        predict_time(root, "P", _LAT_ORIG, _LON_ORIG, 1.0, "STA6")
