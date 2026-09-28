"""Tests for openmeq.relocate.nlloc_to_simul (M9.4 / OpenMEQ-6ld.4).

``_write_nlloc_grid`` below is this task's own minimal NLLoc ``.hdr``/
``.buf`` synthetic-grid builder (OpenMEQ-mev.3's ``tests/helpers_nlloc3d.py``
was not available yet when this was written -- see this task's report). It
mirrors ``external/NonLinLoc/src/GridLib.c``'s ``WriteGrid3dHdr``/
``WriteGrid3dBuf`` exactly: header line ``numx numy numz  origx origy origz
dx dy dz gridType``, an optional ``TRANSFORM ...`` line, and a flat float32
``.buf`` in ``ix*numy*numz + iy*numz + iz`` (x slowest, z fastest) order.

Unit tests (`test_read_nlloc_grid_*`, `test_bld_resampling_*`,
`test_exceeds_compiled_limits_*`) build tiny synthetic grids and never touch
any external tool. The single `requires_hypodd`/`slow` test
(`test_output_accepted_by_real_hypodd`) proves the converted ``vel3D.dat``
is accepted by the real ``get_vel3d.f`` parsing rules end-to-end: it runs
the real ``ph2dt``/``hypoDD`` binaries against a copy of
``external/HYPODD/examples/example1``'s own station/phase inputs, with
``hypoDD.inp`` targeting ``IMOD=9`` and our converted ``vel3D.dat``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Arrival,
    Magnitude,
    Origin,
    OriginQuality,
    OriginUncertainty,
    Pick,
    QuantityError,
    WaveformStreamID,
)

from openmeq.catalog import event_id, event_id_from_time, make_event
from openmeq.config import ConfigError, load_config
from openmeq.external import run
from openmeq.relocate.hypodd_files import (
    build_station_label_map,
    read_vel3d_header,
    write_event_dat,
    write_hypodd_inp_3d,
    write_ph2dt_inp,
    write_phase_dat,
    write_station_dat,
)
from openmeq.relocate.nlloc_to_simul import (
    _build_axis_nodes,
    nlloc_grid_to_vel3d,
    read_nlloc_grid,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"


# ---------------------------------------------------------------------------
# minimal synthetic NLLoc grid builder (this task's own, per the worker
# instructions -- OpenMEQ-mev.3's shared helper was not available yet)
# ---------------------------------------------------------------------------


def _write_nlloc_grid(
    root: Path,
    *,
    file_type: str = "P",
    numx: int,
    numy: int,
    numz: int,
    origx: float,
    origy: float,
    origz: float,
    dx: float,
    dy: float,
    dz: float,
    grid_type: str,
    values: np.ndarray,
    trans: str | None = "TRANSFORM  SIMPLE LatOrig 37.878078  LongOrig -122.243950  RotCW 0.000000",
) -> Path:
    """Write a synthetic ``<root>.<file_type>.mod.{hdr,buf}`` pair, mirroring
    ``external/NonLinLoc/src/GridLib.c``'s own ``WriteGrid3dHdr``/
    ``WriteGrid3dBuf`` exactly (see this module's docstring).

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(numx, numy, numz)``, the grid's *native* (not necessarily
        velocity) node values, already in `grid_type`'s own units.

    Returns
    -------
    pathlib.Path
        The written ``.hdr`` file's path.
    """
    hdr_path = Path(f"{root}.{file_type}.mod.hdr")
    buf_path = Path(f"{root}.{file_type}.mod.buf")
    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    with open(hdr_path, "w") as fh:
        fh.write(
            f"{numx} {numy} {numz}  {origx:f} {origy:f} {origz:f}  "
            f"{dx:f} {dy:f} {dz:f} {grid_type} FLOAT\n"
        )
        if trans is not None:
            fh.write(trans + "\n")

    assert values.shape == (numx, numy, numz)
    # x slowest-varying, z fastest-varying (C order), float32 native
    # byte order -- GridLib.c's own offset formula (module docstring).
    flat = values.astype("<f4").tobytes(order="C")
    assert len(flat) == numx * numy * numz * struct.calcsize("f")
    with open(buf_path, "wb") as fh:
        fh.write(flat)

    return hdr_path


# ---------------------------------------------------------------------------
# read_nlloc_grid
# ---------------------------------------------------------------------------


def test_read_nlloc_grid_matches_known_values(tmp_path):
    """A small, hand-authored NLLoc VELOCITY grid round-trips through
    `read_nlloc_grid` with correct node values at a few spot-checked
    coordinates.
    """
    numx, numy, numz = 4, 3, 5
    values = np.zeros((numx, numy, numz), dtype=np.float64)
    # A simple, hand-verifiable pattern: velocity increases 0.1 km/s per
    # x-index and 1.0 km/s per z-index, constant in y.
    for i in range(numx):
        for j in range(numy):
            for k in range(numz):
                values[i, j, k] = 4.0 + 0.1 * i + 1.0 * k

    hdr = _write_nlloc_grid(
        tmp_path / "vel1d",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=-1.0,
        origy=-1.0,
        origz=-2.0,
        dx=0.5,
        dy=0.5,
        dz=1.0,
        grid_type="VELOCITY",
        values=values,
    )

    grid = read_nlloc_grid(hdr)
    assert (grid.nx, grid.ny, grid.nz) == (numx, numy, numz)
    assert grid.grid_type == "VELOCITY"
    assert grid.transform.kind == "SIMPLE"
    assert grid.transform.lat_orig == pytest.approx(37.878078)
    assert grid.transform.lon_orig == pytest.approx(-122.243950)
    assert grid.transform.rot_deg == pytest.approx(0.0)

    # spot-check a few nodes directly against the hand-authored pattern
    assert grid.velocity_km_s[0, 0, 0] == pytest.approx(4.0)
    assert grid.velocity_km_s[3, 1, 0] == pytest.approx(4.3)
    assert grid.velocity_km_s[0, 2, 4] == pytest.approx(8.0)
    assert grid.velocity_km_s[3, 0, 4] == pytest.approx(8.3)

    # node-coordinate helpers
    assert grid.x_axis()[0] == pytest.approx(-1.0)
    assert grid.x_axis()[-1] == pytest.approx(-1.0 + 3 * 0.5)
    assert grid.z_axis()[-1] == pytest.approx(-2.0 + 4 * 1.0)


def test_read_nlloc_grid_slow_len_conversion(tmp_path):
    """SLOW_LEN grids (docs/nlloc_interface.md: what the bundled sample and
    Podvin-Lecomte both produce) store ``dx / velocity``
    (Vel2Grid3D.c:959) -- confirm the inverse conversion.
    """
    numx = numy = numz = 2
    dx = dy = dz = 0.05
    true_velocity = 5.0
    slow_len_value = dx / true_velocity
    values = np.full((numx, numy, numz), slow_len_value, dtype=np.float64)

    hdr = _write_nlloc_grid(
        tmp_path / "vel_slowlen",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="SLOW_LEN",
        values=values,
    )
    grid = read_nlloc_grid(hdr)
    assert grid.velocity_km_s[0, 0, 0] == pytest.approx(true_velocity)


def test_read_nlloc_grid_double_precision_rejected(tmp_path):
    hdr_path = tmp_path / "bad.P.mod.hdr"
    hdr_path.write_text("2 2 2  0.0 0.0 0.0  1.0 1.0 1.0 VELOCITY DOUBLE\n")
    (tmp_path / "bad.P.mod.buf").write_bytes(b"\x00" * (8 * 8))
    with pytest.raises(ConfigError, match="double precision"):
        read_nlloc_grid(hdr_path)


def test_read_nlloc_grid_unrecognized_grid_type_raises(tmp_path):
    numx = numy = numz = 2
    values = np.ones((numx, numy, numz))
    hdr = _write_nlloc_grid(
        tmp_path / "vel_bad",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        grid_type="NOT_A_REAL_TYPE",
        values=values,
    )
    with pytest.raises(ConfigError, match="unrecognized"):
        read_nlloc_grid(hdr)


def test_read_nlloc_grid_wrong_buf_size_raises(tmp_path):
    hdr_path = tmp_path / "bad.P.mod.hdr"
    hdr_path.write_text("2 2 2  0.0 0.0 0.0  1.0 1.0 1.0 VELOCITY FLOAT\n")
    (tmp_path / "bad.P.mod.buf").write_bytes(b"\x00" * 4)  # way too short
    with pytest.raises(ConfigError, match="expected"):
        read_nlloc_grid(hdr_path)


# ---------------------------------------------------------------------------
# _build_axis_nodes
# ---------------------------------------------------------------------------


def test_build_axis_nodes_covers_domain_and_adds_infinity_nodes():
    nodes = _build_axis_nodes(
        domain_min=0.0, domain_max=1.0, spacing_km=0.5, bld=0.1, infinity_km=10.0, axis_name="x"
    )
    # interior: 0.0, 0.5, 1.0 -- plus one infinity node 10.0 km past each end
    assert nodes == pytest.approx([-10.0, 0.0, 0.5, 1.0, 11.0])


def test_build_axis_nodes_rejects_non_bld_multiple_spacing():
    with pytest.raises(ConfigError, match="not a multiple of bld"):
        _build_axis_nodes(
            domain_min=0.0,
            domain_max=1.0,
            spacing_km=0.37,
            bld=0.1,
            infinity_km=10.0,
            axis_name="x",
        )


# ---------------------------------------------------------------------------
# nlloc_grid_to_vel3d: bld resampling from fine NLLoc spacing
# ---------------------------------------------------------------------------


def test_bld_resampling_from_fine_nlloc_spacing(tmp_path):
    """An NLLoc grid at 0.05 km spacing (PLAN.md §3's default) is correctly
    resampled onto a bld=0.1 km SIMUL2000 grid: assert node count and a few
    interpolated velocity values against hand computation.
    """
    # A small NLLoc grid: 1 km cube at 0.05 km spacing (21 nodes/axis),
    # velocity varies linearly in x only: v(x) = 4.0 + 2.0*x (x in km).
    numx = numy = numz = 21
    dx = dy = dz = 0.05
    origx = origy = origz = 0.0
    x_axis = origx + dx * np.arange(numx)
    values = np.empty((numx, numy, numz))
    for i, x in enumerate(x_axis):
        values[i, :, :] = 4.0 + 2.0 * x

    hdr = _write_nlloc_grid(
        tmp_path / "fine",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=origx,
        origy=origy,
        origz=origz,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)

    out_path = tmp_path / "vel3D.dat"
    nlloc_grid_to_vel3d(
        grid,
        out_path,
        lat_3d=37.878078,
        lon_3d=-122.243950,
        rot_3d=0.0,
        vp_vs_ratio=1.73,
        bld=0.1,
        node_spacing_km=0.2,  # coarser than bld, still a multiple of it
        infinity_km=5.0,
        max_nx=25,
        max_ny=25,
        max_nz=25,
    )

    result = read_vel3d_header(out_path)
    assert result.bld == pytest.approx(0.1)
    # domain [0.0, 1.0] at spacing 0.2 -> 6 interior nodes (0.0, 0.2, ...,
    # 1.0), plus 2 infinity nodes = 8 per axis.
    assert result.nx == result.ny == result.nz == 8
    assert result.xn[0] == pytest.approx(-5.0)
    assert result.xn[-1] == pytest.approx(6.0)
    assert result.xn[1:-1] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # hand computation: v(x) = 4.0 + 2.0*x at each interior x node
    text_lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
    # header(1) + xn(1) + yn(1) + zn(1) + fixed(1) + linked(1) = 6 lines
    # before the P-velocity block's first row (k=0, j=0).
    first_p_row = [float(v) for v in text_lines[6].split()]
    expected = [4.0 + 2.0 * x for x in [-5.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 6.0]]
    # the two infinity nodes clamp to the nearest real edge (x=0.0 and
    # x=1.0 respectively), not linearly extrapolated.
    expected[0] = 4.0 + 2.0 * 0.0
    expected[-1] = 4.0 + 2.0 * 1.0
    assert first_p_row == pytest.approx(expected, abs=1e-6)

    # Vp/Vs block must also be present (constant ratio everywhere)
    n_p_rows = 8 * 8  # nz * ny
    vpvs_row = [float(v) for v in text_lines[6 + n_p_rows].split()]
    assert vpvs_row == pytest.approx([1.73] * 8)


def test_bld_resampling_default_spacing_matches_bld(tmp_path):
    """With `node_spacing_km` left at its default (`bld`), the resampled
    grid uses the finest legal resolution.
    """
    numx = numy = numz = 5
    dx = dy = dz = 0.1
    values = np.full((numx, numy, numz), 5.0)
    hdr = _write_nlloc_grid(
        tmp_path / "grid",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)
    out_path = tmp_path / "vel3D.dat"
    nlloc_grid_to_vel3d(
        grid,
        out_path,
        lat_3d=37.878078,
        lon_3d=-122.243950,
        rot_3d=0.0,
        vp_vs_ratio=1.73,
        bld=0.1,
        infinity_km=1.0,
        max_nx=25,
        max_ny=25,
        max_nz=25,
    )
    result = read_vel3d_header(out_path)
    # domain [0.0, 0.4] at spacing 0.1 (== bld) -> 5 interior nodes + 2
    # infinity nodes = 7.
    assert result.nx == 7


# ---------------------------------------------------------------------------
# fail loudly on exceeding compiled limits
# ---------------------------------------------------------------------------


def test_exceeds_compiled_limits_fails_loudly(tmp_path):
    """An NLLoc grid whose resampled node count would exceed a tiny
    maxnx: assert a clear, actionable error, not silent truncation.
    """
    numx = numy = numz = 21
    dx = dy = dz = 0.05
    values = np.full((numx, numy, numz), 5.0)
    hdr = _write_nlloc_grid(
        tmp_path / "fine2",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)
    out_path = tmp_path / "vel3D.dat"

    with pytest.raises(ConfigError, match="exceeds the compiled vel3d.inc limit"):
        nlloc_grid_to_vel3d(
            grid,
            out_path,
            lat_3d=37.878078,
            lon_3d=-122.243950,
            rot_3d=0.0,
            vp_vs_ratio=1.73,
            bld=0.1,
            node_spacing_km=0.1,  # 1km domain / 0.1km = 10 interior + 2 = 12 nodes
            max_nx=5,  # far too small
            max_ny=25,
            max_nz=25,
        )
    # the file must not have been written at all (no silent partial output)
    assert not out_path.exists()


def test_exceeds_map_span_fails_loudly(tmp_path):
    """A huge `infinity_km` relative to `bld` overflows vel3d.inc's
    ixkms/iykms/izkms compiled map-size limit -- must fail loudly too.
    """
    numx = numy = numz = 3
    values = np.full((numx, numy, numz), 5.0)
    hdr = _write_nlloc_grid(
        tmp_path / "tiny",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=0.1,
        dy=0.1,
        dz=0.1,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)
    with pytest.raises(ConfigError, match="ixkms/iykms/izkms"):
        nlloc_grid_to_vel3d(
            grid,
            tmp_path / "vel3D.dat",
            lat_3d=37.878078,
            lon_3d=-122.243950,
            rot_3d=0.0,
            vp_vs_ratio=1.73,
            bld=0.1,
            infinity_km=2000.0,  # way past the 1000 km (bld=0.1) map limit
        )


def test_bld_illegal_value_rejected(tmp_path):
    numx = numy = numz = 2
    values = np.full((numx, numy, numz), 5.0)
    hdr = _write_nlloc_grid(
        tmp_path / "g",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)
    with pytest.raises(ConfigError, match="bld must be one of"):
        nlloc_grid_to_vel3d(
            grid,
            tmp_path / "vel3D.dat",
            lat_3d=0.0,
            lon_3d=0.0,
            rot_3d=0.0,
            vp_vs_ratio=1.73,
            bld=0.5,
        )


def test_coordinate_frame_mismatch_rejected(tmp_path):
    numx = numy = numz = 2
    values = np.full((numx, numy, numz), 5.0)
    hdr = _write_nlloc_grid(
        tmp_path / "g",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        grid_type="VELOCITY",
        values=values,
    )
    grid = read_nlloc_grid(hdr)  # TRANS SIMPLE lat=37.878078 lon=-122.24395 rot=0
    with pytest.raises(ConfigError, match="do not match"):
        nlloc_grid_to_vel3d(
            grid,
            tmp_path / "vel3D.dat",
            lat_3d=10.0,  # wrong
            lon_3d=-122.243950,
            rot_3d=0.0,
            vp_vs_ratio=1.73,
        )


# ---------------------------------------------------------------------------
# per-node Vp/Vs from a co-located S grid
# ---------------------------------------------------------------------------


def test_per_node_vp_vs_from_s_grid(tmp_path):
    numx = numy = numz = 3
    dx = dy = dz = 1.0
    p_values = np.full((numx, numy, numz), 6.0)
    s_values = np.full((numx, numy, numz), 3.0)  # Vp/Vs = 2.0 everywhere

    p_hdr = _write_nlloc_grid(
        tmp_path / "model",
        file_type="P",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=p_values,
    )
    s_hdr = _write_nlloc_grid(
        tmp_path / "model",
        file_type="S",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=0.0,
        origy=0.0,
        origz=0.0,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=s_values,
    )
    p_grid = read_nlloc_grid(p_hdr)
    s_grid = read_nlloc_grid(s_hdr)

    out_path = tmp_path / "vel3D.dat"
    nlloc_grid_to_vel3d(
        p_grid,
        out_path,
        lat_3d=37.878078,
        lon_3d=-122.243950,
        rot_3d=0.0,
        s_grid=s_grid,
        bld=1.0,
        node_spacing_km=1.0,
        infinity_km=5.0,
        max_nx=25,
        max_ny=25,
        max_nz=25,
    )
    result = read_vel3d_header(out_path)
    text_lines = [ln for ln in out_path.read_text().splitlines() if ln.strip()]
    n_p_rows = result.nz * result.ny
    vpvs_row = [float(v) for v in text_lines[6 + n_p_rows].split()]
    assert vpvs_row == pytest.approx([2.0] * result.nx)


# ---------------------------------------------------------------------------
# requires_hypodd: real get_vel3d.f acceptance test
# ---------------------------------------------------------------------------


def _read_example1_stations(example1_dir: Path) -> dict[str, tuple[float, float]]:
    stations = {}
    with open(example1_dir / "station.dat") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            stations[parts[0]] = (float(parts[1]), float(parts[2]))
    return stations


def _read_example1_events(example1_dir: Path) -> list[dict]:
    events: list[dict] = []
    current = None
    with open(example1_dir / "El16.pha") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                parts = line.split()
                (_, yr, mo, dy, hr, mn, sec, lat, lon, depth, mag, herr, verr, res, cuspid) = parts
                current = {
                    "time": UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mn), float(sec)),
                    "lat": float(lat),
                    "lon": float(lon),
                    "depth_km": float(depth),
                    "mag": float(mag),
                    "herr": float(herr),
                    "verr": float(verr),
                    "res": float(res),
                    "cuspid": int(cuspid),
                    "picks": [],
                }
                events.append(current)
            else:
                parts = line.split()
                sta, tt, wt, pha = parts[0], float(parts[1]), float(parts[2]), parts[3]
                current["picks"].append((sta, tt, wt, pha))
    return events


def _build_example1_catalog(
    stations: dict[str, tuple[float, float]], events: list[dict]
) -> tuple[Catalog, dict[str, int], pd.DataFrame]:
    """Same shape as ``tests/test_relocate_limits.py``'s own helper of the
    same name -- duplicated here (not imported) since ``tests/`` has no
    ``__init__.py`` (not an importable package, ARCHITECTURE.md §8's test
    layout), and this is this task's own proof the converted vel3D.dat is
    accepted end-to-end, using example1's real, already-working 16-event
    dataset rather than a small synthetic one (a tiny synthetic network's
    near-coincident hypocentres was found, in this task's own development,
    to segfault the real simulps ray tracer regardless of the 3D model
    file's content -- confirmed by reproducing the identical segfault with
    example1's own real vel3D.dat substituted in -- so example1's real,
    proven-working geometry is used instead, per this task's notes:
    "run hypoDD IMOD=9 with the output via a copy of example1 inputs").
    """
    cat = Catalog()
    event_id_map: dict[str, int] = {}
    used_stations: set[str] = set()

    for ev in events:
        origin = Origin(
            time=ev["time"],
            latitude=ev["lat"],
            longitude=ev["lon"],
            depth=ev["depth_km"] * 1000.0,
        )
        origin.origin_uncertainty = OriginUncertainty(horizontal_uncertainty=ev["herr"] * 1000.0)
        origin.depth_errors = QuantityError(uncertainty=ev["verr"] * 1000.0)
        origin.quality = OriginQuality(standard_error=ev["res"])

        event = make_event(event_id_from_time(ev["time"]))
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        mag_obj = Magnitude(mag=ev["mag"], magnitude_type="ML")
        event.magnitudes = [mag_obj]
        event.preferred_magnitude_id = mag_obj.resource_id

        for sta, tt, wt, pha in ev["picks"]:
            if sta not in stations:
                continue
            used_stations.add(sta)
            pick = Pick(
                time=ev["time"] + tt,
                phase_hint=pha,
                waveform_id=WaveformStreamID(network_code="XX", station_code=sta),
            )
            event.picks.append(pick)
            weight = None if wt == -1.0 else max(0.0, min(1.0, abs(wt)))
            arrival = Arrival(pick_id=pick.resource_id, phase=pha, time_weight=weight)
            origin.arrivals.append(arrival)

        cat.append(event)
        event_id_map[event_id(event)] = ev["cuspid"]

    rows = [
        {
            "network": "XX",
            "station": label,
            "location": "",
            "latitude": stations[label][0],
            "longitude": stations[label][1],
            "elevation_m": 0.0,
            "depth_m": 0.0,
        }
        for label in sorted(used_stations)
    ]
    station_table = pd.DataFrame(rows)
    return cat, event_id_map, station_table


HYPODD_MISSING = not (
    EXAMPLE1_DIR.exists()
    and (EXTERNAL_DIR / "bin" / "hypoDD").exists()
    and (EXTERNAL_DIR / "bin" / "ph2dt").exists()
)


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD/examples/example1 or bin not available")
def test_output_accepted_by_real_hypodd(tmp_path):
    """The produced vel3D.dat is accepted without error by the real
    hypoDD binary (IMOD=9): builds a synthetic NLLoc P-velocity grid
    covering example1's real 16-event/75-station network footprint,
    converts it, and runs the real ph2dt + hypoDD binaries against a copy
    of example1's own station/phase inputs with hypoDD.inp's IMOD=9 block
    pointing at the converted file -- confirming no ``stop`` from
    get_vel3d.f's ``bld``/S-velocity validation and a completed relocation
    (``hypoDD.reloc`` with all 16 events).
    """
    bin_dir = EXTERNAL_DIR / "bin"

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, event_id_map, station_table = _build_example1_catalog(stations, events)
    n_events = len(cat.events)
    assert n_events == 16  # sanity check on the fixture itself

    label_map = build_station_label_map(station_table, tmp_path / "station_label_map.csv")

    cfg = load_config(EXAMPLE_CONFIG)
    cfg.hypodd.ph2dt.maxdist_km = 120
    cfg.hypodd.ph2dt.maxsep_km = 10
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100
    cfg.hypodd.max_station_dist_km = 120
    cfg.hypodd.velocity_model.type = "3d"
    cfg.hypodd.velocity_model.origin_3d = (37.878078, -122.243950)
    cfg.hypodd.velocity_model.rotation_3d_deg = 0.0
    # IDAT=2 (catalog data only): getdata.f:366 only opens dt.cc when
    # idata is 1 or 3 -- with no real cross-correlation data available
    # here, catalog-only avoids needing a dt.cc file at all (dt.ct is
    # ph2dt's own output, produced below). Default Simul2000RaytraceConfig
    # has iphase=2 (S phases used), so the Vp/Vs block below is required.
    cfg.hypodd.idat = 2

    write_station_dat(tmp_path / "station.dat", station_table, label_map, cfg)
    write_phase_dat(tmp_path / "phase.dat", cat, event_id_map, label_map)
    write_event_dat(tmp_path / "event.dat", cat, event_id_map)
    write_ph2dt_inp(tmp_path / "ph2dt.inp", cfg, station_file="station.dat", phase_file="phase.dat")
    run([str(bin_dir / "ph2dt"), "ph2dt.inp"], cwd=tmp_path)
    assert (tmp_path / "event.sel").exists()
    assert (tmp_path / "station.sel").exists()

    # --- build + convert a synthetic NLLoc 3D grid covering example1's
    # real network footprint (station.dat spans roughly +-60 km around
    # the TRANS origin) and depth range (events down to ~25 km) ---
    numx = numy = 61
    numz = 16
    dx = dy = dz = 2.0
    origx = origy = -60.0
    origz = -2.0
    p_values = np.full((numx, numy, numz), 5.0)
    p_hdr = _write_nlloc_grid(
        tmp_path / "models" / "vel3d",
        file_type="P",
        numx=numx,
        numy=numy,
        numz=numz,
        origx=origx,
        origy=origy,
        origz=origz,
        dx=dx,
        dy=dy,
        dz=dz,
        grid_type="VELOCITY",
        values=p_values,
    )
    nlloc_grid = read_nlloc_grid(p_hdr)

    vel3d_path = tmp_path / "vel3D.dat"
    nlloc_grid_to_vel3d(
        nlloc_grid,
        vel3d_path,
        lat_3d=37.878078,
        lon_3d=-122.243950,
        rot_3d=0.0,
        vp_vs_ratio=1.73,
        bld=1.0,
        node_spacing_km=10.0,
        infinity_km=800.0,
        max_nx=25,
        max_ny=25,
        max_nz=20,
    )
    # M9.3's own header/limit validation (issue notes: validate with
    # read_vel3d_header/Vel3DGrid before the real-binary test).
    grid_header = read_vel3d_header(vel3d_path)
    assert grid_header.bld == 1.0
    assert grid_header.nx <= 25 and grid_header.ny <= 25 and grid_header.nz <= 20

    write_hypodd_inp_3d(tmp_path / "hypoDD.inp", cfg, cfg.hypodd.weighting_rows, vel3d_path)

    run([str(bin_dir / "hypoDD"), "hypoDD.inp"], cwd=tmp_path, timeout=120)

    reloc_path = tmp_path / "hypoDD.reloc"
    assert reloc_path.exists()
    reloc_lines = [line for line in reloc_path.read_text().splitlines() if line.strip()]
    assert len(reloc_lines) == n_events

    # hypoDD.log is Fortran-written and occasionally carries a non-UTF-8
    # byte; read as bytes rather than assuming an encoding (matches
    # tests/test_relocate_limits.py's own end-to-end test).
    log_bytes = (tmp_path / "hypoDD.log").read_bytes()
    assert b"no s-velocities" not in log_bytes.lower()
    assert b"bld must be" not in log_bytes.lower()
    assert b"fatal" not in log_bytes.lower()
