"""Tests for openmeq.relocate.hypodd_limits and the 3D (IMOD=9) writer in
openmeq.relocate.hypodd_files (M9.3 / OpenMEQ-6ld.3).

Unit tests (`test_check_*`, `test_no_recompile_when_within_limits`,
`test_substitute_inc_params_*`) use tiny, fake ``.inc`` files -- not the
real ``external/HYPODD`` tree -- so they run instantly and need no
compiler. The single `requires_hypodd`/`slow` test
(`test_recompile_end_to_end`) copies the *real* ``external/HYPODD``
``include/``+``src/`` into a scratch directory (never the main checkout),
shrinks it to an artificially tiny ``MAXEVE``, and actually recompiles
twice with the real ``gfortran``/``make`` toolchain.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Arrival,
    Event,
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
    Vel3DGrid,
    build_station_label_map,
    read_vel3d_header,
    write_event_dat,
    write_hypodd_inp_3d,
    write_ph2dt_inp,
    write_phase_dat,
    write_station_dat,
)
from openmeq.relocate.hypodd_limits import (
    _substitute_inc_params,
    check_compiled_limits,
    ensure_compiled_limits,
    recompile_hypodd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"
REAL_HYPODD_DIR = EXTERNAL_DIR / "HYPODD"


# ---------------------------------------------------------------------------
# fake .inc fixtures (tiny, not the real tree)
# ---------------------------------------------------------------------------

# Mirrors the real hypoDD.inc's shape, including its commented-out "small
# size problems" alternate parameter block -- so tests exercise the same
# comment-skipping logic the real file requires.
_FAKE_HYPODD_INC = """\
c hypoDD.inc: FAKE, tiny, for tests only -- not the real compiled limits.
	integer*4 MAXEVE, MAXLAY, MAXDATA, MAXSTA, MAXEVE0, MAXDATA0
	integer*4 MAXCL

c parameters for large size problems
      parameter(MAXEVE=   {maxeve},
     &          MAXDATA=  {maxdata},
     &          MAXEVE0=  10,
     &          MAXDATA0= 100,
     &          MAXLAY=   30,
     &          MAXSTA=   {maxsta},
     &          MAXCL=    10)

cc parameters for small size problems (commented out -- must be ignored)
c      parameter(MAXEVE=   1,
c     &          MAXDATA=  1,
c     &          MAXEVE0=  1,
c     &          MAXDATA0= 1,
c     &          MAXLAY=   1,
c     &          MAXSTA=   1,
c     &          MAXCL=    1)
"""

_FAKE_PH2DT_INC = """\
c ph2dt.inc: FAKE, tiny, for tests only.
	integer	MEV, MSTA, MOBS

	parameter(MEV=    {mev},
     &		MSTA=   {msta},
     &		MOBS=   {mobs})
"""

_FAKE_VEL3D_INC = """\
      integer	maxnx,maxny,maxnz,maxnz2
c   maxnx: maximum number of nodes in x direction
      parameter(maxnx= {maxnx})
c   maxny: maximum number of nodes in y direction
      parameter(maxny= {maxny})
c   maxnz: maximum number of nodes in z direction
      parameter(maxnz= {maxnz})
"""


def _write_fake_inc_tree(
    root: Path,
    *,
    maxeve: int = 1000,
    maxdata: int = 100000,
    maxsta: int = 500,
    mev: int = 1000,
    msta: int = 500,
    mobs: int = 200,
    maxnx: int = 25,
    maxny: int = 25,
    maxnz: int = 20,
) -> Path:
    """Write a fake ``<root>/include/{hypoDD,ph2dt,vel3d}.inc`` tree with
    generous defaults (all overridable), returning `root`.
    """
    include_dir = root / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "hypoDD.inc").write_text(
        _FAKE_HYPODD_INC.format(maxeve=maxeve, maxdata=maxdata, maxsta=maxsta)
    )
    (include_dir / "ph2dt.inc").write_text(_FAKE_PH2DT_INC.format(mev=mev, msta=msta, mobs=mobs))
    (include_dir / "vel3d.inc").write_text(
        _FAKE_VEL3D_INC.format(maxnx=maxnx, maxny=maxny, maxnz=maxnz)
    )
    return root


# ---------------------------------------------------------------------------
# small synthetic-catalogue helpers (same shape as test_relocate_hypodd_files.py)
# ---------------------------------------------------------------------------


def _make_event(time: UTCDateTime, n_picks: int = 2) -> Event:
    eid = event_id_from_time(time)
    ev = make_event(eid)
    origin = Origin(time=time, latitude=37.9, longitude=-122.2, depth=8000.0)
    origin.origin_uncertainty = OriginUncertainty(horizontal_uncertainty=150.0)
    origin.depth_errors = QuantityError(uncertainty=500.0)
    origin.quality = OriginQuality(standard_error=0.02)
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    mag_obj = Magnitude(mag=1.5, magnitude_type="ML")
    ev.magnitudes = [mag_obj]
    ev.preferred_magnitude_id = mag_obj.resource_id

    for i in range(n_picks):
        pick = Pick(
            time=time + 1.0 + 0.1 * i,
            phase_hint="P",
            waveform_id=WaveformStreamID(network_code="XX", station_code=f"STA{i}"),
        )
        ev.picks.append(pick)
        arrival = Arrival(pick_id=pick.resource_id, phase="P", time_weight=0.8)
        origin.arrivals.append(arrival)
    return ev


def _station_table(n: int) -> pd.DataFrame:
    rows = [
        {
            "network": "XX",
            "station": f"STA{i}",
            "location": "",
            "latitude": 37.0 + 0.01 * i,
            "longitude": -122.0 - 0.01 * i,
            "elevation_m": 0.0,
            "depth_m": 0.0,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# check_compiled_limits
# ---------------------------------------------------------------------------


def test_check_detects_event_count_over_limit(tmp_path):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD", maxeve=5)  # MEV stays generous
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1) + i * 3600) for i in range(7)])
    station_table = _station_table(2)
    cfg = load_config(EXAMPLE_CONFIG)

    result = check_compiled_limits(cfg, cat, station_table, include_dir=fake_root / "include")

    event_check = result.by_name("event_count")
    assert event_check.current == 7
    assert event_check.limit == 5  # min(MAXEVE=5, MEV=1000)
    assert event_check.exceeded is True
    assert result.any_exceeded is True

    # station/observation/phases-per-event checks are all comfortably
    # within the fake tree's generous defaults.
    assert result.by_name("station_count").exceeded is False
    assert result.by_name("observation_count").exceeded is False
    assert result.by_name("phases_per_event").exceeded is False


def test_check_event_count_uses_smaller_of_maxeve_and_mev(tmp_path):
    # MEV (ph2dt) is the binding limit here, not MAXEVE (hypoDD) -- both
    # binaries matter, so the smaller one governs.
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD", maxeve=1000, mev=4)
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1) + i * 3600) for i in range(6)])
    result = check_compiled_limits(
        load_config(EXAMPLE_CONFIG), cat, _station_table(2), include_dir=fake_root / "include"
    )
    event_check = result.by_name("event_count")
    assert event_check.limit == 4
    assert event_check.exceeded is True


def test_check_detects_3d_grid_over_limit(tmp_path):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD", maxnx=5, maxny=5, maxnz=5)
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1))])
    station_table = _station_table(1)
    grid = Vel3DGrid(bld=1.0, nx=7, ny=3, nz=3, xn=[0.0] * 7, yn=[0.0] * 3, zn=[0.0] * 3)

    result = check_compiled_limits(
        load_config(EXAMPLE_CONFIG),
        cat,
        station_table,
        vel3d_grid=grid,
        include_dir=fake_root / "include",
    )

    assert result.by_name("grid_nx").current == 7
    assert result.by_name("grid_nx").limit == 5
    assert result.by_name("grid_nx").exceeded is True
    assert result.by_name("grid_ny").exceeded is False
    assert result.by_name("grid_nz").exceeded is False
    assert result.any_exceeded is True


def test_check_no_vel3d_grid_means_no_grid_checks(tmp_path):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1))])
    result = check_compiled_limits(
        load_config(EXAMPLE_CONFIG), cat, _station_table(1), include_dir=fake_root / "include"
    )
    names = {c.name for c in result.checks}
    assert not any(n.startswith("grid_") for n in names)


def test_check_reads_currently_compiled_values_not_shipped_defaults(tmp_path):
    """A previous recompile may have already changed the compiled limits;
    `check_compiled_limits` must reflect whatever is on disk, not hardcode
    the shipped 8000/2000/... defaults (work item 3's explicit requirement).
    """
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD", maxeve=9999999, mev=9999999)
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1) + i * 3600) for i in range(50)])
    result = check_compiled_limits(
        load_config(EXAMPLE_CONFIG), cat, _station_table(2), include_dir=fake_root / "include"
    )
    assert result.by_name("event_count").limit == 9999999
    assert result.by_name("event_count").exceeded is False


# ---------------------------------------------------------------------------
# _substitute_inc_params
# ---------------------------------------------------------------------------


def test_substitute_inc_params_preserves_comments_and_continuations():
    text = _FAKE_HYPODD_INC.format(maxeve=8000, maxdata=2800000, maxsta=2000)
    new_text, old_values = _substitute_inc_params(text, {"MAXEVE": 16000, "MAXSTA": 4000})

    assert old_values == {"MAXEVE": 8000, "MAXSTA": 2000}
    assert "MAXEVE=   16000" in new_text
    assert "MAXSTA=   4000" in new_text
    # the untouched parameter keeps its original value
    assert "MAXDATA=  2800000" in new_text
    # the commented-out alternate block must survive completely untouched
    assert "c      parameter(MAXEVE=   1," in new_text
    # continuation markers are untouched
    assert new_text.count("     &") == text.count("     &")


def test_substitute_inc_params_unknown_name_raises():
    text = _FAKE_PH2DT_INC.format(mev=1000, msta=500, mobs=200)
    with pytest.raises(ConfigError, match="NOSUCHPARAM"):
        _substitute_inc_params(text, {"NOSUCHPARAM": 5})


# ---------------------------------------------------------------------------
# ensure_compiled_limits / recompile_hypodd (mocked -- no real compiler)
# ---------------------------------------------------------------------------


def test_no_recompile_when_within_limits(tmp_path, monkeypatch):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")  # generous defaults
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1) + i * 3600) for i in range(3)])
    station_table = _station_table(2)

    mock_recompile = Mock()
    monkeypatch.setattr("openmeq.relocate.hypodd_limits.recompile_hypodd", mock_recompile)

    result = ensure_compiled_limits(
        load_config(EXAMPLE_CONFIG),
        cat,
        station_table,
        external_hypodd_dir=fake_root,
        external_bin_dir=tmp_path / "bin",
    )

    mock_recompile.assert_not_called()
    assert result.any_exceeded is False


def test_ensure_compiled_limits_recompiles_with_headroom(tmp_path, monkeypatch):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD", maxeve=5)
    cat = Catalog(events=[_make_event(UTCDateTime(2025, 1, 1) + i * 3600) for i in range(10)])
    station_table = _station_table(2)

    mock_recompile = Mock()
    monkeypatch.setattr("openmeq.relocate.hypodd_limits.recompile_hypodd", mock_recompile)

    result = ensure_compiled_limits(
        load_config(EXAMPLE_CONFIG),
        cat,
        station_table,
        external_hypodd_dir=fake_root,
        external_bin_dir=tmp_path / "bin",
        headroom_factor=1.5,
    )

    assert result.any_exceeded is True
    mock_recompile.assert_called_once()
    (new_limits, hypodd_dir, bin_dir), _kwargs = mock_recompile.call_args
    assert hypodd_dir == fake_root
    assert bin_dir == tmp_path / "bin"
    # 10 events * 1.5 headroom = 15, applied to both MAXEVE and MEV (the
    # same logical "event_count" quantity governs both binaries).
    assert new_limits["MAXEVE"] == 15
    assert new_limits["MEV"] == 15


def test_recompile_hypodd_unknown_param_raises(tmp_path):
    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")
    with pytest.raises(ConfigError, match="NOSUCHPARAM"):
        recompile_hypodd({"NOSUCHPARAM": 5}, fake_root, tmp_path / "bin")


# ---------------------------------------------------------------------------
# read_vel3d_header
# ---------------------------------------------------------------------------


def test_read_vel3d_header_against_example1(tmp_path):
    if not EXAMPLE1_DIR.exists():
        pytest.skip("external/HYPODD/examples/example1 not available")
    grid = read_vel3d_header(EXAMPLE1_DIR / "vel3D.dat")
    # examples/example1/vel3D.dat's own header line: "1.0 13 19 11 2 computed"
    # -- bld,nx,ny,nz consume only the first 4 tokens.
    assert grid.bld == 1.0
    assert (grid.nx, grid.ny, grid.nz) == (13, 19, 11)
    assert len(grid.xn) == 13
    assert len(grid.yn) == 19
    assert len(grid.zn) == 11
    assert grid.xn[0] == -800.0
    assert grid.zn[-1] == 500.0


def test_read_vel3d_header_multiline_grid(tmp_path):
    # A real vel3D.dat wraps each of xn/yn/zn on one line, but get_vel3d.f's
    # list-directed reads don't require that -- confirm this parser doesn't
    # either, by wrapping zn across two lines.
    path = tmp_path / "vel3D.dat"
    path.write_text("1.0 2 2 3 2 computed\n-10.0 10.0\n-20.0 20.0\n-5.0\n0.0 5.0\n")
    grid = read_vel3d_header(path)
    assert (grid.nx, grid.ny, grid.nz) == (2, 2, 3)
    assert grid.zn == [-5.0, 0.0, 5.0]


def test_read_vel3d_header_empty_file_raises(tmp_path):
    path = tmp_path / "empty.dat"
    path.write_text("")
    with pytest.raises(ConfigError):
        read_vel3d_header(path)


# ---------------------------------------------------------------------------
# write_hypodd_inp_3d field mapping (get_vel3d.f signature confirmation)
# ---------------------------------------------------------------------------


def _data_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("*")
    ]


def test_hypodd_inp_3d_requires_3d_config_fields():
    cfg = load_config(EXAMPLE_CONFIG)
    cfg.hypodd.velocity_model.type = "3d"
    cfg.hypodd.velocity_model.origin_3d = None  # not configured
    with pytest.raises(ConfigError):
        write_hypodd_inp_3d(Path("/dev/null"), cfg, cfg.hypodd.weighting_rows, "vel3D.dat")


def test_get9_3d_block_field_mapping_matches_get_vel3d_signature(tmp_path):
    """Confirms `write_hypodd_inp_3d`'s three IMOD=9 data lines, read back
    field-by-field, match the read order `src/hypoDD/getinp2.f:231-237`
    uses and the argument order `src/hypoDD/get_vel3d.f:6-9`'s subroutine
    signature declares -- both cross-checked against the call site at
    `src/hypoDD/hypoDD.f:449-452`:

        getinp2.f (hypoDD.inp -> variables):
            line 1 (l=14+niter): fn_mod3d               <- filename
            line 2 (l=15+niter): lat_3d, lon_3d, rot_3d
            line 3 (l=16+niter): ipha3d, ndip, iskip, scale1, scale2,
                                  xfac, tlim, nitpb

        hypoDD.f:449-452 call site:
            call get_vel3d(fn_mod3d, ipha3d, lat_3d, lon_3d, rot_3d,
                            ndip, iskip, scale1, scale2, xfac, tlim,
                            nitpb, log)

        get_vel3d.f:6-9 subroutine signature (same order as the call site):
            subroutine get_vel3d(fn_vel3d, iphase, lat_3d, lon_3d, rot_3d,
                                  indip, iiskip, iscale1, iscale2,
                                  ixfac, itlim, initpb, log)

    i.e. `Simul2000RaytraceConfig`'s field order (``iphase, ndip, iskip,
    scale1, scale2, xfac, tlim, nitpb``) is exactly line 3's read order.
    """
    cfg = load_config(EXAMPLE_CONFIG)
    cfg.hypodd.velocity_model.type = "3d"
    cfg.hypodd.velocity_model.origin_3d = (37.878078, -122.243950)
    cfg.hypodd.velocity_model.rotation_3d_deg = 12.5
    s = cfg.hypodd.velocity_model.simul2000
    s.iphase, s.ndip, s.iskip = 2, 9, 4
    s.scale1, s.scale2, s.xfac, s.tlim, s.nitpb = 1.0, 1.0, 1.35, 0.0005, 50

    out = tmp_path / "hypoDD.inp"
    write_hypodd_inp_3d(out, cfg, cfg.hypodd.weighting_rows, tmp_path / "models" / "vel3D.dat")

    lines = _data_lines(out.read_text())
    nset = len(cfg.hypodd.weighting_rows)
    # fixed common-block line count (see test_hypodd_inp_1d_section_order):
    # 10 filenames + IDAT line + clustering line + solution-control line = 13
    imod_idx = 13 + nset
    assert lines[imod_idx] == "9"

    filename_line = lines[imod_idx + 1]
    assert filename_line == "vel3D.dat"  # only the basename is written

    origin_line = lines[imod_idx + 2].split()
    lat, lon, rot = float(origin_line[0]), float(origin_line[1]), float(origin_line[2])
    assert lat == pytest.approx(37.878078)
    assert lon == pytest.approx(-122.243950)
    assert rot == pytest.approx(12.5)

    raytrace_line = lines[imod_idx + 3].split()
    iphase, ndip, iskip = int(raytrace_line[0]), int(raytrace_line[1]), int(raytrace_line[2])
    scale1, scale2, xfac, tlim = (float(v) for v in raytrace_line[3:7])
    nitpb = int(raytrace_line[7])
    assert (iphase, ndip, iskip) == (2, 9, 4)
    assert (scale1, scale2, xfac, tlim) == (1.0, 1.0, 1.35, 0.0005)
    assert nitpb == 50

    cid_idx = imod_idx + 4
    assert lines[cid_idx] == "0"
    assert len(lines) == cid_idx + 1


# ---------------------------------------------------------------------------
# example1 reader (for the requires_hypodd end-to-end test only)
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


HYPODD_MISSING = not (EXAMPLE1_DIR.exists() and (EXTERNAL_DIR / "bin" / "ph2dt").exists())


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt/hypoDD) not available")
def test_recompile_end_to_end(tmp_path, monkeypatch):
    """Shrinks a *scratch copy* of the real ``.inc`` files to an
    artificially tiny ``MAXEVE`` (smaller than example1's real 16-event
    dataset), confirms `check_compiled_limits` flags it, recompiles with a
    larger ``MAXEVE`` (bigger than 16), confirms the check now passes, and
    proves the rebuilt scratch ``hypoDD``/``ph2dt`` binaries actually run
    example1's 16-event dataset end-to-end (16 is strictly between the old
    tiny limit and the new larger one).

    Never touches the main checkout's `external/HYPODD` or `external/bin`:
    everything happens inside `tmp_path`.
    """
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    scratch_hypodd = tmp_path / "HYPODD"
    shutil.copytree(REAL_HYPODD_DIR / "include", scratch_hypodd / "include")
    shutil.copytree(REAL_HYPODD_DIR / "src", scratch_hypodd / "src")
    scratch_bin = tmp_path / "bin"

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, event_id_map, station_table = _build_example1_catalog(stations, events)
    n_events = len(cat.events)
    assert n_events == 16  # sanity check on the fixture itself

    tiny_limit = n_events - 5  # artificially small: below the real dataset
    larger_limit = n_events + 50  # comfortably above it

    # --- shrink (text-edit only, no build yet) and confirm the check flags it ---
    hypodd_inc_path = scratch_hypodd / "include" / "hypoDD.inc"
    text, _old = _substitute_inc_params(hypodd_inc_path.read_text(), {"MAXEVE": tiny_limit})
    hypodd_inc_path.write_text(text)

    cfg = load_config(EXAMPLE_CONFIG)
    result = check_compiled_limits(cfg, cat, station_table, include_dir=scratch_hypodd / "include")
    event_check = result.by_name("event_count")
    assert event_check.current == n_events
    assert event_check.limit == tiny_limit
    assert event_check.exceeded is True

    # --- recompile with a larger limit ---
    recompile_hypodd({"MAXEVE": larger_limit}, scratch_hypodd, scratch_bin)
    assert (scratch_bin / "hypoDD").is_file()
    assert (scratch_bin / "ph2dt").is_file()

    result2 = check_compiled_limits(cfg, cat, station_table, include_dir=scratch_hypodd / "include")
    assert result2.by_name("event_count").limit == larger_limit
    assert result2.by_name("event_count").exceeded is False

    # --- run the rebuilt scratch binaries end-to-end on the real dataset ---
    label_map = build_station_label_map(station_table, tmp_path / "station_label_map.csv")
    cfg.hypodd.ph2dt.maxdist_km = 120
    cfg.hypodd.ph2dt.maxsep_km = 10
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100
    cfg.hypodd.max_station_dist_km = 120

    write_station_dat(tmp_path / "station.dat", station_table, label_map, cfg)
    write_phase_dat(tmp_path / "phase.dat", cat, event_id_map, label_map)
    write_event_dat(tmp_path / "event.dat", cat, event_id_map)
    write_ph2dt_inp(tmp_path / "ph2dt.inp", cfg)

    run([str(scratch_bin / "ph2dt"), "ph2dt.inp"], cwd=tmp_path)
    assert (tmp_path / "event.sel").exists()
    assert (tmp_path / "station.sel").exists()

    shutil.copy(EXAMPLE1_DIR / "dt.cc", tmp_path / "dt.cc")

    from openmeq.relocate.hypodd_files import write_hypodd_inp_1d

    model = pd.DataFrame(
        {
            "depth_km": [0.00, 0.25, 1.50, 2.50, 3.50, 5.00, 6.00, 9.00, 15.00, 25.00],
            "vp": [1.42, 3.24, 4.82, 5.36, 5.60, 5.65, 5.90, 6.15, 6.60, 8.00],
            "vs": [v / 1.73 for v in [1.42, 3.24, 4.82, 5.36, 5.60, 5.65, 5.90, 6.15, 6.60, 8.00]],
        }
    )
    write_hypodd_inp_1d(tmp_path / "hypoDD.inp", cfg, cfg.hypodd.weighting_rows, model)

    run([str(scratch_bin / "hypoDD"), "hypoDD.inp"], cwd=tmp_path, timeout=600)

    reloc_path = tmp_path / "hypoDD.reloc"
    assert reloc_path.exists()
    reloc_lines = [line for line in reloc_path.read_text().splitlines() if line.strip()]
    assert len(reloc_lines) == n_events

    # hypoDD.log is Fortran-written and occasionally carries a non-UTF-8 byte
    # (e.g. a Latin-1 degree sign in a residual-statistics line); read as
    # bytes rather than assuming an encoding.
    log_bytes = (tmp_path / "hypoDD.log").read_bytes()
    assert b"fatal" not in log_bytes.lower()
