"""Tests for OpenMEQ-mev.3 (M5.3): 3D NLLoc model grid support in
``openmeq.nlloc_setup`` (``check_3d_grid_coverage``, ``link_3d_grid``, and
``run_grid2time``'s shared ``GRID2D``/``GRID3D`` auto-detection).

Every ``requires_nlloc`` test here hand-builds its own small, laterally
varying 3D grid via ``tests/helpers_nlloc3d.py`` (see that module's
docstring for the verified ``.hdr``/``.buf`` format and why a hand-built
fixture is used instead of running ``Vel2Grid3D``) rather than depending on
any real-world 3D velocity model -- PLAN.md §1: 3D grids are always
user-supplied, this project never builds one for real use, only for tests.
Grids are kept coarse (2 km node spacing) and just large enough to cover
``synthetic_project``'s own location-grid extent, keeping total fixture size
in the tens of KB (the "keep grids small" disk-space constraint), reusing
the already-generated ``synthetic_project`` session fixture (stations,
config) rather than building a second synthetic dataset.
"""

from __future__ import annotations

import glob
import math
import shutil
from pathlib import Path

import pytest
from helpers_nlloc3d import build_small_3d_velocity_grid
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.config import ConfigError, load_config
from openmeq.external import ExternalToolMissing, bin_path, run
from openmeq.nlloc_setup import (
    NllocSetupError,
    build_1d_grid,
    check_3d_grid_coverage,
    link_3d_grid,
    location_grid_bounds,
    read_grid3d_hdr,
    run_grid2time,
    setup_nlloc,
    write_control_common,
)
from openmeq.paths import RunPaths
from openmeq.stations import station_table as load_station_table

pytestmark = pytest.mark.requires_nlloc


def _require_nlloc_tools() -> None:
    try:
        for name in ("Vel2Grid", "Grid2Time", "NLLoc", "Time2EQ"):
            bin_path(name)
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


def _load_project_copy(synthetic_project, tmp_path):
    """Same pattern as ``tests/test_nlloc_setup.py``'s helper of the same
    name -- see ``tests/conftest.py``'s ``synthetic_project`` usage note:
    a test that writes under ``cfg.project.output_dir`` must not mutate the
    shared session fixture directly."""
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")
    return cfg, truth


# Node spacing (km) for every hand-built grid here -- coarse, since
# check_3d_grid_coverage only cares about extent, and even the
# Grid2Time/NLLoc-recovery test only needs to resolve travel times well
# enough to recover a hypocenter approximately (not the ~cm precision the
# 1D/0.1km-spacing tests get).
GRID_SPACING_KM = 2.0


@pytest.fixture()
def covering_3d_grid(synthetic_project, tmp_path):
    """A small, laterally-varying P+S grid pair covering
    ``synthetic_project``'s own ``location_grid_bounds`` with a
    :data:`GRID_SPACING_KM` margin on every axis. Returns
    ``(cfg, table, bounds, root)``."""
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    table = load_station_table(cfg.stations)
    bounds = location_grid_bounds(cfg, table)
    lat0, lon0 = cfg.nlloc.grid.origin

    root = str(tmp_path / "supplied_3d" / "model3d")
    build_small_3d_velocity_grid(
        root,
        lat0=lat0,
        lon0=lon0,
        x_range_km=(bounds.x_min_km - GRID_SPACING_KM, bounds.x_max_km + GRID_SPACING_KM),
        y_range_km=(bounds.y_min_km - GRID_SPACING_KM, bounds.y_max_km + GRID_SPACING_KM),
        z_range_km=(bounds.z_min_km - GRID_SPACING_KM, bounds.z_max_km + GRID_SPACING_KM),
        spacing_km=GRID_SPACING_KM,
    )
    return cfg, table, bounds, root


def test_check_3d_grid_coverage_accepts_covering_grid(covering_3d_grid):
    """A hand-built grid deliberately covering ``synthetic_project``'s
    location grid extent (acceptance criterion #1) passes for both wave
    types."""
    _cfg, _table, bounds, root = covering_3d_grid
    check_3d_grid_coverage(Path(f"{root}.P.mod.hdr"), _cfg, bounds)
    check_3d_grid_coverage(Path(f"{root}.S.mod.hdr"), _cfg, bounds)


def test_check_3d_grid_coverage_rejects_shallow_grid(synthetic_project, tmp_path):
    """A grid one spacing-layer too shallow (its ``z_max`` falls short of
    the location grid's own ``z_max_km`` by exactly ``GRID_SPACING_KM``)
    raises ``ConfigError`` naming the exact shortfall (acceptance criterion
    #1's second half)."""
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    table = load_station_table(cfg.stations)
    bounds = location_grid_bounds(cfg, table)
    lat0, lon0 = cfg.nlloc.grid.origin

    root = str(tmp_path / "supplied_3d" / "shallow")
    build_small_3d_velocity_grid(
        root,
        lat0=lat0,
        lon0=lon0,
        x_range_km=(bounds.x_min_km - GRID_SPACING_KM, bounds.x_max_km + GRID_SPACING_KM),
        y_range_km=(bounds.y_min_km - GRID_SPACING_KM, bounds.y_max_km + GRID_SPACING_KM),
        # one whole layer too shallow: stops short of bounds.z_max_km.
        z_range_km=(bounds.z_min_km - GRID_SPACING_KM, bounds.z_max_km - GRID_SPACING_KM),
        spacing_km=GRID_SPACING_KM,
    )

    with pytest.raises(ConfigError, match="does not cover the location grid"):
        check_3d_grid_coverage(Path(f"{root}.P.mod.hdr"), cfg, bounds)


def test_check_3d_grid_coverage_rejects_transform_mismatch(covering_3d_grid):
    """A grid whose own ``TRANSFORM`` origin doesn't match
    ``cfg.nlloc.grid.origin`` is rejected even though its extent would
    otherwise cover the location grid -- the design note's TRANS-mismatch
    pitfall."""
    cfg, _table, bounds, _root = covering_3d_grid
    lat0, lon0 = cfg.nlloc.grid.origin

    root = str(_root) + "_mismatched"
    build_small_3d_velocity_grid(
        root,
        lat0=lat0 + 1.0,  # a whole degree off -- not a rounding artifact
        lon0=lon0,
        x_range_km=(bounds.x_min_km - GRID_SPACING_KM, bounds.x_max_km + GRID_SPACING_KM),
        y_range_km=(bounds.y_min_km - GRID_SPACING_KM, bounds.y_max_km + GRID_SPACING_KM),
        z_range_km=(bounds.z_min_km - GRID_SPACING_KM, bounds.z_max_km + GRID_SPACING_KM),
        spacing_km=GRID_SPACING_KM,
    )

    with pytest.raises(ConfigError, match="TRANSFORM"):
        check_3d_grid_coverage(Path(f"{root}.P.mod.hdr"), cfg, bounds)


def test_link_3d_grid_rejects_missing_files(synthetic_project, tmp_path):
    """A ``cfg.nlloc.velocity_model.file`` root with no grid files at all
    raises ``NllocSetupError``, not a confusing coverage-check error."""
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    table = load_station_table(cfg.stations)
    cfg.nlloc.velocity_model.type = "3d"
    cfg.nlloc.velocity_model.file = tmp_path / "does_not_exist"
    paths = RunPaths(cfg)
    paths.ensure()

    with pytest.raises(NllocSetupError, match="missing expected file"):
        link_3d_grid(cfg, paths, table)


def test_run_grid2time_1d_and_3d_paths_both_produce_self_consistent_shapes(
    synthetic_project, tmp_path
):
    """Regression test (acceptance criterion #2): ``run_grid2time`` needs no
    per-path branching -- it reads each phase's own model grid ``.hdr`` at
    run time (``_grid_mode_for_model``) to pick ``GTMODE GRID2D``/``GRID3D``,
    rather than being told which of :func:`build_1d_grid` (1D, always
    ``numx=2``) or :func:`link_3d_grid` (3D, ``numx>2`` here) produced its
    input.

    "Identically shaped" is checked two ways for a station common to both
    setups (``synthetic_project`` shares one station table across both):

    1. **Within each path**, the produced ``<root>.<phase>.<STA>.time.hdr``
       has ``(numy, numz)`` exactly matching its own ``<root>.<phase>.mod.hdr``
       -- true regardless of path (NLLoc's ``InitTimeGrid`` always
       duplicates the model grid's shape) -- but ``numx`` itself differs by
       *mode*, not by which OpenMEQ setup path produced the input: a
       ``GRID2D``-mode output time grid is deliberately written with
       ``numx == 1`` even though its own model grid has ``numx == 2``
       (verified from source, ``external/NonLinLoc/src/Grid2Time1.c:841-847``:
       "need only ix=0 sheet for 2D grids", the model's ``numx=2`` is only
       a Vel2Grid/``VGGRID`` bookkeeping requirement, not meaningful data),
       while a ``GRID3D``-mode output keeps its model's full ``numx``.
    2. **Across paths**, the 1D path's model grid has ``numx == 2`` (its
       ``VGGRID`` convention) while the 3D path's has ``numx > 2`` (a
       genuinely laterally-varying grid) -- proving the shared
       ``run_grid2time`` correctly dispatched ``GRID2D`` for one and
       ``GRID3D`` for the other from the same call, with no error either
       way.
    """
    _require_nlloc_tools()
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    table = load_station_table(cfg.stations)
    station = table.iloc[0].station

    # -- Path A: 1D (build_1d_grid -> Vel2Grid, GTMODE GRID2D) --------------
    paths_1d = RunPaths(cfg)
    paths_1d.ensure()
    build_1d_grid(cfg, paths_1d, table)
    run_grid2time(cfg, paths_1d, table)

    grid_dir_1d = paths_1d.nlloc_grid_root.parent
    root_1d = paths_1d.nlloc_grid_root.name
    model_hdr_1d = read_grid3d_hdr(grid_dir_1d / f"{root_1d}.P.mod.hdr")
    time_hdr_1d = read_grid3d_hdr(grid_dir_1d / f"{root_1d}.P.{station}.time.hdr")

    assert model_hdr_1d.numx == 2
    # GRID2D output is deliberately written with numx=1 (one radial "sheet"),
    # not the model's own numx=2 -- see the docstring's Grid2Time1.c citation.
    assert time_hdr_1d.numx == 1
    assert (time_hdr_1d.numy, time_hdr_1d.numz) == (model_hdr_1d.numy, model_hdr_1d.numz)

    # -- Path B: 3D (link_3d_grid, GTMODE GRID3D) ---------------------------
    cfg.nlloc.velocity_model.type = "3d"
    bounds = location_grid_bounds(cfg, table)
    lat0, lon0 = cfg.nlloc.grid.origin
    src_root = str(tmp_path / "supplied_3d" / "model3d")
    build_small_3d_velocity_grid(
        src_root,
        lat0=lat0,
        lon0=lon0,
        x_range_km=(bounds.x_min_km - GRID_SPACING_KM, bounds.x_max_km + GRID_SPACING_KM),
        y_range_km=(bounds.y_min_km - GRID_SPACING_KM, bounds.y_max_km + GRID_SPACING_KM),
        z_range_km=(bounds.z_min_km - GRID_SPACING_KM, bounds.z_max_km + GRID_SPACING_KM),
        spacing_km=GRID_SPACING_KM,
    )
    cfg.nlloc.velocity_model.file = src_root

    # A second, distinct output_dir so path B's grid_dir doesn't collide
    # with path A's (paths.nlloc_grid_root is derived from cfg.project.output_dir).
    cfg.project.output_dir = tmp_path / "project_3d_output"
    paths_3d = RunPaths(cfg)
    paths_3d.ensure()
    link_3d_grid(cfg, paths_3d, table)
    run_grid2time(cfg, paths_3d, table)

    grid_dir_3d = paths_3d.nlloc_grid_root.parent
    root_3d = paths_3d.nlloc_grid_root.name
    model_hdr_3d = read_grid3d_hdr(grid_dir_3d / f"{root_3d}.P.mod.hdr")
    time_hdr_3d = read_grid3d_hdr(grid_dir_3d / f"{root_3d}.P.{station}.time.hdr")

    assert model_hdr_3d.numx > 2
    assert (time_hdr_3d.numx, time_hdr_3d.numy, time_hdr_3d.numz) == (
        model_hdr_3d.numx,
        model_hdr_3d.numy,
        model_hdr_3d.numz,
    )


def test_setup_nlloc_dispatches_to_3d_path(covering_3d_grid):
    """``setup_nlloc`` itself (not just ``link_3d_grid`` directly) builds
    the full 3D setup when ``cfg.nlloc.velocity_model.type == "3d"``, and
    the manifest skip-unchanged / ``force`` behaviour works the same as the
    1D path (``tests/test_nlloc_setup.py``'s
    ``test_build_1d_grid_and_run_grid2time_all_stations``)."""
    cfg, table, _bounds, root = covering_3d_grid
    cfg.nlloc.velocity_model.type = "3d"
    cfg.nlloc.velocity_model.file = root
    paths = RunPaths(cfg)
    paths.ensure()

    rebuilt_first = setup_nlloc(cfg, paths)
    assert rebuilt_first is True

    grid_dir = paths.nlloc_grid_root.parent
    root_name = paths.nlloc_grid_root.name
    for wave_type in ("P", "S"):
        assert (grid_dir / f"{root_name}.{wave_type}.mod.hdr").exists()
        assert (grid_dir / f"{root_name}.{wave_type}.mod.hdr").is_symlink()
    for row in table.itertuples(index=False):
        for wave_type in ("P", "S"):
            for ext in ("hdr", "buf"):
                assert (grid_dir / f"{root_name}.{wave_type}.{row.station}.time.{ext}").exists()

    rebuilt_second = setup_nlloc(cfg, paths)
    assert rebuilt_second is False, "unchanged 3D config should be a manifest no-op"

    rebuilt_forced = setup_nlloc(cfg, paths, force=True)
    assert rebuilt_forced is True


def test_nlloc_recovers_truth_event_with_3d_grid(covering_3d_grid, tmp_path):
    """Confirms Grid2Time works on a genuinely 3D (laterally varying) grid,
    and that an NLLoc run against a self-consistent truth event (generated
    with Time2EQ against these same grids, the same pattern
    ``tests/test_nlloc_spike.py`` uses) recovers the hypocenter reasonably.

    Uses Time2EQ rather than ``synthetic_project``'s own exact 1D layered
    truth arrivals (OpenMEQ-m1f.5): those aren't consistent with this test's
    own toy 3D velocity field (arbitrary x/z gradients, not the project's 1D
    layer file), so comparing against them would test the wrong thing.
    Tolerance is looser than the 1D end-to-end test's (``tests/test_nlloc_
    setup.py``'s 0.1 km, 0.1 km grid spacing): this grid's
    :data:`GRID_SPACING_KM` (2 km) resolves travel times more coarsely, and
    this test's own toy velocity field is unrelated to the exact-travel-time
    validation OpenMEQ-m1f.5 did against the project's real 1D layer model
    (``tests/locate/test_synthetic_traveltime_validation.py``).
    """
    _require_nlloc_tools()
    cfg, table, bounds, root = covering_3d_grid
    cfg.nlloc.velocity_model.type = "3d"
    cfg.nlloc.velocity_model.file = root
    paths = RunPaths(cfg)
    paths.ensure()

    link_3d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    grid_root = paths.nlloc_grid_root
    lat0, lon0 = cfg.nlloc.grid.origin

    true_depth_km = 5.0
    # ~3 km north-east of the grid origin, well inside bounds and away from
    # the boundary the octree search starts from.
    true_lat = lat0 + 3.0 / 111.19
    true_lon = lon0 + 3.0 / (111.19 * math.cos(math.radians(lat0)))
    true_origin_time = UTCDateTime("2025-06-15T03:00:00")

    loc_dir = tmp_path / "loc3d"
    loc_dir.mkdir()

    eqsta_lines = "\n".join(
        f"EQSTA  {row.station}  {phase}  GAU 0.00   GAU 0.05"
        for row in table.drop_duplicates(subset=["network", "station"]).itertuples(index=False)
        for phase in ("P", "S")
    )
    time2eq_in = (
        write_control_common(cfg, paths) + "\n"
        f"EQFILES {grid_root} {loc_dir / 'synth.obs'}\n"
        "EQMODE SRCE_TO_STA\n"
        f"EQSRCE  TrueEvt  LATLON  {true_lat}  {true_lon}  {true_depth_km}  0.0\n"
        "EQVPVS  -1.0\n"
        "\n"
        f"{eqsta_lines}\n"
        "\n"
        "EQQUAL2ERR 0.05 0.1 0.5 1.0 99999.9\n"
    )
    ctrl_path = loc_dir / "time2eq.in"
    ctrl_path.write_text(time2eq_in)
    run([str(bin_path("Time2EQ")), ctrl_path.name], cwd=loc_dir)

    travel_times: dict[tuple[str, str], float] = {}
    for line in (loc_dir / "synth.obs").read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("PUBLIC_ID"):
            continue
        fields = line.split()
        travel_times[(fields[0], fields[4])] = float(fields[8])
    assert len(travel_times) == 2 * table["station"].nunique()

    event = Event()
    for (sta, phase), tt in travel_times.items():
        pick = Pick()
        pick.time = true_origin_time + tt
        pick.waveform_id = WaveformStreamID(
            network_code="XX", station_code=sta, channel_code="HHZ" if phase == "P" else "HHN"
        )
        pick.phase_hint = phase
        pick.time_errors.uncertainty = 0.05
        pick.onset = "impulsive"
        event.picks.append(pick)
    obs_path = loc_dir / "from_obspy.obs"
    Catalog(events=[event]).write(str(obs_path), format="NLLOC_OBS")

    spacing = cfg.nlloc.grid.spacing_km
    x_num = int((bounds.x_max_km - bounds.x_min_km) / spacing) + 1
    y_num = int((bounds.y_max_km - bounds.y_min_km) / spacing) + 1
    z_num = int((bounds.z_max_km - bounds.z_min_km) / spacing) + 1
    ls = cfg.nlloc.locsearch
    lg = cfg.nlloc.locgau
    lg2 = cfg.nlloc.locgau2
    locqual2err = " ".join(str(v) for v in cfg.nlloc.locqual2err_thresholds_s)

    nlloc_control = (
        write_control_common(cfg, paths) + "\n"
        "LOCSIG OpenMEQ-mev.3 test\n"
        "LOCCOM 3D-grid end-to-end recovery test against a Time2EQ truth event\n"
        "\n"
        f"LOCFILES {obs_path} NLLOC_OBS {grid_root} {loc_dir / 'out'}\n"
        "\n"
        "LOCHYPOUT SAVE_NLLOC_ALL\n"
        "\n"
        f"LOCSEARCH OCT {ls.init_num_cells_x} {ls.init_num_cells_y} {ls.init_num_cells_z} "
        f"{ls.min_node_size} {ls.max_num_nodes} {ls.num_scatter} "
        f"{1 if ls.use_stations_density else 0} {1 if ls.stop_on_min_node_size else 0}\n"
        "\n"
        f"LOCGRID {x_num} {y_num} {z_num} "
        f"{bounds.x_min_km} {bounds.y_min_km} {bounds.z_min_km} "
        f"{spacing} {spacing} {spacing} PROB_DENSITY SAVE\n"
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
    )
    nlloc_ctrl_path = loc_dir / "nlloc.in"
    nlloc_ctrl_path.write_text(nlloc_control)
    run([str(bin_path("NLLoc")), str(nlloc_ctrl_path)], cwd=loc_dir)

    hyp_files = glob.glob(str(loc_dir / "out.[0-9]*.grid0.loc.hyp"))
    assert len(hyp_files) == 1, f"expected exactly one per-event .hyp file, found {hyp_files}"

    import obspy

    origin = obspy.read_events(hyp_files[0], format="NLLOC_HYP")[0].origins[0]

    horiz_dist_km = (
        gps2dist_azimuth(true_lat, true_lon, origin.latitude, origin.longitude)[0] / 1000.0
    )
    depth_diff_km = abs(origin.depth / 1000.0 - true_depth_km)
    total_dist_km = (horiz_dist_km**2 + depth_diff_km**2) ** 0.5

    assert total_dist_km < 3.0, (
        f"recovered location {total_dist_km:.3f} km from true hypocenter "
        f"(horiz {horiz_dist_km:.3f} km, depth diff {depth_diff_km:.3f} km)"
    )
