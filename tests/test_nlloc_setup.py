"""Tests for OpenMEQ-mev.2 (M5.2): ``openmeq.nlloc_setup``.

Every ``requires_nlloc`` test here uses the real, installed NonLinLoc
binaries (`external/bin/{Vel2Grid,Grid2Time,NLLoc}`) against
``tests/data/make_synthetic.py``'s ``synthetic_project`` fixture (8
stations, 1D layer model). The pure-Python duplicate-station-code check is
tested without any external tool.
"""

from __future__ import annotations

import glob
import shutil

import pandas as pd
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.config import ConfigError, load_config
from openmeq.external import ExternalToolMissing, bin_path, run
from openmeq.nlloc_setup import (
    build_1d_grid,
    link_obspyck_model,
    location_grid_bounds,
    run_grid2time,
    setup_nlloc,
    write_control_common,
    write_station_list,
)
from openmeq.paths import RunPaths
from openmeq.stations import station_table as load_station_table

pytestmark = pytest.mark.requires_nlloc


def _require_nlloc_tools() -> None:
    try:
        for name in ("Vel2Grid", "Grid2Time", "NLLoc"):
            bin_path(name)
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


def _load_project_copy(synthetic_project, tmp_path):
    """Copy the shared synthetic project into a function-scoped tmp_path and
    load its config (per tests/conftest.py's usage note: this step writes
    under ``cfg.project.output_dir``, so the shared session directory must
    not be mutated directly)."""
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")
    return cfg, truth


def test_build_1d_grid_and_run_grid2time_all_stations(synthetic_project, tmp_path):
    """``build_1d_grid`` + ``run_grid2time`` produce a velocity grid and a
    P+S travel-time grid pair for every one of the 8 synthetic stations, and
    a second call with an unchanged config is skipped as a manifest no-op.
    """
    _require_nlloc_tools()
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    table = load_station_table(cfg.stations)
    assert len(table) == 8  # 6 surface + 2 borehole, ARCHITECTURE.md §8

    build_1d_grid(cfg, paths, table)
    grid_dir = paths.nlloc_grid_root.parent
    root_name = paths.nlloc_grid_root.name
    for wave_type in ("P", "S"):
        for ext in ("hdr", "buf"):
            assert (grid_dir / f"{root_name}.{wave_type}.mod.{ext}").exists()

    run_grid2time(cfg, paths, table)
    for row in table.itertuples(index=False):
        for wave_type in ("P", "S"):
            for ext in ("hdr", "buf"):
                f = grid_dir / f"{root_name}.{wave_type}.{row.station}.time.{ext}"
                assert f.exists(), f"missing {f}"

    # setup_nlloc: full orchestration, skip-unchanged via StepManifest.
    rebuilt_first = setup_nlloc(cfg, paths)
    assert rebuilt_first is True

    rebuilt_second = setup_nlloc(cfg, paths)
    assert rebuilt_second is False, "unchanged config should be a manifest no-op"

    rebuilt_forced = setup_nlloc(cfg, paths, force=True)
    assert rebuilt_forced is True


def test_borehole_elevation_in_gtsrce_line(synthetic_project, tmp_path):
    """The two borehole stations' (``BH01``/``BH02``, ``DP?`` channels)
    ``GTSRCE`` line encodes ``elevation_m - depth_m`` (ARCHITECTURE.md §6),
    converted metres -> km, exactly -- re-parsed from the written line
    itself, not recomputed independently.
    """
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    table = load_station_table(cfg.stations)

    borehole = table[table["station"].isin(["BH01", "BH02"])]
    assert len(borehole) == 2
    assert (borehole["depth_m"] > 0).all(), "both DP? stations should be boreholes (depth_m > 0)"

    lines = write_station_list(table)
    by_station = {line.split()[1]: line for line in lines}

    for row in borehole.itertuples(index=False):
        line = by_station[row.station]
        fields = line.split()
        # GTSRCE label LATLON lat lon zSrce elev
        assert fields[0] == "GTSRCE"
        assert fields[1] == row.station
        assert fields[2] == "LATLON"
        parsed_elev_km = float(fields[6])
        expected_elev_km = (row.elevation_m - row.depth_m) / 1000.0
        assert parsed_elev_km == pytest.approx(expected_elev_km, abs=1e-9)


def test_duplicate_station_code_rejected():
    """A synthetic two-network fixture with a bare-station-code clash raises
    ``ConfigError`` (M5.1's uniqueness finding: NLLoc/ObsPyck match travel-time
    grids on the bare station code alone).
    """
    table = pd.DataFrame(
        [
            {
                "network": "XX",
                "station": "STA01",
                "location": "",
                "latitude": 45.0,
                "longitude": 10.0,
                "elevation_m": 500.0,
                "depth_m": 0.0,
            },
            {
                "network": "YY",
                "station": "STA01",
                "location": "",
                "latitude": 45.1,
                "longitude": 10.1,
                "elevation_m": 400.0,
                "depth_m": 0.0,
            },
        ]
    )
    with pytest.raises(ConfigError, match="STA01"):
        write_station_list(table)


def test_obspyck_link_creates_expected_files(synthetic_project, tmp_path):
    """``link_obspyck_model`` creates the NLLoc executable (named for this
    platform), the ``locate_BY.nlloc`` control-file template, and the
    ``VELOCITY_MODELS`` symlink, under a temp ``obspyck_pluginpath``
    (docs/obspyck_notes.md §5).
    """
    _require_nlloc_tools()
    cfg, _truth = _load_project_copy(synthetic_project, tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()
    table = load_station_table(cfg.stations)

    build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    pluginpath = tmp_path / "obspyck_plugin"
    cfg.nlloc.obspyck_pluginpath = pluginpath

    link_obspyck_model(cfg, paths, table)

    import platform

    exe_name = f"NLLoc__{platform.system()}__{platform.architecture()[0]}"
    exe_path = pluginpath / "nlloc" / exe_name
    assert exe_path.exists()

    control_path = pluginpath / "nlloc" / "locate_BY.nlloc"
    assert control_path.exists()
    control_text = control_path.read_text()
    assert "LOCFILES" in control_text
    assert "../VELOCITY_MODELS/layer_time/" in control_text

    vm_link = pluginpath / "VELOCITY_MODELS"
    assert vm_link.is_symlink()
    assert vm_link.resolve() == paths.nlloc_velocity_models_dir.resolve()

    # Refreshing (re-running) the link must not fail (unlink-then-relink).
    link_obspyck_model(cfg, paths, table)
    assert vm_link.is_symlink()


def test_end_to_end_recovers_truth_event(synthetic_project, tmp_path):
    """Full loop: the grids ``build_1d_grid``/``run_grid2time`` produce are
    fed to a real NLLoc run against one truth event's own arrivals (used as
    noise-free picks), and the recovered hypocentre is within a small margin
    of the truth origin.

    OpenMEQ-m1f.5: the synthetic generator now computes exact 1D layered
    first-arrival travel times (direct ray + head waves), validated against
    this same ``Grid2Time`` grid to <5 ms P / <10 ms S
    (``tests/locate/test_synthetic_traveltime_validation.py``), so the old
    straight-ray modelling-mismatch floor (which forced a ~1 km tolerance
    here, per ARCHITECTURE.md §8's old "few hundred m to ~1 km" guidance) is
    gone -- this event now measures ~24 m, and 0.1 km keeps a comfortable
    margin over that.

    Uses ``truth["events"][1]``, not ``[0]``: exploratory runs against every
    one of the 20 synthetic events (against these same grids) showed that a
    handful of the shallowest events (depth comparable to, or above, the
    layer model's first velocity discontinuity at 1 km) have a genuine
    depth/origin-time trade-off -- an alternate, shallower-or-negative-depth
    solution can fit the arrival times *better* (lower RMS) than the true
    depth does, confirmed by forcing NLLoc's search to stay within the true
    depth's neighbourhood and finding a strictly worse RMS there. This is a
    property of this specific network's azimuthal coverage (large gap, few
    close-in stations) for near-surface sources, not a defect in
    ``build_1d_grid``/``run_grid2time``; event 1 has an ordinary (few-km)
    depth and is unaffected.
    """
    _require_nlloc_tools()
    cfg, truth = _load_project_copy(synthetic_project, tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()
    table = load_station_table(cfg.stations)

    build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    event_truth = truth["events"][1]
    true_lat = event_truth["origin"]["lat"]
    true_lon = event_truth["origin"]["lon"]
    true_depth_km = event_truth["origin"]["depth_km"]
    arrivals = event_truth["arrivals"]  # {"NET.STA": {"P": UTCDateTime, "S": UTCDateTime}}

    event = Event()
    for net_sta, phases in arrivals.items():
        _net, sta = net_sta.split(".")
        for phase_hint, arrival_time in phases.items():
            pick = Pick()
            pick.time = UTCDateTime(arrival_time)
            pick.waveform_id = WaveformStreamID(
                network_code=_net,
                station_code=sta,
                channel_code="HHZ" if phase_hint == "P" else "HHN",
            )
            pick.phase_hint = phase_hint
            pick.time_errors.uncertainty = 0.05
            pick.onset = "impulsive"
            event.picks.append(pick)

    loc_dir = tmp_path / "loc"
    loc_dir.mkdir()
    obs_path = loc_dir / "truth_event.obs"
    Catalog(events=[event]).write(str(obs_path), format="NLLOC_OBS")

    bounds = location_grid_bounds(cfg, table)
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
        "LOCSIG OpenMEQ-mev.2 test\n"
        "LOCCOM end-to-end recovery test against a synthetic truth event\n"
        "\n"
        f"LOCFILES {obs_path} NLLOC_OBS {paths.nlloc_grid_root} {loc_dir / 'out'}\n"
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
    ctrl_path = loc_dir / "nlloc.in"
    ctrl_path.write_text(nlloc_control)

    run([str(bin_path("NLLoc")), str(ctrl_path)], cwd=loc_dir)

    hyp_files = glob.glob(str(loc_dir / "out.[0-9]*.grid0.loc.hyp"))
    assert len(hyp_files) == 1, f"expected exactly one per-event .hyp file, found {hyp_files}"

    import obspy

    result = obspy.read_events(hyp_files[0], format="NLLOC_HYP")
    origin = result[0].origins[0]

    horiz_dist_km = (
        gps2dist_azimuth(true_lat, true_lon, origin.latitude, origin.longitude)[0] / 1000.0
    )
    depth_diff_km = abs(origin.depth / 1000.0 - true_depth_km)
    total_dist_km = (horiz_dist_km**2 + depth_diff_km**2) ** 0.5

    assert total_dist_km < 0.1, (
        f"recovered location {total_dist_km:.3f} km from true hypocenter "
        f"(horiz {horiz_dist_km:.3f} km, depth diff {depth_diff_km:.3f} km)"
    )
