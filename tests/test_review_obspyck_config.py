"""Tests for OpenMEQ-c3l.2 (M7.2): ``openmeq.review.obspyck_config``.

These tests build their own tiny (2-station, no waveform data) OpenMEQ
project rather than using ``tests/data/make_synthetic.py``'s ~500 MB
``synthetic_project`` fixture: ``generate_obspyck_config`` never reads
waveform data, only ``cfg.stations``/``cfg.nlloc``/``cfg.review`` and the
NLLoc grid files ``build_1d_grid``/``run_grid2time`` produce, so the much
smaller fixture below is enough to exercise every acceptance criterion
without an extra large synthetic-dataset copy (see the worker instructions'
"Disk space" note).

Every test here calls (transitively, via ``generate_obspyck_config`` ->
``openmeq.nlloc_setup.link_obspyck_model``) ``openmeq.external.bin_path("NLLoc")``,
so the whole module is ``requires_nlloc``, matching ``tests/test_nlloc_setup.py``'s
own convention.
"""

from __future__ import annotations

import configparser
import platform
import textwrap
from pathlib import Path

import pytest

from openmeq.config import load_config
from openmeq.external import ExternalToolMissing, bin_path
from openmeq.nlloc_setup import build_1d_grid, run_grid2time
from openmeq.paths import RunPaths
from openmeq.review.obspyck_config import (
    _EXAMPLE_CFG_PATH,
    MANDATORY_BASE_KEYS,
    generate_obspyck_config,
)
from openmeq.stations import station_table as load_station_table

pytestmark = pytest.mark.requires_nlloc


def _require_nlloc_tools() -> None:
    try:
        bin_path("NLLoc")
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


def _write_minimal_project(tmp_path: Path) -> Path:
    """A minimal, valid OpenMEQ project (2 stations, no waveform data)."""
    project = tmp_path / "project"
    project.mkdir()

    (project / "stations.csv").write_text(
        "net,sta,loc,lat,lon,elev_m,depth_m\n"
        "OQ,ST1,,-38.02,176.01,100.0,0.0\n"
        "OQ,ST2,,-37.985,175.975,50.0,0.0\n"
    )

    (project / "vel1d.txt").write_text(
        "# depth_top_km  vp_km_s  vs_km_s  rho_g_cc\n"
        "0.0   3.50  2.00  2.3\n"
        "1.0   4.50  2.60  2.5\n"
        "3.0   5.80  3.35  2.7\n"
    )

    config_yaml = project / "config.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            """\
            project:
              name: obspyck_config_test
              output_dir: ./run

            data:
              waveform_dirs: [./data]
              start: 2025-01-01
              end: 2025-01-02

            parallel:
              n_workers: 1

            stations:
              csv: ./stations.csv

            chunks:
              pre_s: 15
              post_s: 15

            nlloc:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
              grid:
                origin: [-38.00, 176.00]
                size_km: [20, 20, 10]
                spacing_km: 0.5
              obspyck_pluginpath: ./obspyck_plugin

            review:
              enabled: true
              obspyck_config: ./config/obspyck.cfg

            hypodd:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
            """
        )
    )
    return config_yaml


def _parse(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(allow_no_value=True)
    parser.optionxform = str
    read = parser.read(path)
    assert read == [str(path)], f"failed to parse {path}"
    return parser


def test_generated_config_has_required_sections(tmp_path):
    _require_nlloc_tools()
    config_path = _write_minimal_project(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    target = generate_obspyck_config(cfg, paths)
    assert target == Path(cfg.review.obspyck_config)
    assert target.exists()

    parser = _parse(target)
    for section in ("base", "seismic_phases", "seed_id_lookup", "keys", "nonlinloc"):
        assert parser.has_section(section), f"missing [{section}]"

    for key in MANDATORY_BASE_KEYS:
        assert parser.has_option("base", key), f"[base] missing key {key!r}"

    # duration derives from cfg.chunks.pre_s + post_s (15 + 15).
    assert parser.getfloat("base", "duration") == pytest.approx(30.0)
    assert parser.get("base", "pluginpath") == str(cfg.nlloc.obspyck_pluginpath)
    assert parser.get("base", "event_server") == ""
    # ObsPyck's `merge` is a string enum ("", "safe", "overwrite"); "true"
    # is not a valid value (see the module docstring's deviation note).
    assert parser.get("base", "merge") in ("", "safe", "overwrite")

    assert parser.get("seismic_phases", "P")
    assert parser.get("seismic_phases", "S")
    assert parser.options("seed_id_lookup") == []
    assert parser.getfloat("nonlinloc", "default_pick_uncertainty") > 0


def test_no_server_sections_emitted(tmp_path):
    _require_nlloc_tools()
    config_path = _write_minimal_project(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    target = generate_obspyck_config(cfg, paths)
    parser = _parse(target)

    assert not any(s.startswith("server_") for s in parser.sections())
    assert "station_combinations" not in parser.sections()


def test_keys_section_matches_shipped_example(tmp_path):
    _require_nlloc_tools()
    config_path = _write_minimal_project(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    target = generate_obspyck_config(cfg, paths)
    parser = _parse(target)

    example = configparser.ConfigParser(allow_no_value=True)
    example.optionxform = str
    example.read(_EXAMPLE_CFG_PATH)

    assert dict(parser.items("keys")) == dict(example.items("keys"))


def test_pluginpath_layout(tmp_path):
    _require_nlloc_tools()
    try:
        bin_path("Vel2Grid")
        bin_path("Grid2Time")
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")

    config_path = _write_minimal_project(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    table = load_station_table(cfg.stations)
    build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    generate_obspyck_config(cfg, paths)

    pluginpath = Path(cfg.nlloc.obspyck_pluginpath)
    assert (pluginpath / "hyp_2000").is_dir()
    assert list((pluginpath / "hyp_2000").iterdir()) == []
    assert (pluginpath / "focmec").is_dir()
    assert list((pluginpath / "focmec").iterdir()) == []

    exe_name = f"NLLoc__{platform.system()}__{platform.architecture()[0]}"
    exe_path = pluginpath / "nlloc" / exe_name
    assert exe_path.exists()

    control = pluginpath / "nlloc" / "locate_BY.nlloc"
    assert control.exists()
    control_text = control.read_text()

    vm_link = pluginpath / "VELOCITY_MODELS"
    assert vm_link.is_symlink()
    assert vm_link.resolve() == paths.nlloc_velocity_models_dir.resolve()

    # LOCFILES obsFile obsFileType ttimeFileRoot locFilesOutRoot -- the
    # control file's ttime-grid-root must resolve to grids that actually
    # exist on disk (Vel2Grid/Grid2Time output), named by bare station code
    # (docs/obspyck_notes.md's station-naming pitfall).
    locfiles_line = next(line for line in control_text.splitlines() if line.startswith("LOCFILES"))
    grid_root_rel = locfiles_line.split()[3]
    resolved_root = (pluginpath / "nlloc" / grid_root_rel).resolve()
    assert (resolved_root.parent / f"{resolved_root.name}.P.mod.hdr").exists()
    assert (resolved_root.parent / f"{resolved_root.name}.S.mod.hdr").exists()
    for row in table.itertuples(index=False):
        assert (resolved_root.parent / f"{resolved_root.name}.P.{row.station}.time.hdr").exists()
        assert (resolved_root.parent / f"{resolved_root.name}.S.{row.station}.time.hdr").exists()


def test_skip_unchanged(tmp_path):
    _require_nlloc_tools()
    config_path = _write_minimal_project(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    target = generate_obspyck_config(cfg, paths)
    control = Path(cfg.nlloc.obspyck_pluginpath) / "nlloc" / "locate_BY.nlloc"
    target_mtime_1 = target.stat().st_mtime_ns
    control_mtime_1 = control.stat().st_mtime_ns

    target_again = generate_obspyck_config(cfg, paths)
    assert target_again == target
    assert target.stat().st_mtime_ns == target_mtime_1, "ini rewritten despite no config change"
    assert control.stat().st_mtime_ns == control_mtime_1, (
        "pluginpath rebuilt (link_obspyck_model re-run) despite no config change"
    )

    # force=True always regenerates, regardless of the manifest.
    generate_obspyck_config(cfg, paths, force=True)
    assert target.stat().st_mtime_ns >= target_mtime_1

    # Changing a fingerprinted input (cfg.review here) is picked up on the
    # next call even without force=True.
    cfg.review.enabled = not cfg.review.enabled
    target_mtime_forced = target.stat().st_mtime_ns
    generate_obspyck_config(cfg, paths)
    assert target.stat().st_mtime_ns >= target_mtime_forced
