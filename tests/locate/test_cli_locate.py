"""Tests for the ``openmeq locate`` CLI command (M6.5, ``OpenMEQ-xdf.5``).

Covers the CLI wiring (``openmeq/cli.py``'s ``locate`` subcommand,
``--force``), the hard "run ``openmeq pick`` first" dependency check, and
``StepManifest`` skip/``--force`` behaviour -- mirrors
``tests/pick/test_cli_pick.py``'s/``tests/test_chunk_command.py``'s own
structure for their respective commands.

Each test points a fresh ``Config`` (loaded from the shared, session-scoped
``synthetic_project`` fixture) at a private ``tmp_path`` output directory
(worker instructions' "Disk space" section), and writes a small hand-built
``pick/catalog_picks.xml`` directly (following ``tests/locate/
test_locate_step.py``'s own convention: P on the station's Z channel, S on
its horizontal, from ``tests/data/make_synthetic.py``'s truth arrivals) --
this module never runs PhaseNet, so ``openmeq.locate.run``'s own NLLoc grid
setup (``openmeq.nlloc_setup.setup_nlloc``) is the only "real" step
exercised besides location itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from obspy import Catalog
from obspy.core.event import Pick, WaveformStreamID

import openmeq.locate as locate_module
from openmeq.catalog import make_event, write_catalog
from openmeq.cli import main
from openmeq.config import Config, load_config
from openmeq.paths import RunPaths

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import make_synthetic  # noqa: E402

pytestmark = pytest.mark.requires_nlloc

_CHANNELS_BY_STATION = {
    code: channels for code, _kind, _x, _y, _elev_m, _depth_m, channels in make_synthetic.STATIONS
}

_CLEAN_UNCERTAINTY_S = 0.02


def _p_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=_CHANNELS_BY_STATION[station][0],
    )
    pick.phase_hint = "P"
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _s_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=_CHANNELS_BY_STATION[station][1],
    )
    pick.phase_hint = "S"
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute
    (matches ``tests/pick/conftest.py``'s identical helper, duplicated here
    to avoid cross-test-module coupling, per that module's own docstring)."""
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


def _write_clean_picks_catalog(paths: RunPaths, truth: dict) -> None:
    """One clean, well-covered (depth >= 2 km, avoiding the shallow
    depth/origin-time trade-off M5.2 found) full 8-station P+S event."""
    ev = next(e for e in truth["events"] if e["origin"]["depth_km"] >= 2.0)
    event = make_event(ev["event_id_hint"])
    picks = []
    for station_key, times in ev["arrivals"].items():
        network, station = station_key.split(".")
        picks.append(_p_pick(network, station, times["P"]))
        picks.append(_s_pick(network, station, times["S"]))
    event.picks = picks
    write_catalog(Catalog(events=[event]), paths.catalog_picks_xml)


# ---------------------------------------------------------------------------
# test_requires_pick_first
# ---------------------------------------------------------------------------


def test_requires_pick_first(synthetic_project, tmp_path):
    """A fresh output dir with no prior ``openmeq pick`` run fails clearly."""
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    result = runner.invoke(main, ["locate", "-c", str(new_config_path)])

    assert result.exit_code != 0
    assert "openmeq pick" in result.output


# ---------------------------------------------------------------------------
# test_locate_command_runs_end_to_end_and_skip_unchanged
# ---------------------------------------------------------------------------


def test_locate_command_runs_end_to_end_and_skip_unchanged(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    _write_clean_picks_catalog(paths, truth)

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    result = runner.invoke(main, ["locate", "-c", str(new_config_path)])
    assert result.exit_code == 0, result.output

    assert paths.catalog_auto_xml.is_file()
    assert paths.locate_summary_csv.is_file()
    assert paths.removal_log_csv.is_file()
    assert paths.manifest_json("locate").is_file()

    diag_dir = paths.diagnostics_dir("locate")
    assert diag_dir.is_dir()
    assert (diag_dir / "summary.txt").is_file()
    assert list(diag_dir.glob("*.png")), "expected at least one PNG in locate/diagnostics/"

    assert paths.log_file("locate").is_file()

    mtime1 = paths.catalog_auto_xml.stat().st_mtime_ns

    # Second run, unchanged config -> no-op (StepManifest skip-unchanged):
    # locate/catalog_auto.xml is not rewritten.
    result2 = runner.invoke(main, ["locate", "-c", str(new_config_path)])
    assert result2.exit_code == 0, result2.output
    assert paths.catalog_auto_xml.stat().st_mtime_ns == mtime1

    # --force reruns regardless.
    result3 = runner.invoke(main, ["locate", "-c", str(new_config_path), "--force"])
    assert result3.exit_code == 0, result3.output
    assert paths.catalog_auto_xml.stat().st_mtime_ns != mtime1

    # -c is required, and --force is documented (matches every other
    # subcommand's _CONFIG_OPTION convention).
    help_result = runner.invoke(main, ["locate", "--help"])
    assert help_result.exit_code == 0
    assert "--force" in help_result.output


# ---------------------------------------------------------------------------
# test_skip_unchanged_and_force_via_module_run
# ---------------------------------------------------------------------------


def test_skip_unchanged_and_force_via_module_run(synthetic_project, tmp_path, monkeypatch):
    """Spies on ``run_locate`` (the expensive call) to count invocations
    directly, mirroring ``tests/pick/test_cli_pick.py::test_skip_unchanged_and_force``.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    _write_clean_picks_catalog(paths, truth)

    call_count = 0
    real_run_locate = locate_module.run_locate

    def _counting_run_locate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_run_locate(*args, **kwargs)

    monkeypatch.setattr(locate_module, "run_locate", _counting_run_locate)

    locate_module.run(cfg, paths, force=True)
    assert call_count == 1
    assert paths.catalog_auto_xml.is_file()

    locate_module.run(cfg, paths)
    assert call_count == 1

    locate_module.run(cfg, paths, force=True)
    assert call_count == 2
