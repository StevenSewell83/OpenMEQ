"""Tests for the ``openmeq pick`` CLI command (M4.4, ``OpenMEQ-1m7.4``).

Covers the CLI wiring (``openmeq/cli.py``'s ``pick`` subcommand, ``--force``),
the hard "run ``openmeq chunk`` first" dependency check, and
``StepManifest`` skip/``--force`` behaviour -- mirrors
``tests/test_chunk_command.py``'s own structure for the ``chunk`` command.

The flagship end-to-end run reuses the shared, session-scoped
``pick_ready_project`` fixture (``tests/pick/conftest.py``) rather than
re-running ``openmeq detect``/``openmeq chunk`` itself -- see that
fixture's own docstring for why sharing it across this module and
``tests/pick/test_recovery.py`` is safe.

``test_skip_unchanged_and_force`` always starts with an explicit
``force=True`` call before counting anything, rather than assuming its own
first call is "the" first real run -- the shared ``pick_ready_project``
fixture may already have a completed ``openmeq pick`` run on it from
whichever other test in this session touched it first (both this module's
own ``test_pick_command_runs_end_to_end`` and ``tests/pick/
test_recovery.py`` invoke ``openmeq pick`` against the same shared
project), so this test cannot assume it is starting from a clean manifest.
"""

from __future__ import annotations

import pytest
import yaml
from click.testing import CliRunner

import openmeq.pick as pick_module
from openmeq.cli import main
from openmeq.config import load_config


def _phasenet_available() -> bool:
    from openmeq.external import phasenet_dir, phasenet_model_dir

    try:
        model_dir = phasenet_model_dir()
        predict_py = phasenet_dir() / "phasenet" / "predict.py"
    except Exception:
        return False
    return predict_py.is_file() and (model_dir / "checkpoint").is_file()


skip_reason = (
    "external/PhaseNet clone/model not found (set OPENMEQ_EXTERNAL_DIR, or run install.sh)"
)


# ---------------------------------------------------------------------------
# test_requires_chunk_first
# ---------------------------------------------------------------------------


def test_requires_chunk_first(synthetic_project, tmp_path):
    """A fresh output dir with no prior ``openmeq chunk`` run fails clearly."""
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"

    new_config_path = tmp_path / "config.yaml"
    new_config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    runner = CliRunner()
    result = runner.invoke(main, ["pick", "-c", str(new_config_path)])

    assert result.exit_code != 0
    assert "openmeq chunk" in result.output


# ---------------------------------------------------------------------------
# test_pick_command_runs_end_to_end
# ---------------------------------------------------------------------------


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_pick_command_runs_end_to_end(pick_ready_project):
    cfg, paths, config_path, truth = pick_ready_project
    del cfg, truth

    runner = CliRunner()
    result = runner.invoke(main, ["pick", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    assert paths.catalog_picks_xml.is_file()
    assert paths.catalog_picks_xml.stat().st_size > 0

    assert paths.manifest_json("pick").is_file()

    diag_dir = paths.diagnostics_dir("pick")
    assert diag_dir.is_dir()
    assert (diag_dir / "summary.csv").is_file()
    assert list(diag_dir.glob("*.png")), "expected at least one PNG in pick/diagnostics/"

    assert paths.log_file("pick").is_file()

    # -c is required (matches every other subcommand's _CONFIG_OPTION).
    help_result = runner.invoke(main, ["pick", "--help"])
    assert help_result.exit_code == 0
    assert "--force" in help_result.output


# ---------------------------------------------------------------------------
# test_skip_unchanged_and_force
# ---------------------------------------------------------------------------


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_skip_unchanged_and_force(pick_ready_project, monkeypatch):
    """Mirrors ``tests/test_chunk_command.py``'s own manifest
    skip-unchanged/``--force`` test, spying on ``run_phasenet`` (the
    expensive, always-batched call) rather than re-running the whole step
    body to count invocations. See module docstring for why this starts
    with an explicit ``force=True`` call.
    """
    cfg, paths, config_path, truth = pick_ready_project
    del truth

    call_count = 0
    real_run_phasenet = pick_module.run_phasenet

    def _counting_run_phasenet(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_run_phasenet(*args, **kwargs)

    monkeypatch.setattr(pick_module, "run_phasenet", _counting_run_phasenet)

    # Force a known-real run first, regardless of any prior test's state.
    pick_module.run(cfg, paths, config_path=str(config_path), force=True)
    assert call_count == 1
    assert paths.catalog_picks_xml.is_file()
    mtime1 = paths.catalog_picks_xml.stat().st_mtime_ns

    # Unchanged: run_phasenet must not be called again, output mtime unmoved.
    pick_module.run(cfg, paths, config_path=str(config_path))
    assert call_count == 1
    assert paths.catalog_picks_xml.stat().st_mtime_ns == mtime1

    # --force reruns it regardless.
    pick_module.run(cfg, paths, config_path=str(config_path), force=True)
    assert call_count == 2
