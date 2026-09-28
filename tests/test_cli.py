"""Tests for openmeq.cli: the ``openmeq`` click group (M0.4).

Uses click's ``CliRunner`` against a patched copy of ``config/example.yaml``
(only ``project.output_dir`` is redirected to a temp directory, so the run
doesn't pollute the repo) -- the acceptance criterion this satisfies is
"each [subcommand] accepts -c/--config and runs to completion against the
example config, producing at least the logs/<step>.log file and the ensured
directory tree, even though the step body is a stub."
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from openmeq.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"

STEP_COMMANDS = [
    "qc",
    "detect",
    "chunk",
    "pick",
    "locate",
    "review",
    "magnitude",
    "relocate",
    "run",
]


def _patched_example_config(tmp_path: Path) -> tuple[Path, Path]:
    """A copy of config/example.yaml with project.output_dir -> tmp_path/run."""
    text = EXAMPLE_CONFIG.read_text()
    output_dir = tmp_path / "run"
    patched, n = re.subn(
        r"^(\s*output_dir:\s*).*$",
        lambda m: f"{m.group(1)}{output_dir}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert n == 1, "expected exactly one project.output_dir line in config/example.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(patched)
    return config_path, output_dir


def test_help_lists_every_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for step in STEP_COMMANDS:
        assert step in result.output, f"{step!r} missing from `openmeq --help` output"


def test_missing_config_option_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["qc"])
    assert result.exit_code != 0


def test_nonexistent_config_file_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["qc", "-c", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


@pytest.mark.parametrize("step", STEP_COMMANDS)
def test_step_runs_to_completion_against_example_config(tmp_path: Path, step: str) -> None:
    config_path, output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, [step, "-c", str(config_path)])

    assert result.exit_code == 0, result.output

    # `review` (M7.3 / OpenMEQ-c3l.3) is implemented -- unlike every other
    # still-stubbed step, it doesn't print "not implemented"; instead, since
    # config/example.yaml ships with `review.enabled: false` ("true to pause
    # the pipeline for manual ObsPyck review"), it prints a one-line
    # "nothing to do" message and exits 0 without touching
    # locate/catalog_auto.xml (which doesn't exist yet at this point in the
    # real pipeline, `locate` itself still being a stub here).
    expected_substring = "enabled is false" if step == "review" else "not implemented"
    assert expected_substring in result.output

    log_file = output_dir / "logs" / f"{step}.log"
    assert log_file.is_file(), f"{log_file} was not created"
    assert expected_substring in log_file.read_text()

    # The ensured directory tree (ARCHITECTURE.md §3), spot-checked.
    for rel in ("inventory", "qc/diagnostics", "logs", "chunks", "locate/passA", "relocate/cache"):
        assert (output_dir / rel).is_dir(), f"{rel} was not created under {output_dir}"


def test_review_compile_flag_runs(tmp_path: Path) -> None:
    config_path, output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["review", "-c", str(config_path), "--compile"])

    assert result.exit_code == 0, result.output
    assert (output_dir / "logs" / "review.log").is_file()


def test_magnitude_catalog_option_runs(tmp_path: Path) -> None:
    config_path, output_dir = _patched_example_config(tmp_path)
    catalog_path = tmp_path / "external_catalog.xml"
    catalog_path.write_text("<q:quakeml xmlns:q='dummy'/>")
    runner = CliRunner()

    result = runner.invoke(
        main, ["magnitude", "-c", str(config_path), "--catalog", str(catalog_path)]
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "logs" / "magnitude.log").is_file()


def test_two_steps_in_one_process_get_separate_log_files(tmp_path: Path) -> None:
    """Regression check for the "no duplicate/stale handler" logging pitfall
    when several subcommands run in one process (as CliRunner does here)."""
    config_path, output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    result_qc = runner.invoke(main, ["qc", "-c", str(config_path)])
    result_detect = runner.invoke(main, ["detect", "-c", str(config_path)])

    assert result_qc.exit_code == 0, result_qc.output
    assert result_detect.exit_code == 0, result_detect.output

    qc_log = (output_dir / "logs" / "qc.log").read_text()
    detect_log = (output_dir / "logs" / "detect.log").read_text()
    assert "openmeq qc" in qc_log
    assert "openmeq detect" in detect_log
    assert "openmeq detect" not in qc_log
