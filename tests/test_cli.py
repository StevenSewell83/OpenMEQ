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
from obspy import Catalog, UTCDateTime
from obspy.core.event import Origin

from openmeq.catalog import make_event, write_catalog
from openmeq.cli import main
from openmeq.config import load_config
from openmeq.paths import RunPaths

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


#: ``qc`` is exercised end to end in tests/test_qc_command.py:
#: config/example.yaml's data paths are illustrative only and don't exist on
#: disk, so the real qc pipeline can't run against it.
STEP_COMMANDS_UNDER_TEST = [step for step in STEP_COMMANDS if step != "qc"]


@pytest.mark.parametrize("step", STEP_COMMANDS_UNDER_TEST)
def test_step_runs_or_fails_clearly_against_example_config(tmp_path: Path, step: str) -> None:
    """Every step either is still a stub, or is implemented and fails clearly.

    A fresh example.yaml run directory has no upstream outputs, so an
    implemented step must stop with a click error naming the command to run
    first (ARCHITECTURE.md §2) -- never a traceback. A still-stubbed step
    exits 0 and says "not implemented". ``review`` is implemented but ships
    with ``review.enabled: false``, so it exits 0 with a "nothing to do"
    message. This stays valid as more steps are implemented.
    """
    config_path, output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, [step, "-c", str(config_path)])

    assert "Traceback" not in result.output, result.output
    if step == "review":
        assert result.exit_code == 0, result.output
        assert "enabled is false" in result.output
    elif result.exit_code == 0:
        assert "not implemented" in result.output
    else:
        # Implemented: a clear "run the previous step first" message.
        assert "openmeq " in result.output, result.output
        assert result.exception is None or isinstance(result.exception, SystemExit), result.output

    log_file = output_dir / "logs" / f"{step}.log"
    if step != "run":
        assert log_file.is_file(), f"{log_file} was not created"

    # The ensured directory tree (ARCHITECTURE.md §3), spot-checked --
    # ``paths.ensure()`` runs before a step's body, so this holds even when
    # the step fails its dependency check.
    for rel in (
        "0_setup",
        "1_qc/diagnostics",
        "logs",
        "4_data_chunks/chunks",
        "7_location/passA",
        "8_relocation/cache",
    ):
        assert (output_dir / rel).is_dir(), f"{rel} was not created under {output_dir}"


def test_review_compile_flag_without_catalog_auto_fails_clearly(tmp_path: Path) -> None:
    """`review --compile` (M7.4 / OpenMEQ-c3l.4) is implemented -- unlike
    `review` on its own (still `enabled: false` in `config/example.yaml`,
    see the loop test above), it always tries to merge
    `locate/catalog_auto.xml`, which doesn't exist yet in a fresh
    `example.yaml`-based run directory (`locate` itself is still a stub in
    this test file's own scope). This is the same deliberate, CLI-level
    hard-dependency-check pattern `detect` uses (see the loop test above),
    not an incidental `FileNotFoundError`."""
    config_path, _output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["review", "-c", str(config_path), "--compile"])

    assert result.exit_code != 0, result.output
    assert "openmeq locate" in result.output
    assert "catalog_auto.xml" in result.output


def test_review_compile_flag_runs(tmp_path: Path) -> None:
    """With a real (if minimal) `locate/catalog_auto.xml` already in place,
    `review --compile` merges it (no reviewed events, no `state.json`) into
    `review/catalog_final.xml`."""
    config_path, output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    event = make_event("ev20260101T000000_000")
    origin = Origin(
        time=UTCDateTime("2026-01-01T00:00:00"), longitude=176.0, latitude=-38.0, depth=4000.0
    )
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id
    write_catalog(Catalog(events=[event]), paths.catalog_auto_xml)

    runner = CliRunner()
    result = runner.invoke(main, ["review", "-c", str(config_path), "--compile"])

    assert result.exit_code == 0, result.output
    assert (output_dir / "logs" / "review.log").is_file()
    assert (output_dir / "7b_review" / "catalog_final.xml").is_file()


def test_magnitude_catalog_option_runs(tmp_path: Path) -> None:
    """``magnitude`` (M8.3 / OpenMEQ-625.3) is implemented, so ``--catalog``
    must point at a real (if empty) QuakeML catalogue -- an empty catalogue
    is a valid, if trivial, input (zero accepted events, every output file
    still written with "no data" placeholders)."""
    from obspy import Catalog

    from openmeq.catalog import write_catalog

    config_path, output_dir = _patched_example_config(tmp_path)
    catalog_path = tmp_path / "external_catalog.xml"
    write_catalog(Catalog(), catalog_path)
    runner = CliRunner()

    result = runner.invoke(
        main, ["magnitude", "-c", str(config_path), "--catalog", str(catalog_path)]
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "logs" / "magnitude.log").is_file()
    assert (output_dir / "8_relocation" / "magnitude" / "catalog_mag.xml").is_file()
    assert (output_dir / "8_relocation" / "magnitude" / "station_magnitudes.csv").is_file()


def test_step_writes_config_snapshot(tmp_path: Path) -> None:
    """Every step copies the config it ran with to ``config_snapshot.yaml`` (OpenMEQ-9wj)."""
    config_path, output_dir = _patched_example_config(tmp_path)

    result = CliRunner().invoke(main, ["review", "-c", str(config_path)])

    assert result.exit_code == 0, result.output
    snapshot = output_dir / "config_snapshot.yaml"
    assert snapshot.read_text() == Path(config_path).read_text()


def test_two_steps_in_one_process_get_separate_log_files(tmp_path: Path) -> None:
    """Regression check for the "no duplicate/stale handler" logging pitfall
    when several subcommands run in one process (as CliRunner does here).

    Uses ``review`` (exits 0: ``review.enabled`` is false in example.yaml) and
    ``pick`` (fails its "run openmeq chunk first" check). Log separation
    doesn't depend on either step succeeding.
    """
    config_path, output_dir = _patched_example_config(tmp_path)
    runner = CliRunner()

    runner.invoke(main, ["review", "-c", str(config_path)])
    runner.invoke(main, ["pick", "-c", str(config_path)])

    review_log = (output_dir / "logs" / "review.log").read_text()
    pick_log = (output_dir / "logs" / "pick.log").read_text()
    assert "enabled is false" in review_log
    assert "openmeq pick" in pick_log
    assert "openmeq pick" not in review_log
    assert "enabled is false" not in pick_log
