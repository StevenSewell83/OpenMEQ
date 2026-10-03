"""Integration tests for ``openmeq qc`` (M1.7, ``OpenMEQ-5nl.7``).

Wires M1.1-M1.6's already-tested pieces (index, station merge, coverage
report, availability, daily stats, monthly stats + channel flags,
diagnostics) into one ``openmeq qc -c config.yaml`` invocation, exercised
here end to end via click's ``CliRunner`` against a private copy of the
shared ``synthetic_project`` fixture -- this is the epic's own integration
test (ARCHITECTURE.md §3's full ``inventory/``/``qc/`` output layout must
all be produced by one command).

Per ``tests/conftest.py``'s usage note, ``openmeq qc`` writes into
``cfg.project.output_dir`` under the project directory, so this module
copies the whole synthetic project into its own **module**-scoped
``tmp_path_factory`` directory once (the dataset is ~500 MB -- see the
worker instructions' "Disk space" note) rather than once per test
function, and every test in this module shares that one copy/run through
the ``qc_first_run`` fixture below.

OpenMEQ-m1f.6: this module drives its own real ``openmeq qc`` invocation(s)
rather than reading ``tests/conftest.py``'s shared ``full_pipeline_run``
fixture -- its own tests exercise ``StepManifest``'s skip-unchanged/
``--force`` behaviour (real, repeated CLI invocations that mutate the same
output directory), which that issue's own rule keeps as an isolated run
rather than migrating. The whole module is marked ``slow`` (~70s on this
machine, two real ``openmeq qc`` passes) so it doesn't count against the
``-m "not slow"`` budget.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from openmeq.cli import main
from openmeq.config import load_config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qc_project(synthetic_project, tmp_path_factory) -> tuple[Path, dict]:
    """A private, module-scoped copy of the synthetic project.

    ``openmeq qc`` writes into ``cfg.project.output_dir`` (under the
    project directory here), so the shared session-scoped
    ``synthetic_project`` fixture must not be mutated directly (its own
    docstring's usage note) -- copy it once per test module instead.
    """
    config_path, truth = synthetic_project
    project_dir = tmp_path_factory.mktemp("qc_command_project") / "project"
    shutil.copytree(config_path.parent, project_dir)
    return project_dir / "config.yaml", truth


@pytest.fixture(scope="module")
def qc_first_run(qc_project):
    """Invoke ``openmeq qc -c <config>`` once for the whole module.

    Returns ``(result, config_path, output_dir, truth, elapsed_s)``. Every
    test in this module depends on this fixture (directly or transitively)
    so the expensive first run happens exactly once, regardless of which
    test triggers it first or the order tests run in.
    """
    config_path, truth = qc_project
    cfg = load_config(config_path)
    output_dir = cfg.project.output_dir

    runner = CliRunner()
    t0 = time.perf_counter()
    result = runner.invoke(main, ["qc", "-c", str(config_path)])
    elapsed = time.perf_counter() - t0

    print(
        f"\n[openmeq qc integration] full run on the synthetic dataset: "
        f"{elapsed:.2f}s (exit_code={result.exit_code})"
    )

    return result, config_path, output_dir, truth, elapsed


#: ARCHITECTURE.md §3's inventory/ and qc/ output layout this command must
#: produce (qc/diagnostics/*.png and logs/qc.log are checked separately).
_EXPECTED_FILES = [
    "0_setup/index.json",
    "0_setup/stations.xml",
    "0_setup/stations.csv",
    "0_setup/metadata_report.json",
    "1_qc/availability.csv",
    "1_qc/daily_stats.csv",
    "1_qc/monthly_stats.csv",
    "1_qc/channel_flags.json",
    "1_qc/manifest.json",
    "logs/qc.log",
]

#: Files whose mtime is meaningful for "did the qc step actually redo
#: work" -- excludes logs/qc.log, which every invocation appends
#: "starting"/"done" lines to (openmeq/cli.py's ``_dispatch``), even a
#: skipped one, so its mtime always changes and isn't a useful skip/force
#: signal on its own (the "up to date" log *message* is checked instead).
_WORK_OUTPUT_FILES = [rel for rel in _EXPECTED_FILES if rel != "logs/qc.log"]


def test_qc_command_runs_end_to_end(qc_first_run):
    result, _config_path, output_dir, _truth, _elapsed = qc_first_run

    assert result.exit_code == 0, result.output

    for rel in _EXPECTED_FILES:
        path = output_dir / rel
        assert path.is_file(), f"{rel} was not created under {output_dir}"

    diagnostics_dir = output_dir / "1_qc" / "diagnostics"
    pngs = sorted(diagnostics_dir.glob("*.png"))
    assert pngs, f"expected at least one qc/diagnostics/*.png under {diagnostics_dir}"

    log_text = (output_dir / "logs" / "qc.log").read_text()
    assert "openmeq qc" in log_text


def test_qc_command_recovers_known_issues(qc_first_run):
    """The known spiky and flat-lined channel-days (``5nl.9``) are flagged
    not-ok with the expected reasons, and are the *only* channel-days
    flagged not-ok under the default config (the gap channel-day's ~10 min
    of missing data does not cross ``qc.min_availability_pct``'s default,
    per ``tests/test_qc_monthly_flags.py``'s own "zero false positives"
    acceptance criterion -- asserted against ``truth["known_issues"]``
    directly, per the orchestrator note: ``truth["known_issues"]`` is
    guaranteed by M0.8, ARCHITECTURE.md §8)."""
    _result, _config_path, output_dir, truth, _elapsed = qc_first_run

    payload = json.loads((output_dir / "1_qc" / "channel_flags.json").read_text())
    assert payload["version"] == 1
    flags = payload["flags"]

    spike_issue = next(i for i in truth["known_issues"] if i["kind"] == "spikes")
    flat_issue = next(i for i in truth["known_issues"] if i["kind"] == "flat")
    spike_seed_id, spike_date = spike_issue["seed_id"], spike_issue["start"].date.isoformat()
    flat_seed_id, flat_date = flat_issue["seed_id"], flat_issue["start"].date.isoformat()

    # Per the M1.7/5nl.9 acceptance criterion, spelled out exactly:
    assert spike_seed_id == "XX.STA06.00.HHE"
    assert spike_date == "2025-01-02"
    assert flat_seed_id == "XX.STA03.00.HHN"
    assert flat_date == "2025-01-01"

    spike_entry = flags[spike_seed_id][spike_date]
    assert spike_entry["ok"] is False
    assert "spikes" in spike_entry["reasons"]

    flat_entry = flags[flat_seed_id][flat_date]
    assert flat_entry["ok"] is False
    assert "flat" in flat_entry["reasons"]

    not_ok = {
        (seed_id, date_str)
        for seed_id, days in flags.items()
        for date_str, entry in days.items()
        if not entry["ok"]
    }
    assert not_ok == {(spike_seed_id, spike_date), (flat_seed_id, flat_date)}


def test_qc_command_skip_unchanged(qc_first_run):
    """Running the command again with no input changes is a no-op."""
    result, config_path, output_dir, _truth, _elapsed = qc_first_run
    assert result.exit_code == 0, result.output

    watched = [output_dir / rel for rel in _WORK_OUTPUT_FILES]
    mtimes_before = {path: path.stat().st_mtime_ns for path in watched}

    runner = CliRunner()
    second = runner.invoke(main, ["qc", "-c", str(config_path)])
    assert second.exit_code == 0, second.output

    mtimes_after = {path: path.stat().st_mtime_ns for path in watched}
    assert mtimes_after == mtimes_before, "a no-op second run touched some output file(s)"

    log_text = (output_dir / "logs" / "qc.log").read_text()
    assert "up to date" in log_text


def test_qc_command_force_reruns(qc_first_run):
    """``--force`` bypasses skip-unchanged and rewrites every output."""
    result, config_path, output_dir, _truth, _elapsed = qc_first_run
    assert result.exit_code == 0, result.output

    watched = [output_dir / rel for rel in _WORK_OUTPUT_FILES]
    mtimes_before = {path: path.stat().st_mtime_ns for path in watched}

    runner = CliRunner()
    forced = runner.invoke(main, ["qc", "-c", str(config_path), "--force"])
    assert forced.exit_code == 0, forced.output

    mtimes_after = {path: path.stat().st_mtime_ns for path in watched}
    changed = [path for path in watched if mtimes_after[path] == mtimes_before[path]]
    assert not changed, f"--force did not rewrite: {changed}"
