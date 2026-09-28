"""Tests for ``scripts/benchmark.py`` (M0.9).

Fast, static checks (importable/parseable, ``--help`` exits 0) run on every
CI invocation. A real end-to-end run is marked ``slow`` since it needs the
synthetic dataset (~500 MB, ~8 s to generate) -- it reuses the session-scoped
``synthetic_project`` fixture via ``--config`` so it doesn't pay that cost
twice within a test session, and checks the produced JSON matches the
documented schema (``benchmarks/README.md`` / the script's own docstring).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import benchmark  # noqa: E402


def test_benchmark_script_exists():
    assert BENCHMARK_SCRIPT.is_file()


def test_benchmark_script_importable():
    """Already imported at module scope above; this asserts it exposes the
    documented extension point and CLI entry point."""
    assert callable(benchmark.main)
    assert isinstance(benchmark.STAGES, list)
    assert len(benchmark.STAGES) > 0
    names = [name for name, _fn in benchmark.STAGES]
    assert names[0] == "generate_synthetic_dataset"
    assert names[1] == "load_config"
    # Every openmeq.cli pipeline step should appear (data-driven registry).
    for step in ("qc", "detect", "chunk", "pick", "locate", "review", "magnitude", "relocate"):
        assert step in names


def test_benchmark_script_help_exits_0():
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--n-workers" in result.stdout
    assert "--steps" in result.stdout


def test_benchmark_script_rejects_unknown_step(tmp_path):
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_SCRIPT), "--steps", "not_a_real_step"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "unknown step" in result.stderr


def _copy_project(synthetic_project: tuple[Path, dict], tmp_path: Path) -> Path:
    """Copy the shared, session-scoped synthetic project into a private
    ``tmp_path`` before running the benchmark against it.

    ``benchmark.py``'s ``load_config``/pipeline-step stages call
    ``RunPaths(cfg).ensure()``, which creates ``cfg.project.output_dir``
    (``<project>/runs/synthetic/...`` for this fixture) -- a write under the
    project directory. Per ``tests/conftest.py``'s documented usage pattern
    for ``synthetic_project``, any test that does this must run against a
    copy, not the shared original (which every other test in the session
    also reads).
    """
    config_path, _truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    return project_copy / "config.yaml"


@pytest.mark.slow
def test_benchmark_script_run_produces_well_formed_report(synthetic_project, tmp_path):
    config_path = _copy_project(synthetic_project, tmp_path)
    output_dir = tmp_path / "results"

    exit_code = benchmark.main(
        [
            "--config",
            str(config_path),
            "--n-workers",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    json_files = sorted(output_dir.glob("*.json"))
    md_files = sorted(output_dir.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1

    report = json.loads(json_files[0].read_text())

    assert report["schema_version"] == 1
    assert report["generated_at"]
    assert report["n_workers_requested"] == 1
    assert report["n_workers_resolved"] == 1
    assert report["threads_per_worker"] == 1
    assert report["dataset"]["source"] == "config"

    stages = {s["name"]: s for s in report["stages"]}
    assert stages["generate_synthetic_dataset"]["status"] == "skipped"
    assert stages["load_config"]["status"] == "ran"
    assert stages["load_config"]["duration_s"] is not None
    assert stages["load_config"]["peak_rss_kb"] is not None

    # ``qc`` (M1.7) must really run and report a duration.
    assert stages["qc"]["status"] == "ran"
    assert stages["qc"]["duration_s"] is not None
    assert stages["qc"]["peak_rss_kb"] is not None

    # Other steps gain a real ``run()`` as the pipeline is built out; each must
    # be reported either "ran" (with a duration) or "skipped" ("not
    # implemented"), never dropped or "error", so the report stays well-formed.
    #
    # ``magnitude`` (M8.3 / OpenMEQ-625.3) is the one documented exception:
    # it is implemented and has its own hard dependency check (mirroring
    # ``detect``'s own, against qc's outputs) -- it correctly errors here
    # because its required input, ``locate/catalog_auto.xml``, doesn't
    # exist: `locate`'s own CLI wiring (M6.5) hasn't landed in this build
    # yet even though locate's internals (M6.1-M6.4) have (see
    # ``openmeq/locate/__init__.py``, still ``run``-less). Once M6.5 lands
    # and this benchmark run reaches "ran" for locate too, magnitude will
    # naturally follow suit with no test change needed here.
    for step in ("detect", "chunk", "pick", "locate", "magnitude", "relocate"):
        if step == "magnitude" and stages[step]["status"] == "error":
            assert "catalog" in stages[step]["note"].lower()
            continue
        assert stages[step]["status"] in ("ran", "skipped")
        if stages[step]["status"] == "ran":
            assert stages[step]["duration_s"] is not None
        else:
            assert stages[step]["duration_s"] is None
            assert "not implemented" in stages[step]["note"]

    # exit_code is 1 exactly when some stage errored (scripts/benchmark.py's
    # own `main`) -- magnitude's expected error above (until M6.5 lands) is
    # the only one allowed to cause that; every other stage must be clean.
    non_magnitude_errors = [
        s for s in report["stages"] if s["status"] == "error" and s["name"] != "magnitude"
    ]
    assert not non_magnitude_errors, non_magnitude_errors
    assert exit_code == (1 if stages["magnitude"]["status"] == "error" else 0)

    # `review` actually ran (this dataset never sets `review.enabled: true`,
    # so `openmeq.review.run()` took its "nothing to do" early-return path
    # rather than launching ObsPyck -- either way, the stage genuinely
    # executed, so it is reported "ran", not "skipped").
    assert stages["review"]["status"] == "ran"
    assert stages["review"]["duration_s"] is not None
    assert "completed" in stages["review"]["note"]

    for stage in report["stages"]:
        assert set(stage) == {
            "name",
            "status",
            "duration_s",
            "peak_rss_kb",
            "peak_rss_kb_delta",
            "peak_rss_bytes_sampled",
            "note",
        }
        assert stage["status"] in {"ran", "skipped", "error"}


@pytest.mark.slow
def test_benchmark_script_steps_filter(synthetic_project, tmp_path):
    config_path = _copy_project(synthetic_project, tmp_path)
    output_dir = tmp_path / "results"

    exit_code = benchmark.main(
        [
            "--config",
            str(config_path),
            "--n-workers",
            "1",
            "--steps",
            "qc,detect",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0

    report = json.loads(next(output_dir.glob("*.json")).read_text())
    names = {s["name"] for s in report["stages"]}
    assert names == {"generate_synthetic_dataset", "load_config", "qc", "detect"}
