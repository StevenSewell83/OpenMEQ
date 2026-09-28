"""Tests for ``openmeq run`` / ``openmeq.cli.run_all`` (M11.1 / OpenMEQ-m1f.1).

Two tiers:

* **Fast** (this module's majority): ``run_all``'s own orchestration logic
  -- step order, ``--force`` forwarding, ``--from``/``--to`` sub-ranges,
  ``--skip-review``, the ``review`` step's branching rule, and
  failure-stops-the-chain reporting -- exercised against a config built like
  ``tests/test_cli.py``'s own ``config/example.yaml``-based fixture, with
  every step module's real ``run`` monkeypatched out. None of this needs the
  synthetic dataset or any external binary.
* **Slow** (``@pytest.mark.slow``, real ``qc`` -> ``relocate``): one
  module-scoped fixture (``run_all_twice``) drives the real chain twice
  against a private copy of the shared ``synthetic_project`` fixture
  (review disabled) via ``CliRunner`` -- the expensive part -- and every
  slow test below only makes assertions against its already-computed
  result, per this issue's own "share output across assertions" note. A
  third slow test reuses the same populated project for a ``--from``/
  ``--to`` sub-range check.

The end-to-end *correctness* assertions against ``truth`` (recovery of
known events/locations/magnitudes) are M11.2's scope; this module only
proves the chain runs, skips correctly and reports event counts at each
stage.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import pytest
from click.testing import CliRunner
from obspy import Catalog, UTCDateTime
from obspy.core.event import Origin

from openmeq.catalog import accepted, make_event, read_catalog, write_catalog
from openmeq.cli import (
    STEP_ORDER,
    RunAllResult,
    StepFailed,
    _run_review_step,
    main,
    run_all,
)
from openmeq.config import load_config
from openmeq.paths import RunPaths

# tests/ has no __init__.py (matches the rest of the suite, e.g.
# tests/test_relocate_command.py's own `from test_relocate_hypodd_runner
# import ...`) -- reuse test_cli.py's config/example.yaml-patching helper
# rather than duplicating it.
sys.path.insert(0, str(Path(__file__).parent))
from test_cli import _patched_example_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

_STEP_MODULE_NAMES: dict[str, str] = {
    "qc": "openmeq.qc",
    "detect": "openmeq.detect",
    "chunk": "openmeq.chunk",
    "pick": "openmeq.pick",
    "locate": "openmeq.locate",
    "review": "openmeq.review",
    "magnitude": "openmeq.magnitude",
    "relocate": "openmeq.relocate",
}


def _install_fake_steps(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict]],
    *,
    fail_at: str | None = None,
    fail_exc: BaseException | None = None,
) -> None:
    """Replace every step module's real ``run`` with a recorder (and, for
    ``fail_at``, a function that raises), so ``run_all``'s own orchestration
    can be exercised in milliseconds against a config whose data paths
    (``config/example.yaml``'s) don't exist on disk.
    """
    import importlib

    for step, module_name in _STEP_MODULE_NAMES.items():
        module = importlib.import_module(module_name)

        def _make_fake(step_name: str):
            def _fake(cfg, paths, **opts):  # noqa: ANN001, ARG001
                calls.append((step_name, dict(opts)))
                if step_name == fail_at:
                    raise fail_exc if fail_exc is not None else RuntimeError(f"boom in {step_name}")
                return None

            return _fake

        monkeypatch.setattr(module, "run", _make_fake(step), raising=True)


def _minimal_catalog(depth_m: float) -> Catalog:
    """One event with an ``Origin`` at a given depth -- enough for
    ``openmeq review --compile`` (which needs ``locate/catalog_auto.xml``)
    and for magnitude to run against (zero arrivals -> zero station
    magnitudes, but a valid, non-crashing input -- the same pattern
    ``tests/test_cli.py::test_magnitude_catalog_option_runs`` already
    relies on for an empty catalogue).
    """
    event = make_event("ev20260101T000000_000")
    origin = Origin(
        time=UTCDateTime("2026-01-01T00:00:00"), longitude=176.0, latitude=-38.0, depth=depth_m
    )
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id
    return Catalog(events=[event])


# ---------------------------------------------------------------------------
# Fast: order, --force forwarding, --from/--to, --skip-review
# ---------------------------------------------------------------------------


def test_run_all_calls_every_step_in_order_and_forwards_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    result = run_all(cfg, paths, config_path=str(config_path), force=True)

    assert isinstance(result, RunAllResult)
    assert result.ok
    assert [name for name, _opts in calls] == list(STEP_ORDER)
    assert [o.step for o in result.outcomes] == list(STEP_ORDER)
    assert all(o.status in ("ok", "skipped") for o in result.outcomes)
    assert all(o.elapsed_s >= 0 for o in result.outcomes)

    for _name, opts in calls:
        assert opts["force"] is True

    magnitude_opts = dict(next(opts for name, opts in calls if name == "magnitude"))
    assert magnitude_opts["catalog"] is None, (
        "magnitude must always be called with catalog=None -- its own "
        "select_catalog_path is the one place that picks catalog_final.xml "
        "vs catalog_auto.xml (ARCHITECTURE.md, not duplicated in run_all)"
    )

    review_opts = dict(next(opts for name, opts in calls if name == "review"))
    assert review_opts["compile"] is False, (
        "cfg.review.enabled is false in the patched example config, so the "
        "review step must take the safe no-op branch (compile=False), never "
        "the interactive batch loop"
    )


def test_run_all_from_to_subrange(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    result = run_all(cfg, paths, config_path=str(config_path), from_step="chunk", to_step="locate")

    assert [name for name, _opts in calls] == ["chunk", "pick", "locate"]
    assert [o.step for o in result.outcomes] == ["chunk", "pick", "locate"]


def test_run_all_from_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    run_all(cfg, paths, config_path=str(config_path), from_step="magnitude")

    assert [name for name, _opts in calls] == ["magnitude", "relocate"]


def test_run_all_to_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    run_all(cfg, paths, config_path=str(config_path), to_step="chunk")

    assert [name for name, _opts in calls] == ["qc", "detect", "chunk"]


def test_run_all_invalid_range_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    with pytest.raises(click.ClickException):
        run_all(cfg, paths, config_path=str(config_path), from_step="locate", to_step="qc")
    assert calls == []


def test_run_all_skip_review_flag_never_touches_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    cfg.review.enabled = True
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls)

    result = run_all(cfg, paths, config_path=str(config_path), skip_review=True)

    assert "review" not in [name for name, _opts in calls]
    review_outcome = next(o for o in result.outcomes if o.step == "review")
    assert review_outcome.status == "skipped"
    assert "--skip-review" in review_outcome.detail


# ---------------------------------------------------------------------------
# Fast: failure reporting
# ---------------------------------------------------------------------------


def test_run_all_stops_on_failure_and_reports_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(
        monkeypatch, calls, fail_at="chunk", fail_exc=RuntimeError("synthetic failure")
    )

    with pytest.raises(StepFailed) as excinfo:
        run_all(cfg, paths, config_path=str(config_path))

    assert excinfo.value.step == "chunk"
    # Steps after the failing one are never attempted.
    assert [name for name, _opts in calls] == ["qc", "detect", "chunk"]
    message = str(excinfo.value)
    assert "chunk" in message
    assert "synthetic failure" in message
    assert f"openmeq run -c {config_path} --from chunk" in message


def test_run_cli_reports_failure_clearly_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)

    calls: list[tuple[str, dict]] = []
    _install_fake_steps(monkeypatch, calls, fail_at="detect", fail_exc=RuntimeError("boom"))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "-c", str(config_path)])

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "step 'detect' failed" in result.output
    assert "boom" in result.output
    assert "--from detect" in result.output
    # The chain stopped before chunk -- only qc/detect were called.
    assert [name for name, _opts in calls] == ["qc", "detect"]


# ---------------------------------------------------------------------------
# Fast: the review step's branching rule (ARCHITECTURE.md §1's requires_gui
# marker -- `openmeq run` must never launch the interactive batch loop).
# "review-enabled path, with the batch loop mocked": every review-enabled
# test below monkeypatches `openmeq.review.batch.run_review` (the actual
# ObsPyck-launching batch loop) to raise if called at all, proving run_all
# never reaches it in any cfg.review.enabled=True branch.
# ---------------------------------------------------------------------------


def _forbid_batch_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "openmeq run must never launch the interactive review batch loop "
            "(ARCHITECTURE.md §1's requires_gui marker)"
        )

    monkeypatch.setattr("openmeq.review.batch.run_review", _boom)


def test_review_enabled_pending_uses_catalog_auto_and_logs_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """cfg.review.enabled=True, nothing reviewed yet: non-interactive,
    catalog_final.xml is never created, and a clear message names the
    separate `openmeq review` command to run."""
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    cfg.review.enabled = True
    paths = RunPaths(cfg)
    paths.ensure()

    _forbid_batch_loop(monkeypatch)

    status, detail = _run_review_step(
        cfg, paths, config_path=str(config_path), force=False, skip_review=False
    )

    assert status == "skipped"
    assert "catalog_auto.xml" in detail
    assert not paths.catalog_final_xml.exists()

    captured = capsys.readouterr()
    assert f"openmeq review -c {config_path}" in captured.out
    assert "catalog_auto.xml" in captured.out


def test_review_enabled_catalog_final_present_compiles_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cfg.review.enabled=True and review/catalog_final.xml already present
    (a prior manual `openmeq review` + `--compile` session): run_all
    (re)compiles it -- safe, never interactive -- and it stays present for
    magnitude to pick up automatically."""
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    cfg.review.enabled = True
    paths = RunPaths(cfg)
    paths.ensure()

    write_catalog(_minimal_catalog(4000.0), paths.catalog_auto_xml)
    write_catalog(_minimal_catalog(4000.0), paths.catalog_final_xml)

    _forbid_batch_loop(monkeypatch)

    status, detail = _run_review_step(
        cfg, paths, config_path=str(config_path), force=False, skip_review=False
    )

    assert status == "ok"
    assert "catalog_final.xml" in detail
    assert paths.catalog_final_xml.is_file()


def test_review_enabled_reviewed_events_without_compile_also_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cfg.review.enabled=True, review/events/*.xml exist (a prior session
    reviewed some events) but `--compile` was never run: run_all compiles
    them into review/catalog_final.xml rather than leaving them unused."""
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    cfg.review.enabled = True
    paths = RunPaths(cfg)
    paths.ensure()

    write_catalog(_minimal_catalog(4000.0), paths.catalog_auto_xml)
    assert not paths.catalog_final_xml.exists()
    write_catalog(_minimal_catalog(4200.0), paths.review_event_file("ev20260101T000000_000"))

    _forbid_batch_loop(monkeypatch)

    status, _detail = _run_review_step(
        cfg, paths, config_path=str(config_path), force=False, skip_review=False
    )

    assert status == "ok"
    assert paths.catalog_final_xml.is_file()


def test_review_disabled_is_a_safe_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, _output_dir = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    assert cfg.review.enabled is False
    paths = RunPaths(cfg)
    paths.ensure()

    _forbid_batch_loop(monkeypatch)

    status, _detail = _run_review_step(
        cfg, paths, config_path=str(config_path), force=False, skip_review=False
    )

    assert status == "skipped"
    assert not paths.catalog_final_xml.exists()


def test_run_all_review_enabled_catalog_final_used_by_magnitude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration acceptance check: with review.enabled true and a
    genuinely reviewed event on disk (review/events/<id>.xml, as a prior,
    separate `openmeq review` session would leave), `openmeq run`'s
    magnitude step ends up using review/catalog_final.xml -- not
    locate/catalog_auto.xml -- via the real `openmeq review` (--compile)
    and `openmeq magnitude` step modules, driven through run_all's
    --from/--to sub-range (qc..locate faked out; only review and magnitude
    run for real).

    Note: `openmeq review --compile` always *rebuilds* catalog_final.xml
    from locate/catalog_auto.xml + review/events/*.xml + review/state.json
    (openmeq/review/compile.py's own module docstring) -- it is never
    itself a durable, hand-editable file, so this test places a reviewed
    event file rather than pre-writing catalog_final.xml directly (which a
    real compile would simply overwrite from its own inputs anyway).
    """
    config_path, _output_dir = _patched_example_config(tmp_path)
    text = config_path.read_text().replace("enabled: false", "enabled: true", 1)
    config_path.write_text(text)

    cfg = load_config(config_path)
    assert cfg.review.enabled is True
    cfg.parallel.n_workers = 1  # avoid spawning a real process pool for one event
    paths = RunPaths(cfg)
    paths.ensure()

    # catalog_auto.xml (locate's output) and the reviewed event (a prior,
    # separate manual `openmeq review` session's output) disagree on depth --
    # proves which one ends up in catalog_final.xml, and that magnitude reads it.
    event_id = "ev20260101T000000_000"
    write_catalog(_minimal_catalog(4000.0), paths.catalog_auto_xml)
    write_catalog(_minimal_catalog(9999.0), paths.review_event_file(event_id))

    _forbid_batch_loop(monkeypatch)

    calls: list[tuple[str, dict]] = []
    import importlib

    for step in ("qc", "detect", "chunk", "pick", "locate", "relocate"):
        module = importlib.import_module(_STEP_MODULE_NAMES[step])
        monkeypatch.setattr(
            module,
            "run",
            lambda cfg, paths, step_name=step, **opts: calls.append((step_name, opts)),
        )

    run_all(cfg, paths, config_path=str(config_path), from_step="review", to_step="magnitude")

    assert paths.catalog_mag_xml.is_file()
    mag_cat = read_catalog(paths.catalog_mag_xml)
    assert len(mag_cat) == 1
    assert mag_cat[0].preferred_origin().depth == pytest.approx(9999.0)


# ---------------------------------------------------------------------------
# Slow: real end-to-end chain against the synthetic dataset (review disabled)
# ---------------------------------------------------------------------------

EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
_EXTERNAL_MISSING = not (
    (EXTERNAL_DIR / "bin" / "NLLoc").exists()
    and (EXTERNAL_DIR / "bin" / "hypoDD").exists()
    and (EXTERNAL_DIR / "bin" / "ph2dt").exists()
    and (EXTERNAL_DIR / "PhaseNet").is_dir()
)


@pytest.fixture(scope="module", autouse=True)
def _set_external_dir():
    """Module-scoped equivalent of ``monkeypatch.setenv`` -- ``monkeypatch``
    is function-scoped only, and ``run_all_twice`` (below) is module-scoped
    (mirrors ``tests/test_relocate_command.py``'s identical fixture)."""
    old = os.environ.get("OPENMEQ_EXTERNAL_DIR")
    os.environ["OPENMEQ_EXTERNAL_DIR"] = str(EXTERNAL_DIR)
    yield
    if old is None:
        os.environ.pop("OPENMEQ_EXTERNAL_DIR", None)
    else:
        os.environ["OPENMEQ_EXTERNAL_DIR"] = old


@pytest.fixture(scope="module")
def run_all_project(synthetic_project, tmp_path_factory) -> tuple[Path, dict]:
    """A private copy of the shared ``synthetic_project`` fixture -- `openmeq
    run` writes into ``cfg.project.output_dir`` under it, so it must not be
    the shared session directory (``tests/conftest.py``'s own usage note)."""
    config_path, truth = synthetic_project
    project_dir = tmp_path_factory.mktemp("run_all_e2e") / "project"
    shutil.copytree(config_path.parent, project_dir)
    return project_dir / "config.yaml", truth


#: Shared marks for every real-chain (slow) test below.
_slow = pytest.mark.slow
_requires_nlloc = pytest.mark.requires_nlloc
_requires_hypodd = pytest.mark.requires_hypodd
_requires_phasenet = pytest.mark.requires_phasenet
_skip_no_external = pytest.mark.skipif(
    _EXTERNAL_MISSING, reason="external/{bin/NLLoc,bin/hypoDD,bin/ph2dt,PhaseNet} not available"
)


@pytest.fixture(scope="module")
def run_all_twice(run_all_project: tuple[Path, dict]) -> SimpleNamespace:
    """Runs ``openmeq run -c <config>`` twice (real qc -> relocate, review
    disabled) against the same project directory -- the module's one
    expensive real-binary/real-PhaseNet invocation (the second is a fast
    no-op) -- and hands back everything every slow test below needs, so the
    chain only actually runs once per test session (this issue's own "share
    its output across assertions" note).
    """
    config_path, truth = run_all_project
    cfg = load_config(config_path)
    paths = RunPaths(cfg)

    runner = CliRunner()

    t0 = time.perf_counter()
    first_result = runner.invoke(main, ["run", "-c", str(config_path)])
    first_elapsed = time.perf_counter() - t0
    # Fail fast here (rather than in every dependent test) if the real chain
    # itself is broken -- every other slow test in this module assumes it
    # succeeded.
    assert first_result.exit_code == 0, first_result.output

    watched = [
        paths.index_json,
        paths.channel_flags_json,
        paths.detections_csv,
        paths.chunk_completeness_csv,
        paths.catalog_picks_xml,
        paths.catalog_auto_xml,
        paths.catalog_mag_xml,
        paths.catalog_dd_xml,
    ]
    # catalog_dd.xml is deliberately excluded from the "mtime must be
    # unchanged" set: openmeq.relocate.hypodd_runner.run_hypodd's own
    # docstring documents that it always re-derives/rewrites
    # relocate/catalog_dd.xml from hypoDD.reloc even when it skips
    # re-invoking the hypoDD binary itself ("Skip re-invoking hypoDD (but
    # still re-derive catalog_dd.xml)") -- a deliberate M10.4 design choice,
    # not a manifest gap, so its mtime changing on an otherwise-unchanged
    # second `openmeq run` is expected.
    mtime_stable = [p for p in watched if p != paths.catalog_dd_xml]
    mtimes_before = {p: p.stat().st_mtime for p in watched}

    t0 = time.perf_counter()
    second_result = runner.invoke(main, ["run", "-c", str(config_path)])
    second_elapsed = time.perf_counter() - t0

    mtimes_after = {p: p.stat().st_mtime for p in watched}

    print(
        f"\n[run_all e2e] first run: {first_elapsed:.1f}s; "
        f"second (skip-unchanged) run: {second_elapsed:.1f}s"
    )

    return SimpleNamespace(
        config_path=config_path,
        truth=truth,
        cfg=cfg,
        paths=paths,
        first_result=first_result,
        first_elapsed=first_elapsed,
        second_result=second_result,
        second_elapsed=second_elapsed,
        watched=watched,
        mtime_stable=mtime_stable,
        mtimes_before=mtimes_before,
        mtimes_after=mtimes_after,
    )


@_slow
@_requires_nlloc
@_requires_hypodd
@_requires_phasenet
@_skip_no_external
def test_run_all_end_to_end_qc_to_relocate(run_all_twice: SimpleNamespace) -> None:
    """`openmeq run` (review disabled) runs qc -> relocate end to end on the
    synthetic fixture and produces relocate/catalog_dd.xml; the final
    summary table lists every step."""
    r = run_all_twice
    assert r.first_result.exit_code == 0, r.first_result.output
    assert "Traceback" not in r.first_result.output
    assert r.paths.catalog_dd_xml.is_file()

    for step in STEP_ORDER:
        assert step in r.first_result.output, f"{step!r} missing from the summary table"
    assert "status" in r.first_result.output
    assert "wall time" in r.first_result.output

    # Event counts at each stage (informational -- correctness-vs-truth is
    # M11.2's scope, not this task's).
    detections_df = pd.read_csv(r.paths.detections_csv)
    chunk_df = pd.read_csv(r.paths.chunk_completeness_csv)
    picks_cat = read_catalog(r.paths.catalog_picks_xml)
    locate_cat = read_catalog(r.paths.catalog_auto_xml)
    mag_cat = read_catalog(r.paths.catalog_mag_xml)
    dd_cat = read_catalog(r.paths.catalog_dd_xml)
    print(
        "\n[run_all e2e] event counts: "
        f"truth={len(r.truth['events'])} "
        f"detections={len(detections_df)} "
        f"chunked_events={chunk_df['event_id'].nunique()} "
        f"picked={len(picks_cat)} "
        f"located_accepted={len(accepted(locate_cat))}/{len(locate_cat)} "
        f"magnitude={len(mag_cat)} "
        f"relocated={len(dd_cat)} "
        f"(first_run={r.first_elapsed:.1f}s)"
    )
    assert len(dd_cat) > 0


@_slow
@_requires_nlloc
@_requires_hypodd
@_requires_phasenet
@_skip_no_external
def test_run_all_second_invocation_is_fast_noop(run_all_twice: SimpleNamespace) -> None:
    """A second `openmeq run` with no input changes: every step reports
    skipped/unchanged -- verified via log output (every step's own
    "unchanged"/"up to date" message), output-file mtimes being unchanged
    (except relocate/catalog_dd.xml, deliberately excluded -- see
    `run_all_twice`'s own comment), and the second run being much faster
    than the first."""
    r = run_all_twice
    assert r.second_result.exit_code == 0, r.second_result.output

    # Verify via each step's own logs/<step>.log (openmeq.log's console
    # handler binds to the process's original sys.stderr at import time, not
    # click.testing.CliRunner's per-invoke buffer, so these messages don't
    # reliably land in `result.output` -- the on-disk per-step log file,
    # which every step's own manifest-skip branch always logs to, is the
    # robust place to check this).
    expected_phrases = {
        "qc": "qc: up to date, skipping",
        "detect": "detect: unchanged since the last run",
        "chunk": "chunk: unchanged since the last run",
        "pick": "pick: unchanged since the last run",
        "locate": "locate: unchanged since the last run",
        "magnitude": "magnitude: unchanged since the last run",
        "relocate": "hypoDD run unchanged since the last run",
    }
    for step, phrase in expected_phrases.items():
        log_text = r.paths.log_file(step).read_text()
        assert phrase in log_text, f"expected {phrase!r} in logs/{step}.log:\n{log_text}"

    assert {p: r.mtimes_before[p] for p in r.mtime_stable} == {
        p: r.mtimes_after[p] for p in r.mtime_stable
    }, (
        "the second `openmeq run` rewrote at least one output file that "
        "should have been skipped as unchanged"
    )
    assert r.second_elapsed < max(r.first_elapsed / 3, 5.0), (
        f"second run ({r.second_elapsed:.1f}s) should be much faster than the first "
        f"({r.first_elapsed:.1f}s) -- every step should skip its real work"
    )


@_slow
@_requires_nlloc
@_requires_hypodd
@_requires_phasenet
@_skip_no_external
def test_run_all_from_to_subrange_restricts_real_execution(run_all_twice: SimpleNamespace) -> None:
    """``--from qc --to detect --force`` on the already-fully-run project:
    only qc/detect's own outputs are rewritten; every downstream output
    (chunk, pick, locate, magnitude, relocate) is left untouched."""
    r = run_all_twice
    paths = r.paths
    mtimes_before = {p: p.stat().st_mtime for p in r.watched}

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "-c", str(r.config_path), "--from", "qc", "--to", "detect", "--force"]
    )
    assert result.exit_code == 0, result.output

    mtimes_after = {p: p.stat().st_mtime for p in r.watched}
    assert mtimes_after[paths.index_json] != mtimes_before[paths.index_json]
    assert mtimes_after[paths.channel_flags_json] != mtimes_before[paths.channel_flags_json]
    assert mtimes_after[paths.detections_csv] != mtimes_before[paths.detections_csv]
    for p in (
        paths.chunk_completeness_csv,
        paths.catalog_picks_xml,
        paths.catalog_auto_xml,
        paths.catalog_mag_xml,
        paths.catalog_dd_xml,
    ):
        assert mtimes_after[p] == mtimes_before[p], (
            f"{p} should not have been touched by `openmeq run --from qc --to detect`"
        )
