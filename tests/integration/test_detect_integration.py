"""Integration tests for ``openmeq detect`` (M2.3, ``OpenMEQ-ujx.3``).

Covers the CLI wiring (``openmeq/cli.py``'s ``detect`` command),
``openmeq.detect.pipeline.run_detect``'s hard "run ``openmeq qc`` first"
dependency check, manifest skip/force behaviour, the diagnostics files
(``openmeq.detect.diagnostics``), and -- the epic's actual go/no-go gate,
per this task's own design note -- the recovery test against the synthetic
dataset's ~20 known truth events (``docs/ARCHITECTURE.md`` §8).

``openmeq qc`` (M1.7) landed on ``main`` while this task was in flight (it
was concurrent when this task started -- see the original orchestrator
note preserved on the helpers below). The flagship recovery/diagnostics
tests (``_detect_result``) now run the real ``openmeq qc`` then ``openmeq
detect`` CLI commands in sequence, matching this task's own acceptance
criterion verbatim ("after running openmeq qc then openmeq detect on the
synthetic_project config"). ``test_skip_unchanged_and_force`` still builds
``inventory/index.json``/``qc/channel_flags.json`` directly via
``openmeq.index.build_index`` + ``openmeq.qc.flags`` functions (reusing the
shared, session-scoped ``synthetic_qc_tables`` fixture's already-computed
QC ``DataFrame``\\ s) rather than running the real, ~30s ``openmeq qc``
pass again -- that test is only exercising ``StepManifest``'s skip/force
decision (already isolated from both qc's and detect's own correctness),
so a real qc run buys it nothing.

OpenMEQ-m1f.6: ``_detect_result``'s real ``openmeq qc`` + ``openmeq detect``
pair (the latter with ``n_workers=2``, deliberately, to exercise real
``ProcessPoolExecutor`` dispatch across the 2 day-windows) is its own
isolated run, not migrated onto ``tests/conftest.py``'s shared
``full_pipeline_run`` fixture -- it is testing detection itself (including
real parallel dispatch), not reading an already-computed upstream result.
Its two consumers below are marked ``slow`` (~50s on this machine) so they
don't count against the ``-m "not slow"`` budget.

Disk-space note (worker instructions "Disk space" section): no test here
``shutil.copytree``\\ s the ~500 MB synthetic waveform data. Each test points
a fresh ``Config`` (loaded from the shared, session-scoped
``synthetic_project`` fixture's ``config.yaml``) at a private ``tmp_path``
output directory (``cfg.project.output_dir = ...``, the same pattern
``tests/test_chunk_writer.py``/``tests/test_stations.py``/etc. already use)
and re-dumps it, with every path already resolved to absolute by
``load_config``, to a standalone YAML file under that same private
directory -- ``openmeq.parallel.pmap``'s worker processes reload the config
from disk by path (see ``openmeq.detect.worker.detect_window``'s own
docstring), so a real file is needed, but ``waveform_dirs`` itself is only
ever *read*, straight from the shared session fixture's directory.

Recovery threshold (this task's own go/no-go gate, chosen deliberately per
the beads issue's "Pitfalls" note)
------------------------------------------------------------------------------
- **Recall:** at least 90% of the ~20 truth events must be matched by a
  detection within 5 seconds of the true origin time, one-to-one (a
  detection is claimed by at most one truth event -- see
  ``_match_truth_events``).
- **False positives:** the total detection count must not exceed 1.5x the
  true event count. In practice, against this dataset, with
  ``cfg.detection.dedup_window_s`` at its current default (10 s, raised
  from M2.2's original 2 s specifically to merge STA/LTA's separate P/S
  triggers into one detection per event -- see this task's own notes),
  detection recovers **all 20/20 truth events with zero duplicates and zero
  false positives** (verified directly while writing this test); 1.5x is
  intentionally looser than that observed exact match, to avoid a flaky
  test over a legitimate few-percent difference in STA/LTA's trigger
  timing/count across a machine/BLAS-library difference, while still
  catching a real regression (e.g. dedup_window_s accidentally reverted
  towards M2.2's original 2 s, which produced 39 raw triggers for 20 events).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from obspy import UTCDateTime

from openmeq.cli import main
from openmeq.config import Config, load_config
from openmeq.detect.pipeline import run_detect
from openmeq.paths import RunPaths
from openmeq.qc.flags import compute_channel_flags, write_channel_flags

_MATCH_TOLERANCE_S = 5.0
_FALSE_POSITIVE_FACTOR = 1.5


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    ``cfg.model_dump(mode="json")`` renders every ``Path``/``date`` field as
    a plain absolute string/ISO date (``load_config``'s own
    ``_resolve_relative_paths`` already ran when ``cfg`` was first loaded),
    so re-parsing this file never depends on *where* it happens to live --
    unlike the original config file, it need not sit next to
    ``./waveforms``/etc. See the module docstring's "Disk-space note".
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


def _build_qc_outputs_from_tables(
    cfg: Config,
    paths: RunPaths,
    idx,
    availability_df: pd.DataFrame,
    daily_stats_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> None:
    """Write ``inventory/index.json``/``qc/channel_flags.json`` for ``paths``.

    Never via a real ``openmeq qc`` run (M1.7 is concurrent, see module
    docstring) -- ``idx``/``availability_df``/``daily_stats_df``/
    ``monthly_df`` come from the shared ``synthetic_qc_tables`` fixture
    (data-only, not tied to any particular ``output_dir``, so reusing them
    here saves recomputing ~20-40s of QC stats per test).
    """
    idx.write(paths.index_json)
    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    write_channel_flags(flags, paths.channel_flags_json)


@pytest.fixture()
def detect_project(synthetic_project, synthetic_qc_tables, tmp_path):
    """``(cfg, paths, config_path, truth)`` for a private ``tmp_path`` run
    directory, with ``openmeq qc``'s outputs already built (see module
    docstring) -- everything needed for ``openmeq detect`` to run, nothing
    more.
    """
    config_path, truth = synthetic_project
    _qc_cfg, idx, availability_df, daily_stats_df, monthly_df, _truth2 = synthetic_qc_tables

    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    _build_qc_outputs_from_tables(cfg, paths, idx, availability_df, daily_stats_df, monthly_df)

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    return cfg, paths, new_config_path, truth


def _match_truth_events(
    true_events: list[dict], detection_times: list[UTCDateTime], tolerance_s: float
) -> tuple[int, int]:
    """One-to-one nearest-detection matching (this task's own design note:
    "match each truth event to its nearest detection ... report duplicates
    and misses" -- never double-counting one detection against two truth
    events).

    Returns
    -------
    (n_matched, n_unclaimed_detections)
        ``n_matched``: truth events with a distinct detection within
        ``tolerance_s``. ``n_unclaimed_detections``: detections left over
        after every truth event has claimed at most one (duplicates and/or
        false positives).
    """
    unclaimed = list(detection_times)
    n_matched = 0
    for event in sorted(true_events, key=lambda e: e["origin"]["time"]):
        if not unclaimed:
            break
        origin_time = event["origin"]["time"]
        nearest = min(unclaimed, key=lambda t: abs(t - origin_time))
        if abs(nearest - origin_time) <= tolerance_s:
            n_matched += 1
            unclaimed.remove(nearest)
    return n_matched, len(unclaimed)


# ---------------------------------------------------------------------------
# test_requires_qc_first
# ---------------------------------------------------------------------------


def test_requires_qc_first(synthetic_project, tmp_path):
    """A fresh output dir with no prior ``openmeq qc`` run fails clearly."""
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    result = runner.invoke(main, ["detect", "-c", str(new_config_path)])

    assert result.exit_code != 0
    assert "openmeq qc" in result.output


# ---------------------------------------------------------------------------
# test_detect_end_to_end_recovery / test_diagnostics_files_exist
# ---------------------------------------------------------------------------
#
# Both tests check a single real ``openmeq detect`` run (expensive: a real
# STA/LTA pass over the whole synthetic archive), so they share one
# module-scoped run via ``_detect_result`` rather than each triggering their
# own -- they are read-only checks against the same completed run.


@pytest.fixture(scope="module")
def _detect_result(synthetic_project, tmp_path_factory):
    """Run the real ``openmeq qc`` then ``openmeq detect`` CLI commands,
    once for the whole module (both are real, non-trivial passes over the
    synthetic archive) -- this task's own acceptance criterion, verbatim:
    "after running openmeq qc then openmeq detect on the synthetic_project
    config".
    """
    config_path, truth = synthetic_project

    run_dir = tmp_path_factory.mktemp("detect_integration_run")
    cfg = load_config(config_path)
    cfg.project.output_dir = run_dir / "run"
    # Exercise real ProcessPoolExecutor dispatch across the 2 day-windows
    # (this task's own "Performance" design note: "verify ... both windows
    # are actually dispatched to the pool"), rather than the synthetic
    # project's own default n_workers=1.
    cfg.parallel.n_workers = 2

    new_config_path = run_dir / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    qc_result = runner.invoke(main, ["qc", "-c", str(new_config_path)])
    assert qc_result.exit_code == 0, qc_result.output

    detect_result = runner.invoke(main, ["detect", "-c", str(new_config_path)])
    assert detect_result.exit_code == 0, detect_result.output

    paths = RunPaths(cfg)
    return paths, truth


@pytest.mark.slow
def test_detect_end_to_end_recovery(_detect_result):
    paths, truth = _detect_result

    assert paths.detections_csv.is_file()
    df = pd.read_csv(paths.detections_csv)

    true_events = truth["events"]
    detection_times = [UTCDateTime(t) for t in df["time"]]

    n_matched, n_unclaimed = _match_truth_events(true_events, detection_times, _MATCH_TOLERANCE_S)
    n_true = len(true_events)
    n_detections = len(df)

    assert n_matched >= int(0.9 * n_true), (
        f"only matched {n_matched}/{n_true} truth events within {_MATCH_TOLERANCE_S}s "
        f"(one-to-one); {n_unclaimed} detection(s) unclaimed (duplicates/false positives) "
        f"out of {n_detections} total"
    )
    assert n_detections <= _FALSE_POSITIVE_FACTOR * n_true, (
        f"{n_detections} detections for {n_true} true events exceeds the "
        f"{_FALSE_POSITIVE_FACTOR}x false-positive sanity bound "
        f"({n_matched} matched, {n_unclaimed} unclaimed)"
    )


@pytest.mark.slow
def test_diagnostics_files_exist(_detect_result):
    paths, _truth = _detect_result
    diag_dir = paths.detect_diagnostics_dir

    for name in (
        "detections_per_hour.png",
        "coincidence_sum_hist.png",
        "station_participation.png",
        "summary.csv",
    ):
        path = diag_dir / name
        assert path.is_file(), f"{path} was not created"
        assert path.stat().st_size > 0, f"{path} is empty"


# ---------------------------------------------------------------------------
# test_skip_unchanged_and_force
# ---------------------------------------------------------------------------


def test_skip_unchanged_and_force(detect_project, monkeypatch):
    """Mirrors M1.7's own manifest tests (``StepManifest`` skip-unchanged /
    ``--force``). ``openmeq.detect.pipeline.pmap`` is monkeypatched to a
    cheap, call-counting fake standing in for the real (slow) per-window
    STA/LTA pass -- this test is about the manifest's own skip/force
    decision, already isolated from detection correctness (covered by
    ``test_detect_end_to_end_recovery`` above), so nothing here needs real
    waveform data.
    """
    cfg, paths, config_path, _truth = detect_project

    call_count = 0

    def _fake_pmap(func, items, pcfg, desc="", **kwargs):
        nonlocal call_count
        call_count += 1
        return [[] for _ in items]  # no raw triggers -> an empty (but valid) detections.csv

    monkeypatch.setattr("openmeq.detect.pipeline.pmap", _fake_pmap)

    run_detect(cfg, paths, config_path=str(config_path))
    assert call_count == 1
    assert paths.detections_csv.is_file()
    mtime1 = paths.detections_csv.stat().st_mtime_ns

    # Unchanged: pmap must not be called again, and the output file's mtime
    # must not move.
    run_detect(cfg, paths, config_path=str(config_path))
    assert call_count == 1
    assert paths.detections_csv.stat().st_mtime_ns == mtime1

    # --force reruns it regardless.
    run_detect(cfg, paths, config_path=str(config_path), force=True)
    assert call_count == 2
