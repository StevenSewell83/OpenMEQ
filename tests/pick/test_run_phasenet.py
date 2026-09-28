"""Tests for ``openmeq.pick.run_phasenet`` (M4.3, ``OpenMEQ-1m7.3``).

M3.1 (``openmeq.chunk.write_chunks``) and M4.2 (``openmeq.pick.prep``) have
both landed by this point in the build, so -- unlike ``tests/pick/
test_prep.py``, which had to hand-build raw chunks directly from the index
-- these tests build synthetic ``chunks_pn/`` input the "real" way: hand-made
detections at the synthetic dataset's true origin times (:func:`_make_detections_df`)
fed through :func:`openmeq.chunk.write_chunks` (cutting the raw
``chunks/<event_id>/<event_id>.mseed``), then :func:`openmeq.pick.prep.prepare_all`
(producing ``chunks_pn/`` + ``data_list.csv``), then finally
:func:`openmeq.pick.run_phasenet.run_phasenet` under test.

``synthetic_project`` is session-scoped and shared (~500 MB on disk, per its
own usage note); every test here builds its own private
:class:`~openmeq.config.Config` (``cfg.model_copy(deep=True)`` with
``project.output_dir`` repointed into that test's own ``tmp_path``) rather
than copying the whole project directory, matching ``tests/pick/
test_prep.py``'s established pattern.
"""

from __future__ import annotations

import logging
import subprocess

import pandas as pd
import pytest
from obspy import UTCDateTime

import openmeq.external as external_module
from openmeq.chunk import write_chunks
from openmeq.config import load_config
from openmeq.index import build_index
from openmeq.paths import RunPaths
from openmeq.pick.prep import prepare_all
from openmeq.pick.run_phasenet import PICKS_CSV_COLUMNS, PhaseNetRunError, run_phasenet
from openmeq.stations import station_table as compute_station_table

#: ARCHITECTURE.md §4.3's detections.csv columns -- write_chunks only
#: strictly needs "event_id"/"time" (see openmeq.chunk.group_detections_by_day's
#: docstring), but the rest are included so these hand-made rows look like a
#: real detections.csv, not a stripped-down test-only shape.
DETECTIONS_COLUMNS = [
    "event_id",
    "time",
    "duration_s",
    "coincidence_sum",
    "n_stations",
    "stations",
    "trace_ids",
]


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
# shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phasenet_world(synthetic_project):
    """``(cfg, index, truth, station_table)`` for the shared synthetic project.

    ``build_index`` only reads waveform files and idempotently (re)writes
    ``inventory/index.json`` under the *shared* session directory -- safe to
    call directly against ``synthetic_project`` (matches ``tests/pick/
    test_prep.py``'s ``prep_world`` fixture).
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    index = build_index(cfg)
    station_table = compute_station_table(cfg.stations)
    return cfg, index, truth, station_table


def _private_cfg(cfg, tmp_path, name: str = "run"):
    """A deep copy of ``cfg`` with ``project.output_dir`` repointed into
    this test's own ``tmp_path`` -- never write into the shared session
    fixture's output directory."""
    cfg2 = cfg.model_copy(deep=True)
    cfg2.project.output_dir = tmp_path / name
    return cfg2


def _make_detections_df(events: list[dict]) -> pd.DataFrame:
    """Hand-made ``detections.csv``-shaped rows at each event's true origin time."""
    rows = []
    for ev in events:
        origin_time: UTCDateTime = ev["origin"]["time"]
        rows.append(
            {
                "event_id": ev["event_id_hint"],
                "time": origin_time.isoformat() + "Z",
                "duration_s": 30.0,
                "coincidence_sum": 8.0,
                "n_stations": 8,
                "stations": "",
                "trace_ids": "",
            }
        )
    return pd.DataFrame(rows, columns=DETECTIONS_COLUMNS)


def _prepare_synthetic_chunks(cfg, index, paths, events: list[dict], station_table) -> list[str]:
    """Cut raw chunks (:func:`write_chunks`) then prepare ``chunks_pn/``
    (:func:`prepare_all`) for `events`. Returns the event ids prepared."""
    detections_df = _make_detections_df(events)
    write_chunks(detections_df, index, cfg, paths)
    event_ids = [ev["event_id_hint"] for ev in events]
    prepare_all(cfg, paths, event_ids, station_table=station_table)
    return event_ids


# ---------------------------------------------------------------------------
# test_batch_run_synthetic
# ---------------------------------------------------------------------------


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_batch_run_synthetic(phasenet_world, tmp_path, caplog):
    cfg, index, truth, station_table = phasenet_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_batch")
    paths2 = RunPaths(cfg2)

    events = truth["events"][:5]
    _prepare_synthetic_chunks(cfg2, index, paths2, events, station_table)

    with caplog.at_level(logging.INFO, logger="openmeq.pick"):
        picks_csv = run_phasenet(paths2, cfg2)

    assert picks_csv == paths2.phasenet_raw_dir / "picks.csv"
    assert picks_csv.is_file()

    picks = pd.read_csv(picks_csv)
    assert len(picks) > 0, "predict.py produced no picks at all"
    assert set(PICKS_CSV_COLUMNS).issubset(set(picks.columns))

    # --save_prob is always passed -- confirm the sidecar it writes too.
    assert (paths2.phasenet_raw_dir / "result.h5").is_file()

    # A logged runtime (module docstring / acceptance criterion).
    assert "predict.py finished" in caplog.text

    # station_id has no per-channel attribution (docs/phasenet_interface.md §5).
    for station_id in picks["station_id"]:
        assert not str(station_id).endswith((".Z", ".N", ".E", ".1", ".2"))

    # Correctness spot-check (not a full recall study -- see
    # docs/phasenet_interface.md §10 for that): at least one P and one S
    # pick should land within 0.1s of a true arrival somewhere in this subset.
    matched = {"P": False, "S": False}
    for ev in events:
        event_id = ev["event_id_hint"]
        event_picks = picks[picks["file_name"] == f"{event_id}/{event_id}.mseed"]
        for station_key, arrival in ev["arrivals"].items():
            prefix = f"{station_key}."
            candidates = event_picks[event_picks["station_id"].str.startswith(prefix)]
            for phase in ("P", "S"):
                truth_t = arrival[phase]
                phase_rows = candidates[candidates["phase_type"] == phase]
                for _, row in phase_rows.iterrows():
                    if abs(UTCDateTime(row["phase_time"]) - truth_t) <= 0.1:
                        matched[phase] = True

    assert matched["P"], "no P pick recovered within 0.1s of any truth arrival in this subset"
    assert matched["S"], "no S pick recovered within 0.1s of any truth arrival in this subset"


# ---------------------------------------------------------------------------
# test_cpu_forced
# ---------------------------------------------------------------------------


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_cpu_forced(phasenet_world, tmp_path, monkeypatch):
    """``phasenet_device: cpu`` (the default) must set
    ``CUDA_VISIBLE_DEVICES=""`` in the *actual subprocess environment*, and
    the run must still succeed.

    Spies on ``subprocess.run`` (as ``openmeq.external`` calls it) to capture
    the real ``env`` kwarg reaching it -- this checks what the subprocess
    itself receives, not just what ``openmeq.pick.run_phasenet`` intends to
    pass, while still letting the real call through so the run genuinely
    succeeds end-to-end.
    """
    cfg, index, truth, station_table = phasenet_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_cpu")
    assert cfg2.parallel.phasenet_device == "cpu"  # the config default
    paths2 = RunPaths(cfg2)

    events = truth["events"][:1]
    _prepare_synthetic_chunks(cfg2, index, paths2, events, station_table)

    captured: dict = {}
    real_subprocess_run = subprocess.run

    def _spy_subprocess_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return real_subprocess_run(cmd, **kwargs)

    monkeypatch.setattr(external_module.subprocess, "run", _spy_subprocess_run)

    picks_csv = run_phasenet(paths2, cfg2)

    assert picks_csv.is_file()
    assert captured["env"] is not None, (
        "run_phasenet did not pass an env override to subprocess.run"
    )
    assert captured["env"].get("CUDA_VISIBLE_DEVICES") == ""
    # The override is merged over the parent environment, not a replacement.
    assert "PATH" in captured["env"]


# ---------------------------------------------------------------------------
# test_failure_path
# ---------------------------------------------------------------------------


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_failure_path_unresolvable_fname(phasenet_world, tmp_path):
    """A ``data_list.csv`` naming a chunk file that doesn't exist under
    ``chunks_pn_dir`` must raise :class:`PhaseNetRunError` with an
    informative message -- not an uncaught traceback.

    Empirically (confirmed while writing this test): ``predict.py`` does
    *not* crash on an unresolvable ``fname`` -- ``read_mseed`` catches the
    ``FileNotFoundError`` per-file, logs "Error reading ...", and simply
    contributes zero picks for that file. With only one row in
    ``data_list.csv``, the whole run therefore finds 0 picks overall and
    exits 0 -- and ``predict.py`` only writes ``picks.csv`` at all when
    ``len(picks) > 0`` (confirmed by inspection of its own ``main()``), so
    no ``picks.csv`` is written. This exercises the *other* documented
    failure mode from the work-to-do list ("On non-zero exit **or a
    missing picks.csv**, raise PhaseNetRunError"), not a subprocess crash --
    see :func:`test_failure_path_corrupted_data_list_csv` below for the
    non-zero-exit branch.
    """
    cfg, _index, _truth, _station_table = phasenet_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_fail")
    paths2 = RunPaths(cfg2)

    paths2.chunks_pn_dir.mkdir(parents=True, exist_ok=True)
    data_list_csv = paths2.chunks_pn_dir / "data_list.csv"
    data_list_csv.write_text("fname\nmissing_event/missing_event.mseed\n")

    with pytest.raises(PhaseNetRunError) as excinfo:
        run_phasenet(paths2, cfg2)

    message = str(excinfo.value)
    assert "did not write" in message
    assert "picks.csv" in message
    assert not paths2.phasenet_raw_dir.joinpath("picks.csv").exists()


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_failure_path_corrupted_data_list_csv(phasenet_world, tmp_path):
    """A ``data_list.csv`` with no ``fname`` column at all crashes
    ``predict.py`` itself (a real ``KeyError`` traceback, confirmed by
    manual reproduction while writing this test) -- a non-zero exit, the
    other half of the "corrupted data_list.csv (or unresolvable fname)"
    acceptance criterion. Must still surface as :class:`PhaseNetRunError`
    with the captured stderr tail, not an uncaught ``ExternalToolError``."""
    cfg, _index, _truth, _station_table = phasenet_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_fail_corrupt")
    paths2 = RunPaths(cfg2)

    paths2.chunks_pn_dir.mkdir(parents=True, exist_ok=True)
    data_list_csv = paths2.chunks_pn_dir / "data_list.csv"
    data_list_csv.write_text("notfname\nfoo.mseed\n")

    with pytest.raises(PhaseNetRunError) as excinfo:
        run_phasenet(paths2, cfg2)

    message = str(excinfo.value)
    assert "predict.py failed" in message
    assert "fname" in message  # the real KeyError: 'fname' traceback line


def test_failure_path_missing_data_list_csv(phasenet_world, tmp_path):
    """No ``chunks_pn/data_list.csv`` at all (``prepare_all`` never ran) is
    also a :class:`PhaseNetRunError`, raised before any subprocess is
    invoked -- doesn't need PhaseNet installed, so no requires_phasenet skip."""
    cfg, _index, _truth, _station_table = phasenet_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_no_data_list")
    paths2 = RunPaths(cfg2)

    with pytest.raises(PhaseNetRunError) as excinfo:
        run_phasenet(paths2, cfg2)
    assert "data_list.csv" in str(excinfo.value)
