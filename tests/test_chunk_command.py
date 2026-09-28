"""Tests for ``openmeq chunk`` (M3.2, ``OpenMEQ-rq8.2``): chunk completeness
report, diagnostics, the ``openmeq chunk`` CLI command, and its manifest
skip/``--force`` behaviour.

Covers ARCHITECTURE.md §3's ``chunks/completeness.csv``, PLAN.md §5's Chunk
diagnostics row, and ``openmeq/chunk.py``'s ``run`` entry point
(``openmeq/cli.py``'s ``chunk`` subcommand). Mirrors
``tests/integration/test_detect_integration.py``'s own patterns (private
``tmp_path`` output dirs, a re-dumped standalone config file, real CLI
commands via ``CliRunner``) and reuses ``tests/test_chunk_writer.py``'s
hand-built-``detections.csv`` pattern (M2.x/M3.1 were concurrent with this
task, so a real ``openmeq detect`` run is only exercised by the one
end-to-end test that actually needs it -- the completeness/diagnostics
tests build their own ``detections.csv`` rows directly against known truth
event times and the synthetic dataset's own documented gap, matching
``tests/test_chunk_writer.py``'s own approach).

Disk-space note (worker instructions "Disk space" section): every test
here points a fresh ``Config`` (loaded from the shared, session-scoped
``synthetic_project``/``synthetic_qc_tables`` fixtures) at a private
``tmp_path`` output directory, never copying the ~500 MB synthetic
waveform data itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from obspy import UTCDateTime

from openmeq.chunk import run as run_chunk
from openmeq.chunk import write_chunks
from openmeq.chunk_diagnostics import compute_completeness
from openmeq.cli import main
from openmeq.config import Config, load_config
from openmeq.paths import RunPaths

#: Known anomaly location (mirrors tests/data/make_synthetic.py's own
#: constants, duplicated as plain literals -- same convention
#: tests/test_chunk_writer.py already uses, see that module's docstring).
GAP_SEED_ID = "XX.STA05.00.HHZ"
GAP_START = UTCDateTime(2025, 1, 1, 6, 0, 0)
GAP_END = UTCDateTime(2025, 1, 1, 6, 10, 0)

#: A clean (far from any anomaly) real synthetic event, per
#: tests/test_chunk_writer.py's own documented constant.
_CLEAN_EVENT_A = "ev20250101T004107_805"  # 2025-01-01T00:41:07.805563Z

DETECTIONS_COLUMNS = [
    "event_id",
    "time",
    "duration_s",
    "coincidence_sum",
    "n_stations",
    "stations",
    "trace_ids",
]


def _find_event(truth: dict, event_id_hint: str) -> dict:
    for ev in truth["events"]:
        if ev["event_id_hint"] == event_id_hint:
            return ev
    raise KeyError(f"{event_id_hint!r} not found in truth['events']")


def _detection_row(event_id: str, time, stations: list[str], duration_s: float = 5.0) -> dict:
    """One hand-built ``detections/detections.csv`` row (ARCHITECTURE.md §4.3)."""
    return {
        "event_id": event_id,
        "time": str(UTCDateTime(time)),
        "duration_s": duration_s,
        "coincidence_sum": len(stations),
        "n_stations": len(stations),
        "stations": ";".join(stations),
        "trace_ids": "",
    }


def _detections_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DETECTIONS_COLUMNS)


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    See tests/integration/test_detect_integration.py's identical helper for
    the full rationale (duplicated here to avoid cross-test-module
    coupling).
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


@pytest.fixture
def project(synthetic_project, tmp_path):
    """``(cfg, index, paths, truth)`` for one test, private to ``tmp_path``.

    Mirrors ``tests/test_chunk_writer.py``'s own ``project`` fixture.
    """
    from openmeq.index import build_index

    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    index = build_index(cfg)
    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, index, paths, truth


@pytest.fixture
def chunk_project(synthetic_project, synthetic_qc_tables, tmp_path):
    """``(cfg, paths, config_path, truth)`` with ``inventory/index.json``
    already built (from the shared ``synthetic_qc_tables`` fixture, to
    avoid rebuilding it per test) but no ``detections/detections.csv`` --
    each test writes its own.
    """
    config_path, truth = synthetic_project
    _qc_cfg, idx, _avail, _daily, _monthly, _truth2 = synthetic_qc_tables

    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()
    idx.write(paths.index_json)

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    return cfg, paths, new_config_path, truth


# ---------------------------------------------------------------------------
# test_requires_detect_first
# ---------------------------------------------------------------------------


def test_requires_detect_first(synthetic_project, tmp_path):
    """A fresh output dir with no prior ``openmeq detect`` run fails clearly."""
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"

    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    result = runner.invoke(main, ["chunk", "-c", str(new_config_path)])

    assert result.exit_code != 0
    assert "openmeq detect" in result.output


# ---------------------------------------------------------------------------
# test_chunk_command_runs_end_to_end
# ---------------------------------------------------------------------------


def test_chunk_command_runs_end_to_end(synthetic_project, tmp_path):
    """``openmeq qc`` -> ``openmeq detect`` -> ``openmeq chunk`` via
    ``CliRunner`` on the synthetic fixture. About one chunk per truth
    event (~20 total), plus completeness/diagnostics/manifest/log outputs.
    """
    config_path, truth = synthetic_project

    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    new_config_path = tmp_path / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    qc_result = runner.invoke(main, ["qc", "-c", str(new_config_path)])
    assert qc_result.exit_code == 0, qc_result.output

    detect_result = runner.invoke(main, ["detect", "-c", str(new_config_path)])
    assert detect_result.exit_code == 0, detect_result.output

    chunk_result = runner.invoke(main, ["chunk", "-c", str(new_config_path)])
    assert chunk_result.exit_code == 0, chunk_result.output

    paths = RunPaths(cfg)

    detections_df = pd.read_csv(paths.detections_csv)
    n_true = len(truth["events"])
    assert abs(len(detections_df) - n_true) <= max(2, round(0.15 * n_true)), (
        f"expected about {n_true} detections/chunks, got {len(detections_df)}"
    )

    for event_id in detections_df["event_id"]:
        chunk_path = paths.chunk_file(event_id)
        assert chunk_path.is_file(), f"{chunk_path} was not written"
        assert chunk_path.stat().st_size > 0

    assert paths.chunk_completeness_csv.is_file()
    completeness_df = pd.read_csv(paths.chunk_completeness_csv)
    assert list(completeness_df.columns) == ["event_id", "seed_id", "fraction_present"]
    assert set(completeness_df["event_id"]) == set(detections_df["event_id"])

    diag_dir = paths.diagnostics_dir("chunks")
    for name in ("completeness_heatmap.png", "completeness_boxplot.png", "summary.csv"):
        p = diag_dir / name
        assert p.is_file(), f"{p} was not created"
        assert p.stat().st_size > 0

    assert paths.manifest_json("chunks").is_file()
    assert paths.log_file("chunk").is_file()


# ---------------------------------------------------------------------------
# test_completeness_full_and_partial
# ---------------------------------------------------------------------------


def test_completeness_full_and_partial(project):
    """A station with complete data in an event's window is close to 1.0;
    the known-gap channel's affected event shows a fraction roughly
    matching the gap's actual overlap with that event's own +/- 30 s
    window (not just "< 1.0").
    """
    cfg, index, paths, _truth = project
    assert cfg.chunks.pre_s == 30.0 and cfg.chunks.post_s == 30.0  # this test's math assumes it

    # 15 s into the 600 s gap: window = [GAP_START - 15, GAP_START + 45].
    # Overlap with the gap [GAP_START, GAP_START + 600] is 45 s of the 60 s
    # window -> ~15/60 = 0.25 s of real data for the gap channel; every
    # other channel is fully present throughout this same window.
    t = GAP_START + 15.0
    event_id = "ev_completeness_partial_test"
    df = _detections_df([_detection_row(event_id, t, ["XX.STA05", "XX.STA01"])])

    write_chunks(df, index, cfg, paths)
    completeness_df = compute_completeness(df, cfg, paths)

    by_seed_id = completeness_df.set_index("seed_id")["fraction_present"]

    gap_fraction = by_seed_id[GAP_SEED_ID]
    expected_gap_fraction = 15.0 / 60.0
    assert abs(gap_fraction - expected_gap_fraction) < 0.05, (
        f"gap channel fraction_present={gap_fraction} not close to expected "
        f"{expected_gap_fraction} for this window's actual gap overlap"
    )
    assert 0.0 <= gap_fraction < 1.0

    clean_fraction = by_seed_id["XX.STA01.00.HHZ"]
    assert clean_fraction > 0.99


# ---------------------------------------------------------------------------
# test_missing_station_shows_zero_row
# ---------------------------------------------------------------------------


def test_missing_station_shows_zero_row(project):
    """A station with zero data in a given event's window still appears in
    completeness.csv with fraction_present == 0.0, not omitted.
    """
    cfg, index, paths, _truth = project

    # 300 s into the 600 s gap: window = [GAP_START + 270, GAP_START + 330],
    # entirely inside the gap -> zero data at all for the gap channel.
    t = GAP_START + 300.0
    event_id = "ev_missing_station_test"
    df = _detections_df([_detection_row(event_id, t, ["XX.STA05", "XX.STA01"])])

    write_chunks(df, index, cfg, paths)
    completeness_df = compute_completeness(df, cfg, paths)

    rows = completeness_df[
        (completeness_df["event_id"] == event_id) & (completeness_df["seed_id"] == GAP_SEED_ID)
    ]
    assert len(rows) == 1, "the missing station's row must be present, not omitted"
    assert rows.iloc[0]["fraction_present"] == 0.0

    # Every seed id in the wider inventory (not just the ones with data in
    # this event's chunk) must get a row.
    assert set(completeness_df["seed_id"]) == set(index.seed_ids())


# ---------------------------------------------------------------------------
# test_skip_unchanged_and_force
# ---------------------------------------------------------------------------


def test_skip_unchanged_and_force(chunk_project, monkeypatch):
    """Mirrors M1.7/M2.3's own manifest tests (``StepManifest``
    skip-unchanged / ``--force``).
    """
    cfg, paths, config_path, truth = chunk_project

    ev = _find_event(truth, _CLEAN_EVENT_A)
    df = _detections_df([_detection_row(ev["event_id_hint"], ev["origin"]["time"], ["XX.STA01"])])
    df.to_csv(paths.detections_csv, index=False, columns=DETECTIONS_COLUMNS)

    call_count = 0
    real_write_chunks = write_chunks

    def _counting_write_chunks(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_write_chunks(*args, **kwargs)

    monkeypatch.setattr("openmeq.chunk.write_chunks", _counting_write_chunks)

    run_chunk(cfg, paths, config_path=str(config_path))
    assert call_count == 1
    assert paths.chunk_completeness_csv.is_file()
    mtime1 = paths.chunk_completeness_csv.stat().st_mtime_ns

    # Unchanged: write_chunks must not be called again, and the output
    # file's mtime must not move.
    run_chunk(cfg, paths, config_path=str(config_path))
    assert call_count == 1
    assert paths.chunk_completeness_csv.stat().st_mtime_ns == mtime1

    # --force reruns it regardless.
    run_chunk(cfg, paths, config_path=str(config_path), force=True)
    assert call_count == 2
