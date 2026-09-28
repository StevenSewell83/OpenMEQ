"""Tests for openmeq.qc.availability (M1.3, ARCHITECTURE.md §3).

Uses the session-scoped ``synthetic_project`` fixture (tests/conftest.py):
8 stations, 2 days at 200 Hz, with a deliberate gap on
``XX.STA05.00.HHZ`` (2025-01-01 06:00:00 to 06:10:00 UTC), per
``truth["known_issues"]`` (``tests/data/make_synthetic.py``'s
``GAP_SEED_ID``/``GAP_START``/``GAP_END``).

Per ``tests/conftest.py``'s usage note, ``build_index`` only *reads*
waveform files and writes ``inventory/index.json`` under the shared
session ``cfg.project.output_dir`` -- idempotent across repeated calls
(mirrors ``tests/test_index.py``'s own pattern) -- so these tests use the
``synthetic_project`` fixture directly rather than copying the project
directory.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from obspy import UTCDateTime

from openmeq.config import load_config
from openmeq.index import DataIndex, build_index
from openmeq.qc.availability import (
    availability_for_channel_day,
    compute_availability,
    date_range_inclusive,
    write_availability_csv,
)


def _gap_issue(truth: dict) -> dict:
    (issue,) = [i for i in truth["known_issues"] if i["kind"] == "gap"]
    return issue


@pytest.fixture(scope="module")
def indexed_project(synthetic_project, tmp_path_factory):
    """``(config_path, cfg, index, index_path, truth)`` for the shared
    synthetic project. ``index_path`` is this module's own copy of the
    written index JSON (``DataIndex`` itself does not remember the path it
    was last written/loaded from), reused across every test in this file.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    index = build_index(cfg)
    index_path = tmp_path_factory.mktemp("qc_availability_index") / "index.json"
    index.write(index_path)
    return config_path, cfg, index, index_path, truth


def test_full_coverage_channel(indexed_project):
    config_path, cfg, index, index_path, truth = indexed_project

    # Any channel/day other than the known gap channel/day is fully,
    # contiguously covered by construction (make_synthetic.py's module
    # docstring: exactly one gap, on one channel/day only).
    seed_id = "XX.STA01.00.HHZ"
    day = date(2025, 1, 1)

    result = availability_for_channel_day(seed_id, day, str(index_path), str(config_path))
    assert result["coverage_pct"] == pytest.approx(100.0, abs=1e-3)
    assert result["n_gaps"] == 0
    assert result["n_overlaps"] == 0
    assert result["gap_s"] == pytest.approx(0.0, abs=1e-3)


def test_known_gap_channel(indexed_project):
    config_path, cfg, index, index_path, truth = indexed_project
    gap_issue = _gap_issue(truth)
    seed_id = gap_issue["seed_id"]
    gap_start = UTCDateTime(gap_issue["start"])
    gap_end = UTCDateTime(gap_issue["end"])
    injected_gap_s = gap_end - gap_start
    day = gap_start.date

    result = availability_for_channel_day(seed_id, day, str(index_path), str(config_path))

    assert result["seed_id"] == seed_id
    assert result["date"] == day.isoformat()
    assert result["coverage_pct"] < 100.0
    assert result["n_gaps"] >= 1
    assert result["gap_s"] > 0.0
    # Roughly matching the injected 10-minute gap (within a couple of
    # sample periods either side; 200 Hz => period 0.005 s).
    assert result["gap_s"] == pytest.approx(injected_gap_s, abs=1.0)
    expected_pct = 100.0 * (86400.0 - injected_gap_s) / 86400.0
    assert result["coverage_pct"] == pytest.approx(expected_pct, abs=0.01)


def test_overlap_detection(tmp_path):
    """A hand-built two-overlapping-segment index (no real files needed:
    this module never reads waveform data, only index segment metadata --
    see the module docstring's "Design: metadata-only" section)."""
    seed_id = "YY.OVER1.00.HHZ"
    day = date(2025, 6, 1)
    day_start = UTCDateTime(day.isoformat())
    sr = 100.0

    seg1_start = day_start + 1000.0
    seg1_end = day_start + 2000.0
    # Starts before seg1 ends: a genuine overlap.
    seg2_start = day_start + 1500.0
    seg2_end = day_start + 2500.0

    index = DataIndex(version=1, created="2026-01-01T00:00:00Z", roots=["/data"])
    index.channels[seed_id] = [
        ("/data/seg1.mseed", seg1_start.isoformat() + "Z", seg1_end.isoformat() + "Z", sr),
        ("/data/seg2.mseed", seg2_start.isoformat() + "Z", seg2_end.isoformat() + "Z", sr),
    ]
    index_path = tmp_path / "index.json"
    index.write(index_path)

    cfg_path = _write_minimal_config(tmp_path)

    result = availability_for_channel_day(seed_id, day, str(index_path), str(cfg_path))

    assert result["n_overlaps"] >= 1
    # The overlap must not also be miscounted as a gap: only the day's
    # leading/trailing missing time (before seg1, after seg2) are gaps.
    assert result["n_gaps"] == 2
    assert result["coverage_pct"] < 100.0


def test_no_data_day(tmp_path):
    seed_id = "YY.NODATA.00.HHZ"
    day = date(2025, 6, 1)

    index = DataIndex(version=1, created="2026-01-01T00:00:00Z", roots=["/data"])
    index.channels[seed_id] = []
    index_path = tmp_path / "index.json"
    index.write(index_path)

    cfg_path = _write_minimal_config(tmp_path)

    result = availability_for_channel_day(seed_id, day, str(index_path), str(cfg_path))

    assert result["coverage_pct"] == 0.0
    assert result["n_gaps"] == 0
    assert result["n_overlaps"] == 0
    assert result["gap_s"] == pytest.approx(86400.0)


def test_csv_schema(indexed_project, tmp_path):
    config_path, cfg, index, index_path, truth = indexed_project

    seed_ids = index.seed_ids()[:3]
    dates = date_range_inclusive(cfg.data.start, cfg.data.end)
    df = compute_availability(index, seed_ids, dates, cfg)

    assert list(df.columns) == [
        "seed_id",
        "date",
        "coverage_pct",
        "n_gaps",
        "n_overlaps",
        "gap_s",
    ]
    assert len(df) == len(seed_ids) * len(dates)

    csv_path = tmp_path / "availability.csv"
    write_availability_csv(df, csv_path)
    assert csv_path.exists()

    roundtrip = pd.read_csv(csv_path)
    assert list(roundtrip.columns) == list(df.columns)
    assert len(roundtrip) == len(df)
    pd.testing.assert_frame_equal(
        roundtrip.sort_values(["seed_id", "date"]).reset_index(drop=True),
        df.sort_values(["seed_id", "date"]).reset_index(drop=True),
        check_dtype=False,
    )


def _write_minimal_config(tmp_path) -> Path:
    """A minimal valid config YAML, for tests that don't need the full
    synthetic project (only ``availability_for_channel_day``'s ``cfg_path``
    contract needs *some* loadable config; its fields are unused by this
    module's own computation)."""
    waveform_dir = tmp_path / "waveforms"
    waveform_dir.mkdir(exist_ok=True)
    velocity_file = tmp_path / "model.txt"
    velocity_file.write_text("0.0 3.5 2.0 2.3\n", encoding="utf-8")

    config_path = tmp_path / "minimal_config.yaml"
    config_path.write_text(
        f"""
project:
  name: test
  output_dir: {tmp_path / "out"}
data:
  waveform_dirs: ["{waveform_dir}"]
  start: 2025-06-01
  end: 2025-06-01
nlloc:
  velocity_model:
    type: "1d"
    file: "{velocity_file}"
  grid:
    origin: [0.0, 0.0]
    size_km: [10.0, 10.0, 10.0]
    spacing_km: 1.0
hypodd:
  velocity_model:
    type: "1d"
    file: "{velocity_file}"
""",
        encoding="utf-8",
    )
    return Path(config_path)
