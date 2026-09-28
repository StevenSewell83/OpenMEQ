"""Tests for openmeq.index: DataIndex build, JSON round-trip, query and read API.

Uses the session-scoped ``synthetic_project`` fixture (tests/conftest.py):
8 stations (6 surface HH?, 2 borehole DP?, each mapped to Z/1/2), 2 days at
200 Hz, one deliberate gap on ``XX.STA05.00.HHZ`` (2025-01-01 06:00:00 to
06:10:00 UTC, per ``tests/data/make_synthetic.py``'s ``GAP_SEED_ID``/
``GAP_START``/``GAP_END`` and mirrored in ``truth["known_issues"]``).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import obspy
from obspy import UTCDateTime

from openmeq import index as index_mod
from openmeq.config import load_config
from openmeq.index import DataIndex, _scan_file, build_index


def _gap_issue(truth: dict) -> dict:
    (issue,) = [i for i in truth["known_issues"] if i["kind"] == "gap"]
    return issue


def test_build_index_finds_all_synthetic_channels(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)

    idx = build_index(cfg)

    seed_ids = idx.seed_ids()
    # 6 surface stations x HHZ/HHN/HHE + 2 borehole stations x DPZ/DP1/DP2
    assert len(seed_ids) == 8 * 3

    surface_stations = [f"STA0{i}" for i in range(1, 7)]
    borehole_stations = ["BH01", "BH02"]
    for sta in surface_stations:
        for cha in ("HHZ", "HHN", "HHE"):
            assert f"XX.{sta}.00.{cha}" in seed_ids
    for sta in borehole_stations:
        for cha in ("DPZ", "DP1", "DP2"):
            assert f"XX.{sta}.00.{cha}" in seed_ids

    day0 = UTCDateTime(2025, 1, 1)
    day_end = UTCDateTime(2025, 1, 3)  # exclusive-ish upper bound (end of day 2)
    for seed_id in seed_ids:
        segments = idx.channels[seed_id]
        assert len(segments) >= 2  # at least one segment per day
        starts = [UTCDateTime(seg[1]) for seg in segments]
        ends = [UTCDateTime(seg[2]) for seg in segments]
        assert min(starts) >= day0 - 1.0
        assert max(ends) <= day_end + 1.0
        # sorted by start time
        assert starts == sorted(starts)


def test_index_json_roundtrip(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)

    idx = build_index(cfg)

    out_path = tmp_path / "index.json"
    idx.write(out_path)

    with open(out_path, encoding="utf-8") as f:
        raw = json.load(f)
    assert set(raw.keys()) == {"version", "created", "roots", "channels"}
    assert raw["version"] == 1
    assert isinstance(raw["created"], str)
    assert raw["created"].endswith("Z")
    assert isinstance(raw["roots"], list) and len(raw["roots"]) >= 1
    for root in raw["roots"]:
        assert Path(root).is_absolute()

    loaded = DataIndex.load(out_path)
    assert loaded.version == idx.version
    assert loaded.created == idx.created
    assert loaded.roots == idx.roots
    assert loaded.channels == idx.channels

    # Paths inside channels are absolute.
    for segments in raw["channels"].values():
        for abs_path, start_iso, end_iso, sr in segments:
            assert Path(abs_path).is_absolute()
            assert start_iso.endswith("Z")
            assert end_iso.endswith("Z")
            assert isinstance(sr, (int, float))

    # The index written by build_index itself also lands at paths.index_json.
    from openmeq.paths import RunPaths

    paths = RunPaths(cfg)
    assert paths.index_json.exists()
    assert DataIndex.load(paths.index_json).channels == idx.channels


def test_files_time_range(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    seed_id = "XX.STA01.00.HHZ"
    day1_start = UTCDateTime(2025, 1, 1)
    day2_start = UTCDateTime(2025, 1, 2)

    # A window entirely within day 1 should return only day 1's file.
    files_day1 = idx.files(seed_id, day1_start + 10, day1_start + 20)
    assert len(files_day1) == 1
    assert "2025-01-01" in str(files_day1[0])

    # A window spanning the day boundary should return both files, in
    # start-time order.
    files_both = idx.files(seed_id, day1_start + 86390, day2_start + 10)
    assert len(files_both) == 2
    assert "2025-01-01" in str(files_both[0])
    assert "2025-01-02" in str(files_both[1])

    # A window entirely outside the archive returns nothing.
    files_none = idx.files(seed_id, UTCDateTime(2030, 1, 1), UTCDateTime(2030, 1, 2))
    assert files_none == []

    # Unknown seed_id returns nothing rather than raising.
    assert idx.files("ZZ.NOPE.00.HHZ", day1_start, day2_start) == []


def test_read_respects_pad(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    seed_id = "XX.STA02.00.HHZ"
    t1 = UTCDateTime(2025, 1, 1, 3, 0, 0)
    t2 = UTCDateTime(2025, 1, 1, 3, 1, 0)
    pad = 5.0

    stream = idx.read(seed_id, t1, t2, pad=pad)
    assert len(stream) >= 1
    tr = stream.select(id=seed_id)[0]
    dt = 1.0 / tr.stats.sampling_rate
    assert abs(tr.stats.starttime - (t1 - pad)) <= dt
    assert abs(tr.stats.endtime - (t2 + pad)) <= dt


def test_read_unknown_seed_id_returns_empty_stream(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    stream = idx.read("ZZ.NOPE.00.HHZ", UTCDateTime(2025, 1, 1), UTCDateTime(2025, 1, 1, 0, 1))
    assert len(stream) == 0


def test_read_wildcard_seed_ids(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    t1 = UTCDateTime(2025, 1, 1, 1, 0, 0)
    t2 = UTCDateTime(2025, 1, 1, 1, 1, 0)
    stream = idx.read("XX.STA01.00.HH?", t1, t2)
    ids = {tr.get_id() for tr in stream}
    assert ids == {"XX.STA01.00.HHZ", "XX.STA01.00.HHN", "XX.STA01.00.HHE"}


def test_read_known_gap_is_masked(synthetic_project):
    """Reading across the synthetic dataset's known gap yields a masked array.

    The gap's exact (seed_id, start, end) is exposed directly by
    ``truth["known_issues"]`` (kind == "gap"), generated by
    ``tests/data/make_synthetic.py`` from its module-level ``GAP_SEED_ID``/
    ``GAP_START``/``GAP_END`` constants -- used here directly rather than
    re-derived, since the fixture already documents it as ground truth.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    issue = _gap_issue(truth)
    seed_id = issue["seed_id"]
    gap_start = UTCDateTime(issue["start"])
    gap_end = UTCDateTime(issue["end"])

    stream = idx.read(seed_id, gap_start - 30.0, gap_end + 30.0)
    assert len(stream) == 1
    tr = stream[0]
    assert isinstance(tr.data, np.ma.MaskedArray)
    assert tr.data.mask.any()


def test_scan_file_is_headonly(synthetic_project, monkeypatch):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)

    calls = []
    real_read = obspy.read

    def spy_read(path, *args, **kwargs):
        calls.append((args, kwargs))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(index_mod, "read", spy_read)

    waveform_dir = cfg.data.waveform_dirs[0]
    some_file = next(waveform_dir.rglob(cfg.data.file_pattern))
    _scan_file(str(some_file))

    assert calls, "obspy.read was not called by _scan_file"
    assert calls[0][1].get("headonly") is True


def test_corrupt_file_skipped(tmp_path, caplog):
    corrupt = tmp_path / "not_really_mseed.mseed"
    corrupt.write_bytes(b"this is not a valid miniseed file at all, sorry")

    with caplog.at_level(logging.WARNING, logger="openmeq.index"):
        result = _scan_file(str(corrupt))

    assert result == []
    assert any("not_really_mseed.mseed" in record.message for record in caplog.records)


def test_build_index_skips_corrupt_file(synthetic_project, tmp_path):
    """A corrupted file mixed into the archive doesn't abort the whole build."""
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")

    waveform_dir = cfg.data.waveform_dirs[0]
    corrupt_path = waveform_dir / "2025-01-01" / "ZZ.JUNK.00.XXX.mseed"
    corrupt_path.write_bytes(b"not miniseed data, deliberately corrupted for this test")

    idx = build_index(cfg)

    # The corrupt file contributed nothing, and every real channel is
    # still present.
    assert "ZZ.JUNK.00.XXX" not in idx.channels
    assert len(idx.seed_ids()) == 8 * 3


def test_pmap_n_workers_1_matches_parallel(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")

    cfg_serial = cfg.model_copy(deep=True)
    cfg_serial.parallel = cfg_serial.parallel.model_copy(update={"n_workers": 1})
    idx_serial = build_index(cfg_serial)

    cfg_parallel = cfg.model_copy(deep=True)
    cfg_parallel.parallel = cfg_parallel.parallel.model_copy(update={"n_workers": 4})
    idx_parallel = build_index(cfg_parallel)

    assert idx_serial.channels == idx_parallel.channels
