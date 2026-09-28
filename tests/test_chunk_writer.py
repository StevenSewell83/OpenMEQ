"""Tests for openmeq.chunk (M3.1, ``OpenMEQ-rq8.1``).

Detection (M2.x) is being built concurrently with this task, so
``detections/detections.csv`` (ARCHITECTURE.md §4.3) is built **by hand**
here, one row per test, using real truth event times from the
``synthetic_project`` fixture (``tests/data/make_synthetic.py``) --
``event_id_hint`` for the ``event_id`` column (computed the same way a real
detection step would, via ``openmeq.catalog.event_id_from_time``) and
``origin.time`` for the ``time`` column.

Isolation note (worker instructions "Disk space" section): every test here
uses only ``tmp_path`` for anything it writes. Rather than
``shutil.copytree``-ing the whole ~500 MB synthetic project directory per
``tests/conftest.py``'s general guidance, each test instead points a fresh
``Config`` (loaded from the shared, session-scoped fixture's
``config.yaml``) at a private ``tmp_path`` output directory
(``cfg.project.output_dir = tmp_path / "run"``) -- ``waveform_dirs`` are
only ever *read* by this module (never written to), so nothing about the
shared fixture actually needs copying; only the tiny ``inventory/index.json``
and ``chunks/`` this module writes need to land somewhere private, and
``tmp_path`` is exactly that.
"""

from __future__ import annotations

import pandas as pd
import pytest
from obspy import UTCDateTime, read

from openmeq.catalog import event_id_from_time
from openmeq.chunk import group_detections_by_day, write_chunks
from openmeq.config import load_config
from openmeq.index import DataIndex, build_index
from openmeq.paths import RunPaths

#: Known anomaly locations (mirrors tests/data/make_synthetic.py's own
#: constants; duplicated here as plain literals since that module is a test
#: helper, not part of the installed package's public API).
GAP_SEED_ID = "XX.STA05.00.HHZ"
GAP_START = UTCDateTime(2025, 1, 1, 6, 0, 0)
GAP_END = UTCDateTime(2025, 1, 1, 6, 10, 0)

#: Event ids of a handful of the synthetic dataset's real (deterministic --
#: fixed seed) events, picked for specific properties needed below. Printed
#: directly from ``tests/data/make_synthetic.generate()`` with the default
#: seed; re-verified by ``test_synthetic.py::test_deterministic`` staying
#: green, so these values are stable.
_CLEAN_EVENT_A = "ev20250101T004107_805"  # 2025-01-01T00:41:07.805563Z: far from any anomaly
_CLEAN_EVENT_B = "ev20250101T052729_858"  # 2025-01-01T05:27:29.858462Z: still day 1, before the gap
_CLEAN_EVENT_C = "ev20250101T053316_611"  # 2025-01-01T05:33:16.611559Z: still day 1, before the gap
_CLEAN_EVENT_D = "ev20250101T100744_225"  # 2025-01-01T10:07:44.225523Z: day 1, after the gap
#: Deliberately placed ~3 s before the day-1/day-2 boundary by the generator
#: (``BOUNDARY_EVENT_OFFSET_S``) specifically to exercise cross-day windows.
_BOUNDARY_EVENT = "ev20250101T235957_000"  # 2025-01-01T23:59:57.000000Z

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


def _detection_row(
    event_id: str,
    time: UTCDateTime,
    stations: list[str],
    trace_ids: list[str],
    duration_s: float = 5.0,
) -> dict:
    """One hand-built ``detections/detections.csv`` row (ARCHITECTURE.md §4.3)."""
    return {
        "event_id": event_id,
        "time": str(UTCDateTime(time)),
        "duration_s": duration_s,
        "coincidence_sum": len(stations),
        "n_stations": len(stations),
        "stations": ";".join(stations),
        "trace_ids": ";".join(trace_ids),
    }


def _detections_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DETECTIONS_COLUMNS)


@pytest.fixture
def project(synthetic_project, tmp_path):
    """``(cfg, index, paths, truth)`` for one test, private to ``tmp_path``.

    See the module docstring's isolation note: ``cfg.project.output_dir`` is
    redirected to ``tmp_path``, so ``build_index``/``write_chunks`` never
    write into the session-shared ``synthetic_project`` directory.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    index = build_index(cfg)
    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, index, paths, truth


# ---------------------------------------------------------------------------
# group_detections_by_day
# ---------------------------------------------------------------------------


def test_group_detections_by_day_groups_by_calendar_date(project):
    cfg, _index, _paths, truth = project
    a = _find_event(truth, _CLEAN_EVENT_A)
    b = _find_event(truth, _CLEAN_EVENT_B)
    boundary = _find_event(truth, _BOUNDARY_EVENT)

    rows = [
        _detection_row(a["event_id_hint"], UTCDateTime(a["origin"]["time"]), ["XX.STA01"], []),
        _detection_row(b["event_id_hint"], UTCDateTime(b["origin"]["time"]), ["XX.STA01"], []),
        _detection_row(
            boundary["event_id_hint"], UTCDateTime(boundary["origin"]["time"]), ["XX.STA01"], []
        ),
    ]
    groups = group_detections_by_day(_detections_df(rows), cfg)

    # a/b/boundary are all on 2025-01-01 (boundary is 3 s before midnight,
    # per make_synthetic's BOUNDARY_EVENT_OFFSET_S) -> one group, not three.
    assert set(groups.keys()) == {"2025-01-01"}
    assert len(groups["2025-01-01"]) == 3
    event_ids = {row["event_id"] for row in groups["2025-01-01"]}
    assert event_ids == {a["event_id_hint"], b["event_id_hint"], boundary["event_id_hint"]}


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_window_correctness(project):
    cfg, index, paths, truth = project
    ev = _find_event(truth, _CLEAN_EVENT_A)
    t = UTCDateTime(ev["origin"]["time"])
    event_id = ev["event_id_hint"]

    df = _detections_df(
        [_detection_row(event_id, t, ["XX.STA01", "XX.STA02"], ["XX.STA01.00.HHZ"])]
    )
    write_chunks(df, index, cfg, paths)

    chunk_path = paths.chunk_file(event_id)
    assert chunk_path.exists()
    st = read(str(chunk_path))
    assert len(st) > 0

    expected_start = t - cfg.chunks.pre_s
    expected_end = t + cfg.chunks.post_s
    for tr in st:
        dt = 1.0 / tr.stats.sampling_rate
        assert abs(tr.stats.starttime - expected_start) <= dt
        assert abs(tr.stats.endtime - expected_end) <= dt


def test_all_channels_included(project):
    cfg, index, paths, truth = project
    ev = _find_event(truth, _CLEAN_EVENT_D)
    t = UTCDateTime(ev["origin"]["time"])
    event_id = ev["event_id_hint"]

    # Only 2 of the network's 8 stations named in `stations`/`trace_ids` --
    # the written chunk must still hold every channel with data in the
    # window, not only these.
    df = _detections_df(
        [_detection_row(event_id, t, ["XX.STA01", "XX.STA02"], ["XX.STA01.00.HHZ"])]
    )
    rows = write_chunks(df, index, cfg, paths)

    st = read(str(paths.chunk_file(event_id)))
    seed_ids = {tr.get_id() for tr in st}
    expected = set(index.seed_ids())
    assert len(expected) == 8 * 3  # 6 surface x 3 + 2 borehole x 3
    assert seed_ids == expected

    returned_seed_ids = {row["seed_id"] for row in rows if row["event_id"] == event_id}
    assert returned_seed_ids == expected


def test_original_sampling_preserved(project):
    cfg, index, paths, truth = project
    ev = _find_event(truth, _CLEAN_EVENT_D)
    t = UTCDateTime(ev["origin"]["time"])
    event_id = ev["event_id_hint"]

    df = _detections_df([_detection_row(event_id, t, ["XX.STA01"], ["XX.STA01.00.HHZ"])])
    write_chunks(df, index, cfg, paths)

    st = read(str(paths.chunk_file(event_id)))
    assert len(st) > 0
    for tr in st:
        assert tr.stats.sampling_rate == 200.0


def test_gap_spanning_event_writes_split_segments(project):
    cfg, index, paths, _truth = project

    # Default +-30 s is far narrower than the 600 s known gap; widen the
    # chunk window so it spans data - gap - data on both sides.
    cfg.chunks.pre_s = 400.0
    cfg.chunks.post_s = 400.0

    midpoint = GAP_START + (GAP_END - GAP_START) / 2.0
    event_id = event_id_from_time(midpoint)
    df = _detections_df([_detection_row(event_id, midpoint, ["XX.STA05"], [GAP_SEED_ID])])

    write_chunks(df, index, cfg, paths)  # must not raise

    st = read(str(paths.chunk_file(event_id)))
    gap_traces = [tr for tr in st if tr.get_id() == GAP_SEED_ID]
    assert len(gap_traces) >= 2, "expected the gap channel to write as >=2 segments"

    full_npts = int(round((cfg.chunks.pre_s + cfg.chunks.post_s) * 200.0))
    total_npts = sum(tr.stats.npts for tr in gap_traces)
    assert total_npts < full_npts  # a genuine gap, never filled/interpolated


def test_day_boundary_event_window(project):
    cfg, index, paths, truth = project
    ev = _find_event(truth, _BOUNDARY_EVENT)
    t = UTCDateTime(ev["origin"]["time"])
    event_id = ev["event_id_hint"]

    expected_start = t - cfg.chunks.pre_s
    expected_end = t + cfg.chunks.post_s
    day2_start = UTCDateTime(2025, 1, 2)
    assert expected_end > day2_start  # sanity: this window really crosses midnight

    df = _detections_df([_detection_row(event_id, t, ["XX.STA01"], ["XX.STA01.00.HHZ"])])
    write_chunks(df, index, cfg, paths)

    st = read(str(paths.chunk_file(event_id)))
    assert len(st) > 0

    dt = 1.0 / 200.0
    starts = [tr.stats.starttime for tr in st]
    ends = [tr.stats.endtime for tr in st]
    assert abs(min(starts) - expected_start) <= dt
    assert abs(max(ends) - expected_end) <= dt
    # Not truncated at midnight: some data reaches past the day boundary.
    assert any(tr.stats.endtime > day2_start for tr in st)


def test_one_read_per_day_group(project, monkeypatch):
    cfg, index, paths, truth = project
    event_ids = [_CLEAN_EVENT_A, _CLEAN_EVENT_B, _CLEAN_EVENT_C]
    events = [_find_event(truth, eid) for eid in event_ids]
    rows = [
        _detection_row(
            ev["event_id_hint"],
            UTCDateTime(ev["origin"]["time"]),
            ["XX.STA01"],
            ["XX.STA01.00.HHZ"],
        )
        for ev in events
    ]
    df = _detections_df(rows)

    # A/B/C are hours apart (module docstring's constants), well beyond
    # cfg.chunks.max_read_window_s's real default (600 s, M3.2 /
    # OpenMEQ-rq8.2's sub-window-splitting memory fix -- see
    # test_day_group_splits_into_subwindows_on_sparse_detections below for
    # that behaviour). This test's own point predates that fix and is about
    # day-*grouping* only (one read per day group, not per detection) --
    # widen max_read_window_s here so it still exercises exactly that,
    # undiluted by sub-window splitting.
    cfg.chunks.max_read_window_s = 86400.0

    calls = []
    real_read = DataIndex.read

    def spy_read(self, *args, **kwargs):
        calls.append((args, kwargs))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(DataIndex, "read", spy_read)

    write_chunks(df, index, cfg, paths)

    # All three detections fall on the same UTC day, and are all within the
    # widened max_read_window_s above -> exactly one whole-network read for
    # the whole group, not one per detection.
    assert len(calls) == 1
    for ev in events:
        assert paths.chunk_file(ev["event_id_hint"]).exists()


# ---------------------------------------------------------------------------
# Sub-window splitting (M3.2 / OpenMEQ-rq8.2 memory fix)
# ---------------------------------------------------------------------------


def test_split_detections_into_subwindows_groups_by_gap():
    """Unit test for :func:`openmeq.chunk._split_detections_into_subwindows`."""
    from openmeq.chunk import _split_detections_into_subwindows

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    detections = [
        {"event_id": "a", "time": str(t0)},
        {"event_id": "b", "time": str(t0 + 100)},  # 100 s after a: same sub-window
        {"event_id": "c", "time": str(t0 + 100 + 700)},  # 700 s after b: new sub-window
        {"event_id": "d", "time": str(t0 + 100 + 700 + 50)},  # 50 s after c: same sub-window as c
    ]

    subwindows = _split_detections_into_subwindows(detections, max_read_window_s=600.0)

    assert [[det["event_id"] for det in sub] for sub in subwindows] == [["a", "b"], ["c", "d"]]


def test_split_detections_into_subwindows_empty_and_single():
    from openmeq.chunk import _split_detections_into_subwindows

    assert _split_detections_into_subwindows([], max_read_window_s=600.0) == []

    one = [{"event_id": "a", "time": str(UTCDateTime(2025, 1, 1))}]
    assert _split_detections_into_subwindows(one, max_read_window_s=600.0) == [one]


def test_day_group_splits_into_subwindows_on_sparse_detections(project, monkeypatch):
    """The actual memory fix: sparse detections within one day-group now
    trigger more than one (smaller) read, not the single whole-day read
    M3.1 originally always performed -- each read still covers exactly one
    sub-window's detections (the "one read per sub-window" property), never
    falling back to one read per detection either.
    """
    cfg, index, paths, truth = project
    # A/B/C: A is ~4h46m before B; B is ~5m47s before C (test_chunk_writer.py
    # module docstring's own documented gaps) -- with the real 600 s
    # default, A alone starts one sub-window and B+C share a second.
    event_ids = [_CLEAN_EVENT_A, _CLEAN_EVENT_B, _CLEAN_EVENT_C]
    events = [_find_event(truth, eid) for eid in event_ids]
    rows = [
        _detection_row(
            ev["event_id_hint"],
            UTCDateTime(ev["origin"]["time"]),
            ["XX.STA01"],
            ["XX.STA01.00.HHZ"],
        )
        for ev in events
    ]
    df = _detections_df(rows)
    assert cfg.chunks.max_read_window_s == 600.0  # the real shipped default

    calls = []
    real_read = DataIndex.read

    def spy_read(self, *args, **kwargs):
        calls.append((args, kwargs))
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(DataIndex, "read", spy_read)

    write_chunks(df, index, cfg, paths)

    # Two sub-windows ({A}, {B, C}), not one whole-day read and not three
    # per-detection reads.
    assert len(calls) == 2
    for ev in events:
        assert paths.chunk_file(ev["event_id_hint"]).exists()
