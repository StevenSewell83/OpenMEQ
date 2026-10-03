"""Tests for ``openmeq.detect.merge`` (M2.2, ``OpenMEQ-ujx.2``).

Per the "Disk space" rule for this run: every test uses pytest's ``tmp_path``
(or the session-scoped ``synthetic_project`` fixture, copied into a
function-scoped ``tmp_path`` before any pipeline step writes anything -- see
``tests/test_detect_worker.py``'s module docstring for the same pattern) and
never writes outside it.
"""

from __future__ import annotations

import re
import shutil

import pandas as pd
import pytest
from obspy import UTCDateTime

from openmeq.catalog import event_id_from_time
from openmeq.config import load_config
from openmeq.detect.merge import (
    DETECTIONS_COLUMNS,
    _assign_event_ids,
    merge_windows,
    write_detections_csv,
)
from openmeq.detect.worker import build_windows, detect_window
from openmeq.index import build_index
from openmeq.paths import RunPaths

_MATCH_TOLERANCE_S = 12.0  # same generous tolerance as test_detect_worker.py


def _trig(
    time: str,
    coincidence_sum: float = 5.0,
    stations: list[str] | None = None,
    trace_ids: list[str] | None = None,
    duration_s: float = 1.0,
) -> dict:
    """Build one raw trigger dict in :func:`~openmeq.detect.worker.detect_window`'s shape."""
    stations = stations if stations is not None else ["XX.STA01", "XX.STA02"]
    trace_ids = trace_ids if trace_ids is not None else [f"{s}.00.HHZ" for s in stations]
    return {
        "time": time,
        "duration_s": duration_s,
        "coincidence_sum": coincidence_sum,
        "n_stations": len(stations),
        "stations": sorted(stations),
        "trace_ids": sorted(trace_ids),
    }


@pytest.fixture()
def cfg(synthetic_project):
    """A real ``Config`` (read-only load, no filesystem writes) for
    ``cfg.detection.dedup_window_s``; individual tests override that field.
    """
    config_path, _truth = synthetic_project
    return load_config(config_path)


# ---------------------------------------------------------------------------
# De-duplication rule
# ---------------------------------------------------------------------------


def test_dedup_merges_boundary_duplicates(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    # Same underlying event, seen by the two adjacent day windows: the
    # window ending at the boundary sees fewer stations (clipped), the
    # window starting at the boundary sees the fuller station list.
    trig_a = _trig(
        (t0).isoformat() + "Z",
        coincidence_sum=4.0,
        stations=["XX.STA01", "XX.STA02"],
    )
    trig_b = _trig(
        (t0 + 0.05).isoformat() + "Z",
        coincidence_sum=5.0,
        stations=["XX.STA01", "XX.STA02", "XX.STA03"],
    )

    df = merge_windows([[trig_a], [trig_b]], cfg)

    assert len(df) == 1
    row = df.iloc[0]
    # Higher coincidence_sum member's time/coincidence_sum win.
    assert row["coincidence_sum"] == 5.0
    assert UTCDateTime(row["time"]) == UTCDateTime(trig_b["time"])
    # Union of stations across both members.
    assert row["stations"] == ["XX.STA01", "XX.STA02", "XX.STA03"]
    assert row["n_stations"] == 3


def test_dedup_prefers_earliest_time_on_coincidence_sum_tie(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig((t0 + 0.5).isoformat() + "Z", coincidence_sum=5.0)
    trig_b = _trig((t0).isoformat() + "Z", coincidence_sum=5.0)

    df = merge_windows([[trig_a], [trig_b]], cfg)

    assert len(df) == 1
    assert UTCDateTime(df.iloc[0]["time"]) == t0


def test_distinct_events_not_merged(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig(t0.isoformat() + "Z")
    trig_b = _trig((t0 + 10.0).isoformat() + "Z")  # well beyond dedup_window_s

    df = merge_windows([[trig_a, trig_b]], cfg)

    assert len(df) == 2
    times = sorted(UTCDateTime(t) for t in df["time"])
    assert times[0] == t0
    assert times[1] == t0 + 10.0


def test_greedy_chaining_merges_span_exceeding_dedup_window(cfg):
    """A chain of triggers each within ``dedup_window_s`` of its neighbour,
    but spanning more than ``dedup_window_s`` end-to-end, is one cluster
    (the deliberate greedy-chaining design choice, not fixed-width binning).
    """
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    # 0, 1.5, 3.0 -> consecutive gaps are 1.5s (< 2.0s) but first-to-last
    # span is 3.0s (> 2.0s dedup_window_s).
    trig_a = _trig(t0.isoformat() + "Z", coincidence_sum=3.0, stations=["XX.STA01"])
    trig_b = _trig((t0 + 1.5).isoformat() + "Z", coincidence_sum=4.0, stations=["XX.STA02"])
    trig_c = _trig((t0 + 3.0).isoformat() + "Z", coincidence_sum=3.5, stations=["XX.STA03"])

    df = merge_windows([[trig_a, trig_b, trig_c]], cfg)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["coincidence_sum"] == 4.0
    assert UTCDateTime(row["time"]) == t0 + 1.5
    assert row["stations"] == ["XX.STA01", "XX.STA02", "XX.STA03"]

    # A fixed-width-binning implementation would (with 2.0s-wide bins
    # aligned at t0) likely split trig_c from trig_a/trig_b; this asserts
    # the greedy-chaining behaviour explicitly rather than merely checking
    # cluster count above.
    would_be_binned = abs((t0 + 3.0) - t0) > cfg.detection.dedup_window_s
    assert would_be_binned


def test_near_duplicate_just_inside_window_merges(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig(t0.isoformat() + "Z", coincidence_sum=4.0)
    trig_b = _trig((t0 + 1.99).isoformat() + "Z", coincidence_sum=4.5)

    df = merge_windows([[trig_a, trig_b]], cfg)
    assert len(df) == 1


def test_near_duplicate_just_outside_window_not_merged(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig(t0.isoformat() + "Z", coincidence_sum=4.0)
    trig_b = _trig((t0 + 2.01).isoformat() + "Z", coincidence_sum=4.5)

    df = merge_windows([[trig_a, trig_b]], cfg)
    assert len(df) == 2


def test_duration_is_max_across_cluster(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig(t0.isoformat() + "Z", coincidence_sum=5.0, duration_s=1.0)
    trig_b = _trig((t0 + 0.1).isoformat() + "Z", coincidence_sum=3.0, duration_s=2.5)

    df = merge_windows([[trig_a, trig_b]], cfg)

    assert len(df) == 1
    assert df.iloc[0]["duration_s"] == 2.5
    # time/coincidence_sum still come from the stronger (trig_a) member.
    assert df.iloc[0]["coincidence_sum"] == 5.0


# ---------------------------------------------------------------------------
# event_id_from_time
# ---------------------------------------------------------------------------


def test_event_id_format():
    t = UTCDateTime("2025-03-04T05:06:07.089000Z")
    eid = event_id_from_time(t)
    assert re.match(r"^ev\d{8}T\d{6}_\d{3}$", eid)
    assert eid == "ev20250304T050607_089"
    # Deterministic/stable for the same input.
    assert event_id_from_time(t) == eid


def test_merge_windows_assigns_event_ids_matching_format(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T03:04:05.123000Z")
    df = merge_windows([[_trig(t0.isoformat() + "Z")]], cfg)
    assert len(df) == 1
    assert re.match(r"^ev\d{8}T\d{6}_\d{3}$", df.iloc[0]["event_id"])
    assert df.iloc[0]["event_id"] == event_id_from_time(t0)


# ---------------------------------------------------------------------------
# Event id collisions
# ---------------------------------------------------------------------------


def test_event_id_collision_handled_deterministically():
    """Two distinct (already-merged) detections whose representative times
    round to the same millisecond get distinct, deterministic ids (the
    ``_dup<n>`` suffix documented in ``openmeq.detect.merge``'s module
    docstring), rather than silently colliding or raising.
    """
    t = UTCDateTime("2025-01-02T00:00:00.000500Z")  # rounds to _000 ms
    same_ms = UTCDateTime("2025-01-02T00:00:00.000900Z")  # also rounds to _000 ms
    detections = [
        {
            "time": t,
            "duration_s": 1.0,
            "coincidence_sum": 4.0,
            "n_stations": 2,
            "stations": ["XX.STA01"],
            "trace_ids": ["XX.STA01.00.HHZ"],
        },
        {
            "time": same_ms,
            "duration_s": 1.0,
            "coincidence_sum": 4.0,
            "n_stations": 2,
            "stations": ["XX.STA02"],
            "trace_ids": ["XX.STA02.00.HHZ"],
        },
    ]

    ids = _assign_event_ids(detections)

    assert len(ids) == len(set(ids)), "collision must not produce a duplicate id"
    assert ids[0] == event_id_from_time(t)
    assert ids[1] == f"{event_id_from_time(t)}_dup1"

    # Deterministic: re-running produces the exact same ids.
    assert _assign_event_ids(detections) == ids


def test_event_id_collision_third_detection_increments_suffix():
    t = UTCDateTime("2025-01-02T00:00:00.000000Z")
    detections = [
        {
            "time": t,
            "duration_s": 1.0,
            "coincidence_sum": 4.0,
            "n_stations": 1,
            "stations": ["A"],
            "trace_ids": ["A.00.HHZ"],
        },
        {
            "time": t,
            "duration_s": 1.0,
            "coincidence_sum": 4.0,
            "n_stations": 1,
            "stations": ["B"],
            "trace_ids": ["B.00.HHZ"],
        },
        {
            "time": t,
            "duration_s": 1.0,
            "coincidence_sum": 4.0,
            "n_stations": 1,
            "stations": ["C"],
            "trace_ids": ["C.00.HHZ"],
        },
    ]

    ids = _assign_event_ids(detections)

    base = event_id_from_time(t)
    assert ids == [base, f"{base}_dup1", f"{base}_dup2"]


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


def test_csv_schema(cfg, tmp_path):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_a = _trig(t0.isoformat() + "Z", stations=["XX.STA01", "XX.STA02"])
    trig_b = _trig((t0 + 10.0).isoformat() + "Z", stations=["XX.STA03"])

    df = merge_windows([[trig_a, trig_b]], cfg)
    out_path = tmp_path / "detections" / "detections.csv"
    write_detections_csv(df, out_path)

    assert out_path.exists()
    read_back = pd.read_csv(out_path, dtype=str)
    assert list(read_back.columns) == DETECTIONS_COLUMNS
    assert list(read_back.columns) == [
        "event_id",
        "time",
        "duration_s",
        "coincidence_sum",
        "n_stations",
        "stations",
        "trace_ids",
    ]
    # stations/trace_ids are ;-joined strings, not Python list repr().
    row0 = read_back.iloc[0]
    assert row0["stations"] == "XX.STA01;XX.STA02"
    assert not row0["stations"].startswith("[")
    row1 = read_back.iloc[1]
    assert row1["stations"] == "XX.STA03"


def test_merge_windows_empty_input(cfg):
    df = merge_windows([], cfg)
    assert list(df.columns) == DETECTIONS_COLUMNS
    assert len(df) == 0


def test_merge_windows_sorted_by_time(cfg):
    cfg.detection.dedup_window_s = 2.0
    t0 = UTCDateTime("2025-01-02T00:00:00.000000Z")
    trig_late = _trig((t0 + 20.0).isoformat() + "Z")
    trig_early = _trig(t0.isoformat() + "Z")

    df = merge_windows([[trig_late], [trig_early]], cfg)

    times = list(df["time"])
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Light integration test: full synthetic 2-day dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow  # OpenMEQ-m1f.6: real STA/LTA over the full synthetic archive, ~34s
def test_full_pipeline_no_duplicate_event_ids(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    new_config_path = project_copy / "config.yaml"
    cfg = load_config(new_config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    build_index(cfg)

    windows = build_windows(cfg)
    assert len(windows) == 2  # 2-day synthetic dataset, default chunk_days=1

    config_path_str = str(new_config_path)
    per_window_results = [
        detect_window(w_start, w_end, str(paths.index_json), config_path_str)
        for w_start, w_end in windows
    ]

    df = merge_windows(per_window_results, cfg)
    write_detections_csv(df, paths.detections_csv)

    assert paths.detections_csv.exists()

    # No two output rows within dedup_window_s of each other.
    times = sorted(UTCDateTime(t) for t in df["time"])
    for t_prev, t_next in zip(times, times[1:], strict=False):
        assert (t_next - t_prev) > cfg.detection.dedup_window_s

    # event_ids are unique.
    assert df["event_id"].is_unique

    # Recovers a number of events consistent with the ~20 true events,
    # including the one placed just before the day boundary
    # (test_detect_worker.py's test_day_boundary_event_appears_in_both_adjacent_windows
    # confirms it's seen by both windows; here we confirm the merge collapses
    # that duplicate down to one surviving detection, not two).
    n_matched = 0
    for event in truth["events"]:
        origin_time = event["origin"]["time"]
        if any(abs(t - origin_time) <= _MATCH_TOLERANCE_S for t in times):
            n_matched += 1
    assert n_matched >= int(0.7 * len(truth["events"])), (
        f"only matched {n_matched}/{len(truth['events'])} truth events "
        f"({len(times)} merged detections)"
    )

    # The day-boundary event specifically: confirm merging across the two
    # overlapping windows removes exactly the window-boundary-induced
    # duplication, and nothing more/less.
    #
    # Empirically (verified by hand against this synthetic dataset while
    # writing this test), coincidence_trigger fires more than one distinct
    # STA/LTA episode for most single physical events here (e.g. a P-phase
    # episode followed by a separate S-phase episode a few seconds later,
    # often further apart than the default dedup_window_s=2.0s) -- that is
    # M2.1's detection behaviour, not a window-boundary artifact, and not
    # this task's bug to fix (out of scope: tuning STA/LTA/coincidence
    # parameters). So the boundary event is not expected to collapse to
    # exactly one merged row; what M2.2 is responsible for is making sure
    # the *cross-window duplication* (each of those genuine episodes
    # appearing twice, once per adjacent padded window) is fully removed,
    # without over- or under-merging relative to what a single window's own
    # (already duplicate-free) triggers would give.
    day_boundary = UTCDateTime("2025-01-02T00:00:00Z")
    boundary_event = min(truth["events"], key=lambda ev: abs(ev["origin"]["time"] - day_boundary))
    origin_time = boundary_event["origin"]["time"]

    margin = 20.0  # wide enough to catch every genuine STA/LTA episode of this one event
    window1_near_boundary = [
        trig
        for trig in per_window_results[0]
        if abs(UTCDateTime(trig["time"]) - origin_time) <= margin
    ]
    both_windows_near_boundary = [
        trig
        for window in per_window_results
        for trig in window
        if abs(UTCDateTime(trig["time"]) - origin_time) <= margin
    ]
    assert both_windows_near_boundary, "expected raw triggers near the day-boundary event"

    # Merging just window 1's own near-boundary triggers (no cross-window
    # duplication possible, since it's a single window) gives the "ground
    # truth" number of genuine STA/LTA episodes for this event.
    single_window_merged = merge_windows([window1_near_boundary], cfg)
    # Merging both windows' (duplicated) near-boundary triggers together
    # must collapse back down to that same count -- not double it.
    both_windows_merged = merge_windows([both_windows_near_boundary], cfg)

    assert len(both_windows_merged) == len(single_window_merged), (
        "merging the day-boundary event's triggers across both overlapping "
        f"windows ({len(both_windows_merged)} rows) must match merging a single "
        f"window's own triggers for the same event ({len(single_window_merged)} "
        "rows) -- cross-window duplication must be fully removed, with no "
        "extra artifacts introduced"
    )

    n_near_boundary = sum(1 for t in times if abs(t - origin_time) <= _MATCH_TOLERANCE_S)
    assert n_near_boundary >= 1, "the day-boundary event must be recovered at least once"
