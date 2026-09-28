"""Tests for ``openmeq.detect.worker`` (M2.1, ``OpenMEQ-ujx.1``).

Per the orchestrator note on this issue: these tests hand-build
``qc/channel_flags.json`` files rather than running the real QC step
(``openmeq.qc``), so they never depend on which synthetic channels QC itself
would flag (that's being changed concurrently by another task). Every test
below builds its own minimal, explicit flags dict via :func:`_all_ok_flags`
(and mutates it for the exclusion test), covering exactly the seed ids/dates
the index actually has.

Every test that runs a pipeline step (builds an index, writes flags,
produces output) works from a **function-scoped** copy of the shared,
session-scoped ``synthetic_project`` fixture's directory (see
``tests/conftest.py``'s usage-pattern docstring) -- never the shared
directory itself.

OpenMEQ-m1f.6: the four tests below that run a real STA/LTA
``detect_window`` pass over the whole synthetic archive (16-32s each on
this machine -- the per-test ``shutil.copytree`` this module's own
``_setup_project`` helper does is not the expensive part, confirmed
directly: ``test_build_windows_spans_config_range`` uses the same helper
and runs in well under a second) are marked ``slow`` so they don't count
against the ``-m "not slow"`` budget.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from obspy import UTCDateTime
from obspy.signal import trigger as obspy_trigger_module

from openmeq.config import load_config
from openmeq.detect.worker import build_windows, detect_window
from openmeq.index import DataIndex, build_index
from openmeq.paths import RunPaths

# Generous match tolerance: truth origin time + up to this many seconds of
# travel time to the nearest station, plus a little slack for STA/LTA
# on-trigger lag. The synthetic network's aperture is under 15 km with
# vp/vs of a few km/s, so a nearby station's P arrival is at most a couple
# of seconds after the origin; this is deliberately generous rather than
# tightly tuned (see ``tests/data/make_synthetic.py``'s own documented
# travel-time tolerances).
_MATCH_TOLERANCE_S = 12.0


def _setup_project(synthetic_project, tmp_path: Path):
    """Copy the shared synthetic project into ``tmp_path`` and build its index.

    Returns ``(cfg, paths, truth, config_path)``.
    """
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    new_config_path = project_copy / "config.yaml"
    cfg = load_config(new_config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    build_index(cfg)
    return cfg, paths, truth, new_config_path


def _all_ok_flags(index: DataIndex, dates: list[str]) -> dict:
    """A ``channel_flags.json``-shaped dict marking every seed id ``ok`` on every date.

    Hand-built per the orchestrator note (see module docstring) -- this
    never depends on QC's own flagging logic/defaults.
    """
    flags = {
        seed_id: {d: {"ok": True, "reasons": []} for d in dates} for seed_id in index.seed_ids()
    }
    return {"version": 1, "flags": flags}


def _write_flags(paths: RunPaths, flags_dict: dict) -> Path:
    paths.channel_flags_json.parent.mkdir(parents=True, exist_ok=True)
    paths.channel_flags_json.write_text(json.dumps(flags_dict))
    return paths.channel_flags_json


@pytest.mark.slow
def test_recovers_synthetic_events_in_window(synthetic_project, tmp_path):
    cfg, paths, truth, config_path = _setup_project(synthetic_project, tmp_path)
    index = DataIndex.load(paths.index_json)
    dates = ["2025-01-01", "2025-01-02"]
    flags_path = _write_flags(paths, _all_ok_flags(index, dates))

    windows = build_windows(cfg)
    assert len(windows) == 2

    all_triggers = []
    for window_start, window_end in windows:
        all_triggers.extend(
            detect_window(
                window_start, window_end, str(paths.index_json), str(flags_path), str(config_path)
            )
        )

    assert len(all_triggers) > 0

    trigger_times = sorted(UTCDateTime(t["time"]) for t in all_triggers)

    n_matched = 0
    for event in truth["events"]:
        origin_time = event["origin"]["time"]
        if any(abs(tt - origin_time) <= _MATCH_TOLERANCE_S for tt in trigger_times):
            n_matched += 1

    # "yields triggers whose times are within a few seconds of the true
    # origin times for most of them" (acceptance criterion) -- not
    # necessarily all 20 (a low-ML event at a far station may not cross
    # threshold), so require recovery of a clear majority.
    assert n_matched >= int(0.7 * len(truth["events"])), (
        f"only matched {n_matched}/{len(truth['events'])} truth events "
        f"({len(all_triggers)} raw triggers)"
    )


@pytest.mark.slow
def test_flagged_channel_excluded(synthetic_project, tmp_path, monkeypatch):
    cfg, paths, truth, config_path = _setup_project(synthetic_project, tmp_path)
    index = DataIndex.load(paths.index_json)
    dates = ["2025-01-01", "2025-01-02"]
    flags_dict = _all_ok_flags(index, dates)

    excluded_station = "XX.STA01"
    for seed_id in list(flags_dict["flags"]):
        if seed_id.startswith(f"{excluded_station}."):
            for d in dates:
                flags_dict["flags"][seed_id][d] = {"ok": False, "reasons": ["test-excluded"]}

    flags_path = _write_flags(paths, flags_dict)
    windows = build_windows(cfg)

    captured_streams = []
    real_coincidence_trigger = obspy_trigger_module.coincidence_trigger

    def _spy(*args, **kwargs):
        # coincidence_trigger's positional signature is
        # (trigger_type, thr_on, thr_off, stream, thr_coincidence_sum, ...);
        # detect_window calls it positionally, so the stream is args[3].
        captured_streams.append(args[3])
        return real_coincidence_trigger(*args, **kwargs)

    monkeypatch.setattr("openmeq.detect.worker.coincidence_trigger", _spy)

    for window_start, window_end in windows:
        detect_window(
            window_start, window_end, str(paths.index_json), str(flags_path), str(config_path)
        )

    assert captured_streams, "coincidence_trigger was never called"
    for stream in captured_streams:
        stations_present = {f"{tr.stats.network}.{tr.stats.station}" for tr in stream}
        assert excluded_station not in stations_present


@pytest.mark.slow
def test_gap_day_does_not_crash(synthetic_project, tmp_path):
    cfg, paths, truth, config_path = _setup_project(synthetic_project, tmp_path)
    index = DataIndex.load(paths.index_json)
    dates = ["2025-01-01", "2025-01-02"]
    flags_path = _write_flags(paths, _all_ok_flags(index, dates))

    gap_issue = next(i for i in truth["known_issues"] if i["kind"] == "gap")
    gap_date = gap_issue["start"].date.isoformat()
    assert gap_date == "2025-01-01"

    windows = build_windows(cfg)
    gap_window = next(
        (w for w in windows if UTCDateTime(w[0]).date.isoformat() == gap_date),
        windows[0],
    )

    triggers = detect_window(
        gap_window[0], gap_window[1], str(paths.index_json), str(flags_path), str(config_path)
    )
    assert isinstance(triggers, list)
    # Other channels/segments are unaffected by the one channel's gap, so
    # detection over a whole day still finds triggers.
    assert len(triggers) > 0


@pytest.mark.slow
def test_day_boundary_event_appears_in_both_adjacent_windows(synthetic_project, tmp_path):
    cfg, paths, truth, config_path = _setup_project(synthetic_project, tmp_path)
    index = DataIndex.load(paths.index_json)
    dates = ["2025-01-01", "2025-01-02"]
    flags_path = _write_flags(paths, _all_ok_flags(index, dates))

    day_boundary = UTCDateTime("2025-01-02T00:00:00Z")
    boundary_event = min(truth["events"], key=lambda ev: abs(ev["origin"]["time"] - day_boundary))
    assert abs(boundary_event["origin"]["time"] - day_boundary) < 10.0, (
        "expected make_synthetic.py's deliberately-placed boundary event within "
        "a few seconds of the day boundary"
    )

    windows = build_windows(cfg)
    day1_window = next(w for w in windows if UTCDateTime(w[0]).date.isoformat() == "2025-01-01")
    day2_window = next(w for w in windows if UTCDateTime(w[0]).date.isoformat() == "2025-01-02")

    origin_time = boundary_event["origin"]["time"]
    for window_start, window_end in (day1_window, day2_window):
        triggers = detect_window(
            window_start, window_end, str(paths.index_json), str(flags_path), str(config_path)
        )
        trigger_times = [UTCDateTime(t["time"]) for t in triggers]
        assert any(abs(tt - origin_time) <= _MATCH_TOLERANCE_S for tt in trigger_times), (
            f"window {window_start}..{window_end} has no trigger near the boundary "
            f"event's origin time {origin_time}"
        )


def test_build_windows_spans_config_range(synthetic_project, tmp_path):
    cfg, _paths, _truth, config_path = _setup_project(synthetic_project, tmp_path)

    windows = build_windows(cfg)
    expected_start = UTCDateTime(cfg.data.start)
    expected_end = UTCDateTime(cfg.data.end) + 86400.0

    assert UTCDateTime(windows[0][0]) == expected_start
    assert UTCDateTime(windows[-1][1]) == expected_end
    for (_s1, e1), (s2, _e2) in zip(windows, windows[1:], strict=False):
        assert e1 == s2  # contiguous, no gap/overlap

    # chunk_days > 1: an uneven-length range still tiles with no gap/overlap,
    # last window shorter.
    cfg.parallel.chunk_days = 2
    cfg.data.start = expected_start.date
    cfg.data.end = (expected_start + 3 * 86400.0).date  # 4 calendar days total
    windows2 = build_windows(cfg)
    assert len(windows2) == 2
    assert UTCDateTime(windows2[0][0]) == expected_start
    assert UTCDateTime(windows2[0][1]) == expected_start + 2 * 86400.0
    assert UTCDateTime(windows2[1][0]) == expected_start + 2 * 86400.0
    assert UTCDateTime(windows2[1][1]) == expected_start + 4 * 86400.0
