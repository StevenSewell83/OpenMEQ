"""Tests for ``openmeq.detect.sweep`` (OpenMEQ-cl6).

The sweep's triggering and coincidence pieces must reproduce ObsPy's
``trigger_onset`` / ``coincidence_trigger`` exactly, and a sweep grid point
equal to ``cfg.detection`` must give the same detections as
``openmeq.detect.worker.detect_window``.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import coincidence_trigger, trigger_onset

from openmeq.config import load_config
from openmeq.detect.merge import _cluster_triggers, _representative
from openmeq.detect.sweep import (
    choose_days,
    coincidence_candidates,
    compress_cf,
    dedup_times,
    fast_trigger_onset,
    knee_index,
    match_count,
    run_sweep,
    select_events,
)
from openmeq.detect.worker import detect_window
from openmeq.index import build_index
from openmeq.paths import RunPaths

_BIG = 10**9


def _fake_cf(rng, n: int) -> np.ndarray:
    """Smooth positive noise around 1 with occasional bursts, like a recursive STA/LTA."""
    x = np.abs(np.convolve(rng.standard_normal(n), np.ones(25) / 5.0, mode="same")) + 0.3
    for c in rng.integers(0, n, size=n // 400):
        w = int(rng.integers(5, 80))
        x[c : c + w] += rng.uniform(1.0, 9.0)
    return x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_trigger_onset_matches_obspy(seed):
    rng = np.random.default_rng(seed)
    cft = _fake_cf(rng, 20_000)
    pairs = [(on, off) for on in (2.0, 3.0, 4.5, 7.0) for off in (0.8, 1.0, 1.5, 2.5) if off < on]
    comp, pos = compress_cf(cft, min(p[1] for p in pairs), min(p[0] for p in pairs))
    assert len(comp) < len(cft)
    for on, off in pairs:
        expected = np.asarray(trigger_onset(cft, on, off, max_len=_BIG, max_len_delete=False))
        got = fast_trigger_onset(comp, pos, on, off, _BIG)
        np.testing.assert_array_equal(got.reshape(-1, 2), expected.reshape(-1, 2))


def test_fast_trigger_onset_trace_edges():
    cft = np.array([5.0, 5.0, 1.2, 0.5, 0.5, 3.0, 0.2, 0.2, 4.0, 4.0])
    comp, pos = compress_cf(cft, 1.0, 2.0)
    expected = np.asarray(trigger_onset(cft, 2.0, 1.0, max_len=_BIG, max_len_delete=False))
    np.testing.assert_array_equal(fast_trigger_onset(comp, pos, 2.0, 1.0, _BIG), expected)


def _cf_stream(rng, n_sta: int, n: int, sr: float = 50.0) -> Stream:
    """CF traces sharing a few common bursts (events) plus independent noise."""
    t0 = UTCDateTime(2025, 1, 1)
    events = rng.integers(500, n - 500, size=12)
    st = Stream()
    for s in range(n_sta):
        for cha in ("HHZ", "HHN"):
            x = _fake_cf(rng, n)
            for e in events:
                if rng.random() < 0.8:
                    lag = int(rng.integers(0, 60))
                    x[e + lag : e + lag + 40] += 6.0
            st.append(
                Trace(
                    x,
                    header={
                        "network": "XX",
                        "station": f"S{s:02d}",
                        "channel": cha,
                        "sampling_rate": sr,
                        "starttime": t0,
                    },
                )
            )
    return st


@pytest.mark.parametrize("weighted", [False, True])
def test_coincidence_matches_obspy(weighted):
    rng = np.random.default_rng(7)
    st = _cf_stream(rng, 6, 15_000)
    weights = (
        {tr.id: (0.5 if tr.stats.channel == "HHN" else 1.0) for tr in st} if weighted else None
    )
    for on, off, thr in [(3.0, 1.0, 3), (4.0, 1.5, 5), (6.0, 2.0, 4)]:
        trigs = []
        for tr in st:
            sr = tr.stats.sampling_rate
            for a, b in trigger_onset(tr.data, on, off, max_len=_BIG, max_len_delete=False):
                trigs.append(
                    (
                        tr.stats.starttime.timestamp + a / sr,
                        tr.stats.starttime.timestamp + b / sr,
                        tr.id,
                    )
                )
        trigs.sort()
        got = select_events(coincidence_candidates(trigs, weights), thr, 1)
        expected = coincidence_trigger(
            None, on, off, st, thr, trace_ids=weights, max_trigger_length=1e6, details=False
        )
        assert [(e[0], e[1]) for e in got] == [
            (ev["time"].timestamp, ev["coincidence_sum"]) for ev in expected
        ]
        for m in (2, 4):
            n_sta = [len({".".join(t.split(".")[:2]) for t in ev["trace_ids"]}) for ev in expected]
            assert len(select_events(coincidence_candidates(trigs, weights), thr, m)) == sum(
                n >= m for n in n_sta
            )


def test_dedup_times_matches_merge_clustering():
    rng = np.random.default_rng(3)
    t0 = UTCDateTime(2025, 1, 1).timestamp
    times = np.sort(t0 + np.cumsum(rng.exponential(6.0, size=300)))
    events = [(float(t), float(rng.integers(3, 12)), 3) for t in times]
    triggers = [{"time": str(UTCDateTime(t)), "coincidence_sum": c} for t, c, _ in events]
    clusters = _cluster_triggers(triggers, 10.0)
    expected = sorted(_representative(c)[1].timestamp for c in clusters)
    np.testing.assert_allclose(dedup_times(events, 10.0), expected, atol=1e-6)


def test_match_count_one_to_one():
    ref = np.array([10.0, 11.0, 50.0])
    det = np.array([12.0, 30.0, 51.0, 52.0])
    assert match_count(det, ref, (-2.0, 5.0)) == 2  # 12 -> 10, 51 -> 50; 11 has none left
    assert match_count(det, ref, (-1.0, 0.5)) == 0


def test_knee_index_finds_corner():
    x = np.log10(np.array([2, 3, 4, 5, 6, 8, 10, 12.0]))
    y = np.log10(np.array([1e5, 1e4, 1e3, 120, 100, 90, 80, 75.0]))
    i, d = knee_index(x, y)
    assert i == 3 and d > 0.3
    assert knee_index(x[:2], y[:2]) == (-1, 0.0)


def test_choose_days_prefers_full_days():
    counts = pd.Series(
        {"2025-01-01": 2, "2025-01-02": 10, "2025-01-03": 9, "2025-01-04": 10, "2025-01-05": 10}
    )
    assert choose_days(counts, 2) == ["2025-01-02", "2025-01-05"]
    assert choose_days(counts, 10) == ["2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]


@pytest.mark.slow
def test_sweep_point_equals_detect_window(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    project = tmp_path / "project"
    shutil.copytree(config_path.parent, project)
    cfg_path = project / "config.yaml"
    cfg = load_config(cfg_path)
    det = cfg.detection
    assert det.filter.type == "bandpass"
    day = UTCDateTime(cfg.data.start)

    ref_csv = tmp_path / "ref.csv"
    pd.DataFrame({"time": [str(ev["origin"]["time"]) for ev in truth["events"]]}).to_csv(
        ref_csv, index=False
    )
    s = cfg.detection.sweep
    s.days = [day.date]
    s.filters = [(det.filter.freqmin, det.filter.freqmax)]
    s.sta = [det.sta]
    s.lta_factor = [det.lta / det.sta, 2.0 * det.lta / det.sta]
    s.trigger_on = [det.trigger_on, det.trigger_on * 1.5, det.trigger_on * 2.5]
    s.trigger_off = [det.trigger_off]
    s.min_stations = [max(det.min_stations, 1), 4]
    s.trace_coincidence_per_station = det.min_trace_coincidence / max(det.min_stations, 1)
    s.reference = ref_csv
    s.n_workers = 1

    paths = RunPaths(cfg)
    paths.ensure()
    build_index(cfg)
    out = run_sweep(cfg, paths, str(cfg_path))

    points = pd.read_csv(out / "sweep.csv")
    assert {"det_per_day", "recall", "extras_per_day", "f1", "persist_frac"} <= set(points.columns)
    for name in ("knees.csv", "candidates.csv", "items.csv"):
        assert (out / name).is_file()
    assert list(out.glob("lcurve_*.png")) and list(out.glob("recall_*.png"))
    assert not paths.detections_csv.exists()

    # The grid point equal to cfg.detection == detect_window on that day
    # (on-times inside the day, deduplicated as merge_windows does).
    trigs = detect_window(str(day), str(day + 86400), str(paths.index_json), str(cfg_path))
    trigs = [t for t in trigs if day <= UTCDateTime(t["time"]) < day + 86400]
    trigs.sort(key=lambda t: UTCDateTime(t["time"]))
    expected = len(_cluster_triggers(trigs, det.dedup_window_s))
    row = points[
        np.isclose(points["lta"], det.lta)
        & np.isclose(points["trigger_on"], det.trigger_on)
        & (points["min_stations"] == max(det.min_stations, 1))
    ]
    assert len(row) == 1
    assert int(row["n_det"].iloc[0]) == expected
    assert expected > 0
    # higher thresholds never detect more
    cur = points[np.isclose(points["lta"], det.lta) & (points["min_stations"] == 4)].sort_values(
        "trigger_on"
    )
    assert cur["n_det"].is_monotonic_decreasing
