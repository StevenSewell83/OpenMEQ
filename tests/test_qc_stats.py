"""Tests for ``openmeq.qc.stats`` (M1.4, ``OpenMEQ-5nl.4``).

Uses the session-scoped ``synthetic_project`` fixture (``tests/conftest.py``):
2 days at 200 Hz, with ``truth["known_issues"]`` pinning one gap
(``XX.STA05.00.HHZ``), one spiky channel (``XX.STA06.00.HHE``, all of day 2)
and one flat-lined channel (``XX.STA03.00.HHN``, 12:00-13:00 on day 1).

Building the index and computing daily stats for every channel/day is
read-only against the shared session fixture's waveform files (only
``inventory/index.json`` under the fixture's ``runs/synthetic/`` gets
(re)written, deterministically -- same pattern as ``tests/test_index.py``),
so it's safe to share a module-scoped fixture across every test here without
copying the project directory (see ``tests/conftest.py``'s
``synthetic_project`` docstring: copying is only required for tests that
mutate/consume state other tests care about).
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from openmeq.config import QcConfig, load_config
from openmeq.index import build_index
from openmeq.paths import RunPaths
from openmeq.qc.stats import (
    DAILY_STATS_COLUMNS,
    _day_window,
    _write_qc_worker_cfg,
    compute_channel_day_stats,
    compute_daily_stats,
    stats_for_channel_day,
    write_daily_stats_csv,
)

DAY1 = date(2025, 1, 1)
DAY2 = date(2025, 1, 2)


def _issue(truth: dict, kind: str) -> dict:
    (issue,) = [i for i in truth["known_issues"] if i["kind"] == kind]
    return issue


@pytest.fixture(scope="module")
def stats_world(synthetic_project):
    """``(df, truth, cfg, idx)`` -- daily stats for every channel over both days.

    Computed once per test module (indexing + reading ~48 channel-days),
    reused read-only by every test below.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)

    seed_ids = idx.seed_ids()
    dates = [DAY1, DAY2]

    t0 = time.perf_counter()
    df = compute_daily_stats(idx, seed_ids, dates, cfg)
    elapsed = time.perf_counter() - t0

    n = len(seed_ids) * len(dates)
    print(
        f"\n[qc.stats throughput] {n} channel-days in {elapsed:.2f}s "
        f"({n / elapsed:.1f} channel-days/s, n_workers={cfg.parallel.n_workers})"
    )

    return df, truth, cfg, idx


def _row(df, seed_id: str, day: date):
    match = df[(df["seed_id"] == seed_id) & (df["date"] == day.isoformat())]
    assert len(match) == 1, f"expected exactly one row for {seed_id}/{day}, got {len(match)}"
    return match.iloc[0]


# ---------------------------------------------------------------------------
# rms / mean_abs
# ---------------------------------------------------------------------------


def test_rms_mean_abs_matches_manual_calc(stats_world):
    """An ordinary (no known-issue) channel/day: rms/mean_abs match a direct
    NumPy calculation on the same samples read straight from the index."""
    df, truth, cfg, idx = stats_world

    known_seed_ids = {issue["seed_id"] for issue in truth["known_issues"]}
    ordinary_seed_id = "XX.STA01.00.HHZ"
    assert ordinary_seed_id not in known_seed_ids

    day_start, day_end = _day_window(DAY1)
    stream = idx.read([ordinary_seed_id], day_start, day_end)
    assert len(stream) == 1  # no gap on this channel -> single contiguous trace
    x = stream[0].data.astype(np.float64)

    expected_rms = float(np.sqrt(np.mean(x**2)))
    expected_mean_abs = float(np.mean(np.abs(x)))

    row = _row(df, ordinary_seed_id, DAY1)
    assert row["rms"] == pytest.approx(expected_rms, rel=1e-12)
    assert row["mean_abs"] == pytest.approx(expected_mean_abs, rel=1e-12)
    assert int(row["npts"]) == x.size


def test_rms_mean_abs_via_compute_channel_day_stats_direct(synthetic_project):
    """Same check, calling ``compute_channel_day_stats`` directly (bypassing
    the index/pmap plumbing) against a hand-built, gap-free Stream."""
    rng = np.random.default_rng(0)
    data = rng.normal(scale=10.0, size=200 * 3600).astype(np.int32)  # 1 hour @ 200 Hz
    tr = Trace(data=data)
    tr.stats.sampling_rate = 200.0
    tr.stats.starttime = UTCDateTime(2025, 1, 1)
    stream = Stream([tr])

    result = compute_channel_day_stats("XX.TEST.00.HHZ", DAY1, stream, QcConfig())

    x = data.astype(np.float64)
    assert result["rms"] == pytest.approx(float(np.sqrt(np.mean(x**2))), rel=1e-12)
    assert result["mean_abs"] == pytest.approx(float(np.mean(np.abs(x))), rel=1e-12)
    assert result["npts"] == data.size
    assert result["clip_count"] == 0
    assert result["flat_count"] == 0


# ---------------------------------------------------------------------------
# spikes
# ---------------------------------------------------------------------------


def test_known_spiky_channel_flagged_by_stats(stats_world):
    """The known spiky channel/day stands out by spike_count, and both it and
    every other channel measure as heavy-tailed (kurtosis >> 0).

    **Verified fact for M1.5** (real quake pulses in this fixture reach
    hundreds to low thousands of counts against a background noise half
    range of only 5 counts -- e.g. ~2265 counts peak on an *ordinary*
    channel/day): this makes raw per-day kurtosis, on its own, an unreliable
    cross-*channel* discriminator in this dataset -- a channel/day with a
    single very large, very close event pulse can out-rank the deliberately
    spiky channel/day on kurtosis alone (confirmed: ``XX.BH02.00.DP2``'s
    2025-01-02 kurtosis exceeds the known spiky channel's that same day).
    ``spike_count`` (MAD-threshold impulse counting) is the more robust
    discriminator here; M1.5's monthly comparison / flagging logic should
    weigh multiple statistics together rather than trust kurtosis in
    isolation. ``test_kurtosis_and_spike_count_rise_with_injected_spikes``
    below isolates the effect on a controlled trace with no real-event
    confound, showing both statistics behave correctly in the absence of
    that confound.
    """
    df, truth, cfg, idx = stats_world
    issue = _issue(truth, "spikes")
    assert issue["seed_id"] == "XX.STA06.00.HHE"
    day = issue["start"].date  # obspy.UTCDateTime.date -> datetime.date
    assert day == DAY2

    day_rows = df[df["date"] == day.isoformat()]
    spiky_row = _row(df, issue["seed_id"], day)
    other_rows = day_rows[day_rows["seed_id"] != issue["seed_id"]]
    baseline_row = _row(df, issue["seed_id"], DAY1)  # same channel, day before the anomaly

    assert spiky_row["spike_count"] > 0
    # the single highest spike_count of any channel that day ...
    assert spiky_row["spike_count"] == day_rows["spike_count"].max()
    # ... and comfortably above the day's upper quartile, not just barely
    # nosing out the next-highest ordinary channel
    assert spiky_row["spike_count"] > 1.1 * other_rows["spike_count"].quantile(0.75)
    # ... and clearly attributable to the injected anomaly: well above this
    # same channel's own spike_count the day before (no anomaly that day)
    assert spiky_row["spike_count"] > baseline_row["spike_count"] + 200

    assert spiky_row["kurtosis"] > 50.0  # clearly heavy-tailed vs. ~0 for Gaussian/uniform noise
    assert (
        other_rows["kurtosis"] > 50.0
    ).all()  # true of every channel in this fixture (see above)


def test_kurtosis_and_spike_count_rise_with_injected_spikes():
    """Controlled check (no real-event confound): injecting a handful of
    large-amplitude impulses into otherwise plain background noise clearly
    raises both spike_count and kurtosis, isolating the effect that
    ``test_known_spiky_channel_flagged_by_stats`` above shows is muddied by
    real quake pulses in the full synthetic fixture."""
    rng = np.random.default_rng(3)
    n = 200 * 3600  # 1 hour @ 200 Hz
    clean = rng.uniform(-5.0, 5.0, size=n)

    spiky = clean.copy()
    idx = rng.choice(n, size=200, replace=False)
    spiky[idx] = rng.choice([-1.0, 1.0], size=200) * rng.uniform(80.0, 150.0, size=200) * 5.0

    def _stats(x):
        tr = Trace(data=np.round(x).astype(np.int32))
        tr.stats.sampling_rate = 200.0
        tr.stats.starttime = UTCDateTime(2025, 1, 1)
        return compute_channel_day_stats("XX.CTRL.00.HHZ", DAY1, Stream([tr]), QcConfig())

    clean_result = _stats(clean)
    spiky_result = _stats(spiky)

    assert spiky_result["spike_count"] > clean_result["spike_count"]
    assert spiky_result["spike_count"] >= 150  # most of the 200 injected impulses recovered
    assert spiky_result["kurtosis"] > clean_result["kurtosis"]
    assert spiky_result["kurtosis"] > 10.0
    assert clean_result["kurtosis"] < 5.0  # uniform noise alone is not heavy-tailed


# ---------------------------------------------------------------------------
# flat
# ---------------------------------------------------------------------------


def test_known_flat_channel_flagged_by_stats(stats_world):
    df, truth, cfg, idx = stats_world
    issue = _issue(truth, "flat")
    assert issue["seed_id"] == "XX.STA03.00.HHN"
    day = issue["start"].date
    assert day == DAY1

    day_rows = df[df["date"] == day.isoformat()]
    flat_row = _row(df, issue["seed_id"], day)
    other_rows = day_rows[day_rows["seed_id"] != issue["seed_id"]]

    window_samples = int(round((issue["end"] - issue["start"]) * 200.0))  # 1h @ 200 Hz
    assert flat_row["flat_count"] > 0
    # virtually the entire known anomaly window recovered as one flat run
    assert flat_row["flat_count"] == pytest.approx(window_samples, rel=0.01)
    # far above any ordinary channel that day (background noise essentially
    # never repeats the same raw sample value >= flat_run_min_samples times)
    assert flat_row["flat_count"] > 100 * (other_rows["flat_count"].max() + 1)


def test_flat_window_is_almost_entirely_one_run(synthetic_project):
    """Reading exactly the known flat window: flat_count is close to npts."""
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)
    issue = _issue(truth, "flat")

    stream = idx.read([issue["seed_id"]], issue["start"], issue["end"])
    result = compute_channel_day_stats(issue["seed_id"], issue["start"].date, stream, cfg.qc)

    assert result["npts"] > 0
    assert result["flat_count"] / result["npts"] > 0.99


# ---------------------------------------------------------------------------
# clipping
# ---------------------------------------------------------------------------


def test_clip_detection_int32_saturation():
    qc_cfg = QcConfig()  # clip_run_min_samples default 3
    rng = np.random.default_rng(1)

    data = rng.integers(-1000, 1000, size=1000).astype(np.int32)
    run_len = qc_cfg.clip_run_min_samples + 3
    data[500 : 500 + run_len] = 2**31 - 1  # saturated positive run

    tr = Trace(data=data)
    tr.stats.sampling_rate = 200.0
    tr.stats.starttime = UTCDateTime(2025, 1, 1)
    stream = Stream([tr])

    result = compute_channel_day_stats("XX.CLIP.00.HHZ", DAY1, stream, qc_cfg)
    assert result["clip_count"] == run_len

    # a normal-amplitude trace (no saturation run) yields clip_count == 0
    normal_data = rng.integers(-1000, 1000, size=1000).astype(np.int32)
    normal_tr = Trace(data=normal_data)
    normal_tr.stats.sampling_rate = 200.0
    normal_tr.stats.starttime = UTCDateTime(2025, 1, 1)
    normal_result = compute_channel_day_stats("XX.CLIP.00.HHZ", DAY1, Stream([normal_tr]), qc_cfg)
    assert normal_result["clip_count"] == 0


def test_clip_run_shorter_than_threshold_not_counted():
    qc_cfg = QcConfig()
    rng = np.random.default_rng(2)
    data = rng.integers(-1000, 1000, size=1000).astype(np.int32)
    data[10 : 10 + qc_cfg.clip_run_min_samples - 1] = 2**31 - 1  # one sample short

    tr = Trace(data=data)
    tr.stats.sampling_rate = 200.0
    tr.stats.starttime = UTCDateTime(2025, 1, 1)
    result = compute_channel_day_stats("XX.CLIP.00.HHZ", DAY1, Stream([tr]), qc_cfg)
    assert result["clip_count"] == 0


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------


def test_gap_day_does_not_crash(stats_world):
    df, truth, cfg, idx = stats_world
    issue = _issue(truth, "gap")
    assert issue["seed_id"] == "XX.STA05.00.HHZ"
    day = issue["start"].date
    assert day == DAY1

    row = _row(df, issue["seed_id"], day)
    full_day_samples = int(round(86400 * 200.0))
    gap_samples = int(round((issue["end"] - issue["start"]) * 200.0))

    assert 0 < row["npts"] < full_day_samples
    assert row["npts"] == pytest.approx(full_day_samples - gap_samples, abs=5)
    assert np.isfinite(row["rms"])
    assert np.isfinite(row["mean_abs"])


def test_stats_for_channel_day_direct_worker_call(synthetic_project, tmp_path):
    """Exercise the exact top-level worker signature/contract used by pmap."""
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    build_index(cfg)

    index_path = RunPaths(cfg).index_json
    cfg_path = _write_qc_worker_cfg(cfg.qc)
    try:
        result = stats_for_channel_day("XX.STA01.00.HHZ", DAY1, str(index_path), str(cfg_path))
    finally:
        cfg_path.unlink(missing_ok=True)

    assert result["seed_id"] == "XX.STA01.00.HHZ"
    assert result["date"] == DAY1.isoformat()
    assert result["npts"] > 0


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


def test_csv_schema(stats_world, tmp_path):
    df, truth, cfg, idx = stats_world
    out_path = tmp_path / "daily_stats.csv"
    write_daily_stats_csv(df, out_path)

    assert out_path.exists()
    text = out_path.read_text()
    header = text.splitlines()[0]
    assert header == ",".join(DAILY_STATS_COLUMNS)

    import pandas as pd

    reread = pd.read_csv(out_path)
    assert list(reread.columns) == DAILY_STATS_COLUMNS
    assert len(reread) == len(df)


def test_empty_channel_day_no_crash():
    """A channel/day with zero valid samples (empty Stream) returns zeros, not a crash."""
    result = compute_channel_day_stats("XX.NODATA.00.HHZ", DAY1, Stream(), QcConfig())
    assert result["npts"] == 0
    assert result["rms"] == 0.0
    assert result["mean_abs"] == 0.0
    assert result["spike_count"] == 0
    assert result["flat_count"] == 0
    assert result["clip_count"] == 0
    assert np.isnan(result["kurtosis"])


def test_day_window_excludes_next_day_first_sample():
    start, end = _day_window(DAY1)
    assert start == UTCDateTime(2025, 1, 1, 0, 0, 0)
    assert end < UTCDateTime(2025, 1, 2, 0, 0, 0)
    assert end > UTCDateTime(2025, 1, 1, 23, 59, 59)


def test_dates_helper_two_days():
    # sanity: DAY1/DAY2 really are one calendar day apart (used throughout)
    assert DAY2 - DAY1 == timedelta(days=1)
