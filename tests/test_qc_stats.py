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
    _fast_kurtosis,
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


# ---------------------------------------------------------------------------
# OpenMEQ-5nl.8 (throughput/parallel-scaling optimisation): regression test
# against a from-scratch, deliberately naive reference implementation, so the
# optimised code in openmeq/qc/stats.py can keep changing without this test
# ever importing (or trusting) the same fast helpers it's meant to check.
# ---------------------------------------------------------------------------


def _reference_channel_day_stats(seed_id: str, day, stream, qc_cfg: QcConfig) -> dict:
    """Deliberately naive reference implementation.

    Independent of ``openmeq.qc.stats``'s optimised helpers (does not call
    ``_median_mad_spike_count``, ``_fast_kurtosis`` or the boundary-based
    ``_segment_run_counts``): plain ``scipy.stats.kurtosis``, ``np.median``
    with no ``overwrite_input``, ``np.abs(x - median)`` recomputed from
    scratch for the spike comparison, ``np.sum(x**2)`` (not ``np.dot``),
    and a textbook per-sample run-length encoding (cumulative-sum run ids,
    ``np.bincount``, then a full per-sample gather) for flat/clip runs.
    Slower, by design -- this is the "obviously correct by inspection"
    baseline OpenMEQ-5nl.8 optimised the real implementation against.
    """
    from scipy.stats import kurtosis as scipy_kurtosis

    segments = [tr.data for tr in stream.split()] if len(stream) > 0 else []
    segments = [seg for seg in segments if seg.size > 0]
    if not segments:
        return {
            "seed_id": seed_id,
            "date": day.isoformat(),
            "rms": 0.0,
            "mean_abs": 0.0,
            "spike_count": 0,
            "kurtosis": float("nan"),
            "flat_count": 0,
            "clip_count": 0,
            "npts": 0,
        }

    is_int = segments[0].dtype.kind == "i"
    int32_max = 2**31 - 1
    int32_min = -(2**31)

    float_saturation_value = None
    if not is_int:
        concat_raw = np.concatenate(segments)
        if concat_raw.size > 0:
            float_saturation_value = float(concat_raw[np.argmax(np.abs(concat_raw))])

    def run_lengths_per_sample(x):
        n = x.size
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        is_new_run = np.empty(n, dtype=bool)
        is_new_run[0] = True
        is_new_run[1:] = x[1:] != x[:-1]
        run_id = np.cumsum(is_new_run) - 1
        run_len = np.bincount(run_id)
        return run_len[run_id]

    total_npts = 0
    sum_sq = 0.0
    sum_abs = 0.0
    flat_count = 0
    clip_count = 0
    float_segments = []

    for seg in segments:
        total_npts += seg.size

        run_len = run_lengths_per_sample(seg)
        flat_count += int(np.count_nonzero(run_len >= qc_cfg.flat_run_min_samples))
        if is_int:
            is_sat = (seg == int32_max) | (seg == int32_min)
        elif float_saturation_value is not None:
            is_sat = seg == float_saturation_value
        else:
            is_sat = np.zeros_like(seg, dtype=bool)
        clip_count += int(np.count_nonzero((run_len >= qc_cfg.clip_run_min_samples) & is_sat))

        seg_f = seg.astype(np.float64)
        sum_sq += float(np.sum(seg_f**2))
        sum_abs += float(np.sum(np.abs(seg_f)))
        float_segments.append(seg_f)

    rms = float(np.sqrt(sum_sq / total_npts))
    mean_abs = float(sum_abs / total_npts)

    x_all = np.concatenate(float_segments)
    median = np.median(x_all)
    mad = np.median(np.abs(x_all - median))
    spike_count = int(np.count_nonzero(np.abs(x_all - median) > qc_cfg.spike_threshold_mad * mad))
    kurt = float(scipy_kurtosis(x_all, fisher=True))

    return {
        "seed_id": seed_id,
        "date": day.isoformat(),
        "rms": rms,
        "mean_abs": mean_abs,
        "spike_count": spike_count,
        "kurtosis": kurt,
        "flat_count": flat_count,
        "clip_count": clip_count,
        "npts": total_npts,
    }


def _build_sample_channel_day_trace(rng) -> Trace:
    """One synthetic channel-day-shaped trace exercising every statistic at
    once: background noise, a cluster of spikes, a flat-lined run, and a
    saturated (clipped) run -- 2 hours at 200 Hz (1.44M samples), not a
    full 17.28M-sample day, so this regression test stays fast."""
    n = 200 * 3600 * 2
    data = rng.normal(scale=8.0, size=n)

    spike_idx = rng.choice(n, size=100, replace=False)
    data[spike_idx] += rng.choice([-1.0, 1.0], size=100) * rng.uniform(150.0, 400.0, size=100)

    flat_start = n // 4
    flat_len = 80
    data[flat_start : flat_start + flat_len] = 17.0

    clip_start = 3 * n // 4
    clip_len = 6
    data = np.round(data).astype(np.int32)
    data[clip_start : clip_start + clip_len] = 2**31 - 1

    tr = Trace(data=data)
    tr.stats.sampling_rate = 200.0
    tr.stats.starttime = UTCDateTime(2025, 1, 1)
    return tr


def test_optimised_output_matches_reference_implementation():
    """The optimised ``compute_channel_day_stats`` matches the from-scratch
    naive reference on a sample channel-day exercising rms, mean_abs,
    spike_count, kurtosis, flat_count and clip_count together.

    Tolerances (documented, per OpenMEQ-5nl.8's "identical, or within
    documented float tolerance" requirement):

    - ``rms``/``mean_abs``: both implementations do the same pooled
      summation *except* ``sum_sq`` uses ``np.dot`` instead of
      ``np.sum(x**2)`` -- a different summation order for the same
      mathematical sum, so these agree to ``rel=1e-9`` (comfortably above
      float64 rounding noise for a ~1.4M-term reduction, comfortably below
      anything that would matter for a noise statistic).
    - ``spike_count``/``flat_count``/``clip_count``: integer counts from
      exact-equality/threshold comparisons on the *same* underlying
      float64 values either way (only ``kurtosis`` runs in float32) --
      required to match **exactly**.
    - ``kurtosis``: the optimised path computes this in float32 (see
      :func:`openmeq.qc.stats._fast_kurtosis`); ``rel=1e-3`` comfortably
      covers that precision difference while still catching any real
      algorithmic regression.
    """
    rng = np.random.default_rng(20260928)
    tr = _build_sample_channel_day_trace(rng)
    stream = Stream([tr])
    qc_cfg = QcConfig()

    optimised = compute_channel_day_stats("XX.SAMPLE.00.HHZ", DAY1, stream, qc_cfg)
    reference = _reference_channel_day_stats("XX.SAMPLE.00.HHZ", DAY1, stream, qc_cfg)

    assert optimised["npts"] == reference["npts"]
    assert optimised["rms"] == pytest.approx(reference["rms"], rel=1e-9)
    assert optimised["mean_abs"] == pytest.approx(reference["mean_abs"], rel=1e-9)
    assert optimised["spike_count"] == reference["spike_count"]
    assert optimised["flat_count"] == reference["flat_count"]
    assert optimised["clip_count"] == reference["clip_count"]
    assert optimised["kurtosis"] == pytest.approx(reference["kurtosis"], rel=1e-3)

    # Sanity: the sample really does exercise every anomaly type (a
    # regression that zeroed out a whole code path would still "match" a
    # reference computed the same wrong way if the sample didn't).
    assert reference["spike_count"] > 50
    assert reference["flat_count"] >= 80
    assert reference["clip_count"] == 6
    assert reference["kurtosis"] > 5.0


def test_fast_kurtosis_matches_scipy_reference():
    """``_fast_kurtosis`` (float32, custom demean-once implementation)
    matches ``scipy.stats.kurtosis(x, fisher=True)`` (float64) within the
    same documented ``rel=1e-3`` tolerance used above, on both a plain
    noise sample and a heavy-tailed (spiky) one."""
    from scipy.stats import kurtosis as scipy_kurtosis

    rng = np.random.default_rng(7)

    plain = rng.normal(scale=5.0, size=500_000)
    plain_ref = scipy_kurtosis(plain, fisher=True)
    assert _fast_kurtosis(plain) == pytest.approx(plain_ref, rel=1e-3, abs=1e-2)

    spiky = plain.copy()
    idx = rng.choice(plain.size, size=500, replace=False)
    spiky[idx] += rng.uniform(100.0, 500.0, size=500)
    assert _fast_kurtosis(spiky) == pytest.approx(scipy_kurtosis(spiky, fisher=True), rel=1e-3)
