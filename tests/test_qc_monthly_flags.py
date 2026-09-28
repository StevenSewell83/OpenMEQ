"""Tests for ``openmeq.qc.flags`` (M1.5, ``OpenMEQ-5nl.5``): monthly
cross-station comparison (:func:`compute_monthly_stats`) and per-channel-day
flags (:func:`compute_channel_flags` / :func:`write_channel_flags`).

Hand-built-DataFrame tests below construct only the columns each function
under test actually reads (``compute_channel_flags`` only ever looks at
``seed_id``, ``date``, ``coverage_pct`` from an availability-shaped frame and
``seed_id``, ``date``, ``spike_count``, ``npts``, ``flat_count``,
``clip_count`` from a daily-stats-shaped frame -- see ``openmeq/qc/flags.py``),
plus a couple of extra columns for readability/realism where cheap.

``compute_monthly_stats``/``compute_channel_flags`` are typed as taking a
full ``openmeq.config.Config``, but only ever read ``cfg.qc``. Rather than
constructing a full ``Config`` (which requires unrelated required sections
-- ``project``, ``data``, ``nlloc``, ``hypodd`` -- with no bearing on this
module), the hand-built tests pass a minimal ``_FakeCfg`` duck-typed object
exposing just ``.qc``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from openmeq.config import QcConfig
from openmeq.qc.flags import (
    MONTHLY_STATS_COLUMNS,
    compute_channel_flags,
    compute_monthly_stats,
    write_channel_flags,
    write_monthly_stats_csv,
)


class _FakeCfg:
    """Minimal duck-typed stand-in for ``openmeq.config.Config`` exposing
    only ``.qc`` -- see the module docstring."""

    def __init__(self, qc: QcConfig) -> None:
        self.qc = qc


def _cfg(**qc_kwargs) -> _FakeCfg:
    return _FakeCfg(QcConfig(**qc_kwargs))


def _availability_row(seed_id: str, date: str, coverage_pct: float) -> dict:
    return {
        "seed_id": seed_id,
        "date": date,
        "coverage_pct": coverage_pct,
        "n_gaps": 0,
        "n_overlaps": 0,
        "gap_s": 86400.0 * (1.0 - coverage_pct / 100.0),
    }


def _daily_stats_row(
    seed_id: str,
    date: str,
    *,
    rms: float = 1.0,
    mean_abs: float = 1.0,
    spike_count: int = 0,
    kurtosis: float = 0.0,
    flat_count: int = 0,
    longest_flat_run_s: float = 0.0,
    clip_count: int = 0,
    npts: int = 1000,
) -> dict:
    return {
        "seed_id": seed_id,
        "date": date,
        "rms": rms,
        "mean_abs": mean_abs,
        "spike_count": spike_count,
        "kurtosis": kurtosis,
        "flat_count": flat_count,
        "longest_flat_run_s": longest_flat_run_s,
        "clip_count": clip_count,
        "npts": npts,
    }


EMPTY_MONTHLY = pd.DataFrame(columns=MONTHLY_STATS_COLUMNS)


# ---------------------------------------------------------------------------
# compute_monthly_stats
# ---------------------------------------------------------------------------


def test_monthly_stats_median_and_ratio():
    """3 stations, 1 month, known RMS values -> deterministic median/ratio/rank.

    A=10 (mean over 2 days), B=20, C=100 -> network_median=20.
    ratio: A=0.5, B=1.0, C=5.0.
    abs(log(ratio)): A=0.693, B=0.0, C=1.609 -> rank order C(1), A(2), B(3).
    """
    daily = pd.DataFrame(
        [
            {"seed_id": "XX.A.00.HHZ", "date": "2025-01-01", "rms": 10.0},
            {"seed_id": "XX.A.00.HHZ", "date": "2025-01-02", "rms": 10.0},
            {"seed_id": "XX.B.00.HHZ", "date": "2025-01-01", "rms": 20.0},
            {"seed_id": "XX.B.00.HHZ", "date": "2025-01-02", "rms": 20.0},
            {"seed_id": "XX.C.00.HHZ", "date": "2025-01-01", "rms": 100.0},
            {"seed_id": "XX.C.00.HHZ", "date": "2025-01-02", "rms": 100.0},
        ]
    )
    cfg = _cfg(daily_stats=["rms"])

    monthly = compute_monthly_stats(daily, cfg)

    assert list(monthly.columns) == MONTHLY_STATS_COLUMNS
    assert len(monthly) == 3
    assert set(monthly["stat"]) == {"rms"}
    assert set(monthly["month"]) == {"2025-01"}

    def _row(seed_id):
        m = monthly[monthly["seed_id"] == seed_id]
        assert len(m) == 1
        return m.iloc[0]

    row_a, row_b, row_c = _row("XX.A.00.HHZ"), _row("XX.B.00.HHZ"), _row("XX.C.00.HHZ")

    assert row_a["value"] == pytest.approx(10.0)
    assert row_b["value"] == pytest.approx(20.0)
    assert row_c["value"] == pytest.approx(100.0)

    for row in (row_a, row_b, row_c):
        assert row["network_median"] == pytest.approx(20.0)

    assert row_a["ratio"] == pytest.approx(0.5)
    assert row_b["ratio"] == pytest.approx(1.0)
    assert row_c["ratio"] == pytest.approx(5.0)

    assert row_c["rank"] == 1  # abs(log(5.0)) = 1.609, largest deviation
    assert row_a["rank"] == 2  # abs(log(0.5)) = 0.693
    assert row_b["rank"] == 3  # abs(log(1.0)) = 0.0, smallest deviation


def test_monthly_stats_zero_median_guard():
    """network_median == 0: ratio = value (documented convention), except a
    station whose own value is also 0, where ratio is nan (no well-defined
    deviation) rather than 0/0."""
    daily = pd.DataFrame(
        [
            {"seed_id": "XX.A.00.HHZ", "date": "2025-01-01", "spike_count": 0},
            {"seed_id": "XX.B.00.HHZ", "date": "2025-01-01", "spike_count": 0},
            {"seed_id": "XX.C.00.HHZ", "date": "2025-01-01", "spike_count": 5},
        ]
    )
    cfg = _cfg(daily_stats=["spike_count"])

    monthly = compute_monthly_stats(daily, cfg)

    medians = monthly["network_median"].unique()
    assert len(medians) == 1
    assert medians[0] == pytest.approx(0.0)

    row_a = monthly[monthly["seed_id"] == "XX.A.00.HHZ"].iloc[0]
    row_c = monthly[monthly["seed_id"] == "XX.C.00.HHZ"].iloc[0]
    assert pd.isna(row_a["ratio"])  # value == 0 == network_median -> nan
    assert row_c["ratio"] == pytest.approx(5.0)  # value != 0, median == 0 -> ratio = value
    # The well-defined (finite log) deviation ranks ahead of the nan one.
    assert row_c["rank"] == 1


def test_monthly_stats_skips_unconfigured_stat_columns_without_crashing():
    """A stat named in cfg.qc.daily_stats but absent from daily_stats_df's
    columns (e.g. a concurrently-narrowed M1.4 output) must not crash --
    only the columns actually present are aggregated."""
    daily = pd.DataFrame(
        [
            {"seed_id": "XX.A.00.HHZ", "date": "2025-01-01", "rms": 5.0},
            {"seed_id": "XX.B.00.HHZ", "date": "2025-01-01", "rms": 15.0},
        ]
    )
    cfg = _cfg(daily_stats=["rms", "mean_abs", "spike_count", "kurtosis"])

    monthly = compute_monthly_stats(daily, cfg)

    assert set(monthly["stat"]) == {"rms"}
    assert len(monthly) == 2


def test_monthly_stats_empty_input():
    cfg = _cfg()
    monthly = compute_monthly_stats(pd.DataFrame(), cfg)
    assert list(monthly.columns) == MONTHLY_STATS_COLUMNS
    assert monthly.empty


def test_write_monthly_stats_csv(tmp_path):
    daily = pd.DataFrame(
        [
            {"seed_id": "XX.A.00.HHZ", "date": "2025-01-01", "rms": 5.0},
            {"seed_id": "XX.B.00.HHZ", "date": "2025-01-01", "rms": 15.0},
        ]
    )
    cfg = _cfg(daily_stats=["rms"])
    monthly = compute_monthly_stats(daily, cfg)

    path = tmp_path / "qc" / "monthly_stats.csv"
    write_monthly_stats_csv(monthly, path)

    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == MONTHLY_STATS_COLUMNS
    assert len(reloaded) == 2


# ---------------------------------------------------------------------------
# compute_channel_flags
# ---------------------------------------------------------------------------


def test_flags_availability_reason():
    cfg = _cfg()  # default min_availability_pct=80
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 50.0)])
    daily = pd.DataFrame([_daily_stats_row("XX.A.00.HHZ", "2025-01-01")])

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert entry["ok"] is False
    assert "availability<80" in entry["reasons"]


def test_flags_spikes_reason():
    cfg = _cfg()  # default max_spike_fraction=0.01
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 100.0)])
    daily = pd.DataFrame(
        [_daily_stats_row("XX.A.00.HHZ", "2025-01-01", spike_count=200, npts=1000)]
    )

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert entry["ok"] is False
    assert "spikes" in entry["reasons"]
    assert not any(r.startswith("availability") for r in entry["reasons"])


def test_flags_flat_reason():
    cfg = _cfg()  # default max_flat_fraction=0.5
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 100.0)])
    daily = pd.DataFrame([_daily_stats_row("XX.A.00.HHZ", "2025-01-01", flat_count=600, npts=1000)])

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert entry["ok"] is False
    assert "flat" in entry["reasons"]


def test_flags_flat_reason_via_max_flat_run_s():
    """M1.9: a long single flat run flags "flat" via qc.max_flat_run_s even
    when flat_count/npts is nowhere near max_flat_fraction (e.g. a dead
    sensor for an hour out of a long day) -- a small total fraction of a
    long recording is still a real fault worth flagging on duration alone.
    """
    cfg = _cfg()  # default max_flat_run_s=60.0, max_flat_fraction=0.5
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 100.0)])
    daily = pd.DataFrame(
        [
            _daily_stats_row(
                "XX.A.00.HHZ",
                "2025-01-01",
                flat_count=3600,
                longest_flat_run_s=3600.0,  # 1 hour -- far past max_flat_run_s
                npts=17_280_000,  # flat_count/npts ~= 2e-4, nowhere near max_flat_fraction
            )
        ]
    )

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert entry["ok"] is False
    assert "flat" in entry["reasons"]


def test_flags_flat_reason_not_raised_by_short_run_below_both_criteria():
    """Neither criterion crossed -> no "flat" reason (guards against an
    accidental always-true condition)."""
    cfg = _cfg()  # default max_flat_fraction=0.5, max_flat_run_s=60.0
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 100.0)])
    daily = pd.DataFrame(
        [
            _daily_stats_row(
                "XX.A.00.HHZ",
                "2025-01-01",
                flat_count=5,
                longest_flat_run_s=0.05,
                npts=1000,
            )
        ]
    )

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert "flat" not in entry["reasons"]


def test_flags_clipped_reason():
    cfg = _cfg()  # default max_clip_count=0
    availability = pd.DataFrame([_availability_row("XX.A.00.HHZ", "2025-01-01", 100.0)])
    daily = pd.DataFrame([_daily_stats_row("XX.A.00.HHZ", "2025-01-01", clip_count=1)])

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    entry = flags["XX.A.00.HHZ"]["2025-01-01"]
    assert entry["ok"] is False
    assert "clipped" in entry["reasons"]


def test_exclude_channels_overrides():
    """exclude_channels is a hard override: applied even when every other
    stat is perfectly clean, for every date the channel has an entry."""
    cfg = _cfg(exclude_channels=["XX.BAD.00.HHZ"])
    availability = pd.DataFrame(
        [
            _availability_row("XX.BAD.00.HHZ", "2025-01-01", 100.0),
            _availability_row("XX.BAD.00.HHZ", "2025-01-02", 100.0),
        ]
    )
    daily = pd.DataFrame(
        [
            _daily_stats_row("XX.BAD.00.HHZ", "2025-01-01"),
            _daily_stats_row("XX.BAD.00.HHZ", "2025-01-02"),
        ]
    )

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    for date in ("2025-01-01", "2025-01-02"):
        entry = flags["XX.BAD.00.HHZ"][date]
        assert entry["ok"] is False
        assert entry["reasons"] == ["excluded_by_config"]


def test_network_outlier_reason():
    """A monthly ratio beyond monthly_outlier_ratio (or below its
    reciprocal) for any configured stat propagates "network_outlier" to
    every date within that (seed_id, month), on top of any other reasons."""
    cfg = _cfg(monthly_outlier_ratio=3.0, daily_stats=["rms"])
    availability = pd.DataFrame(
        [
            _availability_row("XX.OUT.00.HHZ", "2025-01-05", 100.0),
            _availability_row("XX.OUT.00.HHZ", "2025-01-06", 100.0),
        ]
    )
    daily = pd.DataFrame(
        [
            _daily_stats_row("XX.OUT.00.HHZ", "2025-01-05"),
            _daily_stats_row("XX.OUT.00.HHZ", "2025-01-06"),
        ]
    )
    monthly = pd.DataFrame(
        [
            {
                "seed_id": "XX.OUT.00.HHZ",
                "month": "2025-01",
                "stat": "rms",
                "value": 40.0,
                "network_median": 10.0,
                "ratio": 4.0,
                "rank": 1,
            },
            {
                "seed_id": "XX.OTHER.00.HHZ",
                "month": "2025-01",
                "stat": "rms",
                "value": 10.0,
                "network_median": 10.0,
                "ratio": 1.0,
                "rank": 2,
            },
        ]
    )

    flags = compute_channel_flags(availability, daily, monthly, cfg)

    for date in ("2025-01-05", "2025-01-06"):
        entry = flags["XX.OUT.00.HHZ"][date]
        assert entry["ok"] is False
        assert "network_outlier" in entry["reasons"]


def test_network_outlier_reciprocal_side():
    """A ratio far *below* 1 (below the reciprocal threshold) is just as
    anomalous as one far above it."""
    cfg = _cfg(monthly_outlier_ratio=3.0, daily_stats=["rms"])
    availability = pd.DataFrame([_availability_row("XX.LOW.00.HHZ", "2025-02-01", 100.0)])
    daily = pd.DataFrame([_daily_stats_row("XX.LOW.00.HHZ", "2025-02-01")])
    monthly = pd.DataFrame(
        [
            {
                "seed_id": "XX.LOW.00.HHZ",
                "month": "2025-02",
                "stat": "rms",
                "value": 2.0,
                "network_median": 10.0,
                "ratio": 0.2,  # < 1/3
                "rank": 1,
            }
        ]
    )

    flags = compute_channel_flags(availability, daily, monthly, cfg)

    assert "network_outlier" in flags["XX.LOW.00.HHZ"]["2025-02-01"]["reasons"]


def test_no_data_day_omitted():
    """A channel/day absent from both input DataFrames never appears as a
    key in the output, and a channel/day with coverage_pct == 0 (present in
    availability_df, but "no data at all") is omitted too, per
    ARCHITECTURE.md §4.2."""
    cfg = _cfg()
    availability = pd.DataFrame(
        [
            _availability_row("XX.A.00.HHZ", "2025-01-01", 100.0),
            _availability_row("XX.A.00.HHZ", "2025-01-02", 0.0),  # no data at all
        ]
    )
    daily = pd.DataFrame([_daily_stats_row("XX.A.00.HHZ", "2025-01-01")])

    flags = compute_channel_flags(availability, daily, EMPTY_MONTHLY, cfg)

    assert "XX.A.00.HHZ" in flags
    assert "2025-01-01" in flags["XX.A.00.HHZ"]
    # coverage_pct == 0 -> omitted entirely, not written with a reason.
    assert "2025-01-02" not in flags["XX.A.00.HHZ"]
    # A seed_id/date never present in either input is naturally absent too.
    assert "XX.GHOST.00.HHZ" not in flags


def test_json_schema(tmp_path):
    flags = {
        "XX.A.00.HHZ": {
            "2025-01-01": {"ok": True, "reasons": []},
            "2025-01-02": {"ok": False, "reasons": ["availability<80", "spikes"]},
        }
    }
    path = tmp_path / "qc" / "channel_flags.json"

    write_channel_flags(flags, path)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    assert set(payload.keys()) == {"version", "flags"}
    assert payload["version"] == 1
    assert payload["flags"] == flags

    entry_ok = payload["flags"]["XX.A.00.HHZ"]["2025-01-01"]
    assert set(entry_ok.keys()) == {"ok", "reasons"}
    assert entry_ok["ok"] is True
    assert entry_ok["reasons"] == []

    entry_bad = payload["flags"]["XX.A.00.HHZ"]["2025-01-02"]
    assert entry_bad["ok"] is False
    assert entry_bad["reasons"] == ["availability<80", "spikes"]


# ---------------------------------------------------------------------------
# End-to-end on the synthetic fixture (M1.3/M1.4's known gap/spiky/flat
# channel-days), per the M1.5 issue's acceptance criteria.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qc_world(synthetic_qc_tables):
    """``(cfg, availability_df, daily_stats_df, monthly_df, truth)`` for the
    shared synthetic project.

    OpenMEQ-5nl.9: no longer computes anything itself -- just repackages
    the session-scoped ``synthetic_qc_tables`` fixture (shared with
    ``tests/test_qc_stats.py``, see ``tests/conftest.py``) into the tuple
    shape every test in this module already expects.
    """
    cfg, _idx, availability_df, daily_stats_df, monthly_df, truth = synthetic_qc_tables
    return cfg, availability_df, daily_stats_df, monthly_df, truth


def _issue(truth: dict, kind: str) -> dict:
    (issue,) = [i for i in truth["known_issues"] if i["kind"] == kind]
    return issue


def _one_row(df: pd.DataFrame, seed_id: str, date: str) -> pd.Series:
    m = df[(df["seed_id"] == seed_id) & (df["date"] == date)]
    assert len(m) == 1, f"expected exactly one row for {seed_id}/{date}, got {len(m)}"
    return m.iloc[0]


def test_e2e_monthly_stats_well_formed(qc_world):
    """Sanity check of compute_monthly_stats's output shape on real data:
    every (month, stat) group's ranks are a dense 1..N permutation."""
    _cfg_obj, _avail, _daily, monthly_df, _truth = qc_world

    assert list(monthly_df.columns) == MONTHLY_STATS_COLUMNS
    assert not monthly_df.empty

    for (_month, _stat), group in monthly_df.groupby(["month", "stat"]):
        ranks = sorted(group["rank"].tolist())
        assert ranks == list(range(1, len(group) + 1))


def test_e2e_gap_day_flagged_not_ok(qc_world):
    """The known gap channel/day (10 min missing out of a day, ARCHITECTURE.md
    §8) is flagged not-ok with an "availability<..." reason.

    **Verified fact**: this fixture's injected gap is short enough
    (coverage_pct ~= 99.31%) that it does *not* cross
    ``qc.min_availability_pct``'s default (80%) -- per the M1.5 issue's own
    "Coordination with M1.3" pitfall, checked here rather than assumed. The
    threshold is tightened to just above the actual measured coverage_pct to
    exercise the "availability<..." reason path end-to-end against real
    numbers.
    """
    cfg, availability_df, daily_stats_df, monthly_df, truth = qc_world
    issue = _issue(truth, "gap")
    date_str = issue["start"].date.isoformat()

    avail_row = _one_row(availability_df, issue["seed_id"], date_str)
    coverage_pct = float(avail_row["coverage_pct"])
    assert 0.0 < coverage_pct < 100.0  # a real (partial) gap, not "no data"
    assert coverage_pct >= cfg.qc.min_availability_pct  # doesn't cross the default

    tuned_cfg = cfg.model_copy(deep=True)
    tuned_cfg.qc.min_availability_pct = coverage_pct + 0.05

    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, tuned_cfg)
    entry = flags[issue["seed_id"]][date_str]
    assert entry["ok"] is False
    assert any(r.startswith("availability<") for r in entry["reasons"])


def test_e2e_spiky_day_flagged_not_ok_default_cfg(qc_world):
    """M1.9: the known spiky channel/day is flagged not-ok with a "spikes"
    reason **under the shipped default config, with no threshold tuning**
    -- this is the acceptance criterion this issue exists to satisfy (the
    M1.5-era version of this test had to tighten ``max_spike_fraction``
    below the actual measured fraction because the old spike metric
    couldn't tell a glitch from an earthquake; see ``openmeq/qc/stats.py``'s
    module docstring and ``docs/performance.md`` for the isolated-impulse
    metric that fixes this).

    **Verified numbers** (isolated-impulse ``spike_count``, default
    ``spike_threshold_mad=10``): ``XX.STA06.00.HHE`` 2025-01-02 measures
    spike_count=388/npts=17280000 (fraction ~2.245e-5), comfortably above
    ``qc.max_spike_fraction``'s default (1.2e-5); the next-highest
    *ordinary* channel/day that same day measures only 119
    (fraction ~6.89e-6), comfortably below it.
    """
    cfg, availability_df, daily_stats_df, monthly_df, truth = qc_world
    issue = _issue(truth, "spikes")
    date_str = issue["start"].date.isoformat()

    stats_row = _one_row(daily_stats_df, issue["seed_id"], date_str)
    spike_fraction = float(stats_row["spike_count"]) / float(stats_row["npts"])
    assert spike_fraction > cfg.qc.max_spike_fraction  # crosses the *default* threshold

    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    entry = flags[issue["seed_id"]][date_str]
    assert entry["ok"] is False
    assert "spikes" in entry["reasons"]


def test_e2e_flat_day_flagged_not_ok_default_cfg(qc_world):
    """M1.9: the known flat-lined channel/day is flagged not-ok with a
    "flat" reason **under the shipped default config, with no threshold
    tuning** -- via the new ``longest_flat_run_s`` / ``qc.max_flat_run_s``
    duration criterion, not ``max_flat_fraction`` (the injected flat run is
    only ~4.17% of the day, still well below ``max_flat_fraction``'s
    default of 50%, exactly as it was before M1.9 -- the fix is the new
    *duration* criterion, not a retuned fraction).

    **Verified numbers**: ``XX.STA03.00.HHN`` 2025-01-01's
    ``longest_flat_run_s`` measures ~3600s (the whole 1-hour injected
    window), comfortably above ``qc.max_flat_run_s``'s default (60s); every
    ordinary channel/day's longest run is under 1 second.
    """
    cfg, availability_df, daily_stats_df, monthly_df, truth = qc_world
    issue = _issue(truth, "flat")
    date_str = issue["start"].date.isoformat()

    stats_row = _one_row(daily_stats_df, issue["seed_id"], date_str)
    flat_fraction = float(stats_row["flat_count"]) / float(stats_row["npts"])
    assert flat_fraction <= cfg.qc.max_flat_fraction  # still doesn't cross the fraction criterion
    # crosses the duration criterion instead
    assert float(stats_row["longest_flat_run_s"]) > cfg.qc.max_flat_run_s

    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    entry = flags[issue["seed_id"]][date_str]
    assert entry["ok"] is False
    assert "flat" in entry["reasons"]


def test_e2e_default_cfg_runs_clean_channel_ok(qc_world):
    """An ordinary channel/day (no known issue, default thresholds) is ok."""
    cfg, availability_df, daily_stats_df, monthly_df, truth = qc_world
    known_seed_ids = {issue["seed_id"] for issue in truth["known_issues"]}
    ordinary_seed_id = "XX.STA01.00.HHZ"
    assert ordinary_seed_id not in known_seed_ids

    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    entry = flags[ordinary_seed_id]["2025-01-01"]
    assert entry["ok"] is True


def test_e2e_zero_false_positives_on_ordinary_channel_days(qc_world):
    """M1.9 acceptance criterion: under the shipped default config, every
    channel/day *other than* the known spiky and flat ones is ``ok`` (the
    known gap channel/day is exempted too -- per this issue's documented
    gap policy, a 10-minute gap out of a day is acceptable and, by design,
    does not flag; it must merely appear in the availability output, which
    ``test_e2e_gap_day_flagged_not_ok`` above's own coverage_pct assertion
    already confirms it does)."""
    cfg, availability_df, daily_stats_df, monthly_df, truth = qc_world

    flagged_seed_id_dates = {
        (issue["seed_id"], issue["start"].date.isoformat())
        for issue in truth["known_issues"]
        if issue["kind"] in ("spikes", "flat")
    }
    gap_issue = _issue(truth, "gap")
    exempt = flagged_seed_id_dates | {(gap_issue["seed_id"], gap_issue["start"].date.isoformat())}

    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)

    false_positives = []
    for seed_id, days in flags.items():
        for date_str, entry in days.items():
            if (seed_id, date_str) in exempt:
                continue
            if not entry["ok"]:
                false_positives.append((seed_id, date_str, entry["reasons"]))

    assert false_positives == []
    assert entry["reasons"] == []
