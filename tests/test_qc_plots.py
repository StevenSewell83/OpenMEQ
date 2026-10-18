"""Tests for ``openmeq.qc.plots`` (M1.6, ``OpenMEQ-5nl.6``).

Per the worker instructions for this task, these tests work **only from
small hand-built DataFrames/dicts** -- never from
``tests/conftest.py``'s ``synthetic_project`` fixture, which would recompute
QC over the ~500 MB synthetic dataset (slow, and this module never
recomputes QC anyway, only visualises already-computed tables). Exactly one
test (``test_integration_reads_hand_built_csvs_and_json``) is a light
integration test that round-trips through the on-disk CSV/JSON formats
(ARCHITECTURE.md §3/§4.2), still built by hand rather than computed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from openmeq.qc.plots import (
    SUMMARY_COLUMNS,
    make_qc_diagnostics,
    summarize_flags,
)


def _png_ok(path: Path) -> None:
    assert path.exists(), f"{path} was not created"
    assert path.stat().st_size > 0, f"{path} is empty"


# ---------------------------------------------------------------------------
# Hand-built fixtures (small: 3 stations x 5 days x 2 months)
# ---------------------------------------------------------------------------

SEED_IDS = ["XX.S01..HHZ", "XX.S02..HHZ", "XX.S03..HHZ"]
DATES = [f"2025-01-{d:02d}" for d in range(1, 6)] + [f"2025-02-{d:02d}" for d in range(1, 6)]
DAILY_STATS = ["rms", "spike_count"]


def _availability_df() -> pd.DataFrame:
    rows = []
    for i, seed_id in enumerate(SEED_IDS):
        for date in DATES:
            coverage = 100.0 - 5.0 * i  # station S03 has slightly worse coverage
            rows.append(
                {
                    "seed_id": seed_id,
                    "date": date,
                    "coverage_pct": coverage,
                    "n_gaps": 0,
                    "n_overlaps": 0,
                    "gap_s": 86400.0 * (1.0 - coverage / 100.0),
                }
            )
    return pd.DataFrame(rows)


def _daily_stats_df() -> pd.DataFrame:
    rows = []
    for i, seed_id in enumerate(SEED_IDS):
        for j, date in enumerate(DATES):
            rows.append(
                {
                    "seed_id": seed_id,
                    "date": date,
                    "rms": 1.0 + 0.1 * i + 0.01 * j,
                    "mean_abs": 0.8 + 0.1 * i,
                    "spike_count": i * 2,
                    "kurtosis": 3.0 + i,
                    "flat_count": 0,
                    "clip_count": 0,
                    "npts": 1000,
                }
            )
    return pd.DataFrame(rows)


def _monthly_df() -> pd.DataFrame:
    # One row per (seed_id, month, stat), matching
    # openmeq.qc.flags.MONTHLY_STATS_COLUMNS. Station S03 (index 2) is the
    # clear network outlier on "rms" in both months, so it should rank 1.
    rows = []
    for month in ["2025-01", "2025-02"]:
        for stat in DAILY_STATS:
            values = {"XX.S01..HHZ": 1.0, "XX.S02..HHZ": 1.05, "XX.S03..HHZ": 5.0}
            median = 1.05
            for seed_id, value in values.items():
                ratio = value / median
                rows.append(
                    {
                        "seed_id": seed_id,
                        "month": month,
                        "stat": stat,
                        "value": value,
                        "network_median": median,
                        "ratio": ratio,
                    }
                )
    df = pd.DataFrame(rows)
    # rank 1 = largest abs(log(ratio)) within each (month, stat) group.
    df["rank"] = (
        df.groupby(["month", "stat"])["ratio"].rank(ascending=False, method="first").astype(int)
    )
    return df[["seed_id", "month", "stat", "value", "network_median", "ratio", "rank"]]


def _flags() -> dict:
    flags: dict[str, dict[str, dict]] = {}
    for i, seed_id in enumerate(SEED_IDS):
        flags[seed_id] = {}
        for j, date in enumerate(DATES):
            if i == 2 and j % 2 == 0:
                flags[seed_id][date] = {"ok": False, "reasons": ["network_outlier"]}
            elif i == 2:
                flags[seed_id][date] = {"ok": False, "reasons": ["spikes"]}
            else:
                flags[seed_id][date] = {"ok": True, "reasons": []}
    return flags


# ---------------------------------------------------------------------------
# test_expected_files_created
# ---------------------------------------------------------------------------


def test_expected_files_created(tmp_path):
    out_dir = tmp_path / "qc" / "diagnostics"
    make_qc_diagnostics(
        _availability_df(),
        _daily_stats_df(),
        _monthly_df(),
        _flags(),
        out_dir,
        stats=DAILY_STATS,
    )

    _png_ok(out_dir / "availability_heatmap.png")
    # rms is gain-dependent: monthly (group-relative) only, no daily plot.
    assert not (out_dir / "daily_rms.png").exists()
    _png_ok(out_dir / "daily_spike_count.png")
    for stat in DAILY_STATS:
        _png_ok(out_dir / f"monthly_{stat}.png")
    _png_ok(out_dir / "flag_overview.png")

    summary_path = out_dir / "summary.csv"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0


def test_expected_files_created_with_default_stats(tmp_path):
    """Omitting ``stats`` discovers the stat list from the DataFrames
    themselves (see the module's "Tolerance to changing stat columns")."""
    out_dir = tmp_path / "diag"
    make_qc_diagnostics(_availability_df(), _daily_stats_df(), _monthly_df(), _flags(), out_dir)

    # daily_stats_df has 6 stat columns (rms, mean_abs, spike_count,
    # kurtosis, flat_count, clip_count); monthly_df only has the 2 in
    # DAILY_STATS -- both should be picked up automatically.
    for stat in ["spike_count", "kurtosis", "flat_count", "clip_count"]:
        _png_ok(out_dir / f"daily_{stat}.png")
    for stat in ["rms", "mean_abs"]:
        assert not (out_dir / f"daily_{stat}.png").exists()
    for stat in DAILY_STATS:
        _png_ok(out_dir / f"monthly_{stat}.png")
    _png_ok(out_dir / "availability_heatmap.png")
    _png_ok(out_dir / "flag_overview.png")


# ---------------------------------------------------------------------------
# test_summary_csv_content
# ---------------------------------------------------------------------------


def test_summary_csv_content(tmp_path):
    out_dir = tmp_path / "diag"
    flags = _flags()
    make_qc_diagnostics(
        _availability_df(), _daily_stats_df(), _monthly_df(), flags, out_dir, stats=DAILY_STATS
    )

    df = pd.read_csv(out_dir / "summary.csv")
    assert list(df.columns) == SUMMARY_COLUMNS
    assert set(df["seed_id"]) == set(SEED_IDS)
    assert len(df) == len(SEED_IDS)

    assert (df["pct_days_ok"] >= 0).all()
    assert (df["pct_days_ok"] <= 100).all()

    s01 = df.loc[df["seed_id"] == "XX.S01..HHZ"].iloc[0]
    assert s01["pct_days_ok"] == pytest.approx(100.0)
    assert s01["n_days_flagged"] == 0
    assert s01["worst_reason"] == "none"

    s03 = df.loc[df["seed_id"] == "XX.S03..HHZ"].iloc[0]
    assert s03["n_days_flagged"] == len(DATES)
    assert s03["pct_days_ok"] == pytest.approx(0.0)
    # Half the days are "network_outlier" (j even: 0,2,4,6,8 -> 5 of 10),
    # the other half "spikes" (5 of 10) -- a tie, broken alphabetically.
    assert s03["worst_reason"] == "network_outlier"


def test_summarize_flags_matches_make_qc_diagnostics_summary(tmp_path):
    flags = _flags()
    df = summarize_flags(flags)
    out_dir = tmp_path / "diag"
    make_qc_diagnostics(
        _availability_df(), _daily_stats_df(), _monthly_df(), flags, out_dir, stats=DAILY_STATS
    )
    on_disk = pd.read_csv(out_dir / "summary.csv")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), on_disk, check_dtype=False)


def test_summarize_flags_empty():
    df = summarize_flags({})
    assert list(df.columns) == SUMMARY_COLUMNS
    assert len(df) == 0


# ---------------------------------------------------------------------------
# test_no_direct_matplotlib_import
# ---------------------------------------------------------------------------


def test_no_direct_matplotlib_import():
    module_path = Path(__file__).resolve().parent.parent / "openmeq" / "qc" / "plots.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "matplotlib", (
                    f"openmeq/qc/plots.py must not import matplotlib directly "
                    f"(found `import {alias.name}`)"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.split(".")[0] == "matplotlib", (
                f"openmeq/qc/plots.py must not import matplotlib directly "
                f"(found `from {node.module} import ...`)"
            )


# ---------------------------------------------------------------------------
# Tolerance to extra/missing stat columns (M1.8/M1.9 are concurrently
# changing openmeq.qc.stats -- see the module's docstring)
# ---------------------------------------------------------------------------


def test_tolerates_missing_stat_column(tmp_path):
    out_dir = tmp_path / "diag"
    daily_df = _daily_stats_df().drop(columns=["spike_count"])
    # Should not raise even though "spike_count" is requested but missing.
    make_qc_diagnostics(
        _availability_df(), daily_df, _monthly_df(), _flags(), out_dir, stats=DAILY_STATS
    )
    assert not (out_dir / "daily_spike_count.png").exists()  # skipped, logged


def test_tolerates_extra_stat_column(tmp_path):
    out_dir = tmp_path / "diag"
    daily_df = _daily_stats_df()
    daily_df["longest_flat_run"] = 0  # e.g. a future M1.9 column
    make_qc_diagnostics(_availability_df(), daily_df, _monthly_df(), _flags(), out_dir)
    _png_ok(out_dir / "daily_longest_flat_run.png")


def test_tolerates_missing_monthly_stat(tmp_path):
    out_dir = tmp_path / "diag"
    monthly_df = _monthly_df()
    monthly_df = monthly_df[monthly_df["stat"] != "spike_count"]
    make_qc_diagnostics(
        _availability_df(), _daily_stats_df(), monthly_df, _flags(), out_dir, stats=DAILY_STATS
    )
    _png_ok(out_dir / "monthly_rms.png")
    assert not (out_dir / "monthly_spike_count.png").exists()  # skipped, logged


# ---------------------------------------------------------------------------
# Degenerate input: single station, empty tables
# ---------------------------------------------------------------------------


def test_single_station(tmp_path):
    out_dir = tmp_path / "diag"
    seed_id = "XX.S01..HHZ"
    avail = pd.DataFrame(
        {
            "seed_id": [seed_id] * len(DATES),
            "date": DATES,
            "coverage_pct": [100.0] * len(DATES),
            "n_gaps": [0] * len(DATES),
            "n_overlaps": [0] * len(DATES),
            "gap_s": [0.0] * len(DATES),
        }
    )
    daily = pd.DataFrame(
        {
            "seed_id": [seed_id] * len(DATES),
            "date": DATES,
            "rms": [1.0] * len(DATES),
            "npts": [1000] * len(DATES),
        }
    )
    monthly = pd.DataFrame(
        {
            "seed_id": [seed_id, seed_id],
            "month": ["2025-01", "2025-02"],
            "stat": ["rms", "rms"],
            "value": [1.0, 1.0],
            "network_median": [1.0, 1.0],
            "ratio": [1.0, 1.0],
            "rank": [1, 1],
        }
    )
    flags = {seed_id: {d: {"ok": True, "reasons": []} for d in DATES}}
    make_qc_diagnostics(avail, daily, monthly, flags, out_dir, stats=["rms"])
    _png_ok(out_dir / "availability_heatmap.png")
    _png_ok(out_dir / "monthly_rms.png")
    _png_ok(out_dir / "flag_overview.png")


def test_empty_inputs(tmp_path):
    out_dir = tmp_path / "diag"
    empty_avail = pd.DataFrame(columns=["seed_id", "date", "coverage_pct"])
    empty_daily = pd.DataFrame(columns=["seed_id", "date", "npts"])
    empty_monthly = pd.DataFrame(
        columns=["seed_id", "month", "stat", "value", "network_median", "ratio", "rank"]
    )
    written = make_qc_diagnostics(
        empty_avail, empty_daily, empty_monthly, {}, out_dir, stats=["rms"]
    )
    assert written == []  # nothing to draw; summary.csv is still written
    df = pd.read_csv(out_dir / "summary.csv")
    assert len(df) == 0
    assert list(df.columns) == SUMMARY_COLUMNS


# ---------------------------------------------------------------------------
# Light integration test: hand-built tables written to and re-read from the
# real on-disk CSV/JSON formats (ARCHITECTURE.md §3/§4.2), still hand-built
# (never computed from the synthetic archive).
# ---------------------------------------------------------------------------


def test_integration_reads_hand_built_csvs_and_json(tmp_path):
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()

    _availability_df().to_csv(qc_dir / "availability.csv", index=False)
    _daily_stats_df().to_csv(qc_dir / "daily_stats.csv", index=False)
    _monthly_df().to_csv(qc_dir / "monthly_stats.csv", index=False)
    with open(qc_dir / "channel_flags.json", "w", encoding="utf-8") as f:
        json.dump({"version": 1, "flags": _flags()}, f)

    availability_df = pd.read_csv(qc_dir / "availability.csv")
    daily_stats_df = pd.read_csv(qc_dir / "daily_stats.csv")
    monthly_df = pd.read_csv(qc_dir / "monthly_stats.csv")
    with open(qc_dir / "channel_flags.json", encoding="utf-8") as f:
        flags = json.load(f)["flags"]

    out_dir = qc_dir / "diagnostics"
    make_qc_diagnostics(
        availability_df, daily_stats_df, monthly_df, flags, out_dir, stats=DAILY_STATS
    )

    _png_ok(out_dir / "availability_heatmap.png")
    _png_ok(out_dir / "daily_spike_count.png")
    for stat in DAILY_STATS:
        _png_ok(out_dir / f"monthly_{stat}.png")
    _png_ok(out_dir / "flag_overview.png")
    summary_df = pd.read_csv(out_dir / "summary.csv")
    assert set(summary_df["seed_id"]) == set(SEED_IDS)


# ---------------------------------------------------------------------------
# Instrument groups, labels and flag categories (OpenMEQ-gyb)
# ---------------------------------------------------------------------------


def _grouped_instruments() -> pd.DataFrame:
    # S01/S02 share a group with 4x different sensitivities; S03 is alone.
    return pd.DataFrame(
        {
            "seed_id": SEED_IDS,
            "group": ["GS-11D", "GS-11D", "L4C"],
            "sensitivity": [1.0, 4.0, 1.0],
        }
    )


def test_row_table_labels_and_group_order():
    from openmeq.qc.plots import _row_table

    rows = _row_table(SEED_IDS, _grouped_instruments())
    assert list(rows["group"]) == ["GS-11D", "GS-11D", "L4C"]  # largest group first
    assert list(rows["label"]) == ["S01.HHZ", "S02.HHZ", "S03.HHZ"]
    # Without instruments everything is one group.
    assert set(_row_table(SEED_IDS, None)["group"]) == {"all"}
    # Several location codes at one station -> STA.LOC.CHA labels.
    two_locs = _row_table(["XX.S01.00.HHZ", "XX.S01.10.HHZ"], None)
    assert list(two_locs["label"]) == ["S01.00.HHZ", "S01.10.HHZ"]


def test_gain_dependent_monthly_ratio_is_group_relative_and_sensitivity_corrected():
    from openmeq.qc.plots import _monthly_ratios, _row_table

    rows = _row_table(SEED_IDS, _grouped_instruments())
    monthly = pd.DataFrame(
        {
            "seed_id": SEED_IDS * 2,
            "month": ["2025-01"] * 3 + ["2025-02"] * 3,
            "stat": "rms",
            # S02 reads 4x S01 in counts but has 4x the sensitivity: same ground motion.
            "value": [1.0, 4.0, 50.0, 1.0, 4.0, 150.0],
            "network_median": 4.0,
            "ratio": 1.0,
            "rank": 1,
        }
    )
    ratio, own = _monthly_ratios(monthly, "rms", rows)
    assert ratio.loc["XX.S01..HHZ"].tolist() == pytest.approx([1.0, 1.0])
    assert ratio.loc["XX.S02..HHZ"].tolist() == pytest.approx([1.0, 1.0])
    # Single-station group: against its own median over the run (100).
    assert own == {"XX.S03..HHZ"}
    assert ratio.loc["XX.S03..HHZ"].tolist() == pytest.approx([0.5, 1.5])


def test_flag_category_priority():
    from openmeq.qc.plots import FLAG_CATEGORIES, _flag_category

    value = {label: v for v, _c, label, _p in FLAG_CATEGORIES}
    assert _flag_category({"ok": True, "reasons": []}) == value["ok"]
    assert (
        _flag_category({"ok": False, "reasons": ["network_outlier", "spikes"]}) == value["spikes"]
    )
    assert (
        _flag_category({"ok": False, "reasons": ["spikes", "availability<80"]})
        == value["low availability"]
    )
    assert _flag_category({"ok": False, "reasons": ["something_new"]}) == value["other"]


def test_grouped_run_writes_expected_set(tmp_path):
    out_dir = tmp_path / "diag"
    written = make_qc_diagnostics(
        _availability_df(),
        _daily_stats_df(),
        _monthly_df(),
        _flags(),
        out_dir,
        stats=["rms", "spike_count", "kurtosis"],
        instruments=_grouped_instruments(),
        title="proj",
    )
    names = sorted(p.name for p in written)
    assert names == [
        "availability_heatmap.png",
        "daily_kurtosis.png",
        "daily_spike_count.png",
        "flag_overview.png",
        "monthly_rms.png",
        "monthly_spike_count.png",
    ]
    for p in written:
        _png_ok(p)
