"""Tests for openmeq.magnitude.diagnostics_extended's pure GR statistics and
figure builders (OpenMEQ-02o.6). The pipeline on/off gating is covered in
``tests/test_magnitude_run.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openmeq.magnitude.diagnostics_extended import (
    MC_MAXC_CORRECTION,
    _sample_event_ids,
    b_value_mle,
    gr_statistics,
    mc_max_curvature,
    plot_gr_mc_bvalue,
    plot_residual_vs_distance,
)


def _synthetic_gr(n: int, b: float, m_min: float, seed: int = 1) -> np.ndarray:
    """Continuous GR sample above ``m_min``: M - m_min ~ Exp(beta = b ln 10)."""
    rng = np.random.default_rng(seed)
    return m_min + rng.exponential(1.0 / (b * np.log(10.0)), size=n)


def test_b_value_mle_recovers_b_of_one():
    mags = _synthetic_gr(3000, b=1.0, m_min=0.95)
    stats, boot = gr_statistics(mags)

    assert stats.status == "ok"
    assert stats.b == pytest.approx(1.0, abs=0.08)
    assert 0.0 < stats.b_std < 0.1
    assert stats.mc == pytest.approx(1.0 + MC_MAXC_CORRECTION, abs=0.11)
    assert len(boot) > 150
    # Direct estimator agrees at the same Mc.
    assert b_value_mle(mags, stats.mc) == pytest.approx(stats.b)


def test_gr_statistics_deterministic():
    mags = _synthetic_gr(500, b=1.2, m_min=0.5, seed=7)
    a, _ = gr_statistics(mags)
    b, _ = gr_statistics(mags)
    assert a == b


def test_mc_max_curvature_empty_is_nan():
    assert np.isnan(mc_max_curvature([]))


def test_gr_tiny_catalogue_insufficient(tmp_path):
    png, csv = tmp_path / "gr.png", tmp_path / "gr.csv"
    stats = plot_gr_mc_bvalue(np.array([1.0, 1.2, 1.5, 2.0, 0.8]), png, csv)

    assert stats.status == "insufficient events"
    assert np.isnan(stats.b)
    assert png.is_file() and png.stat().st_size > 0
    row = pd.read_csv(csv).iloc[0]
    assert row["status"] == "insufficient events"
    assert row["n_events"] == 5


def test_gr_empty_catalogue(tmp_path):
    stats = plot_gr_mc_bvalue(np.array([]), tmp_path / "gr.png", tmp_path / "gr.csv")
    assert stats.status == "no data"
    assert (tmp_path / "gr.png").is_file()


def test_gr_fit_writes_csv(tmp_path):
    stats = plot_gr_mc_bvalue(
        _synthetic_gr(400, 1.0, 0.95), tmp_path / "gr.png", tmp_path / "gr.csv"
    )
    row = pd.read_csv(tmp_path / "gr.csv").iloc[0]
    assert row["status"] == "ok"
    assert row["b"] == pytest.approx(stats.b)
    assert row["n_bootstrap"] == 200


def test_residual_vs_distance(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    station_df = pd.DataFrame(
        {
            "event_id": np.repeat([f"ev{i}" for i in range(10)], 6),
            "net": "XX",
            "sta": np.tile([f"S{j}" for j in range(6)], 10),
            "distance_km": rng.uniform(5, 100, n),
            "component_used": rng.choice(["HHN", "HHE", "HHZ"], n),
            "station_ml": rng.normal(1.5, 0.2, n),
            "included": True,
        }
    )
    event_table = pd.DataFrame({"event_id": [f"ev{i}" for i in range(10)], "ml": 1.5})
    plot_residual_vs_distance(station_df, event_table, tmp_path / "r.png")
    plot_residual_vs_distance(station_df.iloc[:0], event_table, tmp_path / "empty.png")
    assert (tmp_path / "r.png").is_file()
    assert (tmp_path / "empty.png").is_file()


def test_sample_event_ids_even_and_capped():
    df = pd.DataFrame({"event_id": np.repeat([f"ev{i:02d}" for i in range(21)], 2)})
    assert _sample_event_ids(df, 3) == ["ev00", "ev10", "ev20"]
    assert _sample_event_ids(df.iloc[:4], 10) == ["ev00", "ev01"]
    assert _sample_event_ids(pd.DataFrame(), 5) == []
