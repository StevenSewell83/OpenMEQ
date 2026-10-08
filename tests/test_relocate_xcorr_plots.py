"""Tests for openmeq.relocate.xcorr_plots (obspy xcorr_pick_correction QC plots)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import ValidationError

from openmeq.config import DiagnosticsConfig
from openmeq.relocate.xcorr_plots import _plot_pair_item, select_pairs

SR = 200.0
START = 1_700_000_000.0
SEED = "XX.STA1.00.HHZ"


def _write_cache(path, pulse_at: float) -> None:
    """One-channel ``relocate/cache``-style .npz with a pulse at START + pulse_at."""
    t = np.arange(int(4 * SR)) / SR
    data = np.exp(-(((t - pulse_at) / 0.02) ** 2)) * np.sin(2 * np.pi * 10 * (t - pulse_at))
    data += 0.01 * np.random.default_rng(0).standard_normal(t.size)
    np.savez(
        path,
        seed_ids=np.array([SEED]),
        sampling_rate=np.array(SR),
        **{f"{SEED}__data": data, f"{SEED}__start": np.array(START)},
    )


def test_select_pairs_interval_and_cap():
    pairs = [(i, i + 1) for i in range(10)]
    assert select_pairs(pairs, 3, 100) == [(0, (0, 1)), (3, (3, 4)), (6, (6, 7)), (9, (9, 10))]
    assert select_pairs(pairs, 3, 2) == [(0, (0, 1)), (3, (3, 4))]
    assert len(select_pairs(pairs, 1, 100)) == 10


def test_xcorr_plots_config_defaults_and_bounds():
    xp = DiagnosticsConfig().relocate.xcorr_plots
    assert (xp.enabled, xp.pair_interval, xp.max_pairs) == (False, 100, 20)
    with pytest.raises(ValidationError):
        DiagnosticsConfig(relocate={"xcorr_plots": {"pair_interval": 0}})


def test_plot_pair_item_recovers_shift_and_writes_png(tmp_path):
    from obspy import UTCDateTime

    _write_cache(tmp_path / "a.npz", 2.0)
    _write_cache(tmp_path / "b.npz", 2.0)
    out_dir = tmp_path / "xcorr_pairs"
    out_dir.mkdir(parents=True)
    # Pick 2 is 15 ms late relative to its waveform: correction should be ~ -0.015 s.
    item = {
        "pair_index": 0,
        "cuspid_i": 1,
        "cuspid_j": 2,
        "event_i": "a",
        "event_j": "b",
        "npz_i": str(tmp_path / "a.npz"),
        "npz_j": str(tmp_path / "b.npz"),
        "out_dir": str(out_dir),
        "picks_i": {"XX.STA1": {"P": str(UTCDateTime(START + 2.0))}},
        "picks_j": {"XX.STA1": {"P": str(UTCDateTime(START + 2.015))}},
        "station_phases": [("STA1", "XX.STA1", "P"), ("STA1", "XX.STA1", "S")],
        "kept": {"STA1|P": (0.95, 0.01)},
        "window_p": (0.05, 0.2),
        "window_s": (0.05, 0.3),
        "maxlag_s": 0.05,
        "components": {"P": ["Z"], "S": ["N", "E"]},
    }
    figs_before = set(plt.get_fignums())
    rows = _plot_pair_item(item)
    p_row = next(r for r in rows if r["phase"] == "P")
    assert p_row["obspy_pick2_corr_s"] == pytest.approx(-0.015, abs=0.003)
    assert p_row["obspy_cc"] > 0.9
    assert p_row["in_dtcc"] is True
    assert p_row["png"] == "a__b_STA1_P_HHZ.png"
    assert (out_dir / "a__b_STA1_P_HHZ.png").stat().st_size > 0
    s_row = next(r for r in rows if r["phase"] == "S")
    assert s_row["error"] == "pick missing at one event"
    assert set(plt.get_fignums()) <= figs_before  # closes every figure it opens
