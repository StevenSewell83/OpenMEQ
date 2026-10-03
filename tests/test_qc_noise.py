"""Tests for the hourly noise statistics, instrument table and noise figures (OpenMEQ-h8j)."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from obspy import Trace, UTCDateTime
from obspy.core.inventory import (
    Channel,
    InstrumentSensitivity,
    Inventory,
    Network,
    Response,
    Station,
)
from obspy.core.inventory.util import Equipment

from openmeq.config import QcNoiseConfig
from openmeq.qc.instruments import instrument_table
from openmeq.qc.noise import NOISE_COLUMNS, hourly_noise_stats
from openmeq.qc.noise_plots import make_noise_diagnostics, noise_table

DAY = date(2026, 1, 1)
FS = 100.0


def _trace(data: np.ndarray, start_s: float = 0.0, station: str = "STA1") -> Trace:
    tr = Trace(data=np.asarray(data, dtype=np.float64))
    tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel = (
        "XX",
        station,
        "",
        "EHZ",
    )
    tr.stats.sampling_rate = FS
    tr.stats.starttime = UTCDateTime(DAY.isoformat()) + start_s
    return tr


def _sine(n: int, amplitude: float, freq: float = 10.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / FS)


# ---------------------------------------------------------------------------
# hourly_noise_stats
# ---------------------------------------------------------------------------


def test_sine_in_core_band_gives_known_level_kurtosis_and_crest():
    amplitude = 1000.0
    row = hourly_noise_stats([_trace(_sine(int(86400 * FS), amplitude))], DAY, QcNoiseConfig())

    assert set(row) == set(NOISE_COLUMNS)
    assert row["noise_n_windows"] == 24
    assert row["noise_completeness"] == pytest.approx(1.0)
    # A 10 Hz sine passes the 5-25 Hz filter unchanged: RMS = A / sqrt(2).
    assert row["noise_rms_db_median"] == pytest.approx(
        20 * math.log10(amplitude / math.sqrt(2)), abs=0.2
    )
    assert row["noise_spread_db"] == pytest.approx(0.0, abs=0.2)
    assert row["noise_kurt_median"] == pytest.approx(1.5, abs=0.05)  # a sine's kurtosis
    assert row["noise_crest_db_median"] == pytest.approx(20 * math.log10(math.sqrt(2)), abs=0.2)
    assert row["noise_ratio_low_core_db"] < -20  # nothing in 1-5 Hz
    assert row["noise_dead_frac"] == 0.0


def test_half_day_of_data_gives_half_completeness_and_twelve_windows():
    row = hourly_noise_stats(
        [_trace(_sine(int(43200 * FS), 100.0), start_s=43200.0)], DAY, QcNoiseConfig()
    )
    assert row["noise_completeness"] == pytest.approx(0.5)
    assert row["noise_n_windows"] == 12


def test_two_segments_fill_their_own_windows():
    first = _trace(_sine(int(3 * 3600 * FS), 100.0))
    second = _trace(_sine(int(2 * 3600 * FS), 100.0), start_s=10 * 3600.0)
    row = hourly_noise_stats([first, second], DAY, QcNoiseConfig())
    assert row["noise_n_windows"] == 5


def test_flatlined_channel_is_dead_not_quiet():
    row = hourly_noise_stats([_trace(np.zeros(int(86400 * FS)))], DAY, QcNoiseConfig())
    assert row["noise_dead_frac"] == 1.0
    assert math.isnan(row["noise_rms_db_median"])


def test_core_band_above_nyquist_returns_empty_row():
    cfg = QcNoiseConfig(core_band=(5.0, 60.0))  # Nyquist of 100 Hz data is 50 Hz
    row = hourly_noise_stats([_trace(_sine(int(3600 * FS), 100.0))], DAY, cfg)
    assert row["noise_n_windows"] == 0
    assert math.isnan(row["noise_rms_db_median"])


def test_no_traces_returns_empty_row():
    row = hourly_noise_stats([], DAY, QcNoiseConfig())
    assert row["noise_completeness"] == 0.0
    assert row["noise_n_windows"] == 0


def test_window_must_divide_a_day():
    with pytest.raises(ValueError, match="divide a day"):
        QcNoiseConfig(window_s=7000.0)


# ---------------------------------------------------------------------------
# instrument_table
# ---------------------------------------------------------------------------


def _channel(code: str, sensor: str, sens: float | None, start: str) -> Channel:
    ch = Channel(code, "", 0.0, 0.0, 0.0, 0.0, start_date=UTCDateTime(start), sample_rate=FS)
    ch.sensor = Equipment(model=sensor)
    if sens is not None:
        ch.response = Response(
            instrument_sensitivity=InstrumentSensitivity(sens, 10.0, "M/S", "COUNTS")
        )
    return ch


def _inventory() -> Inventory:
    sta1 = Station(
        "STA1",
        0.0,
        0.0,
        0.0,
        channels=[
            _channel("EHZ", "OLD-SENSOR", 1e8, "2008-01-01"),
            _channel("EHZ", "GS-11D", 2e8, "2008-06-01"),  # later epoch, also open-ended
        ],
    )
    sta2 = Station("STA2", 0.0, 0.0, 0.0, channels=[_channel("HQZ", "IESE", -3e7, "2010-01-01")])
    sta3 = Station("STA3", 0.0, 0.0, 0.0, channels=[_channel("EHZ", "GS-11D", None, "2010-01-01")])
    return Inventory(networks=[Network("XX", stations=[sta1, sta2, sta3])])


def test_instrument_table_epochs_overrides_and_fallbacks():
    seed_ids = ["XX.STA1..EHZ", "XX.STA2..EHZ", "XX.STA3..EHZ", "XX.STA4..EHZ"]
    table = instrument_table(
        _inventory(), seed_ids, UTCDateTime("2026-02-15"), {"MY-GROUP": ["XX.STA3"]}
    ).set_index("seed_id")

    # Latest-starting open epoch wins.
    assert table.loc["XX.STA1..EHZ", "group"] == "GS-11D"
    assert table.loc["XX.STA1..EHZ", "sensitivity"] == pytest.approx(2e8)
    assert table.loc["XX.STA1..EHZ", "sens_flag"] == "ok"
    # EHZ data, HQZ metadata: same component; the sign is dropped.
    assert table.loc["XX.STA2..EHZ", "group"] == "IESE"
    assert table.loc["XX.STA2..EHZ", "sensitivity"] == pytest.approx(3e7)
    assert table.loc["XX.STA2..EHZ", "sens_flag"] == "assumed_component"
    # Config override; no response -> missing sensitivity.
    assert table.loc["XX.STA3..EHZ", "group"] == "MY-GROUP"
    assert table.loc["XX.STA3..EHZ", "group_source"] == "config"
    assert table.loc["XX.STA3..EHZ", "sens_flag"] == "missing"
    assert math.isnan(table.loc["XX.STA3..EHZ", "sensitivity"])
    # No metadata at all.
    assert table.loc["XX.STA4..EHZ", "group"] == "unknown"


# ---------------------------------------------------------------------------
# noise_table / make_noise_diagnostics
# ---------------------------------------------------------------------------


def _daily_stats(seed_ids: list[str], n_days: int = 5) -> pd.DataFrame:
    rows = []
    for i, seed_id in enumerate(seed_ids):
        for d in range(n_days):
            row = {c: 1.0 for c in NOISE_COLUMNS}
            row.update(
                seed_id=seed_id,
                date=f"2026-01-{d + 1:02d}",
                noise_completeness=1.0,
                noise_n_windows=24,
                noise_rms_db_median=100.0 + i,
                noise_rms_db_p5=98.0 + i,
                noise_rms_db_p95=104.0 + i,
                noise_kurt_p95=5.0 + d,
                noise_crest_db_p95=15.0,
                noise_dead_frac=0.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_noise_table_converts_levels_with_sensitivity_only():
    seed_ids = ["XX.STA1..EHZ", "XX.STA3..EHZ"]
    instruments = instrument_table(_inventory(), seed_ids, UTCDateTime("2026-01-03"))
    df = noise_table(_daily_stats(seed_ids), instruments).set_index("seed_id")

    sta1 = df.loc["XX.STA1..EHZ"].iloc[0]
    assert sta1["rms_db_median"] == pytest.approx(100.0 - 20 * math.log10(2e8))
    assert sta1["units"] == "dB re 1 m/s"
    assert sta1["spread_db"] == 1.0  # unchanged by the sensitivity
    sta3 = df.loc["XX.STA3..EHZ"].iloc[0]
    assert sta3["rms_db_median"] == pytest.approx(101.0)  # no sensitivity: still counts
    assert sta3["units"] == "dB re 1 count"


def test_make_noise_diagnostics_writes_one_set_per_group(tmp_path):
    seed_ids = ["XX.STA1..EHZ", "XX.STA2..EHZ", "XX.STA3..EHZ", "XX.STA1..EHN"]
    instruments = instrument_table(_inventory(), seed_ids, UTCDateTime("2026-01-03"))
    df = noise_table(_daily_stats(seed_ids), instruments)

    written = make_noise_diagnostics(
        df,
        tmp_path,
        start=date(2026, 1, 1),
        end=date(2026, 1, 7),
        noise_cfg=QcNoiseConfig(),
        title="t",
    )

    names = sorted(p.name for p in written)
    assert names == sorted(
        [f"{g}_{k}.png" for g in ("gs-11d", "iese") for k in ("heatmap", "strips", "overlay")]
        + ["impulsiveness.png"]
    )
    assert all(p.is_file() for p in written)


def test_group_without_levels_is_skipped(tmp_path):
    seed_ids = ["XX.STA1..EHZ", "XX.STA2..EHZ"]
    stats = _daily_stats(seed_ids)
    stats.loc[stats.seed_id == "XX.STA2..EHZ", "noise_rms_db_median"] = np.nan
    df = noise_table(stats, instrument_table(_inventory(), seed_ids, UTCDateTime("2026-01-03")))

    written = make_noise_diagnostics(
        df, tmp_path, start=date(2026, 1, 1), end=date(2026, 1, 5), noise_cfg=QcNoiseConfig()
    )

    assert not any(p.name.startswith("iese_") for p in written)
    assert (tmp_path / "gs-11d_heatmap.png").is_file()
