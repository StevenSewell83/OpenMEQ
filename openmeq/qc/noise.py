"""Hourly-window band-limited noise statistics per channel-day (OpenMEQ-h8j).

Ported from MicroQuake's ``scripts/plot_daily_noise_stats.py`` (the
``projects/Olkaria/noise_daily`` figures), so both tools report the same
numbers. Each channel-day is cut into ``cfg.qc.noise.window_s`` windows
(hourly by default); every window gets a band-limited RMS in the core band
(default 5-25 Hz, above the corner of every common short-period sensor) and
the low band (default 1-5 Hz). The day is then summarised across its windows:

- ``noise_rms_db_median`` -- the site noise floor, robust to events/spikes.
- ``noise_rms_db_p5``/``_p95`` and ``noise_spread_db`` (p95 - p5) -- separates
  a floor change from transient bursts.
- ``noise_ratio_low_core_db`` -- median low/core band ratio, a per-station
  response/coupling drift tracker (not comparable between sensor models).
- ``noise_kurt_*`` (Gaussian noise = 3) and ``noise_crest_db_*`` (peak/RMS;
  a Gaussian hour is ~12-13 dB) -- impulsiveness. Both are scale-invariant,
  so they compare across instruments without any response.
- ``noise_completeness``, ``noise_n_windows`` and ``noise_dead_frac``
  (windows whose core-band RMS is below ``dead_rms_counts``: flatlined
  hardware, not quiet ground).

Levels here are **dB re 1 count**. :mod:`openmeq.qc.noise_plots` converts
them to dB re 1 m/s with each channel's scalar sensitivity: dividing every
sample by a constant shifts a dB level by exactly ``-20 log10(sens)``, so the
waveform pass needs no metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
from obspy import UTCDateTime
from scipy.signal import butter, sosfilt

from openmeq.log import get_logger

logger = get_logger("qc.noise")

#: Columns this module adds to ``qc/daily_stats.csv``.
NOISE_COLUMNS = [
    "noise_completeness",
    "noise_n_windows",
    "noise_rms_db_median",
    "noise_rms_db_p5",
    "noise_rms_db_p95",
    "noise_spread_db",
    "noise_ratio_low_core_db",
    "noise_kurt_median",
    "noise_kurt_p95",
    "noise_crest_db_median",
    "noise_crest_db_p95",
    "noise_dead_frac",
]

_TINY = 1e-30


def empty_noise_stats() -> dict:
    """Noise columns for a channel-day with no usable samples."""
    row = {col: float("nan") for col in NOISE_COLUMNS}
    row["noise_completeness"] = 0.0
    row["noise_n_windows"] = 0
    return row


def _band_ok(band: Sequence[float], fs: float) -> bool:
    return 0.0 < band[0] < band[1] < 0.5 * fs


def hourly_noise_stats(traces, day: date, noise_cfg) -> dict:
    """Noise columns for one channel-day.

    Parameters
    ----------
    traces : iterable of obspy.Trace
        The day's contiguous, unmasked segments (one seed id, one sampling
        rate). Each is filtered on its own, so a gap never joins two
        segments inside the filter.
    day : datetime.date
        The calendar day (UTC) the windows are laid on.
    noise_cfg : openmeq.config.QcNoiseConfig

    Returns
    -------
    dict
        Keys :data:`NOISE_COLUMNS`.
    """
    traces = [tr for tr in traces if tr.stats.npts > 0]
    if not traces:
        return empty_noise_stats()

    fs = float(traces[0].stats.sampling_rate)
    window_s = float(noise_cfg.window_s)
    n_win = int(round(86400.0 / window_s))
    expected = window_s * fs
    core_band = tuple(noise_cfg.core_band)
    low_band = tuple(noise_cfg.low_band)
    if not _band_ok(core_band, fs):
        logger.warning(
            "%s %s: core band %s Hz not below Nyquist (%.1f Hz); noise stats skipped",
            traces[0].id,
            day,
            core_band,
            0.5 * fs,
        )
        return empty_noise_stats()
    sos_core = butter(4, core_band, btype="bandpass", fs=fs, output="sos")
    sos_low = (
        butter(4, low_band, btype="bandpass", fs=fs, output="sos")
        if _band_ok(low_band, fs)
        else None
    )

    ss_core = np.zeros(n_win)
    s4_core = np.zeros(n_win)
    max_core = np.zeros(n_win)
    ss_low = np.zeros(n_win)
    n_samp = np.zeros(n_win)
    day_start = UTCDateTime(day.isoformat())

    for tr in traces:
        if float(tr.stats.sampling_rate) != fs:
            continue
        data = np.asarray(tr.data, dtype=np.float64)
        data = data - data.mean()
        offs = float(tr.stats.starttime - day_start)
        # A trace is contiguous, so each window is one slice [i0, i1) of it:
        # sample i is at offs + i/fs, in window floor((offs + i/fs) / window_s).
        bounds = np.ceil((np.arange(n_win + 1) * window_s - offs) * fs).astype(np.int64)
        bounds = np.clip(bounds, 0, data.size)
        if bounds[0] == bounds[-1]:
            continue
        # Filter one band at a time so only one extra day-length array is
        # alive at once (a 200 Hz day is ~140 MB of float64).
        core = sosfilt(sos_core, data)
        for w in range(n_win):
            seg = core[bounds[w] : bounds[w + 1]]
            if seg.size == 0:
                continue
            sq = seg * seg
            ss_core[w] += float(sq.sum())
            s4_core[w] += float(np.dot(sq, sq))
            max_core[w] = max(max_core[w], float(np.abs(seg).max()))
            n_samp[w] += seg.size
        del core
        if sos_low is not None:
            low = sosfilt(sos_low, data)
            for w in range(n_win):
                seg = low[bounds[w] : bounds[w + 1]]
                ss_low[w] += float(np.dot(seg, seg))
            del low

    row = empty_noise_stats()
    row["noise_completeness"] = float(min(n_samp.sum() / (expected * n_win), 1.0))
    valid = n_samp >= noise_cfg.min_window_frac * expected
    row["noise_n_windows"] = int(valid.sum())
    if not valid.any():
        return row

    n = n_samp[valid]
    core_rms = np.sqrt(ss_core[valid] / n)
    dead = core_rms < noise_cfg.dead_rms_counts
    row["noise_dead_frac"] = float(dead.mean())
    live = ~dead
    if not live.any():
        return row

    db = 20.0 * np.log10(np.maximum(core_rms[live], _TINY))
    p5, p50, p95 = np.percentile(db, [5, 50, 95])
    row["noise_rms_db_median"] = float(p50)
    row["noise_rms_db_p5"] = float(p5)
    row["noise_rms_db_p95"] = float(p95)
    row["noise_spread_db"] = float(p95 - p5)
    if sos_low is not None:
        low_rms = np.sqrt(ss_low[valid][live] / n[live])
        ratio = 20.0 * np.log10(np.maximum(low_rms, _TINY) / np.maximum(core_rms[live], _TINY))
        row["noise_ratio_low_core_db"] = float(np.median(ratio))
    kurt = n[live] * s4_core[valid][live] / np.maximum(ss_core[valid][live] ** 2, _TINY)
    crest = 20.0 * np.log10(
        np.maximum(max_core[valid][live], _TINY) / np.maximum(core_rms[live], _TINY)
    )
    row["noise_kurt_median"] = float(np.median(kurt))
    row["noise_kurt_p95"] = float(np.percentile(kurt, 95))
    row["noise_crest_db_median"] = float(np.median(crest))
    row["noise_crest_db_p95"] = float(np.percentile(crest, 95))
    return row
