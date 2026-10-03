"""Daily noise statistics per channel-day (M1.4, ``OpenMEQ-5nl.4``; spike
metric and longest-flat-run column revised in M1.9, ``OpenMEQ-5nl.9``).

PLAN.md §6 Step 1's "Daily noise statistics per channel" (RMS, mean absolute
amplitude, MAD-based spikiness, kurtosis, flat-lined and clipped sample
counts), written to ``qc/daily_stats.csv`` (ARCHITECTURE.md §3):
``seed_id,date,rms,mean_abs,spike_count,kurtosis,flat_count,longest_flat_run_s,
clip_count,npts``.

Isolated-impulse spike metric (M1.9)
-------------------------------------
M1.5's real-data findings (``docs/performance.md``/the M1.9 issue) showed
that the original "large samples vs. the day's global MAD" definition of
``spike_count`` cannot tell a glitch from an earthquake: a real seismic
event pushes thousands of samples past a global MAD threshold too (the
whole point of an event is that it's a large deviation from background),
so every channel's ``spike_count`` on a day with real events was already
1300-1850, swamping the ~600-above-baseline signature of the deliberately
spiky synthetic channel.

``spike_count`` is now an **isolated-impulse** count: earthquakes are smooth
over several consecutive samples (a real seismic pulse has most of its
energy below the Nyquist-adjacent band and rides on top of, not instead of,
its neighbours), so a short running median tracks straight through an
event's samples with very little per-sample deviation. An electronic
glitch/spike, by construction, differs sharply from its immediate
neighbours, so it leaves a large residual against that same running
median. Concretely (the running median in
:func:`compute_channel_day_stats`, the threshold/count in
:func:`_spike_count_from_residual`):

    med_i = running_median(x, k=cfg.qc.spike_median_window)[i]   # k~5-7, odd
    r_i   = x_i - med_i
    spike_count = count(|r_i| > cfg.qc.spike_threshold_mad * MAD(r))

``MAD(r)`` here is ``median(|r|)`` (not re-centered on ``median(r)``): the
residual is already centered near zero by construction (subtracting a
local running median removes any local trend/DC level), so re-centering
would cost a second full-array ``np.median`` pass (~100 ms on a full
17.28M-sample day) for a correction that is negligible in practice --
documented approximation, not a silent one (see
:func:`_spike_count_from_residual`'s docstring).

The running median itself is :func:`scipy.ndimage.median_filter`
(``size=cfg.qc.spike_median_window``, ``mode="nearest"`` so the day's first/
last ``k//2`` samples reuse the edge value rather than wrapping or
reflecting past the day boundary) -- see ``docs/performance.md`` §9 for the
benchmark this choice is based on (scipy's rank-order filter beat both a
hand-rolled sorting-network-of-7 and a strided-window ``np.partition``
approach by a wide margin) and for why the candidate-window/subsampled-MAD
optimisation suggested by the issue was benchmarked but not shipped as the
default (it trades the "identical, or within documented float tolerance"
correctness bar this module holds itself to -- see OpenMEQ-5nl.8's own
"considered and rejected" precedent -- for a speedup that the full-array
``median_filter`` approach turned out not to need to stay fast enough).

Parallelism
-----------
PLAN.md §4's "QC | channel x day" row is still the logical unit of work,
but the ``pmap`` *task* granularity is one seed id x a batch of up to
``cfg.parallel.chunk_days`` consecutive days (:func:`_stats_batch`),
mirroring ``openmeq.qc.availability``'s M1.3 batching exactly
(OpenMEQ-5nl.8: batching amortises per-task ``ProcessPoolExecutor``
scheduling/IPC overhead at production scale). The top-level, picklable
per-channel-day worker is still :func:`stats_for_channel_day`, which takes
a ``seed_id``/``day`` pair plus two *paths* -- never a waveform object or
the in-memory :class:`~openmeq.config.Config`/:class:`~openmeq.index.DataIndex`
themselves. Both the index and the small ``qc:``-only config are loaded
**once per worker process** and cached thereafter
(:func:`_cached_index`/:func:`_cached_qc_cfg`, OpenMEQ-5nl.8) rather than
reloaded on every call -- a ``ProcessPoolExecutor`` worker is long-lived
and handles many items over its lifetime, all sharing the same index/
config for one ``compute_daily_stats`` call; see ``docs/performance.md``
for the production-scale evidence this mattered for.
:func:`compute_daily_stats` is the orchestration entry point: it takes the
already-loaded ``index``/``cfg`` (for a convenient, direct API), and
internally bridges to the path-based worker contract.

``cfg_path`` design note
-------------------------
:func:`stats_for_channel_day` only ever needs ``cfg.qc.spike_threshold_mad``,
``cfg.qc.spike_median_window``, ``cfg.qc.flat_run_min_samples`` and
``cfg.qc.clip_run_min_samples`` -- not the rest of the (much larger) project
:class:`~openmeq.config.Config`
(NLLoc grids, hypoDD weighting tables, ...), and those unrelated sections
don't all round-trip cleanly through a bare YAML dump (e.g. the
``station_corrections: dict | Path`` union). So ``cfg_path`` here is a
small, self-contained YAML file holding just the ``qc:`` section (in
exactly ``config/example.yaml``'s documented shape), written once by
:func:`compute_daily_stats` to a temp file and loaded with
``QcConfig.model_validate`` inside the worker -- not
``openmeq.config.load_config`` on the user's actual project config. This
keeps worker IPC payloads tiny (a handful of scalars) and avoids coupling
this module to config sections it has no business reading, while still
following the "workers receive paths, not objects" rule.

Masked arrays / gaps
---------------------
``DataIndex.read`` merges with ``stream.merge(method=0)``, which yields a
masked array across any real time gap (ARCHITECTURE.md §2). Every
statistic here needs an explicit compress step first --
:meth:`obspy.Stream.split` breaks a gappy stream into fully-contiguous,
non-masked per-segment traces (one gap channel-day may yield 2+ segments).
Per-segment results are then pooled:

- ``rms``/``mean_abs``: additive-safe -- accumulate ``sum(x**2)`` /
  ``sum(abs(x))`` and total ``npts`` per segment, then divide once at the
  end. This is arithmetically identical to concatenating first (it's just
  computed via partial sums), so it agrees with a direct NumPy calculation
  on ungapped data to floating-point tolerance.
- kurtosis: **not** additive across segments (a moment of moments is not
  the true moment). Computed once on the concatenation of every segment's
  valid samples for the day -- acceptable at this dataset's scale (a
  documented memory caveat for very high-rate/long-duration production
  channels, out of scope to optimise here; a streaming/approximate
  two-pass alternative would be needed then).
- isolated-impulse ``spike_count`` (M1.9): the *residual* ``r = x -
  running_median(x, k)`` is computed **per segment** (a running-median
  window must never be allowed to span a real time gap -- see
  :func:`compute_channel_day_stats`), but the residuals are then pooled
  (concatenated) before the single MAD/threshold/count pass, since a count
  over a concatenation of independently-computed residuals is additive
  even though a median itself is not.
- flat-run / clip-run / longest-flat-run counts: **must** stay per-segment
  (a run cannot legitimately span a real time gap), computed with a
  vectorised run-length
  encoding (no per-sample Python loop).

Clip detection order
----------------------
ObsPy decodes STEIM-compressed miniSEED into native ``int32`` NumPy arrays.
The clip check runs against each segment's *original* ``trace.data``
(``dtype.kind == "i"`` -> compare directly against ``2**31 - 1`` /
``-(2**31)``) before any cast to ``float64`` for the other statistics, so an
in-range int32 value is never mistakenly compared as if it had been
losslessly rescaled. If the data is already float (not this project's
STEIM-decoded case, but a documented fallback), the saturation value is
instead the single most extreme-magnitude sample observed anywhere in that
channel-day, and a "clipped" run is one of >= ``clip_run_min_samples``
consecutive samples exactly equal to that value.

Thresholds are MAD-based (``qc.spike_threshold_mad``), not tuned to any one
noise model, so they stay meaningful if the synthetic background-noise
generator changes (e.g. to coloured noise, OpenMEQ-qlg.10) -- median/MAD are
robust to the noise's exact distribution shape in a way a fixed-amplitude
threshold would not be.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from obspy import UTCDateTime
from scipy.ndimage import median_filter

from openmeq.config import Config, QcConfig
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.paths import RunPaths
from openmeq.qc.noise import NOISE_COLUMNS, empty_noise_stats, hourly_noise_stats

logger = get_logger("qc.stats")

#: Exact column order, ARCHITECTURE.md §3's ``qc/daily_stats.csv`` row.
DAILY_STATS_COLUMNS = [
    "seed_id",
    "date",
    "rms",
    "mean_abs",
    "spike_count",
    "kurtosis",
    "flat_count",
    "longest_flat_run_s",
    "clip_count",
    "npts",
    *NOISE_COLUMNS,
]

#: int32 full-scale saturation values (miniSEED/STEIM-decoded integer data).
_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)

#: Tiny epsilon (s) subtracted from a day's exclusive upper bound so the
#: read window is ``[00:00:00.000000, 24:00:00.000000)`` and never pulls in
#: the next day's first sample, regardless of the channel's sampling rate.
_DAY_END_EPS = 1e-6


def _day_window(day: date) -> tuple[UTCDateTime, UTCDateTime]:
    """UTC ``[day_start, day_end)`` window for a calendar ``day``."""
    day_start = UTCDateTime(day.year, day.month, day.day)
    day_end = day_start + 86400.0 - _DAY_END_EPS
    return day_start, day_end


def _segment_run_counts(
    x: np.ndarray,
    *,
    flat_run_min_samples: int,
    clip_run_min_samples: int,
    is_int: bool,
    float_saturation_value: float | None,
) -> tuple[int, int, int]:
    """``(flat_count, clip_count, longest_run_samples)`` for one contiguous
    segment's raw samples.

    ``longest_run_samples`` (M1.9, ``OpenMEQ-5nl.9``) is the length of the
    single longest run of an identical consecutive value in this segment --
    the same equality-based run definition ``flat_count`` uses, but the
    *maximum* run length rather than the total sample count across every
    run that clears ``flat_run_min_samples``. It backs the
    ``longest_flat_run_s`` column (a real fault -- e.g. a dead sensor for an
    hour -- should flag on its own *duration*, not only once it accumulates
    enough total flat samples to cross ``max_flat_fraction`` of the whole
    day). Background noise essentially never repeats the same raw sample
    value even ``flat_run_min_samples`` times in a row, so an ordinary
    channel/day's ``longest_run_samples`` is typically 1-3, regardless of
    ``flat_run_min_samples``.

    ``x`` is the segment's *original* dtype array (int32 in the normal
    miniSEED case) -- equality-based run detection, so no float cast is
    needed (or wanted, for the clip check -- see the module docstring).

    Run-length via *boundaries*, not a per-sample expansion
    ----------------------------------------------------------------------
    The original implementation (see git history) built a full ``run_len``
    array the same size as ``x`` (one ``np.bincount`` plus a fancy-index
    gather, ``run_len[run_id]``, back out to per-sample size) purely to
    then threshold and count it. Profiling on a full 200 Hz channel-day
    (17.28M samples) showed that per-sample expansion -- not the O(n)
    boundary scan itself, which is unavoidable -- was the dominant cost
    (OpenMEQ-5nl.8). This version never re-expands past run granularity:
    ``run_starts``/``run_lengths`` are sized by the *number of runs*
    (typically close to ``n`` for real noise, since consecutive raw
    samples rarely repeat exactly, but still cheaper than an explicit
    per-sample gather), and the qualifying-run totals are two masked
    ``.sum()`` reductions over that run-sized array -- no array the size
    of ``x`` is ever built except the single boundary-comparison pass.
    """
    n = x.size
    if n == 0:
        return 0, 0, 0

    is_run_start = np.empty(n, dtype=bool)
    is_run_start[0] = True
    np.not_equal(x[1:], x[:-1], out=is_run_start[1:])
    run_starts = np.flatnonzero(is_run_start)
    # Equivalent to ``np.diff(run_starts, append=n)`` but a little cheaper:
    # ``numpy.diff`` is a generic, axis-aware wrapper (extra overhead for a
    # 1-D, no-``prepend``, single-``append`` case); this writes directly
    # into a pre-sized output array via plain slicing/subtraction.
    run_lengths = np.empty(run_starts.size, dtype=np.int64)
    run_lengths[:-1] = run_starts[1:]
    run_lengths[:-1] -= run_starts[:-1]
    run_lengths[-1] = n - run_starts[-1]

    flat_count = int(run_lengths[run_lengths >= flat_run_min_samples].sum())
    longest_run_samples = int(run_lengths.max())

    run_values = x[run_starts]
    if is_int:
        is_saturation_value = (run_values == _INT32_MAX) | (run_values == _INT32_MIN)
    elif float_saturation_value is not None:
        is_saturation_value = run_values == float_saturation_value
    else:
        is_saturation_value = np.zeros(run_values.shape, dtype=bool)

    clip_mask = (run_lengths >= clip_run_min_samples) & is_saturation_value
    clip_count = int(run_lengths[clip_mask].sum())
    return flat_count, clip_count, longest_run_samples


def _clamped_odd_window(window: int, n: int) -> int:
    """``window`` clamped to ``[1, n]`` and forced odd (a running-median
    filter needs an odd, centered footprint). Only matters for a very
    short trailing/leading segment right next to a gap -- the overwhelming
    majority of segments are a full day's worth of samples, far longer
    than ``spike_median_window`` (default 7)."""
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    return max(w, 1)


def _spike_count_from_residual(r_all: np.ndarray, spike_threshold_mad: float) -> int:
    """Isolated-impulse ``spike_count`` (M1.9, ``OpenMEQ-5nl.9``) from a
    channel-day's already-computed residual ``r = x - running_median(x, k)``
    (see the module docstring's "Isolated-impulse spike metric" section for
    the rationale, and :func:`compute_channel_day_stats` for where ``r`` is
    computed *per segment*, then pooled, before this is called).

        mad = median(abs(r))            # see "MAD(r) convention" below
        spike_count = count(abs(r) > spike_threshold_mad * mad)

    ``MAD(r)`` convention -- deliberately *not* re-centered
    -------------------------------------------------------
    Unlike the day's raw-sample MAD (the metric this replaces computed
    ``median(abs(x - median(x)))``, i.e. centered on ``x``'s own median),
    this uses ``median(abs(r))`` directly, with no ``r - median(r)``
    re-centering step. ``r`` is already centered near zero *by
    construction*: subtracting a local running median removes any local
    trend or DC offset a raw sample array might have. Re-centering would
    need a second full-array :func:`numpy.median` call (~100 ms on a full
    17.28M-sample day, comparable to the running-median filter itself --
    see ``docs/performance.md``), for a correction that every benchmark
    channel/day in this fixture's real synthetic data changes by a
    negligible amount (``median(r)`` is within noise of zero in practice).
    Documented approximation, not a silent one -- if a future noise model
    ever gives residuals a real non-zero bias, this would need revisiting.
    """
    if r_all.size == 0:
        return 0
    dev = np.abs(r_all)
    # ``overwrite_input=True`` is safe here for the same reason as the old
    # raw-sample MAD computation: ``dev``'s *order* is never read again --
    # only the returned median *value*, and the final ``count_nonzero``
    # comparison below is order-independent (a pure count).
    mad = float(np.median(dev, overwrite_input=True))
    return int(np.count_nonzero(dev > spike_threshold_mad * mad))


def _fast_kurtosis(x: np.ndarray) -> float:
    """Excess (Fisher) kurtosis, matching ``scipy.stats.kurtosis(x, fisher=True)``
    (its default ``bias=True``, i.e. the plain biased/population estimator
    ``m4 / m2**2 - 3``, no small-sample correction) to within documented
    float32 tolerance (see below).

    ``scipy.stats.kurtosis`` computes the 2nd and 4th central moments via
    two independent calls to its internal ``_moment``, each of which
    re-demeans ``x`` from scratch and raises the demeaned array to the
    target power with :func:`numpy.power` (a transcendental-style,
    order-4-generic routine, and in float64). This instead:

    - demeans once and gets the 4th power via two elementwise
      *multiplications* (``d2 = d * d``; ``d2 * d2``) rather than ``d**4``
      -- multiplication is a single hardware instruction per element where
      a generic integer power is not;
    - runs entirely in ``float32`` (cast once up front). Kurtosis has no
      tight-tolerance consumer (unlike ``rms``/``mean_abs``, which stay
      float64 -- see :func:`compute_channel_day_stats`): every assertion
      against it in ``tests/test_qc_stats.py`` is a wide-margin inequality
      (e.g. ``> 50``, ``> 10``, ``< 5``), and OpenMEQ-5nl.8 profiling found
      float32 cut this function's cost by roughly 1.5x end to end
      (including the cast) on a full 17.28M-sample channel-day, dominated
      by the elementwise ``mean``/multiply reductions rather than
      (surprisingly) the partition-based median/MAD, which barely benefits
      from float32 (see ``docs/performance.md``).

    Verified against ``scipy.stats.kurtosis`` (float64) on a sample
    channel-day, within a documented relative tolerance, in
    ``tests/test_qc_stats.py::test_fast_kurtosis_matches_scipy_reference``.

    A zero-variance (or single-sample) input divides by zero the same way
    ``scipy.stats.kurtosis`` does, yielding ``nan`` (with a runtime
    warning) rather than raising.
    """
    x32 = x.astype(np.float32, copy=False)
    mean = x32.mean()
    d = x32 - mean
    d2 = d * d
    m2 = d2.mean()
    m4 = (d2 * d2).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(m4 / (m2 * m2) - 3.0)


def _empty_stats(seed_id: str, day: date) -> dict:
    """Result for a channel-day with zero valid (non-masked) samples."""
    return {
        "seed_id": seed_id,
        "date": day.isoformat(),
        "rms": 0.0,
        "mean_abs": 0.0,
        "spike_count": 0,
        "kurtosis": float("nan"),
        "flat_count": 0,
        "longest_flat_run_s": 0.0,
        "clip_count": 0,
        "npts": 0,
        **empty_noise_stats(),
    }


def compute_channel_day_stats(seed_id: str, day: date, stream, qc_cfg: QcConfig) -> dict:
    """Compute one channel-day's daily noise statistics from an already-read ``Stream``.

    Separated from :func:`stats_for_channel_day` so it can be unit-tested
    directly against a hand-built ``Stream``/``Trace`` (e.g. the int32
    clip-detection test) without going through the index/config-file
    machinery.

    Parameters
    ----------
    seed_id : str
    day : datetime.date
    stream : obspy.Stream
        Typically ``DataIndex.read([seed_id], t1, t2)`` -- may hold masked
        traces across gaps; split here before any arithmetic.
    qc_cfg : openmeq.config.QcConfig

    Returns
    -------
    dict
        One row, keys matching :data:`DAILY_STATS_COLUMNS`.
    """
    # ``Stream.split()`` (needed to break a *masked* array at a gap into
    # contiguous, unmasked per-segment traces -- see the module docstring)
    # unconditionally deep-copies trace metadata even when there is
    # nothing to split. The overwhelming majority of channel-days have no
    # gap at all (a single already-contiguous trace straight out of
    # ``DataIndex.read``), so skip it entirely unless at least one trace
    # is actually masked (OpenMEQ-5nl.8 profiling: ~10 ms/channel-day on a
    # full 200 Hz day, pure overhead in the common no-gap case).
    # Captured before ``segments`` is overwritten with bare ``.data`` arrays
    # below -- needed to convert ``longest_flat_run_samples`` into
    # ``longest_flat_run_s``. A channel/day is one seed id at one fixed
    # sampling rate, so any trace's own rate applies to the whole day.
    sampling_rate = float(stream[0].stats.sampling_rate) if len(stream) > 0 else None

    if len(stream) > 0 and any(np.ma.is_masked(tr.data) for tr in stream):
        traces = list(stream.split())
    else:
        traces = list(stream)
    traces = [tr for tr in traces if tr.data.size > 0]
    segments = [tr.data for tr in traces]

    if not segments:
        return _empty_stats(seed_id, day)

    is_int = segments[0].dtype.kind == "i"

    float_saturation_value: float | None = None
    if not is_int:
        concat_raw = segments[0] if len(segments) == 1 else np.concatenate(segments)
        if concat_raw.size > 0:
            float_saturation_value = float(concat_raw[np.argmax(np.abs(concat_raw))])

    total_npts = 0
    sum_sq = 0.0
    sum_abs = 0.0
    flat_count = 0
    clip_count = 0
    longest_flat_run_samples = 0
    float_segments: list[np.ndarray] = []
    residual_segments: list[np.ndarray] = []

    for seg in segments:
        n = seg.size
        total_npts += n

        seg_flat, seg_clip, seg_longest = _segment_run_counts(
            seg,
            flat_run_min_samples=qc_cfg.flat_run_min_samples,
            clip_run_min_samples=qc_cfg.clip_run_min_samples,
            is_int=is_int,
            float_saturation_value=float_saturation_value,
        )
        flat_count += seg_flat
        clip_count += seg_clip
        # A flat run cannot legitimately span a real time gap (module
        # docstring), so the day's longest run is the max across segments,
        # never a sum.
        longest_flat_run_samples = max(longest_flat_run_samples, seg_longest)

        # NOTE: ``mean_abs`` is deliberately *not* accumulated off a
        # narrower (e.g. int32) view of the raw samples: ``np.abs`` on
        # int32 silently overflows for exactly ``-(2**31)`` (two's
        # complement has no positive counterpart for that value), and
        # ``-(2**31)`` is precisely one of this module's own two int32
        # saturation values (see the module docstring's "Clip detection
        # order") -- a negatively-clipped channel would otherwise corrupt
        # ``mean_abs`` silently. ``seg_f`` (float64) has no such overflow.
        seg_f = seg.astype(np.float64, copy=False)
        sum_abs += float(np.sum(np.abs(seg_f)))
        # ``np.dot(seg_f, seg_f)`` computes the same sum of squares as
        # ``np.sum(seg_f**2)`` without ever materialising the squared
        # array -- one BLAS-level pass instead of an elementwise square
        # (full-size temp) followed by a separate reduction. Verified
        # (``tests/test_qc_stats.py::test_dot_sum_of_squares_matches_manual_sum_within_tolerance``)
        # to agree with the naive elementwise-square-then-sum to double
        # precision's practical limit (a different summation order can
        # differ in the last handful of ULPs for a 17M-term reduction);
        # the existing ``rel=1e-12`` manual-calculation test
        # (``test_rms_mean_abs_matches_manual_calc``) still passes at that
        # tolerance.
        sum_sq += float(np.dot(seg_f, seg_f))
        float_segments.append(seg_f)

        # Isolated-impulse residual (M1.9): computed *per segment*, not on
        # the cross-segment concatenation, so a real time gap never lets
        # the running-median window span two physically-discontinuous
        # segments (which would manufacture a handful of spurious
        # large-residual samples right at the join). The residuals are
        # pooled below for a single MAD/threshold/count over the whole day
        # -- additive, unlike a median itself, since it's ultimately a
        # count over the concatenation of independently-computed residuals.
        window = _clamped_odd_window(qc_cfg.spike_median_window, seg_f.size)
        med_local = median_filter(seg_f, size=window, mode="nearest")
        residual_segments.append(seg_f - med_local)

    rms = float(np.sqrt(sum_sq / total_npts))
    mean_abs = float(sum_abs / total_npts)

    x_all = float_segments[0] if len(float_segments) == 1 else np.concatenate(float_segments)
    r_all = (
        residual_segments[0] if len(residual_segments) == 1 else np.concatenate(residual_segments)
    )
    spike_count = _spike_count_from_residual(r_all, qc_cfg.spike_threshold_mad)
    kurt = _fast_kurtosis(x_all)

    longest_flat_run_s = longest_flat_run_samples / sampling_rate if sampling_rate else 0.0

    return {
        **hourly_noise_stats(traces, day, qc_cfg.noise),
        "seed_id": seed_id,
        "date": day.isoformat(),
        "rms": rms,
        "mean_abs": mean_abs,
        "spike_count": spike_count,
        "kurtosis": kurt,
        "flat_count": flat_count,
        "longest_flat_run_s": longest_flat_run_s,
        "clip_count": clip_count,
        "npts": total_npts,
    }


def _load_qc_cfg(cfg_path: str | Path) -> QcConfig:
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return QcConfig.model_validate(raw.get("qc", {}))


#: Per-worker-*process* caches, keyed by the path string each worker was
#: called with. ``ProcessPoolExecutor`` workers are long-lived (one spawned
#: interpreter processes many submitted items over the pool's lifetime, not
#: one process per item -- see ``openmeq.parallel``'s module docstring), so
#: without this, every single channel-day item re-parsed the *entire*
#: ``inventory/index.json`` (and re-validated the small ``qc:`` YAML) from
#: scratch, even though every item in one ``compute_daily_stats`` call
#: shares the same index/config (OpenMEQ-5nl.8: "cache the DataIndex per
#: worker process"). Each worker process gets its own module-level dict (no
#: cross-process sharing -- ``spawn`` gives each worker a fresh module
#: import), and a given ``(index_path, cfg_path)`` pair is only ever loaded
#: once per worker's lifetime. Safe across separate ``pmap`` calls too:
#: each call to :func:`openmeq.parallel.pmap` creates a brand-new process
#: pool (and therefore brand-new, empty caches) when ``n_workers > 1``; the
#: ``n_workers == 1`` path runs in the *calling* process, where this same
#: cache also applies (harmlessly -- a single process handling every item
#: in that run) and is why the in-process serial test suite sees a speedup
#: too.
_worker_index_cache: dict[str, DataIndex] = {}
_worker_qc_cfg_cache: dict[str, QcConfig] = {}


def _cached_index(index_path: str) -> DataIndex:
    index = _worker_index_cache.get(index_path)
    if index is None:
        index = DataIndex.load(index_path)
        _worker_index_cache[index_path] = index
    return index


def _cached_qc_cfg(cfg_path: str) -> QcConfig:
    qc_cfg = _worker_qc_cfg_cache.get(cfg_path)
    if qc_cfg is None:
        qc_cfg = _load_qc_cfg(cfg_path)
        _worker_qc_cfg_cache[cfg_path] = qc_cfg
    return qc_cfg


def stats_for_channel_day(seed_id: str, day: date, index_path: str, cfg_path: str) -> dict:
    """Top-level, picklable per-``(seed_id, day)`` worker for :func:`openmeq.parallel.pmap`.

    Reads real waveform samples (not header-only) for one channel-day via
    ``DataIndex.read`` and computes its daily noise statistics.

    Parameters
    ----------
    seed_id : str
        Exact ``NET.STA.LOC.CHA`` seed id.
    day : datetime.date
        Calendar day (UTC) to compute statistics for.
    index_path : str
        Path to a ``DataIndex``-format JSON file (``inventory/index.json``).
        Loaded once per worker *process* and cached thereafter (see
        :data:`_worker_index_cache`), not reloaded on every call.
    cfg_path : str
        Path to a small YAML file holding just a ``qc:`` section (see the
        module docstring's "``cfg_path`` design note"). Also cached per
        worker process.

    Returns
    -------
    dict
        One row, keys matching :data:`DAILY_STATS_COLUMNS`.
    """
    index = _cached_index(index_path)
    qc_cfg = _cached_qc_cfg(cfg_path)
    day_start, day_end = _day_window(day)
    stream = index.read([seed_id], day_start, day_end)
    return compute_channel_day_stats(seed_id, day, stream, qc_cfg)


def _stats_batch(item: tuple[str, list[str], str, str]) -> list[dict]:
    """Top-level ``pmap`` worker: one seed id x a batch of calendar days.

    Mirrors ``openmeq.qc.availability``'s ``_availability_batch`` shape
    exactly (OpenMEQ-5nl.8's "chunk the work so each worker handles several
    channel-days"): ``days_iso`` are ISO ``YYYY-MM-DD`` strings (picklable
    primitives), and this delegates to :func:`stats_for_channel_day` once
    per day. Batching more than one day per task
    (``cfg.parallel.chunk_days > 1``) only reduces per-task scheduling/IPC
    overhead -- the *index*/*config* caching that makes repeated calls
    within one task (and across tasks handled by the same long-lived
    worker process) cheap lives in :func:`_cached_index`/
    :func:`_cached_qc_cfg`, not in this batching itself; batching mainly
    amortises ``ProcessPoolExecutor``'s per-task pickling/scheduling
    overhead over several channel-days instead of one.
    """
    seed_id, days_iso, index_path, cfg_path = item
    return [
        stats_for_channel_day(seed_id, date.fromisoformat(day_iso), index_path, cfg_path)
        for day_iso in days_iso
    ]


def _write_qc_worker_cfg(qc_cfg: QcConfig) -> Path:
    """Write ``qc_cfg`` to a small standalone temp YAML file for worker IPC.

    M1.5 found orphaned ``/tmp/openmeq_qc_stats_cfg_*.yaml`` files left
    behind on this shared machine. :func:`compute_daily_stats`'s own
    ``try``/``finally`` around :func:`~openmeq.parallel.pmap` already
    unlinks the file it gets back from this function on every normal exit
    path (including a worker-raised exception, since that propagates
    through ``pmap`` into the ``try`` block) -- but *this* function itself
    had no such guard: ``tempfile.mkstemp`` creates the file on disk before
    a single byte is written to it, and if ``qc_cfg.model_dump`` or
    ``yaml.safe_dump`` ever raised (e.g. a future non-YAML-safe field), the
    exception would propagate out of this function with the empty temp
    file never unlinked -- no caller has a path to it yet to clean up. The
    ``try``/``except`` below closes that gap: on any failure to populate
    the file, it is removed before the exception re-raises.
    """
    fd, tmp_name = tempfile.mkstemp(prefix="openmeq_qc_stats_cfg_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"qc": qc_cfg.model_dump(mode="json")}, f)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def compute_daily_stats(
    index: DataIndex,
    seed_ids: list[str],
    dates: list[date],
    cfg: Config,
) -> pd.DataFrame:
    """Daily noise statistics for every ``seed_ids`` x ``dates`` channel-day.

    One :func:`openmeq.parallel.pmap` unit of work per seed id x a batch of
    up to ``cfg.parallel.chunk_days`` consecutive calendar days (PLAN.md
    §4's channel x day unit of work, batched exactly like
    ``openmeq.qc.availability.compute_availability`` -- OpenMEQ-5nl.8),
    each running :func:`stats_for_channel_day` once per day inside
    :func:`_stats_batch`. Batching amortises per-task
    ``ProcessPoolExecutor`` scheduling/IPC overhead over several
    channel-days; within one task (and across every task handled by the
    same long-lived worker process), the index/config are loaded once and
    cached (:func:`_cached_index`/:func:`_cached_qc_cfg`), not once per
    channel-day.

    Parameters
    ----------
    index : openmeq.index.DataIndex
        Already-loaded/-built index. Written to ``RunPaths(cfg).index_json``
        if not already present there (idempotent -- workers reload the
        index from that path rather than having it pickled into every
        work item).
    seed_ids : list of str
        Exact seed ids to compute (e.g. ``index.seed_ids()``, filtered by
        the caller as needed).
    dates : list of datetime.date
        Calendar days (UTC) to compute.
    cfg : openmeq.config.Config

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`DAILY_STATS_COLUMNS`, in order, one row per
        ``(seed_id, day)``.
    """
    if not seed_ids or not dates:
        return pd.DataFrame(columns=DAILY_STATS_COLUMNS)

    paths = RunPaths(cfg)
    index_path = paths.index_json
    if not index_path.exists():
        index.write(index_path)

    chunk_days = max(1, cfg.parallel.chunk_days)
    sorted_dates = sorted(dates)
    date_chunks = [
        sorted_dates[i : i + chunk_days] for i in range(0, len(sorted_dates), chunk_days)
    ]

    cfg_path = _write_qc_worker_cfg(cfg.qc)
    try:
        items = [
            (seed_id, [d.isoformat() for d in chunk], str(index_path), str(cfg_path))
            for seed_id in seed_ids
            for chunk in date_chunks
        ]
        logger.info(
            "Computing daily QC stats for %d channel(s) x %d day(s) "
            "= %d channel-day(s) (%d task(s), chunk_days=%d)",
            len(seed_ids),
            len(sorted_dates),
            len(seed_ids) * len(sorted_dates),
            len(items),
            chunk_days,
        )
        batches = pmap(_stats_batch, items, cfg.parallel, desc="qc-daily-stats")
    finally:
        cfg_path.unlink(missing_ok=True)

    results = [row for batch in batches for row in batch]
    if results:
        df = pd.DataFrame(results, columns=DAILY_STATS_COLUMNS)
    else:
        df = pd.DataFrame(columns=DAILY_STATS_COLUMNS)
    return df


def write_daily_stats_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``df`` to ``path`` as CSV (``qc/daily_stats.csv``, ARCHITECTURE.md §3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=DAILY_STATS_COLUMNS)
