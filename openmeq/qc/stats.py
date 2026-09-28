"""Daily noise statistics per channel-day (M1.4, ``OpenMEQ-5nl.4``).

PLAN.md §6 Step 1's "Daily noise statistics per channel" (RMS, mean absolute
amplitude, MAD-based spikiness, kurtosis, flat-lined and clipped sample
counts), written to ``qc/daily_stats.csv`` (ARCHITECTURE.md §3):
``seed_id,date,rms,mean_abs,spike_count,kurtosis,flat_count,clip_count,npts``.

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
``cfg.qc.flat_run_min_samples`` and ``cfg.qc.clip_run_min_samples`` -- not
the rest of the (much larger) project :class:`~openmeq.config.Config`
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
- ``median``/MAD/spike count/kurtosis: **not** additive across segments (a
  median of medians is not the true median). These are computed once on the
  concatenation of every segment's valid samples for the day -- acceptable
  at this dataset's scale (a documented memory caveat for very
  high-rate/long-duration production channels, out of scope to optimise
  here; a streaming/approximate two-pass alternative would be needed then).
- flat-run / clip-run counts: **must** stay per-segment (a run cannot
  legitimately span a real time gap), computed with a vectorised run-length
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

from openmeq.config import Config, QcConfig
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.paths import RunPaths

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
    "clip_count",
    "npts",
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
) -> tuple[int, int]:
    """``(flat_count, clip_count)`` for one contiguous segment's raw samples.

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
        return 0, 0

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

    run_values = x[run_starts]
    if is_int:
        is_saturation_value = (run_values == _INT32_MAX) | (run_values == _INT32_MIN)
    elif float_saturation_value is not None:
        is_saturation_value = run_values == float_saturation_value
    else:
        is_saturation_value = np.zeros(run_values.shape, dtype=bool)

    clip_mask = (run_lengths >= clip_run_min_samples) & is_saturation_value
    clip_count = int(run_lengths[clip_mask].sum())
    return flat_count, clip_count


def _median_mad_spike_count(
    x_all: np.ndarray, spike_threshold_mad: float
) -> tuple[float, float, int]:
    """``(median, mad, spike_count)`` for one channel-day's concatenated valid samples.

    Same two order-statistics computations as the original implementation
    (a median is not additive across segments -- see the module
    docstring), but with two changes that only affect *speed*, not the
    result:

    - ``overwrite_input=True`` on both :func:`numpy.median` calls. Per
      ``numpy.median``'s own source, this skips an internal defensive
      copy (``np.partition`` otherwise copies the whole array before
      partitioning in place) and instead partitions ``x_all``/``dev`` in
      place -- safe here because neither array's *order* is read again
      afterwards (the caller only wants the median/MAD *values*, and
      :func:`_fast_kurtosis`, called separately on the still-unsorted-but-
      value-preserved ``x_all``, is order-independent). On a full
      17.28M-sample channel-day this removes two full-array copies
      (OpenMEQ-5nl.8 profiling).
    - ``dev = abs(x_all - median)`` is computed once and reused for both
      the MAD and the spike threshold comparison, instead of being
      recomputed from scratch for the spike count (the original
      implementation called ``np.abs(x_all - median)`` twice).
    """
    median = float(np.median(x_all, overwrite_input=True))
    dev = np.abs(x_all - median)
    mad = float(np.median(dev, overwrite_input=True))
    spike_count = int(np.count_nonzero(dev > spike_threshold_mad * mad))
    return median, mad, spike_count


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
        "clip_count": 0,
        "npts": 0,
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
    if len(stream) > 0 and any(np.ma.is_masked(tr.data) for tr in stream):
        segments = [tr.data for tr in stream.split()]
    else:
        segments = [tr.data for tr in stream]
    segments = [seg for seg in segments if seg.size > 0]

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
    float_segments: list[np.ndarray] = []

    for seg in segments:
        n = seg.size
        total_npts += n

        seg_flat, seg_clip = _segment_run_counts(
            seg,
            flat_run_min_samples=qc_cfg.flat_run_min_samples,
            clip_run_min_samples=qc_cfg.clip_run_min_samples,
            is_int=is_int,
            float_saturation_value=float_saturation_value,
        )
        flat_count += seg_flat
        clip_count += seg_clip

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

    rms = float(np.sqrt(sum_sq / total_npts))
    mean_abs = float(sum_abs / total_npts)

    x_all = float_segments[0] if len(float_segments) == 1 else np.concatenate(float_segments)
    median, mad, spike_count = _median_mad_spike_count(x_all, qc_cfg.spike_threshold_mad)
    kurt = _fast_kurtosis(x_all)

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
    """Write ``qc_cfg`` to a small standalone temp YAML file for worker IPC."""
    fd, tmp_name = tempfile.mkstemp(prefix="openmeq_qc_stats_cfg_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"qc": qc_cfg.model_dump(mode="json")}, f)
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
