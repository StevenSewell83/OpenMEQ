"""Daily noise statistics per channel-day (M1.4, ``OpenMEQ-5nl.4``).

PLAN.md §6 Step 1's "Daily noise statistics per channel" (RMS, mean absolute
amplitude, MAD-based spikiness, kurtosis, flat-lined and clipped sample
counts), written to ``qc/daily_stats.csv`` (ARCHITECTURE.md §3):
``seed_id,date,rms,mean_abs,spike_count,kurtosis,flat_count,clip_count,npts``.

Parallelism
-----------
One unit of work per channel-day (PLAN.md §4's "QC | channel x day" row),
run through :func:`openmeq.parallel.pmap` exactly like the miniSEED index's
header scan (:mod:`openmeq.index`) parallelizes per file: the top-level,
picklable worker is :func:`stats_for_channel_day`, which takes a
``seed_id``/``day`` pair plus two *paths* -- never a waveform object or the
in-memory :class:`~openmeq.config.Config`/:class:`~openmeq.index.DataIndex`
themselves -- and does its own ``DataIndex.load`` + ``index.read`` inside
the worker process. :func:`compute_daily_stats` is the orchestration
entry point: it takes the already-loaded ``index``/``cfg`` (for a
convenient, direct API), and internally bridges to the path-based worker
contract.

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
from scipy.stats import kurtosis as _scipy_kurtosis

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


def _run_length_per_sample(x: np.ndarray) -> np.ndarray:
    """Vectorised run-length encoding: for each sample, the length of the run
    of identical consecutive values it belongs to.

    Uses ``np.diff(x) == 0`` (a value equals its predecessor) to find run
    boundaries, then a cumulative sum to assign a run id per sample, then
    ``np.bincount`` to get each run's length back out -- no per-sample
    Python loop.
    """
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    is_new_run = np.empty(n, dtype=bool)
    is_new_run[0] = True
    is_new_run[1:] = x[1:] != x[:-1]
    run_id = np.cumsum(is_new_run) - 1
    run_len = np.bincount(run_id)
    return run_len[run_id]


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
    """
    if x.size == 0:
        return 0, 0

    run_len = _run_length_per_sample(x)
    flat_count = int(np.count_nonzero(run_len >= flat_run_min_samples))

    if is_int:
        is_saturation_value = (x == _INT32_MAX) | (x == _INT32_MIN)
    elif float_saturation_value is not None:
        is_saturation_value = x == float_saturation_value
    else:
        is_saturation_value = np.zeros_like(x, dtype=bool)

    clip_mask = (run_len >= clip_run_min_samples) & is_saturation_value
    clip_count = int(np.count_nonzero(clip_mask))
    return flat_count, clip_count


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
    segments = [tr.data for tr in stream.split()] if len(stream) > 0 else []
    segments = [seg for seg in segments if seg.size > 0]

    if not segments:
        return _empty_stats(seed_id, day)

    is_int = segments[0].dtype.kind == "i"

    float_saturation_value: float | None = None
    if not is_int:
        concat_raw = np.concatenate(segments)
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

        seg_f = seg.astype(np.float64, copy=False)
        sum_sq += float(np.sum(seg_f**2))
        sum_abs += float(np.sum(np.abs(seg_f)))
        float_segments.append(seg_f)

    rms = float(np.sqrt(sum_sq / total_npts))
    mean_abs = float(sum_abs / total_npts)

    x_all = np.concatenate(float_segments)
    median = np.median(x_all)
    mad = np.median(np.abs(x_all - median))
    spike_count = int(np.count_nonzero(np.abs(x_all - median) > qc_cfg.spike_threshold_mad * mad))
    kurt = float(_scipy_kurtosis(x_all, fisher=True))

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
        Path to a ``DataIndex``-format JSON file (``inventory/index.json``),
        loaded fresh in this worker process.
    cfg_path : str
        Path to a small YAML file holding just a ``qc:`` section (see the
        module docstring's "``cfg_path`` design note").

    Returns
    -------
    dict
        One row, keys matching :data:`DAILY_STATS_COLUMNS`.
    """
    index = DataIndex.load(index_path)
    qc_cfg = _load_qc_cfg(cfg_path)
    day_start, day_end = _day_window(day)
    stream = index.read([seed_id], day_start, day_end)
    return compute_channel_day_stats(seed_id, day, stream, qc_cfg)


def _stats_worker(item: tuple[str, date, str, str]) -> dict:
    """``pmap``-compatible wrapper: unpacks one ``(seed_id, day, index_path,
    cfg_path)`` item and calls :func:`stats_for_channel_day`.

    A separate top-level function (rather than a lambda/closure) because
    ``pmap`` pickles ``func`` to send it to worker processes.
    """
    seed_id, day, index_path, cfg_path = item
    return stats_for_channel_day(seed_id, day, index_path, cfg_path)


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

    One :func:`openmeq.parallel.pmap` unit of work per ``(seed_id, day)``
    pair (PLAN.md §4), each running :func:`stats_for_channel_day` in a
    worker process.

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
    paths = RunPaths(cfg)
    index_path = paths.index_json
    if not index_path.exists():
        index.write(index_path)

    cfg_path = _write_qc_worker_cfg(cfg.qc)
    try:
        items = [
            (seed_id, day, str(index_path), str(cfg_path)) for seed_id in seed_ids for day in dates
        ]
        logger.info(
            "Computing daily QC stats for %d channel(s) x %d day(s) = %d channel-day(s)",
            len(seed_ids),
            len(dates),
            len(items),
        )
        results = pmap(_stats_worker, items, cfg.parallel, desc="qc-daily-stats")
    finally:
        cfg_path.unlink(missing_ok=True)

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
