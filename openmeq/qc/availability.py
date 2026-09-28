"""QC availability metrics per channel-day (M1.3, ARCHITECTURE.md §3/§4.1).

Writes ``qc/availability.csv`` (columns ``seed_id,date,coverage_pct,n_gaps,
n_overlaps,gap_s``), one row per ``(seed_id, UTC calendar day)``, consumed by
M1.5 (channel flags) and M1.6 (availability heatmap). This module only
*reports* the numbers -- deciding pass/fail thresholds is M1.5's job
(``qc.min_availability_pct`` etc. are read there, not here), and
``cfg.qc.exclude_channels`` is deliberately **not** applied in this module:
availability is computed and reported for every channel; exclusion is a
downstream flagging decision.

Design: metadata-only, no waveform reads
-----------------------------------------
``openmeq.index.DataIndex`` already stores, per seed id, every on-disk
segment's ``(path, start, end, sampling_rate)`` (ARCHITECTURE.md §4.1) --
exactly what a header-only ``obspy.read(path, headonly=True)`` would give
back. Rather than re-opening every overlapping file for the day window
(``index.files(...)`` + ``obspy.read(..., headonly=True)`` + ``Stream.
get_gaps()``, as a first-cut design would), this module computes coverage,
gaps and overlaps **directly from the index's segment metadata**, with zero
file I/O per channel-day. This is a deliberate deviation from a
literal-``get_gaps()`` implementation, for speed: at production scale
(months x many stations), skipping a redundant header re-parse per
channel-day (the index build already paid that cost once) removes an I/O
round trip from the single most-repeated unit of work in the QC step
(PLAN.md §4: channel x day).

:func:`_gaps_and_overlaps` reimplements ``obspy.Stream.get_gaps()``'s
sign/delta convention (positive delta = gap, negative = overlap, and a
one-sample-period correction between two real segments -- see
``obspy.core.stream.Stream.get_gaps`` for the reference algorithm this
mirrors) directly on ``(start, end, sampling_rate)`` tuples, generalized two
ways beyond the literal per-consecutive-trace-pair algorithm:

1. **Running merged coverage**, not just consecutive-pair deltas: a
   segment fully contained inside an earlier, longer one (possible when a
   file holds several overlapping segments) is correctly swallowed rather
   than spuriously reported as a gap/overlap against whichever segment
   happens to sort next to it.
2. **Day-boundary coverage**: a channel whose first segment starts after
   the day's ``00:00:00`` (or whose last segment ends before the day's
   ``24:00:00``) is missing data at the start/end of the day. Literal
   ``get_gaps()`` never reports this (it only compares *between* traces,
   never against the stream's implied window), which would silently make
   ``coverage_pct`` read 100% for a channel that, say, only came online at
   06:00. Since ``coverage_pct = 100 * (day_length_s - gap_s) / day_length_s``
   is only meaningful if ``gap_s`` is the *total* missing time in the day,
   boundary gaps are counted here (without a delta-time inclusion of the
   per-real-trace one-sample-period correction there is no preceding real
   trace at the day boundaries).

A channel-day with **zero** overlapping segments is special-cased
(coverage_pct=0.0, n_gaps=0, n_overlaps=0, gap_s=86400.0) per the "no data
yet" convention -- not run through the general algorithm, which would
otherwise report a single 86400s boundary "gap"; the ARCHITECTURE.md
contract does not distinguish "no data" from "a real gap" in this CSV
(that distinction is M1.5's ``channel_flags.json`` reason strings, e.g.
``no_data`` vs ``availability<80``).

Parallel unit and worker signature
-----------------------------------
Per PLAN.md §4, channel x day is QC's unit of parallel work.
:func:`availability_for_channel_day` is the top-level, picklable
per-channel-day worker: its parameters are a seed id, a ``date``, and two
*paths* (not a live ``DataIndex``/``Config``), so it pickles trivially
across the ``ProcessPoolExecutor`` (``spawn``) that ``openmeq.parallel.pmap``
uses. :func:`compute_availability` is handed already-loaded ``DataIndex``/
``Config`` objects (its own documented signature), so it hands workers
*paths* rather than pickling those objects into every task item: the
index is written to ``RunPaths(cfg).index_json`` (reusing it as-is if an
earlier step, e.g. ``openmeq.index.build_index``, already wrote it there --
no redundant re-serialization), and the config is written once to a small
temporary file, cleaned up in a ``finally`` after the parallel map (mirrors
``openmeq.qc.stats.compute_daily_stats``'s own index-reuse pattern --
OpenMEQ-5nl.7 simplified this from unconditionally writing *both* to a
fresh ``tempfile.TemporaryDirectory`` every call). Each worker process then
loads them once and caches them at module level (keyed by resolved path),
so a batch of many channel-days processed by the same worker only pays the
load cost once (mirrors the per-process caching pattern in
ARCHITECTURE.md §6b). When
``cfg.parallel.chunk_days > 1``, one ``pmap`` item batches several
consecutive calendar days for one seed id into a single task (reduces
per-task scheduling overhead for long runs, per the issue's Performance
note); at test-dataset scale (8 stations x 3 comp x 2 days = 48 units) this
is trivial either way.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

from openmeq.config import Config, load_config
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.paths import RunPaths

logger = get_logger("qc.availability")

#: ARCHITECTURE.md §3's exact column order for qc/availability.csv.
COLUMNS = ["seed_id", "date", "coverage_pct", "n_gaps", "n_overlaps", "gap_s"]

DAY_LENGTH_S = 86400.0

#: Tolerance (seconds) below which a computed delta is treated as float
#: noise rather than a real gap/overlap. Comfortably smaller than any
#: realistic sample period (e.g. 5 ms at 200 Hz) while safely above
#: UTCDateTime timestamp float64 precision at this time scale (~1e-7 s).
_EPS_S = 1e-4

# ---------------------------------------------------------------------------
# Per-process caches (ARCHITECTURE.md §6b pattern: loaded once per worker
# process, keyed by resolved path; no cross-process sharing, which is fine
# since each worker process only ever sees one (index_path, cfg_path) pair
# per compute_availability call).
# ---------------------------------------------------------------------------

_INDEX_CACHE: dict[str, DataIndex] = {}
_CONFIG_CACHE: dict[str, Config] = {}


def _load_index_cached(index_path: str) -> DataIndex:
    key = str(Path(index_path).resolve())
    index = _INDEX_CACHE.get(key)
    if index is None:
        index = DataIndex.load(key)
        _INDEX_CACHE[key] = index
    return index


def _load_config_cached(cfg_path: str) -> Config:
    key = str(Path(cfg_path).resolve())
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = load_config(key)
        _CONFIG_CACHE[key] = cfg
    return cfg


# ---------------------------------------------------------------------------
# Core metadata-only gap/overlap algorithm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClippedSegment:
    start: UTCDateTime
    end: UTCDateTime
    sampling_rate: float


def _clipped_segments(
    index: DataIndex, seed_id: str, day_start: UTCDateTime, day_end: UTCDateTime
) -> list[_ClippedSegment]:
    """Segments of ``seed_id`` overlapping ``[day_start, day_end]``, clipped to it.

    Mirrors ``DataIndex.files``'s own overlap test (``start <= t2 and end >=
    t1``) so day-window semantics stay consistent with the rest of the
    index API. Each segment's start/end is clipped to the day window before
    being handed to :func:`_gaps_and_overlaps`, so a file that spans
    midnight (or several days) only contributes the portion actually inside
    this day (ARCHITECTURE.md's day-boundary handling note).
    """
    out: list[_ClippedSegment] = []
    for _path, start_iso, end_iso, sampling_rate in index.channels.get(seed_id, []):
        start = UTCDateTime(start_iso)
        end = UTCDateTime(end_iso)
        if start <= day_end and end >= day_start:
            clipped_start = start if start > day_start else day_start
            clipped_end = end if end < day_end else day_end
            out.append(_ClippedSegment(clipped_start, clipped_end, float(sampling_rate)))
    out.sort(key=lambda seg: (seg.start, seg.end))
    return out


def _gaps_and_overlaps(
    segments: list[_ClippedSegment], day_start: UTCDateTime, day_end: UTCDateTime
) -> tuple[int, int, float]:
    """Count gaps/overlaps and sum gap duration for one channel-day's segments.

    ``segments`` must be non-empty (the zero-segment "no data" case is
    special-cased by the caller, :func:`availability_for_channel_day`).

    Convention (mirrors ``obspy.Stream.get_gaps()``, generalized -- see the
    module docstring): walking segments in start order, ``delta`` is the
    time between the running merged coverage's end and the next segment's
    start, minus one sample period (of the segment that set the current
    coverage end) when that end came from a real segment -- a positive
    delta is a gap, a negative delta is an overlap between the two.
    Day-boundary gaps (before the first segment, after the last) use the
    raw delta with no sample-period correction, since there is no real
    preceding/following trace to derive one from.

    Returns
    -------
    (n_gaps, n_overlaps, gap_s) : tuple[int, int, float]
        Counts and the summed duration of **positive**-delta (gap) entries
        only -- overlapping duration is never subtracted from ``gap_s``,
        matching the issue's documented ``coverage_pct`` formula (using
        merged-timeline gap durations directly avoids double-counting
        overlaps that summing raw trace lengths would introduce).
    """
    n_gaps = 0
    n_overlaps = 0
    gap_s = 0.0

    current_end = day_start
    current_sr: float | None = None  # sampling rate that produced current_end

    for seg in segments:
        if current_sr is None:
            delta = seg.start.timestamp - current_end.timestamp
        else:
            period = 1.0 / current_sr if current_sr > 0 else 0.0
            delta = seg.start.timestamp - current_end.timestamp - period

        if delta > _EPS_S:
            n_gaps += 1
            gap_s += delta
        elif delta < -_EPS_S:
            n_overlaps += 1

        if seg.end > current_end:
            current_end = seg.end
            current_sr = seg.sampling_rate
        # else: seg is fully contained in already-covered time; ignore.

    if current_sr is None:
        trailing = day_end.timestamp - current_end.timestamp
    else:
        period = 1.0 / current_sr if current_sr > 0 else 0.0
        trailing = day_end.timestamp - current_end.timestamp - period
    if trailing > _EPS_S:
        n_gaps += 1
        gap_s += trailing

    return n_gaps, n_overlaps, gap_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def availability_for_channel_day(seed_id: str, day: date, index_path: str, cfg_path: str) -> dict:
    """Availability metrics for one ``(seed_id, day)`` -- the parallel unit.

    Top-level and picklable: ``index_path``/``cfg_path`` are plain strings
    (not a live ``DataIndex``/``Config``), per PLAN.md §4's "workers are
    given file paths and parameters, not waveform objects". Both are loaded
    (and cached per worker process, see :func:`_load_index_cached`/
    :func:`_load_config_cached`) rather than passed as objects, so this
    function survives pickling when ``openmeq.parallel.pmap`` runs it with
    ``n_workers > 1``. ``cfg`` is loaded for contract symmetry with the
    issue's specified signature and to fail fast on a bad config path; this
    module's own computation needs no config field (``cfg.qc.
    exclude_channels`` is explicitly not applied here -- see the module
    docstring).

    The day is queried as ``[day 00:00:00, day 24:00:00)`` UTC. No
    waveform data is read -- see the module docstring's "Design:
    metadata-only, no waveform reads".

    Parameters
    ----------
    seed_id : str
        Exact ``NET.STA.LOC.CHA`` seed id (not a pattern).
    day : datetime.date
        UTC calendar day to compute availability for.
    index_path : str
        Path to a ``DataIndex``-written JSON file (``inventory/index.json``).
    cfg_path : str
        Path to a YAML config file loadable by ``openmeq.config.load_config``.

    Returns
    -------
    dict
        ``{"seed_id", "date", "coverage_pct", "n_gaps", "n_overlaps",
        "gap_s"}`` -- one row of ``qc/availability.csv``. ``date`` is an
        ISO ``YYYY-MM-DD`` string.
    """
    _load_config_cached(cfg_path)
    index = _load_index_cached(index_path)

    day_start = UTCDateTime(day.isoformat())
    day_end = day_start + DAY_LENGTH_S

    segments = _clipped_segments(index, seed_id, day_start, day_end)

    if not segments:
        return {
            "seed_id": seed_id,
            "date": day.isoformat(),
            "coverage_pct": 0.0,
            "n_gaps": 0,
            "n_overlaps": 0,
            "gap_s": DAY_LENGTH_S,
        }

    n_gaps, n_overlaps, gap_s = _gaps_and_overlaps(segments, day_start, day_end)
    gap_s = min(gap_s, DAY_LENGTH_S)
    coverage_pct = 100.0 * (DAY_LENGTH_S - gap_s) / DAY_LENGTH_S
    coverage_pct = min(100.0, max(0.0, coverage_pct))

    return {
        "seed_id": seed_id,
        "date": day.isoformat(),
        "coverage_pct": coverage_pct,
        "n_gaps": n_gaps,
        "n_overlaps": n_overlaps,
        "gap_s": gap_s,
    }


def _availability_batch(args: tuple[str, list[str], str, str]) -> list[dict]:
    """Top-level ``pmap`` worker: one seed id x a batch of calendar days.

    ``days_iso`` are ISO ``YYYY-MM-DD`` strings (picklable primitives, per
    PLAN.md §4), converted back to ``date`` objects here before delegating
    to :func:`availability_for_channel_day` once per day. Batching more
    than one day per task (``cfg.parallel.chunk_days > 1``) is purely a
    scheduling-overhead optimization (see the module docstring); the result
    is the same list of per-day rows either way.
    """
    seed_id, days_iso, index_path, cfg_path = args
    return [
        availability_for_channel_day(seed_id, date.fromisoformat(day_iso), index_path, cfg_path)
        for day_iso in days_iso
    ]


def date_range_inclusive(start: date, end: date) -> list[date]:
    """``[start, end]`` inclusive, one entry per UTC calendar day.

    Small helper for callers building ``compute_availability``'s ``dates``
    argument from ``cfg.data.start``/``cfg.data.end`` (ARCHITECTURE.md §3's
    "dates spans cfg.data.start to cfg.data.end inclusive").
    """
    if end < start:
        return []
    n_days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n_days)]


def compute_availability(
    index: DataIndex, seed_ids: list[str], dates: list[date], cfg: Config
) -> pd.DataFrame:
    """Availability metrics for every ``(seed_id, date)`` pair, in parallel.

    Orchestrates :func:`availability_for_channel_day` over the full
    ``seed_ids x dates`` cross-product via ``openmeq.parallel.pmap``,
    batching ``cfg.parallel.chunk_days`` consecutive days per seed id into
    one task when ``chunk_days > 1`` (PLAN.md §4: channel x day is the unit
    of parallel work; batching is purely a scheduling-overhead reduction at
    production scale, see the module docstring).

    ``index``/``cfg`` are already-loaded objects (unlike the picklable
    per-task worker, which takes paths): ``index`` is written to
    ``RunPaths(cfg).index_json`` only if not already present there (an
    earlier step, e.g. ``openmeq.index.build_index``, typically already
    wrote it -- no redundant re-serialization of a potentially large
    index), and ``cfg`` is written once to a small temporary file, unlinked
    in a ``finally`` after the parallel map. Workers load from those paths
    (and cache per process), so a large index is never repeatedly pickled
    into every task item -- only two small path strings are.

    ``seed_ids`` is used as given, with **no** ``cfg.qc.exclude_channels``
    filtering (that belongs to channel flagging, M1.5): this function
    reports availability for every channel it is asked about.

    Parameters
    ----------
    index : openmeq.index.DataIndex
    seed_ids : list of str
        Exact seed ids to compute, typically ``index.seed_ids()``.
    dates : list of datetime.date
        UTC calendar days to compute, typically
        ``date_range_inclusive(cfg.data.start, cfg.data.end)``.
    cfg : openmeq.config.Config

    Returns
    -------
    pandas.DataFrame
        Columns exactly ``seed_id,date,coverage_pct,n_gaps,n_overlaps,
        gap_s`` (ARCHITECTURE.md §3), sorted by ``seed_id`` then ``date``.
    """
    if not seed_ids or not dates:
        return pd.DataFrame(columns=COLUMNS)

    chunk_days = max(1, cfg.parallel.chunk_days)
    sorted_dates = sorted(dates)
    date_chunks = [
        sorted_dates[i : i + chunk_days] for i in range(0, len(sorted_dates), chunk_days)
    ]

    # Reuse the index already on disk (e.g. written moments earlier by
    # openmeq.index.build_index, as openmeq.qc.run does) rather than
    # unconditionally re-serializing it to a fresh temp copy every call --
    # mirrors openmeq.qc.stats.compute_daily_stats's own pattern.
    paths = RunPaths(cfg)
    index_path = paths.index_json
    if not index_path.exists():
        index.write(index_path)

    # A small standalone temp file, not a whole TemporaryDirectory (only
    # the config needs one -- the index is handled above). Written as
    # plain JSON text (a valid YAML subset), read back by load_config's
    # yaml.safe_load -- see the module docstring; avoids relying on
    # yaml.safe_dump's scalar-style choices for date-shaped strings
    # (cfg.data.start/end) round-tripping ambiguously. Every path in a
    # loaded Config is already absolute (load_config resolved them against
    # the *original* config file's directory), so re-resolving relative to
    # this temp file's directory on reload is a no-op.
    fd, tmp_name = tempfile.mkstemp(prefix="openmeq_qc_availability_cfg_", suffix=".yaml")
    os.close(fd)
    cfg_path = Path(tmp_name)
    try:
        cfg_path.write_text(json.dumps(cfg.model_dump(mode="json")), encoding="utf-8")

        items = [
            (seed_id, [d.isoformat() for d in chunk], str(index_path), str(cfg_path))
            for seed_id in seed_ids
            for chunk in date_chunks
        ]

        logger.info(
            "Computing availability for %d channel(s) x %d day(s) (%d task(s), chunk_days=%d)",
            len(seed_ids),
            len(sorted_dates),
            len(items),
            chunk_days,
        )

        batches = pmap(_availability_batch, items, cfg.parallel, desc="qc-availability")
    finally:
        cfg_path.unlink(missing_ok=True)

    rows = [row for batch in batches for row in batch]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["seed_id", "date"], kind="stable").reset_index(drop=True)
    return df


def write_availability_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``df`` to ``path`` as ``qc/availability.csv`` (ARCHITECTURE.md §3).

    Creates parent directories as needed. Does not reorder/validate
    columns -- callers are expected to pass :func:`compute_availability`'s
    output (or something with the same shape) straight through.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
