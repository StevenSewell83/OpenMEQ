"""Per-event waveform chunk writing (M3.1, ``OpenMEQ-rq8.1``, ARCHITECTURE.md §3/§4.3).

Implements PLAN.md §6 Step 3: cut a +/- ``cfg.chunks.pre_s``/``post_s``
window around every detection in ``detections/detections.csv`` and write it
as one miniSEED file per event (``chunks/<event_id>/<event_id>.mseed``,
:meth:`openmeq.paths.RunPaths.chunk_file`), holding **every** channel in the
network at its **original** sampling rate (no resampling -- that is M4.2's
``chunks_pn/`` job, a separate output tree).

Grouping ("each source file opened once per worker")
-----------------------------------------------------
PLAN.md §4's Chunk row requires "detections grouped by source file, so each
file is opened once per worker". This module reads that as **grouping by
UTC calendar day** (:func:`group_detections_by_day`): every detection whose
trigger time falls on the same UTC date is handled by one
:func:`chunk_day_group` task, which performs exactly **one** whole-network
:meth:`openmeq.index.DataIndex.read` covering that group's overall time
span, then cuts every one of that day's detections out of the single
resulting in-memory ``Stream`` via ``Stream.slice`` (never a second read for
the same event/day). ``DataIndex.read`` itself only opens a given on-disk
miniSEED file once per call (:meth:`~openmeq.index.DataIndex.files`
resolves each file at most once, ARCHITECTURE.md §2), so a single
per-day-group read is exactly "each file opened once" in practice --
without this module needing its own explicit source-file-to-detections
grouping structure, which would be considerably more complex for the same
result. Parallelism (via ``openmeq.parallel.pmap``) is therefore across
day-groups, not detections: a day with many detections still costs one
whole-network read for that day (the whole point of this grouping), so
throughput scales with the number of days a survey spans, not the number
of detections in it.

A detection near a day boundary is still assigned to the UTC calendar date
of its own trigger time (so every detection belongs to exactly one group),
but a day-group's *read window* is **not** ``[day_start, day_end)`` -- it is
``[min(detection window start), max(detection window end)]`` across every
detection actually in that group (:func:`chunk_day_group`), padded by a
small safety margin (see below). Since that window is derived from the
detections' own times/half-widths rather than from midnight, it naturally
extends into the neighbouring day's files when a detection's own +/-
``pre_s``/``post_s`` window crosses midnight -- no special-cased boundary
logic is needed; :meth:`~openmeq.index.DataIndex.read` simply finds and
reads whichever on-disk files overlap that window, regardless of which
calendar day they belong to.

``Stream.slice`` vs. ``Stream.copy().trim``
---------------------------------------------
Each detection's chunk is cut from the day-group's shared ``Stream`` with
``Stream.slice(t1, t2)``, not ``Stream.copy().trim(t1, t2)``. ``.slice()``
returns a *new* ``Stream`` of ``Trace`` objects that are themselves
lightweight views (new ``Trace``/``Stats`` wrapping the same underlying
data buffer, sliced), **without mutating** the day-group's original
``Stream`` -- essential here, since every other detection in the group
needs to cut its own, independent chunk from that same original ``Stream``
afterwards. ``Stream.slice`` also conveniently drops any ``Trace`` that
does not overlap the requested window at all (``keep_empty_traces=False``,
its default) rather than yielding zero-length traces that would then need
filtering out before writing.

Masked writes / gap spanning (``Stream.split()``)
----------------------------------------------------
``DataIndex.read`` merges with ``stream.merge(method=0)``, which produces a
masked array across a real gap (ARCHITECTURE.md §2). miniSEED cannot write
a masked ``Trace`` as one continuous record, so the day-group's ``Stream``
is split into fully contiguous, unmasked segments (``Stream.split()``)
**before** any per-detection slicing. A detection whose window spans the
known gap therefore ends up with two (or more) separate ``Trace`` segments
for that channel in its output file -- exactly the desired behaviour (never
a gap-filled trace, and never a write failure).

Original sampling preserved
------------------------------
No resample/decimate call appears anywhere in this module -- chunk writing
is a pure windowing/cut operation over whatever sampling rate each channel
was recorded at. Differing sampling rates across stations/channels are
written into the same output file exactly as read; miniSEED natively
supports mixed-rate traces in one file.

Safety margin
----------------
:data:`_READ_WINDOW_SAFETY_MARGIN_S` (5.0 s) is added on each side of a
day-group's computed read window, on top of ``cfg.chunks.pre_s``/
``post_s``. It exists purely to be safe against clock/sample-rounding at
the read boundary (e.g. ``UTCDateTime`` subtraction/floating-point
comparisons at an exact window edge) -- it is not load-bearing for
correctness (each detection's own exact ``pre_s``/``post_s`` window is
always inside ``[min(times) - pre_s, max(times) + post_s]`` by
construction), just cheap insurance. 5 s is several orders of magnitude
larger than a sample period at any plausible sampling rate, and small
relative to the default 30 s ``pre_s``/``post_s`` half-widths, so it does
not materially inflate the read.

``cfg_path`` design (worker IPC)
-----------------------------------
Mirrors ``openmeq.qc.stats``'s ``cfg_path`` convention exactly (see that
module's docstring for the full rationale): :func:`chunk_day_group` only
ever needs ``cfg.chunks.pre_s``/``post_s`` -- not the rest of the (much
larger) project :class:`~openmeq.config.Config` -- so :func:`write_chunks`
writes just the ``chunks:`` section to a small, self-contained temp YAML
file, loaded (and cached per worker *process*, :data:`_worker_chunks_cfg_cache`)
inside the worker via :class:`~openmeq.config.ChunksConfig.model_validate`,
not ``openmeq.config.load_config`` on the user's real project config. The
``DataIndex`` is loaded from ``index_path`` (``inventory/index.json``) and
cached the same way (:data:`_worker_index_cache`), so a long-lived
``ProcessPoolExecutor`` worker handling several day-groups over its
lifetime only pays the index-parse cost once.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from obspy import UTCDateTime

from openmeq.config import ChunksConfig
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("chunk")

#: Extra padding (seconds), on top of ``cfg.chunks.pre_s``/``post_s``, added
#: to a day-group's computed read window. See the module docstring's
#: "Safety margin" section.
_READ_WINDOW_SAFETY_MARGIN_S = 5.0


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_detections_by_day(detections_df: pd.DataFrame, cfg: Config) -> dict[str, list[dict]]:
    """Group detection rows by the UTC calendar date of their trigger ``time``.

    Each group is handled by exactly one :func:`chunk_day_group` task (one
    whole-network read per group -- see the module docstring). A detection
    whose own +/- ``cfg.chunks.pre_s``/``post_s`` window crosses midnight is
    still assigned here to the UTC date of its trigger ``time`` (so every
    detection belongs to exactly one group); the group's *read* window is
    computed separately, from the group's actual min/max detection times,
    by :func:`chunk_day_group` itself, and naturally covers the adjacent
    day's files in that case (no special-casing needed here).

    Parameters
    ----------
    detections_df : pandas.DataFrame
        ``detections/detections.csv``'s columns (ARCHITECTURE.md §4.3):
        at least ``event_id`` and ``time`` (an ISO-8601 ``...Z`` string, or
        anything ``obspy.UTCDateTime`` accepts).
    cfg : openmeq.config.Config
        Unused by the grouping decision itself (grouping is solely by the
        calendar date of ``time`` -- see above); accepted for API symmetry
        with :func:`chunk_day_group`/:func:`write_chunks` and in case a
        future contract change needs it (e.g. a configurable grouping
        granularity).

    Returns
    -------
    dict
        ``{"YYYY-MM-DD": [detection_row_dict, ...]}``, each group's rows
        sorted by ``time``.
    """
    del cfg  # not needed for the grouping decision itself; see docstring.
    groups: dict[str, list[dict]] = {}
    for row in detections_df.to_dict("records"):
        day = UTCDateTime(row["time"]).date.isoformat()
        groups.setdefault(day, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: UTCDateTime(row["time"]))
    return groups


# ---------------------------------------------------------------------------
# Worker-process caches + ``cfg_path`` IPC (mirrors ``openmeq.qc.stats``)
# ---------------------------------------------------------------------------

#: Per-worker-*process* caches, keyed by the path string each worker was
#: called with -- see ``openmeq.qc.stats``'s identical caches for the full
#: rationale (long-lived ``ProcessPoolExecutor`` workers handle many
#: day-groups over their lifetime; without this every task would re-parse
#: the whole ``inventory/index.json`` from scratch).
_worker_index_cache: dict[str, DataIndex] = {}
_worker_chunks_cfg_cache: dict[str, ChunksConfig] = {}


def _cached_index(index_path: str) -> DataIndex:
    index = _worker_index_cache.get(index_path)
    if index is None:
        index = DataIndex.load(index_path)
        _worker_index_cache[index_path] = index
    return index


def _load_chunks_cfg(cfg_path: str | Path) -> ChunksConfig:
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ChunksConfig.model_validate(raw.get("chunks", {}))


def _cached_chunks_cfg(cfg_path: str) -> ChunksConfig:
    chunks_cfg = _worker_chunks_cfg_cache.get(cfg_path)
    if chunks_cfg is None:
        chunks_cfg = _load_chunks_cfg(cfg_path)
        _worker_chunks_cfg_cache[cfg_path] = chunks_cfg
    return chunks_cfg


def _write_chunks_worker_cfg(chunks_cfg: ChunksConfig) -> Path:
    """Write ``chunks_cfg`` to a small standalone temp YAML file for worker IPC."""
    fd, tmp_name = tempfile.mkstemp(prefix="openmeq_chunks_cfg_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"chunks": chunks_cfg.model_dump(mode="json")}, f)
    return tmp_path


# ---------------------------------------------------------------------------
# Per-day-group worker
# ---------------------------------------------------------------------------


def chunk_day_group(
    day: str,
    detections: list[dict],
    index_path: str,
    cfg_path: str,
    chunks_dir: str,
) -> list[dict]:
    """Top-level, picklable ``pmap`` worker: write every chunk for one day-group.

    Performs exactly **one** whole-network :meth:`~openmeq.index.DataIndex.read`
    covering this group's overall window, then cuts and writes each
    detection's own chunk from that single in-memory ``Stream`` (see the
    module docstring).

    Parameters
    ----------
    day : str
        This group's UTC calendar date (``"YYYY-MM-DD"``, as produced by
        :func:`group_detections_by_day`). Used only for logging/diagnostics
        here -- the actual read window is derived from ``detections``'
        own times, not from ``day``.
    detections : list of dict
        This day's detection rows (``event_id``, ``time``, ...;
        ARCHITECTURE.md §4.3), as produced by
        :func:`group_detections_by_day`.
    index_path : str
        Path to a ``DataIndex``-format JSON file (``inventory/index.json``).
        Loaded once per worker *process* and cached thereafter.
    cfg_path : str
        Path to a small YAML file holding just a ``chunks:`` section (see
        the module docstring's ``cfg_path`` design note). Also cached per
        worker process.
    chunks_dir : str
        ``chunks/`` output directory
        (:attr:`openmeq.paths.RunPaths.chunk_dir`). Each event's chunk is
        written to ``<chunks_dir>/<event_id>/<event_id>.mseed``
        (:meth:`openmeq.paths.RunPaths.chunk_file`), creating the per-event
        directory as needed.

    Returns
    -------
    list of dict
        One ``{"event_id", "seed_id", "npts_present"}`` row per (event,
        channel) actually written (summed across any gap-split segments of
        that channel within one event's chunk) -- for M3.2 to optionally
        cross-check against, recomputed independently from the written
        files (this function does not compute ``fraction_present`` itself).
    """
    if not detections:
        return []

    index = _cached_index(index_path)
    chunks_cfg = _cached_chunks_cfg(cfg_path)
    pre_s = chunks_cfg.pre_s
    post_s = chunks_cfg.post_s

    times = [UTCDateTime(det["time"]) for det in detections]
    window_start = min(times) - pre_s - _READ_WINDOW_SAFETY_MARGIN_S
    window_end = max(times) + post_s + _READ_WINDOW_SAFETY_MARGIN_S

    # "All channels" per ARCHITECTURE.md §3/PLAN.md §6 Step 3 -- every
    # channel in the whole network, not just this detection's own
    # ``stations`` column.
    seed_ids = index.seed_ids()

    # The one whole-network read for this entire day-group (see module
    # docstring). ``.merge(method=0)`` is already what ``DataIndex.read``
    # itself does internally; the extra call here is a cheap no-op
    # (idempotent) that keeps this call site self-documenting about the
    # invariant it relies on before ``.split()``.
    stream = index.read(seed_ids, window_start, window_end).merge(method=0).split()

    chunks_root = Path(chunks_dir)
    results: list[dict] = []
    n_written = 0
    for det in detections:
        event_id = det["event_id"]
        t = UTCDateTime(det["time"])
        t1 = t - pre_s
        t2 = t + post_s

        # New Trace views, not mutating ``stream`` -- every other detection
        # in this group still needs to cut its own chunk from the same
        # original ``stream`` afterwards (module docstring).
        chunk = stream.slice(t1, t2)

        if len(chunk) == 0:
            logger.warning(
                "Event %s (day-group %s): no channel has any data in [%s, %s]; no chunk written",
                event_id,
                day,
                t1,
                t2,
            )
            continue

        event_dir = chunks_root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        out_path = event_dir / f"{event_id}.mseed"
        chunk.write(str(out_path), format="MSEED")
        n_written += 1

        npts_by_seed_id: dict[str, int] = {}
        for tr in chunk:
            npts_by_seed_id[tr.get_id()] = npts_by_seed_id.get(tr.get_id(), 0) + tr.stats.npts
        for seed_id, npts_present in npts_by_seed_id.items():
            results.append({"event_id": event_id, "seed_id": seed_id, "npts_present": npts_present})

    logger.info(
        "Day-group %s: wrote %d/%d chunk(s) from one %d-channel read [%s, %s]",
        day,
        n_written,
        len(detections),
        len(seed_ids),
        window_start,
        window_end,
    )
    return results


def _chunk_day_group_item(item: tuple[str, list[dict], str, str, str]) -> list[dict]:
    """``pmap``-compatible single-argument adapter around :func:`chunk_day_group`."""
    day, detections, index_path, cfg_path, chunks_dir = item
    return chunk_day_group(day, detections, index_path, cfg_path, chunks_dir)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_chunks(
    detections_df: pd.DataFrame,
    index: DataIndex,
    cfg: Config,
    paths: RunPaths,
) -> list[dict]:
    """Write every event's raw waveform chunk (``chunks/<event_id>/<event_id>.mseed``).

    Groups ``detections_df`` by UTC calendar day
    (:func:`group_detections_by_day`), then runs :func:`chunk_day_group`
    once per day-group through :func:`openmeq.parallel.pmap` (parallel
    across day-groups -- see the module docstring's "Grouping" section for
    why that is the efficiency-relevant unit here, not per-detection).

    Parameters
    ----------
    detections_df : pandas.DataFrame
        ``detections/detections.csv``'s columns (ARCHITECTURE.md §4.3).
    index : openmeq.index.DataIndex
        Already-loaded/-built index. Written to ``paths.index_json`` if not
        already present there (idempotent -- workers reload the index from
        that path rather than having it pickled into every work item, same
        pattern as ``openmeq.qc.stats.compute_daily_stats``).
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths

    Returns
    -------
    list of dict
        Every ``{"event_id", "seed_id", "npts_present"}`` row from every
        day-group (see :func:`chunk_day_group`'s return value), for a
        caller that wants them (e.g. M3.2's cross-check); ``write_chunks``
        itself does not write any CSV of these (that would be
        ``chunks/completeness.csv``, an independent recomputation owned by
        M3.2, out of this task's scope).
    """
    if detections_df.empty:
        logger.info("No detections to chunk.")
        return []

    index_path = paths.index_json
    if not index_path.exists():
        index.write(index_path)

    groups = group_detections_by_day(detections_df, cfg)

    cfg_path = _write_chunks_worker_cfg(cfg.chunks)
    try:
        items = [
            (day, day_detections, str(index_path), str(cfg_path), str(paths.chunk_dir))
            for day, day_detections in sorted(groups.items())
        ]
        logger.info(
            "Writing waveform chunks for %d detection(s) across %d day-group(s)",
            len(detections_df),
            len(items),
        )
        t0 = time.monotonic()
        batches = pmap(_chunk_day_group_item, items, cfg.parallel, desc="chunk")
        elapsed = max(time.monotonic() - t0, 1e-9)
    finally:
        cfg_path.unlink(missing_ok=True)

    results = [row for batch in batches for row in batch]
    n_events = len({row["event_id"] for row in results})
    logger.info("Wrote %d chunk(s) in %.2fs (%.1f chunks/s)", n_events, elapsed, n_events / elapsed)
    return results
