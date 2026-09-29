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

Sub-window splitting (M3.2 / ``OpenMEQ-rq8.2``, memory fix)
-------------------------------------------------------------
As originally built (M3.1), :func:`chunk_day_group` performed exactly
**one** whole-network read spanning ``[min(detection window start),
max(detection window end)]`` across an *entire day's* detections. On a
real network (e.g. 50 channels x a full 200 Hz day) that single read is
~3.5 GB, and multiplied across ``n_workers`` concurrent day-group tasks it
can exhaust memory -- flagged in the M3.1 orchestrator review.

:func:`chunk_day_group` now further splits each day group's (already
time-sorted) detections into **sub-windows**
(:func:`_split_detections_into_subwindows`): a new sub-window starts
whenever two consecutive detections' trigger times are more than
``cfg.chunks.max_read_window_s`` apart. Each sub-window still gets exactly
**one** whole-network read (the "one read per sub-window" property the
issue's notes require -- a cluster of nearby detections, e.g. an
aftershock sequence, is still read in a single call), but the read's time
span is now bounded by ``max_read_window_s`` plus the padding below,
rather than by the day's full first-to-last-detection span. The previous
single-read-per-day behaviour is exactly reproduced when every detection
in the day is within ``max_read_window_s`` of its neighbour (e.g. a dense
swarm, or a small/short-duration dataset) -- splitting only kicks in when
detections are genuinely spread out. Each sub-window's ``Stream`` is
dropped (falls out of scope) before the next sub-window is read, so peak
memory is bounded by one sub-window's span, not the whole day's.

**M11.3 follow-up (``OpenMEQ-m1f.3``): confirmed ``DataIndex.read`` does
not decode whole day files.** A later note on this issue asked whether a
still-observed ~830 MB process RSS while chunking a sparse day (8
stations) meant :meth:`~openmeq.index.DataIndex.read` was decoding a whole
day file even for a short sub-window read. Measured directly (M11.3):
reading a 60 s window out of a full 17.28M-sample/day file costs ~3.4 ms
and ~1.1 MB of Python-visible allocation, versus ~83 ms/~70 MB for the
whole file -- proportional to the *window*, not the file (ObsPy's native
miniSEED selective decode, exactly as this module's own docstring already
documented). The 830 MB figure is explained instead by
``resource.getrusage(...).ru_maxrss`` being a **whole-process**,
monotonically non-decreasing high-water mark (``scripts/benchmark.py``'s
own docstring flags this same caveat): by the time ``chunk`` runs, ``qc``
and ``detect`` have already touched several GB in the same process (at
``n_workers=1`` especially, where ``pmap`` runs every window in-process,
not a subprocess), and that earlier peak is still counted against
``chunk``'s number. A ``psutil``-sampled, stage-local peak (not
``ru_maxrss``) put ``chunk``'s own contribution at under 300 MB on the
same run. No code change was needed here; see ``docs/performance.md``
for the full measurement.

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
from typing import TYPE_CHECKING, Any

import click
import pandas as pd
import yaml
from obspy import UTCDateTime

from openmeq.chunk_diagnostics import (
    compute_completeness,
    make_chunk_diagnostics,
    make_chunk_extended_diagnostics,
    write_completeness_csv,
)
from openmeq.config import ChunksConfig
from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
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
# Sub-window splitting (M3.2 / OpenMEQ-rq8.2, memory fix -- see module docstring)
# ---------------------------------------------------------------------------


def _split_detections_into_subwindows(
    detections: list[dict], max_read_window_s: float
) -> list[list[dict]]:
    """Split one day-group's (time-sorted) detections into read sub-windows.

    A new sub-window starts whenever two consecutive detections' trigger
    times are more than ``max_read_window_s`` apart -- see the module
    docstring's "Sub-window splitting" section. Every detection appears in
    exactly one sub-window, in its original (time-sorted) order.

    Parameters
    ----------
    detections : list of dict
        One day-group's detection rows, already sorted by ``time``
        (:func:`group_detections_by_day`'s own contract -- this function
        does not re-sort).
    max_read_window_s : float
        ``cfg.chunks.max_read_window_s``. Must be positive; a detection
        list containing only one detection always yields a single
        one-element sub-window regardless of this value.

    Returns
    -------
    list of list of dict
        Non-empty sub-window groups, in time order. Empty only if
        ``detections`` itself is empty.
    """
    if not detections:
        return []
    subwindows: list[list[dict]] = [[detections[0]]]
    prev_time = UTCDateTime(detections[0]["time"])
    for det in detections[1:]:
        t = UTCDateTime(det["time"])
        if t - prev_time > max_read_window_s:
            subwindows.append([])
        subwindows[-1].append(det)
        prev_time = t
    return subwindows


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
    max_read_window_s = chunks_cfg.max_read_window_s

    # "All channels" per ARCHITECTURE.md §3/PLAN.md §6 Step 3 -- every
    # channel in the whole network, not just this detection's own
    # ``stations`` column.
    seed_ids = index.seed_ids()

    # M3.2 / OpenMEQ-rq8.2 memory fix: split this day-group's (already
    # time-sorted) detections into sub-windows so a single read never spans
    # more than ~max_read_window_s of a sparse day (module docstring's
    # "Sub-window splitting" section) -- still exactly one whole-network
    # read per sub-window (unchanged from M3.1 for a dense/short day, where
    # this yields a single sub-window covering the whole group).
    subwindows = _split_detections_into_subwindows(detections, max_read_window_s)

    chunks_root = Path(chunks_dir)
    results: list[dict] = []
    n_written = 0
    n_reads = 0
    for sub_detections in subwindows:
        times = [UTCDateTime(det["time"]) for det in sub_detections]
        window_start = min(times) - pre_s - _READ_WINDOW_SAFETY_MARGIN_S
        window_end = max(times) + post_s + _READ_WINDOW_SAFETY_MARGIN_S

        # The one whole-network read for this sub-window (see module
        # docstring). ``.merge(method=0)`` is already what ``DataIndex.read``
        # itself does internally; the extra call here is a cheap no-op
        # (idempotent) that keeps this call site self-documenting about the
        # invariant it relies on before ``.split()``.
        stream = index.read(seed_ids, window_start, window_end).merge(method=0).split()
        n_reads += 1

        for det in sub_detections:
            event_id = det["event_id"]
            t = UTCDateTime(det["time"])
            t1 = t - pre_s
            t2 = t + post_s

            # New Trace views, not mutating ``stream`` -- every other
            # detection in this sub-window still needs to cut its own
            # chunk from the same original ``stream`` afterwards (module
            # docstring).
            chunk = stream.slice(t1, t2)

            if len(chunk) == 0:
                logger.warning(
                    "Event %s (day-group %s): no channel has any data in [%s, %s]; "
                    "no chunk written",
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
                results.append(
                    {"event_id": event_id, "seed_id": seed_id, "npts_present": npts_present}
                )

        # Free this sub-window's data before reading the next one -- the
        # whole point of splitting is to bound peak memory to one
        # sub-window's span, not the day's full span (module docstring).
        del stream

    logger.info(
        "Day-group %s: wrote %d/%d chunk(s) from %d read(s) across %d sub-window(s) "
        "(%d channels, max_read_window_s=%.0f)",
        day,
        n_written,
        len(detections),
        n_reads,
        len(subwindows),
        len(seed_ids),
        max_read_window_s,
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


# ---------------------------------------------------------------------------
# Pipeline orchestration / CLI registration (M3.2, ``OpenMEQ-rq8.2``)
# ---------------------------------------------------------------------------
#
# ``run(cfg, paths, **opts)`` is what ``openmeq/cli.py`` looks for on this
# module (ARCHITECTURE.md §2's ``openmeq.cli`` contract) -- registering the
# ``openmeq chunk`` step needs no change to ``cli.py`` itself. Mirrors
# ``openmeq.detect.pipeline.run_detect``'s own "require inputs, fingerprint,
# skip-unchanged, do the work, mark done, log elapsed time" shape.

#: ``RunPaths.diagnostics_dir``/``manifest_json``'s step-folder-name argument
#: -- the output folder (``chunks/``), matching ``paths.chunk_completeness_csv``
#: and ``paths.diagnostics_dir("chunks")``'s own convention (ARCHITECTURE.md §2).
STEP_NAME = "chunks"

#: The single :class:`~openmeq.manifest.StepManifest` entry key for this
#: step -- one whole-run unit of work, mirroring
#: ``openmeq.detect.pipeline.MANIFEST_KEY``.
MANIFEST_KEY = "chunk"


def _require_detect_outputs(paths: RunPaths) -> None:
    """Raise a clear, actionable error if ``openmeq detect``'s outputs are missing.

    ``openmeq chunk`` reads ``detections/detections.csv`` (to know which
    events to cut) and ``inventory/index.json`` (to find/read the waveform
    files) -- the former written by ``openmeq detect`` (M2.3), the latter by
    ``openmeq qc`` (M1.7). A fresh run directory, or one where only ``qc``
    has run, is the expected first-run-out-of-order mistake; this is a
    deliberate, CLI-level hard dependency check (mirroring
    ``openmeq.detect.pipeline._require_qc_outputs``), not an incidental
    ``FileNotFoundError`` raised deeper inside :func:`write_chunks`.

    Raises
    ------
    click.ClickException
        Naming every missing path and telling the user to run ``openmeq
        detect`` first.
    """
    missing = [p for p in (paths.detections_csv, paths.index_json) if not p.is_file()]
    if not missing:
        return
    names = ", ".join(str(p) for p in missing)
    message = (
        f"openmeq chunk: missing required input(s): {names}. Run `openmeq detect` first "
        "(and `openmeq qc` before that, if inventory/index.json is missing)."
    )
    logger.error(message)
    raise click.ClickException(message)


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    force: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq chunk`` step entry point: write every event's raw waveform
    chunk, then recompute and report completeness.

    1. Requires ``detections/detections.csv``/``inventory/index.json`` to
       already exist (:func:`_require_detect_outputs`).
    2. Skips (unless ``force``) if ``cfg.chunks`` and both input files'
       mtimes/sizes match the last recorded fingerprint
       (:class:`openmeq.manifest.StepManifest`) and ``chunks/
       completeness.csv`` still exists -- the manifest fingerprint includes
       ``detections.csv``'s own mtime (not just ``cfg.chunks``), so new
       detections with no config change still trigger re-chunking.
    3. :func:`write_chunks` (M3.1).
    4. :func:`openmeq.chunk_diagnostics.compute_completeness` /
       :func:`~openmeq.chunk_diagnostics.write_completeness_csv` /
       :func:`~openmeq.chunk_diagnostics.make_chunk_diagnostics` (M3.2),
       recomputed independently from the just-written chunk files (never
       from :func:`write_chunks`'s own in-memory return value -- see
       :mod:`openmeq.chunk_diagnostics`'s module docstring). Then
       :func:`~openmeq.chunk_diagnostics.make_chunk_extended_diagnostics`
       (opt-in ``chunks/diagnostics/extended/``; a no-op unless
       ``cfg.diagnostics.extended_enabled("chunk")``).
    5. Marks the manifest entry done and logs the elapsed time.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        Unused by this step (:func:`write_chunks` manages its own small
        worker-IPC config file internally, unlike ``openmeq detect``) --
        accepted only because ``openmeq/cli.py``'s dispatcher always
        supplies it (ARCHITECTURE.md §2's ``run(cfg, paths, **opts)``
        contract).
    force : bool, keyword-only, optional
        ``--force``: rerun even if the manifest says nothing changed.
    **_ignored
        Absorbs any other step-generic kwargs the dispatcher forwards.
    """
    del config_path  # unused -- see docstring
    t0 = time.perf_counter()
    paths.ensure()
    step_logger = get_logger("chunk")

    _require_detect_outputs(paths)

    fp = fingerprint(cfg.chunks, paths.detections_csv, paths.index_json)
    manifest = StepManifest(paths, STEP_NAME)
    if not force and manifest.is_done(MANIFEST_KEY, fp) and paths.chunk_completeness_csv.is_file():
        step_logger.info(
            "chunk: unchanged since the last run (cfg.chunks, detections/detections.csv and "
            "inventory/index.json all match the last recorded fingerprint) -- skipping "
            "(use --force to rerun)."
        )
        return

    detections_df = pd.read_csv(paths.detections_csv)
    index = DataIndex.load(paths.index_json)

    write_chunks(detections_df, index, cfg, paths)

    completeness_df = compute_completeness(detections_df, cfg, paths)
    write_completeness_csv(completeness_df, paths.chunk_completeness_csv)
    make_chunk_diagnostics(paths, completeness_df)
    make_chunk_extended_diagnostics(cfg, paths, list(detections_df["event_id"]))

    manifest.mark_done(MANIFEST_KEY, fp)

    elapsed = time.perf_counter() - t0
    step_logger.info(
        "chunk: %d event(s), %d completeness row(s) (%.2fs).",
        len(detections_df),
        len(completeness_df),
        elapsed,
    )
