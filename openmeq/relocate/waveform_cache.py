"""Per-event pre-filtered waveform cache for cross-correlation (M10.1 /
``OpenMEQ-16d.1``, ``docs/ARCHITECTURE.md`` §6f).

Cross-correlation (M10.3) needs the same short waveform window, filtered the
same way, potentially many times per event (once per linked pair). Rather
than have M10.3 re-read/re-filter the raw ``chunks/<event_id>/<event_id>.mseed``
file for every pair, this module builds a per-event cache **once**, up
front: :func:`build_cache` reads each event's raw chunk, applies the
``cfg.hypodd.xcorr.filter`` bandpass exactly once, resamples every trace to
one project-wide common rate, cuts a tight (but generously padded) window
around each pick, and writes the result to
``relocate/cache/<event_id>.npz`` (:meth:`openmeq.paths.RunPaths.relocate_cache_file`).

Fixes from ``docs/hypoddpy_audit.md`` §6 this module is responsible for
------------------------------------------------------------------------
- **Defect #2** (hardcoded 1s-before/5s-after stream read window, ignoring
  the configured before/after bounds): the cached window is always derived
  directly from ``cfg.hypodd.xcorr.window_p``/``window_s`` (see "Window
  sizing" below), never a fixed literal.
- **Defect #6** (an in-memory, per-run-regenerated sequential id used as an
  unvalidated cross-correlation cache key): the cache is keyed on the
  stable, never-changing ``event_id`` (``docs/ARCHITECTURE.md`` §4.4) --
  both as the output filename and as the :class:`openmeq.manifest.StepManifest`
  key -- and is only ever treated as valid when that key's fingerprint
  (config section + input file identity, below) still matches; there is no
  separate sequential/in-memory id anywhere in this module.
- **Defect #8** (sampling-rate mismatch causes a pair to be silently
  skipped, no resampling attempted): every cached trace is resampled to one
  fixed, documented, project-wide ``cfg.hypodd.xcorr.resample_rate_hz``
  (added by this task -- see ``openmeq.config.XcorrConfig``), so M10.3 never
  has to reconcile two different sampling rates itself.

Window sizing
-------------
For a station with a P and/or S pick in the event, the cached window for
each relevant channel is the union, over every applicable pick, of::

    [pick_time + window[0] - maxlag_s - filter_edge_margin_s,
     pick_time + window[1] + maxlag_s + filter_edge_margin_s]

using ``cfg.hypodd.xcorr.window_p``/``window_s`` (each an already-signed
``(before, after)`` offset pair relative to the pick, e.g. ``(-0.05, 0.2)``
-- ``openmeq.config.XcorrConfig``'s literal convention, matching
``config/example.yaml``'s sketch) for whichever phase(s) apply, plus
``cfg.hypodd.xcorr.maxlag_s`` (room for M10.3's own lag search) and a small
edge-effect margin, ``cfg.hypodd.xcorr.filter_edge_margin_s`` (default 0.5 s), for the
bandpass filter applied in this same step. Which channel(s) a phase applies
to comes from ``cfg.hypodd.xcorr.components`` (``{"P": [...], "S": [...]}``,
matched against the channel's *mapped* last character, after
``openmeq.stations.apply_channel_map`` -- ``docs/ARCHITECTURE.md`` §6): a P
window applies to every channel whose mapped last character is one of
``components["P"]`` (typically just ``"Z"``); an S window applies to every
channel whose mapped last character is one of ``components["S"]``
(typically the horizontals, ``"1"``/``"2"``/``"N"``/``"E"``). A channel
with no applicable window at all (no pick at that station covers it) is not
cached -- this cache is deliberately **not** a full-chunk copy (the design
note's own "Pitfalls": caching more than necessary defeats the purpose).

Processing order (once per event, not once per pair)
-----------------------------------------------------
For each channel that has at least one applicable window: demean, detrend
(linear), taper (``cfg.hypodd.xcorr.taper_pct``), bandpass filter
(``cfg.hypodd.xcorr.filter``: band, ``corners``, ``zerophase`` --
preserves the pick-time alignment M10.3's lag search depends on) -- all
applied to the *whole* available chunk segment first, then resampled to
``cfg.hypodd.xcorr.resample_rate_hz`` (skipped if already at that rate), and
only *then* sliced down to the padded window above. Filtering/resampling
before slicing (not after) means the taper's edge effects fall on the wide
chunk boundary, not on the tight cache window's own edges, and the bandpass
is computed exactly once per channel here -- never repeated per pair by
M10.3, which would both waste CPU and (per the design note) subtly change
the signal's spectral content each time.

Gap handling
------------
The raw chunk may hold more than one contiguous ``Trace`` segment for the
same channel (``openmeq.chunk``'s own ``merge().split()`` convention, when a
real data gap falls inside the chunked window). Each segment is windowed
independently; if more than one segment yields a non-empty slice for the
same final (mapped) seed id, the segment with the most samples is kept
(logged), rather than raising or arbitrarily picking the first.

npz layout (``relocate/cache/<event_id>.npz``, this module's own choice --
documented here since M10.3 depends on it directly)
------------------------------------------------------------------------
- ``"seed_ids"``: a 1-D array of ``str`` -- every seed id (``NET.STA.LOC.CHA``,
  post channel-mapping) cached for this event. May be empty (an event with
  no P/S picks still gets a valid, trivially-empty cache file -- this is
  different from an event being *skipped* below, which never writes a file
  at all).
- ``"sampling_rate"``: a scalar ``float64`` -- ``cfg.hypodd.xcorr.resample_rate_hz``,
  the one rate every array below is sampled at.
- for each seed id in ``"seed_ids"``: ``"<seed_id>__data"`` (``float32``,
  the windowed, filtered, resampled trace) and ``"<seed_id>__start"``
  (scalar ``float64``, that trace's absolute start time as POSIX seconds --
  deliberately a plain float, not a pickled ``obspy.UTCDateTime``, so the
  cache format stays simple/portable and independent of ObsPy's pickle
  compatibility across versions, per the design note).

Skip-unchanged and missing/corrupt chunks
-------------------------------------------
:func:`build_cache` is keyed, via :class:`openmeq.manifest.StepManifest`
(``relocate/manifest.json``), on the event's own stable ``event_id`` (never
a transient sequential id -- defect #6 above), fingerprinted
(:func:`openmeq.manifest.fingerprint`) on ``cfg.hypodd.xcorr`` (every field
that affects the cache: window/filter/maxlag/resample rate/components) plus
the raw chunk file's on-disk identity (mtime/size) plus that event's own
picks (net_sta/phase/time) -- so a config change, a rewritten chunk, or a
changed pick all correctly invalidate the cache, but re-running with
nothing changed rewrites no ``.npz`` file at all.

An event whose chunk file is missing, or fails to parse (corrupt/truncated
miniSEED), is logged (with the reason) and skipped -- never marked done in
the manifest (so a later run, once the chunk is fixed, retries it
automatically) and the run as a whole continues with the remaining events;
it never raises.

Parallelism
-----------
One event per :func:`openmeq.parallel.pmap` work item (PLAN.md §4's own
"xcorr (dt.cc): event pair (batched)" row notes the cache itself is built
**per event**, since a pair's window is derived from each event's own picks
independently -- the pairing happens later, in M10.3). The picks catalogue
(``paths.catalog_mag_xml``) is read once, in the main process, and only the
small per-event pick list (not the whole catalogue, and not the whole
:class:`~openmeq.config.Config`) is sent to each worker -- mirroring
``openmeq.chunk``'s own ``cfg_path`` worker-IPC convention for the
``stations``/``xcorr`` config sections a worker actually needs.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from obspy import UTCDateTime, read

from openmeq.catalog import event_id as _event_id_of
from openmeq.catalog import read_catalog
from openmeq.config import StationsConfig, XcorrConfig
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap
from openmeq.stations import apply_channel_map

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate.waveform_cache")

#: Manifest step name (``relocate/manifest.json``,
#: ``docs/ARCHITECTURE.md`` §6c-2: shared with M9.5's ``"hypodd_setup"``
#: key in the same file -- keys are per-entry, not per-file; an event_id
#: never collides with that string).
_MANIFEST_STEP = "relocate"


# ---------------------------------------------------------------------------
# Picks: read the catalogue once, in the main process
# ---------------------------------------------------------------------------


def _event_picks(event) -> list[dict[str, str]]:
    """Every non-rejected P/S pick on ``event``, as small picklable dicts.

    Returns ``[{"net_sta": "NET.STA", "phase_hint": "P"|"S", "time": iso}, ...]``.
    Rejected picks (``evaluation_status == "rejected"``,
    ``docs/ARCHITECTURE.md`` §5) and picks with any other ``phase_hint`` are
    excluded.
    """
    picks: list[dict[str, str]] = []
    for pick in event.picks:
        if pick.evaluation_status == "rejected":
            continue
        phase = pick.phase_hint
        if phase not in ("P", "S"):
            continue
        wf = pick.waveform_id
        if wf is None or not wf.network_code or not wf.station_code:
            continue
        picks.append(
            {
                "net_sta": f"{wf.network_code}.{wf.station_code}",
                "phase_hint": phase,
                "time": str(pick.time),
            }
        )
    return picks


def _load_picks_by_event(paths: RunPaths, wanted: set[str]) -> dict[str, list[dict[str, str]]]:
    """``{event_id: [pick dicts]}`` for every ``wanted`` event id found in
    ``paths.catalog_mag_xml``.

    An event id in ``wanted`` but absent from the catalogue (or the
    catalogue file not existing yet) simply has no entry -- treated
    downstream as "no picks", not an error; the chunk-missing/corrupt check
    is the only thing that actually skips an event.
    """
    catalog_path = Path(paths.catalog_mag_xml)
    if not catalog_path.exists():
        logger.warning(
            "Waveform cache: %s does not exist; building with no picks for every event "
            "(every cache will have 0 seed_ids until the magnitude catalogue exists)",
            catalog_path,
        )
        return {}
    catalog = read_catalog(catalog_path)
    picks_by_event: dict[str, list[dict[str, str]]] = {}
    for event in catalog:
        eid = _event_id_of(event)
        if eid in wanted:
            picks_by_event[eid] = _event_picks(event)
    return picks_by_event


# ---------------------------------------------------------------------------
# Worker-process caches + ``cfg_path`` IPC (mirrors ``openmeq.chunk``)
# ---------------------------------------------------------------------------

_worker_stations_cfg_cache: dict[str, StationsConfig] = {}
_worker_xcorr_cfg_cache: dict[str, XcorrConfig] = {}


def _write_worker_cfg(cfg: Config) -> Path:
    """Write just the ``stations``/``xcorr`` sections a worker needs to a
    small standalone temp YAML file (worker IPC; see module docstring)."""
    fd, tmp_name = tempfile.mkstemp(prefix="openmeq_wavecache_cfg_", suffix=".yaml")
    os.close(fd)
    tmp_path = Path(tmp_name)
    payload = {
        "stations": cfg.stations.model_dump(mode="json"),
        "xcorr": cfg.hypodd.xcorr.model_dump(mode="json"),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)
    return tmp_path


def _cached_worker_cfg(cfg_path: str) -> tuple[StationsConfig, XcorrConfig]:
    stations_cfg = _worker_stations_cfg_cache.get(cfg_path)
    xcorr_cfg = _worker_xcorr_cfg_cache.get(cfg_path)
    if stations_cfg is None or xcorr_cfg is None:
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        stations_cfg = StationsConfig.model_validate(raw.get("stations", {}))
        xcorr_cfg = XcorrConfig.model_validate(raw.get("xcorr", {}))
        _worker_stations_cfg_cache[cfg_path] = stations_cfg
        _worker_xcorr_cfg_cache[cfg_path] = xcorr_cfg
    return stations_cfg, xcorr_cfg


# ---------------------------------------------------------------------------
# Per-event worker
# ---------------------------------------------------------------------------


def _picks_by_station(
    picks: list[dict[str, str]],
) -> dict[str, dict[str, list[UTCDateTime]]]:
    by_station: dict[str, dict[str, list[UTCDateTime]]] = {}
    for pick in picks:
        entry = by_station.setdefault(pick["net_sta"], {"P": [], "S": []})
        entry[pick["phase_hint"]].append(UTCDateTime(pick["time"]))
    return by_station


def _applicable_windows(
    comp: str,
    station_picks: dict[str, list[UTCDateTime]],
    xcorr_cfg: XcorrConfig,
) -> list[tuple[UTCDateTime, UTCDateTime]]:
    """Every ``(start, end)`` padded window ``comp`` (a mapped channel's last
    character) is relevant to, given this station's P/S pick times."""
    windows: list[tuple[UTCDateTime, UTCDateTime]] = []
    maxlag = xcorr_cfg.maxlag_s
    p_components = set(xcorr_cfg.components.get("P", []))
    s_components = set(xcorr_cfg.components.get("S", []))
    pad = maxlag + xcorr_cfg.filter_edge_margin_s
    if comp in p_components:
        for t in station_picks.get("P", []):
            before, after = xcorr_cfg.window_p
            windows.append((t + before - pad, t + after + pad))
    if comp in s_components:
        for t in station_picks.get("S", []):
            before, after = xcorr_cfg.window_s
            windows.append((t + before - pad, t + after + pad))
    return windows


def _process_and_slice(
    trace,
    win_start: UTCDateTime,
    win_end: UTCDateTime,
    xcorr_cfg: XcorrConfig,
):
    """Demean/detrend/taper/filter/resample the whole segment, then slice.

    Returns ``None`` if anything fails or the sliced result is empty --
    never raises (the caller logs and continues with other channels).
    """
    proc = trace.copy()
    try:
        proc.detrend("demean")
        proc.detrend("linear")
        proc.taper(max_percentage=xcorr_cfg.taper_pct, type="hann")
        proc.filter(
            "bandpass",
            freqmin=xcorr_cfg.filter.freqmin,
            freqmax=xcorr_cfg.filter.freqmax,
            corners=xcorr_cfg.filter.corners,
            zerophase=xcorr_cfg.filter.zerophase,
        )
        if abs(proc.stats.sampling_rate - xcorr_cfg.resample_rate_hz) > 1e-6:
            proc.resample(xcorr_cfg.resample_rate_hz)
        sliced = proc.slice(win_start, win_end)
    except Exception as exc:  # noqa: BLE001 -- never let one bad channel abort the event
        logger.warning("Failed to process/slice %s: %s", trace.get_id(), exc)
        return None
    if sliced is None or sliced.stats.npts == 0:
        return None
    return sliced


def build_event_cache(
    event_id: str,
    chunk_file: str,
    cache_file: str,
    cfg_path: str,
    picks: list[dict[str, str]],
) -> dict[str, Any]:
    """Build one event's ``relocate/cache/<event_id>.npz``.

    Top-level, picklable ``pmap`` worker (see module docstring). Does not
    itself decide skip-unchanged (the caller, :func:`build_cache`, only
    dispatches events whose manifest fingerprint changed) and does not
    write to the shared :class:`~openmeq.manifest.StepManifest` (workers
    never touch shared mutable state, ``openmeq.manifest``'s own
    concurrency assumption) -- it only returns a small result dict for the
    caller to log/mark-done from.

    Parameters
    ----------
    event_id : str
        The stable event id (never a transient sequential id -- audit
        defect #6).
    chunk_file : str
        Path to ``chunks/<event_id>/<event_id>.mseed``.
    cache_file : str
        Destination path, ``relocate/cache/<event_id>.npz``
        (:meth:`openmeq.paths.RunPaths.relocate_cache_file`).
    cfg_path : str
        Path to the small YAML file written by :func:`_write_worker_cfg`
        (``stations``/``xcorr`` sections only).
    picks : list of dict
        This event's picks, as produced by :func:`_event_picks` (may be
        empty).

    Returns
    -------
    dict
        ``{"event_id", "status", "reason", "n_seed_ids", "bytes"}``.
        ``status`` is one of ``"built"``, ``"missing_chunk"`` or
        ``"unreadable_chunk"``. ``"reason"``/``"bytes"`` are ``None``
        unless relevant.
    """
    stations_cfg, xcorr_cfg = _cached_worker_cfg(cfg_path)

    chunk_path = Path(chunk_file)
    if not chunk_path.exists():
        return {
            "event_id": event_id,
            "status": "missing_chunk",
            "reason": f"chunk file not found: {chunk_path}",
            "n_seed_ids": 0,
            "bytes": None,
        }

    try:
        stream = read(str(chunk_path))
    except Exception as exc:  # noqa: BLE001 -- corrupt/truncated input must not crash the run
        return {
            "event_id": event_id,
            "status": "unreadable_chunk",
            "reason": f"{type(exc).__name__}: {exc}",
            "n_seed_ids": 0,
            "bytes": None,
        }

    # Defensive: the chunk file is already gap-split, contiguous miniSEED
    # (openmeq.chunk's own convention), but re-merge/re-split so this module
    # never depends on the on-disk file already being in that exact shape.
    stream = stream.merge(method=0).split()
    stream = apply_channel_map(stream, stations_cfg)

    picks_by_station = _picks_by_station(picks)

    best: dict[str, tuple[np.ndarray, float, int]] = {}
    for trace in stream:
        net_sta = f"{trace.stats.network}.{trace.stats.station}"
        station_picks = picks_by_station.get(net_sta)
        if not station_picks:
            continue
        channel = trace.stats.channel
        comp = channel[-1] if channel else ""
        windows = _applicable_windows(comp, station_picks, xcorr_cfg)
        if not windows:
            continue
        win_start = min(w[0] for w in windows)
        win_end = max(w[1] for w in windows)

        sliced = _process_and_slice(trace, win_start, win_end, xcorr_cfg)
        if sliced is None:
            continue

        seed_id = sliced.get_id()
        npts = sliced.stats.npts
        existing = best.get(seed_id)
        if existing is not None and existing[2] >= npts:
            if existing[2] > npts:
                logger.debug(
                    "Event %s: %s has multiple overlapping chunk segments in its cache "
                    "window (a gap fell inside it); keeping the longer one (%d vs %d samples)",
                    event_id,
                    seed_id,
                    existing[2],
                    npts,
                )
            continue
        data32 = sliced.data.astype(np.float32)
        start_ts = float(sliced.stats.starttime.timestamp)
        best[seed_id] = (data32, start_ts, npts)

    arrays: dict[str, np.ndarray] = {}
    seed_ids = sorted(best.keys())
    for seed_id in seed_ids:
        data, start, _npts = best[seed_id]
        arrays[f"{seed_id}__data"] = data
        arrays[f"{seed_id}__start"] = np.array(start, dtype=np.float64)
    arrays["seed_ids"] = np.array(seed_ids, dtype=str)
    arrays["sampling_rate"] = np.array(xcorr_cfg.resample_rate_hz, dtype=np.float64)

    out_path = Path(cache_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    return {
        "event_id": event_id,
        "status": "built",
        "reason": None,
        "n_seed_ids": len(seed_ids),
        "bytes": out_path.stat().st_size,
    }


def _build_event_cache_item(
    item: tuple[str, str, str, str, list[dict[str, str]]],
) -> dict[str, Any]:
    """``pmap``-compatible single-argument adapter around :func:`build_event_cache`."""
    event_id, chunk_file, cache_file, cfg_path, picks = item
    return build_event_cache(event_id, chunk_file, cache_file, cfg_path, picks)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_cache(cfg: Config, paths: RunPaths, event_ids: Iterable[str]) -> None:
    """Build (or skip-unchanged) ``relocate/cache/<event_id>.npz`` for every
    id in ``event_ids``.

    Reads ``paths.catalog_mag_xml`` once (in this process) for picks, then
    runs one :func:`build_event_cache` work item per event needing a
    (re)build through :func:`openmeq.parallel.pmap` (event ids already
    unchanged -- per :class:`openmeq.manifest.StepManifest`, keyed on the
    event's own stable id -- are never dispatched at all, so their
    ``.npz`` file is left byte-for-byte untouched).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    event_ids : iterable of str
        Every event id to (re)build a cache for (typically every accepted
        event in the catalogue hypoDD will relocate).
    """
    event_ids = list(event_ids)
    if not event_ids:
        logger.info("Waveform cache: no events given; nothing to do.")
        return

    Path(paths.relocate_cache_dir).mkdir(parents=True, exist_ok=True)
    picks_by_event = _load_picks_by_event(paths, set(event_ids))

    manifest = StepManifest(paths, _MANIFEST_STEP)
    xcorr_cfg = cfg.hypodd.xcorr

    to_process: list[tuple[str, str, list[dict[str, str]]]] = []
    n_unchanged = 0
    for event_id in event_ids:
        chunk_file = paths.chunk_file(event_id)
        picks = picks_by_event.get(event_id, [])
        fp = fingerprint(xcorr_cfg, chunk_file, picks)
        if manifest.is_done(event_id, fp):
            n_unchanged += 1
            continue
        to_process.append((event_id, fp, picks))

    if not to_process:
        logger.info(
            "Waveform cache: %d/%d event(s) already up to date; nothing to (re)build.",
            n_unchanged,
            len(event_ids),
        )
        return

    cfg_path = _write_worker_cfg(cfg)
    try:
        items = [
            (
                event_id,
                str(paths.chunk_file(event_id)),
                str(paths.relocate_cache_file(event_id)),
                str(cfg_path),
                picks,
            )
            for event_id, _fp, picks in to_process
        ]
        logger.info(
            "Waveform cache: building %d event(s) (%d already up to date)",
            len(items),
            n_unchanged,
        )
        t0 = time.monotonic()
        results = pmap(_build_event_cache_item, items, cfg.parallel, desc="relocate-cache")
        elapsed = max(time.monotonic() - t0, 1e-9)
    finally:
        cfg_path.unlink(missing_ok=True)

    n_built = 0
    total_bytes = 0
    for (event_id, fp, _picks), result in zip(to_process, results, strict=True):
        status = result["status"]
        if status == "built":
            manifest.mark_done(event_id, fp)
            n_built += 1
            total_bytes += result["bytes"] or 0
        else:
            logger.warning(
                "Waveform cache: event %s skipped (%s): %s",
                event_id,
                status,
                result.get("reason"),
            )

    avg_kb = (total_bytes / n_built / 1024.0) if n_built else 0.0
    logger.info(
        "Waveform cache: built %d/%d event(s) in %.2fs (%.2f events/s, %.1f KB/event avg, "
        "%d skipped)",
        n_built,
        len(to_process),
        elapsed,
        n_built / elapsed,
        avg_kb,
        len(to_process) - n_built,
    )
