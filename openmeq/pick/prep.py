"""PhaseNet chunk preprocessing (M4.2, ``OpenMEQ-1m7.2``).

PLAN.md §6 Step 4 "Preparation" / §1 constraint ("we decimate 200 -> 100 Hz
ourselves"): turns a raw ``chunks/<event_id>/<event_id>.mseed`` chunk (M3.1,
all channels, original sampling rate) into a PhaseNet-ready
``chunks_pn/<event_id>/<event_id>.mseed`` (100 Hz, exactly 3 components per
station, channel codes ending in ``Z``/``N``/``E``/``1``/``2``), plus the
``chunks_pn/data_list.csv`` PhaseNet needs (:mod:`openmeq.pick.data_list`).

Pipeline (:func:`prepare_chunk`, one call per event)
-----------------------------------------------------
a. Read the chunk; merge multi-segment traces, filling gaps up to
   ``cfg.phasenet.max_gap_fill_s`` by linear interpolation and zero-filling
   anything still masked afterwards (a gap longer than that, or a gap that
   touches the very start/end of the trace with nothing to interpolate
   from) -- the trace is never dropped.
b. ``detrend("demean")`` then ``detrend("linear")``.
c. Hann taper (``cfg.phasenet.taper_pct``).
d. Optional zero-phase high-pass (``cfg.phasenet.highpass_hz``), before
   decimation.
e. Anti-alias decimate each trace individually (verified per-trace, not
   assumed uniform) from its own sampling rate to
   ``cfg.phasenet.target_rate_hz``: obspy's ``lowpass_cheby_2`` (the filter
   ``Trace.decimate`` uses) applied forward and backward, i.e. zero-phase,
   then ``Trace.decimate(no_filter=True)``. ``Trace.decimate``'s own filter
   is causal and delayed 5-20 Hz onsets by 20-30 ms at 200 -> 100 Hz, which
   shifted every PhaseNet pick late (OpenMEQ-tr9). A trace already at the
   target rate is a no-op; a non-integer ratio raises ``ValueError``.
f. **Location-code collation**: PhaseNet groups by ``NET.STA.LOC.BandInst``
   (docs/phasenet_interface.md §4.6), so two location codes at one
   ``NET.STA`` would otherwise look like two different stations, each
   potentially zero-filled to a spuriously "complete" 3-component group
   (ARCHITECTURE.md §6: "location codes are ignored when two share a
   site"). Exactly one location code per ``NET.STA`` per event survives:
   whichever has the most channels present (ties broken alphabetically by
   location code); the rest are dropped and the choice is logged.
g. :func:`openmeq.stations.apply_channel_map` -- after this, every
   channel's last character is ``Z``, ``N``/``E`` (rotated, known
   orientation) or ``1``/``2`` (unoriented); the original code is preserved
   in ``trace.stats.openmeq_original_channel``.
h. **Ensure exactly 3 components per station** (grouped by
   ``NET.STA.LOC.BandInst``, PhaseNet's own grouping key): a station with
   fewer than 3 present gets a zero-filled synthetic trace (header copied
   from a real trace at that station, data all zero) per missing
   component, recorded in ``chunks_pn/prep_report.csv``
   (:data:`PREP_REPORT_COLUMNS`) -- only stations that needed at least one
   zero-filled component get a row (a fully-complete station is not
   reported). Whether the missing set is ``{N, E}``/one of them or
   ``{"1", "2"}``/one of them is inferred from whatever horizontal(s) *are*
   present; a station missing horizontals entirely defaults to the
   unoriented ``{"1", "2"}`` convention (the safe default -- it never
   asserts a known orientation that wasn't observed).
i. Write ``chunks_pn/<event_id>/<event_id>.mseed``.

:func:`prepare_all` is the orchestration entry point PLAN.md §4's "PhaseNet
prep: event" row describes (M4.3's batch runner calls this): one
:func:`openmeq.parallel.pmap` unit of work per event, skip-unchanged via
:class:`openmeq.manifest.StepManifest` (fingerprint = ``cfg.phasenet`` +
``cfg.stations`` + the raw chunk file's mtime/size), then
:func:`openmeq.pick.data_list.write_data_list`.

Per-event zero-fill sidecar files
----------------------------------
:func:`prepare_chunk` itself only ever touches its own event's directory
(``chunks_pn/<event_id>/``), so it can run in the worker processes
:func:`openmeq.parallel.pmap` spawns with no risk of two workers writing the
same file: any zero-fill rows it produces are written to a private sidecar,
``chunks_pn/<event_id>/_prep_report.json`` (not part of the documented
layout -- an implementation detail), and :func:`prepare_all` aggregates
every event's sidecar into the one public ``chunks_pn/prep_report.csv`` the
run directory layout documents, deleting the sidecars afterward. A test that
calls :func:`prepare_chunk` directly (not through :func:`prepare_all`) can
still assert on the sidecar directly, or on the log warnings, without
needing the aggregation step.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import Stream, Trace, read
from obspy.signal.filter import lowpass_cheby_2

from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap
from openmeq.paths import RunPaths
from openmeq.pick.data_list import write_data_list
from openmeq.stations import STATION_TABLE_COLUMNS, apply_channel_map
from openmeq.stations import station_table as compute_station_table

if TYPE_CHECKING:
    from openmeq.config import Config

logger = get_logger("pick.prep")

#: ``chunks_pn/prep_report.csv`` column order (ARCHITECTURE.md §3 extension,
#: M4.2 / OpenMEQ-1m7.2 -- see this module's docstring, item (h)).
PREP_REPORT_COLUMNS = ["event_id", "seed_id_base", "n_components_found", "zero_filled_components"]

#: Sidecar filename written under ``chunks_pn/<event_id>/`` by
#: :func:`prepare_chunk` (private -- not part of the documented run
#: directory layout; aggregated and removed by :func:`prepare_all`).
_SIDECAR_NAME = "_prep_report.json"

#: A ratio must be within this fraction of an integer to count as "integer"
#: (guards against float round-off in ``sampling_rate / target_rate_hz``,
#: not intended to tolerate a genuinely non-integer decimation factor).
_RATIO_TOL = 1e-6

#: Part of every event's skip-unchanged fingerprint: bump it when the
#: preparation itself changes so existing ``chunks_pn/`` files are redone
#: (2: zero-phase anti-alias filter, OpenMEQ-tr9).
_PREP_VERSION = 2


# ---------------------------------------------------------------------------
# a. merge + gap fill
# ---------------------------------------------------------------------------


def _masked_run_lengths(mask: np.ndarray) -> list[int]:
    """Lengths of every maximal run of ``True`` in a 1-D boolean array."""
    if not mask.any():
        return []
    # Boundaries: a run starts where the previous value differs (or at 0).
    is_start = np.empty(mask.size, dtype=bool)
    is_start[0] = mask[0]
    np.logical_and(mask[1:], ~mask[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)

    is_end = np.empty(mask.size, dtype=bool)
    is_end[-1] = mask[-1]
    np.logical_and(mask[:-1], ~mask[1:], out=is_end[:-1])
    ends = np.flatnonzero(is_end)
    return (ends - starts + 1).tolist()


def _linear_interpolate_masked(data) -> np.ndarray:
    """Fill a masked array's gaps by linear interpolation between the
    nearest valid samples on either side (``numpy.interp``, which holds the
    nearest valid value constant for a masked run at either edge)."""
    filled = np.ma.getdata(data).astype(np.float64, copy=True)
    mask = np.ma.getmaskarray(data)
    if not mask.any():
        return filled
    valid_idx = np.flatnonzero(~mask)
    if valid_idx.size == 0:
        return np.zeros(filled.shape, dtype=np.float64)
    all_idx = np.arange(filled.size)
    interpolated = np.interp(all_idx, valid_idx, filled[valid_idx])
    filled[mask] = interpolated[mask]
    return filled


def _merge_and_fill_gaps(stream: Stream, event_id: str, max_gap_fill_s: float) -> Stream:
    """Merge multi-segment traces and fill gaps (see module docstring, step a).

    Every output trace is unmasked, plain ``float64`` data -- never dropped,
    regardless of how long a gap is.
    """
    merged = stream.copy()
    merged.merge(method=0)  # masked arrays over any gap; never fills

    out = Stream()
    for tr in merged:
        if not np.ma.is_masked(tr.data):
            tr.data = np.asarray(tr.data, dtype=np.float64)
            out += tr
            continue

        mask = np.ma.getmaskarray(tr.data)
        run_lengths = _masked_run_lengths(mask)
        max_run_s = (max(run_lengths) / tr.stats.sampling_rate) if run_lengths else 0.0

        if mask.all():
            logger.warning(
                "event %s: %s has no valid samples at all in this chunk -- zero-filling",
                event_id,
                tr.id,
            )
            tr.data = np.zeros(tr.data.shape, dtype=np.float64)
        elif max_run_s <= max_gap_fill_s:
            tr.data = _linear_interpolate_masked(tr.data)
        else:
            logger.warning(
                "event %s: %s has a %.3fs gap (> phasenet.max_gap_fill_s=%.3fs) -- "
                "zero-filling the remaining masked samples",
                event_id,
                tr.id,
                max_run_s,
                max_gap_fill_s,
            )
            tr.data = np.ma.filled(tr.data.astype(np.float64), 0.0)
        out += tr
    return out


# ---------------------------------------------------------------------------
# b-d. demean/detrend, taper, highpass
# ---------------------------------------------------------------------------


def _demean_detrend_taper_filter(stream: Stream, taper_pct: float, highpass_hz: float) -> None:
    for tr in stream:
        tr.detrend("demean")
        tr.detrend("linear")
    stream.taper(max_percentage=taper_pct, type="hann")
    if highpass_hz > 0:
        stream.filter("highpass", freq=highpass_hz, zerophase=True)


# ---------------------------------------------------------------------------
# e. anti-alias decimation
# ---------------------------------------------------------------------------


def _decimate_to_target(stream: Stream, event_id: str, target_rate_hz: float) -> None:
    for tr in stream:
        sr = tr.stats.sampling_rate
        if abs(sr - target_rate_hz) < 1e-9:
            continue
        ratio = sr / target_rate_hz
        rounded = round(ratio)
        if rounded < 1 or abs(ratio - rounded) > _RATIO_TOL:
            raise ValueError(
                f"event {event_id}: {tr.id} sampling rate {sr} Hz is not an integer multiple "
                f"of phasenet.target_rate_hz={target_rate_hz} Hz (ratio={ratio!r})"
            )
        factor = int(rounded)
        if factor == 1:
            continue
        # Same filter and corner as Trace.decimate(no_filter=False), but
        # forward-backward so onsets are not delayed (module docstring, e).
        freq = sr * 0.5 / factor
        data = lowpass_cheby_2(tr.data, freq, sr, maxorder=12)
        tr.data = np.ascontiguousarray(lowpass_cheby_2(data[::-1], freq, sr, maxorder=12)[::-1])
        tr.decimate(factor=factor, no_filter=True)


# ---------------------------------------------------------------------------
# f. location-code collation
# ---------------------------------------------------------------------------


def _collate_location_codes(stream: Stream, event_id: str) -> Stream:
    groups: dict[tuple[str, str], dict[str, list[Trace]]] = defaultdict(lambda: defaultdict(list))
    for tr in stream:
        groups[(tr.stats.network, tr.stats.station)][tr.stats.location].append(tr)

    out = Stream()
    for (net, sta), by_loc in groups.items():
        if len(by_loc) == 1:
            (only_traces,) = by_loc.values()
            out += Stream(only_traces)
            continue

        # Most complete (most channels) wins; ties broken alphabetically by
        # location code (sorted() iteration order + max()'s "first max wins").
        best_loc = max(sorted(by_loc), key=lambda loc: len(by_loc[loc]))
        dropped = {loc: len(traces) for loc, traces in by_loc.items() if loc != best_loc}
        logger.info(
            "event %s: %s.%s has multiple location codes %s -- keeping %r "
            "(%d channel(s)), dropping %s",
            event_id,
            net,
            sta,
            {loc: len(traces) for loc, traces in sorted(by_loc.items())},
            best_loc,
            len(by_loc[best_loc]),
            dropped,
        )
        out += Stream(by_loc[best_loc])
    return out


# ---------------------------------------------------------------------------
# h. ensure exactly 3 components per station
# ---------------------------------------------------------------------------

_ORIENTED_HORIZONTALS = {"N", "E"}
_UNORIENTED_HORIZONTALS = {"1", "2"}


def _target_component_set(present_chars: set) -> set:
    if present_chars & _ORIENTED_HORIZONTALS:
        return {"Z"} | _ORIENTED_HORIZONTALS
    if present_chars & _UNORIENTED_HORIZONTALS:
        return {"Z"} | _UNORIENTED_HORIZONTALS
    # Z-only (or nothing recognizable): default to the unoriented
    # convention -- never assert a known orientation that wasn't observed.
    return {"Z"} | _UNORIENTED_HORIZONTALS


def _ensure_three_components(stream: Stream, event_id: str) -> tuple[Stream, list[dict]]:
    groups: dict[str, list[Trace]] = defaultdict(list)
    for tr in stream:
        groups[tr.id[:-1]].append(tr)  # NET.STA.LOC.BandInst, PhaseNet's own key

    out = Stream()
    report_rows: list[dict] = []
    for seed_id_base, traces in groups.items():
        present = {tr.stats.channel[-1]: tr for tr in traces}
        originally_present = sorted(present)
        target = _target_component_set(set(present))
        missing = sorted(target - set(present))

        for missing_char in missing:
            ref = next(iter(present.values()))
            synth = ref.copy()
            synth.stats.channel = ref.stats.channel[:-1] + missing_char
            synth.data = np.zeros(np.asarray(ref.data).shape, dtype=np.float64)
            present[missing_char] = synth
            logger.warning(
                "event %s: %s missing component %r (had %s) -- zero-filling a synthetic trace",
                event_id,
                seed_id_base,
                missing_char,
                originally_present,
            )

        if missing:
            report_rows.append(
                {
                    "event_id": event_id,
                    "seed_id_base": seed_id_base,
                    "n_components_found": len(traces),
                    "zero_filled_components": ",".join(missing),
                }
            )

        for char in sorted(target):
            out += present[char]

    return out, report_rows


def _write_sidecar(paths: RunPaths, event_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    sidecar = paths.chunks_pn_dir / event_id / _SIDECAR_NAME
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(rows))


# ---------------------------------------------------------------------------
# prepare_chunk
# ---------------------------------------------------------------------------


def prepare_chunk(
    event_id: str,
    chunk_path: str | Path,
    station_table: pd.DataFrame,
    cfg: Config,
) -> Path:
    """Turn one raw chunk into a PhaseNet-ready ``chunks_pn/<event_id>/<event_id>.mseed``.

    See the module docstring for the full a-i pipeline. Top-level and
    picklable (:func:`openmeq.parallel.pmap`'s requirement), and touches only
    ``chunks_pn/<event_id>/`` on disk, so it is safe to call from a
    ``pmap`` worker process alongside every other event's call.

    Parameters
    ----------
    event_id : str
    chunk_path : str or pathlib.Path
        The raw chunk to prepare, normally ``paths.chunk_file(event_id)``
        (``chunks/<event_id>/<event_id>.mseed``, M3.1's output format).
    station_table : pandas.DataFrame
        ``openmeq.stations.station_table(cfg.stations)`` columns
        (currently used only for a coverage sanity check -- a station in
        the chunk that :func:`openmeq.stations.apply_channel_map` maps but
        that isn't in ``station_table`` at all is logged, not an error).
    cfg : openmeq.config.Config

    Returns
    -------
    pathlib.Path
        ``chunks_pn/<event_id>/<event_id>.mseed`` (``paths.chunk_pn_file(event_id)``).
    """
    chunk_path = Path(chunk_path)
    paths = RunPaths(cfg)

    stream = read(str(chunk_path))

    stream = _merge_and_fill_gaps(stream, event_id, cfg.phasenet.max_gap_fill_s)
    _demean_detrend_taper_filter(stream, cfg.phasenet.taper_pct, cfg.phasenet.highpass_hz)
    _decimate_to_target(stream, event_id, cfg.phasenet.target_rate_hz)
    stream = _collate_location_codes(stream, event_id)
    stream = apply_channel_map(stream, cfg.stations)
    stream, report_rows = _ensure_three_components(stream, event_id)

    if station_table is not None and len(station_table) > 0:
        known = {f"{r.network}.{r.station}" for r in station_table.itertuples(index=False)}
        seen = {f"{tr.stats.network}.{tr.stats.station}" for tr in stream}
        unknown = seen - known
        if unknown:
            logger.info(
                "event %s: station(s) %s not present in station_table (proceeding anyway)",
                event_id,
                sorted(unknown),
            )

    out_path = paths.chunk_pn_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Every trace is float64 by this point (detrend/taper/filter/decimate,
    # plus any zero-filled synthetic trace); write it as such explicitly --
    # otherwise obspy falls back to each trace's stale
    # ``stats.mseed.encoding`` (inherited from the original int32 STEIM2
    # input) and warns that it doesn't match the now-float data.
    for tr in stream:
        tr.data = tr.data.astype(np.float64, copy=False)
    stream.write(str(out_path), format="MSEED", encoding="FLOAT64")

    _write_sidecar(paths, event_id, report_rows)

    return out_path


# ---------------------------------------------------------------------------
# prepare_all: orchestration (pmap over events, skip-unchanged, data_list.csv)
# ---------------------------------------------------------------------------


def _prepare_chunk_worker(item: tuple[str, str, list[dict], Config]) -> str:
    event_id, chunk_path_str, station_records, cfg = item
    station_df = pd.DataFrame(station_records, columns=STATION_TABLE_COLUMNS)
    out_path = prepare_chunk(event_id, chunk_path_str, station_df, cfg)
    return str(out_path)


def _collect_prep_report(paths: RunPaths, event_ids: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for event_id in event_ids:
        sidecar = paths.chunks_pn_dir / event_id / _SIDECAR_NAME
        if sidecar.is_file():
            rows.extend(json.loads(sidecar.read_text()))
    return pd.DataFrame(rows, columns=PREP_REPORT_COLUMNS)


def write_prep_report_csv(paths: RunPaths, event_ids: list[str]) -> Path:
    """Aggregate every event's zero-fill sidecar into ``chunks_pn/prep_report.csv``.

    Only events that had at least one zero-filled component contribute a
    row (see the module docstring, step h). Safe to call repeatedly --
    always rebuilt from the current sidecars, never appended to.
    """
    df = _collect_prep_report(paths, event_ids)
    out_path = paths.prep_report_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, columns=PREP_REPORT_COLUMNS)
    return out_path


def prepare_all(
    cfg: Config,
    paths: RunPaths,
    event_ids: list[str],
    station_table: pd.DataFrame | None = None,
) -> Path:
    """Prepare every event (pmap, skip-unchanged), then write ``data_list.csv``.

    PLAN.md §4's "PhaseNet prep: event" row: one :func:`openmeq.parallel.pmap`
    unit of work per event id, skipping any event already done for the
    current ``cfg.phasenet``/``cfg.stations``/chunk-file fingerprint
    (:class:`openmeq.manifest.StepManifest`, keyed by ``event_id``).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    event_ids : list of str
        Every event to prepare (e.g. from ``chunks/completeness.csv`` or a
        directory listing of ``chunks/``).
    station_table : pandas.DataFrame, optional
        Defaults to ``openmeq.stations.station_table(cfg.stations)`` if not
        given.

    Returns
    -------
    pathlib.Path
        ``paths.data_list_csv``, written for every ``event_ids`` regardless
        of which ones were freshly (re)processed this call.
    """
    if station_table is None:
        station_table = compute_station_table(cfg.stations)

    manifest = StepManifest(paths, "chunks_pn")

    def _fp(event_id: str) -> str:
        return fingerprint(cfg.phasenet, cfg.stations, paths.chunk_file(event_id), _PREP_VERSION)

    pending = [event_id for event_id in event_ids if not manifest.is_done(event_id, _fp(event_id))]

    if pending:
        station_records = station_table.to_dict("records")
        items = [
            (event_id, str(paths.chunk_file(event_id)), station_records, cfg)
            for event_id in pending
        ]
        pmap(_prepare_chunk_worker, items, cfg.parallel, desc="phasenet-prep")
        for event_id in pending:
            manifest.mark_done(event_id, _fp(event_id))

    write_prep_report_csv(paths, event_ids)
    return write_data_list(paths, event_ids)


__all__ = [
    "PREP_REPORT_COLUMNS",
    "prepare_chunk",
    "prepare_all",
    "write_prep_report_csv",
]
