"""PhaseNet ``picks.csv`` -> ``pick/catalog_picks.xml`` conversion (M4.4, ``OpenMEQ-1m7.4``).

Final task of the M4 epic (ARCHITECTURE.md §3/§5). Builds one
``obspy.core.event.Event`` per chunk (``openmeq.catalog.make_event``), with
one automatic :class:`~obspy.core.event.origin.Pick` per surviving
``(station, phase)`` in :data:`openmeq.pick.run_phasenet.PICKS_CSV_COLUMNS`'s
``picks.csv`` -- **no origins** (ARCHITECTURE.md §3: "events with automatic
picks, no origins").

Grouping picks by event
------------------------
``picks.csv``'s ``file_name`` column is ``<event_id>/<event_id>.mseed``
(M4.2's ``chunks_pn/data_list.csv`` convention, M4.3's own docstring) -- the
event id is that path's first component, never re-derived from the row's
own ``begin_time``.

Deduplication: highest probability wins
-----------------------------------------
A station can carry several ``P`` or several ``S`` rows for one event
(``docs/phasenet_interface.md`` §5's own confirmed example: PhaseNet can
double-fire a spurious extra ``P`` near a strong ``S`` onset) -- this is a
**project-level decision not specified anywhere else**, so it is documented
here prominently: at most one :class:`Pick` survives per
``(event_id, station_id, phase_type)``, the row with the highest
``phase_score``, ties broken by the earliest ``phase_time`` (deterministic,
stable) then row order. This is the one gap ARCHITECTURE.md §5's QuakeML
conventions section left genuinely open (the channel-attribution rule below
was, on inspection, already fully specified there) -- flagged here and in
the same-commit ARCHITECTURE.md addition.

Channel attribution (P -> Z, S -> first horizontal present)
----------------------------------------------------------------
ARCHITECTURE.md §5 already specifies this rule in full: PhaseNet picks are
per-station (instrument), not per-channel, so a pick's
:attr:`~obspy.core.event.origin.Pick.waveform_id` must name one concrete,
real channel. A P pick goes on the station's ``Z`` channel; an S pick goes
on the first horizontal *present*, in preference order ``N``, ``E``, ``1``,
``2`` (:data:`S_PICK_CHANNEL_PREFERENCE`) -- a station is either fully
oriented (``Z``/``N``/``E``, ``openmeq.stations.apply_channel_map`` rotated
it) or fully unoriented (``Z``/``1``/``2``), never a mix
(``openmeq.pick.prep``'s ``_target_component_set``), so in practice only one
of the two horizontal pairs is ever actually present for a given station;
the four-entry preference order exists so this function never has to know
which convention a given station uses.

``picks.csv``'s own ``station_id`` (``NET.STA.LOC.BandInst``, no component
character -- ``docs/phasenet_interface.md`` §5) is resolved to one concrete
:class:`~obspy.core.event.base.WaveformStreamID` by reading
``chunks_pn/<event_id>/<event_id>.mseed``'s actual trace headers for that
prefix (:meth:`~obspy.core.stream.Stream.select`-style match on
``trace.id[:-1] == station_id``) and picking the Z/first-horizontal trace
among them, per the rule above.

**Why ``trace.stats.channel`` (not ``trace.stats.openmeq_original_channel``)
is read here**: ``trace.stats.openmeq_original_channel``
(``openmeq.stations.apply_channel_map``'s own record of a trace's
pre-mapping channel code, ARCHITECTURE.md §6) is an in-process-only
``obspy.core.trace.Stats`` attribute -- MiniSEED's fixed SEED header has no
slot for it, so it does **not** survive ``chunks_pn/<event_id>/<event_id>.mseed``
being written and re-read (M4.2's ``prepare_chunk`` writes it once; this
module reads it back fresh in a later step/process, same as every other
QuakeML-chain handoff, ARCHITECTURE.md §5's "every step reads the previous
step's actual on-disk output"). This module therefore uses whatever channel
code is **actually on disk** in ``chunks_pn/``'s own trace headers as the
"original channel code" ARCHITECTURE.md §5 asks for -- correct for every
mapping this project's ``stations.channel_map`` convention actually uses (an
already-standard ``Z``/``N``/``E`` channel is never remapped at all, and
every configured borehole ``channel_map`` entry in ``config/example.yaml``
maps its own last character to itself, e.g. ``DPZ -> Z``), and it is
guaranteed (by construction) to always be "a real channel present in the
corresponding ``chunks_pn`` mseed" -- exactly this task's own acceptance
criterion. The one case this does not perfectly reproduce is a genuinely
*rotated* horizontal pair (``cfg.stations.orientations`` configured, an
unoriented ``1``/``2`` pair relabelled ``N``/``E`` by
``apply_channel_map``'s own rotation step): the waveform_id then names the
post-rotation ``N``/``E`` code rather than the true physically-recorded
``1``/``2`` SEED channel. Flagged as a follow-up (persist
``openmeq_original_channel`` via a small per-event sidecar in
``chunks_pn/``, mirroring M4.2's own zero-fill-report sidecar pattern) for
whoever next needs that exact distinction -- out of this task's scope, and
not exercised by the synthetic fixture (``config/example.yaml`` ships
``orientations: {}``).

Probability -> ``extra``
--------------------------
``phase_score`` is stored via ``openmeq.catalog.set_extra(pick,
"probability", phase_score)`` (ARCHITECTURE.md §5), so it survives the
QuakeML round-trip via ``openmeq.catalog.get_extra``.

Rejected picks (``rejected=``, extended QAQC only)
----------------------------------------------------
:func:`build_catalog` optionally appends one record per ``picks.csv`` row
it *drops* to a caller-supplied ``rejected`` list (OpenMEQ-02o.4's
``pick_zoom_<event_id>.png`` ``show_rejected`` overlay) -- a pure side
channel: the catalogue returned is identical whether or not ``rejected``
is given. Two reasons exist here: ``"duplicate"`` (lost the
highest-probability dedupe above) and ``"unresolved channel"``. There is no
probability-threshold reason: ``cfg.phasenet.min_p_prob``/``min_s_prob``
are applied inside PhaseNet's own ``predict.py``
(``openmeq.pick.run_phasenet``'s ``--min_p_prob``/``--min_s_prob``), so a
below-threshold candidate never reaches ``picks.csv`` at all (the zoom
figure recovers those from ``result.h5`` instead,
:mod:`openmeq.pick.extended_diagnostics`).

No origins
-----------
Per ARCHITECTURE.md §3, ``pick/catalog_picks.xml`` holds "events with
automatic picks, no origins" -- :func:`build_catalog` never creates an
``Origin`` or sets ``event.preferred_origin_id``; that is M6's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.core.event import Catalog, Pick, WaveformStreamID

from openmeq.catalog import make_event, set_extra, write_catalog
from openmeq.log import get_logger
from openmeq.pick.run_phasenet import PICKS_CSV_COLUMNS

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("pick.to_quakeml")

#: S-pick channel attribution preference order (project-level decision, see
#: module docstring; matches ARCHITECTURE.md §5's "Channel for PhaseNet
#: picks" convention). P picks always attribute to the ``Z`` channel.
S_PICK_CHANNEL_PREFERENCE = ["N", "E", "1", "2"]

#: ``method_id`` every PhaseNet-sourced pick carries (ARCHITECTURE.md §5).
PHASENET_METHOD_ID = "smi:openmeq/phasenet"


def _event_id_from_file_name(file_name: str) -> str:
    """``"<event_id>/<event_id>.mseed"`` (M4.2's ``data_list.csv`` convention) -> ``event_id``."""
    return Path(str(file_name)).parts[0]


def _dedupe_best_per_station_phase(df: pd.DataFrame) -> pd.DataFrame:
    """At most one row per ``(station_id, phase_type)``: the highest ``phase_score``.

    Ties broken by earliest ``phase_time`` then original row order (both via
    a stable sort before ``drop_duplicates``'s own stable "first wins").
    """
    ordered = df.sort_values(
        by=["phase_score", "phase_time"],
        ascending=[False, True],
        kind="mergesort",  # stable, so equal-score/time ties keep original row order
    )
    return ordered.drop_duplicates(subset=["station_id", "phase_type"], keep="first")


def _rejected_record(event_id: str, row: pd.Series, reason: str) -> dict:
    """One ``build_catalog(rejected=...)`` entry (see module docstring)."""
    return {
        "event_id": event_id,
        "station_id": str(row["station_id"]),
        "phase_type": str(row["phase_type"]),
        "time": UTCDateTime(row["phase_time"]),
        "probability": float(row["phase_score"]),
        "reason": reason,
    }


def _read_chunk_pn_stream(paths: RunPaths, event_id: str) -> Stream:
    chunk_path = paths.chunk_pn_file(event_id)
    if not chunk_path.is_file():
        logger.warning(
            "event %s: %s not found -- cannot resolve any pick channel for this event",
            event_id,
            chunk_path,
        )
        return Stream()
    try:
        return read(str(chunk_path))
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole conversion
        logger.warning("event %s: could not read %s: %s", event_id, chunk_path, exc)
        return Stream()


def _resolve_waveform_id(
    stream: Stream, station_id: str, phase_type: str
) -> WaveformStreamID | None:
    """Resolve ``picks.csv``'s station-level ``station_id`` to one concrete channel.

    Parameters
    ----------
    stream : obspy.Stream
        ``chunks_pn/<event_id>/<event_id>.mseed``'s traces.
    station_id : str
        ``NET.STA.LOC.BandInst`` (no component character,
        ``docs/phasenet_interface.md`` §5).
    phase_type : str
        ``"P"`` or ``"S"``.

    Returns
    -------
    obspy.core.event.base.WaveformStreamID or None
        ``None`` if no matching trace/preferred component was found (logged
        by the caller).
    """
    candidates = {tr.stats.channel: tr for tr in stream if tr.id[:-1] == station_id}
    if not candidates:
        return None

    if phase_type == "P":
        preference = ["Z"]
    elif phase_type == "S":
        preference = S_PICK_CHANNEL_PREFERENCE
    else:  # pragma: no cover - picks.csv's phase_type is always P/S (docs/phasenet_interface.md §5)
        raise ValueError(f"unexpected phase_type {phase_type!r} (expected 'P' or 'S')")

    for last_char in preference:
        for channel_code, tr in candidates.items():
            if channel_code.endswith(last_char):
                return WaveformStreamID(
                    network_code=tr.stats.network,
                    station_code=tr.stats.station,
                    location_code=tr.stats.location,
                    channel_code=channel_code,
                )
    return None


def _build_pick(row: pd.Series, waveform_id: WaveformStreamID, pick_uncertainty_s: float) -> Pick:
    pick = Pick(
        time=UTCDateTime(row["phase_time"]),
        waveform_id=waveform_id,
        phase_hint=row["phase_type"],
        method_id=PHASENET_METHOD_ID,
        evaluation_mode="automatic",
    )
    pick.time_errors.uncertainty = pick_uncertainty_s
    set_extra(pick, "probability", float(row["phase_score"]))
    return pick


def build_catalog(
    picks_csv: str | Path,
    paths: RunPaths,
    cfg: Config,
    station_table: pd.DataFrame | None = None,
    *,
    rejected: list[dict] | None = None,
) -> Catalog:
    """Convert PhaseNet's ``picks.csv`` into one ``Event`` per chunk, with picks only.

    Parameters
    ----------
    picks_csv : str or pathlib.Path
        ``pick/phasenet_raw/picks.csv`` (:func:`openmeq.pick.run_phasenet.run_phasenet`'s
        output), with at least :data:`~openmeq.pick.run_phasenet.PICKS_CSV_COLUMNS`.
    paths : openmeq.paths.RunPaths
        ``paths.chunk_pn_file(event_id)`` locates each event's
        ``chunks_pn/`` mseed for channel resolution (see module docstring).
    cfg : openmeq.config.Config
        ``cfg.phasenet.pick_uncertainty_s`` becomes every
        ``Pick.time_errors.uncertainty`` (PhaseNet emits no timing
        uncertainty, only a probability -- ARCHITECTURE.md §5). Not part of
        the beads issue's abbreviated signature sketch
        (``build_catalog(picks_csv, paths, station_table)``) -- added
        because the config value it reads did not exist to be threaded
        through any other way; see this task's final report for the full
        note.
    station_table : pandas.DataFrame, optional
        ``openmeq.stations.station_table(cfg.stations)`` columns. Used only
        for an informational coverage check (a ``picks.csv`` station not in
        ``station_table`` is logged, never an error) -- mirrors
        :func:`openmeq.pick.prep.prepare_chunk`'s identical, optional use of
        it. Defaults to skipping the check if not given.
    rejected : list of dict, keyword-only, optional
        If given, one ``{"event_id", "station_id", "phase_type", "time",
        "probability", "reason"}`` dict is appended per dropped
        ``picks.csv`` row (``reason`` ``"duplicate"`` or ``"unresolved
        channel"``, see module docstring). Never changes the returned
        catalogue.

    Returns
    -------
    obspy.Catalog
        One ``Event`` (``openmeq.catalog.make_event(event_id)``) per
        distinct event id in ``picks_csv``'s ``file_name`` column, each
        holding only :class:`~obspy.core.event.origin.Pick` objects (no
        origins, ARCHITECTURE.md §3) -- an event with zero surviving picks
        never appears (every event in the returned catalogue has at least
        one pick).
    """
    picks_csv = Path(picks_csv)
    df = pd.read_csv(picks_csv)
    missing = [c for c in PICKS_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{picks_csv} is missing expected column(s) {missing}")

    if station_table is not None and len(station_table) > 0:
        known = {f"{r.network}.{r.station}" for r in station_table.itertuples(index=False)}
    else:
        known = None

    df = df.copy()
    df["event_id"] = df["file_name"].map(_event_id_from_file_name)

    catalog = Catalog()
    n_picks_total = 0
    for event_id, group in df.groupby("event_id", sort=True):
        stream = _read_chunk_pn_stream(paths, event_id)
        best = _dedupe_best_per_station_phase(group)
        if rejected is not None:
            for _, row in group.drop(index=best.index).iterrows():
                rejected.append(_rejected_record(event_id, row, "duplicate"))

        event = make_event(event_id)
        for _, row in best.iterrows():
            station_id = str(row["station_id"])
            phase_type = str(row["phase_type"])

            if known is not None:
                sta_key = ".".join(station_id.split(".")[:2])
                if sta_key not in known:
                    logger.info(
                        "event %s: station %s (from picks.csv) not present in station_table "
                        "(proceeding anyway)",
                        event_id,
                        sta_key,
                    )

            waveform_id = _resolve_waveform_id(stream, station_id, phase_type)
            if waveform_id is None:
                logger.warning(
                    "event %s: could not resolve a %s-channel for station_id=%r in %s "
                    "-- dropping this pick",
                    event_id,
                    "Z" if phase_type == "P" else "/".join(S_PICK_CHANNEL_PREFERENCE),
                    station_id,
                    paths.chunk_pn_file(event_id),
                )
                if rejected is not None:
                    rejected.append(_rejected_record(event_id, row, "unresolved channel"))
                continue

            pick = _build_pick(row, waveform_id, cfg.phasenet.pick_uncertainty_s)
            event.picks.append(pick)
            n_picks_total += 1

        if event.picks:
            catalog.events.append(event)
        else:
            logger.warning(
                "event %s: every candidate pick's channel failed to resolve -- "
                "no Event written for it",
                event_id,
            )

    logger.info(
        "build_catalog: %d event(s), %d pick(s) total from %s",
        len(catalog.events),
        n_picks_total,
        picks_csv,
    )
    return catalog


def write_catalog_picks(catalog: Catalog, paths: RunPaths) -> Path:
    """Write ``catalog`` to ``pick/catalog_picks.xml`` (:func:`openmeq.catalog.write_catalog`)."""
    write_catalog(catalog, paths.catalog_picks_xml)
    return paths.catalog_picks_xml


__all__ = [
    "PHASENET_METHOD_ID",
    "S_PICK_CHANNEL_PREFERENCE",
    "build_catalog",
    "write_catalog_picks",
]
