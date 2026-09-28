"""Event-level accept/reject decisions and the M6 output artifacts (M6.4 / OpenMEQ-xdf.4).

This module owns every *final* per-event accept/reject decision
(ARCHITECTURE.md §5: never delete, always flag) and the row shapes for the
three headline M6 output artifacts (ARCHITECTURE.md §3):
``locate/catalog_auto.xml``, ``locate/summary.csv`` and
``locate/removal_log.csv``. :func:`finalize_event` is the per-event entry
point; :func:`write_summary_csv`/:func:`write_removal_log_csv` are the two
CSV writers. ``openmeq.locate.locate_step.run_locate`` (M6.4's other half)
is the orchestration that calls all of pass A, pass B and this module per
event and assembles the three files.

Design decisions
-----------------
**1. The event-level rejection order is fixed and "first failing reason
wins".** Per the issue's own "Work to do" §1: ``rms > max_rms_s`` ->
``n_phases_total < min_phases`` -> ``n_p < min_p_phases`` (re-checked here,
even though pass A already checked it once, because pass B's own residual
filtering -- ``openmeq.locate.pass_b`` -- can drop *more* P picks after pass
A already accepted them) -> ``n_s < min_s_phases`` -> ``gap_deg >
max_gap_deg`` -> ``err_h_km > max_err_h_km`` -> ``err_z_km > max_err_z_km``.
This ordering is a judgement call, not specified by ``docs/PLAN.md`` --
flagged here for reviewer sign-off, so a reviewer can override it later
without re-deriving the whole function. The ``reason`` string written to
``summary.csv``/used for ``openmeq.catalog.reject_event`` is exactly the
matching ``openmeq.config.EventsFilterConfig`` attribute name
(``"max_rms_s"``, ``"min_phases"``, ``"min_p_phases"``, ``"min_s_phases"``,
``"max_gap_deg"``, ``"max_err_h_km"``, ``"max_err_z_km"``) -- the same
convention ``openmeq.locate.pass_a.run_pass_a`` already uses for its own
``"min_p_phases"`` rejection reason, kept consistent here rather than
inventing a second vocabulary.

**2. A criterion whose underlying value is missing is never evaluated
(never counted as a failure).** ``rms``/``gap``/``err_h_km``/``err_z_km``
all come from the pass-B ``Origin``'s ``quality``/``origin_uncertainty``/
``depth_errors`` sub-objects, any of which can be ``None`` -- either because
a hand-built test ``Origin`` doesn't set every field, or (see decision #3
below) because pass B never produced an origin at all. Rather than treat a
missing value as an automatic pass *or* an automatic fail, each of those
four checks is skipped entirely when its value is ``None``, falling through
to the next criterion in the fixed order. ``n_phases_total``/``n_p``/``n_s``
are never missing (they come directly from
``len(pass_b_result.accepted_p)``/``len(pass_b_result.accepted_s)``, always
computable, even ``0``), so they are always evaluated.

**3. A total pass-B NLLoc failure (``pass_b_result.origin is None``) is
checked before the fixed-order criteria, not folded into them.** The
issue's own fixed-order list implicitly assumes pass B produced an origin
(it starts with ``rms > max_rms_s``, which needs one). ``PassBResult.origin
is None`` only when *every* pass-B relocation attempt raised
``NLLocFailed`` (``openmeq.locate.pass_b``'s own design decision #5) --
there is no ``Origin`` to read ``rms``/``lat``/``lon``/... from at all, so
this function cannot fall through to decision #2's per-field skip logic for
*every* field and still end up with something to write.  In the overwhelming
majority of configurations this case is already caught by decision #2 alone
(``n_p == n_s == 0`` when ``origin is None``, trivially below any sane
``min_phases``/``min_p_phases``), but a project could configure
``min_phases: 0`` / ``min_p_phases: 0`` -- an explicit check here avoids the
alternative of ever calling ``event.preferred_origin_id = None.resource_id``
or dividing ``None`` by 1000. The rejection reason in this case is
``pass_b_result.reason`` (``f"nlloc_failed:{message}"``), not a criterion
name, mirroring how pass A's own ``"nlloc_failed:..."`` reason (rather than
``"min_p_phases"``) is used when *its* failure was an NLLoc crash rather
than a residual-filtering shortfall.

**4. A rejected event still gets a preferred `Origin`, even when neither
pass ever produced one.** ``openmeq.catalog.accepted``/``reject_event`` both
key off ``event.preferred_origin()`` (ARCHITECTURE.md §5); an event with no
preferred origin at all is treated as *not* rejected by ``accepted()``'s own
contract. So a total pass-A-or-pass-B NLLoc failure -- which otherwise would
leave ``event.origins`` empty -- gets an empty placeholder ``Origin()``
(no time/lat/lon/quality set) appended and made preferred purely so the
rejection flag can attach to *something*. Every numeric ``summary.csv``
field for such a row is null (every field but ``event_id``/``status``/
``reason``), which is directly visible in the file rather than hidden by
a silently-omitted row -- ``docs/ARCHITECTURE.md`` §3, "every event,
rejected ones flagged, never dropped".

**5. `OriginQuality.minimum_distance`/`maximum_distance`/`median_distance`
are converted from degrees to km, in place, on the final origin.** M6.1's
own note (``openmeq.locate.nlloc_runner.locate_event``) already converts
every `Arrival.distance` from obspy's QuakeML-standard degrees to km, but
obspy's ``NLLOC_HYP`` reader leaves ``OriginQuality``'s own three distance
summary fields in degrees (``kilometer2degrees`` is baked into that reader,
confirmed directly in ``obspy/io/nlloc/core.py``). Left unconverted, the
same origin would report *arrival* distances in km right next to *origin
quality* distances in degrees -- silently inconsistent within the same
`Origin` object. This module converts all three fields to km (via
``obspy.geodetics.kilometers2degrees``'s inverse,
``obspy.geodetics.degrees2kilometers``) on the origin actually written into
``catalog_auto.xml``, for consistency with ARCHITECTURE.md §1's project-wide
km convention for the location step and with `Arrival.distance`. There is no
``summary.csv`` column these three fields feed (that file's own distance-free
column list is unchanged), so this only affects `OriginQuality` as written
into the QuakeML catalogue; ``docs/ARCHITECTURE.md`` §5 is updated in this
commit to record the decision next to the ``Arrival.distance`` note it
extends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from obspy.core.event import Origin
from obspy.geodetics import degrees2kilometers

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import iso_z, reject_event

if TYPE_CHECKING:
    from obspy.core.event import Event

    from openmeq.config import Config, EventsFilterConfig
    from openmeq.locate.pass_a import PassAResult, RemovalRecord
    from openmeq.locate.pass_b import PassBResult

#: ``locate/summary.csv``'s exact column order (ARCHITECTURE.md §3).
SUMMARY_COLUMNS = [
    "event_id",
    "time",
    "lat",
    "lon",
    "depth_km",
    "rms",
    "gap",
    "n_p",
    "n_s",
    "err_h_km",
    "err_z_km",
    "status",
    "reason",
]

#: ``locate/removal_log.csv``'s exact column order (ARCHITECTURE.md §3).
#: ``RemovalRecord.pass_label`` maps to this file's ``pass`` column (``pass``
#: itself is a Python keyword, see that class's own docstring).
REMOVAL_LOG_COLUMNS = ["event_id", "pass", "iteration", "seed_id", "phase", "residual_s", "reason"]

#: The three ``OriginQuality`` distance fields obspy's ``NLLOC_HYP`` reader
#: leaves in degrees (decision #5 above) -- converted to km in place.
_QUALITY_DISTANCE_FIELDS = ("minimum_distance", "maximum_distance", "median_distance")

_PASS_A_METHOD_ID = "smi:openmeq/nlloc/passA"
_PASS_B_METHOD_ID = "smi:openmeq/nlloc/passB"

#: The step name passed to ``openmeq.catalog.reject_event`` for every
#: event-level rejection this module makes -- distinct from
#: ``"locate/passA"``/``"locate/passB"``, which ``openmeq.locate.pass_a
#: .iterate_until_stable`` uses for *pick*-level rejections.
_REJECT_STEP = "locate"


@dataclass(frozen=True)
class SummaryRow:
    """One row of ``locate/summary.csv`` (ARCHITECTURE.md §3).

    Every field but ``event_id``/``n_p``/``n_s``/``status``/``reason`` is
    ``None`` when the underlying `Origin` (or a specific value on it) is
    unavailable -- written as an empty CSV cell (:func:`write_summary_csv`).
    """

    event_id: str
    time: str | None
    lat: float | None
    lon: float | None
    depth_km: float | None
    rms: float | None
    gap: float | None
    n_p: int
    n_s: int
    err_h_km: float | None
    err_z_km: float | None
    status: Literal["accepted", "rejected"]
    reason: str


@dataclass
class FinalizedEvent:
    """Result of :func:`finalize_event`.

    Attributes
    ----------
    event : obspy.core.event.Event
        The same `Event` passed in, mutated in place: the final `Origin`
        (pass A's or pass B's, or a placeholder, see the module docstring)
        appended to ``event.origins`` and made ``event.preferred_origin_id``,
        and (if rejected) flagged via ``openmeq.catalog.reject_event``.
    summary : SummaryRow
        The ``locate/summary.csv`` row for this event.
    """

    event: Event
    summary: SummaryRow


@dataclass(frozen=True)
class _OriginMetrics:
    """Values read off a (possibly ``None``) `Origin`, once, shared between
    :func:`_evaluate_criteria` and :func:`_summary_row` so the two never
    drift out of sync about how a field is derived."""

    time_iso: str | None
    lat: float | None
    lon: float | None
    depth_km: float | None
    rms: float | None
    gap: float | None
    err_h_km: float | None
    err_z_km: float | None


def _origin_metrics(origin: Origin | None) -> _OriginMetrics:
    if origin is None:
        return _OriginMetrics(None, None, None, None, None, None, None, None)

    quality = origin.quality
    ou = origin.origin_uncertainty
    de = origin.depth_errors

    err_h_km = None
    if ou is not None and ou.horizontal_uncertainty is not None:
        err_h_km = ou.horizontal_uncertainty / 1000.0  # OriginUncertainty is metres.

    err_z_km = None
    if de is not None and de.uncertainty is not None:
        err_z_km = de.uncertainty / 1000.0  # depth_errors is metres.

    return _OriginMetrics(
        time_iso=iso_z(origin.time) if origin.time is not None else None,
        lat=origin.latitude,
        lon=origin.longitude,
        # QuakeML Origin.depth is metres.
        depth_km=origin.depth / 1000.0 if origin.depth is not None else None,
        rms=quality.standard_error if quality is not None else None,
        gap=quality.azimuthal_gap if quality is not None else None,
        err_h_km=err_h_km,
        err_z_km=err_z_km,
    )


def _evaluate_criteria(
    metrics: _OriginMetrics, n_p: int, n_s: int, events_cfg: EventsFilterConfig
) -> str | None:
    """The fixed-order event-criteria check (module docstring decision #1).

    Returns the failing criterion's name, or ``None`` if every criterion
    passes (or was skipped for a missing value, decision #2).
    """
    if metrics.rms is not None and metrics.rms > events_cfg.max_rms_s:
        return "max_rms_s"
    if (n_p + n_s) < events_cfg.min_phases:
        return "min_phases"
    if n_p < events_cfg.min_p_phases:
        return "min_p_phases"
    if n_s < events_cfg.min_s_phases:
        return "min_s_phases"
    if metrics.gap is not None and metrics.gap > events_cfg.max_gap_deg:
        return "max_gap_deg"
    if metrics.err_h_km is not None and metrics.err_h_km > events_cfg.max_err_h_km:
        return "max_err_h_km"
    if metrics.err_z_km is not None and metrics.err_z_km > events_cfg.max_err_z_km:
        return "max_err_z_km"
    return None


def _summary_row(
    event_id: str,
    metrics: _OriginMetrics,
    n_p: int,
    n_s: int,
    status: Literal["accepted", "rejected"],
    reason: str,
) -> SummaryRow:
    return SummaryRow(
        event_id=event_id,
        time=metrics.time_iso,
        lat=metrics.lat,
        lon=metrics.lon,
        depth_km=metrics.depth_km,
        rms=metrics.rms,
        gap=metrics.gap,
        n_p=n_p,
        n_s=n_s,
        err_h_km=metrics.err_h_km,
        err_z_km=metrics.err_z_km,
        status=status,
        reason=reason,
    )


def _convert_quality_distances_to_km(origin: Origin) -> None:
    """Decision #5: convert ``OriginQuality``'s min/max/median distance
    fields from obspy's degrees to km, in place, on the final origin."""
    quality = origin.quality
    if quality is None:
        return
    for field in _QUALITY_DISTANCE_FIELDS:
        value = getattr(quality, field, None)
        if value is not None:
            setattr(quality, field, degrees2kilometers(value))


def _flag_rejected(event: Event, origin: Origin | None, method_id: str, reason: str) -> Origin:
    """Append ``origin`` (or a placeholder, decision #4) to ``event``, make
    it preferred, and reject the event. Returns the origin actually used."""
    if origin is None:
        origin = Origin()
    origin.method_id = method_id
    _convert_quality_distances_to_km(origin)
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    reject_event(event, reason=reason, step=_REJECT_STEP)
    return origin


def finalize_event(
    event: Event, pass_a: PassAResult, pass_b: PassBResult | None, cfg: Config
) -> FinalizedEvent:
    """Make the final accept/reject decision for one event and build its
    ``summary.csv`` row.

    Parameters
    ----------
    event : obspy.core.event.Event
        Must already carry ``pass_a``'s (and, if run, ``pass_b``'s) pick
        rejections (``openmeq.locate.pass_a.iterate_until_stable`` mutates
        the same `Pick` objects on ``event.picks`` in place) -- this
        function only adds the final `Origin` and the event-level rejection
        flag, never touching picks itself.
    pass_a : openmeq.locate.pass_a.PassAResult
        Pass A's result for this event (M6.2 / OpenMEQ-xdf.2).
    pass_b : openmeq.locate.pass_b.PassBResult or None
        Pass B's result for this event (M6.3 / OpenMEQ-xdf.3), or ``None``
        if the caller never ran pass B (only valid when
        ``pass_a.status == "rejected"``, matching
        ``openmeq.locate.pass_b.run_pass_b``'s own early-return for that
        case -- a caller is free to skip calling pass B at all in that
        situation and pass ``None`` here instead).
    cfg : openmeq.config.Config
        ``cfg.nlloc.filter.events`` holds the event-level thresholds.

    Returns
    -------
    FinalizedEvent
    """
    eid = get_event_id(event)
    events_cfg = cfg.nlloc.filter.events

    if pass_a.status == "rejected":
        n_p = len(pass_a.accepted_picks)
        n_s = 0  # Pass A only ever locates P picks.
        origin = _flag_rejected(
            event, pass_a.origin, _PASS_A_METHOD_ID, pass_a.reason or "rejected"
        )
        # `origin` is either pass_a.origin (real data) or the placeholder
        # decision #4 creates when pass_a.origin is None -- either way its
        # own fields (all null on a placeholder) are the single source of
        # truth for this row, so no separate branch is needed here.
        metrics = _origin_metrics(origin)
        row = _summary_row(eid, metrics, n_p, n_s, "rejected", pass_a.reason or "rejected")
        return FinalizedEvent(event=event, summary=row)

    if pass_b is None:
        raise ValueError(
            f"event {eid!r}: pass_b is required when pass_a.status == 'ok' "
            "(only a rejected pass A may skip pass B)"
        )

    n_p = len(pass_b.accepted_p)
    n_s = len(pass_b.accepted_s)

    if pass_b.origin is None:
        # Decision #3: a total pass-B NLLoc failure, checked ahead of the
        # fixed-order criteria (which assume an origin exists).
        reason = pass_b.reason or "nlloc_failed"
        origin = _flag_rejected(event, None, _PASS_B_METHOD_ID, reason)
        metrics = _origin_metrics(origin)
        row = _summary_row(eid, metrics, n_p, n_s, "rejected", reason)
        return FinalizedEvent(event=event, summary=row)

    metrics = _origin_metrics(pass_b.origin)
    reason = _evaluate_criteria(metrics, n_p, n_s, events_cfg)

    if reason is not None:
        origin = _flag_rejected(event, pass_b.origin, _PASS_B_METHOD_ID, reason)
        metrics = _origin_metrics(origin)
        row = _summary_row(eid, metrics, n_p, n_s, "rejected", reason)
        return FinalizedEvent(event=event, summary=row)

    origin = pass_b.origin
    origin.method_id = _PASS_B_METHOD_ID
    _convert_quality_distances_to_km(origin)
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    row = _summary_row(eid, metrics, n_p, n_s, "accepted", "")
    return FinalizedEvent(event=event, summary=row)


def write_summary_csv(rows: list[SummaryRow], path) -> None:
    """Write ``locate/summary.csv`` (ARCHITECTURE.md §3), one row per event.

    ``None`` fields are written as empty CSV cells (pandas' default `NaN`
    rendering for a missing value).

    Parameters
    ----------
    rows : list[SummaryRow]
    path : str or pathlib.Path
        Destination path, e.g. ``RunPaths.locate_summary_csv``. Created
        (with its parent directory) if it doesn't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "event_id": r.event_id,
            "time": r.time,
            "lat": r.lat,
            "lon": r.lon,
            "depth_km": r.depth_km,
            "rms": r.rms,
            "gap": r.gap,
            "n_p": r.n_p,
            "n_s": r.n_s,
            "err_h_km": r.err_h_km,
            "err_z_km": r.err_z_km,
            "status": r.status,
            "reason": r.reason,
        }
        for r in rows
    ]
    df = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    df.to_csv(path, index=False, columns=SUMMARY_COLUMNS)


def write_removal_log_csv(records: list[RemovalRecord], path) -> None:
    """Write ``locate/removal_log.csv`` (ARCHITECTURE.md §3), one row per
    removed pick (across both passes, and pass B's S-prediction screening,
    ``iteration=0``).

    ``residual_s`` is always populated: every :class:`openmeq.locate.pass_a
    .RemovalRecord` (from residual filtering *and* pass B's S-prediction
    screening, whose ``residual_s`` is the predicted-vs-actual time
    difference, ``openmeq.locate.pass_b``'s own design decision #3) carries
    a real value -- ``RemovalRecord.residual_s`` is a required ``float``,
    never ``None``.

    Parameters
    ----------
    records : list[openmeq.locate.pass_a.RemovalRecord]
    path : str or pathlib.Path
        Destination path, e.g. ``RunPaths.removal_log_csv``. Created (with
        its parent directory) if it doesn't exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event_id": r.event_id,
            "pass": r.pass_label,
            "iteration": r.iteration,
            "seed_id": r.seed_id,
            "phase": r.phase,
            "residual_s": r.residual_s,
            "reason": r.reason,
        }
        for r in records
    ]
    df = pd.DataFrame(rows, columns=REMOVAL_LOG_COLUMNS)
    df.to_csv(path, index=False, columns=REMOVAL_LOG_COLUMNS)
