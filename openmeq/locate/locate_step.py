"""Locate step orchestration (M6.4 / OpenMEQ-xdf.4): pass A -> pass B ->
finalize, per event, parallelised over events.

:func:`run_locate` is the function ``openmeq locate`` (``openmeq.locate.run``,
M6.5 / OpenMEQ-xdf.5) calls. Reads ``pick/catalog_picks.xml`` (M4.4's
output: events with automatic picks, no origins yet), runs
:func:`openmeq.locate.pass_a.run_pass_a`, :func:`openmeq.locate.pass_b.run_pass_b`
and :func:`openmeq.locate.finalize.finalize_event` for every event, and
writes the three M6 output artifacts (ARCHITECTURE.md §3):
``locate/catalog_auto.xml``, ``locate/summary.csv``,
``locate/removal_log.csv``.

**Per-event parallelism.** Per the M6 epic's cross-cutting design note (also
restated in this issue's own "DESIGN" section), the ``openmeq.parallel.pmap``
unit function (:func:`_locate_one_event`) wraps all three steps -- pass A,
pass B, finalize -- together for *one* event, rather than parallelising each
step separately (as ``openmeq.locate.pass_b.run_pass_b_many`` alone would).
This halves the number of process-pool round trips per event (one dispatch
instead of two) and keeps each worker's per-event NLLoc grid/context cache
(``openmeq.locate.nlloc_runner``'s own per-process cache) warm across all
three steps for that event, rather than potentially cold-starting it twice
if pass A and pass B landed in different worker processes.

**Per-pass diagnostics (M6.5, additive).** :func:`run_locate` now also
returns a :class:`LocateRunResult`, which carries -- alongside the three
on-disk artifacts it already wrote -- one :class:`PassDiagnostics` record
per event per pass (``pass_a``/``pass_b``), for
``openmeq.locate.diagnostics.make_locate_diagnostics`` to consume. This is
an additive change only: every field ``run_locate`` and ``_locate_one_event``
previously returned/wrote is unchanged, and no existing caller that ignores
the return value (e.g. ``tests/locate/test_locate_step.py``) is affected.

``PassDiagnostics`` is deliberately built entirely from **plain,
picklable-and-back-safe data** (floats/strings/dicts), extracted *inside*
the ``pmap`` worker, right after ``run_pass_a``/``run_pass_b`` return --
never from the ``PassAResult``/``PassBResult`` objects themselves after they
have crossed back over the process-pool boundary. The reason:
``obspy.core.event.Arrival.pick_id.get_referred_object()`` resolves a
``ResourceIdentifier`` via an in-process registry that a pickled/unpickled
``Origin`` cannot be relied on to have preserved, so any diagnostics code
that needs to know "which pick does this residual/arrival belong to, and
what phase is it" must do that resolution in the same process the
``Origin``/``Pick`` objects were created in (mirrors
``openmeq.locate.pass_a.iterate_until_stable``'s own in-worker residual
extraction into a plain ``dict[str, float]``, which this module reuses
rather than re-deriving).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from obspy import Catalog

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import iso_z, read_catalog, write_catalog
from openmeq.locate.finalize import (
    SummaryRow,
    finalize_event,
    horizontal_error_km,
    write_removal_log_csv,
    write_summary_csv,
)
from openmeq.locate.pass_a import PassAResult, RemovalRecord, run_pass_a
from openmeq.locate.pass_b import PassBResult, run_pass_b
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from obspy.core.event import Event

    from openmeq.config import Config
    from openmeq.paths import RunPaths


@dataclass(frozen=True)
class PassDiagnostics:
    """Plain-data, per-event, per-pass diagnostic record for
    ``openmeq.locate.diagnostics`` (M6.5) -- see module docstring for why
    every field here is a plain float/str/dict rather than an obspy object.

    Attributes
    ----------
    event_id : str
    ran : bool
        ``False`` only for pass B on an event pass A already rejected
        (``openmeq.locate.pass_b.run_pass_b``'s own early-return) -- every
        other field is then at its default/empty value.
    status : str or None
        ``"ok"``/``"rejected"`` (the pass's own status), or ``None`` if
        ``ran`` is ``False``.
    reason : str or None
    n_iterations : int
    stabilized : bool or None
        ``history[-1]``'s implied convergence: ``True`` if the last
        iteration attempted removed nothing, ``False`` if it still removed
        picks (``max_iter`` reached) or the pass's own ``NLLocFailed``
        fallback fired. ``None`` if ``ran`` is ``False`` or ``history`` is
        empty.
    time, lat, lon, depth_km, rms, gap, err_h_km, err_z_km : float or None
        Read directly off the pass's own final ``Origin`` (``pass.origin``,
        the last *successful* relocation) -- ``None`` if that pass never
        produced one. Distances (``gap``... not a distance, kept in
        degrees, matching ``Origin.quality.azimuthal_gap``'s own unit) are
        otherwise as stored on the ``Origin`` (ARCHITECTURE.md §1's km
        convention already applies to the origin itself by this point).
    residuals : dict[str, float]
        ``seed_id -> time_residual`` from this pass's own *last*
        ``IterationRecord`` (``history[-1].residuals``) -- empty if
        ``history`` is empty.
    phase_by_seed_id : dict[str, str]
        ``seed_id -> "P"/"S"``, built from ``event.picks`` (stable across
        iterations/passes -- a pick's own phase never changes once created).
        Lets a consumer split ``residuals`` by phase without needing
        ``Arrival.pick_id`` resolution.
    n_picks_removed : int
        ``len(pass.removed)``.
    distances_km : dict[str, float]
        ``seed_id -> epicentral distance (km)`` for every pick in
        ``residuals``, read off the same last iteration's ``Origin``
        arrivals (``Arrival.distance``, already km per
        ``openmeq.locate.nlloc_runner.locate_event``). Added for the
        extended QAQC residual-vs-distance plot (OpenMEQ-02o.5); defaults
        to empty so every existing constructor call is unaffected. A seed
        id with no arrival distance is simply absent.
    """

    event_id: str
    ran: bool
    status: str | None
    reason: str | None
    n_iterations: int
    stabilized: bool | None
    time: str | None
    lat: float | None
    lon: float | None
    depth_km: float | None
    rms: float | None
    gap: float | None
    err_h_km: float | None
    err_z_km: float | None
    residuals: dict[str, float] = field(default_factory=dict)
    phase_by_seed_id: dict[str, str] = field(default_factory=dict)
    n_picks_removed: int = 0
    distances_km: dict[str, float] = field(default_factory=dict)


def _arrival_distances_km(event: Event, origin) -> dict[str, float]:
    """``seed_id -> Arrival.distance`` (km) for ``origin``'s arrivals, in-worker.

    Matches each arrival to its pick by ``pick_id`` string against
    ``event.picks``' own ``resource_id``s rather than via
    ``get_referred_object()`` -- same in-worker rule as the module
    docstring, and immune to the resource registry altogether.
    """
    if origin is None:
        return {}
    seed_by_rid = {str(p.resource_id): p.waveform_id.id for p in event.picks}
    out: dict[str, float] = {}
    for arrival in origin.arrivals:
        if arrival.distance is None or arrival.pick_id is None:
            continue
        seed_id = seed_by_rid.get(str(arrival.pick_id))
        if seed_id is not None:
            out[seed_id] = float(arrival.distance)
    return out


@dataclass
class LocateRunResult:
    """Result of :func:`run_locate` (M6.5, additive -- see module docstring).

    Attributes
    ----------
    events : list[obspy.core.event.Event]
        The same events written to ``locate/catalog_auto.xml``.
    summary_rows : list[openmeq.locate.finalize.SummaryRow]
        The same rows written to ``locate/summary.csv``.
    removal_records : list[openmeq.locate.pass_a.RemovalRecord]
        The same records written to ``locate/removal_log.csv``.
    pass_a : dict[str, PassDiagnostics]
        ``event_id -> PassDiagnostics`` for pass A, one entry per input event.
    pass_b : dict[str, PassDiagnostics]
        ``event_id -> PassDiagnostics`` for pass B, one entry per input
        event (``ran=False`` for an event pass A already rejected).
    """

    events: list[Event]
    summary_rows: list[SummaryRow]
    removal_records: list[RemovalRecord]
    pass_a: dict[str, PassDiagnostics]
    pass_b: dict[str, PassDiagnostics]


def _origin_fields(origin) -> tuple:
    """``(time, lat, lon, depth_km, rms, gap, err_h_km, err_z_km)`` read
    directly off ``origin`` (or all-``None`` if ``origin is None``) -- no
    ``Arrival``/``pick_id`` resolution, safe to call in-worker before this
    module's own diagnostics record crosses back over the process-pool
    boundary (module docstring)."""
    if origin is None:
        return (None, None, None, None, None, None, None, None)
    quality = origin.quality
    ou = origin.origin_uncertainty
    de = origin.depth_errors
    err_h_km = horizontal_error_km(ou)
    err_z_km = de.uncertainty / 1000.0 if de and de.uncertainty is not None else None
    return (
        iso_z(origin.time) if origin.time is not None else None,
        origin.latitude,
        origin.longitude,
        origin.depth / 1000.0 if origin.depth is not None else None,
        quality.standard_error if quality is not None else None,
        quality.azimuthal_gap if quality is not None else None,
        err_h_km,
        err_z_km,
    )


def _pass_diagnostics(event: Event, result: PassAResult | PassBResult | None) -> PassDiagnostics:
    """Build a :class:`PassDiagnostics` for one event/pass, in-worker (see
    module docstring). ``result is None`` covers pass B's own early-return
    for a pass-A-rejected event (``run_pass_b``'s ``PassBResult`` still
    carries pass A's own fields through in that case, so this is only
    reached when the caller explicitly has nothing to report, never used by
    :func:`_locate_one_event` itself -- kept for a future caller that might
    skip calling pass B at all)."""
    eid = get_event_id(event)
    if result is None:
        return PassDiagnostics(
            event_id=eid,
            ran=False,
            status=None,
            reason=None,
            n_iterations=0,
            stabilized=None,
            time=None,
            lat=None,
            lon=None,
            depth_km=None,
            rms=None,
            gap=None,
            err_h_km=None,
            err_z_km=None,
        )

    time_, lat, lon, depth_km, rms, gap, err_h_km, err_z_km = _origin_fields(result.origin)

    history = result.history
    residuals: dict[str, float] = dict(history[-1].residuals) if history else {}
    stabilized: bool | None = None
    if history:
        stabilized = not history[-1].removed and history[-1].nlloc_failed_reason is None

    phase_by_seed_id = {p.waveform_id.id: p.phase_hint for p in event.picks}
    distances_km = _arrival_distances_km(event, history[-1].origin) if history else {}

    return PassDiagnostics(
        event_id=eid,
        ran=True,
        status=result.status,
        reason=result.reason,
        n_iterations=result.n_iterations,
        stabilized=stabilized,
        time=time_,
        lat=lat,
        lon=lon,
        depth_km=depth_km,
        rms=rms,
        gap=gap,
        err_h_km=err_h_km,
        err_z_km=err_z_km,
        residuals=residuals,
        phase_by_seed_id=phase_by_seed_id,
        n_picks_removed=len(result.removed),
        distances_km=distances_km,
    )


def _locate_one_event(
    item: tuple[Event, RunPaths, Config],
) -> tuple[Event, SummaryRow, list[RemovalRecord], PassDiagnostics, PassDiagnostics]:
    """``openmeq.parallel.pmap``-compatible per-event unit: pass A -> pass B
    -> finalize, for one event (module docstring's "per-event parallelism"
    note).

    Also builds this event's pass A/B :class:`PassDiagnostics` (M6.5,
    module docstring) here, in-worker, right after each pass's own
    ``PassAResult``/``PassBResult`` is produced -- before it crosses back
    over the process-pool boundary.
    """
    event, paths, cfg = item

    pass_a_result = run_pass_a(event, paths, cfg)
    pass_a_diag = _pass_diagnostics(event, pass_a_result)
    if pass_a_result.status == "rejected":
        # PLAN.md's step ordering places the min_p_phases gate before pass B
        # begins (openmeq.locate.pass_b.run_pass_b's own early-return would
        # do the same if called, but skipping the call entirely avoids a
        # pointless S-prediction lookup against an already-rejected event).
        finalized = finalize_event(event, pass_a_result, None, cfg)
        removed = list(pass_a_result.removed)
        pass_b_diag = _pass_diagnostics(event, None)
    else:
        pass_b_result = run_pass_b(event, pass_a_result, paths, cfg)
        finalized = finalize_event(event, pass_a_result, pass_b_result, cfg)
        removed = list(pass_a_result.removed) + list(pass_b_result.removed)
        pass_b_diag = _pass_diagnostics(event, pass_b_result)

    return finalized.event, finalized.summary, removed, pass_a_diag, pass_b_diag


def run_locate(paths: RunPaths, cfg: Config) -> LocateRunResult:
    """Locate every event in ``pick/catalog_picks.xml`` and write the three
    M6 output artifacts.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config

    Returns
    -------
    LocateRunResult
        The same events/rows/records written to disk (below), plus a
        per-event pass A/B :class:`PassDiagnostics` record for
        ``openmeq.locate.diagnostics.make_locate_diagnostics`` (M6.5,
        additive -- see module docstring).

    Writes
    ------
    ``locate/catalog_auto.xml``
        Every event from the input catalogue, exactly once, each carrying
        its final `Origin` (pass A's, pass B's, or a rejection-only
        placeholder, ``openmeq.locate.finalize``) -- rejected events are
        flagged (``openmeq.catalog.reject_event``), never dropped
        (ARCHITECTURE.md §5).
    ``locate/summary.csv``
        One row per event, ``openmeq.locate.finalize.SUMMARY_COLUMNS``.
    ``locate/removal_log.csv``
        One row per pick removed by either pass (residual filtering) or
        pass B's S-prediction screening,
        ``openmeq.locate.finalize.REMOVAL_LOG_COLUMNS``.
    """
    input_catalog = read_catalog(paths.catalog_picks_xml)

    items = [(event, paths, cfg) for event in input_catalog]
    results = pmap(
        _locate_one_event, items, cfg.parallel, desc="locate: pass A + pass B + finalize"
    )

    events: list[Event] = []
    summary_rows: list[SummaryRow] = []
    removal_records: list[RemovalRecord] = []
    pass_a_diagnostics: dict[str, PassDiagnostics] = {}
    pass_b_diagnostics: dict[str, PassDiagnostics] = {}
    for event, summary_row, removed, pass_a_diag, pass_b_diag in results:
        events.append(event)
        summary_rows.append(summary_row)
        removal_records.extend(removed)
        pass_a_diagnostics[pass_a_diag.event_id] = pass_a_diag
        pass_b_diagnostics[pass_b_diag.event_id] = pass_b_diag

    out_catalog = Catalog(
        events=events, resource_id=input_catalog.resource_id, description=input_catalog.description
    )
    write_catalog(out_catalog, paths.catalog_auto_xml)
    write_summary_csv(summary_rows, paths.locate_summary_csv)
    write_removal_log_csv(removal_records, paths.removal_log_csv)

    # Defensive: every input event_id appears exactly once in summary.csv
    # (tests/locate/test_locate_step.py's own acceptance criterion) -- a
    # cheap, deterministic check here fails loudly rather than shipping a
    # silently-mismatched file.
    input_ids = [get_event_id(ev) for ev in input_catalog]
    summary_ids = [row.event_id for row in summary_rows]
    if sorted(summary_ids) != sorted(input_ids):
        raise AssertionError(
            "locate/summary.csv event_ids do not reconcile with pick/catalog_picks.xml: "
            f"{len(summary_ids)} summary rows vs {len(input_ids)} input events"
        )

    return LocateRunResult(
        events=events,
        summary_rows=summary_rows,
        removal_records=removal_records,
        pass_a=pass_a_diagnostics,
        pass_b=pass_b_diagnostics,
    )
