"""Pass B: S screening against the predicted S arrival, then P + S iterative
location and residual filtering (M6.3 / OpenMEQ-xdf.3).

PLAN.md §6 Step 5, Pass B:

1. Screen S picks against the S arrival predicted from the pass-A location;
   drop those outside ``s_predicted_window_s``.
2. Run NLLoc with the remaining P + S picks.
3. Filter P and S residuals separately (``p_and_s.p``, ``p_and_s.s``).
4. Relocate; repeat until stable or ``max_iter`` is reached.

:func:`run_pass_b` is the public per-event entry point; :func:`run_pass_b_many`
is a thin parallel-over-events wrapper around it (via ``openmeq.parallel.pmap``)
for a caller (M6.4) that already has every event's :class:`PassAResult` in hand.

Pass B never runs on an event pass A already rejected (PLAN.md's step
ordering places the ``min_p_phases`` gate before pass B begins) -- see
:func:`run_pass_b`'s early-return.

Design decisions
-----------------
**1. The S-prediction mechanism is `openmeq.locate.ttime_lookup.predict_time`
-- a direct grid read, not another NLLoc/Time2EQ subprocess.** See that
module's own docstring for the full justification (speed) and the exact
grid geometry/interpolation approximation (nearest node, bounded error).

**2. The relocate-and-filter loop is `openmeq.locate.pass_a.iterate_until_stable`,
reused verbatim** (same module pass A already exposes it from, per that
module's own docstring: "pass B ... reuses it verbatim with a different
phase set and per-phase thresholds"). Passing
``phase_thresholds={"P": cfg.nlloc.filter.p_and_s.p, "S": cfg.nlloc.filter.p_and_s.s}``
gets independent P/S thresholds "for free": ``iterate_until_stable`` already
computes each phase's k-sigma purely from that phase's own residuals in the
current iteration (see that function's docstring), so a tight P threshold
and a looser S threshold never interact.

**3. S-prediction screening is "iteration 0" for removal-log purposes.**
It happens once, before the relocate-and-filter loop starts (there is no
origin yet to re-predict from between loop iterations -- the loop's own
origin only exists once NLLoc has actually run), so every
:class:`openmeq.locate.pass_a.RemovalRecord` this screening step produces
uses ``iteration=0`` to distinguish it from the loop's own 1-based
iterations. Its ``residual_s`` is the signed
``pick.time - predicted_arrival_time`` difference that got it dropped --
not an NLLoc arrival residual (there isn't one yet: the pick was screened
out *before* ever being handed to NLLoc), but the same field the issue's own
wording implies ("the residual that got this pick dropped").

**4. No arrival for `openmeq.catalog.reject_pick` at the screening stage.**
That shared helper's ``arrival`` parameter is now optional (``None``,
minimal/additive change made in this commit, see the module's own
docstring) precisely for this case -- there is no ``Origin``/``Arrival`` for
a pick that never reached NLLoc.

**5. No event-level status beyond `"ok"`/`"rejected"` for an internal NLLoc
failure.** Per the issue's own "Out of scope": this task does not apply
``min_phases``/``min_s_phases``/RMS/gap/error thresholds -- M6.4 owns all
final event-level rejection. So, mirroring `PassAResult`'s own convention,
``PassBResult.status`` is ``"rejected"`` only when every relocation attempt
raised ``NLLocFailed`` (``iterate_until_stable``'s own "0 accepted picks
this iteration" contract, surfaced as ``origin is None``), never for a
merely-thin surviving pick set -- that judgement belongs to M6.4.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import reject_pick
from openmeq.locate.pass_a import IterationRecord, PassAResult, RemovalRecord, iterate_until_stable
from openmeq.locate.ttime_lookup import predict_time
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from obspy.core.event import Event, Origin, Pick

    from openmeq.config import Config
    from openmeq.paths import RunPaths


@dataclass
class PassBResult:
    """Result of :func:`run_pass_b`.

    Attributes
    ----------
    origin : obspy.core.event.Origin or None
        The last successful relocation (mirrors :class:`PassAResult.origin
        <openmeq.locate.pass_a.PassAResult>`'s own convention: ``None`` only
        if every relocation attempt raised ``NLLocFailed``, kept even if
        ``status == "rejected"`` so M6.4/M6.5 always have something to
        report). For an event pass A already rejected, this is
        ``pass_a_result.origin`` unchanged (pass B never ran).
    accepted_p : list[obspy.core.event.Pick]
        Surviving P picks after filtering.
    accepted_s : list[obspy.core.event.Pick]
        Surviving S picks after screening and filtering.
    removed : list[openmeq.locate.pass_a.RemovalRecord]
        Every pick removed, both by the S-prediction screening step
        (``iteration=0``) and by the relocate-and-filter loop (iterations
        1..N), in that order.
    n_iterations : int
        Number of ``locate_event`` calls made by the relocate-and-filter
        loop (``0`` if pass A had already rejected the event, so the loop
        never ran).
    status : {"ok", "rejected"}
        ``"rejected"`` if pass A already rejected the event (status/reason
        passed through unchanged), or if every pass-B relocation attempt
        raised ``NLLocFailed``. Never set for a merely-thin surviving pick
        set -- see module docstring, design decision #5.
    reason : str or None
        ``None`` if ``status == "ok"``; otherwise pass A's own reason
        (passed through), or ``f"nlloc_failed:{message}"``.
    history : list[openmeq.locate.pass_a.IterationRecord]
        Per-iteration diagnostics from the relocate-and-filter loop (empty
        if pass A had already rejected the event) -- M6.4's removal log and
        M6.5's diagnostic plots consume this the same way they consume
        ``PassAResult.history``.
    """

    origin: Origin | None
    accepted_p: list[Pick]
    accepted_s: list[Pick]
    removed: list[RemovalRecord]
    n_iterations: int
    status: Literal["ok", "rejected"]
    reason: str | None
    history: list[IterationRecord] = field(default_factory=list)


def _screen_s_picks(
    event: Event, origin: Origin, paths: RunPaths, s_predicted_window_s: float
) -> tuple[list[Pick], list[RemovalRecord]]:
    """Predict each not-yet-rejected S pick's arrival from ``origin`` (via
    :func:`openmeq.locate.ttime_lookup.predict_time`) and drop those whose
    actual pick time differs from the prediction by more than
    ``s_predicted_window_s``. Returns ``(surviving_s_picks, removed_records)``.
    """
    eid = get_event_id(event)
    grid_root = paths.nlloc_grid_root
    depth_km = origin.depth / 1000.0  # QuakeML/obspy Origin.depth is metres.

    surviving: list[Pick] = []
    removed: list[RemovalRecord] = []
    for pick in event.picks:
        if pick.phase_hint != "S" or pick.evaluation_status == "rejected":
            continue
        station = pick.waveform_id.station_code
        travel_time_s = predict_time(
            grid_root, "S", origin.latitude, origin.longitude, depth_km, station
        )
        predicted_arrival = origin.time + travel_time_s
        diff_s = pick.time - predicted_arrival
        if abs(diff_s) > s_predicted_window_s:
            reject_pick(pick, None, "s_predicted_window", step="locate/passB")
            removed.append(
                RemovalRecord(
                    event_id=eid,
                    pass_label="B",
                    iteration=0,
                    seed_id=pick.waveform_id.id,
                    phase="S",
                    residual_s=diff_s,
                    reason="s_predicted_window",
                )
            )
        else:
            surviving.append(pick)
    return surviving, removed


def run_pass_b(
    event: Event, pass_a_result: PassAResult, paths: RunPaths, cfg: Config
) -> PassBResult:
    """Pass B: S-prediction screening, then iterative P+S location and
    residual filtering.

    Never runs (returns immediately) if ``pass_a_result.status ==
    "rejected"`` -- PLAN.md's step ordering places the ``min_p_phases`` gate
    before pass B begins.

    Otherwise: predicts each S pick's arrival from ``pass_a_result.origin``
    (:func:`openmeq.locate.ttime_lookup.predict_time`) and screens out those
    farther than ``cfg.nlloc.filter.p_and_s.s_predicted_window_s`` away, then
    combines pass A's final accepted P picks with the surviving S picks as
    the starting set for :func:`openmeq.locate.pass_a.iterate_until_stable`,
    filtering P and S residuals against ``cfg.nlloc.filter.p_and_s.p``/``.s``
    independently, up to ``cfg.nlloc.filter.p_and_s.max_iter`` iterations.

    Parameters
    ----------
    event : obspy.core.event.Event
        Must already carry its PhaseNet (or hand-built) P and S picks in
        ``event.picks``.
    pass_a_result : openmeq.locate.pass_a.PassAResult
        Pass A's result for this same event (M6.2 / OpenMEQ-xdf.2).
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config

    Returns
    -------
    PassBResult
    """
    if pass_a_result.status == "rejected":
        return PassBResult(
            origin=pass_a_result.origin,
            accepted_p=list(pass_a_result.accepted_picks),
            accepted_s=[],
            removed=[],
            n_iterations=0,
            status="rejected",
            reason=pass_a_result.reason,
            history=[],
        )

    p_and_s = cfg.nlloc.filter.p_and_s
    assert pass_a_result.origin is not None  # status == "ok" implies a successful origin.

    screened_s_picks, screening_removed = _screen_s_picks(
        event, pass_a_result.origin, paths, p_and_s.s_predicted_window_s
    )

    initial_picks = list(pass_a_result.accepted_picks) + screened_s_picks

    result = iterate_until_stable(
        event=event,
        picks=initial_picks,
        paths=paths,
        cfg=cfg,
        pass_name="passB",
        removal_pass_label="B",
        phase_thresholds={"P": p_and_s.p, "S": p_and_s.s},
        max_iter=p_and_s.max_iter,
        step="locate/passB",
    )

    accepted_p = [p for p in result.accepted_picks if p.phase_hint == "P"]
    accepted_s = [p for p in result.accepted_picks if p.phase_hint == "S"]

    status: Literal["ok", "rejected"] = "ok"
    reason: str | None = None
    if result.origin is None:
        status = "rejected"
        reason = result.last_nlloc_failed_reason

    return PassBResult(
        origin=result.origin,
        accepted_p=accepted_p,
        accepted_s=accepted_s,
        removed=screening_removed + result.removed,
        n_iterations=result.n_iterations,
        status=status,
        reason=reason,
        history=result.history,
    )


def _run_pass_b_item(
    item: tuple[Event, PassAResult, RunPaths, Config],
) -> PassBResult:
    """``openmeq.parallel.pmap``-compatible adapter for :func:`run_pass_b`
    (``pmap``'s ``func`` takes exactly one, picklable, item)."""
    event, pass_a_result, paths, cfg = item
    return run_pass_b(event, pass_a_result, paths, cfg)


def run_pass_b_many(
    events: list[Event],
    pass_a_results: Mapping[str, PassAResult],
    paths: RunPaths,
    cfg: Config,
) -> dict[str, PassBResult]:
    """Run :func:`run_pass_b` over many events in parallel
    (``cfg.parallel``, via ``openmeq.parallel.pmap``).

    A thin convenience wrapper for a caller (M6.4's orchestration) that
    already has every event's pass-A result in hand and just wants pass B
    run across the whole catalogue -- ``run_pass_b`` itself stays a plain
    per-event function so it is trivially unit-testable on its own.

    Parameters
    ----------
    events : list[obspy.core.event.Event]
    pass_a_results : Mapping[str, openmeq.locate.pass_a.PassAResult]
        ``event_id -> PassAResult``, keyed by
        ``openmeq.catalog.event_id`` -- must have an entry for every event
        in ``events``.
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config

    Returns
    -------
    dict[str, PassBResult]
        ``event_id -> PassBResult``, one entry per event in ``events``.
    """
    items = [(event, pass_a_results[get_event_id(event)], paths, cfg) for event in events]
    results = pmap(_run_pass_b_item, items, cfg.parallel, desc="locate: pass B")
    return {get_event_id(event): result for event, result in zip(events, results, strict=True)}
