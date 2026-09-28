"""Pass A: P-only iterative location and residual filtering (M6.2 / OpenMEQ-xdf.2).

PLAN.md §6 Step 5, Pass A:

1. Run NLLoc with P picks only.
2. Drop P picks with ``|residual| > p_only.max_residual_s`` or more than
   ``residual_sigma``·σ from the rest.
3. Relocate; repeat until no picks are removed or ``max_iter`` is reached.
4. Reject events with fewer than ``min_p_phases`` P picks.

:func:`run_pass_a` is the public entry point. It is a thin wrapper around
:func:`iterate_until_stable`, the reusable iterate-locate-and-filter helper
this module also exposes: pass B (M6.3 / OpenMEQ-xdf.3) reuses it verbatim
with a different phase set (``{"P", "S"}``) and per-phase thresholds
(``cfg.nlloc.filter.p_and_s.p``/``.s``) rather than reimplementing the loop.

Design decisions
-----------------
**1. "Remove all offenders per iteration", not "remove only the single worst
offender".** PLAN.md §6 step 2 says, in the plural, "Drop P **picks** with
|residual| > ...", and step 3 relocates and repeats "until no picks are
removed" -- both read as an explicit description of the "drop everything
over threshold, then relocate" rule, not "drop one, then relocate". The
issue's own worked steps (its "Work to do" §1.c/d) match this exactly: step
c computes a whole *set* of picks over threshold from one iteration's
residuals, and step d rejects each one in that set before the next
relocation. This module follows that literal, doubly-confirmed contract
rather than the more conservative "single worst offender per iteration"
alternative, for two reasons:

* **Correctness is preserved by the relocate-and-recheck loop itself.**
  Every removal decision is judged against a residual field that is
  *recomputed from scratch* after every relocation. A pick that only looked
  bad because a genuinely bad pick was still contaminating the origin gets
  a fresh, fair residual next iteration; removing several offenders in one
  pass is not a one-shot, irrevocable judgement -- it is one more round of
  the same iterate-and-recheck procedure that the single-worst-offender
  strategy also relies on. The event-level ``min_p_phases`` check afterwards
  (§4 above) is the backstop if a pass over-removes.
* **It is the faster of two contracts that are equally safe in practice.**
  With ``residual_sigma`` defaulting to 3 and ``max_residual_s`` a tight
  absolute bound (0.10 s), both thresholds are conservative enough that
  "more than one bad pick in a single iteration" is the interesting case
  precisely when several picks are *genuinely* bad (e.g. a mispick on
  several stations at once) -- exactly when removing them all in one round
  trip is both correct and desirable (PLAN.md §4: "speed is a design goal";
  each relocation is a real NLLoc subprocess). The single-worst-offender
  rule would need up to ``N`` extra relocate rounds to remove ``N``
  simultaneously-bad picks, for no accuracy benefit given the point above.

**2. The "sigma" in the k-sigma test is a raw, uncentered RMS, not
``numpy.std``.** The k-sigma condition is ``abs(residual) > residual_sigma
* std(residuals-this-iteration)``, where ``std`` is the **population**
standard deviation (``ddof=0``) computed **on the raw residuals, without
first subtracting their mean** -- i.e. ``sqrt(mean(residual**2))``, not
``sqrt(mean((residual - mean(residual))**2))``. This is a design decision
this module makes (PLAN.md §3 does not specify it), for two reasons:

* NLLoc's ``LOCMETH EDT_OT_WT`` (the method this project always uses,
  ``cfg.nlloc.locmeth``) already solves for the origin time that minimizes
  the *weighted* residual sum, so a clean pick set's residuals are already
  centered on (approximately) zero by construction -- explicitly
  subtracting the mean again is largely redundant work.
* It degrades far more sensibly at small sample sizes, which matters a lot
  here since a phase's residual sample size shrinks every time a pick is
  dropped. With the classical mean-subtracted formula, a *single* residual
  has ``std == 0`` exactly (it equals its own mean), so the k-sigma test
  would flag *any* nonzero residual, however small -- clearly not intended.
  With the raw-RMS formula, a single residual ``r`` gives ``sigma ==
  abs(r)``, so ``abs(r) > residual_sigma * sigma`` becomes ``1 >
  residual_sigma``, which is false for every ``residual_sigma >= 1``
  (``config/example.yaml``'s default is 3) -- the k-sigma test simply never
  fires with one data point, which is the only sane behaviour: there is no
  "rest" to be an outlier from. The flat ``max_residual_s`` bound is still
  fully in effect regardless of sample size, so a single wildly-off residual
  is still caught by it.

**3. The ``min_p_phases`` check runs once, after the loop -- not
per-iteration.** Picks are only ever removed, never re-added, so the
accepted-pick count is monotonically non-increasing across iterations; a
single post-loop check is therefore sufficient (this is also stated
verbatim in the issue's own "DESIGN" section).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import reject_pick
from openmeq.locate.nlloc_runner import NLLocFailed, locate_event

if TYPE_CHECKING:
    from obspy.core.event import Event, Origin, Pick

    from openmeq.config import Config
    from openmeq.paths import RunPaths


class _PhaseThreshold(Protocol):
    """Structural type for a per-phase residual-filter threshold.

    Both :class:`openmeq.config.POnlyFilterConfig` (pass A) and
    :class:`openmeq.config.PhaseResidualFilterConfig` (pass B's ``p``/``s``)
    already have exactly these two attributes, so either can be passed
    straight through as a ``phase_thresholds`` value with no conversion.
    """

    max_residual_s: float
    residual_sigma: float


@dataclass(frozen=True)
class RemovalRecord:
    """One pick removed by :func:`iterate_until_stable`, at one iteration.

    This is the row shape ``locate/removal_log.csv`` (ARCHITECTURE.md §3:
    ``event_id,pass,iteration,seed_id,phase,residual_s,reason``) is built
    from (M6.4's job -- this module only produces the in-memory records).
    The field is named ``pass_label`` rather than ``pass`` because ``pass``
    is a Python keyword and cannot be an attribute name; it maps directly to
    the CSV's ``pass`` column.

    Attributes
    ----------
    event_id : str
    pass_label : str
        ``"A"`` or ``"B"`` -- short pass label for the CSV/log, distinct
        from ``locate_event``'s own ``pass_name`` argument (``"passA"``/
        ``"passB"``, used for on-disk directory naming).
    iteration : int
        The 1-based iteration this pick was dropped at.
    seed_id : str
        The pick's full SEED id (``pick.waveform_id.id``, ``NET.STA.LOC.CHA``).
    phase : str
        ``"P"`` or ``"S"`` (``pick.phase_hint``).
    residual_s : float
        The time residual that got this pick dropped.
    reason : str
        ``"residual>max"`` or ``"residual>k_sigma"`` -- ``"residual>max"``
        is used whenever both conditions trigger (the issue's own
        tie-break rule).
    """

    event_id: str
    pass_label: str
    iteration: int
    seed_id: str
    phase: str
    residual_s: float
    reason: str


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics from one :func:`iterate_until_stable` call.

    One of these is appended per iteration actually attempted (i.e. per
    :func:`openmeq.locate.nlloc_runner.locate_event` call, whether it
    succeeded or raised :class:`NLLocFailed`). This is the structure M6.4
    (removal log) and M6.5 (per-pass diagnostic plots, ARCHITECTURE.md §5:
    "P and S residual distributions (all, per station)") are expected to
    consume -- ``residuals`` carries *every* located pick's residual this
    iteration (not just the ones that got removed), so a full
    residual-distribution plot doesn't need to re-run anything.

    Attributes
    ----------
    iteration : int
        1-based iteration number.
    n_picks_in : int
        Number of picks handed to ``locate_event`` this iteration (before
        any removal at the end of this same iteration).
    origin : obspy.core.event.Origin or None
        The relocated origin, or ``None`` if this iteration's
        ``locate_event`` call raised :class:`NLLocFailed`.
    residuals : dict[str, float]
        ``seed_id -> time_residual`` for every pick this iteration actually
        produced a residual for (empty if ``origin`` is ``None``). Keyed by
        seed id rather than ``(seed_id, phase)`` because P and S picks
        always live on different channels (ARCHITECTURE.md §5), so seed id
        alone is already unambiguous.
    sigma_by_phase : dict[str, float]
        ``phase -> sigma`` (the raw-RMS value described in the module
        docstring) actually used for that phase's k-sigma test this
        iteration. Only populated for phases with at least one residual.
    removed : list[RemovalRecord]
        Picks dropped at the end of this iteration (empty if the iteration
        was stable, or if it raised ``NLLocFailed``).
    nlloc_failed_reason : str or None
        ``f"nlloc_failed:{message}"`` if this iteration's ``locate_event``
        call raised :class:`NLLocFailed`, else ``None``.
    """

    iteration: int
    n_picks_in: int
    origin: Origin | None
    residuals: dict[str, float]
    sigma_by_phase: dict[str, float]
    removed: list[RemovalRecord]
    nlloc_failed_reason: str | None


@dataclass
class IterateResult:
    """Result of :func:`iterate_until_stable`.

    Attributes
    ----------
    origin : obspy.core.event.Origin or None
        The last *successful* relocation, or ``None`` if every iteration
        (including the first) raised :class:`NLLocFailed`.
    accepted_picks : list[obspy.core.event.Pick]
        The surviving pick set. Empty if the loop ended via
        ``NLLocFailed`` (treated as "0 accepted picks this iteration",
        per the issue's own rule).
    removed : list[RemovalRecord]
        Every pick removed, across every iteration, in removal order.
    history : list[IterationRecord]
        One entry per iteration attempted.
    n_iterations : int
        Number of ``locate_event`` calls made (successful or not).
    stabilized : bool
        ``True`` if the loop ended because an iteration removed nothing
        (the normal, "converged" ending). ``False`` if it ended because
        ``max_iter`` was reached while still removing picks, or because
        ``NLLocFailed`` cut the loop short.
    last_nlloc_failed_reason : str or None
        Set if the loop ended because of an ``NLLocFailed`` (the last such
        reason seen); ``None`` otherwise.
    """

    origin: Origin | None
    accepted_picks: list[Pick]
    removed: list[RemovalRecord]
    history: list[IterationRecord]
    n_iterations: int
    stabilized: bool
    last_nlloc_failed_reason: str | None


@dataclass
class PassAResult:
    """Result of :func:`run_pass_a`.

    Attributes
    ----------
    origin : obspy.core.event.Origin or None
        The last successful relocation (kept even if ``status ==
        "rejected"``, so M6.4/M6.5 always have something to report).
    accepted_picks : list[obspy.core.event.Pick]
        Surviving P picks after filtering.
    removed : list[RemovalRecord]
        Every P pick removed, across every iteration.
    n_iterations : int
        Number of ``locate_event`` calls made.
    status : {"ok", "rejected"}
        ``"rejected"`` only for the ``min_p_phases`` shortfall (this
        function never calls ``reject_event`` itself -- see module
        docstring / M6.2 issue: M6.4 owns all final event-rejection
        bookkeeping in one place).
    reason : str or None
        ``None`` if ``status == "ok"``; otherwise ``"min_p_phases"``, or
        ``f"nlloc_failed:{message}"`` if the shortfall was caused by an
        ``NLLocFailed`` rather than residual filtering.
    history : list[IterationRecord]
        Per-iteration diagnostics (M6.4's removal log, M6.5's diagnostic
        plots) -- see :class:`IterationRecord`.
    """

    origin: Origin | None
    accepted_picks: list[Pick]
    removed: list[RemovalRecord]
    n_iterations: int
    status: Literal["ok", "rejected"]
    reason: str | None
    history: list[IterationRecord] = field(default_factory=list)


def _raw_rms(residuals: list[float]) -> float:
    """``sqrt(mean(residual**2))`` -- see the module docstring's design
    decision #2 (a population std computed *without* subtracting the mean
    first)."""
    if not residuals:
        return 0.0
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def iterate_until_stable(
    event: Event,
    picks: list[Pick],
    paths: RunPaths,
    cfg: Config,
    *,
    pass_name: str,
    removal_pass_label: str,
    phase_thresholds: Mapping[str, _PhaseThreshold],
    max_iter: int,
    step: str,
) -> IterateResult:
    """Relocate-and-filter loop, reused by pass A (this module) and pass B
    (M6.3 / OpenMEQ-xdf.3).

    Repeatedly calls :func:`openmeq.locate.nlloc_runner.locate_event` with
    the current surviving pick set, computes each phase's residuals from
    the returned origin's arrivals, drops any pick over that phase's
    threshold (see the module docstring for the exact rule and the reasons
    behind it), and stops when either an iteration removes nothing (stable)
    or ``max_iter`` is reached. A dropped pick is flagged via
    :func:`openmeq.catalog.reject_pick` (never deleted, ARCHITECTURE.md §5)
    and is never re-added in a later iteration.

    Parameters
    ----------
    event : obspy.core.event.Event
    picks : list[obspy.core.event.Pick]
        The initial pick set (already filtered by the caller to "accepted,
        correct phase(s) for this pass" -- this function does not decide
        inclusion at the start, only removal as it iterates).
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config
    pass_name : str
        ``"passA"`` or ``"passB"``, forwarded to ``locate_event`` (and thus
        ``paths.locate_pass_dir``) verbatim.
    removal_pass_label : str
        ``"A"`` or ``"B"`` -- the short label stamped onto every
        :class:`RemovalRecord` this call produces.
    phase_thresholds : Mapping[str, _PhaseThreshold]
        ``phase_hint -> threshold`` (e.g. ``{"P": cfg.nlloc.filter.p_only}``
        for pass A, or ``{"P": cfg.nlloc.filter.p_and_s.p, "S":
        cfg.nlloc.filter.p_and_s.s}`` for pass B). Only picks whose
        ``phase_hint`` is a key here are ever candidates for removal;
        picks of any other phase present in ``picks`` are passed to NLLoc
        but never filtered by this function.
    max_iter : int
    step : str
        Passed straight through to ``reject_pick`` (e.g. ``"locate/passA"``).

    Returns
    -------
    IterateResult
    """
    eid = get_event_id(event)
    current = list(picks)
    history: list[IterationRecord] = []
    removed_all: list[RemovalRecord] = []
    last_good_origin: Origin | None = None
    last_nlloc_failed_reason: str | None = None
    stabilized = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        try:
            origin = locate_event(event, current, paths, cfg, pass_name, iteration)
        except NLLocFailed as exc:
            last_nlloc_failed_reason = f"nlloc_failed:{exc}"
            history.append(
                IterationRecord(
                    iteration=iteration,
                    n_picks_in=len(current),
                    origin=None,
                    residuals={},
                    sigma_by_phase={},
                    removed=[],
                    nlloc_failed_reason=last_nlloc_failed_reason,
                )
            )
            # "Treat NLLocFailed as 0 accepted picks this iteration" -- the
            # caller's post-loop min_p_phases check picks this up.
            current = []
            break

        last_good_origin = origin
        n_picks_in = len(current)

        arrival_by_pick_id: dict[int, object] = {}
        for arrival in origin.arrivals:
            referred = arrival.pick_id.get_referred_object() if arrival.pick_id else None
            if referred is not None:
                arrival_by_pick_id[id(referred)] = arrival

        residuals_by_phase: dict[str, list[tuple[Pick, object, float]]] = {}
        residuals_out: dict[str, float] = {}
        for pick in current:
            arrival = arrival_by_pick_id.get(id(pick))
            if arrival is None or arrival.time_residual is None:
                continue
            residual = arrival.time_residual
            residuals_out[pick.waveform_id.id] = residual
            residuals_by_phase.setdefault(pick.phase_hint, []).append((pick, arrival, residual))

        sigma_by_phase: dict[str, float] = {}
        to_remove: list[tuple[Pick, object, float, str]] = []
        for phase, threshold in phase_thresholds.items():
            entries = residuals_by_phase.get(phase)
            if not entries:
                continue
            sigma = _raw_rms([r for _, _, r in entries])
            sigma_by_phase[phase] = sigma
            for pick, arrival, residual in entries:
                abs_r = abs(residual)
                if abs_r > threshold.max_residual_s:
                    to_remove.append((pick, arrival, residual, "residual>max"))
                elif abs_r > threshold.residual_sigma * sigma:
                    to_remove.append((pick, arrival, residual, "residual>k_sigma"))

        removed_this_iter: list[RemovalRecord] = []
        for pick, arrival, residual, reason in to_remove:
            reject_pick(pick, arrival, reason, step=step)
            removed_this_iter.append(
                RemovalRecord(
                    event_id=eid,
                    pass_label=removal_pass_label,
                    iteration=iteration,
                    seed_id=pick.waveform_id.id,
                    phase=pick.phase_hint,
                    residual_s=residual,
                    reason=reason,
                )
            )
        removed_all.extend(removed_this_iter)

        history.append(
            IterationRecord(
                iteration=iteration,
                n_picks_in=n_picks_in,
                origin=origin,
                residuals=residuals_out,
                sigma_by_phase=sigma_by_phase,
                removed=removed_this_iter,
                nlloc_failed_reason=None,
            )
        )

        if not removed_this_iter:
            stabilized = True
            break

        removed_ids = {id(pick) for pick, _, _, _ in to_remove}
        current = [p for p in current if id(p) not in removed_ids]

    return IterateResult(
        origin=last_good_origin,
        accepted_picks=current,
        removed=removed_all,
        history=history,
        n_iterations=iteration,
        stabilized=stabilized,
        last_nlloc_failed_reason=last_nlloc_failed_reason,
    )


def run_pass_a(event: Event, paths: RunPaths, cfg: Config) -> PassAResult:
    """Pass A: iterative P-only location and residual filtering.

    Starts from every ``phase_hint == "P"`` pick on ``event`` that is not
    already ``evaluation_status == "rejected"`` (defensive, for idempotent
    re-runs -- M4.4's output should have none rejected yet), then iterates
    :func:`iterate_until_stable` up to ``cfg.nlloc.filter.p_only.max_iter``
    times against ``cfg.nlloc.filter.p_only``'s thresholds. After the loop,
    rejects the *event* (by returning ``status="rejected"``, never by
    calling ``openmeq.catalog.reject_event`` here -- M6.4 owns all final
    event-rejection bookkeeping in one place, per the issue) if fewer than
    ``cfg.nlloc.filter.events.min_p_phases`` picks survived.

    Parameters
    ----------
    event : obspy.core.event.Event
        Must already carry its PhaseNet (or hand-built) picks in
        ``event.picks``.
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config

    Returns
    -------
    PassAResult
    """
    initial_picks = [
        p for p in event.picks if p.phase_hint == "P" and p.evaluation_status != "rejected"
    ]
    p_only = cfg.nlloc.filter.p_only

    result = iterate_until_stable(
        event=event,
        picks=initial_picks,
        paths=paths,
        cfg=cfg,
        pass_name="passA",
        removal_pass_label="A",
        phase_thresholds={"P": p_only},
        max_iter=p_only.max_iter,
        step="locate/passA",
    )

    min_p_phases = cfg.nlloc.filter.events.min_p_phases
    if len(result.accepted_picks) < min_p_phases:
        status: Literal["ok", "rejected"] = "rejected"
        reason: str | None = result.last_nlloc_failed_reason or "min_p_phases"
    else:
        status = "ok"
        reason = None

    return PassAResult(
        origin=result.origin,
        accepted_picks=result.accepted_picks,
        removed=result.removed,
        n_iterations=result.n_iterations,
        status=status,
        reason=reason,
        history=result.history,
    )
