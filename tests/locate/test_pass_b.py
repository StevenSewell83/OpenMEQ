"""Acceptance tests for M6.3 / OpenMEQ-xdf.3: ``openmeq.locate.pass_b``.

Same conventions as ``tests/locate/test_pass_a.py``: pick catalogues are
built by hand from ``tests/data/make_synthetic.py``'s truth arrivals, P on
the station's Z channel, S on its first horizontal (ARCHITECTURE.md §5).
Because the synthetic picks are near-perfect, deliberately bad picks are
injected to exercise the S-prediction screen and the P/S residual filters --
**one bad pick per phase per test** (never two or more simultaneously, per
M6.2's own finding, referenced by this issue's notes: several
simultaneously-bad, mutually-consistent picks can pull the solution instead
of being individually rejected).

``nlloc_grids`` (``tests/locate/conftest.py``, session-scoped) builds the
synthetic network's NLLoc grids once for the whole session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from obspy.core.event import Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import make_event
from openmeq.locate.pass_a import RemovalRecord, run_pass_a
from openmeq.locate.pass_b import PassBResult, run_pass_b, run_pass_b_many

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import make_synthetic  # noqa: E402

pytestmark = pytest.mark.requires_nlloc

_CHANNELS_BY_STATION = {
    code: channels for code, _kind, _x, _y, _elev_m, _depth_m, channels in make_synthetic.STATIONS
}

_CLEAN_UNCERTAINTY_S = 0.02

#: Late shift applied to the deliberately-bad S pick in the screening test --
#: comfortably above ``s_predicted_window_s`` (default 0.3 s).
_BAD_S_SCREEN_SHIFT_S = 1.0


def _pick(network: str, station: str, phase: str, channel: str, time, uncertainty_s: float) -> Pick:
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=channel,
    )
    pick.phase_hint = phase
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _p_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    return _pick(network, station, "P", _CHANNELS_BY_STATION[station][0], time, uncertainty_s)


def _s_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    return _pick(network, station, "S", _CHANNELS_BY_STATION[station][1], time, uncertainty_s)


def _pick_a_well_covered_event(truth: dict) -> dict:
    """Same convention as ``test_nlloc_runner.py``/``test_pass_a.py``: avoid
    the shallow events with a genuine depth/origin-time trade-off (M5.2
    finding)."""
    candidates = [ev for ev in truth["events"] if ev["origin"]["depth_km"] >= 2.0]
    assert candidates, "expected at least one non-shallow synthetic event"
    return candidates[0]


def _total_error_km(origin, truth_origin: dict) -> float:
    horiz_km = (
        gps2dist_azimuth(
            truth_origin["lat"], truth_origin["lon"], origin.latitude, origin.longitude
        )[0]
        / 1000.0
    )
    depth_diff_km = abs(origin.depth / 1000.0 - truth_origin["depth_km"])
    return (horiz_km**2 + depth_diff_km**2) ** 0.5


def _reset_picks(picks: list[Pick]) -> None:
    """Clear any rejection flagged by a previous trial run (see
    ``test_pass_b_filters_p_and_s_residuals_independently``'s calibration
    loop, which reuses the same `Pick` objects across several trials)."""
    for p in picks:
        p.evaluation_status = None
        p.comments = []


def _clean_p_and_s_picks(ev: dict) -> list[Pick]:
    picks = []
    for station_key, times in ev["arrivals"].items():
        network, station = station_key.split(".")
        picks.append(_p_pick(network, station, times["P"]))
        picks.append(_s_pick(network, station, times["S"]))
    return picks


def test_pass_b_screens_late_s_pick_before_residual_loop(nlloc_grids):
    """One deliberately late S pick (1 s past the S time predicted from the
    pass-A origin) is screened out with reason "s_predicted_window", and
    never reaches the residual-filter loop at all."""
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])

    picks = _clean_p_and_s_picks(ev)
    event.picks = picks

    bad_station_key = next(iter(ev["arrivals"]))
    bad_network, bad_station = bad_station_key.split(".")
    bad_s_pick = next(
        p for p in picks if p.phase_hint == "S" and p.waveform_id.station_code == bad_station
    )
    bad_s_pick.time = bad_s_pick.time + _BAD_S_SCREEN_SHIFT_S

    pass_a_result = run_pass_a(event, paths, cfg)
    assert pass_a_result.status == "ok"

    result = run_pass_b(event, pass_a_result, paths, cfg)

    assert isinstance(result, PassBResult)
    screening_records = [r for r in result.removed if r.reason == "s_predicted_window"]
    assert len(screening_records) == 1
    bad_record = screening_records[0]
    assert isinstance(bad_record, RemovalRecord)
    assert bad_record.phase == "S"
    assert bad_record.pass_label == "B"
    assert bad_record.iteration == 0  # screened before the relocate-and-filter loop.
    assert bad_record.event_id == ev["event_id_hint"]
    assert bad_record.seed_id == bad_s_pick.waveform_id.id

    # The pick itself is flagged (never deleted, ARCHITECTURE.md §5), with
    # no Arrival available yet at this stage (see openmeq.catalog.reject_pick's
    # now-optional arrival parameter).
    assert bad_s_pick.evaluation_status == "rejected"
    assert any("s_predicted_window" in c.text for c in bad_s_pick.comments)
    assert bad_s_pick not in result.accepted_s

    # It never reached the residual-filter loop: it never appears in any
    # iteration's residuals (it was never handed to locate_event at all).
    for iteration_record in result.history:
        assert bad_s_pick.waveform_id.id not in iteration_record.residuals

    # Every other S pick, and every P pick, survives.
    good_s_picks = [p for p in picks if p.phase_hint == "S" and p is not bad_s_pick]
    assert len(result.accepted_s) == len(good_s_picks)
    p_picks = [p for p in picks if p.phase_hint == "P"]
    assert len(result.accepted_p) == len(p_picks)
    assert result.status == "ok"


def test_pass_b_uses_s_picks_and_converges_near_truth(nlloc_grids):
    """A clean P+S pick set converges close to the true location, and the
    pass-B origin measurably differs from the pass-A-only (P-only) origin --
    confirming the S picks were actually used."""
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])
    event.picks = _clean_p_and_s_picks(ev)

    pass_a_result = run_pass_a(event, paths, cfg)
    assert pass_a_result.status == "ok"
    assert pass_a_result.origin is not None

    result = run_pass_b(event, pass_a_result, paths, cfg)

    assert result.status == "ok"
    assert result.origin is not None
    assert _total_error_km(result.origin, ev["origin"]) < 3.0
    assert len(result.accepted_s) >= 4  # S picks genuinely made it into the solve.

    pass_a_origin = pass_a_result.origin
    pass_b_origin = result.origin
    assert (
        pass_a_origin.latitude,
        pass_a_origin.longitude,
        pass_a_origin.depth,
        pass_a_origin.time,
    ) != (
        pass_b_origin.latitude,
        pass_b_origin.longitude,
        pass_b_origin.depth,
        pass_b_origin.time,
    )


def test_pass_b_filters_p_and_s_residuals_independently(nlloc_grids):
    """An S residual that would fail the (tighter) P threshold, but passes
    the (looser) configured S threshold, survives -- P and S thresholds are
    applied independently.

    The shift that lands an S pick's *actual* NLLoc residual strictly
    between the two configured thresholds is found empirically at test
    time (a handful of trial relocations), rather than assumed from a
    hardcoded "this shift produces that residual" arithmetic guess: this
    project's synthetic travel times are currently straight-ray, but a
    concurrent task (OpenMEQ-m1f.5) is replacing them with exact layered
    ray tracing, and the pass-B iterative solve's own origin-time trade-off
    means the resulting residual is not simply "the applied shift" anyway
    (confirmed empirically while writing this test: e.g. a 0.12 s shift on
    this event/station produced a ~0.07 s residual, not 0.12 s, because
    NLLoc's ``EDT_OT_WT`` partially absorbs a single outlying pick's shift
    into the fitted origin time).
    """
    cfg, paths, truth, _table = nlloc_grids
    p_and_s = cfg.nlloc.filter.p_and_s
    lower, upper = p_and_s.p.max_residual_s, p_and_s.s.max_residual_s
    assert lower < upper
    margin = (upper - lower) * 0.15
    target_low, target_high = lower + margin, upper - margin
    assert target_low < target_high, "p/s thresholds too close together for this test's margin"

    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])
    picks = _clean_p_and_s_picks(ev)
    event.picks = picks

    bad_station_key = next(iter(ev["arrivals"]))
    _bad_network, bad_station = bad_station_key.split(".")
    bad_s_pick = next(
        p for p in picks if p.phase_hint == "S" and p.waveform_id.station_code == bad_station
    )
    base_time = bad_s_pick.time

    pass_a_result = run_pass_a(event, paths, cfg)
    assert pass_a_result.status == "ok"

    # All of these candidate shifts stay well inside the S-prediction
    # screening window (default 0.3 s), so the pick always survives
    # screening and reaches the residual-filter loop.
    candidate_fractions = (0.50, 0.65, 0.35, 0.80, 0.40, 0.70, 0.55, 0.85, 0.30, 0.60, 0.75, 0.45)
    found = None
    for frac in candidate_fractions:
        shift_s = lower + frac * (upper - lower)
        assert shift_s < p_and_s.s_predicted_window_s
        _reset_picks(picks)
        bad_s_pick.time = base_time + shift_s
        trial = run_pass_b(event, pass_a_result, paths, cfg)
        if trial.status != "ok" or not trial.history:
            continue
        residual = trial.history[0].residuals.get(bad_s_pick.waveform_id.id)
        if residual is None:
            continue
        if target_low < abs(residual) < target_high:
            found = (shift_s, residual, trial)
            break

    assert found is not None, (
        "could not empirically find an S-pick time shift whose resulting residual lands "
        f"between the P ({lower}s) and S ({upper}s) thresholds for this event/station"
    )
    _shift_s, _residual, result = found

    assert result.status == "ok"
    assert bad_s_pick.evaluation_status != "rejected"
    assert bad_s_pick in result.accepted_s
    assert not any(r.seed_id == bad_s_pick.waveform_id.id for r in result.removed)


def test_pass_b_skips_already_rejected_event(nlloc_grids):
    """Pass B never runs on an event pass A already rejected -- it returns
    immediately with the same status/reason."""
    cfg, paths, truth, _table = nlloc_grids
    cfg = cfg.model_copy(deep=True)
    cfg.nlloc.filter.p_only.max_iter = 1
    cfg.nlloc.filter.events.min_p_phases = 999  # guarantee a "rejected" pass A.

    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])
    event.picks = _clean_p_and_s_picks(ev)

    pass_a_result = run_pass_a(event, paths, cfg)
    assert pass_a_result.status == "rejected"

    result = run_pass_b(event, pass_a_result, paths, cfg)

    assert result.status == "rejected"
    assert result.reason == pass_a_result.reason
    assert result.origin is pass_a_result.origin
    assert result.accepted_s == []
    assert result.removed == []
    assert result.n_iterations == 0
    assert result.history == []


def test_run_pass_b_many_processes_multiple_events(nlloc_grids):
    """``run_pass_b_many`` wires ``run_pass_b`` through ``openmeq.parallel.pmap``
    over several events, keyed by event id."""
    cfg, paths, truth, _table = nlloc_grids
    cfg = cfg.model_copy(deep=True)
    cfg.parallel.n_workers = 1  # serial, in-process (ARCHITECTURE.md's pmap note).

    candidates = [ev for ev in truth["events"] if ev["origin"]["depth_km"] >= 2.0][:2]
    assert len(candidates) == 2, "expected at least 2 non-shallow synthetic events"

    events = []
    pass_a_results = {}
    for ev in candidates:
        event = make_event(ev["event_id_hint"])
        event.picks = _clean_p_and_s_picks(ev)
        events.append(event)
        pass_a_results[ev["event_id_hint"]] = run_pass_a(event, paths, cfg)

    results = run_pass_b_many(events, pass_a_results, paths, cfg)

    assert set(results.keys()) == {ev["event_id_hint"] for ev in candidates}
    for result in results.values():
        assert isinstance(result, PassBResult)
        assert result.status == "ok"
