"""Acceptance tests for M6.2 / OpenMEQ-xdf.2: ``openmeq.locate.pass_a``.

Pick catalogues are built by hand here (following ARCHITECTURE.md §5's
convention: P on the station's Z channel), from ``tests/data/make_synthetic
.py``'s truth arrivals -- same convention ``tests/locate/test_nlloc_runner
.py`` (M6.1) uses. Because the synthetic picks are near-perfect, "bad" P
picks are injected deliberately (time-shifted by 0.3-1 s), and the tests
prove they get rejected while good ones survive.

``nlloc_grids`` (``tests/locate/conftest.py``, session-scoped) builds the
synthetic 8-station network's NLLoc grids exactly once for the whole
session. Tests that mutate ``cfg.nlloc.filter`` thresholds work on a
``model_copy(deep=True)`` so they never affect the shared, session-scoped
``cfg`` other tests in this file rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from obspy.core.event import Arrival, Origin, Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import make_event
from openmeq.locate.pass_a import RemovalRecord, run_pass_a

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import make_synthetic  # noqa: E402

pytestmark = pytest.mark.requires_nlloc

_CHANNELS_BY_STATION = {
    code: channels for code, _kind, _x, _y, _elev_m, _depth_m, channels in make_synthetic.STATIONS
}

_CLEAN_UNCERTAINTY_S = 0.02

#: Time shift (s) applied to a deliberately bad P pick -- comfortably above
#: both ``p_only.max_residual_s`` (0.10 s default) and any plausible
#: k-sigma bound, and within the "0.3-1 s" range the issue calls for.
_BAD_SHIFT_S = 0.5


def _p_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    vert_channel = _CHANNELS_BY_STATION[station][0]
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=vert_channel,
    )
    pick.phase_hint = "P"
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _pick_a_well_covered_event(truth: dict) -> dict:
    """Same convention as ``test_nlloc_runner.py``: avoid the shallow
    events with a genuine depth/origin-time trade-off (M5.2 finding)."""
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


def test_pass_a_removes_bad_pick_and_relocates(nlloc_grids):
    """One deliberately bad P pick (residual >> max_residual_s) is removed
    after iteration 1, and the returned origin is close to the truth using
    the remaining good picks."""
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])

    picks = []
    bad_station = None
    for station_key, times in ev["arrivals"].items():
        network, station = station_key.split(".")
        if bad_station is None:
            bad_station = station
            picks.append(_p_pick(network, station, times["P"] + _BAD_SHIFT_S))
        else:
            picks.append(_p_pick(network, station, times["P"]))
    event.picks = picks

    result = run_pass_a(event, paths, cfg)

    assert result.status == "ok"
    assert len(result.removed) == 1
    bad_record = result.removed[0]
    assert isinstance(bad_record, RemovalRecord)
    assert bad_record.phase == "P"
    assert bad_record.reason == "residual>max"
    assert bad_record.iteration == 1
    assert bad_record.event_id == ev["event_id_hint"]

    # The bad pick itself is flagged (never deleted, ARCHITECTURE.md §5).
    bad_pick = next(p for p in picks if p.waveform_id.station_code == bad_station)
    assert bad_pick.evaluation_status == "rejected"
    assert any("residual>max" in c.text for c in bad_pick.comments)
    assert bad_pick not in result.accepted_picks

    # Every other pick survives, unrejected.
    good_picks = [p for p in picks if p is not bad_pick]
    assert len(result.accepted_picks) == len(good_picks)
    for p in good_picks:
        assert p.evaluation_status != "rejected"

    assert result.origin is not None
    assert _total_error_km(result.origin, ev["origin"]) < 3.0


def test_pass_a_min_p_phases_rejection(nlloc_grids):
    """Fewer than ``min_p_phases`` good picks survive -> the event is
    reported rejected with reason "min_p_phases" (not an NLLoc failure)."""
    cfg, paths, truth, _table = nlloc_grids
    cfg = cfg.model_copy(deep=True)
    # A single iteration: the removal that drops us below min_p_phases must
    # be the *last* one attempted, or the next iteration's relocation with
    # too few phases would raise NLLocFailed instead (a different, and also
    # normal, termination path -- see
    # test_pass_a_nllocfailed_is_treated_as_zero_accepted_picks).
    cfg.nlloc.filter.p_only.max_iter = 1
    # Raised above the default (4) so that removing just the single bad
    # pick below (leaving 4 good ones) is still a shortfall. A *single*
    # deliberately bad pick, rather than several at once, keeps this
    # deterministic: with two or more simultaneously bad picks sharing a
    # geometry/azimuth, NLLoc's EDT_OT_WT can occasionally find a solution
    # that instead favours their (mutually consistent) shared bias over the
    # true location -- a real location-robustness edge case, but not the
    # one this test is about.
    cfg.nlloc.filter.events.min_p_phases = 5

    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])

    # 5 stations, well above NLLoc's own LOCMETH min_number_phases=4, so the
    # single iteration locates successfully; 1 of the 5 is deliberately bad,
    # leaving 4 good picks -- fewer than this test's min_p_phases (5).
    stations = ["STA01", "STA02", "STA03", "STA04", "STA05"]
    bad_station = "STA01"
    picks = []
    for station in stations:
        station_key = f"{make_synthetic.NETWORK}.{station}"
        times = ev["arrivals"][station_key]
        if station == bad_station:
            picks.append(_p_pick(make_synthetic.NETWORK, station, times["P"] + _BAD_SHIFT_S))
        else:
            picks.append(_p_pick(make_synthetic.NETWORK, station, times["P"]))
    event.picks = picks

    result = run_pass_a(event, paths, cfg)

    assert result.n_iterations == 1
    assert len(result.removed) == 1
    assert result.removed[0].reason == "residual>max"
    assert len(result.accepted_picks) == 4
    assert result.status == "rejected"
    assert result.reason == "min_p_phases"
    # The last successful origin is still kept for M6.4/M6.5 to report.
    assert result.origin is not None


def test_pass_a_clean_event_stops_after_one_iteration(nlloc_grids, monkeypatch):
    """A clean event (no bad picks) stabilizes after exactly 1 iteration."""
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])

    picks = [
        _p_pick(*station_key.split("."), times["P"])
        for station_key, times in ev["arrivals"].items()
    ]
    event.picks = picks

    from openmeq.locate.nlloc_runner import locate_event as real_locate_event

    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_locate_event(*args, **kwargs)

    monkeypatch.setattr("openmeq.locate.pass_a.locate_event", _spy)

    result = run_pass_a(event, paths, cfg)

    assert len(calls) == 1
    assert result.n_iterations == 1
    assert result.removed == []
    assert result.status == "ok"
    assert len(result.accepted_picks) == len(picks)


def test_pass_a_max_iter_reached_without_stabilizing(nlloc_grids, monkeypatch):
    """If every iteration keeps removing something, the loop still returns
    the last origin/pick set once ``max_iter`` is hit -- not an exception."""
    cfg, paths, truth, _table = nlloc_grids
    cfg = cfg.model_copy(deep=True)
    cfg.nlloc.filter.p_only.max_iter = 3

    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])

    picks = [
        _p_pick(*station_key.split("."), times["P"])
        for station_key, times in list(ev["arrivals"].items())[:6]
    ]
    event.picks = picks

    def _fake_locate_event(_event, picks_to_use, _paths, _cfg, _pass_name, _iteration):
        origin = Origin(time=picks_to_use[0].time, latitude=0.0, longitude=0.0, depth=0.0)
        for i, pick in enumerate(picks_to_use):
            # Always condemn whichever pick is currently first -- guarantees
            # a removal every iteration, so the loop never stabilizes
            # before max_iter.
            residual = 999.0 if i == 0 else 0.0
            origin.arrivals.append(Arrival(pick_id=pick.resource_id, time_residual=residual))
        return origin

    monkeypatch.setattr("openmeq.locate.pass_a.locate_event", _fake_locate_event)

    result = run_pass_a(event, paths, cfg)

    assert result.n_iterations == 3
    assert len(result.removed) == 3
    assert len(result.accepted_picks) == len(picks) - 3
    assert result.origin is not None
    # No exception escaped, and a normal (non-stabilized) termination.
    assert result.status in ("ok", "rejected")


def test_pass_a_nllocfailed_is_treated_as_zero_accepted_picks(nlloc_grids, monkeypatch):
    """``NLLocFailed`` is treated as "0 accepted picks this iteration" and
    surfaces as a "nlloc_failed:..." reason rather than the generic
    "min_p_phases" string."""
    from openmeq.locate.nlloc_runner import NLLocFailed

    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)
    event = make_event(ev["event_id_hint"])
    picks = [
        _p_pick(*station_key.split("."), times["P"])
        for station_key, times in list(ev["arrivals"].items())[:5]
    ]
    event.picks = picks

    def _boom(_event, _picks_to_use, _paths, _cfg, pass_name, iteration):
        raise NLLocFailed(ev["event_id_hint"], pass_name, iteration, "synthetic failure")

    monkeypatch.setattr("openmeq.locate.pass_a.locate_event", _boom)

    result = run_pass_a(event, paths, cfg)

    assert result.n_iterations == 1
    assert result.accepted_picks == []
    assert result.origin is None
    assert result.status == "rejected"
    assert result.reason is not None
    assert result.reason.startswith("nlloc_failed:")
