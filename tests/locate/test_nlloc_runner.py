"""Acceptance tests for M6.1 / OpenMEQ-xdf.1: ``openmeq.locate.nlloc_runner``.

Pick catalogues are built by hand here -- one `obspy.core.event.Pick` per
truth arrival (`tests/data/make_synthetic.py`'s ``truth["events"][i]
["arrivals"]``), following ARCHITECTURE.md §5's convention (P on the
station's Z channel, S on the first horizontal in N/E/1/2 priority order)
-- rather than through PhaseNet picking (M4.x), which is being built
concurrently and is out of scope for this task (per the issue's
orchestrator note).

``nlloc_grids`` (``tests/locate/conftest.py``, session-scoped) builds the
synthetic 8-station network's NLLoc grids exactly once for the whole
session via ``openmeq.nlloc_setup.setup_nlloc``.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pytest
from obspy.core.event import Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import make_event
from openmeq.locate.nlloc_runner import DuplicatePickError, NLLocFailed, locate_event
from openmeq.parallel import ParallelSettings, pmap

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import make_synthetic  # noqa: E402

pytestmark = pytest.mark.requires_nlloc

#: station code -> (vertical, horizontal1, horizontal2) channel codes, per
#: tests/data/make_synthetic.py's STATIONS constant. The tuple order
#: already matches ARCHITECTURE.md §5's S-channel priority (N before E, 1
#: before 2), so channels[1] is always the correct S channel.
_CHANNELS_BY_STATION = {
    code: channels for code, _kind, _x, _y, _elev_m, _depth_m, channels in make_synthetic.STATIONS
}

#: Uncertainty (s) given to every hand-built pick: small and uniform,
#: matching a "clean (noise-free) pick set" per the acceptance criterion.
_CLEAN_UNCERTAINTY_S = 0.02


def _build_clean_picks(arrivals: dict) -> list[Pick]:
    """One P pick (Z channel) and one S pick (first horizontal) per station
    in ``arrivals`` (``truth["events"][i]["arrivals"]``, keyed by "NET.STA")."""
    picks = []
    for station_key, times in arrivals.items():
        network, station = station_key.split(".")
        channels = _CHANNELS_BY_STATION[station]
        vert, horiz1 = channels[0], channels[1]
        for phase, channel in (("P", vert), ("S", horiz1)):
            pick = Pick()
            pick.time = times[phase]
            pick.waveform_id = WaveformStreamID(
                network_code=network,
                station_code=station,
                location_code="00",
                channel_code=channel,
            )
            pick.phase_hint = phase
            pick.time_errors.uncertainty = _CLEAN_UNCERTAINTY_S
            pick.onset = "impulsive"
            pick.evaluation_mode = "automatic"
            picks.append(pick)
    return picks


def _total_error_km(origin, truth_origin: dict) -> tuple[float, float, float]:
    horiz_km = (
        gps2dist_azimuth(
            truth_origin["lat"], truth_origin["lon"], origin.latitude, origin.longitude
        )[0]
        / 1000.0
    )
    depth_diff_km = abs(origin.depth / 1000.0 - truth_origin["depth_km"])
    total_km = (horiz_km**2 + depth_diff_km**2) ** 0.5
    return total_km, horiz_km, depth_diff_km


def _pick_a_well_covered_event(truth: dict) -> dict:
    """A deterministic, not-pathologically-shallow event for the single-event
    strict-tolerance tests below.

    Per the M5.2 finding (this issue's orchestrator note): some shallow
    synthetic events have a genuine depth/origin-time trade-off, so a
    strict per-event tolerance is only checked against an event picked to
    avoid that; :func:`test_batch_locate_events_recovery_and_throughput`
    below judges the *rest* of the catalogue statistically instead.
    """
    candidates = [ev for ev in truth["events"] if ev["origin"]["depth_km"] >= 2.0]
    assert candidates, "expected at least one non-shallow synthetic event"
    return candidates[0]


def test_locate_event_recovers_true_hypocenter_clean_picks(nlloc_grids):
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)

    event = make_event(ev["event_id_hint"])
    picks = _build_clean_picks(ev["arrivals"])

    origin = locate_event(event, picks, paths, cfg, "passA", 1)

    total_km, horiz_km, depth_km = _total_error_km(origin, ev["origin"])
    assert total_km < 3.0, (
        f"recovered location {total_km:.3f} km from truth "
        f"(horiz {horiz_km:.3f} km, depth diff {depth_km:.3f} km)"
    )


def test_locate_event_arrivals_resolve_to_original_picks(nlloc_grids):
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)

    event = make_event(ev["event_id_hint"])
    picks = _build_clean_picks(ev["arrivals"])

    origin = locate_event(event, picks, paths, cfg, "passA", 2)

    assert len(origin.arrivals) > 0
    for arrival in origin.arrivals:
        referred = arrival.pick_id.get_referred_object()
        assert referred is not None
        assert any(referred is p for p in picks), (
            "arrival.pick_id did not resolve to one of the original Pick objects"
        )


def test_locate_event_duplicate_picks_raises_before_subprocess(nlloc_grids, monkeypatch):
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)

    event = make_event(ev["event_id_hint"])
    picks = _build_clean_picks(ev["arrivals"])
    # Duplicate the first pick's (station_code, phase_hint) pair.
    picks.append(picks[0])

    def _boom(*_args, **_kwargs):
        raise AssertionError("NLLoc subprocess must not run for a duplicate pick set")

    monkeypatch.setattr("openmeq.locate.nlloc_runner.run", _boom)

    with pytest.raises(DuplicatePickError):
        locate_event(event, picks, paths, cfg, "passA", 3)


def test_locate_event_unsolvable_pick_set_raises_nllocfailed(nlloc_grids):
    cfg, paths, truth, _table = nlloc_grids
    ev = _pick_a_well_covered_event(truth)

    event = make_event(ev["event_id_hint"])
    all_picks = _build_clean_picks(ev["arrivals"])
    # Only P+S from a single station: well under LOCMETH's default
    # min_number_phases=4, so NLLoc rejects the event (0 located, exit 0,
    # no per-event .hyp file written).
    one_station = next(iter(_CHANNELS_BY_STATION))
    picks = [p for p in all_picks if p.waveform_id.station_code == one_station]
    assert 0 < len(picks) <= 2

    with pytest.raises(NLLocFailed):
        locate_event(event, picks, paths, cfg, "passA", 4)


def _locate_one(item):
    """Top-level, picklable ``pmap`` work function (openmeq.parallel's contract)."""
    event, picks, paths, cfg, pass_name, iteration = item
    return locate_event(event, picks, paths, cfg, pass_name, iteration)


@pytest.mark.slow
def test_batch_locate_events_recovery_and_throughput(nlloc_grids):
    """Locate every synthetic event in parallel via ``openmeq.parallel.pmap``
    (each worker runs NLLoc in its own ``iter_NN`` temp dir, with the
    shared grids read-only), then check recovery *statistically* across
    the batch (the M5.2 finding above) and report locations/second.
    """
    cfg, paths, truth, _table = nlloc_grids

    items = []
    for i, ev in enumerate(truth["events"]):
        event = make_event(ev["event_id_hint"])
        picks = _build_clean_picks(ev["arrivals"])
        items.append((event, picks, paths, cfg, "passA", 10 + i))

    pcfg = ParallelSettings(n_workers="auto", threads_per_worker=1)

    start = time.perf_counter()
    origins = pmap(_locate_one, items, pcfg, desc="locate_event throughput")
    elapsed = time.perf_counter() - start

    rate = len(items) / elapsed if elapsed > 0 else float("inf")
    print(
        f"\nlocate_event throughput: {len(items)} events in {elapsed:.2f}s "
        f"= {rate:.1f} locations/s (n_workers={pcfg.n_workers})"
    )

    total_errors = [
        _total_error_km(origin, ev["origin"])[0]
        for origin, ev in zip(origins, truth["events"], strict=True)
    ]
    median_err = statistics.median(total_errors)
    # Generous, statistical bound (ARCHITECTURE.md §8: expect a few hundred
    # m to ~1 km typical, with some shallow events trading depth for origin
    # time) -- not a per-event assertion.
    assert median_err < 3.0, (
        f"median total location error {median_err:.3f} km across {len(items)} events"
    )
    assert max(total_errors) < 10.0, f"worst-case total location error {max(total_errors):.3f} km"
