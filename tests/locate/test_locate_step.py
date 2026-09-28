"""Acceptance test for M6.4 / OpenMEQ-xdf.4: ``openmeq.locate.locate_step.run_locate``.

Builds a hand-crafted ``pick/catalog_picks.xml`` (three events, following the
same hand-built-pick convention as ``tests/locate/test_pass_a.py``/
``test_pass_b.py``, from ``tests/data/make_synthetic.py``'s truth arrivals)
covering the three outcomes ``run_locate`` must produce correctly in one
end-to-end run:

- a clean event -> accepted;
- an event with injected bad P picks that drops it below ``min_p_phases``
  during pass A -> rejected by pass A, before pass B ever runs;
- an event with only P picks (no S) -> passes pass A/B but fails M6.4's own
  ``min_s_phases`` event criterion.

Uses the session-scoped ``nlloc_grids`` fixture (``tests/locate/conftest.py``)
for the NLLoc grids; the synthetic project's own generated config already
sets ``parallel.n_workers: 1`` (``tests/data/make_synthetic.py``), so
``openmeq.parallel.pmap`` runs serially/in-process here -- no multiprocessing
pickling to debug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from obspy import Catalog
from obspy.core.event import Pick, WaveformStreamID

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import make_event, read_catalog, write_catalog
from openmeq.locate.finalize import REMOVAL_LOG_COLUMNS, SUMMARY_COLUMNS
from openmeq.locate.locate_step import run_locate

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import make_synthetic  # noqa: E402

pytestmark = pytest.mark.requires_nlloc

_CHANNELS_BY_STATION = {
    code: channels for code, _kind, _x, _y, _elev_m, _depth_m, channels in make_synthetic.STATIONS
}

_CLEAN_UNCERTAINTY_S = 0.02
_BAD_SHIFT_S = 0.5


def _p_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=_CHANNELS_BY_STATION[station][0],
    )
    pick.phase_hint = "P"
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _s_pick(network: str, station: str, time, uncertainty_s: float = _CLEAN_UNCERTAINTY_S) -> Pick:
    pick = Pick()
    pick.time = time
    pick.waveform_id = WaveformStreamID(
        network_code=network,
        station_code=station,
        location_code="00",
        channel_code=_CHANNELS_BY_STATION[station][1],
    )
    pick.phase_hint = "S"
    pick.time_errors.uncertainty = uncertainty_s
    pick.onset = "impulsive"
    pick.evaluation_mode = "automatic"
    return pick


def _well_covered_events(truth: dict, n: int) -> list[dict]:
    candidates = [ev for ev in truth["events"] if ev["origin"]["depth_km"] >= 2.0]
    assert len(candidates) >= n
    return candidates[:n]


def test_run_locate_end_to_end(nlloc_grids):
    cfg, paths, truth, _table = nlloc_grids
    ev_clean, ev_bad, ev_no_s = _well_covered_events(truth, 3)

    # Shared, per-run thresholds: max_iter=1 (deterministic single-iteration
    # residual filtering, matching test_pass_a.py's own
    # test_pass_a_min_p_phases_rejection convention) and min_p_phases raised
    # to 5 so ev_bad's single injected bad pick (5 stations, 1 bad -> 4
    # survivors) is rejected by pass A.
    cfg = cfg.model_copy(deep=True)
    cfg.nlloc.filter.p_only.max_iter = 1
    cfg.nlloc.filter.p_and_s.max_iter = 1
    cfg.nlloc.filter.events.min_p_phases = 5

    # -- Event 1: clean, full 8-station P+S network -> accepted.
    event_clean = make_event(ev_clean["event_id_hint"])
    picks_clean = []
    for station_key, times in ev_clean["arrivals"].items():
        network, station = station_key.split(".")
        picks_clean.append(_p_pick(network, station, times["P"]))
        picks_clean.append(_s_pick(network, station, times["S"]))
    event_clean.picks = picks_clean

    # -- Event 2: 5 stations, 1 deliberately bad P pick -> pass A rejects
    # (min_p_phases=5, 4 survive after the bad one is dropped).
    event_bad = make_event(ev_bad["event_id_hint"])
    stations = list(ev_bad["arrivals"].keys())[:5]
    picks_bad = []
    for i, station_key in enumerate(stations):
        network, station = station_key.split(".")
        times = ev_bad["arrivals"][station_key]
        p_time = times["P"] + _BAD_SHIFT_S if i == 0 else times["P"]
        picks_bad.append(_p_pick(network, station, p_time))
    event_bad.picks = picks_bad

    # -- Event 3: P-only (no S picks at all) -> clears pass A/B but fails
    # M6.4's own min_s_phases criterion (default 1).
    event_no_s = make_event(ev_no_s["event_id_hint"])
    picks_no_s = []
    for station_key, times in ev_no_s["arrivals"].items():
        network, station = station_key.split(".")
        picks_no_s.append(_p_pick(network, station, times["P"]))
    event_no_s.picks = picks_no_s

    input_catalog = Catalog(events=[event_clean, event_bad, event_no_s])
    write_catalog(input_catalog, paths.catalog_picks_xml)

    run_locate(paths, cfg)

    assert paths.catalog_auto_xml.exists()
    assert paths.locate_summary_csv.exists()
    assert paths.removal_log_csv.exists()

    out_catalog = read_catalog(paths.catalog_auto_xml)
    assert len(out_catalog) == 3

    summary = pd.read_csv(paths.locate_summary_csv)
    assert list(summary.columns) == SUMMARY_COLUMNS

    input_ids = {ev["event_id_hint"] for ev in (ev_clean, ev_bad, ev_no_s)}
    assert set(summary["event_id"]) == input_ids
    # Reconciling row counts: every input event_id appears exactly once.
    assert summary["event_id"].value_counts().max() == 1

    by_id = {row.event_id: row for row in summary.itertuples()}

    clean_row = by_id[ev_clean["event_id_hint"]]
    assert clean_row.status == "accepted"
    assert pd.isna(clean_row.reason) or clean_row.reason == ""
    assert clean_row.n_p == 8
    assert clean_row.n_s == 8

    bad_row = by_id[ev_bad["event_id_hint"]]
    assert bad_row.status == "rejected"
    assert bad_row.reason == "min_p_phases"

    no_s_row = by_id[ev_no_s["event_id_hint"]]
    assert no_s_row.status == "rejected"
    assert no_s_row.reason == "min_s_phases"
    assert no_s_row.n_s == 0

    # Every rejected event's summary.csv status matches its catalog's own
    # preferred-origin evaluation_status (ARCHITECTURE.md §5).
    events_by_id = {get_event_id(ev): ev for ev in out_catalog}
    for row in summary.itertuples():
        origin = events_by_id[row.event_id].preferred_origin()
        assert origin is not None
        is_rejected_in_catalog = origin.evaluation_status == "rejected"
        assert (row.status == "rejected") == is_rejected_in_catalog

    removal_log = pd.read_csv(paths.removal_log_csv)
    assert list(removal_log.columns) == REMOVAL_LOG_COLUMNS
    bad_removals = removal_log[removal_log["event_id"] == ev_bad["event_id_hint"]]
    assert len(bad_removals) == 1
    assert bad_removals.iloc[0]["reason"] == "residual>max"
    assert bad_removals.iloc[0]["phase"] == "P"
    assert not pd.isna(bad_removals.iloc[0]["residual_s"])
