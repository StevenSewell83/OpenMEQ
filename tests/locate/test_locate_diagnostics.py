"""Acceptance test for M6.5 / OpenMEQ-xdf.5: ``openmeq.locate.diagnostics``.

Runs the real ``openmeq.locate.locate_step.run_locate`` against a small
hand-built ``pick/catalog_picks.xml`` (same hand-built-pick convention as
``tests/locate/test_locate_step.py``, from ``tests/data/make_synthetic.py``'s
truth arrivals), covering a clean accepted event, an event pass A rejects
(``min_p_phases``), and an event pass B/``finalize`` rejects
(``min_s_phases``) -- so every pass's own "removed picks"/"removed events"
diagnostics have some real, non-empty content to draw, while still
exercising the "zero rejection/removal is valid" guard for whichever
individual reason/phase never fires.

Then calls :func:`openmeq.locate.diagnostics.make_locate_diagnostics`
directly on ``run_locate``'s own :class:`~openmeq.locate.locate_step.LocateRunResult`
and asserts every expected PNG + ``summary.txt`` exists and that
``summary.txt``'s counts are parseable (flat ``key: value`` lines).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from obspy import Catalog
from obspy.core.event import Pick, WaveformStreamID

from openmeq.catalog import make_event, write_catalog
from openmeq.locate.diagnostics import make_locate_diagnostics
from openmeq.locate.diagnostics_extended import STATION_RESIDUAL_COLUMNS, extended_dir
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


@pytest.fixture(scope="module")
def locate_diagnostics_result(nlloc_grids):
    """Runs ``run_locate`` once for this module: a clean event, a pass-A
    rejection, and a pass-B/finalize rejection (``min_s_phases``)."""
    cfg, paths, truth, _table = nlloc_grids
    ev_clean, ev_bad, ev_no_s = _well_covered_events(truth, 3)

    cfg = cfg.model_copy(deep=True)
    cfg.nlloc.filter.p_only.max_iter = 1
    cfg.nlloc.filter.p_and_s.max_iter = 1
    cfg.nlloc.filter.events.min_p_phases = 5

    event_clean = make_event(ev_clean["event_id_hint"])
    picks_clean = []
    for station_key, times in ev_clean["arrivals"].items():
        network, station = station_key.split(".")
        picks_clean.append(_p_pick(network, station, times["P"]))
        picks_clean.append(_s_pick(network, station, times["S"]))
    event_clean.picks = picks_clean

    event_bad = make_event(ev_bad["event_id_hint"])
    stations = list(ev_bad["arrivals"].keys())[:5]
    picks_bad = []
    for i, station_key in enumerate(stations):
        network, station = station_key.split(".")
        times = ev_bad["arrivals"][station_key]
        p_time = times["P"] + _BAD_SHIFT_S if i == 0 else times["P"]
        picks_bad.append(_p_pick(network, station, p_time))
    event_bad.picks = picks_bad

    event_no_s = make_event(ev_no_s["event_id_hint"])
    picks_no_s = []
    for station_key, times in ev_no_s["arrivals"].items():
        network, station = station_key.split(".")
        picks_no_s.append(_p_pick(network, station, times["P"]))
    event_no_s.picks = picks_no_s

    input_catalog = Catalog(events=[event_clean, event_bad, event_no_s])
    write_catalog(input_catalog, paths.catalog_picks_xml)

    result = run_locate(paths, cfg)
    return cfg, paths, result


def test_diagnostics_outputs_and_summary(locate_diagnostics_result):
    cfg, paths, result = locate_diagnostics_result

    diagnostics_dir = make_locate_diagnostics(paths, cfg, result)
    assert diagnostics_dir == paths.diagnostics_dir("locate")

    expected_pngs = []
    for pass_label in ("A", "B"):
        for phase in ("P", "S"):
            expected_pngs.append(f"pass{pass_label}_residuals_{phase}.png")
            expected_pngs.append(f"pass{pass_label}_residuals_{phase}_per_station.png")
        expected_pngs.append(f"pass{pass_label}_map_sections.png")
        expected_pngs.append(f"pass{pass_label}_rms_hist.png")
        expected_pngs.append(f"pass{pass_label}_gap_hist.png")
        expected_pngs.append(f"pass{pass_label}_err_h_hist.png")
        expected_pngs.append(f"pass{pass_label}_err_z_hist.png")
        expected_pngs.append(f"pass{pass_label}_removed_picks_by_reason.png")
        expected_pngs.append(f"pass{pass_label}_removed_events_by_reason.png")

    for name in expected_pngs:
        p = diagnostics_dir / name
        assert p.is_file(), f"{p} was not created"
        assert p.stat().st_size > 0

    summary_path = diagnostics_dir / "summary.txt"
    assert summary_path.is_file()

    metrics: dict[str, str] = {}
    for line in summary_path.read_text().splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        metrics[key.strip()] = value.strip()

    # Basic parseability + sanity of the required counts (work item #1:
    # "events in / located / rejected-by-reason counts, mean/median
    # residuals per pass").
    assert int(metrics["events_in"]) == 3
    assert int(metrics["events_accepted"]) == 1
    assert int(metrics["events_rejected"]) == 2
    assert int(metrics["rejected_reason_min_p_phases"]) == 1
    assert int(metrics["rejected_reason_min_s_phases"]) == 1

    assert int(metrics["passA_n_events_run"]) == 3
    assert int(metrics["passA_n_rejected"]) == 1
    assert int(metrics["passB_n_events_run"]) == 2  # pass B never runs for the pass-A rejection

    # Pass A's own P residual mean/median must be a parseable float (it
    # always has some P data on this fixture); pass A's S residuals are
    # expected to be empty (pass A only ever locates P) -- an empty string
    # is still valid, parseable "no data" per this module's own contract.
    float(metrics["passA_residual_P_mean_s"])
    float(metrics["passA_residual_P_median_s"])
    assert metrics["passA_residual_S_mean_s"] == ""
    assert metrics["passA_residual_S_median_s"] == ""

    assert int(metrics["total_picks_removed"]) >= 1


def test_extended_diagnostics_from_real_run(locate_diagnostics_result):
    """OpenMEQ-02o.5: real NLLoc arrivals populate ``PassDiagnostics.distances_km``
    for every residual, and enabling ``diagnostics.locate.extended`` writes the
    extended set (per-station CSV sane) alongside the baseline."""
    cfg, paths, result = locate_diagnostics_result

    for diag_map in (result.pass_a, result.pass_b):
        for d in diag_map.values():
            assert set(d.distances_km) == set(d.residuals)
            assert all(0.0 <= km < 1000.0 for km in d.distances_km.values())
    assert any(d.distances_km for d in result.pass_b.values())

    cfg = cfg.model_copy(deep=True)
    cfg.diagnostics.locate.extended = True
    make_locate_diagnostics(paths, cfg, result)
    out = extended_dir(paths)
    for pass_label in ("A", "B"):
        for name in (
            "res_vs_distance.png",
            "station_res_map.png",
            "station_residuals.csv",
            "depth_hist_grid.png",
        ):
            assert (out / f"pass{pass_label}_{name}").is_file()

    table = pd.read_csv(out / "passB_station_residuals.csv")
    assert list(table.columns) == STATION_RESIDUAL_COLUMNS
    assert set(table["phase"]) == {"P", "S"}
    assert (table["n"] >= 1).all()
    assert table["mean"].abs().max() < 1.0


def test_diagnostics_zero_rejection_does_not_raise(nlloc_grids):
    """A single clean, fully-accepted event: zero pick removals, zero event
    rejections -- must not raise (work item #1's own "zero-rejection ...
    is an expected, valid outcome" guard).

    Reuses the shared, session-scoped ``nlloc_grids`` paths directly (rather
    than a fresh ``RunPaths`` over a different output directory, which would
    have no NLLoc grids under it) -- safe because ``run_locate``/
    ``make_locate_diagnostics`` fully rewrite ``pick/catalog_picks.xml`` and
    every ``locate/`` output from the single event built here before this
    test reads any of them back, exactly like every other test sharing this
    fixture (``tests/locate/test_locate_step.py``).
    """
    cfg, paths, truth, _table = nlloc_grids
    (ev_clean,) = _well_covered_events(truth, 1)

    cfg = cfg.model_copy(deep=True)

    event_clean = make_event(ev_clean["event_id_hint"])
    picks = []
    for station_key, times in ev_clean["arrivals"].items():
        network, station = station_key.split(".")
        picks.append(_p_pick(network, station, times["P"]))
        picks.append(_s_pick(network, station, times["S"]))
    event_clean.picks = picks

    write_catalog(Catalog(events=[event_clean]), paths.catalog_picks_xml)

    result = run_locate(paths, cfg)
    diagnostics_dir = make_locate_diagnostics(paths, cfg, result)

    summary_path = diagnostics_dir / "summary.txt"
    assert summary_path.is_file()
    text = summary_path.read_text()
    assert "events_rejected: 0" in text
    assert "total_picks_removed: 0" in text
