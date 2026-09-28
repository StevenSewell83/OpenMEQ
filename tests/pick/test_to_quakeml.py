"""Tests for ``openmeq.pick.to_quakeml`` (M4.4, ``OpenMEQ-1m7.4``).

Two layers:

- **Unit tests** (no PhaseNet needed): a hand-built ``picks.csv`` (with a
  deliberate probability-based duplicate, mirroring M4.3's own "a station
  may get several P or S picks" finding) plus a hand-built
  ``chunks_pn/<event_id>/<event_id>.mseed`` (one oriented ``Z/N/E`` station,
  one unoriented ``Z/1/2`` station, matching ``openmeq.pick.prep``'s two
  supported conventions) exercise :func:`~openmeq.pick.to_quakeml.build_catalog`
  directly -- fast, run every time.
- **Integration test** (``requires_phasenet``): real PhaseNet picks over a
  handful of the synthetic dataset's events, converted end to end,
  mirroring ``tests/pick/test_run_phasenet.py``'s own ``phasenet_world`` /
  ``_prepare_synthetic_chunks`` pattern (hand-built detections at each
  event's true origin time, so ``event_id == event_id_hint`` by
  construction -- this module's own concern is the *conversion*, not
  detection-trigger-time recovery, which ``tests/integration/
  test_detect_integration.py`` already covers, and
  ``tests/pick/test_recovery.py`` covers again end to end for the pick
  step specifically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import get_extra, read_catalog
from openmeq.chunk import write_chunks
from openmeq.config import load_config
from openmeq.index import build_index
from openmeq.paths import RunPaths
from openmeq.pick.prep import prepare_all
from openmeq.pick.run_phasenet import PICKS_CSV_COLUMNS, run_phasenet
from openmeq.pick.to_quakeml import (
    PHASENET_METHOD_ID,
    S_PICK_CHANNEL_PREFERENCE,
    build_catalog,
    write_catalog_picks,
)
from openmeq.stations import station_table as compute_station_table

DETECTIONS_COLUMNS = [
    "event_id",
    "time",
    "duration_s",
    "coincidence_sum",
    "n_stations",
    "stations",
    "trace_ids",
]


def _phasenet_available() -> bool:
    from openmeq.external import phasenet_dir, phasenet_model_dir

    try:
        model_dir = phasenet_model_dir()
        predict_py = phasenet_dir() / "phasenet" / "predict.py"
    except Exception:
        return False
    return predict_py.is_file() and (model_dir / "checkpoint").is_file()


skip_reason = (
    "external/PhaseNet clone/model not found (set OPENMEQ_EXTERNAL_DIR, or run install.sh)"
)


# ---------------------------------------------------------------------------
# Unit-test fixtures: a hand-built chunks_pn/ + picks.csv
# ---------------------------------------------------------------------------

_START = UTCDateTime(2025, 6, 1, 0, 0, 0)
_NPTS = 6000  # 60 s @ 100 Hz
_SR = 100.0


def _make_trace(network, station, location, channel, start=_START) -> Trace:
    rng = np.random.default_rng(abs(hash((network, station, location, channel))) % (2**32))
    data = rng.normal(scale=1.0, size=_NPTS).astype(np.float64)
    return Trace(
        data=data,
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "starttime": start,
            "sampling_rate": _SR,
        },
    )


@pytest.fixture
def hand_built_project(tmp_path, synthetic_project):
    """A private project with a hand-built ``chunks_pn/<event_id>/<event_id>.mseed``
    for one event holding two stations: ``XX.ORNT`` (oriented ``HHZ/HHN/HHE``)
    and ``XX.UNOR`` (unoriented ``DPZ/DP1/DP2``) -- exercises both of
    :func:`openmeq.pick.prep`'s supported component conventions without
    needing a real PhaseNet run.
    """
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    event_id = "ev20250601T000000_000"
    stream = Stream(
        [
            _make_trace("XX", "ORNT", "00", "HHZ"),
            _make_trace("XX", "ORNT", "00", "HHN"),
            _make_trace("XX", "ORNT", "00", "HHE"),
            _make_trace("XX", "UNOR", "00", "DPZ"),
            _make_trace("XX", "UNOR", "00", "DP1"),
            _make_trace("XX", "UNOR", "00", "DP2"),
        ]
    )
    chunk_pn_path = paths.chunk_pn_file(event_id)
    chunk_pn_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(chunk_pn_path), format="MSEED")

    return cfg, paths, event_id


def _write_picks_csv(paths: RunPaths, rows: list[dict]) -> str:
    df = pd.DataFrame(rows, columns=PICKS_CSV_COLUMNS)
    out_path = paths.phasenet_raw_dir / "picks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def _pick_row(station_id, phase_type, phase_time, phase_score, file_name) -> dict:
    return {
        "station_id": station_id,
        "begin_time": str(_START),
        "phase_index": 0,
        "phase_time": str(phase_time),
        "phase_score": phase_score,
        "phase_type": phase_type,
        "file_name": file_name,
        "phase_amplitude": 0.0,
        "phase_amp": 0.0,
    }


# ---------------------------------------------------------------------------
# test_channel_attribution_and_dedup
# ---------------------------------------------------------------------------


def test_channel_attribution_and_dedup(hand_built_project):
    """P -> Z; S -> first horizontal present in ``S_PICK_CHANNEL_PREFERENCE``
    order; duplicate P rows for the same station collapse to the
    highest-probability one.
    """
    cfg, paths, event_id = hand_built_project
    file_name = f"{event_id}/{event_id}.mseed"

    rows = [
        # XX.ORNT: two P rows (duplicate -> highest score wins) + one S row.
        _pick_row("XX.ORNT.00.HH", "P", _START + 20.0, 0.60, file_name),
        _pick_row("XX.ORNT.00.HH", "P", _START + 20.05, 0.95, file_name),
        _pick_row("XX.ORNT.00.HH", "S", _START + 32.0, 0.80, file_name),
        # XX.UNOR: unoriented station -- S must land on "1" (first of
        # N,E,1,2 actually present).
        _pick_row("XX.UNOR.00.DP", "P", _START + 21.0, 0.70, file_name),
        _pick_row("XX.UNOR.00.DP", "S", _START + 33.0, 0.65, file_name),
    ]
    picks_csv = _write_picks_csv(paths, rows)

    catalog = build_catalog(picks_csv, paths, cfg)
    assert len(catalog.events) == 1
    event = catalog.events[0]
    assert get_event_id(event) == event_id

    # Dedup: exactly one P pick for XX.ORNT, the higher-probability one.
    ornt_p_picks = [
        p for p in event.picks if p.phase_hint == "P" and p.waveform_id.station_code == "ORNT"
    ]
    assert len(ornt_p_picks) == 1
    assert ornt_p_picks[0].time == UTCDateTime(_START + 20.05)
    assert get_extra(ornt_p_picks[0], "probability") == pytest.approx(0.95)

    # Channel attribution.
    by_key = {(p.waveform_id.station_code, p.phase_hint): p for p in event.picks}
    assert by_key[("ORNT", "P")].waveform_id.channel_code == "HHZ"
    assert by_key[("ORNT", "S")].waveform_id.channel_code == "HHN"  # N preferred over E
    assert by_key[("UNOR", "P")].waveform_id.channel_code == "DPZ"
    assert by_key[("UNOR", "S")].waveform_id.channel_code == "DP1"  # first of N,E,1,2 present

    assert S_PICK_CHANNEL_PREFERENCE == ["N", "E", "1", "2"]

    for pick in event.picks:
        assert pick.phase_hint in ("P", "S")
        assert str(pick.method_id) == PHASENET_METHOD_ID
        assert pick.evaluation_mode == "automatic"
        assert pick.time_errors.uncertainty == pytest.approx(cfg.phasenet.pick_uncertainty_s)


# ---------------------------------------------------------------------------
# test_round_trip_and_missing_channel
# ---------------------------------------------------------------------------


def test_round_trip_and_missing_channel(hand_built_project, tmp_path):
    """Full ``write_catalog``/``read_catalog`` round trip preserves everything
    this task's acceptance criteria check; an unresolvable station_id (no
    matching trace at all) is dropped with a warning, not a crash.
    """
    cfg, paths, event_id = hand_built_project
    file_name = f"{event_id}/{event_id}.mseed"

    rows = [
        _pick_row("XX.ORNT.00.HH", "P", _START + 20.0, 0.9, file_name),
        _pick_row("XX.ORNT.00.HH", "S", _START + 32.0, 0.85, file_name),
        # A station not present in chunks_pn at all -- must be dropped, not raise.
        _pick_row("XX.GHOST.00.HH", "P", _START + 22.0, 0.5, file_name),
    ]
    picks_csv = _write_picks_csv(paths, rows)

    catalog = build_catalog(picks_csv, paths, cfg)
    assert len(catalog.events) == 1
    event = catalog.events[0]
    assert len(event.picks) == 2  # the GHOST station's pick was dropped

    write_catalog_picks(catalog, paths)
    assert paths.catalog_picks_xml.is_file()

    cat2 = read_catalog(paths.catalog_picks_xml)
    assert len(cat2.events) == 1
    ev2 = cat2.events[0]
    assert len(ev2.picks) >= 1

    # Acceptance-criteria checks against the round-tripped catalogue.
    chunk_pn_channels = set()
    from obspy import read

    stream = read(str(paths.chunk_pn_file(event_id)))
    chunk_pn_channels = {tr.stats.channel for tr in stream}

    for pick in ev2.picks:
        assert pick.phase_hint in ("P", "S")
        assert pick.waveform_id.channel_code in chunk_pn_channels
        prob = get_extra(pick, "probability")
        assert prob is not None
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert pick.time_errors.uncertainty == pytest.approx(cfg.phasenet.pick_uncertainty_s)
        assert str(pick.method_id) == PHASENET_METHOD_ID
        assert pick.evaluation_mode == "automatic"


# ---------------------------------------------------------------------------
# test_missing_columns_raises
# ---------------------------------------------------------------------------


def test_missing_columns_raises(hand_built_project):
    cfg, paths, _event_id = hand_built_project
    bad_csv = paths.phasenet_raw_dir / "picks.csv"
    bad_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"station_id": ["XX.ORNT.00.HH"]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing expected column"):
        build_catalog(str(bad_csv), paths, cfg)


# ---------------------------------------------------------------------------
# Integration: real PhaseNet -> real conversion (requires_phasenet)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phasenet_world(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    index = build_index(cfg)
    station_table = compute_station_table(cfg.stations)
    return cfg, index, truth, station_table


def _make_detections_df(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        origin_time: UTCDateTime = ev["origin"]["time"]
        rows.append(
            {
                "event_id": ev["event_id_hint"],
                "time": origin_time.isoformat() + "Z",
                "duration_s": 30.0,
                "coincidence_sum": 8.0,
                "n_stations": 8,
                "stations": "",
                "trace_ids": "",
            }
        )
    return pd.DataFrame(rows, columns=DETECTIONS_COLUMNS)


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_build_catalog_real_phasenet(phasenet_world, tmp_path):
    cfg, index, truth, station_table = phasenet_world
    cfg2 = cfg.model_copy(deep=True)
    cfg2.project.output_dir = tmp_path / "run"
    paths2 = RunPaths(cfg2)

    events = truth["events"][:4]
    detections_df = _make_detections_df(events)
    write_chunks(detections_df, index, cfg2, paths2)
    event_ids = [ev["event_id_hint"] for ev in events]
    prepare_all(cfg2, paths2, event_ids, station_table=station_table)
    picks_csv = run_phasenet(paths2, cfg2)

    catalog = build_catalog(picks_csv, paths2, cfg2, station_table=station_table)
    assert len(catalog.events) >= 1
    for event in catalog:
        assert len(event.picks) >= 1
        for pick in event.picks:
            assert pick.phase_hint in ("P", "S")
            assert get_extra(pick, "probability") is not None

    write_catalog_picks(catalog, paths2)
    assert paths2.catalog_picks_xml.is_file()
    cat2 = read_catalog(paths2.catalog_picks_xml)
    assert len(cat2.events) == len(catalog.events)
