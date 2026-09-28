"""Tests for openmeq.magnitude.amplitude (M8.1): Wood-Anderson amplitude
measurement per event.

The synthetic_project fixture recovery test (M0.8 note) builds located
catalogues by hand -- truth origins plus truth picks via openmeq.catalog --
since location (M6.x) is being built concurrently; see
tests/data/make_synthetic.py's "Wood-Anderson ML calibration" docstring
section for the exact calibration this test's tolerance is based on.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import Origin, Pick, WaveformStreamID
from obspy.core.inventory import Channel, Inventory, Network, Response, Station
from obspy.core.inventory.response import InstrumentSensitivity, PolesZerosResponseStage
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import make_event
from openmeq.config import MagnitudeConfig, MlConfig, load_config
from openmeq.index import build_index
from openmeq.magnitude.amplitude import (
    WOOD_ANDERSON_PAZ,
    _hypocentral_distance_km,
    _simulate_wood_anderson,
    clear_response_cache,
    measure_event_amplitudes,
)
from openmeq.magnitude.relations import station_ml
from openmeq.stations import load_inventory

NETWORK = "XX"
LOCATION = "00"


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_response_cache()
    yield
    clear_response_cache()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flat_velocity_response(sensitivity: float) -> Response:
    """A flat, velocity-sensitive (M/S -> COUNTS) response -- matches
    tests/data/make_synthetic.py's own ``_flat_response`` fixture convention,
    so removing it requires the same integration step a real project's
    broadband sensor response would."""
    stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=sensitivity,
        stage_gain_frequency=1.0,
        input_units="M/S",
        output_units="COUNTS",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[],
        poles=[],
        normalization_factor=1.0,
    )
    sensitivity_obj = InstrumentSensitivity(
        value=sensitivity, frequency=1.0, input_units="M/S", output_units="COUNTS"
    )
    return Response(instrument_sensitivity=sensitivity_obj, response_stages=[stage])


def _flat_displacement_response(sensitivity: float) -> Response:
    """A flat, displacement-sensitive (M -> COUNTS) response -- used only by
    ``test_wa_simulation_known_response`` so ``remove_response(output="DISP")``
    is a pure gain divide with no extra integration, letting the analytic
    check isolate the Wood-Anderson simulation step itself."""
    stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=sensitivity,
        stage_gain_frequency=1.0,
        input_units="M",
        output_units="COUNTS",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[],
        poles=[],
        normalization_factor=1.0,
    )
    sensitivity_obj = InstrumentSensitivity(
        value=sensitivity, frequency=1.0, input_units="M", output_units="COUNTS"
    )
    return Response(instrument_sensitivity=sensitivity_obj, response_stages=[stage])


def _make_trace(
    net: str,
    sta: str,
    cha: str,
    data: np.ndarray,
    starttime: UTCDateTime,
    sampling_rate: float = 100.0,
    loc: str = LOCATION,
) -> Trace:
    tr = Trace(data=data.astype(np.float64))
    tr.stats.network = net
    tr.stats.station = sta
    tr.stats.location = loc
    tr.stats.channel = cha
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = starttime
    return tr


def _sine_burst(n: int, sampling_rate: float, freq_hz: float, amplitude: float) -> np.ndarray:
    t = np.arange(n) / sampling_rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _make_station(
    net: str,
    sta: str,
    lat: float,
    lon: float,
    elevation_m: float,
    depth_m: float,
    channels: list[str],
    sensitivity: float | None,
) -> Station:
    station = Station(code=sta, latitude=lat, longitude=lon, elevation=elevation_m, channels=[])
    for cha in channels:
        response = _flat_velocity_response(sensitivity) if sensitivity is not None else None
        station.channels.append(
            Channel(
                code=cha,
                location_code=LOCATION,
                latitude=lat,
                longitude=lon,
                elevation=elevation_m,
                depth=depth_m,
                azimuth=0.0,
                dip=-90.0 if cha.endswith("Z") else 0.0,
                sample_rate=100.0,
                response=response,
            )
        )
    return station


def _make_inventory(net: str, stations: list[Station]) -> Inventory:
    network = Network(code=net, stations=stations)
    return Inventory(networks=[network], source="test")


def _pick(net: str, sta: str, cha: str, phase: str, time: UTCDateTime) -> Pick:
    return Pick(
        time=time,
        phase_hint=phase,
        evaluation_mode="automatic",
        waveform_id=WaveformStreamID(
            network_code=net, station_code=sta, location_code=LOCATION, channel_code=cha
        ),
    )


def _make_event_with_origin(event_id: str, origin_time, lat: float, lon: float, depth_m: float):
    event = make_event(event_id)
    origin = Origin(time=origin_time, latitude=lat, longitude=lon, depth=depth_m)
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    return event, origin


# ---------------------------------------------------------------------------
# test_wa_simulation_known_response
# ---------------------------------------------------------------------------


def test_wa_simulation_known_response():
    """A synthetic trace with a known, simple (flat, displacement-sensitive)
    PAZ response and a known true ground-displacement sinusoid, run through
    remove_response + WA simulate, recovers the analytically expected peak
    amplitude (accounting for wa_sensitivity) to within 1%."""
    sensitivity = 1.0e6
    freq_hz = 2.0
    sampling_rate = 100.0
    duration_s = 20.0
    n = int(duration_s * sampling_rate)
    amplitude_m = 1.0e-4  # true ground displacement, metres

    response = _flat_displacement_response(sensitivity)
    counts = sensitivity * _sine_burst(n, sampling_rate, freq_hz, amplitude_m)
    tr = _make_trace(NETWORK, "STA1", "HHN", counts, UTCDateTime(2025, 1, 1), sampling_rate)

    ml_cfg = MlConfig()
    out = _simulate_wood_anderson(tr, response, ml_cfg)

    # Measure well inside the trace, away from the taper/filter edges.
    interior = out.slice(tr.stats.starttime + 5, tr.stats.starttime + duration_s - 5)
    measured = float(np.max(np.abs(interior.data)))

    from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp

    h = paz_2_amplitude_value_of_freq_resp(WOOD_ANDERSON_PAZ, freq_hz)
    expected = amplitude_m * h * ml_cfg.wa_sensitivity

    assert measured == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# window selection
# ---------------------------------------------------------------------------


def _simple_3comp_station_setup(net="XX", sta="STA1", sensitivity=2.0e8):
    lat, lon = 37.80, -122.20
    station = _make_station(
        net, sta, lat, lon, 600.0, 0.0, ["HHZ", "HHN", "HHE"], sensitivity=sensitivity
    )
    inv = _make_inventory(net, [station])
    return inv, lat, lon


def test_window_selection_uses_s_pick(caplog):
    inv, lat, lon = _simple_3comp_station_setup()
    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 2.0
    s_time = t0 + 5.0

    event, _origin = _make_event_with_origin("evtest0000000000_000", t0, lat, lon, 3000.0)
    event.picks.append(_pick(NETWORK, "STA1", "HHZ", "P", p_time))
    event.picks.append(_pick(NETWORK, "STA1", "HHN", "S", s_time))

    n = int(20.0 * 100.0)
    stream = Stream(
        [
            _make_trace(NETWORK, "STA1", cha, _sine_burst(n, 100.0, 5.0, 2000.0), t0)
            for cha in ("HHZ", "HHN", "HHE")
        ]
    )

    cfg = MlConfig()
    mag_cfg = MagnitudeConfig(ml=cfg)
    results = measure_event_amplitudes(event, stream, inv, mag_cfg)

    assert len(results) == 1
    got = results[0]
    assert got.window_start == s_time - cfg.window_before_s
    assert got.window_end == s_time + cfg.window_after_s


def test_window_fallback_no_s_pick(caplog):
    inv, lat, lon = _simple_3comp_station_setup()
    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 2.0

    event, _origin = _make_event_with_origin("evtest0000000001_000", t0, lat, lon, 3000.0)
    event.picks.append(_pick(NETWORK, "STA1", "HHZ", "P", p_time))

    n = int(20.0 * 100.0)
    stream = Stream(
        [
            _make_trace(NETWORK, "STA1", cha, _sine_burst(n, 100.0, 5.0, 2000.0), t0)
            for cha in ("HHZ", "HHN", "HHE")
        ]
    )

    mag_cfg = MagnitudeConfig()
    with caplog.at_level(logging.WARNING, logger="openmeq.magnitude"):
        results = measure_event_amplitudes(event, stream, inv, mag_cfg)

    assert len(results) == 1
    got = results[0]
    assert got.window_start == p_time + 1.0
    assert got.window_end == p_time + 8.0
    assert any("no S pick" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# component selection
# ---------------------------------------------------------------------------


def test_horizontal_component_selection(caplog):
    net = NETWORK
    lat, lon = 37.80, -122.20
    sta_both = _make_station(
        net, "STA1", lat, lon, 600.0, 0.0, ["HHZ", "HHN", "HHE"], sensitivity=2.0e8
    )
    sta_zonly = _make_station(net, "STA2", lat, lon, 600.0, 0.0, ["HHZ"], sensitivity=2.0e8)
    inv = _make_inventory(net, [sta_both, sta_zonly])

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 2.0
    s_time = t0 + 5.0

    event, _origin = _make_event_with_origin("evtest0000000002_000", t0, lat, lon, 3000.0)
    for sta in ("STA1", "STA2"):
        event.picks.append(_pick(net, sta, "HHZ", "P", p_time))
        event.picks.append(_pick(net, sta, "HHN", "S", s_time))

    n = int(20.0 * 100.0)
    traces = [
        _make_trace(net, "STA1", "HHZ", _sine_burst(n, 100.0, 5.0, 500.0), t0),
        _make_trace(net, "STA1", "HHN", _sine_burst(n, 100.0, 5.0, 800.0), t0),
        _make_trace(net, "STA1", "HHE", _sine_burst(n, 100.0, 5.0, 3000.0), t0),  # larger
        _make_trace(net, "STA2", "HHZ", _sine_burst(n, 100.0, 5.0, 900.0), t0),
    ]
    stream = Stream(traces)

    mag_cfg = MagnitudeConfig()
    with caplog.at_level(logging.INFO, logger="openmeq.magnitude"):
        results = measure_event_amplitudes(event, stream, inv, mag_cfg)

    by_sta = {r.sta: r for r in results}
    assert by_sta["STA1"].component_used == "HHE"
    assert by_sta["STA2"].component_used == "HHZ"
    assert any("no horizontal channels present" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


def test_distance_calculation():
    origin_lat, origin_lon, origin_depth_m = 37.80, -122.20, 5000.0
    sta_lat, sta_lon = 37.85, -122.15
    elevation_m, depth_m = 600.0, 300.0  # borehole sensor 300 m below surface

    station = _make_station(
        NETWORK, "STA1", sta_lat, sta_lon, elevation_m, depth_m, ["HHZ"], sensitivity=2.0e8
    )
    inv = _make_inventory(NETWORK, [station])
    origin = Origin(
        time=UTCDateTime(2025, 1, 1),
        latitude=origin_lat,
        longitude=origin_lon,
        depth=origin_depth_m,
    )

    trace = _make_trace(NETWORK, "STA1", "HHZ", np.zeros(10), UTCDateTime(2025, 1, 1))
    got_km = _hypocentral_distance_km(origin, inv, trace)

    horizontal_m, _az, _baz = gps2dist_azimuth(origin_lat, origin_lon, sta_lat, sta_lon)
    horizontal_km = horizontal_m / 1000.0
    effective_elevation_m = elevation_m - depth_m
    vertical_km = origin_depth_m / 1000.0 + effective_elevation_m / 1000.0
    expected_km = math.hypot(horizontal_km, vertical_km)

    assert got_km == pytest.approx(expected_km, abs=0.01)


# ---------------------------------------------------------------------------
# skips
# ---------------------------------------------------------------------------


def test_skips_missing_response(caplog):
    net = NETWORK
    lat, lon = 37.80, -122.20
    sta_ok = _make_station(net, "STA1", lat, lon, 600.0, 0.0, ["HHZ"], sensitivity=2.0e8)
    # STA2 has a Station entry (e.g. CSV-only metadata) but no channels at
    # all -- so it can never have a response, matching a station known only
    # from the stations CSV (openmeq.stations.load_inventory).
    sta_no_response = Station(
        code="STA2", latitude=lat, longitude=lon, elevation=600.0, channels=[]
    )
    inv = _make_inventory(net, [sta_ok, sta_no_response])

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 2.0

    event, _origin = _make_event_with_origin("evtest0000000003_000", t0, lat, lon, 3000.0)
    event.picks.append(_pick(net, "STA1", "HHZ", "P", p_time))
    event.picks.append(_pick(net, "STA2", "HHZ", "P", p_time))

    n = int(20.0 * 100.0)
    stream = Stream(
        [
            _make_trace(net, "STA1", "HHZ", _sine_burst(n, 100.0, 5.0, 2000.0), t0),
            _make_trace(net, "STA2", "HHZ", _sine_burst(n, 100.0, 5.0, 2000.0), t0),
        ]
    )

    mag_cfg = MagnitudeConfig()
    with caplog.at_level(logging.WARNING, logger="openmeq.magnitude"):
        results = measure_event_amplitudes(event, stream, inv, mag_cfg)

    stas = {r.sta for r in results}
    assert "STA1" in stas
    assert "STA2" not in stas
    assert any("no response" in rec.message for rec in caplog.records)


def test_skips_flat_window(caplog):
    net = NETWORK
    lat, lon = 37.80, -122.20
    station = _make_station(net, "STA1", lat, lon, 600.0, 0.0, ["HHZ"], sensitivity=2.0e8)
    inv = _make_inventory(net, [station])

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 2.0

    event, _origin = _make_event_with_origin("evtest0000000004_000", t0, lat, lon, 3000.0)
    event.picks.append(_pick(net, "STA1", "HHZ", "P", p_time))

    n = int(20.0 * 100.0)
    stream = Stream([_make_trace(net, "STA1", "HHZ", np.zeros(n), t0)])

    mag_cfg = MagnitudeConfig()
    with caplog.at_level(logging.WARNING, logger="openmeq.magnitude"):
        results = measure_event_amplitudes(event, stream, inv, mag_cfg)

    assert results == []
    assert any("flat-lined" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# synthetic_project recovery
# ---------------------------------------------------------------------------


def _z_and_horizontal_channels(inv: Inventory, net: str, sta: str) -> tuple[str, str]:
    for network in inv.networks:
        if network.code != net:
            continue
        for station in network.stations:
            if station.code != sta:
                continue
            codes = [c.code for c in station.channels]
            z = next(c for c in codes if c.endswith("Z"))
            horiz = next(c for c in codes for last in ("N", "E", "1", "2") if c.endswith(last))
            return z, horiz
    raise AssertionError(f"station {net}.{sta} not found in inventory")


def _truth_distance_km(ev: dict, station_row: dict) -> float:
    origin = ev["origin"]
    horizontal_m, _az, _baz = gps2dist_azimuth(
        origin["lat"], origin["lon"], station_row["lat"], station_row["lon"]
    )
    horizontal_km = horizontal_m / 1000.0
    effective_elevation_m = station_row["elev_m"] - station_row["depth_m"]
    vertical_km = origin["depth_km"] + effective_elevation_m / 1000.0
    return math.hypot(horizontal_km, vertical_km)


def test_synthetic_project_recovery(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    # Read-only against the shared session fixture's waveforms/inventory;
    # only the (tiny) index.json this test builds is redirected to this
    # test's own tmp_path, so the ~500 MB shared synthetic dataset is never
    # copied or mutated (see tests/conftest.py's usage note).
    cfg.project.output_dir = tmp_path / "run"

    inv = load_inventory(cfg.stations)
    index = build_index(cfg)

    stations_by_key = {f"{row['net']}.{row['sta']}": row for row in truth["stations"]}

    all_recovered_mls = []
    for ev in truth["events"][:6]:
        event_id = ev["event_id_hint"]
        origin_time = UTCDateTime(ev["origin"]["time"])
        depth_m = ev["origin"]["depth_km"] * 1000.0
        event, _origin = _make_event_with_origin(
            event_id, origin_time, ev["origin"]["lat"], ev["origin"]["lon"], depth_m
        )

        s_times = []
        for station_key, arrivals in ev["arrivals"].items():
            net, sta = station_key.split(".", 1)
            p_time = UTCDateTime(arrivals["P"])
            s_time = UTCDateTime(arrivals["S"])
            s_times.append(s_time)
            z_code, horiz_code = _z_and_horizontal_channels(inv, net, sta)
            event.picks.append(_pick(net, sta, z_code, "P", p_time))
            event.picks.append(_pick(net, sta, horiz_code, "S", s_time))

        t1 = origin_time - 5.0
        t2 = max(s_times) + 15.0
        stream = index.read(f"{NETWORK}.*", t1, t2)
        assert len(stream) > 0

        results = measure_event_amplitudes(event, stream, inv, cfg.magnitude)
        assert len(results) >= 6, f"{event_id}: too few stations recovered ({len(results)})"

        station_mls = []
        for r in results:
            assert math.isfinite(r.amplitude_m)
            assert r.amplitude_m > 0
            truth_row = stations_by_key[f"{r.net}.{r.sta}"]
            truth_km = _truth_distance_km(ev, truth_row)
            assert r.distance_km == pytest.approx(truth_km, rel=0.05)

            ml = station_ml(
                amplitude_m=r.amplitude_m, distance_km=r.distance_km, cfg=cfg, net=r.net, sta=r.sta
            )
            station_mls.append(ml)

        event_ml = float(np.median(station_mls))
        all_recovered_mls.append((event_id, event_ml, ev["ml"]))
        assert event_ml == pytest.approx(ev["ml"], abs=0.15), (
            f"{event_id}: recovered event ML {event_ml:.3f} vs truth {ev['ml']:.3f}"
        )
