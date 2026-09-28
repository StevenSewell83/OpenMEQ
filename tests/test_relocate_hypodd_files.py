"""Tests for openmeq.relocate.hypodd_files (M9.2 / OpenMEQ-6ld.2).

Unit tests build small synthetic catalogues directly from ObsPy objects
(the M0.8 synthetic-dataset fixture isn't merged yet). The two
``requires_hypodd`` tests reconstruct examples/example1's own events/picks/
stations with a small test-only reader and run the real ph2dt/hypoDD
binaries against this module's freshly-written files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Arrival,
    Event,
    Magnitude,
    Origin,
    OriginQuality,
    OriginUncertainty,
    Pick,
    QuantityError,
    WaveformStreamID,
)

from openmeq.catalog import event_id, event_id_from_time, make_event, reject_pick
from openmeq.config import ConfigError, load_config
from openmeq.external import ExternalToolMissing, bin_path, run
from openmeq.relocate.hypodd_files import (
    build_event_id_map,
    build_station_label_map,
    depth_m_to_km,
    invert_event_id_map,
    write_event_dat,
    write_hypodd_inp_1d,
    write_ph2dt_inp,
    write_phase_dat,
    write_station_dat,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"


# ---------------------------------------------------------------------------
# small synthetic-catalogue helpers
# ---------------------------------------------------------------------------


def _make_event(
    time: UTCDateTime,
    lat: float = 37.9,
    lon: float = -122.2,
    depth_m: float = 8000.0,
    mag: float = 1.5,
    picks: list[tuple[str, str, float, float | None]] | None = None,
) -> Event:
    """Build a minimal Event with one Origin and, optionally, picks/arrivals.

    ``picks`` is a list of ``(net_sta, phase, travel_time_s, time_weight)``;
    ``time_weight=None`` means "no weight info" (should become phase.dat's
    -1.0 sentinel).
    """
    eid = event_id_from_time(time)
    ev = make_event(eid)
    origin = Origin(time=time, latitude=lat, longitude=lon, depth=depth_m)
    origin.origin_uncertainty = OriginUncertainty(horizontal_uncertainty=150.0)
    origin.depth_errors = QuantityError(uncertainty=500.0)
    origin.quality = OriginQuality(standard_error=0.02)
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    mag_obj = Magnitude(mag=mag, magnitude_type="ML")
    ev.magnitudes = [mag_obj]
    ev.preferred_magnitude_id = mag_obj.resource_id

    for net_sta, phase, tt, weight in picks or []:
        net, sta = net_sta.split(".", 1)
        pick = Pick(
            time=time + tt,
            phase_hint=phase,
            waveform_id=WaveformStreamID(network_code=net, station_code=sta),
        )
        ev.picks.append(pick)
        arrival = Arrival(pick_id=pick.resource_id, phase=phase, time_weight=weight)
        origin.arrivals.append(arrival)

    return ev


def _station_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["network", "station", "location", "latitude", "longitude", "elevation_m", "depth_m"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# depth conversion
# ---------------------------------------------------------------------------


def test_depth_m_to_km():
    assert depth_m_to_km(9800.0) == pytest.approx(9.8)
    assert depth_m_to_km(0.0) == 0.0


# ---------------------------------------------------------------------------
# event id (cuspid) mapping
# ---------------------------------------------------------------------------


def test_event_id_map_stable_across_runs(tmp_path):
    map_path = tmp_path / "event_id_map.csv"
    ev1 = _make_event(UTCDateTime("2025-01-01T00:00:00"))
    ev2 = _make_event(UTCDateTime("2025-01-02T00:00:00"))
    cat = Catalog(events=[ev1, ev2])

    id_map1 = build_event_id_map(cat, map_path)
    assert set(id_map1.values()) == {1, 2}
    assert id_map1[event_id(ev1)] < id_map1[event_id(ev2)]

    # Add a new event *earlier* than both existing ones -- it must still get
    # the next integer, not be inserted chronologically (existing cuspids
    # are never renumbered).
    ev0 = _make_event(UTCDateTime("2024-06-01T00:00:00"))
    cat2 = Catalog(events=[ev1, ev2, ev0])
    id_map2 = build_event_id_map(cat2, map_path)

    assert id_map2[event_id(ev1)] == id_map1[event_id(ev1)]
    assert id_map2[event_id(ev2)] == id_map1[event_id(ev2)]
    assert id_map2[event_id(ev0)] == 3

    # re-reading from disk reproduces the same mapping
    id_map3 = build_event_id_map(Catalog(events=[ev1, ev2, ev0]), map_path)
    assert id_map3 == id_map2

    inverse = invert_event_id_map(id_map2)
    assert inverse[id_map2[event_id(ev1)]] == event_id(ev1)


# ---------------------------------------------------------------------------
# station label mapping
# ---------------------------------------------------------------------------


def test_station_label_map_handles_collision(tmp_path):
    # Two stations differing only in network, same 5-char station code:
    # disambiguated with a digit suffix (fits within 7 characters).
    table = _station_table(
        [
            {
                "network": "AA",
                "station": "ABCDE",
                "location": "",
                "latitude": 1.0,
                "longitude": 1.0,
                "elevation_m": 0.0,
                "depth_m": 0.0,
            },
            {
                "network": "BB",
                "station": "ABCDE",
                "location": "",
                "latitude": 2.0,
                "longitude": 2.0,
                "elevation_m": 0.0,
                "depth_m": 0.0,
            },
        ]
    )
    label_map = build_station_label_map(table, tmp_path / "labels.csv")
    assert label_map["AA.ABCDE"] == "ABCDE"
    assert label_map["BB.ABCDE"] == "ABCDE0"
    assert len(label_map["BB.ABCDE"]) <= 7

    # Two stations sharing a *7-character* bare code: no spare character to
    # append a disambiguating digit -- must raise a clear, actionable error,
    # not silently mismap.
    table7 = _station_table(
        [
            {
                "network": "AA",
                "station": "ABCDEFG",
                "location": "",
                "latitude": 1.0,
                "longitude": 1.0,
                "elevation_m": 0.0,
                "depth_m": 0.0,
            },
            {
                "network": "BB",
                "station": "ABCDEFG",
                "location": "",
                "latitude": 2.0,
                "longitude": 2.0,
                "elevation_m": 0.0,
                "depth_m": 0.0,
            },
        ]
    )
    with pytest.raises(ConfigError, match="ABCDEFG"):
        build_station_label_map(table7, tmp_path / "labels7.csv")


def test_station_label_map_stable_across_runs(tmp_path):
    map_path = tmp_path / "labels.csv"
    table1 = _station_table(
        [
            {
                "network": "XX",
                "station": "AAA",
                "location": "",
                "latitude": 1.0,
                "longitude": 1.0,
                "elevation_m": 0.0,
                "depth_m": 0.0,
            }
        ]
    )
    map1 = build_station_label_map(table1, map_path)
    table2 = pd.concat(
        [
            table1,
            _station_table(
                [
                    {
                        "network": "XX",
                        "station": "BBB",
                        "location": "",
                        "latitude": 2.0,
                        "longitude": 2.0,
                        "elevation_m": 0.0,
                        "depth_m": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    map2 = build_station_label_map(table2, map_path)
    assert map2["XX.AAA"] == map1["XX.AAA"]
    assert map2["XX.BBB"] == "BBB"


# ---------------------------------------------------------------------------
# station.dat
# ---------------------------------------------------------------------------


def test_station_dat_elevation_units_and_sign(tmp_path, caplog):
    table = _station_table(
        [
            {
                "network": "XX",
                "station": "SURF",
                "location": "",
                "latitude": 10.0,
                "longitude": 20.0,
                "elevation_m": 500.0,
                "depth_m": 0.0,
            },
            {
                "network": "XX",
                "station": "BORE",
                "location": "",
                "latitude": 11.0,
                "longitude": 21.0,
                "elevation_m": 100.0,
                "depth_m": 150.0,
            },
        ]
    )
    label_map = {"XX.SURF": "SURF", "XX.BORE": "BORE"}
    cfg = load_config(EXAMPLE_CONFIG)
    out = tmp_path / "station.dat"

    with caplog.at_level("WARNING"):
        write_station_dat(out, table, label_map, cfg)

    lines = {line.split()[0]: line for line in out.read_text().splitlines()}
    assert float(lines["SURF"].split()[3]) == pytest.approx(500.0)
    assert float(lines["BORE"].split()[3]) == pytest.approx(-50.0)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("BORE" in msg for msg in warnings)
    assert not any("SURF" in msg for msg in warnings)


# ---------------------------------------------------------------------------
# phase.dat
# ---------------------------------------------------------------------------


def test_phase_dat_rejected_picks_excluded(tmp_path):
    t = UTCDateTime("2025-03-01T00:00:00")
    ev = _make_event(
        t,
        picks=[
            ("XX.AAA", "P", 1.0, 0.8),
            ("XX.BBB", "P", 1.5, 0.8),
        ],
    )
    # Reject the second pick the same way every other step does (§5):
    origin = ev.preferred_origin()
    rejected_pick = ev.picks[1]
    rejected_arrival = origin.arrivals[1]
    reject_pick(rejected_pick, rejected_arrival, "bad-snr", "test")
    assert rejected_arrival.time_weight == 0  # the pipeline's own convention

    cat = Catalog(events=[ev])
    event_id_map = {event_id(ev): 1}
    label_map = {"XX.AAA": "AAA", "XX.BBB": "BBB"}
    out = tmp_path / "phase.dat"
    write_phase_dat(out, cat, event_id_map, label_map)

    text = out.read_text()
    assert "AAA" in text
    assert "BBB" not in text


def test_phase_dat_no_weight_writes_sentinel(tmp_path):
    t = UTCDateTime("2025-03-01T00:00:00")
    ev = _make_event(t, picks=[("XX.AAA", "P", 1.0, None)])
    cat = Catalog(events=[ev])
    event_id_map = {event_id(ev): 1}
    label_map = {"XX.AAA": "AAA"}
    out = tmp_path / "phase.dat"
    write_phase_dat(out, cat, event_id_map, label_map)

    data_line = next(line for line in out.read_text().splitlines() if line.startswith("AAA"))
    weight = float(data_line.split()[2])
    assert weight == pytest.approx(-1.0)


def test_phase_dat_weight_clamped_to_unit_interval(tmp_path):
    t = UTCDateTime("2025-03-01T00:00:00")
    ev = _make_event(t, picks=[("XX.AAA", "P", 1.0, 1.5)])
    cat = Catalog(events=[ev])
    event_id_map = {event_id(ev): 1}
    label_map = {"XX.AAA": "AAA"}
    out = tmp_path / "phase.dat"
    write_phase_dat(out, cat, event_id_map, label_map)
    data_line = next(line for line in out.read_text().splitlines() if line.startswith("AAA"))
    assert float(data_line.split()[2]) == pytest.approx(1.0)


def test_phase_dat_header_time_encoding(tmp_path):
    # The same event/time used in docs/hypoddpy_audit.md's worked example
    # (examples/example1's first event): hour=2, minute=19, second=58.71.
    t = UTCDateTime(1985, 1, 24, 2, 19, 58.71)
    ev = _make_event(t, lat=37.8832, lon=-122.2415, depth_m=9800.0, mag=1.4)
    cat = Catalog(events=[ev])
    event_id_map = {event_id(ev): 38542}
    out = tmp_path / "phase.dat"
    write_phase_dat(out, cat, event_id_map, {})

    header = next(line for line in out.read_text().splitlines() if line.startswith("#"))
    fields = header.split()
    # "#" YR MO DY HR MN SEC.SS LAT LON DEPTH_KM MAG HERR VERR RES CUSPID
    assert fields[1:7] == ["1985", "1", "24", "2", "19", "58.71"]
    assert fields[-1] == "38542"


# ---------------------------------------------------------------------------
# event.dat
# ---------------------------------------------------------------------------


def test_event_dat_time_encoding(tmp_path):
    t = UTCDateTime(1985, 1, 24, 2, 19, 58.71)
    ev = _make_event(t, lat=37.8832, lon=-122.2415, depth_m=9800.0, mag=1.4)
    cat = Catalog(events=[ev])
    event_id_map = {event_id(ev): 38542}
    out = tmp_path / "event.dat"
    write_event_dat(out, cat, event_id_map)

    line = out.read_text().splitlines()[0]
    fields = line.split()
    # hand computation: hour*1e6 + minute*1e4 + second*100 + centisecond
    # = 2*1000000 + 19*10000 + 58*100 + 71 = 2195871 (matches
    # examples/example1/event.dat's own first row for this same event).
    assert int(fields[0]) == 19850124
    assert int(fields[1]) == 2195871
    assert int(fields[-1]) == 38542


# ---------------------------------------------------------------------------
# ph2dt.inp
# ---------------------------------------------------------------------------


def test_ph2dt_inp_fields(tmp_path):
    cfg = load_config(EXAMPLE_CONFIG)
    out = tmp_path / "ph2dt.inp"
    write_ph2dt_inp(out, cfg)

    lines = [
        line for line in out.read_text().splitlines() if line.strip() and not line.startswith("*")
    ]
    assert lines[0] == "station.dat"
    assert lines[1] == "phase.dat"
    fields = lines[2].split()
    assert len(fields) == 7
    assert float(fields[0]) == cfg.hypodd.ph2dt.minwght
    assert float(fields[1]) == cfg.hypodd.ph2dt.maxdist_km


# ---------------------------------------------------------------------------
# hypoDD.inp (1D)
# ---------------------------------------------------------------------------


def _data_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("*")
    ]


def test_hypodd_inp_1d_section_order(tmp_path):
    cfg = load_config(EXAMPLE_CONFIG)
    model = pd.DataFrame(
        {
            "depth_km": [0.0, 1.0, 3.0],
            "vp": [3.5, 4.5, 5.8],
            "vs": [2.0, 2.6, 3.35],
            "rho": [2.3, 2.5, 2.7],
        }
    )
    out = tmp_path / "hypoDD.inp"
    write_hypodd_inp_1d(out, cfg, cfg.hypodd.weighting_rows, model)

    written = _data_lines(out.read_text())
    bundled = _data_lines((EXAMPLE1_DIR / "hypoDD.inp").read_text())

    fixed_names = [
        "hypoDD_2",
        "dt.cc",
        "dt.ct",
        "event.sel",
        "station.sel",
        "hypoDD.loc",
        "hypoDD.reloc",
        "hypoDD.sta",
        "hypoDD.res",
        "hypoDD.src",
    ]
    assert written[:10] == fixed_names
    assert bundled[:10] == fixed_names

    assert len(written) == len(bundled)  # both use a 5-row weighting scheme

    assert len(written[10].split()) == 3  # IDAT IPHA DIST
    assert len(written[11].split()) == 5  # OBSCC OBSCT MINDIST MAXDIST MAXGAP
    assert len(written[12].split()) == 4  # ISTART ISOLV IAQ NSET
    nset = int(written[12].split()[3])
    assert nset == len(cfg.hypodd.weighting_rows) == 5

    for row in written[13 : 13 + nset]:
        assert len(row.split()) == 10  # NITER..DAMP

    imod_idx = 13 + nset
    assert written[imod_idx] == "1"  # IMOD = 1 (not 0), per M9.1's audit

    for model_line in written[imod_idx + 1 : imod_idx + 4]:
        assert model_line.split()[-1] == "-9"

    cid_idx = imod_idx + 4
    assert written[cid_idx] == "0"  # relocate all clusters
    assert len(written) == cid_idx + 1  # blank ID list: nothing follows CID


# ---------------------------------------------------------------------------
# example1 round-trip: a small test-only reader for El16.pha/event.dat/
# station.dat, reconstructed as a synthetic QuakeML Catalog.
# ---------------------------------------------------------------------------


def _read_example1_stations(example1_dir: Path) -> dict[str, tuple[float, float]]:
    stations = {}
    with open(example1_dir / "station.dat") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            stations[parts[0]] = (float(parts[1]), float(parts[2]))
    return stations


def _read_example1_events(example1_dir: Path) -> list[dict]:
    events: list[dict] = []
    current = None
    with open(example1_dir / "El16.pha") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                parts = line.split()
                (_, yr, mo, dy, hr, mn, sec, lat, lon, depth, mag, herr, verr, res, cuspid) = parts
                current = {
                    "time": UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mn), float(sec)),
                    "lat": float(lat),
                    "lon": float(lon),
                    "depth_km": float(depth),
                    "mag": float(mag),
                    "herr": float(herr),
                    "verr": float(verr),
                    "res": float(res),
                    "cuspid": int(cuspid),
                    "picks": [],
                }
                events.append(current)
            else:
                parts = line.split()
                sta, tt, wt, pha = parts[0], float(parts[1]), float(parts[2]), parts[3]
                current["picks"].append((sta, tt, wt, pha))
    return events


def _build_example1_catalog(
    stations: dict[str, tuple[float, float]], events: list[dict]
) -> tuple[Catalog, dict[str, int], pd.DataFrame]:
    """Reconstruct example1's own events/picks/stations as a synthetic
    QuakeML Catalog + event_id_map + station_table, preserving the original
    cuspids and station labels exactly (as opposed to
    `build_event_id_map`/`build_station_label_map`'s own auto-assignment)
    so the produced dt.ct is directly comparable to the bundled one.
    """
    cat = Catalog()
    event_id_map: dict[str, int] = {}
    used_stations: set[str] = set()

    for ev in events:
        origin = Origin(
            time=ev["time"],
            latitude=ev["lat"],
            longitude=ev["lon"],
            depth=ev["depth_km"] * 1000.0,
        )
        origin.origin_uncertainty = OriginUncertainty(horizontal_uncertainty=ev["herr"] * 1000.0)
        origin.depth_errors = QuantityError(uncertainty=ev["verr"] * 1000.0)
        origin.quality = OriginQuality(standard_error=ev["res"])

        event = make_event(event_id_from_time(ev["time"]))
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        mag_obj = Magnitude(mag=ev["mag"], magnitude_type="ML")
        event.magnitudes = [mag_obj]
        event.preferred_magnitude_id = mag_obj.resource_id

        for sta, tt, wt, pha in ev["picks"]:
            if sta not in stations:
                # 2 of example1's own El16.pha entries reference a station
                # missing from its own station.dat; ph2dt itself just skips
                # that one observation (ph2dt.f:607-616), so we drop it too
                # rather than crash on an unmapped station label.
                continue
            used_stations.add(sta)
            pick = Pick(
                time=ev["time"] + tt,
                phase_hint=pha,
                waveform_id=WaveformStreamID(network_code="XX", station_code=sta),
            )
            event.picks.append(pick)
            # El16.pha's own convention: exactly -1.000 is the "no info"
            # sentinel (-> no Arrival.time_weight, OpenMEQ's equivalent);
            # this raw dataset also has a single -0.500 outlier, an
            # overloaded negative encoding of a real 0.5 quality score
            # (confirmed against ph2dt.f's `abs(p_wghtr(...))` averaging,
            # which reproduces the bundled dt.ct's weight for that pick only
            # if the magnitude, not just "negative => no info", survives).
            weight = None if wt == -1.0 else max(0.0, min(1.0, abs(wt)))
            arrival = Arrival(pick_id=pick.resource_id, phase=pha, time_weight=weight)
            origin.arrivals.append(arrival)

        cat.append(event)
        event_id_map[event_id(event)] = ev["cuspid"]

    rows = [
        {
            "network": "XX",
            "station": label,
            "location": "",
            "latitude": stations[label][0],
            "longitude": stations[label][1],
            "elevation_m": 0.0,
            "depth_m": 0.0,
        }
        for label in sorted(used_stations)
    ]
    station_table = pd.DataFrame(rows)
    return cat, event_id_map, station_table


def _parse_dt_ct(path: Path) -> dict[tuple[int, int], list[tuple[str, float, float, float, str]]]:
    pairs: dict[tuple[int, int], list] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                _, c1, c2 = line.split()
                current = (int(c1), int(c2))
                pairs[current] = []
            elif line.strip():
                parts = line.split()
                sta = parts[0]
                t1, t2, wt = float(parts[1]), float(parts[2]), float(parts[3])
                pha = parts[4]
                pairs[current].append((sta, t1, t2, wt, pha))
    return pairs


HYPODD_MISSING = not (EXAMPLE1_DIR.exists() and (EXTERNAL_DIR / "bin" / "ph2dt").exists())


@pytest.mark.requires_hypodd
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt) not available")
def test_example1_ph2dt_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, event_id_map, station_table = _build_example1_catalog(stations, events)
    label_map = build_station_label_map(station_table, tmp_path / "station_label_map.csv")

    cfg = load_config(EXAMPLE_CONFIG)
    # Match examples/example1/ph2dt.inp's own parameters exactly, so the
    # produced dt.ct is directly comparable to the bundled one.
    cfg.hypodd.ph2dt.maxdist_km = 120
    cfg.hypodd.ph2dt.maxsep_km = 10
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100

    write_station_dat(tmp_path / "station.dat", station_table, label_map, cfg)
    write_phase_dat(tmp_path / "phase.dat", cat, event_id_map, label_map)
    write_ph2dt_inp(tmp_path / "ph2dt.inp", cfg)

    try:
        ph2dt = bin_path("ph2dt")
    except ExternalToolMissing:
        pytest.skip("ph2dt binary not available")

    run([str(ph2dt), "ph2dt.inp"], cwd=tmp_path)

    written = _parse_dt_ct(tmp_path / "dt.ct")
    bundled = _parse_dt_ct(EXAMPLE1_DIR / "dt.ct")

    assert set(written) == set(bundled)
    for key, bundled_lines in bundled.items():
        # Key each pair's lines by (station, phase) -- unique within one
        # event-pair block -- rather than a positional sort, since sorting
        # full tuples (which include the ph2dt-averaged weight) can reorder
        # same-station P/S lines relative to each other for no functional
        # reason.
        written_by_key = {(sta, pha): (t1, t2, wt) for sta, t1, t2, wt, pha in written[key]}
        bundled_by_key = {(sta, pha): (t1, t2, wt) for sta, t1, t2, wt, pha in bundled_lines}
        assert set(written_by_key) == set(bundled_by_key)
        for sta_pha, (t1_b, t2_b, wt_b) in bundled_by_key.items():
            t1_w, t2_w, wt_w = written_by_key[sta_pha]
            assert t1_w == pytest.approx(t1_b, abs=5e-3)
            assert t2_w == pytest.approx(t2_b, abs=5e-3)
            assert wt_w == pytest.approx(wt_b, abs=5e-3)


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (hypoDD) not available")
def test_example1_hypodd_reloc_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, event_id_map, station_table = _build_example1_catalog(stations, events)
    label_map = build_station_label_map(station_table, tmp_path / "station_label_map.csv")

    cfg = load_config(EXAMPLE_CONFIG)
    cfg.hypodd.ph2dt.maxdist_km = 120
    cfg.hypodd.ph2dt.maxsep_km = 10
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100
    # Match example1's actual working IDAT/IPHA/DIST and clustering values
    # (the bundled hypoDD.inp; our own defaults for these already match).
    cfg.hypodd.max_station_dist_km = 120

    write_station_dat(tmp_path / "station.dat", station_table, label_map, cfg)
    write_phase_dat(tmp_path / "phase.dat", cat, event_id_map, label_map)
    write_event_dat(tmp_path / "event.dat", cat, event_id_map)
    write_ph2dt_inp(tmp_path / "ph2dt.inp", cfg)

    try:
        ph2dt = bin_path("ph2dt")
        hypodd = bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("ph2dt/hypoDD binaries not available")

    run([str(ph2dt), "ph2dt.inp"], cwd=tmp_path)
    assert (tmp_path / "event.sel").exists()
    assert (tmp_path / "station.sel").exists()

    # M9 doesn't do cross-correlation (M10.1/M10.3); copy the bundled dt.cc
    # in verbatim, per this task's acceptance criteria.
    (tmp_path / "dt.cc").write_text((EXAMPLE1_DIR / "dt.cc").read_text())

    model = pd.DataFrame(
        {
            "depth_km": [0.00, 0.25, 1.50, 2.50, 3.50, 5.00, 6.00, 9.00, 15.00, 25.00],
            "vp": [1.42, 3.24, 4.82, 5.36, 5.60, 5.65, 5.90, 6.15, 6.60, 8.00],
            "vs": [v / 1.73 for v in [1.42, 3.24, 4.82, 5.36, 5.60, 5.65, 5.90, 6.15, 6.60, 8.00]],
        }
    )
    write_hypodd_inp_1d(tmp_path / "hypoDD.inp", cfg, cfg.hypodd.weighting_rows, model)

    run([str(hypodd), "hypoDD.inp"], cwd=tmp_path, timeout=600)

    reloc_path = tmp_path / "hypoDD.reloc"
    assert reloc_path.exists()

    def _read_reloc(path: Path) -> dict[int, tuple[float, float, float, float]]:
        out = {}
        with open(path) as fh:
            for line in fh:
                fields = line.split()
                cuspid = int(fields[0])
                lat, lon, depth_km = float(fields[1]), float(fields[2]), float(fields[3])
                mag = float(fields[16])
                out[cuspid] = (lat, lon, depth_km, mag)
        return out

    written_reloc = _read_reloc(reloc_path)
    bundled_reloc = _read_reloc(EXAMPLE1_DIR / "hypoDD.reloc")

    # Same event set relocated.
    assert set(written_reloc) == set(bundled_reloc)

    # NOTE: this run uses IMOD=1 (variable vp/vs) and ISOLV=2 (LSQR), this
    # project's documented choice (docs/ARCHITECTURE.md §6c) -- the bundled
    # hypoDD.reloc may have been produced with a different historical
    # ISOLV/weighting scheme (examples/example1/hypoDD.inp.v1 ships an
    # ISOLV=2 variant alongside the current file's ISOLV=1), so exact
    # numeric reproduction isn't guaranteed. In practice this run reproduces
    # the bundled locations to within ~130 m for every event (verified
    # while developing this test); 0.5 km gives headroom for machine/
    # compiler differences without silently accepting a broken writer.
    import math

    km_per_deg_lat = 111.19
    for cuspid, (lat_w, lon_w, depth_w, mag_w) in written_reloc.items():
        lat_b, lon_b, depth_b, mag_b = bundled_reloc[cuspid]
        km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat_b))
        dlat_km = (lat_w - lat_b) * km_per_deg_lat
        dlon_km = (lon_w - lon_b) * km_per_deg_lon
        ddepth_km = depth_w - depth_b
        offset_km = math.sqrt(dlat_km**2 + dlon_km**2 + ddepth_km**2)
        assert offset_km < 0.5, f"event {cuspid} relocated {offset_km:.3f} km from bundled"
        assert mag_w == pytest.approx(mag_b, abs=1e-6)
