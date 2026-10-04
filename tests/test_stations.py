"""Tests for ``openmeq.stations`` (M1.2, OpenMEQ-5nl.2).

Uses the ``synthetic_project`` fixture (8 stations: 6 surface ``HH?``, 2
borehole ``DP?``) for the merge/table/round-trip tests, and small hand-built
config/streams/inventories for the channel-map, rotation and coverage tests
(ARCHITECTURE.md §6, §2, §3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime, read_inventory
from obspy.core.inventory import Channel, Inventory, Network, Station

from openmeq.config import ConfigError, StationsConfig, load_config
from openmeq.paths import RunPaths
from openmeq.stations import (
    MAX_STATION_CODE_LEN,
    STATION_TABLE_COLUMNS,
    apply_channel_map,
    check_coverage,
    load_inventory,
    station_table,
    write_merged_inventory,
)

NETWORK = "XX"


def _make_trace(net, sta, loc, cha, data, starttime=None, sampling_rate=100.0) -> Trace:
    tr = Trace(data=np.asarray(data, dtype=np.float64))
    tr.stats.network = net
    tr.stats.station = sta
    tr.stats.location = loc
    tr.stats.channel = cha
    tr.stats.sampling_rate = sampling_rate
    tr.stats.starttime = starttime or UTCDateTime(2025, 1, 1)
    return tr


# ---------------------------------------------------------------------------
# load_inventory / station_table against the synthetic fixture
# ---------------------------------------------------------------------------


def test_merge_all_sources(synthetic_project):
    config_path, truth = synthetic_project
    cfg = load_config(config_path)

    inv = load_inventory(cfg.stations)
    stations = [sta for net in inv.networks for sta in net.stations]
    assert len(stations) == 8

    table = station_table(cfg.stations)
    assert list(table.columns) == STATION_TABLE_COLUMNS
    assert len(table) == 8
    # Borehole depths carried through from the CSV.
    bh01 = table[table["station"] == "BH01"].iloc[0]
    assert bh01["depth_m"] == pytest.approx(300.0)
    surface = table[table["station"] == "STA01"].iloc[0]
    assert surface["depth_m"] == pytest.approx(0.0)


def test_csv_only_station_included(tmp_path, synthetic_project):
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)

    # A station that exists only in a CSV, not in the synthetic StationXML.
    csv_path = tmp_path / "extra_stations.csv"
    pd.DataFrame(
        [
            {
                "net": NETWORK,
                "sta": "CSVONLY",
                "loc": "00",
                "lat": 37.9,
                "lon": -122.3,
                "elev_m": 123.0,
                "depth_m": 42.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    stations_cfg = StationsConfig(
        stationxml=cfg.stations.stationxml,
        csv=csv_path,
    )

    inv = load_inventory(stations_cfg)
    codes = {sta.code for net in inv.networks for sta in net.stations}
    assert "CSVONLY" in codes
    assert len(codes) == 9  # 8 from XML + 1 CSV-only

    table = station_table(stations_cfg)
    row = table[table["station"] == "CSVONLY"].iloc[0]
    assert row["depth_m"] == pytest.approx(42.0)
    assert row["latitude"] == pytest.approx(37.9)
    assert row["longitude"] == pytest.approx(-122.3)
    assert row["elevation_m"] == pytest.approx(123.0)


def test_station_code_too_long_fails_at_load(tmp_path):
    csv_path = tmp_path / "stations.csv"
    pd.DataFrame(
        [
            {
                "net": NETWORK,
                "sta": "A" * (MAX_STATION_CODE_LEN + 1),
                "loc": "00",
                "lat": 0.0,
                "lon": 0.0,
                "elev_m": 0.0,
                "depth_m": 0.0,
            }
        ]
    ).to_csv(csv_path, index=False)
    cfg = StationsConfig(csv=csv_path)
    with pytest.raises(ConfigError, match="7-character"):
        load_inventory(cfg)


def _epoch(code, start, end=None, loc="10") -> Channel:
    return Channel(
        code=code,
        location_code=loc,
        latitude=0.0,
        longitude=0.0,
        elevation=0.0,
        depth=0.0,
        sample_rate=100.0,
        start_date=UTCDateTime(start),
        end_date=None if end is None else UTCDateTime(end),
    )


def test_every_channel_epoch_kept_and_overlaps_closed(tmp_path, caplog):
    """OpenMEQ-hud: all epochs of one (loc, code) survive the merge, and an
    epoch left open (or ending after the next one starts) is ended at the
    next epoch's start, so exactly one epoch is active at any time."""
    epochs = [
        _epoch("EHZ", "2007-05-20", "2011-03-06"),
        _epoch("EHZ", "2011-06-20"),  # current, open
        _epoch("EHZ", "2008-01-01"),  # superseded but never closed
        _epoch("EHN", "2009-01-01", "2030-01-01"),  # ends after the next one starts
        _epoch("EHN", "2012-01-01"),
    ]
    inv = Inventory(
        networks=[
            Network(
                code=NETWORK,
                stations=[
                    Station(
                        code="ARAZ", latitude=0.0, longitude=0.0, elevation=0.0, channels=epochs
                    )
                ],
            )
        ],
        source="test",
    )
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    inv.write(str(xml_dir / "ARAZ.xml"), format="STATIONXML")

    merged = load_inventory(StationsConfig(stationxml_dir=xml_dir))
    channels = merged[0][0].channels
    assert len(channels) == 5

    def _active(code, t):
        return [c for c in channels if c.code == code and c.is_active(UTCDateTime(t))]

    (ehz_2026,) = _active("EHZ", "2026-02-15")
    assert ehz_2026.start_date == UTCDateTime("2011-06-20")
    (ehz_2010,) = _active("EHZ", "2010-01-01")
    assert ehz_2010.start_date == UTCDateTime("2008-01-01")
    assert ehz_2010.end_date == UTCDateTime("2011-06-20")
    (ehn_2026,) = _active("EHN", "2026-02-15")
    assert ehn_2026.start_date == UTCDateTime("2012-01-01")
    assert sum("overlaps the next epoch" in r.getMessage() for r in caplog.records) >= 2


def test_station_code_unique_across_networks_fails_at_load(tmp_path):
    inv = Inventory(
        networks=[
            Network(
                code="XX", stations=[Station(code="DUP", latitude=0, longitude=0, elevation=0)]
            ),
            Network(
                code="YY", stations=[Station(code="DUP", latitude=1, longitude=1, elevation=0)]
            ),
        ]
    )
    xml_path = tmp_path / "dup.xml"
    inv.write(str(xml_path), format="STATIONXML")
    cfg = StationsConfig(stationxml=xml_path)
    with pytest.raises(ConfigError, match="unique across networks"):
        load_inventory(cfg)


# ---------------------------------------------------------------------------
# apply_channel_map
# ---------------------------------------------------------------------------


def test_apply_channel_map_relabels_and_preserves_original():
    cfg = StationsConfig(channel_map={"DPZ": "Z", "DP1": "1", "DP2": "2"})
    stream = Stream(
        [
            _make_trace("XX", "BH01", "00", "DPZ", [1, 2, 3]),
            _make_trace("XX", "BH01", "00", "DP1", [4, 5, 6]),
            _make_trace("XX", "BH01", "00", "DP2", [7, 8, 9]),
        ]
    )
    out = apply_channel_map(stream, cfg)
    by_orig = {tr.stats.openmeq_original_channel: tr for tr in out}
    assert set(by_orig) == {"DPZ", "DP1", "DP2"}
    assert by_orig["DPZ"].stats.channel == "DPZ"
    assert by_orig["DP1"].stats.channel == "DP1"
    assert by_orig["DP2"].stats.channel == "DP2"
    # Original stream is untouched.
    assert stream[0].stats.channel == "DPZ"
    assert not hasattr(stream[0].stats, "openmeq_original_channel")


def test_apply_channel_map_collision_keeps_first(caplog):
    cfg = StationsConfig(channel_map={"DP1": "Z", "DP2": "Z"})
    stream = Stream(
        [
            _make_trace("XX", "WEIRD", "00", "DP1", [1, 2, 3]),
            _make_trace("XX", "WEIRD", "00", "DP2", [4, 5, 6]),
        ]
    )
    out = apply_channel_map(stream, cfg)
    mapped = [tr for tr in out if tr.stats.channel == "DPZ"]
    assert len(mapped) == 1
    assert mapped[0].stats.openmeq_original_channel == "DP1"
    unmapped = [tr for tr in out if tr.stats.channel == "DP2"]
    assert len(unmapped) == 1
    assert not hasattr(unmapped[0].stats, "openmeq_original_channel")


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_with_known_orientation():
    cfg = StationsConfig(
        channel_map={"DP1": "1", "DP2": "2"},
        orientations={"XX.BH01.00.DP1": 0.0, "XX.BH01.00.DP2": 90.0},
    )
    c1 = [1.0, -2.0, 3.5, 0.0]
    c2 = [4.0, 5.0, -6.5, 2.0]
    stream = Stream(
        [
            _make_trace("XX", "BH01", "00", "DP1", c1),
            _make_trace("XX", "BH01", "00", "DP2", c2),
        ]
    )
    out = apply_channel_map(stream, cfg)
    by_channel = {tr.stats.channel: tr for tr in out}
    assert set(by_channel) == {"DPN", "DPE"}
    # Orthogonal 0deg/90deg reduces to the identity: N == c1, E == c2.
    np.testing.assert_allclose(by_channel["DPN"].data, c1, atol=1e-9)
    np.testing.assert_allclose(by_channel["DPE"].data, c2, atol=1e-9)
    assert by_channel["DPN"].stats.openmeq_original_channel == "DP1"
    assert by_channel["DPE"].stats.openmeq_original_channel == "DP2"


def test_rotation_skipped_without_known_orientation():
    cfg = StationsConfig(channel_map={"DP1": "1", "DP2": "2"}, orientations={})
    stream = Stream(
        [
            _make_trace("XX", "BH02", "00", "DP1", [1.0, 2.0, 3.0]),
            _make_trace("XX", "BH02", "00", "DP2", [4.0, 5.0, 6.0]),
        ]
    )
    out = apply_channel_map(stream, cfg)
    codes = sorted(tr.stats.channel for tr in out)
    assert codes == ["DP1", "DP2"]


def test_rotation_skipped_for_lone_horizontal():
    cfg = StationsConfig(
        channel_map={"DP1": "1"},
        orientations={"XX.BH01.00.DP1": 0.0},
    )
    stream = Stream([_make_trace("XX", "BH01", "00", "DP1", [1.0, 2.0, 3.0])])
    out = apply_channel_map(stream, cfg)
    assert out[0].stats.channel == "DP1"


# ---------------------------------------------------------------------------
# check_coverage
# ---------------------------------------------------------------------------


def _small_inventory() -> Inventory:
    station = Station(
        code="STA01",
        latitude=1.0,
        longitude=2.0,
        elevation=3.0,
        channels=[
            Channel(
                code="HHZ",
                location_code="00",
                latitude=1.0,
                longitude=2.0,
                elevation=3.0,
                depth=0.0,
            ),
            Channel(
                code="HHN",
                location_code="00",
                latitude=1.0,
                longitude=2.0,
                elevation=3.0,
                depth=0.0,
            ),
        ],
    )
    return Inventory(networks=[Network(code="XX", stations=[station])])


def test_check_coverage_detects_mismatch():
    inv = _small_inventory()
    # Data has HHZ (matches) and an extra channel with no metadata (HHE);
    # metadata has HHN with no data.
    seed_ids = ["XX.STA01.00.HHZ", "XX.STA01.00.HHE"]
    report = check_coverage(seed_ids, inv)
    assert report["data_without_metadata"] == ["XX.STA01.00.HHE"]
    assert report["metadata_without_data"] == ["XX.STA01.00.HHN"]


def test_check_coverage_accepts_plain_iterable_not_data_index():
    # Per the orchestrator note: check_coverage must not require an
    # openmeq.index.DataIndex -- any iterable of seed_id strings works,
    # e.g. a generator or a set, standing in for DataIndex.seed_ids().
    inv = _small_inventory()

    def seed_id_gen():
        yield "XX.STA01.00.HHZ"

    report = check_coverage(seed_id_gen(), inv)
    assert report["data_without_metadata"] == []
    assert report["metadata_without_data"] == ["XX.STA01.00.HHN"]


# ---------------------------------------------------------------------------
# write_merged_inventory
# ---------------------------------------------------------------------------


def test_write_merged_inventory_files(tmp_path, synthetic_project):
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    inv = load_inventory(cfg.stations)
    write_merged_inventory(inv, paths, cfg.stations)

    assert paths.stations_xml.exists()
    round_tripped = read_inventory(str(paths.stations_xml))
    assert sum(len(net.stations) for net in round_tripped.networks) == 8

    assert paths.stations_csv.exists()
    df = pd.read_csv(paths.stations_csv)
    assert list(df.columns) == STATION_TABLE_COLUMNS
    assert len(df) == 8
