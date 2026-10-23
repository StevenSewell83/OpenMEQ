"""Tests for ``scripts/macban_man_picks.py`` (MacBan manual .evl/.pha -> QuakeML)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from obspy import UTCDateTime

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import macban_man_picks as mp  # noqa: E402

# Header blank seconds ('23:  .251'), a blank-padded minute (' 0: 2'), and
# picks after midnight.
EVL = """\
 My_Network            >> Local/Regional Event             Q=C     Manual mode
    DATE     ORIGIN-TIME  LATITUDE  LONGITUDE  DEPTH     ML   RMS   NL/NM  GAP
 2018/01/05 23:59:  .251 14.10309N 121.22331E   2.17    -.3  .040s   3/ 2  142
 ERRORS: (68%)      .067s    .198km     .161km   .43km   .1  IT 9      [2019/08/08 09:41:44]
 ERROR ELLIPSE: MajAx/2 =   .313km  MinAx/2 =   .229km  MajAxisAzimuth = 336
  STATION  CHN  DEL AZM EMG  INT  ARRIVAL_TIME  WAVE    O-C    WT     AMP     TAU  MAG  INFO
   1 MB-10 chz    2  57 131  0 1  23:59: 1.123  +IP1    .037  1.00  .161E+05    7  -.2   .34
   2 MB-10 che    2  57 131  0 1   0: 0: 1.710   IS1   -.011   .50                       .48
   3 MB-6A chz    4 126 102  0 4   0: 0: 0.111  -IP1    .035   .25                       .99
"""

PHA = """\
MB-10chz P+I 20180105 235901.123  1                     7      161 .04
MB-6Achz P-I 20180106  0 000.111  1                     6      100 .04
MB-10che     20180106  0            SI 0001.710  1               0 .00
"""


@pytest.fixture
def evl(tmp_path: Path) -> Path:
    p = tmp_path / "20180105_235958_NET_evt1.evl"
    p.write_text(EVL.replace("\n", "\r\n"))
    p.with_suffix(".pha").write_text(PHA)
    return p


def test_parse_evl(evl: Path) -> None:
    ev = mp.parse_evl(evl)
    assert ev.time == UTCDateTime("2018-01-05T23:59:00.251")
    assert (ev.lat, ev.lon, ev.depth_km, ev.ml, ev.quality) == (
        14.10309,
        121.22331,
        2.17,
        -0.3,
        "C",
    )
    assert (ev.n_phases, ev.n_stations, ev.gap, ev.iterations) == (3, 2, 142.0, 9)
    assert (ev.err_lat_km, ev.err_z_km, ev.ell_major_km, ev.ell_azimuth) == (
        0.198,
        0.43,
        0.313,
        336.0,
    )
    p, s, p2 = ev.arrivals
    assert (p.station, p.channel, p.phase, p.polarity, p.onset) == ("MB-10", "chz", "P", "+", "I")
    assert (p.residual, p.weight, p.amplitude, p.station_ml) == (0.037, 1.0, 16100.0, -0.2)
    assert s.time == UTCDateTime("2018-01-06T00:00:01.710")  # after midnight
    assert (s.phase, s.polarity, s.amplitude) == ("S", "", None)
    assert (p2.polarity, p2.weight, p2.int_col) == ("-", 0.25, "0 4")


def test_pha_matches_evl(evl: Path) -> None:
    ev = mp.parse_evl(evl)
    want = sorted((a.station, a.channel, round(a.time.timestamp, 3)) for a in ev.arrivals)
    assert sorted(mp.pha_times(evl.with_suffix(".pha"))) == want


def test_build_event(evl: Path) -> None:
    eid, event = mp.build_event(mp.parse_evl(evl))
    assert eid == "ev20180105T235900_251"
    origin = event.preferred_origin()
    assert origin.depth == pytest.approx(2170.0)
    assert origin.origin_uncertainty.max_horizontal_uncertainty == pytest.approx(313.0)
    assert [a.pick_id for a in origin.arrivals] == [p.resource_id for p in event.picks]
    assert event.picks[2].waveform_id.get_seed_string() == "MB.MB6A..EHZ"
    assert event.picks[2].polarity == "negative"
    assert event.preferred_magnitude().mag == -0.3
    assert len(event.station_magnitudes) == 1


def test_find_evl_dedups(tmp_path: Path, evl: Path) -> None:
    (tmp_path / "all").mkdir()
    (tmp_path / "all" / evl.name).write_text(EVL)
    (tmp_path / "20180105_235958_NET_reg1.evl").write_text(EVL)
    assert [p.name for p in mp.find_evl(tmp_path)] == [evl.name]
