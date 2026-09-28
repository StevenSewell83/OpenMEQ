"""Spike test for OpenMEQ-mev.1 (M5.1).

Hand-builds a minimal NonLinLoc control-file project end to end -- 1D layer
model -> Vel2Grid -> Grid2Time (P and S, including a borehole station at a
*negative* effective elevation, ARCHITECTURE.md §6) -> a synthetic event ->
NLLoc -- entirely with literal control-file text and `subprocess`/obspy, with
no dependency on any not-yet-built OpenMEQ NLLoc code (M5.2/M6.1 are out of
scope, per the issue). This proves the installed `external/bin` NonLinLoc
binaries and the installed obspy version work together as documented in
docs/nlloc_interface.md, which is the source of truth for the exact control
statement syntax used below (cite the relevant section there for any change).

Uses `openmeq.external.bin_path`/`run` (already-built shared helpers, per the
worker instructions) purely to locate/execute the binaries; the control-file
content and pipeline logic are written out by hand here, not delegated to any
OpenMEQ NLLoc-setup module.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth

from openmeq.external import ExternalToolMissing, bin_path, run

pytestmark = pytest.mark.requires_nlloc


def _require_nlloc_tools() -> dict[str, Path]:
    try:
        return {name: bin_path(name) for name in ("Vel2Grid", "Grid2Time", "NLLoc")}
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


# 1D layer file, exactly the ARCHITECTURE.md §6a format/example.
VEL1D = """\
# depth_top_km  vp_km_s  vs_km_s  rho_g_cc
0.0   3.50  2.00  2.3
1.0   4.50  2.60  2.5
3.0   5.80  3.35  2.7
"""

# Geographic origin for the TRANS SIMPLE projection (arbitrary).
LAT_ORIG = 45.0
LON_ORIG = 10.0

# Stations: label, lat, lon, elevation_km (effective elevation per
# ARCHITECTURE.md §6: elevation_m - depth_m; BH1 is a 500 m borehole at
# elevation_m=0, so its effective elevation is -0.5 km).
STATIONS = [
    ("ST1", 45.020, 10.020, 0.5),
    ("ST2", 44.985, 10.030, 0.3),
    ("ST3", 45.030, 9.975, 0.8),
    ("ST4", 44.978, 9.965, 0.1),
    ("BH1", 45.005, 10.005, -0.5),
]

# Known true hypocenter used to build the synthetic event.
TRUE_LAT = 45.008
TRUE_LON = 10.008
TRUE_DEPTH_KM = 3.0
TRUE_ORIGIN_TIME = UTCDateTime("2025-06-15T03:00:00")


VEL2GRID_IN = f"""\
CONTROL 1 54321
TRANS SIMPLE {LAT_ORIG} {LON_ORIG} 0.0

VGOUT ./model/vel1d
VGTYPE P
VGTYPE S
VGGRID  2 401 161  0.0 0.0 -3.0  0.05 0.05 0.05  SLOW_LEN

LAYER    0.0     3.50   0    2.00   0    2.3   0
LAYER    1.0     4.50   0    2.60   0    2.5   0
LAYER    3.0     5.80   0    3.35   0    2.7   0
"""


def _grid2time_in(wave_type: str) -> str:
    srce_lines = "\n".join(
        f"GTSRCE  {label}  LATLON  {lat}  {lon}  0.0  {elev_km}"
        for label, lat, lon, elev_km in STATIONS
    )
    return f"""\
CONTROL 1 54321
TRANS SIMPLE {LAT_ORIG} {LON_ORIG} 0.0

GTFILES  ./model/vel1d  ./time/vel1d  {wave_type}
GTMODE GRID2D ANGLES_NO

{srce_lines}

GT_PLFD  1.0e-3  0
"""


NLLOC_IN = f"""\
CONTROL 1 54321
TRANS SIMPLE {LAT_ORIG} {LON_ORIG} 0.0

LOCSIG OpenMEQ-mev.1 spike test
LOCCOM minimal worked example, borehole station BH1 at negative elevation

LOCFILES ./obs/from_obspy.obs NLLOC_OBS ./time/vel1d ./loc/mini

LOCHYPOUT SAVE_NLLOC_ALL

LOCSEARCH OCT 10 10 10 0.01 5000 2000 0 1

LOCGRID  101 101 61  -5.0 -5.0 -1.0  0.1 0.1 0.1  PROB_DENSITY SAVE

LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1.0 6 -1.0 1

LOCGAU 0.1 0.0
LOCGAU2 0.01 0.05 2.0

LOCPHASEID  P   P
LOCPHASEID  S   S

LOCQUAL2ERR 0.05 0.1 0.5 1.0 99999.9

LOCANGLES ANGLES_NO 5
"""


def _parse_time2eq_obs(path: Path) -> dict[tuple[str, str], float]:
    """Parse a Time2EQ-generated NLLOC_OBS file into {(station, phase): travel_time_s}."""
    times: dict[tuple[str, str], float] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("PUBLIC_ID"):
            continue
        fields = line.split()
        station, phase = fields[0], fields[4]
        times[(station, phase)] = float(fields[8])
    return times


def test_nlloc_end_to_end_recovers_hypocenter(tmp_path):
    """Vel2Grid -> Grid2Time (P, S; incl. borehole) -> Time2EQ -> obspy
    NLLOC_OBS write -> NLLoc -> obspy NLLOC_HYP read, and check the recovered
    location is within 5 km of the known synthetic hypocenter.
    """
    _require_nlloc_tools()

    for sub in ("model", "time", "obs", "loc"):
        (tmp_path / sub).mkdir()

    (tmp_path / "vel1d.txt").write_text(VEL1D)
    (tmp_path / "vel2grid.in").write_text(VEL2GRID_IN)

    run([str(bin_path("Vel2Grid")), "vel2grid.in"], cwd=tmp_path)
    assert (tmp_path / "model" / "vel1d.P.mod.buf").exists()
    assert (tmp_path / "model" / "vel1d.S.mod.buf").exists()

    for wave_type in ("P", "S"):
        ctrl = tmp_path / f"grid2time_{wave_type}.in"
        ctrl.write_text(_grid2time_in(wave_type))
        run([str(bin_path("Grid2Time")), ctrl.name], cwd=tmp_path)

    # Every station's P and S time grid exists, including the borehole
    # (BH1, negative effective elevation) -- confirms GTSRCE accepts a
    # negative "elev" and Grid2Time runs it through without error.
    for label, *_ in STATIONS:
        for wave_type in ("P", "S"):
            assert (tmp_path / "time" / f"vel1d.{wave_type}.{label}.time.buf").exists()

    # Generate exact synthetic travel times from the installed Grid2Time
    # output with Time2EQ (zero timing error), so the "true" arrival times
    # are self-consistent with the same travel-time engine NLLoc will invert.
    eqsta_lines = "\n".join(
        f"EQSTA  {label}  {phase}  GAU 0.00   GAU 0.05"
        for label, *_ in STATIONS
        for phase in ("P", "S")
    )
    time2eq_in = f"""\
CONTROL 1 54321
TRANS SIMPLE {LAT_ORIG} {LON_ORIG} 0.0

EQFILES ./time/vel1d ./obs/synth.obs
EQMODE SRCE_TO_STA
EQSRCE  TrueEvt  LATLON  {TRUE_LAT}  {TRUE_LON}  {TRUE_DEPTH_KM}  0.0
EQVPVS  -1.0

{eqsta_lines}

EQQUAL2ERR 0.05 0.1 0.5 1.0 99999.9
"""
    (tmp_path / "time2eq.in").write_text(time2eq_in)
    run([str(bin_path("Time2EQ")), "time2eq.in"], cwd=tmp_path)
    travel_times = _parse_time2eq_obs(tmp_path / "obs" / "synth.obs")
    assert len(travel_times) == 2 * len(STATIONS)

    # Build the NLLoc input phase file with obspy's own NLLOC_OBS writer
    # (confirms Catalog.write(..., format="NLLOC_OBS") produces a file NLLoc
    # accepts, per docs/nlloc_interface.md point 5).
    event = Event()
    original_picks = []
    for (station, phase), tt in travel_times.items():
        pick = Pick()
        pick.time = TRUE_ORIGIN_TIME + tt
        pick.waveform_id = WaveformStreamID(
            network_code="XX",
            station_code=station,
            channel_code="HHZ" if phase == "P" else "HHN",
        )
        pick.phase_hint = phase
        pick.time_errors.uncertainty = 0.05
        pick.onset = "impulsive"
        event.picks.append(pick)
        original_picks.append(pick)
    Catalog(events=[event]).write(str(tmp_path / "obs" / "from_obspy.obs"), format="NLLOC_OBS")

    (tmp_path / "nlloc.in").write_text(NLLOC_IN)
    run([str(bin_path("NLLoc")), "nlloc.in"], cwd=tmp_path)

    hyp_files = glob.glob(str(tmp_path / "loc" / "mini.[0-9]*.grid0.loc.hyp"))
    assert len(hyp_files) == 1, f"expected exactly one per-event .hyp file, found {hyp_files}"

    import obspy  # local import: keep the obspy dependency visible at the call site

    result_from_scratch = obspy.read_events(hyp_files[0], format="NLLOC_HYP")
    assert len(result_from_scratch) == 1

    # Also confirm the picks= round trip (docs/nlloc_interface.md point 5):
    # arrivals re-link to the *original* Pick objects by station_code +
    # phase_hint.
    result = obspy.read_events(hyp_files[0], format="NLLOC_HYP", picks=original_picks)
    event_out = result[0]
    origin = event_out.origins[0]

    assert len(event_out.picks) == len(original_picks)
    assert all(pick in original_picks for pick in event_out.picks)

    horiz_dist_km = (
        gps2dist_azimuth(TRUE_LAT, TRUE_LON, origin.latitude, origin.longitude)[0] / 1000.0
    )
    depth_diff_km = abs(origin.depth / 1000.0 - TRUE_DEPTH_KM)
    total_dist_km = (horiz_dist_km**2 + depth_diff_km**2) ** 0.5

    assert total_dist_km < 5.0, (
        f"recovered location {total_dist_km:.3f} km from true hypocenter "
        f"(horiz {horiz_dist_km:.3f} km, depth diff {depth_diff_km:.3f} km)"
    )
