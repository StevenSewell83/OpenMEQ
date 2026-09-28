#!/usr/bin/env python
"""Manual/smoke check for the ObsPyck local-file launch path (requires_gui).

This is the reusable artifact from the OpenMEQ-c3l.1 spike: it reproduces
steps 3-5 of that spike (launch ObsPyck offline against a local chunk +
StationXML + automatic QuakeML, run NLLoc via the pluginpath/nlloc +
VELOCITY_MODELS layout, make a pick change, save locally) end to end, so
M7.2/M7.3 can sanity-check their own config generator / batch wrapper
against a known-good baseline without re-deriving the spike from scratch.

Requires a display or QT_QPA_PLATFORM=offscreen (default set below if
unset), PyQt5 + obspyck importable, and NLLoc/Vel2Grid/Grid2Time reachable
via openmeq.external.bin_path (set OPENMEQ_EXTERNAL_DIR if needed). Skips
(exit 0, prints "SKIP: ...") when any of these are unavailable, the same
way a pytest.mark.requires_gui test would be auto-skipped.

NOTE: tests/conftest.py's `synthetic_project` fixture (OpenMEQ-qlg.8) did
not exist yet when this spike ran, so this script builds its own tiny
throwaway 2-station/1D-model fixture inline instead of depending on it.
M7.3 should point this script (or its logic) at the real fixture once it
lands, to confirm nothing about the real dataset's conventions (station
naming, channel codes, event id format) breaks these findings.

Two ObsPyck 0.8.2 bugs discovered during the spike are worked around here
by monkeypatching (documented at length in docs/obspyck_notes.md):

1. ``obspyck.util.gk2lonlat`` hard-codes EPSG:31468 (Gauss-Krueger zone 4,
   Bavaria) regardless of the NLLoc TRANS actually used, corrupting
   Origin.longitude/latitude for any non-Bavaria project. Patched in BOTH
   ``obspyck.util`` and ``obspyck.obspyck`` (the latter did
   ``from .util import gk2lonlat``, binding its own name).
2. The NonLinLoc build this project uses (v7.1.x) inserts a literal ``>``
   token in each NLLoc PHASE output line that ObsPyck 0.8.2's
   ``loadNLLocOutput`` (fixed whitespace-split column indices) does not
   expect, crashing with ``ValueError`` on every "do NLLoc" click. Worked
   around by wrapping ``PROGRAMS['nlloc']['Call']`` to strip stray ``>``
   tokens from the summary file before ObsPyck re-reads it.

Usage::

    QT_QPA_PLATFORM=offscreen OPENMEQ_EXTERNAL_DIR=/path/to/external \\
        python scripts/obspyck_spike_smoke.py
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REF_LAT, REF_LON = -38.00, 176.00
STATIONS = [
    # name, lat, lon, elev_m, P_offset_s, S_offset_s
    ("ST1", -38.020000, 176.010000, 100.0, 10.10, 11.40),
    ("ST2", -37.985000, 175.975000, 50.0, 10.40, 11.90),
]
NET = "OQ"
SR = 100.0
DUR = 30.0


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}")
    sys.exit(0)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_prereqs():
    try:
        import obspyck  # noqa: F401
        import PyQt5  # noqa: F401
    except ImportError as exc:
        _skip(f"obspyck/PyQt5 not importable: {exc}")

    try:
        from openmeq.external import ExternalToolMissing, bin_path

        for tool in ("NLLoc", "Vel2Grid", "Grid2Time"):
            bin_path(tool)
    except ExternalToolMissing as exc:
        _skip(f"NLLoc toolchain not available: {exc}")
    except ImportError as exc:
        _fail(f"openmeq.external not importable (is this the repo's own venv?): {exc}")


def build_fixture(root: Path):
    """Build a throwaway local event (mseed + StationXML + single-event
    QuakeML) and a matching NLLoc pluginpath/VELOCITY_MODELS layout."""
    import numpy as np
    from obspy import Stream, Trace, UTCDateTime
    from obspy.core.event import Catalog, Event, Origin, Pick, ResourceIdentifier, WaveformStreamID
    from obspy.core.inventory import Channel, Inventory, Network, Site, Station
    from obspy.core.inventory.response import InstrumentSensitivity, Response

    from openmeq.external import bin_path, run

    t0 = UTCDateTime("2026-01-01T00:00:00.000000Z")
    event_id = "ev20260101T000010_100"

    st = Stream()
    npts = int(DUR * SR)
    rng = np.random.default_rng(42)
    for name, *_rest, p_off, s_off in STATIONS:
        for cha in ("HHZ", "HHN", "HHE"):
            data = rng.normal(0, 5, npts).astype(np.float32)
            off = p_off if cha == "HHZ" else s_off
            idx = int(off * SR)
            if idx < npts:
                data[idx : idx + 20] += np.linspace(250, 0, min(20, npts - idx))
            tr = Trace(data=data)
            tr.stats.update(
                dict(
                    network=NET,
                    station=name,
                    location="",
                    channel=cha,
                    sampling_rate=SR,
                    starttime=t0,
                )
            )
            st.append(tr)
    mseed_path = root / f"{event_id}.mseed"
    st.write(str(mseed_path), format="MSEED")

    net = Network(code=NET, stations=[])
    for name, lat, lon, elev, *_ in STATIONS:
        channels = []
        for cha, azimuth, dip in (("HHZ", 0, -90), ("HHN", 0, 0), ("HHE", 90, 0)):
            resp = Response(
                instrument_sensitivity=InstrumentSensitivity(
                    value=1.0, frequency=1.0, input_units="M/S", output_units="COUNTS"
                )
            )
            channels.append(
                Channel(
                    code=cha,
                    location_code="",
                    latitude=lat,
                    longitude=lon,
                    elevation=elev,
                    depth=0.0,
                    azimuth=azimuth,
                    dip=dip,
                    sample_rate=SR,
                    response=resp,
                )
            )
        net.stations.append(
            Station(
                code=name,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                site=Site(name="OpenMEQ spike smoke"),
                channels=channels,
                start_date=t0,
            )
        )
    stations_path = root / "stations.xml"
    inv = Inventory(networks=[net], source="OpenMEQ-c3l.1 spike smoke")
    inv.write(str(stations_path), format="STATIONXML")

    picks = []
    for name, *_rest, p_off, s_off in STATIONS:
        for phase, off in (("P", p_off), ("S", s_off)):
            wid = WaveformStreamID(
                network_code=NET,
                station_code=name,
                location_code="",
                channel_code="HHZ" if phase == "P" else "HHN",
            )
            picks.append(
                Pick(
                    time=t0 + off,
                    waveform_id=wid,
                    phase_hint=phase,
                    method_id=ResourceIdentifier("smi:openmeq/phasenet"),
                    evaluation_mode="automatic",
                )
            )
    origin = Origin(
        time=t0 + 10.0,
        longitude=REF_LON + 0.005,
        latitude=REF_LAT - 0.003,
        depth=4000.0,
        method_id=ResourceIdentifier("smi:openmeq/nlloc/passB"),
    )
    ev = Event(resource_id=ResourceIdentifier(f"smi:openmeq/event/{event_id}"))
    ev.picks = picks
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    event_path = root / f"{event_id}.xml"
    Catalog(events=[ev]).write(str(event_path), format="QUAKEML")

    # --- NLLoc pluginpath / VELOCITY_MODELS (throwaway 1D model+grid) ---
    plugin = root / "pluginpath"
    (plugin / "nlloc").mkdir(parents=True)
    (plugin / "hyp_2000").mkdir()  # setup_external_programs copies these
    (plugin / "focmec").mkdir()  # unconditionally too; must exist even if unused

    import platform

    exe_name = "__".join(["NLLoc", platform.system(), platform.architecture()[0]])
    shutil.copy(bin_path("NLLoc"), plugin / "nlloc" / exe_name)
    os.chmod(plugin / "nlloc" / exe_name, 0o755)

    build_dir = root / "nlloc_build"
    (build_dir / "model").mkdir(parents=True)
    (build_dir / "time").mkdir(parents=True)

    vg_in = build_dir / "vg.in"
    vg_in.write_text(
        "CONTROL 1 54321\n"
        f"TRANS  SIMPLE  {REF_LAT:.2f} {REF_LON:.2f}  0.0\n"
        "VGOUT  ./model/layer\nVGTYPE P\nVGTYPE S\n"
        "VGGRID  2 41 21  0.0 0.0 -2.0  0.5 0.5 0.5  SLOW_LEN\n"
        "LAYER    0.0     3.50    0    2.00    0    2.3\n"
        "LAYER    1.0     4.50    0    2.60    0    2.5\n"
        "LAYER    3.0     5.80    0    3.35    0    2.7\n"
    )
    stations_txt = build_dir / "stations.txt"
    stations_txt.write_text(
        "\n".join(
            f"GTSRCE  {name}  LATLON  {lat:.6f}  {lon:.6f}  0  {elev / 1000.0:.3f}"
            for name, lat, lon, elev, *_ in STATIONS
        )
        + "\n"
    )
    run([str(bin_path("Vel2Grid")), "vg.in"], cwd=build_dir)
    for phase in ("P", "S"):
        gt_in = build_dir / f"gt_{phase}.in"
        gt_in.write_text(
            "CONTROL 1 54321\n"
            f"TRANS  SIMPLE  {REF_LAT:.2f} {REF_LON:.2f}  0.0\n"
            f"GTFILES  ./model/layer  ./time/layer  {phase}\n"
            "GTMODE GRID2D ANGLES_YES\n"
            "INCLUDE stations.txt\n"
            "GT_PLFD  1.0e-3  0\n"
        )
        run([str(bin_path("Grid2Time")), f"gt_{phase}.in"], cwd=build_dir)

    vmodels = plugin / "VELOCITY_MODELS" / "layer_time"
    shutil.copytree(build_dir / "time", vmodels)

    control = plugin / "nlloc" / "locate_BY.nlloc"
    control.write_text(
        "CONTROL 1 54321\n"
        f"TRANS  SIMPLE  {REF_LAT:.2f} {REF_LON:.2f}  0.0\n"
        "LOCSIG OpenMEQ spike smoke\nLOCCOM throwaway 1D model\n"
        "LOCFILES  nlloc.obs  NLLOC_OBS  ../VELOCITY_MODELS/layer_time/layer  nlloc\n"
        "LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_HYPOINV_SUM\n"
        "LOCSEARCH  OCT 10 10 4 0.01 20000 5000 0 1\n"
        "LOCGRID  41 41 21  -10.0 -10.0 -2.0  0.5 0.5 0.5   PROB_DENSITY  SAVE\n"
        "LOCMETH EDT_OT_WT 9999.0 4 -1 -1 1.68 6 -1.0 1\n"
        "LOCGAU 0.2 0.0\nLOCGAU2 0.02 0.05 2.0\n"
        "LOCPHASEID  P   P p Pn Pg\nLOCPHASEID  S   S s Sn Sg\n"
        "LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9\n"
        "LOCANGLES ANGLES_YES 5\n"
        "LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9\n"
    )

    cfg_path = root / "obspyck.cfg"
    _write_config(cfg_path, plugin)

    return {
        "mseed": mseed_path,
        "stations": stations_path,
        "event": event_path,
        "config": cfg_path,
        "event_id": event_id,
        "t0": t0,
    }


def _write_config(path: Path, pluginpath: Path):
    path.write_text(
        f"""[base]
duration = {int(DUR)}
starttime_offset = 0
pluginpath = {pluginpath}
merge =
event_server =
ignore_gse2_chksum_error = true
verbosity = normal
zero_mean = true
normalization = true
no_metadata = false
spectrogram_colormap = viridis
magnitude_pick_color = green
magnitude_picker_width = 10
physical_units = velocity
taper_max_length = 5
taper_max_percentage = 0.05
taper_type = cosine

[misc]
allow_multiple_picks_with_same_seed_id = false
scrollWheelPercentage = 0.30
scrollWheelInvert = false
merge_events_in_catalog = false

[seismic_phases]
P = red
S = blue

[seed_id_lookup]

[gui_defaults]
filter = false
filter_type = bandpass
filter_highpass = 0.01
filter_lowpass = 0.05
filter_corners = 3
sta = 0.5
lta = 10.0
ar_f1 = 1.0
ar_f2 = 20.0
ar_sta_p = 0.1
ar_lta_p = 1.0
ar_sta_s = 1.0
ar_lta_s = 4.0
ar_m_p = 2
ar_m_s = 8
ar_l_p = 0.1
ar_l_s = 0.2
spectrogram_wlen = 0.4
spectrogram_overlap = 0.9
spectrogram_log = false

[keys]
setPick = a
setPickError = s
delPick = q
setMagMin = a
setMagMax = s
delMagMinMax = q
switchPhase = control
prevStream = y
nextStream = x
switchWheelZoomAxis = shift
scrollWheelZoom = alt
setWeight0 = 0
setWeight1 = 1
setWeight2 = 2
setWeight3 = 3
setPolU = +
setPolD = -
setOnsetI = i
setOnsetE = e

[nonlinloc]
default_pick_uncertainty = 0.05

[matplotlibrc]
lines.linewidth = 1.0
font.size = 10

[ar_picker]
f1 = 1.0
f2 = 20.0
lta_p = 1.0
sta_p = 0.1
lta_s = 4.0
sta_s = 1.0
m_p = 2
m_s = 8
l_p = 0.1
l_s = 0.2
"""
    )


def apply_bugfix_patches(lat0: float, lon0: float):
    """Monkeypatch the two ObsPyck 0.8.2 bugs found in the OpenMEQ-c3l.1
    spike. See the module docstring and docs/obspyck_notes.md."""
    import obspyck.obspyck as op
    import obspyck.util as ou

    r_km = 6371.0

    def _fixed_gk2lonlat(x, y, m_to_km=True):
        dlat = (y / r_km) * (180.0 / math.pi)
        dlon = (x / (r_km * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
        return (lon0 + dlon, lat0 + dlat)

    ou.gk2lonlat = _fixed_gk2lonlat
    op.gk2lonlat = _fixed_gk2lonlat

    prog_dict = ou.PROGRAMS["nlloc"]
    original_call = prog_dict["Call"]

    def _wrapped_call(prog_dict, controlfilename):
        result = original_call(prog_dict, controlfilename)
        summary = prog_dict["files"]["summary"]
        if os.path.isfile(summary):
            with open(summary) as f:
                lines = f.readlines()
            in_phase = False
            fixed = []
            for line in lines:
                if line.startswith("PHASE ID"):
                    in_phase = True
                    fixed.append(line)
                elif line.startswith("END_PHASE"):
                    in_phase = False
                    fixed.append(line)
                elif in_phase and line.strip():
                    parts = [p for p in line.split() if p != ">"]
                    fixed.append(" ".join(parts) + "\n")
                else:
                    fixed.append(line)
            with open(summary, "w") as f:
                f.writelines(fixed)
        return result

    prog_dict["Call"] = _wrapped_call


def run_smoke() -> bool:
    check_prereqs()

    root = Path(tempfile.mkdtemp(prefix="obspyck_smoke_"))
    ok = True
    try:
        fx = build_fixture(root)
        work = root / "work"
        work.mkdir()
        os.chdir(work)

        import optparse
        from configparser import ConfigParser

        import obspyck.obspyck as op
        import obspyck.util as ou
        from PyQt5 import QtWidgets

        parser = optparse.OptionParser()
        for opt_args, opt_kwargs in ou.COMMANDLINE_OPTIONS:
            parser.add_option(*opt_args, **opt_kwargs)
        argv = [
            "-c",
            str(fx["config"]),
            "-t",
            str(fx["t0"]),
            "-d",
            str(int(DUR)),
            "--event",
            str(fx["event"]),
            str(fx["mseed"]),
            str(fx["stations"]),
        ]
        options, args = parser.parse_args(argv)

        config = ConfigParser(allow_no_value=True)
        config.optionxform = str
        config.read(options.config_file)
        ou.set_matplotlib_defaults(config)
        keys = {k: config.get("keys", k) for k in config.options("keys")}
        ou.check_keybinding_conflicts(keys)
        clients, streams, inventories = ou.fetch_waveforms_with_metadata(options, args, config)

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        window = op.ObsPyck(clients, streams, options, keys, config, inventories)

        n_picks = len(window.catalog[0].picks)
        print(f"picks preloaded: {n_picks}")
        if n_picks != 2 * len(STATIONS):
            print(f"FAIL: expected {2 * len(STATIONS)} preloaded picks, got {n_picks}")
            ok = False

        apply_bugfix_patches(REF_LAT, REF_LON)
        before = set(os.listdir(work))
        window.on_qToolButton_doNlloc_clicked()
        origin = window.catalog[0].origins[-1]
        print(f"NLLoc origin: lon={origin.longitude} lat={origin.latitude} depth_m={origin.depth}")
        if not (170 < origin.longitude < 180 and -40 < origin.latitude < -36):
            print("FAIL: NLLoc origin lon/lat is not near the expected reference point")
            ok = False

        window.catalog[0].picks[0].time = window.catalog[0].picks[0].time + 0.37
        window.setXMLEventID()
        window.save_event_locally()
        after = set(os.listdir(work))
        new_xml = [f for f in (after - before) if f.startswith("obspyck_") and f.endswith(".xml")]
        print(f"new save file(s): {new_xml}")
        if len(new_xml) != 1:
            print("FAIL: expected exactly one new obspyck_*.xml save file")
            ok = False
        else:
            from obspy import read_events

            saved = read_events(str(work / new_xml[0]))
            if len(saved[0].picks) != n_picks:
                print("FAIL: saved QuakeML pick count does not match")
                ok = False
    finally:
        os.chdir(Path(__file__).resolve().parent)
        shutil.rmtree(root, ignore_errors=True)

    return ok


if __name__ == "__main__":
    if run_smoke():
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)
