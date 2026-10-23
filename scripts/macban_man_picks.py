#!/usr/bin/env python
"""MacBan manual catalogue (``man_picks``) -> QuakeML + CSV + per-event miniSEED.

The MacBan 2018-2019 ``man_picks`` archive holds the operator's manually
picked and located events, one set of files per event::

    YYYYMMDD_HHMMSS_NET_evtN.evl    locator printout: origin, 68 % errors,
                                    error ellipse, one line per arrival
                                    (distance, azimuth, take-off angle,
                                    pick, residual, weight, amplitude, ML)
    YYYYMMDD_HHMMSS_NET_evtN.pha    Hypo71 phase cards (the same picks)
    YYYYMMDD_HHMMSS_NET_evtN.suds   the event's PC-SUDS waveform window

plus ``*_NET_regN.suds`` regional recordings with no picks (skipped). Each
year has monthly folders and an ``events YYYY all`` folder with the same
files; files are de-duplicated by name.

The ``.evl`` is parsed (fixed columns); the ``.pha`` is used only to check
that it holds the same picks. Written to ``OUT_DIR``::

    catalog.xml     QuakeML: one Event per .evl with its Origin (manual,
                    reviewed), OriginUncertainty (ellipse, 68 %), ML
                    Magnitude + StationMagnitudes, Picks and Arrivals
    events.csv      one row per event
    picks.csv       one row per pick, with its arrival fields
    waveforms/<event_id>.mseed   the event's SUDS window (STEIM2)
    waveforms/<event_id>.png     with --plot (or --plot-only, which redraws from
                                 the existing .mseed): Z/N/E record sections
                                 with the picks

Codes follow ``openmeq.suds`` (and so the MacBan miniSEED archive):
``MB-02``/``chz`` -> ``MB.MB02..EHZ``. Event ids are
``openmeq.catalog.event_id_from_time(origin time)``. Units: QuakeML metres
for depth/uncertainties, km in the CSVs.

``.evl`` columns not part of QuakeML are kept as ``openmeq`` extras:
``evl_file``, ``quality`` (A-D) and ``iterations`` on the event/origin;
``hypo_weight_code``, ``amplitude``, ``period``, ``station_ml``, ``evl_int``
and ``evl_info`` (the ``INT`` and ``INFO`` columns, meaning undocumented)
on the pick.

Usage::

    python scripts/macban_man_picks.py \\
        /media/steve/fast9100/datasets/MacBan/man_picks \\
        /media/steve/fast9100/datasets/MacBan/man_picks_extracted [--plot | --plot-only]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Catalog, Stream, Trace, UTCDateTime, read
from obspy.core.event import (
    Arrival,
    Comment,
    CreationInfo,
    EventDescription,
    Magnitude,
    Origin,
    OriginQuality,
    OriginUncertainty,
    Pick,
    QuantityError,
    ResourceIdentifier,
    StationMagnitude,
    WaveformStreamID,
)
from obspy.geodetics import kilometers2degrees

from openmeq.catalog import event_id_from_time, iso_z, make_event, set_extra, write_catalog
from openmeq.suds import Codes, channel_code, read_suds, station_code

METHOD_ID = "smi:openmeq/method/macban_manual_evl"
_CODES = Codes()
_FILE_RE = re.compile(r"^\d{8}_\d{6}_NET_evt\w*\.evl$", re.IGNORECASE)
_KM_PER_DEG = 111.19492664455873


@dataclass
class EvlArrival:
    station: str  # SUDS name, e.g. MB-04
    channel: str  # SUDS name, e.g. chz
    dist_km: float
    azimuth: float
    takeoff: float
    int_col: str
    time: UTCDateTime
    phase: str
    onset: str
    polarity: str
    weight_code: int | None
    residual: float | None
    weight: float | None
    amplitude: float | None
    period: float | None
    station_ml: float | None
    info: float | None


@dataclass
class EvlEvent:
    path: Path
    quality: str
    time: UTCDateTime
    lat: float
    lon: float
    depth_km: float
    ml: float | None
    rms: float | None
    n_phases: int
    n_stations: int
    gap: float | None
    err_t: float | None = None
    err_lat_km: float | None = None
    err_lon_km: float | None = None
    err_z_km: float | None = None
    err_ml: float | None = None
    iterations: int | None = None
    processed: UTCDateTime | None = None
    ell_major_km: float | None = None
    ell_minor_km: float | None = None
    ell_azimuth: float | None = None
    arrivals: list[EvlArrival] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# .evl / .pha parsing
# --------------------------------------------------------------------------- #


def _num(s: str) -> float | None:
    s = s.strip().rstrip("skm")
    return float(s) if s else None


def _hms(date: UTCDateTime, s: str) -> UTCDateTime:
    """``'18:23:  .251'`` / ``'20: 8:13.151'`` on ``date`` -> time (blank = 0)."""
    h, m, sec = (p.strip() or "0" for p in s.split(":"))
    return date + int(h) * 3600 + int(m) * 60 + float(sec)


def parse_evl(path: Path) -> EvlEvent:
    lines = path.read_text(encoding="ascii", errors="replace").replace("\r", "").split("\n")
    q = re.search(r"Q=([A-Z])", lines[0])
    o = lines[2]
    # ' 2018/08/02 19:11:49.644 14.10144N 121.21830E   1.63     .3  .045s  12/ 4  151'
    day = UTCDateTime.strptime(o[1:11], "%Y/%m/%d")
    lat = float(o[24:33]) * (1 if o[33] == "N" else -1)
    lon = float(o[34:44]) * (1 if o[44] == "E" else -1)
    nl, nm = o[66:74].split("/")
    ev = EvlEvent(
        path=path,
        quality=q.group(1) if q else "",
        time=_hms(day, o[12:24]),
        lat=lat,
        lon=lon,
        depth_km=float(o[45:52]),
        ml=_num(o[52:59]),
        rms=_num(o[59:66]),
        n_phases=int(nl),
        n_stations=int(nm),
        gap=_num(o[74:]),
    )
    for line in lines[3:5]:
        if line.startswith(" ERRORS:"):
            f = line[14:].split()
            ev.err_t, ev.err_lat_km, ev.err_lon_km, ev.err_z_km, ev.err_ml = map(_num, f[:5])
            it = re.search(r"IT\s*(\d+)", line)
            ev.iterations = int(it.group(1)) if it else None
            ts = re.search(r"\[(.+)\]", line)
            ev.processed = UTCDateTime.strptime(ts.group(1), "%Y/%m/%d %H:%M:%S") if ts else None
        elif line.startswith(" ERROR ELLIPSE:"):
            v = re.findall(r"=\s*([\d.]+)", line)
            ev.ell_major_km, ev.ell_minor_km, ev.ell_azimuth = map(float, v[:3])
    for line in lines[6:]:
        if not re.match(r"^ +\d+ \S", line):
            continue
        # '   2 MB-04 chz    2 277 134  0 1  19:11:50.348  +IP1   -.032  1.00  .100E+05'
        # '    5  -.6  1.00' (one line; split here for width)
        t = _hms(day, line[33:46])
        if t < ev.time - 3600:  # pick after midnight
            t += 86400
        wave = line[48:52]
        ev.arrivals.append(
            EvlArrival(
                station=line[5:10].strip(),
                channel=line[11:14].strip(),
                dist_km=float(line[14:19]),
                azimuth=float(line[19:23]),
                takeoff=float(line[23:27]),
                int_col=line[27:33].strip(),
                time=t,
                polarity=wave[0].strip(),
                onset=wave[1],
                phase=wave[2],
                weight_code=int(wave[3]) if wave[3].isdigit() else None,
                residual=_num(line[52:61]),
                weight=_num(line[61:67]),
                amplitude=_num(line[67:77]),
                period=_num(line[77:82]),
                station_ml=_num(line[82:87]),
                info=_num(line[87:]),
            )
        )
    if len(ev.arrivals) != ev.n_phases:
        raise ValueError(f"{path.name}: {len(ev.arrivals)} arrival lines, header NL={ev.n_phases}")
    return ev


def pha_times(path: Path) -> list[tuple[str, str, float]]:
    """(station, channel, epoch s) of every pick in a Hypo71 ``.pha`` file."""
    out = []
    for line in path.read_text(encoding="ascii", errors="replace").replace("\r", "").split("\n"):
        if not line.startswith("MB-"):
            continue
        sta, cha = line[0:5], line[5:8]
        base = UTCDateTime.strptime(line[13:21] + line[22:24].replace(" ", "0"), "%Y%m%d%H")
        # P: minute/seconds in cols 24-31; S (P columns blank): in cols 39-46
        mmss = line[24:32] if line[9] == "P" else line[39:47]
        t = base + int(mmss[:2]) * 60 + float(mmss[2:])
        out.append((sta, cha, round(t.timestamp, 3)))
    return out


# --------------------------------------------------------------------------- #
# QuakeML
# --------------------------------------------------------------------------- #


def _wid(sta: str, cha: str) -> WaveformStreamID:
    return WaveformStreamID(
        network_code=_CODES.network,
        station_code=station_code(sta),
        location_code=_CODES.location,
        channel_code=channel_code(cha, _CODES),
    )


def build_event(ev: EvlEvent):
    eid = event_id_from_time(ev.time)
    event = make_event(eid)
    event.event_type = "earthquake"
    event.event_type_certainty = "known"
    event.event_descriptions.append(EventDescription(text=ev.path.stem, type="earthquake name"))
    set_extra(event, "evl_file", ev.path.name)
    set_extra(event, "quality", ev.quality)
    ci = CreationInfo(agency_id="MacBan", creation_time=ev.processed)

    origin = Origin(
        resource_id=ResourceIdentifier(f"smi:openmeq/origin/{eid}/evl"),
        time=ev.time,
        latitude=ev.lat,
        longitude=ev.lon,
        depth=ev.depth_km * 1000.0,
        method_id=ResourceIdentifier(METHOD_ID),
        evaluation_mode="manual",
        evaluation_status="reviewed",
        creation_info=ci,
    )
    origin.time_errors = QuantityError(uncertainty=ev.err_t, confidence_level=68)
    if ev.err_lat_km is not None:
        origin.latitude_errors = QuantityError(
            uncertainty=ev.err_lat_km / _KM_PER_DEG, confidence_level=68
        )
    if ev.err_lon_km is not None:
        origin.longitude_errors = QuantityError(
            uncertainty=ev.err_lon_km / (_KM_PER_DEG * math.cos(math.radians(ev.lat))),
            confidence_level=68,
        )
    if ev.err_z_km is not None:
        origin.depth_errors = QuantityError(uncertainty=ev.err_z_km * 1000.0, confidence_level=68)
    ou = OriginUncertainty(confidence_level=68)
    if ev.err_lat_km is not None and ev.err_lon_km is not None:
        ou.horizontal_uncertainty = math.hypot(ev.err_lat_km, ev.err_lon_km) * 1000.0
    if ev.ell_major_km is not None:
        ou.max_horizontal_uncertainty = ev.ell_major_km * 1000.0
        ou.min_horizontal_uncertainty = ev.ell_minor_km * 1000.0
        ou.azimuth_max_horizontal_uncertainty = ev.ell_azimuth
        ou.preferred_description = "uncertainty ellipse"
    elif ou.horizontal_uncertainty is not None:
        ou.preferred_description = "horizontal uncertainty"
    origin.origin_uncertainty = ou
    origin.quality = OriginQuality(
        used_phase_count=ev.n_phases,
        used_station_count=ev.n_stations,
        standard_error=ev.rms,
        azimuthal_gap=ev.gap,
        minimum_distance=kilometers2degrees(min(a.dist_km for a in ev.arrivals)),
        maximum_distance=kilometers2degrees(max(a.dist_km for a in ev.arrivals)),
    )
    set_extra(origin, "quality", ev.quality)
    if ev.iterations is not None:
        set_extra(origin, "iterations", ev.iterations)

    for a in ev.arrivals:
        pick = Pick(
            resource_id=ResourceIdentifier(
                f"smi:openmeq/pick/{eid}/{station_code(a.station)}.{a.channel}.{a.phase}"
            ),
            time=a.time,
            waveform_id=_wid(a.station, a.channel),
            phase_hint=a.phase,
            onset={"I": "impulsive", "E": "emergent"}.get(a.onset),
            polarity={"+": "positive", "-": "negative"}.get(a.polarity),
            evaluation_mode="manual",
            evaluation_status="reviewed",
            method_id=ResourceIdentifier(METHOD_ID),
            creation_info=ci,
        )
        for key, val in (
            ("hypo_weight_code", a.weight_code),
            ("amplitude", a.amplitude),
            ("period", a.period),
            ("station_ml", a.station_ml),
            ("evl_int", a.int_col),
            ("evl_info", a.info),
        ):
            if val is not None and val != "":
                set_extra(pick, key, val)
        event.picks.append(pick)
        origin.arrivals.append(
            Arrival(
                pick_id=pick.resource_id,
                phase=a.phase,
                distance=kilometers2degrees(a.dist_km),
                azimuth=a.azimuth,
                takeoff_angle=a.takeoff,
                time_residual=a.residual,
                time_weight=a.weight,
            )
        )
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id

    if ev.ml is not None:
        mag = Magnitude(
            resource_id=ResourceIdentifier(f"smi:openmeq/magnitude/{eid}/evl"),
            mag=ev.ml,
            magnitude_type="ML",
            origin_id=origin.resource_id,
            method_id=ResourceIdentifier(METHOD_ID),
            evaluation_mode="manual",
            creation_info=ci,
        )
        if ev.err_ml is not None:
            mag.mag_errors = QuantityError(uncertainty=ev.err_ml)
        for a in ev.arrivals:
            if a.station_ml is None:
                continue
            sm = StationMagnitude(
                resource_id=ResourceIdentifier(
                    f"smi:openmeq/stationmagnitude/{eid}/{station_code(a.station)}.{a.channel}"
                ),
                origin_id=origin.resource_id,
                mag=a.station_ml,
                station_magnitude_type="ML",
                waveform_id=_wid(a.station, a.channel),
            )
            event.station_magnitudes.append(sm)
        mag.station_count = len({sm.waveform_id.station_code for sm in event.station_magnitudes})
        event.magnitudes.append(mag)
        event.preferred_magnitude_id = mag.resource_id
    event.comments.append(Comment(text=f"source:{ev.path.name}"))
    return eid, event


# --------------------------------------------------------------------------- #
# Waveforms
# --------------------------------------------------------------------------- #


def suds_stream(path: Path) -> tuple[Stream, list[str]]:
    traces, _, warnings = read_suds(path)
    st = Stream()
    for t in traces:
        data = t["data"].astype(np.int32) if t["data"].dtype.kind == "i" else t["data"]
        st.append(
            Trace(
                data=np.ascontiguousarray(data),
                header=dict(
                    network=_CODES.network,
                    station=station_code(t["sta"]),
                    location=_CODES.location,
                    channel=channel_code(t["cha"], _CODES),
                    starttime=UTCDateTime(t["begin"]),
                    sampling_rate=t["rate"],
                ),
            )
        )
    st.merge(method=-1)
    st.sort()
    return st, warnings


def plot_event(st: Stream, ev: EvlEvent, eid: str, out: Path) -> None:
    """Record section, one panel per component (Z, N, E), stations sorted by distance.

    Each panel marks the picks made on that channel solid (P blue, S red) and
    the station's picks on its other channels dotted, for reference.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = ev.time
    tmax = max(a.time for a in ev.arrivals) - t0 + 3.0
    st = st.copy()
    st.detrend("demean")
    st.filter("bandpass", freqmin=2.0, freqmax=30.0, corners=2)
    st.trim(t0 - 2.0, t0 + tmax)
    dist: dict[str, float] = {}
    for a in ev.arrivals:
        dist.setdefault(station_code(a.station), a.dist_km)
    stations = sorted({tr.stats.station for tr in st} & dist.keys(), key=dist.get)
    if not stations:
        return
    fig, axes = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(15, 1.2 + 0.6 * len(stations))
    )
    for ax, comp in zip(axes, "ZNE", strict=True):
        for i, sta in enumerate(stations):
            for tr in st.select(station=sta, component=comp):
                d = tr.data / (np.abs(tr.data).max() or 1.0)
                ax.plot(tr.times(reftime=t0), d * 0.45 + i, "k", lw=0.5)
            for a in ev.arrivals:
                if station_code(a.station) != sta:
                    continue
                own = a.channel[-1].upper() == comp
                ax.vlines(
                    a.time - t0,
                    i - 0.45,
                    i + 0.45,
                    color="tab:blue" if a.phase == "P" else "tab:red",
                    lw=1.2 if own else 0.8,
                    ls="-" if own else ":",
                )
        ax.set_title(f"EH{comp}")
        ax.set_xlabel("s after origin")
    axes[0].set_yticks(range(len(stations)), [f"{s} ({dist[s]:.0f} km)" for s in stations])
    axes[0].set_xlim(-2.0, tmax)
    axes[0].set_ylim(-0.7, len(stations) - 0.3)
    fig.suptitle(
        f"{eid}  {iso_z(t0)}  ML {ev.ml}  depth {ev.depth_km} km  (2-30 Hz; P blue, S red; "
        "solid = picked on this channel, dotted = picked on another)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def find_evl(in_dir: Path) -> list[Path]:
    """Every ``*_NET_evt*.evl`` under ``in_dir``, de-duplicated by file name."""
    seen: dict[str, Path] = {}
    for p in sorted(in_dir.rglob("*.evl")):
        if _FILE_RE.match(p.name):
            seen.setdefault(p.name, p)
    return [seen[k] for k in sorted(seen)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_dir", type=Path, help="man_picks root (searched recursively)")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--plot", action="store_true", help="write a record-section PNG per event")
    ap.add_argument("--no-waveforms", action="store_true", help="catalogue and CSVs only")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="only (re)draw the PNGs, from the waveforms/*.mseed of an earlier run",
    )
    args = ap.parse_args(argv)

    evls = find_evl(args.in_dir)
    if not evls:
        ap.error(f"no *_NET_evt*.evl files under {args.in_dir}")
    wf_dir = args.out_dir / "waveforms"
    wf_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        n = 0
        for path in evls:
            ev = parse_evl(path)
            eid = event_id_from_time(ev.time)
            mseed = wf_dir / f"{eid}.mseed"
            if mseed.exists():
                plot_event(read(str(mseed)), ev, eid, wf_dir / f"{eid}.png")
                n += 1
        print(f"{n} plots -> {wf_dir}")
        return 0

    cat = Catalog(description="MacBan 2018-2019 manual locations (man_picks .evl)")
    ev_rows, pk_rows, problems = [], [], []
    ids: dict[str, str] = {}
    for path in evls:
        ev = parse_evl(path)
        eid, event = build_event(ev)
        if eid in ids:
            problems.append(f"{path.name}: event id {eid} also from {ids[eid]}")
        ids[eid] = path.name

        pha = path.with_suffix(".pha")
        if pha.exists():
            want = sorted((a.station, a.channel, round(a.time.timestamp, 3)) for a in ev.arrivals)
            if sorted(pha_times(pha)) != want:
                problems.append(f"{path.name}: .pha picks differ from .evl")
        else:
            problems.append(f"{path.name}: no .pha")

        suds = path.with_suffix(".suds")
        n_tr, stations, covered, wf = 0, [], 0, ""
        if suds.exists() and not args.no_waveforms:
            st, warns = suds_stream(suds)
            problems += [f"{suds.name}: {w}" for w in warns]
            wf = f"waveforms/{eid}.mseed"
            st.write(str(args.out_dir / wf), format="MSEED", encoding="STEIM2")
            n_tr = len(st)
            stations = sorted({tr.stats.station for tr in st})
            for a in ev.arrivals:
                sel = st.select(
                    station=station_code(a.station), channel=channel_code(a.channel, _CODES)
                )
                covered += any(tr.stats.starttime <= a.time <= tr.stats.endtime for tr in sel)
            if covered < len(ev.arrivals):
                problems.append(
                    f"{path.name}: {len(ev.arrivals) - covered} picks outside the SUDS window"
                )
            if args.plot:
                plot_event(st, ev, eid, wf_dir / f"{eid}.png")
        elif not suds.exists():
            problems.append(f"{path.name}: no .suds")
        cat.events.append(event)

        ev_rows.append(
            dict(
                event_id=eid,
                origin_time=iso_z(ev.time),
                latitude=ev.lat,
                longitude=ev.lon,
                depth_km=ev.depth_km,
                ml=ev.ml,
                ml_err=ev.err_ml,
                quality=ev.quality,
                rms_s=ev.rms,
                n_phases=ev.n_phases,
                n_stations=ev.n_stations,
                gap_deg=ev.gap,
                err_time_s=ev.err_t,
                err_lat_km=ev.err_lat_km,
                err_lon_km=ev.err_lon_km,
                err_depth_km=ev.err_z_km,
                ellipse_major_km=ev.ell_major_km,
                ellipse_minor_km=ev.ell_minor_km,
                ellipse_azimuth_deg=ev.ell_azimuth,
                iterations=ev.iterations,
                processed=iso_z(ev.processed) if ev.processed else "",
                n_picks_p=sum(a.phase == "P" for a in ev.arrivals),
                n_picks_s=sum(a.phase == "S" for a in ev.arrivals),
                waveform_file=wf,
                n_traces=n_tr,
                waveform_stations=" ".join(stations),
                picks_in_waveform=covered if wf else None,
                source=str(path.relative_to(args.in_dir)),
            )
        )
        for a in ev.arrivals:
            pk_rows.append(
                dict(
                    event_id=eid,
                    seed_id=".".join(
                        (
                            _CODES.network,
                            station_code(a.station),
                            _CODES.location,
                            channel_code(a.channel, _CODES),
                        )
                    ),
                    station=station_code(a.station),
                    channel=channel_code(a.channel, _CODES),
                    phase=a.phase,
                    time=iso_z(a.time),
                    tt_s=round(a.time - ev.time, 3),
                    onset=a.onset,
                    polarity=a.polarity,
                    weight_code=a.weight_code,
                    weight=a.weight,
                    residual_s=a.residual,
                    distance_km=a.dist_km,
                    azimuth_deg=a.azimuth,
                    takeoff_deg=a.takeoff,
                    amplitude=a.amplitude,
                    period=a.period,
                    station_ml=a.station_ml,
                    evl_int=a.int_col,
                    evl_info=a.info,
                )
            )

    write_catalog(cat, args.out_dir / "catalog.xml")
    pd.DataFrame(ev_rows).to_csv(args.out_dir / "events.csv", index=False)
    pd.DataFrame(pk_rows).to_csv(args.out_dir / "picks.csv", index=False)
    (args.out_dir / "problems.txt").write_text("".join(f"{p}\n" for p in problems))
    print(
        f"{len(cat)} events, {len(pk_rows)} picks -> {args.out_dir} "
        f"({len(problems)} problems, see problems.txt)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
