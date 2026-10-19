"""PC-SUDS event files -> miniSEED day-file archive (``openmeq.suds``, OpenMEQ-ekd).

Used two ways: by the ``qc`` step when ``data.convert`` is configured
(:func:`openmeq.ingest.convert_raw_data`, so ``openmeq run`` converts raw
data before indexing it), and standalone via ``scripts/suds2mseed.py``.

Input
-----
Every ``*.suds`` file found recursively under ``IN_DIR``. File names must
start with the event time, ``YYYYMMDD_HHMMSS`` (e.g.
``20180901_034300_NET.suds``); it is used to group files by day.

ObsPy has no SUDS reader, so this parses the struct stream directly. Each
SUDS struct is a 12-byte tag (``'S'``, machine char ``'6'`` = PC little
endian, ``short`` struct id, ``long`` struct length, ``long`` trailing data
length) followed by the struct and then its data. Only two structs matter:

* ``STATIONCOMP`` (id 5) -- station coordinates (``st_lat``/``st_long``
  doubles, ``elev`` float in m).
* ``DESCRIPTRACE`` (id 7) -- one trace: ``begintime`` (double, epoch s,
  UTC), ``datatype`` (``'s'``/``'l'``/``'i'``/``'f'``/``'d'``), ``length``,
  ``rate``; the samples follow as the struct's data.

The MacBan files use an extended variant in which both structs carry 32
extra bytes after the standard layout: ``network[8]``, ``station[16]``,
``channel[8]`` (e.g. ``NET`` / ``MB-02`` / ``chz``). Those long names are
used for the output codes because the classic 4-char ``st_name`` is
inconsistent across that archive (``002z``, ``MB1v``, ``CH0`` ...); plain
PC-SUDS structs fall back to ``st_name``/``component``. Other structs
(``FEATURE`` picks, ``CHANSET``) are skipped.

A struct tag that fails the sync check (a few files contain a block of
garbage in place of one trace) is handled by scanning forward to the next
valid tag; the skipped byte range is reported.

Output
------
* One STEIM2 miniSEED day file per channel, in the RK-NM archive layout::

      OUT_DIR/YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed

  The triggered event windows are merged into each day file with
  ``Stream.merge(method=-1)``: contiguous windows and overlaps with
  identical samples join; gaps (and the rare overlap whose samples
  disagree) stay as separate segments.
* ``OUT_DIR/stations.xml`` -- one epoch per station at the median of its
  per-file GPS fixes (fixes that are invalid, (0, 0), or >1 km from the
  median are dropped), spanning the station's data. No instrument response:
  SUDS gain fields are unset in the MacBan files.

Codes: SUDS station ``MB-02`` -> ``MB02`` (non-alphanumerics dropped, max 5
chars); channel ``chz``/``chn``/``che`` (or component ``v``) ->
``<band><inst>Z/N/E``, i.e. ``EHZ`` by default.

Existing day files are never overwritten unless ``overwrite=True``.
"""

from __future__ import annotations

import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Station

from openmeq.log import get_logger
from openmeq.parallel import ParallelConfigLike, pmap

logger = get_logger("suds")

_TAG = struct.Struct("<ccHii")
_STATIONCOMP, _DESCRIPTRACE = 5, 7
_STATIONCOMP_LEN, _DESCRIPTRACE_LEN = 76, 64  # standard PC-SUDS struct sizes
_DAY_MARGIN = 3600.0
_DTYPES = {b"s": "<i2", b"l": "<i4", b"i": "<i4", b"f": "<f4", b"d": "<f8"}
_RESYNC = re.compile(rb"S6[\x01-\x30]\x00")
_FNAME_TIME = re.compile(r"^(\d{8}_\d{6})")
_MAX_COORD_DEV_M = 1000.0


@dataclass(frozen=True)
class Codes:
    network: str = "MB"
    location: str = ""
    band: str = "E"
    inst: str = "H"


# --------------------------------------------------------------------------- #
# SUDS parsing
# --------------------------------------------------------------------------- #


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("ascii", "replace").strip()


def _names(s: bytes, std_len: int) -> tuple[str, str, str]:
    """(network, station, channel) from the 32-byte extension after ``std_len``."""
    ext = s[std_len:]
    if len(ext) < 32:
        # Plain PC-SUDS: fall back to the STATIDENT at the head of the struct.
        return _cstr(s[0:4]), _cstr(s[4:9]), s[9:10].decode("ascii", "replace")
    return _cstr(ext[0:8]), _cstr(ext[8:24]), _cstr(ext[24:32])


def read_suds(path: Path) -> tuple[list[dict], dict[str, tuple], list[str]]:
    """Parse one SUDS file -> (traces, {station: (lat, lon, elev)}, warnings).

    Each trace is a dict with ``net``, ``sta``, ``cha`` (SUDS names),
    ``begin`` (epoch s), ``rate`` (Hz) and ``data`` (numpy array).
    """
    b = path.read_bytes()
    traces, coords, warnings = [], {}, []
    p = 0
    while p + _TAG.size <= len(b):
        sync, mach, sid, slen, dlen = _TAG.unpack_from(b, p)
        if (
            sync != b"S"
            or mach != b"6"
            or slen < 0
            or dlen < 0
            or p + _TAG.size + slen + dlen > len(b)
        ):
            m = _RESYNC.search(b, p + 1)
            nxt = m.start() if m else len(b)
            warnings.append(f"bad struct tag at byte {p}; skipped {nxt - p} bytes")
            p = nxt
            continue
        s = b[p + _TAG.size : p + _TAG.size + slen]
        data_off = p + _TAG.size + slen
        if sid == _STATIONCOMP:
            _, sta, _ = _names(s, _STATIONCOMP_LEN)
            lat, lon, elev = struct.unpack_from("<ddf", s, 16)
            coords[sta] = (round(lat, 6), round(lon, 6), round(float(elev), 1))
        elif sid == _DESCRIPTRACE:
            net, sta, cha = _names(s, _DESCRIPTRACE_LEN)
            begin, _, dtype, _, _, _, n, rate = struct.unpack_from("<dhcchhif", s, 12)
            if dtype not in _DTYPES:
                warnings.append(f"{sta}.{cha}: unsupported datatype {dtype!r}, trace skipped")
            elif rate <= 0:
                warnings.append(f"{sta}.{cha}: sample rate {rate}, trace skipped")
            else:
                dt = np.dtype(_DTYPES[dtype])
                n = min(n, dlen // dt.itemsize)
                data = np.frombuffer(b, dtype=dt, count=n, offset=data_off)
                traces.append(
                    dict(net=net, sta=sta, cha=cha, begin=begin, rate=float(rate), data=data)
                )
        p = data_off + dlen
    return traces, coords, warnings


def filename_time(src: Path) -> UTCDateTime:
    """Event time from the SUDS filename (``20180901_034300_NET.suds``)."""
    m = _FNAME_TIME.match(src.name)
    if not m:
        raise ValueError(f"{src.name}: name does not start with YYYYMMDD_HHMMSS")
    return UTCDateTime.strptime(m.group(1), "%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def station_code(sta: str) -> str:
    """SUDS station name -> SEED station code: ``MB-02`` -> ``MB02``."""
    return re.sub(r"[^A-Za-z0-9]", "", sta).upper()[:5]


def channel_code(cha: str, codes: Codes) -> str:
    """SUDS channel name -> SEED channel code: ``chz`` -> ``EHZ``."""
    comp = cha[-1:].upper()
    return f"{codes.band}{codes.inst}{ {'V': 'Z'}.get(comp, comp) }"


def day_path(out_dir: Path, seed_id: str, day: UTCDateTime) -> Path:
    _, sta, _, cha = seed_id.split(".")
    return (
        out_dir
        / f"{day.year}"
        / sta
        / f"{cha}.D"
        / f"{seed_id}.D.{day.year}.{day.julday:03d}.mseed"
    )


def build_day_stream(day: UTCDateTime, srcs: list[Path], codes: Codes) -> tuple[Stream, dict]:
    """Merged stream for ``day`` from every SUDS file in ``srcs`` + per-day metadata.

    ``srcs`` should include files named up to ``_DAY_MARGIN`` either side of
    the day (trace begintimes can precede the filename time by minutes, and
    windows last a few minutes); traces are trimmed to the day.
    """
    t0, t1 = day, day + 86400
    st = Stream()
    fixes: dict[str, list] = defaultdict(list)
    warnings = []
    n_files = 0
    for src in srcs:
        traces, coords, warns = read_suds(src)
        if t0 <= filename_time(src) < t1:  # report each file once, on its own day
            n_files += 1
            warnings += [f"{src}: {w}" for w in warns]
            for k, v in coords.items():
                fixes[station_code(k)].append(v)
        for t in traces:
            start = UTCDateTime(t["begin"])
            if start >= t1 or start + len(t["data"]) / t["rate"] <= t0:
                continue
            data = t["data"]
            if data.dtype.kind == "i":
                data = data.astype(np.int32)
            st.append(
                Trace(
                    data=np.ascontiguousarray(data),
                    header=dict(
                        network=codes.network,
                        station=station_code(t["sta"]),
                        location=codes.location,
                        channel=channel_code(t["cha"], codes),
                        starttime=start,
                        sampling_rate=t["rate"],
                    ),
                )
            )
    st.trim(t0, t1 - 1e-6, nearest_sample=False)
    st.traces = [tr for tr in st if tr.stats.npts]
    st.merge(method=-1)
    st.sort()
    meta = dict(
        day=str(day.date),
        n_files=n_files,
        coords=dict(fixes),
        channels={tr.stats.station: set() for tr in st},
        rates={tr.stats.station: tr.stats.sampling_rate for tr in st},
        times={},
        warnings=warnings,
    )
    for tr in st:
        sta = tr.stats.station
        meta["channels"][sta].add(tr.stats.channel)
        a, b = meta["times"].get(sta, (np.inf, -np.inf))
        meta["times"][sta] = (
            min(a, tr.stats.starttime.timestamp),
            max(b, tr.stats.endtime.timestamp),
        )
    return st, meta


def convert_day(item: tuple[UTCDateTime, list[Path], Path, Codes, bool, bool]) -> dict:
    """Worker: write one day's channel files (unless they exist / dry run)."""
    day, srcs, out_dir, codes, overwrite, dry_run = item
    st, meta = build_day_stream(day, srcs, codes)
    meta.update(written=0, skipped=0, segments=0)
    for seed_id in sorted({tr.id for tr in st}):
        part = st.select(id=seed_id)
        dst = day_path(out_dir, seed_id, day)
        if dst.exists() and not overwrite:
            meta["skipped"] += 1
            continue
        meta["written"] += 1
        meta["segments"] += len(part)
        if dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        enc = "STEIM2" if all(tr.data.dtype == np.int32 for tr in part) else "FLOAT64"
        tmp = dst.with_suffix(".mseed.tmp")
        part.write(str(tmp), format="MSEED", encoding=enc, reclen=4096)
        tmp.replace(dst)
    return meta


def build_inventory(results: list[dict], network: str, location: str) -> Inventory:
    """One epoch per station at the median of its per-file GPS fixes.

    The SUDS coordinates are re-recorded in every event file and jitter by
    ~10 m; some are (0, 0, 0) or garbage. Fixes outside the valid range or
    more than ``_MAX_COORD_DEV_M`` from the station median are discarded.
    """
    fixes: dict[str, list] = defaultdict(list)
    span: dict[str, list[float]] = {}
    chans: dict[str, set] = defaultdict(set)
    rates: dict[str, float] = {}
    for r in results:
        for sta, (t0, t1) in r["times"].items():
            e = span.setdefault(sta, [t0, t1])
            e[0], e[1] = min(e[0], t0), max(e[1], t1)
            chans[sta].update(r["channels"][sta])
            rates[sta] = r["rates"][sta]
        for sta, sta_fixes in r["coords"].items():
            for lat, lon, elev in sta_fixes:
                if abs(lat) <= 90 and abs(lon) <= 180 and (lat, lon) != (0, 0):
                    fixes[sta].append((lat, lon, elev))
    stations = []
    for sta in sorted(span):
        if not fixes[sta]:
            logger.warning("%s: no valid coordinates, left out of stations.xml", sta)
            continue
        a = np.array(fixes[sta])
        med = np.median(a, axis=0)
        dev = np.hypot(a[:, 0] - med[0], (a[:, 1] - med[1]) * np.cos(np.radians(med[0]))) * 111.2e3
        lat, lon, elev = (
            round(float(x), d)
            for x, d in zip(np.median(a[dev <= _MAX_COORD_DEV_M], axis=0), (6, 6, 1), strict=True)
        )
        t0, t1 = UTCDateTime(span[sta][0]), UTCDateTime(span[sta][1])
        cha_objs = [
            Channel(
                code=c,
                location_code=location,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                depth=0.0,
                sample_rate=rates[sta],
                start_date=t0,
                end_date=t1,
                azimuth={"Z": 0.0, "N": 0.0, "E": 90.0}.get(c[-1]),
                dip={"Z": -90.0}.get(c[-1], 0.0),
            )
            for c in sorted(chans[sta])
        ]
        stations.append(
            Station(
                code=sta,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                start_date=t0,
                end_date=t1,
                channels=cha_objs,
            )
        )
    return Inventory(networks=[Network(code=network, stations=stations)], source="openmeq.suds")


def plan_days(srcs: list[Path]) -> list[tuple[UTCDateTime, list[Path]]]:
    """(day, SUDS files that may hold data for it) for every day the files can reach."""
    timed = sorted((filename_time(s), s) for s in srcs)
    # A window can spill into the neighbouring day even when that day has no
    # event file of its own, so plan every day within the margin of a file.
    dates = {(t + dt).date for t, _ in timed for dt in (-_DAY_MARGIN, 0, _DAY_MARGIN)}
    days = [UTCDateTime(d) for d in sorted(dates)]
    return [
        (day, [s for t, s in timed if day - _DAY_MARGIN <= t < day + 86400 + _DAY_MARGIN])
        for day in days
    ]


def convert(
    in_dir: Path,
    out_dir: Path,
    pcfg: ParallelConfigLike,
    codes: Codes | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    stationxml: bool = True,
    pattern: str = "*.suds",
) -> list[dict]:
    """Convert every SUDS file under ``in_dir``; returns the per-day results."""
    codes = codes or Codes()
    srcs = sorted(in_dir.rglob(pattern))
    if not srcs:
        raise FileNotFoundError(f"no {pattern} files under {in_dir}")
    plan = plan_days(srcs)
    logger.info("%d SUDS files over %d days -> %s", len(srcs), len(plan), out_dir)
    items = [(day, day_srcs, out_dir, codes, overwrite, dry_run) for day, day_srcs in plan]
    results = pmap(convert_day, items, pcfg, desc="suds -> miniSEED")
    for r in results:
        for w in r["warnings"]:
            logger.warning("%s", w)
    skipped = sum(r["skipped"] for r in results)
    if skipped:
        logger.warning("%d day files already existed and were left as-is (no overwrite)", skipped)
    verb = "would write" if dry_run else "wrote"
    logger.info(
        "%s %d day files (%d segments) from %d SUDS files",
        verb,
        sum(r["written"] for r in results),
        sum(r["segments"] for r in results),
        sum(r["n_files"] for r in results),
    )
    if stationxml:
        inv = build_inventory(results, codes.network, codes.location)
        xml = out_dir / "stations.xml"
        if dry_run:
            logger.info("would write %s (%d stations)", xml, len(inv.networks[0].stations))
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            inv.write(str(xml), format="STATIONXML")
            logger.info("wrote %s (%d stations)", xml, len(inv.networks[0].stations))
    return results
