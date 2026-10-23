"""CD-1.1 recorder files -> miniSEED day-file archive (``openmeq.cd11``).

Used two ways, like :mod:`openmeq.suds`: by ``openmeq convert`` and the
``qc`` step when ``data.convert.format: cd11``
(:func:`openmeq.ingest.convert_raw_data`), and standalone via
``scripts/cd11tomseed.py``.

Input
-----
Every ``*.cd11`` file found recursively under ``IN_DIR``, e.g. the MacBan
Feb 2021 recorders::

    IN_DIR/MB-01/cont/20210212/20210212_000000_C_MB-01.cd11   (continuous, 15 min)
    IN_DIR/MB-01/event/20210212/20210212_000300_E_MB-01.cd11  (triggered windows)

The file name must end in ``_<station>.cd11``; it is used to group files by
station (one worker per station).

Each file is one CD-1.1 data frame written **little-endian** (the standard
is big-endian): the 36-byte frame header (type 5, creator ``MB-01``,
destination ``FILE_CON``/``FILE_EVT``), the data-frame header (channel
count, frame length, nominal time ``YYYYDDD HH:MM:SS.sss``, channel string
padded to 4 bytes), then a run of channel subframes -- one per channel per
minute to the end of the file, not just ``channel count`` of them. Each
subframe carries its own site / channel / location (``MB-01`` ``chz``
``01``), data type (``s4`` = int32), start time, sample count, a status
block and the samples, so subframes are decoded independently.

Output
------
One STEIM2 miniSEED day file per channel, in the RK-NM archive layout::

    OUT_DIR/YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed

Continuous and event subframes are merged with ``Stream.merge(method=-1)``:
identical overlaps collapse, event windows fill gaps in the continuous
recording, and overlaps whose samples disagree stay as separate segments
(reported). No StationXML: the files carry no coordinates, and their calib
factor is 0, so samples are written as raw counts.

Codes: ``MB-7A`` -> ``MB7A``, ``chz`` -> ``<band><inst>Z`` (``EHZ`` by
default). The source location code (``01``) is replaced by
``Codes.location``. Existing day files are never overwritten unless
``overwrite=True``.
"""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from openmeq.log import get_logger
from openmeq.parallel import ParallelConfigLike, pmap
from openmeq.suds import Codes, channel_code, day_path, station_code

logger = get_logger("cd11")

_FRAME_HDR = 36  # type, trailer offset, creator[8], destination[8], sequence(8), series
_DTYPES = {b"s4": "<i4", b"s2": "<i2", b"t4": "<f4", b"t8": "<f8"}


def _time(b: bytes) -> UTCDateTime:
    """CD-1.1 time string ``2021043 00:00:00.000`` -> UTCDateTime."""
    s = b.decode("ascii")
    return UTCDateTime(year=int(s[:4]), julday=int(s[4:7])) + (
        int(s[8:10]) * 3600 + int(s[11:13]) * 60 + float(s[14:20])
    )


def read_cd11(path: Path) -> list[dict]:
    """Every channel subframe in one little-endian CD-1.1 file.

    Returns dicts with ``sta``, ``cha``, ``start`` (UTCDateTime), ``rate``
    (Hz) and ``data`` (numpy array, native CD-1.1 sample type).
    """
    b = path.read_bytes()
    (ftype,) = struct.unpack_from("<i", b, 0)
    if ftype != 5:
        raise ValueError(f"{path}: frame type {ftype}, expected 5 (data frame)")
    q = _FRAME_HDR + 28  # channel count, frame time length, nominal time[20]
    (slen,) = struct.unpack_from("<i", b, q)
    q += 4 + (slen + 3) // 4 * 4
    out = []
    while q + 8 <= len(b):
        (clen,) = struct.unpack_from("<i", b, q)
        if clen <= 0 or q + 4 + clen > len(b):
            raise ValueError(f"{path}: bad subframe length {clen} at byte {q}")
        # flags[4], site[5], chan[3], loc[2], data type[2], calib factor, calib period
        desc = b[q + 8 : q + 32]
        start = _time(b[q + 32 : q + 52])
        tlen, nsamp, stsz = struct.unpack_from("<iii", b, q + 52)  # ms, samples, status size
        r = q + 64 + (stsz + 3) // 4 * 4
        (dsz,) = struct.unpack_from("<i", b, r)
        dtype = _DTYPES.get(desc[14:16])
        if dtype is None:
            raise ValueError(f"{path}: unsupported data type {desc[14:16]!r}")
        data = np.frombuffer(b, dtype=dtype, count=nsamp, offset=r + 4)
        if dsz != data.nbytes:
            raise ValueError(f"{path}: data size {dsz} != {nsamp} samples of {dtype}")
        out.append(
            dict(
                sta=desc[4:9].split(b"\0")[0].decode("ascii").strip(),
                cha=desc[9:12].decode("ascii").strip(),
                start=start,
                rate=nsamp / (tlen / 1000.0),
                data=data,
            )
        )
        q += 4 + clen
    return out


def file_station(src: Path) -> str:
    """``20210212_000000_C_MB-7A.cd11`` -> ``MB7A``."""
    return station_code(src.stem.rsplit("_", 1)[-1])


def build_stream(srcs: list[Path], codes: Codes) -> Stream:
    """Merged stream of every subframe in ``srcs``."""
    st = Stream()
    for src in srcs:
        for sf in read_cd11(src):
            data = sf["data"]
            data = data.astype(np.int32 if data.dtype.kind == "i" else np.float64)
            st.append(
                Trace(
                    data=data,
                    header=dict(
                        network=codes.network,
                        station=station_code(sf["sta"]),
                        location=codes.location,
                        channel=channel_code(sf["cha"], codes),
                        starttime=sf["start"],
                        sampling_rate=sf["rate"],
                    ),
                )
            )
    st.merge(method=-1)
    st.sort()
    return st


def convert_station(item: tuple[str, list[Path], Path, Codes, bool, bool]) -> dict:
    """Worker: write one station's day files (unless they exist / dry run)."""
    sta, srcs, out_dir, codes, overwrite, dry_run = item
    st = build_stream(srcs, codes)
    meta = dict(station=sta, n_files=len(srcs), written=0, skipped=0, segments=0, warnings=[])
    for seed_id in sorted({tr.id for tr in st}):
        segs = st.select(id=seed_id)
        overlaps = sum(
            1 for a, b in zip(segs, segs[1:], strict=False) if b.stats.starttime <= a.stats.endtime
        )
        if overlaps:
            meta["warnings"].append(f"{seed_id}: {overlaps} overlap(s) with differing samples")
    if not len(st):
        return meta
    day = UTCDateTime(min(tr.stats.starttime for tr in st).date)
    t_last = max(tr.stats.endtime for tr in st)
    while day <= t_last:
        st_day = st.slice(day, day + 86400 - 1e-6, nearest_sample=False)
        st_day.traces = [tr for tr in st_day if tr.stats.npts]
        for seed_id in sorted({tr.id for tr in st_day}):
            part = st_day.select(id=seed_id)
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
        day += 86400
    return meta


def convert(
    in_dir: Path,
    out_dir: Path,
    pcfg: ParallelConfigLike,
    codes: Codes | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    pattern: str = "*.cd11",
) -> list[dict]:
    """Convert every CD-1.1 file under ``in_dir``; returns the per-station results."""
    codes = codes or Codes()
    srcs = sorted(in_dir.rglob(pattern))
    if not srcs:
        raise FileNotFoundError(f"no {pattern} files under {in_dir}")
    by_sta: dict[str, list[Path]] = defaultdict(list)
    for src in srcs:
        by_sta[file_station(src)].append(src)
    logger.info("%d CD-1.1 files from %d stations -> %s", len(srcs), len(by_sta), out_dir)
    items = [
        (sta, files, out_dir, codes, overwrite, dry_run) for sta, files in sorted(by_sta.items())
    ]
    results = pmap(convert_station, items, pcfg, desc="cd11 -> miniSEED")
    for r in results:
        for w in r["warnings"]:
            logger.warning("%s", w)
    skipped = sum(r["skipped"] for r in results)
    if skipped:
        logger.warning("%d day files already existed and were left as-is (no overwrite)", skipped)
    verb = "would write" if dry_run else "wrote"
    logger.info(
        "%s %d day files (%d segments) from %d CD-1.1 files",
        verb,
        sum(r["written"] for r in results),
        sum(r["segments"] for r in results),
        len(srcs),
    )
    return results
