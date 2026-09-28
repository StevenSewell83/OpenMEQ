"""hypoDD/ph2dt 1D input file writers: station.dat, phase.dat, event.dat,
ph2dt.inp, hypoDD.inp.

Plan key: M9.2 / OpenMEQ-6ld.2. Every file format below is written exactly
as confirmed against the bundled ``external/HYPODD`` source and
``examples/example1`` by the M9 epic's design note (``bd show OpenMEQ-6ld``)
and ``docs/hypoddpy_audit.md`` §6 (master defect table) -- see those for
file:line citations against ``getdata.f``, ``getinp2.f`` and ``ph2dt.f``.
Do not re-derive formats from generic hypoDD documentation.

This module fixes every defect ``docs/hypoddpy_audit.md`` §6 assigns to
M9.2:

- #11: station labels are mapped to a unique ``character*7`` hypoDD label
  instead of writing an unmapped, routinely-too-long ``NET.STA``.
- #7: a negative effective station elevation (boreholes) is written as-is
  (not silently clamped) but logged loudly, once per run, naming every
  affected station -- hypoDD itself still clamps it internally unless
  ``IMOD=5``, which this module does not select.
- #3: ``hypoDD.inp``'s 1D block targets ``IMOD=1`` (variable vp/vs ratio),
  not the legacy ``IMOD=0``.
- #4/#14: ``IAQ`` and the ``OBSCC``/``OBSCT`` event-clustering line are
  config-driven, not hardcoded (hypoddpy hardcodes ``OBSCC=OBSCT=0``,
  silently disabling clustering).
- #5: the weighting table defaults to example1's real 5-row scheme, not a
  minimal 2-row default.
- #12: ``phase.dat`` weight is derived from ``Arrival.time_weight``,
  written as hypoDD's own ``-1`` "no info" sentinel when absent, not a
  constant ``1.0``/``0.0``.
- #6: the event-id (cuspid) and station-label mappings are persisted,
  append-only CSVs (``relocate/work/*.csv``), not regenerated in memory on
  every run.

See ``docs/ARCHITECTURE.md`` §6c for the summarised contract.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Origin

from openmeq.catalog import accepted, event_id
from openmeq.config import Config, WeightingRow
from openmeq.config import ConfigError as _ConfigError
from openmeq.log import get_logger

logger = get_logger("relocate.hypodd_files")

#: hypoDD station labels are ``character*7`` (getdata.f:51).
STATION_LABEL_MAX_LEN = 7


# ---------------------------------------------------------------------------
# shared helpers (ARCHITECTURE.md §6c "Pitfalls": one conversion helper, used
# everywhere, rather than re-deriving the metres<->km conversion per file)
# ---------------------------------------------------------------------------


def depth_m_to_km(depth_m: float) -> float:
    """Convert a QuakeML depth (metres, positive down) to hypoDD's km convention.

    Used by every hypoDD/ph2dt writer in this module (``event.dat``,
    ``phase.dat``'s header) and, in M10.4, when reading ``hypoDD.reloc``
    back -- the one shared, tested depth conversion helper the M9 epic
    design note's "Pitfalls" section calls for.

    Parameters
    ----------
    depth_m : float
        Depth in metres (QuakeML ``Origin.depth`` convention).

    Returns
    -------
    float
        Depth in km.
    """
    return depth_m / 1000.0


def _preferred_origin(event: Event) -> Origin:
    origin = event.preferred_origin()
    if origin is None:
        if not event.origins:
            raise ValueError(f"Event {event.resource_id} has no origins")
        origin = event.origins[0]
    return origin


def _net_sta(waveform_id) -> str:
    return f"{waveform_id.network_code}.{waveform_id.station_code}"


def _preferred_magnitude(event: Event) -> float:
    mag = event.preferred_magnitude()
    if mag is not None and mag.mag is not None:
        return float(mag.mag)
    if event.magnitudes:
        for m in event.magnitudes:
            if m.mag is not None:
                return float(m.mag)
    return 0.0


def _origin_errors_km(origin: Origin) -> tuple[float, float]:
    """Horizontal/vertical location error, in km, for event.dat/phase.dat.

    Returns ``(0.0, 0.0)`` for whichever component isn't populated (both
    are optional in QuakeML).
    """
    herr = 0.0
    ou = origin.origin_uncertainty
    if ou is not None and ou.horizontal_uncertainty is not None:
        herr = ou.horizontal_uncertainty / 1000.0
    verr = 0.0
    if origin.depth_errors is not None and origin.depth_errors.uncertainty is not None:
        verr = origin.depth_errors.uncertainty / 1000.0
    return herr, verr


def _origin_residual_s(origin: Origin) -> float:
    if origin.quality is not None and origin.quality.standard_error is not None:
        return float(origin.quality.standard_error)
    return 0.0


def _unpacked_time_fields(t: UTCDateTime) -> tuple[int, int, int, int, int, float]:
    """``phase.dat``'s header time encoding: separate YR MO DY HR MN fields
    plus a decimal-seconds value (confirmed ``ph2dt.f:474-477``), as opposed
    to ``event.dat``'s single concatenated integer (see
    :func:`_time_to_concat_int`) -- two different encodings of the same
    physical value, deliberately kept as two separate, separately-tested
    helpers per the M9 epic design note's "Pitfalls" section.
    """
    t = UTCDateTime(t)
    sec = t.second + t.microsecond / 1.0e6
    return t.year, t.month, t.day, t.hour, t.minute, sec


def _time_to_concat_int(t: UTCDateTime) -> int:
    """``event.dat``'s concatenated ``HHMMSSCC`` time integer.

    ``hour*1000000 + minute*10000 + second*100 + centisecond``, confirmed by
    reverse-engineering ``examples/example1/event.dat`` against
    ``El16.pha``'s header for the same event (hour=2, minute=19,
    second=58.71 -> ``2195871``) and cross-checked against
    ``hypoDD.f:1292-1295``'s inverse computation.
    """
    t = UTCDateTime(t)
    centisecond = int(round(t.microsecond / 10000.0))
    second, minute, hour = t.second, t.minute, t.hour
    if centisecond == 100:
        centisecond = 0
        second += 1
        if second == 60:
            second = 0
            minute += 1
            if minute == 60:
                minute = 0
                hour += 1
    return hour * 1000000 + minute * 10000 + second * 100 + centisecond


def _date_int(t: UTCDateTime) -> int:
    """``event.dat``'s ``YYYYMMDD`` date integer."""
    t = UTCDateTime(t)
    return t.year * 10000 + t.month * 100 + t.day


# ---------------------------------------------------------------------------
# persisted id-mapping csv helpers
# ---------------------------------------------------------------------------


def _read_str_int_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return {}
    return {key: int(value) for key, value in rows[1:]}


def _write_str_int_map(path: Path, mapping: dict[str, int], header: tuple[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for key, value in mapping.items():
            writer.writerow([key, value])


def _read_str_str_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return {}
    return dict(rows[1:])


def _write_str_str_map(path: Path, mapping: dict[str, str], header: tuple[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for key, value in mapping.items():
            writer.writerow([key, value])


# ---------------------------------------------------------------------------
# event id (cuspid) mapping
# ---------------------------------------------------------------------------


def build_event_id_map(catalog: Catalog, map_path: Path | str) -> dict[str, int]:
    """Build (and persist) the ``event_id`` <-> hypoDD ``cuspid`` mapping.

    Sorts *newly seen* accepted events by origin time and assigns the next
    integer cuspids (1..N on a first build); an existing mapping file at
    ``map_path`` is loaded first and reused across runs -- existing rows
    are never renumbered, only new events are appended (docs/hypoddpy_audit.md
    §6 defect #6: hypoddpy's template regenerates this mapping in memory on
    every run, a correctness risk for a resumable pipeline whose
    cross-correlation cache is keyed by cuspid pairs).

    Parameters
    ----------
    catalog : obspy.Catalog
        The full catalogue (rejected events are filtered out via
        `openmeq.catalog.accepted`).
    map_path : pathlib.Path or str
        Where the mapping is persisted, e.g.
        ``relocate/work/event_id_map.csv`` (columns ``event_id,cuspid``).

    Returns
    -------
    dict[str, int]
        ``{event_id: cuspid}`` for every accepted event in `catalog`,
        including any pre-existing entries carried over from `map_path`.
    """
    map_path = Path(map_path)
    existing = _read_str_int_map(map_path)

    new_events: list[tuple[str, UTCDateTime]] = []
    for ev in accepted(catalog):
        eid = event_id(ev)
        if eid not in existing:
            new_events.append((eid, _preferred_origin(ev).time))
    new_events.sort(key=lambda pair: pair[1])

    next_cuspid = max(existing.values(), default=0) + 1
    updated = dict(existing)
    for eid, _ in new_events:
        updated[eid] = next_cuspid
        next_cuspid += 1

    _write_str_int_map(map_path, updated, ("event_id", "cuspid"))
    return updated


def invert_event_id_map(event_id_map: dict[str, int]) -> dict[int, str]:
    """Return the ``cuspid -> event_id`` inverse of `build_event_id_map`'s output."""
    return {cuspid: eid for eid, cuspid in event_id_map.items()}


# ---------------------------------------------------------------------------
# station label mapping
# ---------------------------------------------------------------------------


def build_station_label_map(station_table: pd.DataFrame, map_path: Path | str) -> dict[str, str]:
    """Build (and persist) the ``NET.STA`` <-> hypoDD station-label mapping.

    hypoDD station labels are ``character*7`` (`getdata.f:51`), while
    OpenMEQ's ``NET.STA`` convention (ARCHITECTURE.md §6) can exceed 7
    characters. Each station's bare ``STA`` code is tried first; on a
    collision across networks (same bare code, per ARCHITECTURE.md §6 the
    ``NET.STA`` pair is unique but the bare code need not be), a
    single-digit suffix is appended, deterministically, in ``NET.STA`` sort
    order. A collision that cannot be disambiguated within 7 characters
    (the bare code is already 7 characters, or all ten digit-suffixed
    variants are taken) raises `openmeq.config.ConfigError` naming the
    colliding station -- never a silent mismapping.

    Like `build_event_id_map`, an existing mapping file at `map_path` is
    loaded first and reused across runs: existing labels are never
    reassigned, only new stations are appended.

    Parameters
    ----------
    station_table : pandas.DataFrame
        Columns per `openmeq.stations.station_table`'s contract:
        ``network, station, location, latitude, longitude, elevation_m,
        depth_m`` (only ``network``/``station`` are used here).
    map_path : pathlib.Path or str
        Where the mapping is persisted, e.g.
        ``relocate/work/station_label_map.csv`` (columns ``net_sta,label``).

    Returns
    -------
    dict[str, str]
        ``{"NET.STA": label}`` for every station in `station_table`,
        including any pre-existing entries carried over from `map_path`.

    Raises
    ------
    openmeq.config.ConfigError
        If a station's bare code collides with another's already-assigned
        label and cannot be disambiguated within 7 characters.
    """
    map_path = Path(map_path)
    label_map = _read_str_str_map(map_path)
    used_labels = set(label_map.values())

    keys = sorted(
        {
            f"{net}.{sta}"
            for net, sta in zip(station_table["network"], station_table["station"], strict=True)
        }
    )

    for key in keys:
        if key in label_map:
            continue
        bare = key.split(".", 1)[1]
        label = _assign_station_label(bare, key, used_labels)
        label_map[key] = label
        used_labels.add(label)

    _write_str_str_map(map_path, label_map, ("net_sta", "label"))
    return label_map


def _assign_station_label(bare: str, key: str, used_labels: set[str]) -> str:
    if len(bare) <= STATION_LABEL_MAX_LEN and bare not in used_labels:
        return bare
    if len(bare) <= STATION_LABEL_MAX_LEN - 1:
        for suffix in range(10):
            candidate = f"{bare}{suffix}"
            if candidate not in used_labels:
                return candidate
    if len(bare) >= STATION_LABEL_MAX_LEN:
        reason = (
            f"its {STATION_LABEL_MAX_LEN}-character bare station code collides with "
            "another station's hypoDD label, leaving no spare character to append "
            "a disambiguating digit"
        )
    else:
        reason = "every single-digit-suffixed variant of its bare station code is already taken"
    raise _ConfigError(
        f"Cannot map station {key!r} (bare code {bare!r}) to a unique hypoDD station "
        f"label of at most {STATION_LABEL_MAX_LEN} characters: {reason}. Rename one of "
        "the colliding stations, or shorten its code, to disambiguate within 7 characters."
    )


# ---------------------------------------------------------------------------
# station.dat
# ---------------------------------------------------------------------------


def write_station_dat(
    path: Path | str,
    station_table: pd.DataFrame,
    label_map: dict[str, str],
    cfg: Config,
) -> None:
    """Write hypoDD's ``station.dat``: ``label lat lon elevation_m`` per station.

    Elevation is the ARCHITECTURE.md §6 *effective* elevation
    (``elevation_m - depth_m``, negative for a downhole borehole sensor),
    written in **metres** -- hypoDD divides by 1000 internally
    (`getdata.f:274`), so this writer must not pre-convert to km.

    Any station with a negative effective elevation is written as-is (the
    true value), but logged once, at WARNING level, naming every affected
    station: `getdata.f:275-278` silently zeroes a negative station
    elevation unless ``IMOD=5`` (the straight-ray-only forward model), and
    this module targets ``IMOD=1`` (docs/hypoddpy_audit.md §6 defect #7).
    This is a real, easy-to-miss data-quality issue for any project with
    boreholes; OpenMEQ surfaces it loudly rather than silently replicating
    hypoDD's own silent clamp or silently switching forward models.

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, e.g. ``relocate/work/station.dat``.
    station_table : pandas.DataFrame
        `openmeq.stations.station_table`'s contract: columns ``network,
        station, location, latitude, longitude, elevation_m, depth_m``.
    label_map : dict[str, str]
        ``{"NET.STA": label}``, from `build_station_label_map`.
    cfg : openmeq.config.Config
        Accepted for interface symmetry with the other writers in this
        module (and so a future IMOD-aware policy, e.g. M9.3's 3D writer,
        can read it); not otherwise used by the 1D writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `cfg` isn't otherwise used by the 1D writer -- see docstring.

    rows = []
    negative = []
    ordered = station_table.sort_values(["network", "station"])
    for _, row in ordered.iterrows():
        net_sta = f"{row['network']}.{row['station']}"
        label = label_map[net_sta]
        elevation_m = float(row["elevation_m"]) - float(row["depth_m"])
        if elevation_m < 0:
            negative.append((net_sta, elevation_m))
        rows.append((label, float(row["latitude"]), float(row["longitude"]), elevation_m))

    if negative:
        names = ", ".join(f"{net_sta} ({elv:.1f} m)" for net_sta, elv in negative)
        logger.warning(
            "%d station(s) have a negative effective elevation (elevation_m - depth_m) "
            "and will be silently zeroed by hypoDD unless IMOD=5 is selected "
            "(getdata.f:275-278); station.dat still records the true value: %s",
            len(negative),
            names,
        )

    with open(path, "w") as fh:
        for label, lat, lon, elevation_m in rows:
            fh.write(f"{label:<7s} {lat:10.6f} {lon:11.6f} {elevation_m:9.2f}\n")


# ---------------------------------------------------------------------------
# phase.dat
# ---------------------------------------------------------------------------


def write_phase_dat(
    path: Path | str,
    catalog: Catalog,
    event_id_map: dict[str, int],
    label_map: dict[str, str],
) -> None:
    """Write ph2dt's ``phase.dat``: one ``#``-header line per event followed
    by one line per pick.

    Header: ``# YR MO DY HR MN SEC.SS LAT LON DEPTH_KM MAG HERR VERR RES
    CUSPID`` -- the *unpacked* time (confirmed `ph2dt.f:474-477`), not
    `event.dat`'s concatenated integer form.

    Phase line: ``STA TRAVEL_TIME_S WEIGHT PHASE`` (confirmed
    `ph2dt.f:483-484`). ``TRAVEL_TIME_S = pick.time - origin.time``.
    ``WEIGHT`` is `Arrival.time_weight` clamped to ``[0, 1]``, or ``-1.0``
    (hypoDD's own "no info" sentinel, confirmed `ph2dt.f:485`: a weight
    ``>= 0`` and ``< minwght`` is skipped, but a negative weight always
    passes) when the pick has no `Arrival.time_weight` at all -- writing
    ``0.0`` instead would mean "lowest possible quality" and could be
    filtered out unintentionally.

    A pick whose `evaluation_status` is ``"rejected"`` is **excluded
    entirely**, not merely weighted to zero: ARCHITECTURE.md §5's rejection
    convention sets ``arrival.time_weight = 0`` on rejection, which would
    otherwise look like a deliberately-low-quality but *present* pick to
    ph2dt's ``MINWGHT`` filter and still be included with the default
    ``minwght=0``.

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, e.g. ``relocate/work/phase.dat``.
    catalog : obspy.Catalog
        The full catalogue; rejected events are filtered via
        `openmeq.catalog.accepted`.
    event_id_map : dict[str, int]
        ``{event_id: cuspid}``, from `build_event_id_map`. Every accepted
        event in `catalog` must have an entry.
    label_map : dict[str, str]
        ``{"NET.STA": label}``, from `build_station_label_map`. Every
        station referenced by a written pick must have an entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        for event in accepted(catalog):
            eid = event_id(event)
            cuspid = event_id_map[eid]
            origin = _preferred_origin(event)
            yr, mo, dy, hr, mn, sec = _unpacked_time_fields(origin.time)
            depth_km = depth_m_to_km(origin.depth) if origin.depth is not None else 0.0
            mag = _preferred_magnitude(event)
            herr, verr = _origin_errors_km(origin)
            res = _origin_residual_s(origin)

            fh.write(
                f"# {yr:4d} {mo:2d} {dy:2d} {hr:2d} {mn:2d} {sec:5.2f} "
                f"{origin.latitude:9.4f} {origin.longitude:10.4f} {depth_km:8.2f} "
                f"{mag:4.2f} {herr:5.2f} {verr:5.2f} {res:5.2f} {cuspid:7d}\n"
            )

            picks_by_id = {str(pick.resource_id): pick for pick in event.picks}
            for arrival in origin.arrivals:
                pick = picks_by_id.get(str(arrival.pick_id))
                if pick is None:
                    continue
                if pick.evaluation_status == "rejected":
                    continue

                label = label_map[_net_sta(pick.waveform_id)]
                travel_time_s = pick.time - origin.time
                weight = arrival.time_weight
                weight = -1.0 if weight is None else max(0.0, min(1.0, weight))
                phase = arrival.phase or pick.phase_hint or "P"

                fh.write(f"{label:<7s} {travel_time_s:8.3f} {weight:7.3f}   {phase}\n")


# ---------------------------------------------------------------------------
# event.dat
# ---------------------------------------------------------------------------


def write_event_dat(
    path: Path | str,
    catalog: Catalog,
    event_id_map: dict[str, int],
) -> None:
    """Write hypoDD's ``event.dat``: one line per event.

    ``YYYYMMDD HHMMSSCC LAT LON DEPTH_KM MAG HERR_KM ZERR_KM RES_S CUSPID``
    (10 fields; confirmed `getdata.f:159-188`'s ``j.eq.10`` branch). The
    time is a single **concatenated integer** (``HHMMSSCC``, see
    `_time_to_concat_int`) -- a different encoding from `phase.dat`'s
    unpacked header fields for the same physical value.

    This module always writes the 10-field form (no trailing ``FIX`` flag):
    OpenMEQ does not currently support fixing specific events' locations, so
    the simpler 10-field form is equivalent to the 11-field form with
    ``FIX=0`` for every event (`getdata.f:178-188`'s ``j.eq.11`` branch, vs.
    ``j.eq.10``'s implicit ``ev_fix(i) = 0``).

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, e.g. ``relocate/work/event.dat``.
    catalog : obspy.Catalog
        The full catalogue; rejected events are filtered via
        `openmeq.catalog.accepted`.
    event_id_map : dict[str, int]
        ``{event_id: cuspid}``, from `build_event_id_map`. Every accepted
        event in `catalog` must have an entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        for event in accepted(catalog):
            eid = event_id(event)
            cuspid = event_id_map[eid]
            origin = _preferred_origin(event)
            date_int = _date_int(origin.time)
            time_int = _time_to_concat_int(origin.time)
            depth_km = depth_m_to_km(origin.depth) if origin.depth is not None else 0.0
            mag = _preferred_magnitude(event)
            herr, verr = _origin_errors_km(origin)
            res = _origin_residual_s(origin)

            fh.write(
                f"{date_int:8d} {time_int:9d} {origin.latitude:9.4f} "
                f"{origin.longitude:10.4f} {depth_km:8.3f} {mag:5.2f} {herr:7.2f} "
                f"{verr:7.2f} {res:6.2f} {cuspid:10d}\n"
            )


# ---------------------------------------------------------------------------
# ph2dt.inp
# ---------------------------------------------------------------------------


def write_ph2dt_inp(
    path: Path | str,
    cfg: Config,
    station_file: str = "station.dat",
    phase_file: str = "phase.dat",
) -> None:
    """Write ``ph2dt.inp``: station/phase file paths, then one line of 7
    numeric fields (``MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS``,
    confirmed by `examples/example1/ph2dt.inp`'s own comments and
    `ph2dt.f:474`'s read).

    Comment lines (starting with ``*``) are freely interspersed for
    readability -- ``ph2dt.f:234`` skips any line starting with ``*`` in
    column 1 or 2, but does **not** skip blank lines, so this writer never
    emits one.

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, e.g. ``relocate/work/ph2dt.inp``.
    cfg : openmeq.config.Config
        ``cfg.hypodd.ph2dt``'s fields are used directly.
    station_file, phase_file : str, optional
        Filenames referenced by the written ``ph2dt.inp`` (relative to
        wherever ph2dt is run from); default to this module's own
        `write_station_dat`/`write_phase_dat` output names.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = cfg.hypodd.ph2dt

    with open(path, "w") as fh:
        fh.write("* ph2dt.inp - input control file for program ph2dt (written by openmeq)\n")
        fh.write("* Input station file:\n")
        fh.write(f"{station_file}\n")
        fh.write("* Input phase file:\n")
        fh.write(f"{phase_file}\n")
        fh.write("*MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS\n")
        fh.write(
            f"{p.minwght:g} {p.maxdist_km:g} {p.maxsep_km:g} "
            f"{p.maxngh:d} {p.minlnk:d} {p.minobs:d} {p.maxobs:d}\n"
        )


# ---------------------------------------------------------------------------
# hypoDD.inp (1D, IMOD=1)
# ---------------------------------------------------------------------------


def write_hypodd_inp_1d(
    path: Path | str,
    cfg: Config,
    weighting_rows: list[WeightingRow],
    model: pd.DataFrame,
) -> None:
    """Write ``hypoDD.inp`` targeting ``IMOD=1`` (1D, per-layer variable
    vp/vs ratio), matching the exact section order confirmed against
    `examples/example1/hypoDD.inp` (quoted in the M9 epic's design note;
    see `docs/ARCHITECTURE.md` §6c for the summary).

    References `event.sel`/`station.sel` (ph2dt's filtered output) as the
    event/station files, per the standard hypoDD workflow -- not the raw
    `event.dat`/`station.dat` this module also writes. Always writes
    ``CID=0`` (relocate all clusters) with a blank ``ID`` list (relocate
    every event in the cluster); M10.4's ``parallel_clusters`` option
    relocates clusters independently instead of extending this writer.

    Every data line here matches one counted, comment-skipped "line" in
    `getinp2.f`'s parser (`line(1:1).eq.'*'` skips a comment, but a blank
    line is *not* skipped and would desynchronise the parser) -- comment
    lines are freely interspersed, blank lines never are.

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, e.g. ``relocate/work/hypoDD.inp``.
    cfg : openmeq.config.Config
        ``cfg.hypodd``'s ``idat``, ``ipha``, ``max_station_dist_km``,
        ``clustering``, ``istart``, ``isolv``, ``iaq`` fields are used
        directly.
    weighting_rows : list[openmeq.config.WeightingRow]
        The iterative data-weighting/re-weighting table (hypoDD.inp's
        ``NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP`` rows).
        Passed explicitly (rather than always reading
        ``cfg.hypodd.weighting_rows``) so callers can override it, e.g. for
        a smaller test fixture.
    model : pandas.DataFrame
        The 1D layer model, ARCHITECTURE.md §6a's contract: columns
        ``depth_km, vp, vs`` (and optionally ``rho``, unused here), as
        returned by `openmeq.velocity.read_layers`. ``RATIO`` (Vp/Vs per
        layer) is computed as ``vp / vs``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = cfg.hypodd
    c = h.clustering

    top = [float(v) for v in model["depth_km"]]
    vel = [float(v) for v in model["vp"]]
    ratio = [float(vp) / float(vs) for vp, vs in zip(model["vp"], model["vs"], strict=True)]

    with open(path, "w") as fh:
        fh.write("hypoDD_2\n")
        fh.write("* input file selection\n")
        fh.write("dt.cc\n")
        fh.write("dt.ct\n")
        fh.write("* event file (ph2dt's filtered output):\n")
        fh.write("event.sel\n")
        fh.write("* station file (ph2dt's filtered output):\n")
        fh.write("station.sel\n")
        fh.write("* output file selection\n")
        fh.write("hypoDD.loc\n")
        fh.write("hypoDD.reloc\n")
        fh.write("hypoDD.sta\n")
        fh.write("hypoDD.res\n")
        fh.write("hypoDD.src\n")
        fh.write("* data type selection: IDAT IPHA DIST\n")
        fh.write(f"{h.idat:d} {h.ipha:d} {h.max_station_dist_km:g}\n")
        fh.write("* event clustering: OBSCC OBSCT MINDIST MAXDIST MAXGAP\n")
        fh.write(f"{c.obscc:d} {c.obsct:d} {c.mindist_km:g} {c.maxdist_km:g} {c.maxgap_deg:g}\n")
        fh.write("* solution control: ISTART ISOLV IAQ NSET\n")
        fh.write(f"{h.istart:d} {h.isolv:d} {h.iaq:d} {len(weighting_rows):d}\n")
        fh.write("* data weighting/re-weighting:\n")
        fh.write("* NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP\n")
        for row in weighting_rows:
            fh.write(
                f"{row.niter:d} {row.wtccp:g} {row.wtccs:g} {row.wrcc:g} {row.wdcc:g} "
                f"{row.wtctp:g} {row.wtcts:g} {row.wrct:g} {row.wdct:g} {row.damp:g}\n"
            )
        fh.write("* IMOD: 1 = 1D, layered, variable vp/vs ratio per layer\n")
        fh.write("1\n")
        fh.write(" ".join(f"{v:g}" for v in top) + " -9\n")
        fh.write(" ".join(f"{v:g}" for v in vel) + " -9\n")
        fh.write(" ".join(f"{v:g}" for v in ratio) + " -9\n")
        fh.write("* event selection: CID (0 = all)\n")
        fh.write("0\n")
        fh.write("* ID (blank = all events in the selected cluster)\n")
