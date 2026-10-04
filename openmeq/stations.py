"""Station metadata: StationXML file/folder + CSV merge, channel mapping.

Implements ``openmeq.stations`` (ARCHITECTURE.md §2 table, §6): merging a
StationXML file, a folder of StationXML files, and a CSV into one
``obspy.Inventory`` and a flat station table; relabelling/rotating channel
codes ahead of QC, detection and PhaseNet prep (``apply_channel_map``); and a
data-vs-metadata coverage report.

Design notes
------------
* **Merge precedence** (``load_inventory``): StationXML file, then StationXML
  folder, then CSV -- each layer only fills in stations/channels the earlier
  layers didn't already provide. Two stations sharing ``NET.STA`` (even with
  different location codes) merge into a single `Station` object
  (ARCHITECTURE.md §6: "location codes are ignored when two share a site");
  channel *epochs* are de-duplicated by ``(location_code, code,
  start_date)`` so different location codes' channels, and every epoch of
  each, all survive under that one station (OpenMEQ-hud: de-duplicating by
  ``(location_code, code)`` alone kept only the first epoch -- RK-NM's ARAZ
  lost its 2011-06-> epoch and had no response for 2026).
* **Overlapping epochs are closed** (:func:`_close_overlapping_epochs`): an
  epoch with no end date (or one ending after the next epoch of the same
  ``(location_code, code)`` starts) is ended at that next epoch's start, with
  a warning. Vendor StationXML often leaves superseded epochs open; obspy
  and ObsPyck then return the *first* match (RK-NM RT01: the 2008-07
  LE-3Dlite response instead of the 2008-10-> GS-11D one). This is a
  real merge (walking and extending `Inventory.networks`/`Station.channels`
  in place), not ``Inventory + Inventory`` concatenation, which would create
  duplicate `Network`/`Station` objects for the same codes.
* **CSV rows carry no channel code** (``net,sta,loc,lat,lon,elev_m,depth_m``,
  PLAN.md §3), so a CSV-only station is merged in as a `Station` with no
  `Channel` children -- it can still appear in ``station_table`` and be
  reported as "metadata without data" is moot for it (it declares no
  channels to be missing data for), but it cannot supply a response or a
  channel-level seed id.
* **Station codes must be unique across networks, and <= 7 characters**
  (ARCHITECTURE.md §6, the hypoDD limit): checked once, while merging, and
  raised as `openmeq.config.ConfigError` (the shared "user-fixable
  config/metadata problem" exception, per that module's docstring) so a bad
  station list fails clearly at load time rather than downstream in NLLoc/
  hypoDD.
* **Depth vs elevation**: this module stores ``elevation_m``/``depth_m``
  separately in ``station_table``; ARCHITECTURE.md §6's *effective*
  elevation (``elevation_m - depth_m``) used by NLLoc/hypoDD is computed by
  those later steps (M5/M9), not here.
* **check_coverage takes a plain iterable of seed_ids**, not an
  ``openmeq.index.DataIndex``. The M1.1 index module is being built
  concurrently with this task, so this module deliberately does not import
  it (and per the design note below, never scans miniSEED itself either) --
  it only needs *some* iterable of ``"NET.STA.LOC.CHA"`` strings. The
  intended wiring (e.g. for a later CLI/QC-step task, M1.7 or whichever
  wires ``openmeq qc``) is::

      from openmeq.index import DataIndex
      from openmeq.stations import check_coverage, write_metadata_report

      index = DataIndex.load(paths.index_json)
      inv = load_inventory(cfg.stations)
      report = check_coverage(index.seed_ids(), inv)
      write_metadata_report(report, paths)

  ``DataIndex.seed_ids()`` (ARCHITECTURE.md §2) already returns exactly such
  an iterable, so no adapter is needed once M1.1 lands.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import Stream, read_inventory
from obspy.core.inventory import Inventory, Network, Station

from openmeq.config import ConfigError
from openmeq.log import get_logger

if TYPE_CHECKING:
    from openmeq.config import StationsConfig
    from openmeq.paths import RunPaths

logger = get_logger("stations")

#: ``station_table`` / ``inventory/stations.csv`` column order (ARCHITECTURE.md §2).
STATION_TABLE_COLUMNS = [
    "network",
    "station",
    "location",
    "latitude",
    "longitude",
    "elevation_m",
    "depth_m",
]

#: hypoDD's compiled station-code field width (ARCHITECTURE.md §6).
MAX_STATION_CODE_LEN = 7

#: ``stations_cfg.csv`` columns (PLAN.md §3).
_CSV_COLUMNS = ["net", "sta", "loc", "lat", "lon", "elev_m", "depth_m"]


# ---------------------------------------------------------------------------
# load_inventory
# ---------------------------------------------------------------------------


def load_inventory(stations_cfg: StationsConfig) -> Inventory:
    """Merge every configured station-metadata source into one `Inventory`.

    Merges, in this precedence order (later sources only fill gaps the
    earlier ones left; XML always wins over CSV for the same
    ``NET.STA.LOC.CHA``):

    1. ``stations_cfg.stationxml`` (a single StationXML file), if set.
    2. Every ``*.xml``/``*.stationxml`` in ``stations_cfg.stationxml_dir``
       (sorted for determinism), if set.
    3. ``stations_cfg.csv`` (``net,sta,loc,lat,lon,elev_m,depth_m``), if set
       -- for any ``NET.STA`` not already present, or to fill in
       latitude/longitude/elevation an XML entry lacks. A CSV row carries no
       channel code, so a CSV-only station is merged in with no channels.

    Two stations sharing ``NET.STA`` (regardless of location code) are
    merged into one `Station`; its channels are the union, de-duplicated by
    ``(location_code, code, start_date)``, of every source's channel epochs
    for that station (ARCHITECTURE.md §6: location codes are ignored when
    merging duplicate sites -- but no channel or epoch is dropped).
    Overlapping epochs of one channel are then closed
    (:func:`_close_overlapping_epochs`).

    Parameters
    ----------
    stations_cfg : openmeq.config.StationsConfig
        ``cfg.stations``.

    Returns
    -------
    obspy.core.inventory.Inventory
        The merged inventory (``source="openmeq merged inventory"``).

    Raises
    ------
    openmeq.config.ConfigError
        If none of ``stationxml``/``stationxml_dir``/``csv`` is set, if a
        station code is longer than :data:`MAX_STATION_CODE_LEN` characters,
        or if the same bare station code is used by two different networks
        (ARCHITECTURE.md §6 -- NLLoc/hypoDD match on the bare code).
    """
    if not (stations_cfg.stationxml or stations_cfg.stationxml_dir or stations_cfg.csv):
        raise ConfigError(
            "stations config has none of stationxml/stationxml_dir/csv set -- "
            "at least one station metadata source is required"
        )

    networks_by_code: dict[str, Network] = {}
    stations_by_net_code: dict[str, dict[str, Station]] = {}
    station_owner_network: dict[str, str] = {}

    def _merge_in(src_inv: Inventory) -> None:
        for net in src_inv.networks:
            target_net = networks_by_code.get(net.code)
            if target_net is None:
                target_net = Network(
                    code=net.code,
                    stations=[],
                    description=net.description,
                    start_date=net.start_date,
                    end_date=net.end_date,
                )
                networks_by_code[net.code] = target_net
                stations_by_net_code[net.code] = {}
            sta_index = stations_by_net_code[net.code]

            for sta in net.stations:
                code = sta.code
                if len(code) > MAX_STATION_CODE_LEN:
                    raise ConfigError(
                        f"station code {net.code}.{code!r} is {len(code)} characters, "
                        f"more than the {MAX_STATION_CODE_LEN}-character hypoDD limit "
                        "(ARCHITECTURE.md §6)"
                    )
                owner = station_owner_network.get(code)
                if owner is not None and owner != net.code:
                    raise ConfigError(
                        f"station code {code!r} is used by both network {owner!r} and "
                        f"{net.code!r} -- station codes must be unique across networks "
                        "(ARCHITECTURE.md §6)"
                    )
                station_owner_network[code] = net.code

                existing = sta_index.get(code)
                if existing is None:
                    existing = Station(
                        code=code,
                        latitude=sta.latitude,
                        longitude=sta.longitude,
                        elevation=sta.elevation,
                        channels=[],
                        site=sta.site,
                        creation_date=sta.creation_date,
                    )
                    sta_index[code] = existing
                    target_net.stations.append(existing)
                else:
                    if existing.latitude is None and sta.latitude is not None:
                        existing.latitude = sta.latitude
                    if existing.longitude is None and sta.longitude is not None:
                        existing.longitude = sta.longitude
                    if existing.elevation is None and sta.elevation is not None:
                        existing.elevation = sta.elevation

                existing_keys = {_epoch_key(c) for c in existing.channels}
                for cha in sta.channels:
                    key = _epoch_key(cha)
                    if key not in existing_keys:
                        existing.channels.append(cha)
                        existing_keys.add(key)

    if stations_cfg.stationxml:
        _merge_in(read_inventory(str(stations_cfg.stationxml)))

    if stations_cfg.stationxml_dir:
        xml_dir = Path(stations_cfg.stationxml_dir)
        paths = sorted(set(xml_dir.glob("*.xml")) | set(xml_dir.glob("*.stationxml")))
        for p in paths:
            _merge_in(read_inventory(str(p)))

    if stations_cfg.csv:
        _merge_in(_inventory_from_csv(stations_cfg.csv))

    for net in networks_by_code.values():
        for sta in net.stations:
            _close_overlapping_epochs(net.code, sta)

    return Inventory(networks=list(networks_by_code.values()), source="openmeq merged inventory")


def _epoch_key(cha) -> tuple[str, str, int | None]:
    """``(location_code, code, start_date in ns)`` -- one channel epoch.

    ``UTCDateTime`` itself is not hashable, hence ``.ns``.
    """
    return (cha.location_code, cha.code, None if cha.start_date is None else cha.start_date.ns)


def _close_overlapping_epochs(network_code: str, station: Station) -> None:
    """End each channel epoch no later than the next epoch of the same channel starts.

    Epochs of one ``(location_code, code)`` are sorted by start date; an
    epoch whose ``end_date`` is missing or after the next epoch's
    ``start_date`` is ended at that start date (logged as a warning). See the
    module docstring's "Overlapping epochs are closed" note.
    """
    by_key: dict[tuple[str, str], list] = defaultdict(list)
    for cha in station.channels:
        by_key[(cha.location_code, cha.code)].append(cha)
    for (loc, code), epochs in by_key.items():
        epochs = [e for e in epochs if e.start_date is not None]
        epochs.sort(key=lambda e: e.start_date)
        for cur, nxt in zip(epochs, epochs[1:], strict=False):
            if cur.end_date is None or cur.end_date > nxt.start_date:
                logger.warning(
                    "%s.%s.%s.%s: epoch starting %s %s overlaps the next epoch "
                    "(starting %s) -- ending it at %s",
                    network_code,
                    station.code,
                    loc,
                    code,
                    cur.start_date,
                    "has no end date and"
                    if cur.end_date is None
                    else f"ends {cur.end_date}, which",
                    nxt.start_date,
                    nxt.start_date,
                )
                cur.end_date = nxt.start_date


def _read_stations_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"net": str, "sta": str, "loc": str})
    missing = [c for c in _CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ConfigError(
            f"{csv_path}: missing required column(s) {missing}: expected {_CSV_COLUMNS}"
        )
    return df


def _inventory_from_csv(csv_path: str | Path) -> Inventory:
    """Build a minimal (channel-less) `Inventory` from a stations CSV.

    Each row becomes one `Station` (no `Channel` children -- the CSV format,
    PLAN.md §3, carries no channel code).
    """
    df = _read_stations_csv(csv_path)
    networks: dict[str, Network] = {}
    for row in df.itertuples(index=False):
        net = networks.setdefault(row.net, Network(code=row.net, stations=[]))
        net.stations.append(
            Station(
                code=row.sta,
                latitude=float(row.lat),
                longitude=float(row.lon),
                elevation=float(row.elev_m),
                channels=[],
            )
        )
    return Inventory(networks=list(networks.values()), source="openmeq csv station metadata")


# ---------------------------------------------------------------------------
# station_table
# ---------------------------------------------------------------------------


def _csv_depth_and_location(csv_path: str | Path) -> tuple[dict[str, float], dict[str, str]]:
    df = _read_stations_csv(csv_path)
    depth_by_key: dict[str, float] = {}
    loc_by_key: dict[str, str] = {}
    for row in df.itertuples(index=False):
        key = f"{row.net}.{row.sta}"
        depth_by_key[key] = float(row.depth_m)
        loc_by_key[key] = str(row.loc)
    return depth_by_key, loc_by_key


def _station_rows(
    inv: Inventory, depth_by_key: dict[str, float], loc_by_key: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for net in inv.networks:
        for sta in net.stations:
            key = f"{net.code}.{sta.code}"
            if key in loc_by_key:
                location = loc_by_key[key]
            elif sta.channels:
                location = sta.channels[0].location_code
            else:
                location = ""
            rows.append(
                {
                    "network": net.code,
                    "station": sta.code,
                    "location": location,
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation_m": sta.elevation,
                    "depth_m": depth_by_key.get(key, 0.0),
                }
            )
    return pd.DataFrame(rows, columns=STATION_TABLE_COLUMNS)


def station_table(stations_cfg: StationsConfig) -> pd.DataFrame:
    """One row per merged station: the flat table ARCHITECTURE.md §2 fixes.

    Built from :func:`load_inventory` (one row per ``NET.STA``) plus
    ``stations_cfg.csv``'s ``depth_m`` column (StationXML has no native
    borehole-depth field in this schema, so ``depth_m`` always comes from
    the CSV row for that ``NET.STA`` when one exists, defaulting to ``0.0``
    for a station only covered by XML metadata).

    Parameters
    ----------
    stations_cfg : openmeq.config.StationsConfig
        ``cfg.stations``.

    Returns
    -------
    pandas.DataFrame
        Exactly the columns in :data:`STATION_TABLE_COLUMNS` (``network,
        station, location, latitude, longitude, elevation_m, depth_m``), one
        row per merged station.
    """
    inv = load_inventory(stations_cfg)
    depth_by_key: dict[str, float] = {}
    loc_by_key: dict[str, str] = {}
    if stations_cfg.csv:
        depth_by_key, loc_by_key = _csv_depth_and_location(stations_cfg.csv)
    return _station_rows(inv, depth_by_key, loc_by_key)


# ---------------------------------------------------------------------------
# apply_channel_map
# ---------------------------------------------------------------------------


def apply_channel_map(stream: Stream, stations_cfg: StationsConfig) -> Stream:
    """Relabel (and, where possible, orient) each trace's channel code.

    For every trace whose ``trace.stats.channel`` is a key in
    ``stations_cfg.channel_map``: records the original code in
    ``trace.stats.openmeq_original_channel`` (ARCHITECTURE.md §6) and
    rewrites the channel code's *last character* to the mapped value
    (``"Z"``, ``"1"`` or ``"2"``); the rest of the code (band/instrument
    characters) is left alone.

    Then, for every horizontal pair mapped to unoriented ``"1"``/``"2"``
    at the same station/location/band+instrument prefix: if *both* original
    full SEED ids (``NET.STA.LOC.CHA`` with the *original* channel code) are
    keys in ``stations_cfg.orientations``, the pair is rotated to true N/E
    (dip assumed 0) via the 2x2 solve documented in the issue design notes,
    and the channel codes' last characters become ``"N"``/``"E"``. A
    horizontal with no known orientation, or whose partner is missing or
    also unoriented, is left as unoriented ``"1"``/``"2"`` (never rotate a
    single horizontal alone).

    If ``channel_map`` would map two different original channel codes at
    the same station/location to the *same* target character (e.g. an
    inconsistent/duplicated map, or a station that genuinely carries both
    codes), the first-encountered trace wins: later colliding traces are
    left unmapped (original channel, no ``openmeq_original_channel``) and a
    warning is logged -- this never raises.

    Parameters
    ----------
    stream : obspy.Stream
        Input stream; not mutated (a shallow copy, with `Trace.stats`
        deep-copied per trace, is returned).
    stations_cfg : openmeq.config.StationsConfig
        ``cfg.stations`` (``channel_map`` and ``orientations``).

    Returns
    -------
    obspy.Stream
        A new `Stream` with channel codes relabelled (and, where
        applicable, horizontal pairs rotated to N/E).
    """
    channel_map = stations_cfg.channel_map
    orientations = stations_cfg.orientations
    out = stream.copy()

    seen_targets: dict[tuple[str, str, str, str], int] = {}
    for i, tr in enumerate(out):
        original = tr.stats.channel
        if original not in channel_map:
            continue
        mapped_last = channel_map[original]
        mapped_code = original[:-1] + mapped_last
        target_key = (tr.stats.network, tr.stats.station, tr.stats.location, mapped_code)
        if target_key in seen_targets:
            logger.warning(
                "channel_map collision at %s.%s.%s: both %r and %r map to %r -- "
                "keeping the first, leaving this trace unmapped",
                tr.stats.network,
                tr.stats.station,
                tr.stats.location,
                out[seen_targets[target_key]].stats.get(
                    "openmeq_original_channel", out[seen_targets[target_key]].stats.channel
                ),
                original,
                mapped_code,
            )
            continue
        seen_targets[target_key] = i
        tr.stats.openmeq_original_channel = original
        tr.stats.channel = mapped_code

    _rotate_oriented_pairs(out, orientations)
    return out


def _rotate_oriented_pairs(stream: Stream, orientations: dict[str, float]) -> None:
    groups: dict[tuple[str, str, str, str], list] = defaultdict(list)
    for tr in stream:
        code = tr.stats.channel
        if code[-1] in ("1", "2") and "openmeq_original_channel" in tr.stats:
            key = (tr.stats.network, tr.stats.station, tr.stats.location, code[:-1])
            groups[key].append(tr)

    for (net, sta, loc, _prefix), traces in groups.items():
        by_component = {tr.stats.channel[-1]: tr for tr in traces}
        if "1" not in by_component or "2" not in by_component:
            continue  # a lone horizontal -- never rotate alone
        tr1, tr2 = by_component["1"], by_component["2"]
        seed1 = f"{net}.{sta}.{loc}.{tr1.stats.openmeq_original_channel}"
        seed2 = f"{net}.{sta}.{loc}.{tr2.stats.openmeq_original_channel}"
        if seed1 not in orientations or seed2 not in orientations:
            continue  # unknown orientation -- stays unoriented "1"/"2"
        if tr1.stats.npts != tr2.stats.npts or tr1.stats.starttime != tr2.stats.starttime:
            logger.warning(
                "cannot rotate %s/%s at %s.%s.%s: mismatched npts/starttime",
                seed1,
                seed2,
                net,
                sta,
                loc,
            )
            continue

        az1 = np.radians(orientations[seed1])
        az2 = np.radians(orientations[seed2])
        rot_matrix = np.array([[np.cos(az1), np.sin(az1)], [np.cos(az2), np.sin(az2)]])
        rot_inv = np.linalg.inv(rot_matrix)
        c1c2 = np.vstack([tr1.data.astype(np.float64), tr2.data.astype(np.float64)])
        n_data, e_data = rot_inv @ c1c2

        tr1.data = n_data
        tr1.stats.channel = tr1.stats.channel[:-1] + "N"
        tr2.data = e_data
        tr2.stats.channel = tr2.stats.channel[:-1] + "E"


# ---------------------------------------------------------------------------
# write_merged_inventory
# ---------------------------------------------------------------------------


def write_merged_inventory(
    inv: Inventory, paths: RunPaths, stations_cfg: StationsConfig | None = None
) -> None:
    """Write the merged inventory to ``inventory/stations.xml`` + ``.csv``.

    Parameters
    ----------
    inv : obspy.core.inventory.Inventory
        The merged inventory, e.g. from :func:`load_inventory`.
    paths : openmeq.paths.RunPaths
        Used for ``paths.stations_xml``/``paths.stations_csv``.
    stations_cfg : openmeq.config.StationsConfig, optional
        When given, ``inventory/stations.csv``'s ``depth_m``/``location``
        columns are computed the same way :func:`station_table` computes
        them (threading ``depth_m`` through from ``stations_cfg.csv``).
        Without it, ``depth_m`` defaults to ``0.0`` and ``location`` is
        taken from each station's first channel (or ``""``) -- callers that
        already have ``cfg.stations`` in hand should pass it for a fully
        accurate CSV; this is the recommended wiring wherever both ``inv``
        and ``cfg.stations`` are available (e.g. a future ``openmeq qc``
        step: ``write_merged_inventory(inv, paths, cfg.stations)``).
    """
    paths.stations_xml.parent.mkdir(parents=True, exist_ok=True)
    inv.write(str(paths.stations_xml), format="STATIONXML")

    depth_by_key: dict[str, float] = {}
    loc_by_key: dict[str, str] = {}
    if stations_cfg is not None and stations_cfg.csv:
        depth_by_key, loc_by_key = _csv_depth_and_location(stations_cfg.csv)
    table = _station_rows(inv, depth_by_key, loc_by_key)
    table.to_csv(paths.stations_csv, index=False)


# ---------------------------------------------------------------------------
# check_coverage
# ---------------------------------------------------------------------------


def check_coverage(seed_ids: Iterable[str], inv: Inventory) -> dict:
    """Compare data seed ids against the inventory's channel seed ids.

    Deliberately takes a plain iterable of ``"NET.STA.LOC.CHA"`` strings
    rather than an ``openmeq.index.DataIndex`` (see the module docstring's
    wiring note) -- this module never scans miniSEED or imports
    ``openmeq.index`` itself.

    Parameters
    ----------
    seed_ids : Iterable[str]
        Every ``"NET.STA.LOC.CHA"`` seed id known to have data (e.g.
        ``DataIndex.seed_ids()``).
    inv : obspy.core.inventory.Inventory
        The merged station metadata, e.g. from :func:`load_inventory`.

    Returns
    -------
    dict
        ``{"data_without_metadata": [seed_id, ...], "metadata_without_data":
        [seed_id, ...]}``, both sorted -- the exact schema written to
        ``inventory/metadata_report.json`` (ARCHITECTURE.md §3). Robust to
        an inventory with only some components present (e.g. Z but no
        horizontals) or an ``iterable`` with no overlap at all; never
        raises for a shape mismatch between the two sides.
    """
    data_ids = {str(s) for s in seed_ids}
    meta_ids = {
        f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
        for net in inv.networks
        for sta in net.stations
        for cha in sta.channels
    }
    return {
        "data_without_metadata": sorted(data_ids - meta_ids),
        "metadata_without_data": sorted(meta_ids - data_ids),
    }


def write_metadata_report(report: dict, paths: RunPaths) -> None:
    """Write a :func:`check_coverage` result to ``inventory/metadata_report.json``.

    Parameters
    ----------
    report : dict
        The dict returned by :func:`check_coverage`.
    paths : openmeq.paths.RunPaths
        Used for ``paths.metadata_report_json``.
    """
    paths.metadata_report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata_report_json.write_text(json.dumps(report, indent=2, sort_keys=True))
