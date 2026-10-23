"""Export an OpenMEQ QuakeML catalogue to CSV tables and ESRI shapefiles (OpenMEQ-0ac).

``openmeq export -c config.yaml`` turns the final NonLinLoc catalogue
(``locate/catalog_auto.xml`` by default, or any OpenMEQ QuakeML via
``--catalog``: ``review/catalog_final.xml``, ``magnitude/catalog_mag.xml``,
``relocate/catalog_dd.xml``) into GIS- and spreadsheet-friendly files::

    <out>/<stem>_events.csv      one row per event (rejected ones flagged)
    <out>/<stem>_arrivals.csv    one row per pick, joined to its arrival on
                                 the preferred origin (blank if unused; for a
                                 hypoDD origin, which has no arrivals, the
                                 NLLoc origin it was relocated from)
    <out>/<stem>_stations.csv    0_setup/stations.csv + per-station pick stats
    <out>/<stem>_fields.csv      column dictionary: CSV name, shapefile
                                 name, unit and description, per table
    <out>/<stem>_events.shp      hypocentres (points; events with no origin
                                 coordinates are CSV-only)
    <out>/<stem>_ellipses.shp    68 % horizontal error ellipses (polygons)
    <out>/<stem>_stations.shp    stations (points)

Shapefiles are WGS84 (EPSG:4326, ``.prj`` written) with UTF-8 attributes
(``.cpg``). DBF field names are limited to 10 characters, so each column has
a fixed short name; ``_fields.csv`` maps them back.

Units follow ARCHITECTURE.md §1: km for depths, distances and location
errors, seconds for times. Two conversions happen here rather than in the
catalogue: QuakeML metres (``Origin.depth``, ``depth_errors``,
``OriginUncertainty``) become km, and NonLinLoc's ``latitude_errors``/
``longitude_errors`` (1-D marginals in great-circle degrees, per the NLLoc
comment on every origin) become km. ``ConfidenceEllipsoid`` axis lengths are
already km as written by ObsPy's ``NLLOC_HYP`` reader (they equal the
``STATISTICS`` line's ``Len1``/``Len2``/``Len3``) and are passed through.

Every ``openmeq`` ``extra`` value on the event and its preferred origin
(``magnitude_*``, ``hypodd_*``, ...) is appended as its own column, so the
exporter needs no change when a later step adds one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from obspy.geodetics import degrees2kilometers

from openmeq.catalog import EXTRA_NAMESPACE, get_extra, iso_z, read_catalog
from openmeq.catalog import event_id as get_event_id
from openmeq.locate.finalize import horizontal_error_km
from openmeq.log import get_logger

if TYPE_CHECKING:
    from obspy import Catalog
    from obspy.core.event import Event, Origin

    from openmeq.paths import RunPaths

log = get_logger(__name__)

#: WGS84 geographic CRS, as ESRI WKT, for every ``.prj`` written here.
_WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
    '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)

#: Vertices per error-ellipse polygon.
_ELLIPSE_VERTICES = 72

_STATISTICS_RE = re.compile(r"STATISTICS\s+(.*)")
_NLLOC_STATUS_RE = re.compile(r'^NLLOC\s+"[^"]*"\s+"([^"]*)"\s+"([^"]*)"')


@dataclass(frozen=True)
class Column:
    """One exported column: CSV name, DBF name (<= 10 chars), DBF type, unit, meaning.

    ``kind`` is the pyshp field type: ``"C"`` text, ``"N"`` number (``decimal``
    places; 0 = integer), ``"D"`` date, ``"L"`` logical.
    """

    name: str
    shp: str
    kind: str
    unit: str
    description: str
    decimal: int = 0
    size: int = 0

    def dbf_args(self) -> tuple[str, str, int, int]:
        if self.kind == "C":
            return self.shp, "C", self.size or 80, 0
        if self.kind == "D":
            return self.shp, "D", 8, 0
        if self.kind == "L":
            return self.shp, "L", 1, 0
        return self.shp, "N", self.size or 18, self.decimal


def _c(name, shp, unit, description, size=40):
    return Column(name, shp, "C", unit, description, size=size)


def _f(name, shp, unit, description, decimal=6):
    return Column(name, shp, "N", unit, description, decimal=decimal)


def _i(name, shp, unit, description):
    return Column(name, shp, "N", unit, description, decimal=0, size=10)


EVENT_COLUMNS: list[Column] = [
    _c("event_id", "event_id", "", "OpenMEQ event id (ev<detection trigger time>)"),
    _c("status", "status", "", "accepted / rejected (never deleted, ARCHITECTURE.md §5)", 10),
    _c("reason", "reason", "", "rejection reason: <step>:<criterion> (blank if accepted)", 120),
    _c("time", "time", "UTC", "origin time, ISO-8601", 27),
    Column("date", "date", "D", "UTC", "origin date (for GIS time filters)"),
    _f("lat", "lat", "deg", "latitude, WGS84"),
    _f("lon", "lon", "deg", "longitude, WGS84"),
    _f("depth_km", "depth_km", "km", "depth below sea level (NLLoc grid datum)", 4),
    _c("depth_type", "depth_type", "", "QuakeML Origin.depth_type", 30),
    _f("err_t_s", "err_t_s", "s", "origin time uncertainty", 4),
    _f("err_lat_km", "err_lat_km", "km", "1-D marginal latitude error (NLLoc covariance)", 4),
    _f("err_lon_km", "err_lon_km", "km", "1-D marginal longitude error (NLLoc covariance)", 4),
    _f("err_h_km", "err_h_km", "km", "horizontal error used by locate filters", 4),
    _f("err_z_km", "err_z_km", "km", "depth error (1-D marginal)", 4),
    _f("err_conf", "err_conf", "%", "confidence level of the errors", 1),
    _f("ell_min_km", "ell_min_km", "km", "2-D horizontal error ellipse semi-minor axis", 4),
    _f("ell_max_km", "ell_max_km", "km", "2-D horizontal error ellipse semi-major axis", 4),
    _f("ell_az_deg", "ell_az", "deg", "azimuth of ellipse semi-major axis (clockwise from N)", 2),
    _f("ellz_maj_km", "ez_maj_km", "km", "3-D confidence ellipsoid semi-major axis", 4),
    _f("ellz_int_km", "ez_int_km", "km", "3-D confidence ellipsoid semi-intermediate axis", 4),
    _f("ellz_min_km", "ez_min_km", "km", "3-D confidence ellipsoid semi-minor axis", 4),
    _f("ellz_maj_plunge", "ez_plunge", "deg", "ellipsoid major axis plunge", 2),
    _f("ellz_maj_az", "ez_az", "deg", "ellipsoid major axis azimuth", 2),
    _f("ellz_maj_rot", "ez_rot", "deg", "ellipsoid major axis rotation", 2),
    _f("rms_s", "rms_s", "s", "RMS travel-time residual (OriginQuality.standard_error)", 4),
    _f("gap_deg", "gap_deg", "deg", "primary azimuthal gap", 2),
    _f("gap2_deg", "gap2_deg", "deg", "secondary azimuthal gap", 2),
    _i("n_phases", "n_phases", "", "phases used by the locator (OriginQuality)"),
    _i("n_p", "n_p", "", "P arrivals with non-zero weight (hypoDD: from the NLLoc origin)"),
    _i("n_s", "n_s", "", "S arrivals with non-zero weight (hypoDD: from the NLLoc origin)"),
    _i("n_sta", "n_sta", "", "stations used (OriginQuality)"),
    _i("n_picks", "n_picks", "", "picks on the event (all, including rejected)"),
    _f("dist_min_km", "dmin_km", "km", "nearest station distance", 3),
    _f("dist_med_km", "dmed_km", "km", "median station distance", 3),
    _f("dist_max_km", "dmax_km", "km", "farthest station distance", 3),
    _f("nll_x_km", "nll_x_km", "km", "NLLoc STATISTICS ExpectX (grid coordinates)", 4),
    _f("nll_y_km", "nll_y_km", "km", "NLLoc STATISTICS ExpectY (grid coordinates)", 4),
    _f("nll_z_km", "nll_z_km", "km", "NLLoc STATISTICS ExpectZ (grid coordinates)", 4),
    _f("cov_xx", "cov_xx", "km2", "NLLoc posterior covariance XX", 6),
    _f("cov_xy", "cov_xy", "km2", "NLLoc posterior covariance XY", 6),
    _f("cov_xz", "cov_xz", "km2", "NLLoc posterior covariance XZ", 6),
    _f("cov_yy", "cov_yy", "km2", "NLLoc posterior covariance YY", 6),
    _f("cov_yz", "cov_yz", "km2", "NLLoc posterior covariance YZ", 6),
    _f("cov_zz", "cov_zz", "km2", "NLLoc posterior covariance ZZ", 6),
    _c("nll_status", "nll_status", "", "NLLoc location status (LOCATED, REJECTED, ...)", 20),
    _c("nll_message", "nll_msg", "", "NLLoc status message", 120),
    _c("method", "method", "", "locator of the preferred origin (Origin.method_id)", 60),
    _c("author", "author", "", "Origin.creation_info.author (e.g. NLLoc version)", 80),
    _c("created", "created", "UTC", "Origin.creation_info.creation_time", 27),
    _i("n_origins", "n_origins", "", "origins on the event (pass A/B, hypoDD, ...)"),
    _f("mag", "mag", "", "preferred magnitude", 3),
    _c("mag_type", "mag_type", "", "preferred magnitude type", 10),
    _f("mag_err", "mag_err", "", "preferred magnitude uncertainty", 3),
    _i("mag_n_sta", "mag_n_sta", "", "stations contributing to the preferred magnitude"),
]

ARRIVAL_COLUMNS: list[Column] = [
    _c("event_id", "event_id", "", "OpenMEQ event id"),
    _c("seed_id", "seed_id", "", "NET.STA.LOC.CHA"),
    _c("network", "network", "", "network code", 8),
    _c("station", "station", "", "station code", 8),
    _c("location", "location", "", "location code", 8),
    _c("channel", "channel", "", "channel code", 8),
    _c("phase", "phase", "", "phase hint of the pick", 8),
    _c("pick_time", "pick_time", "UTC", "pick time, ISO-8601", 27),
    _f("pick_unc_s", "pick_unc_s", "s", "pick time uncertainty", 4),
    _f("probability", "prob", "", "picker probability (openmeq extra)", 3),
    _c("pick_method", "pick_meth", "", "Pick.method_id", 60),
    _c("eval_mode", "eval_mode", "", "automatic / manual", 10),
    _c("pick_status", "pick_stat", "", "Pick.evaluation_status (rejected = filtered)", 12),
    _c("pick_reason", "pick_rsn", "", "pick rejection reason: <step>:<criterion>", 120),
    Column("used", "used", "L", "", "pick has an arrival with non-zero weight"),
    _f("travel_time_s", "tt_s", "s", "pick time - origin time", 4),
    _f("residual_s", "res_s", "s", "arrival time residual (observed - predicted)", 4),
    _f("weight", "weight", "", "arrival time weight", 4),
    _f("distance_km", "dist_km", "km", "epicentral distance", 3),
    _f("azimuth_deg", "az_deg", "deg", "event-to-station azimuth", 2),
    _f("takeoff_deg", "takeoff", "deg", "takeoff angle", 2),
]

STATION_COLUMNS: list[Column] = [
    _c("network", "network", "", "network code", 8),
    _c("station", "station", "", "station code", 8),
    _c("location", "location", "", "location code", 8),
    _f("latitude", "lat", "deg", "latitude, WGS84"),
    _f("longitude", "lon", "deg", "longitude, WGS84"),
    _f("elevation_m", "elev_m", "m", "elevation above sea level", 1),
    _f("depth_m", "depth_m", "m", "sensor depth below surface", 1),
    _i("n_p_used", "n_p_used", "", "P arrivals used across exported events"),
    _i("n_s_used", "n_s_used", "", "S arrivals used across exported events"),
    _i("n_rejected", "n_rej", "", "picks rejected by the location filters"),
    _f("p_res_mean_s", "p_res_mn", "s", "mean P residual (used arrivals)", 4),
    _f("p_res_std_s", "p_res_sd", "s", "std of P residuals (used arrivals)", 4),
    _f("s_res_mean_s", "s_res_mn", "s", "mean S residual (used arrivals)", 4),
    _f("s_res_std_s", "s_res_sd", "s", "std of S residuals (used arrivals)", 4),
]


@dataclass
class ExportResult:
    """Paths of everything :func:`export_catalog` wrote."""

    files: list[Path]
    n_events: int
    n_mapped: int


def _km(value_m) -> float | None:
    return value_m / 1000.0 if value_m is not None else None


def _rejection_reason(comments) -> str:
    """The last ``rejected:<step>:<reason>`` comment, as ``<step>:<reason>``."""
    reason = ""
    for c in comments:
        text = c.text or ""
        if text.startswith("rejected:"):
            reason = text[len("rejected:") :]
    return reason


def _nlloc_comments(origin: Origin) -> tuple[dict[str, float], str | None, str | None]:
    """Parse the NLLoc ``STATISTICS`` and ``NLLOC "..." "STATUS" "msg"`` comments."""
    stats: dict[str, float] = {}
    status = message = None
    for c in origin.comments:
        text = c.text or ""
        m = _STATISTICS_RE.search(text)
        if m:
            tokens = m.group(1).split()
            for key, value in zip(tokens[::2], tokens[1::2], strict=False):
                try:
                    stats[key] = float(value)
                except ValueError:
                    pass
        m = _NLLOC_STATUS_RE.match(text)
        if m:
            status, message = m.group(1), m.group(2)
    return stats, status, message


def _openmeq_extras(obj) -> dict:
    extra = getattr(obj, "extra", None) or {}
    return {
        k: get_extra(obj, k)
        for k, v in extra.items()
        if isinstance(v, Mapping) and v.get("namespace") == EXTRA_NAMESPACE
    }


def _is_used(arrival) -> bool:
    return arrival is not None and (arrival.time_weight is None or arrival.time_weight > 0)


def _arrival_origin(event: Event) -> Origin:
    """The origin whose arrivals describe the event's phase data.

    The preferred origin, unless it has no arrivals (hypoDD's, which
    ``openmeq.relocate`` adds without any): then the most recent origin that
    has them, i.e. the NLLoc pass-B location the relocation started from.
    """
    origin = event.preferred_origin()
    if origin.arrivals:
        return origin
    for candidate in reversed(event.origins):
        if candidate.arrivals:
            return candidate
    return origin


def event_row(event: Event) -> tuple[dict, dict]:
    """One events-table row (fixed columns) and the event's openmeq extras."""
    origin = event.preferred_origin()
    row: dict = {c.name: None for c in EVENT_COLUMNS}
    row["event_id"] = get_event_id(event)
    rejected = origin is not None and origin.evaluation_status == "rejected"
    row["status"] = "rejected" if rejected else "accepted"
    row["reason"] = _rejection_reason(event.comments) if rejected else ""
    row["n_picks"] = len(event.picks)
    row["n_origins"] = len(event.origins)

    extras = _openmeq_extras(event)
    if origin is not None:
        _fill_origin(row, origin, event)
        extras.update(_openmeq_extras(origin))

    mag = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
    if mag is not None:
        row["mag"] = mag.mag
        row["mag_type"] = mag.magnitude_type
        row["mag_err"] = mag.mag_errors.uncertainty if mag.mag_errors else None
        row["mag_n_sta"] = mag.station_count
    return row, extras


def _fill_origin(row: dict, origin: Origin, event: Event) -> None:
    if origin.time is not None:
        row["time"] = iso_z(origin.time)
        row["date"] = origin.time.datetime.date()
    row["lat"] = origin.latitude
    row["lon"] = origin.longitude
    row["depth_km"] = _km(origin.depth)
    row["depth_type"] = origin.depth_type
    if origin.time_errors:
        row["err_t_s"] = origin.time_errors.uncertainty
    if origin.latitude_errors and origin.latitude_errors.uncertainty is not None:
        row["err_lat_km"] = degrees2kilometers(origin.latitude_errors.uncertainty)
    if origin.longitude_errors and origin.longitude_errors.uncertainty is not None:
        row["err_lon_km"] = degrees2kilometers(origin.longitude_errors.uncertainty)
    if origin.depth_errors:
        row["err_z_km"] = _km(origin.depth_errors.uncertainty)
        row["err_conf"] = origin.depth_errors.confidence_level

    ou = origin.origin_uncertainty
    row["err_h_km"] = horizontal_error_km(ou)
    if ou is not None:
        row["ell_min_km"] = _km(ou.min_horizontal_uncertainty)
        row["ell_max_km"] = _km(ou.max_horizontal_uncertainty)
        row["ell_az_deg"] = ou.azimuth_max_horizontal_uncertainty
        if ou.confidence_level is not None:
            row["err_conf"] = ou.confidence_level
        ce = ou.confidence_ellipsoid
        if ce is not None:
            row["ellz_maj_km"] = ce.semi_major_axis_length
            row["ellz_int_km"] = ce.semi_intermediate_axis_length
            row["ellz_min_km"] = ce.semi_minor_axis_length
            row["ellz_maj_plunge"] = ce.major_axis_plunge
            row["ellz_maj_az"] = ce.major_axis_azimuth
            row["ellz_maj_rot"] = ce.major_axis_rotation

    q = origin.quality
    if q is not None:
        row["rms_s"] = q.standard_error
        row["gap_deg"] = q.azimuthal_gap
        row["gap2_deg"] = q.secondary_azimuthal_gap
        row["n_phases"] = q.used_phase_count
        row["n_sta"] = q.used_station_count if (q.used_station_count or 0) >= 0 else None
        # Already km on finalized origins (openmeq.locate.finalize decision #5).
        row["dist_min_km"] = q.minimum_distance
        row["dist_med_km"] = q.median_distance
        row["dist_max_km"] = q.maximum_distance

    picks = {str(p.resource_id): p for p in event.picks}
    n_p = n_s = 0
    for a in _arrival_origin(event).arrivals:
        if not _is_used(a):
            continue
        pick = picks.get(str(a.pick_id))
        phase = (a.phase or (pick.phase_hint if pick else "") or "").upper()
        if phase.startswith("P"):
            n_p += 1
        elif phase.startswith("S"):
            n_s += 1
    row["n_p"], row["n_s"] = n_p, n_s

    stats, status, message = _nlloc_comments(origin)
    for col, key in (
        ("nll_x_km", "ExpectX"),
        ("nll_y_km", "Y"),
        ("nll_z_km", "Z"),
        ("cov_xx", "CovXX"),
        ("cov_xy", "XY"),
        ("cov_xz", "XZ"),
        ("cov_yy", "YY"),
        ("cov_yz", "YZ"),
        ("cov_zz", "ZZ"),
    ):
        row[col] = stats.get(key)
    row["nll_status"], row["nll_message"] = status, message
    row["method"] = str(origin.method_id) if origin.method_id else None
    ci = origin.creation_info
    if ci is not None:
        row["author"] = ci.author
        row["created"] = iso_z(ci.creation_time) if ci.creation_time else None


def arrival_rows(event: Event) -> list[dict]:
    """One row per pick on ``event``, joined to its arrival (:func:`_arrival_origin`)."""
    eid = get_event_id(event)
    origin = event.preferred_origin()
    arrivals = {}
    if origin is not None:
        arrivals = {str(a.pick_id): a for a in _arrival_origin(event).arrivals}
    rows = []
    for pick in sorted(event.picks, key=lambda p: (p.waveform_id.get_seed_string(), p.time)):
        wid = pick.waveform_id
        a = arrivals.get(str(pick.resource_id))
        tt = None
        if origin is not None and origin.time is not None and pick.time is not None:
            tt = pick.time - origin.time
        rows.append(
            {
                "event_id": eid,
                "seed_id": wid.get_seed_string(),
                "network": wid.network_code,
                "station": wid.station_code,
                "location": wid.location_code or "",
                "channel": wid.channel_code,
                "phase": pick.phase_hint or (a.phase if a else None),
                "pick_time": iso_z(pick.time) if pick.time else None,
                "pick_unc_s": pick.time_errors.uncertainty if pick.time_errors else None,
                "probability": get_extra(pick, "probability"),
                "pick_method": str(pick.method_id) if pick.method_id else None,
                "eval_mode": pick.evaluation_mode,
                "pick_status": pick.evaluation_status,
                "pick_reason": _rejection_reason(pick.comments),
                "used": _is_used(a) and pick.evaluation_status != "rejected",
                "travel_time_s": tt,
                "residual_s": a.time_residual if a else None,
                "weight": a.time_weight if a else None,
                "distance_km": a.distance if a else None,
                "azimuth_deg": a.azimuth if a else None,
                "takeoff_deg": a.takeoff_angle if a else None,
            }
        )
    return rows


def station_table(stations_csv: Path, arrivals: pd.DataFrame) -> pd.DataFrame:
    """``0_setup/stations.csv`` with per-station pick/residual statistics."""
    sta = pd.read_csv(stations_csv, dtype={"location": str}, keep_default_na=False)
    sta = sta.replace({"": None}).fillna({"location": ""})
    for col in ("latitude", "longitude", "elevation_m", "depth_m"):
        if col in sta:
            sta[col] = pd.to_numeric(sta[col], errors="coerce")
    if arrivals.empty:
        stats = pd.DataFrame(columns=["network", "station"])
    else:
        a = arrivals.copy()
        a["P"] = a["phase"].fillna("").str.upper().str.startswith("P")
        a["S"] = a["phase"].fillna("").str.upper().str.startswith("S")
        used = a[a["used"]]
        grp = ["network", "station"]
        stats = pd.DataFrame(
            {
                "n_p_used": used[used["P"]].groupby(grp).size(),
                "n_s_used": used[used["S"]].groupby(grp).size(),
                "n_rejected": a[a["pick_status"] == "rejected"].groupby(grp).size(),
                "p_res_mean_s": used[used["P"]].groupby(grp)["residual_s"].mean(),
                "p_res_std_s": used[used["P"]].groupby(grp)["residual_s"].std(),
                "s_res_mean_s": used[used["S"]].groupby(grp)["residual_s"].mean(),
                "s_res_std_s": used[used["S"]].groupby(grp)["residual_s"].std(),
            }
        ).reset_index()
    # One row per (network, station): the location code only matters for the
    # station's coordinates, which are the same for every location.
    sta = sta.drop_duplicates(subset=["network", "station"])
    out = sta.merge(stats, on=["network", "station"], how="left")
    for col in ("n_p_used", "n_s_used", "n_rejected"):
        out[col] = out[col].fillna(0).astype(int)
    return out[[c.name for c in STATION_COLUMNS if c.name in out.columns]]


def _extra_columns(names: list[str], taken: set[str]) -> list[Column]:
    """Columns for the openmeq extras, with unique <= 10 char DBF names."""
    cols = []
    for name in names:
        base = re.sub(r"[^A-Za-z0-9_]", "_", name)
        for prefix in ("magnitude_", "hypodd_"):
            if base.startswith(prefix):
                base = prefix[:3] + "_" + base[len(prefix) :]
        shp, n = base[:10], 1
        while shp.lower() in taken:
            suffix = str(n)
            shp, n = base[: 10 - len(suffix)] + suffix, n + 1
        taken.add(shp.lower())
        cols.append(Column(name, shp, "C", "", "openmeq extra (event / preferred origin)", size=80))
    return cols


def _typed_extra_columns(extras: list[dict], cols: list[Column]) -> list[Column]:
    """Make an extra column numeric in the DBF when every value is numeric
    (judged on the raw values: a DataFrame column would already be float)."""
    typed = []
    for col in cols:
        values = [ex[col.name] for ex in extras if ex.get(col.name) is not None]
        numeric = len(values) and all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in values
        )
        if numeric:
            is_int = all(isinstance(v, int) for v in values)
            col = Column(
                col.name, col.shp, "N", col.unit, col.description, decimal=0 if is_int else 6
            )
        typed.append(col)
    return typed


def _nullable_ints(df: pd.DataFrame, columns: list[Column]) -> None:
    """Integer columns as pandas ``Int64`` so a missing value does not turn
    the CSV column into floats (``12.0``)."""
    for col in columns:
        if col.kind == "N" and col.decimal == 0 and col.name in df:
            df[col.name] = pd.to_numeric(df[col.name], errors="coerce").round().astype("Int64")


def _dbf_value(col: Column, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "" if col.kind == "C" else None
    if col.kind == "N":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(v):
            return None
        return int(round(v)) if col.decimal == 0 else v
    if col.kind == "L":
        return bool(value)
    if col.kind == "D":
        return value
    # DBF sizes are bytes; strip so a cut never ends in pad-like spaces.
    raw = str(value).encode("utf-8")[: col.size or 80]
    return raw.decode("utf-8", errors="ignore").rstrip()


def _sidecars(base: Path) -> list[Path]:
    prj, cpg = base.with_suffix(".prj"), base.with_suffix(".cpg")
    prj.write_text(_WGS84_PRJ)
    cpg.write_text("UTF-8")
    return [prj, cpg]


def _write_shapefile(base: Path, df: pd.DataFrame, columns: list[Column], geometry) -> list[Path]:
    """Write ``base``.shp/.shx/.dbf/.prj/.cpg; ``geometry(row)`` adds the shape
    to the writer and returns True, or returns False to skip the row."""
    import shapefile  # pyshp

    with shapefile.Writer(str(base), encoding="utf-8") as w:
        w.autoBalance = False
        for col in columns:
            w.field(*col.dbf_args())
        for rec in df.to_dict("records"):
            if geometry(w, rec):
                w.record(*[_dbf_value(c, rec.get(c.name)) for c in columns])
    return [base.with_suffix(s) for s in (".shp", ".shx", ".dbf")] + _sidecars(base)


def _ellipse_ring(lat, lon, a_km, b_km, az_deg) -> list[tuple[float, float]]:
    """Clockwise (shapefile outer-ring order) lon/lat ring of an ellipse with
    semi-major ``a_km`` along azimuth ``az_deg`` and semi-minor ``b_km``."""
    az = math.radians(az_deg)
    km_per_deg = degrees2kilometers(1.0)
    coslat = max(math.cos(math.radians(lat)), 1e-6)
    ring = []
    for k in range(_ELLIPSE_VERTICES + 1):
        t = 2 * math.pi * k / _ELLIPSE_VERTICES
        # Parametric ellipse in its own frame, then rotated by the azimuth;
        # increasing t with (sin, cos) ordering runs clockwise from north.
        u, v = a_km * math.cos(t), b_km * math.sin(t)
        north = u * math.cos(az) - v * math.sin(az)
        east = u * math.sin(az) + v * math.cos(az)
        ring.append((lon + east / (km_per_deg * coslat), lat + north / km_per_deg))
    return ring


def _point(w, rec) -> bool:
    lat, lon = rec.get("lat", rec.get("latitude")), rec.get("lon", rec.get("longitude"))
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return False
    w.point(float(lon), float(lat))
    return True


def _ellipse(w, rec) -> bool:
    vals = [rec.get(k) for k in ("lat", "lon", "ell_max_km", "ell_min_km", "ell_az_deg")]
    if any(v is None or pd.isna(v) for v in vals):
        return False
    lat, lon, a, b, az = (float(v) for v in vals)
    w.poly([_ellipse_ring(lat, lon, a, b, az)])
    return True


def _fields_table(tables: dict[str, list[Column]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": table,
                "column": c.name,
                "shp_field": c.shp,
                "unit": c.unit,
                "description": c.description,
            }
            for table, cols in tables.items()
            for c in cols
        ]
    )


def export_catalog(
    catalog: Catalog | str | Path,
    out_dir: str | Path,
    *,
    stem: str = "catalog",
    stations_csv: str | Path | None = None,
    accepted_only: bool = False,
    shapefiles: bool = True,
) -> ExportResult:
    """Write the CSV tables (and, by default, shapefiles) for ``catalog``.

    Parameters
    ----------
    catalog : obspy.Catalog or path
        An OpenMEQ catalogue (``smi:openmeq/event/<id>`` resource ids).
    out_dir : path
        Created if missing.
    stem : str
        File-name prefix, e.g. ``"catalog_auto"``.
    stations_csv : path, optional
        ``0_setup/stations.csv``; the station table/shapefile is skipped
        without it.
    accepted_only : bool
        Drop rejected events (and their picks) from every output.
    shapefiles : bool
        Also write the ``.shp`` layers.
    """
    if not isinstance(catalog, (str, Path)):
        cat = catalog
    else:
        cat = read_catalog(catalog)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, extras, picks = [], [], []
    for ev in cat:
        row, ex = event_row(ev)
        if accepted_only and row["status"] == "rejected":
            continue
        rows.append(row)
        extras.append(ex)
        picks.extend(arrival_rows(ev))

    extra_names = sorted({k for ex in extras for k in ex})
    extra_cols = _extra_columns(extra_names, {c.shp.lower() for c in EVENT_COLUMNS})
    for row, ex in zip(rows, extras, strict=True):
        for name in extra_names:
            row[name] = ex.get(name)
    events = pd.DataFrame(rows, columns=[c.name for c in EVENT_COLUMNS] + extra_names)
    extra_cols = _typed_extra_columns(extras, extra_cols)
    event_cols = EVENT_COLUMNS + extra_cols
    _nullable_ints(events, event_cols)
    arrivals = pd.DataFrame(picks, columns=[c.name for c in ARRIVAL_COLUMNS])

    files: list[Path] = []
    path = out_dir / f"{stem}_events.csv"
    events.drop(columns=["date"]).to_csv(path, index=False)
    files.append(path)
    path = out_dir / f"{stem}_arrivals.csv"
    arrivals.to_csv(path, index=False)
    files.append(path)

    tables = {"events": [c for c in event_cols if c.name != "date"], "arrivals": ARRIVAL_COLUMNS}

    stations = None
    if stations_csv is not None and Path(stations_csv).exists():
        stations = station_table(Path(stations_csv), arrivals)
        path = out_dir / f"{stem}_stations.csv"
        stations.to_csv(path, index=False)
        files.append(path)
        tables["stations"] = [c for c in STATION_COLUMNS if c.name in stations.columns]
    elif stations_csv is not None:
        log.warning("export: %s not found; skipping the station table", stations_csv)

    path = out_dir / f"{stem}_fields.csv"
    shp_tables = dict(tables)
    shp_tables["events"] = event_cols
    _fields_table(shp_tables).to_csv(path, index=False)
    files.append(path)

    n_mapped = int(events["lat"].notna().sum()) if not events.empty else 0
    if shapefiles:
        files += _write_shapefile(out_dir / f"{stem}_events", events, event_cols, _point)
        files += _write_shapefile(out_dir / f"{stem}_ellipses", events, event_cols, _ellipse)
        if stations is not None:
            files += _write_shapefile(
                out_dir / f"{stem}_stations", stations, tables["stations"], _point
            )

    log.info(
        "export: %d events (%d with coordinates), %d picks -> %s",
        len(events),
        n_mapped,
        len(arrivals),
        out_dir,
    )
    return ExportResult(files=files, n_events=len(events), n_mapped=n_mapped)


def run_export(
    paths: RunPaths,
    catalog: str | Path | None = None,
    out_dir: str | Path | None = None,
    accepted_only: bool = False,
    shapefiles: bool = True,
) -> ExportResult:
    """``openmeq export``: defaults to ``locate/catalog_auto.xml`` ->
    ``locate/export/``; a ``--catalog`` elsewhere exports next to itself, into
    ``<catalog dir>/export/``."""
    catalog = Path(catalog) if catalog is not None else paths.catalog_auto_xml
    if not catalog.exists():
        raise FileNotFoundError(f"catalogue not found: {catalog} (run `openmeq locate` first)")
    out_dir = Path(out_dir) if out_dir is not None else catalog.parent / "export"
    return export_catalog(
        catalog,
        out_dir,
        stem=catalog.stem,
        stations_csv=paths.stations_csv,
        accepted_only=accepted_only,
        shapefiles=shapefiles,
    )
