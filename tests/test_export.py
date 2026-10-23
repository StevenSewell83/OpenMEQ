"""Tests for ``openmeq.export`` / ``openmeq export`` (OpenMEQ-0ac)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import shapefile
from click.testing import CliRunner
from obspy import Catalog, UTCDateTime
from obspy.core.event import (
    Arrival,
    Comment,
    ConfidenceEllipsoid,
    CreationInfo,
    Magnitude,
    Origin,
    OriginQuality,
    OriginUncertainty,
    Pick,
    QuantityError,
    WaveformStreamID,
)

from openmeq.catalog import make_event, reject_event, set_extra, write_catalog
from openmeq.export import _ellipse_ring, export_catalog

T0 = UTCDateTime(2018, 3, 10, 13, 9, 18.25)

_STATS = (
    "Note: ...\nSTATISTICS ExpectX 1.712603 Y 0.850694 Z 2.537700  CovXX 0.085065 "
    "XY -0.0140784 XZ 0.0773724 YY 0.126443 YZ -0.056236 ZZ 0.286899 EllAz1  95.2343 "
    "Dip1  -19.782 Len1  0.454078 Az2  10.1174 Dip2  13.3153 Len2  0.624301 Len3  1.078628e+00"
)


def _pick(sta, phase, dt, cha="EHZ", status=None, prob=0.9):
    p = Pick(
        time=T0 + dt,
        time_errors=QuantityError(uncertainty=0.05),
        waveform_id=WaveformStreamID("MB", sta, "", cha),
        phase_hint=phase,
        evaluation_mode="automatic",
        evaluation_status=status,
    )
    set_extra(p, "probability", prob)
    if status == "rejected":
        p.comments.append(Comment(text="rejected:locate/passB:residual>0.3"))
    return p


def _nlloc_origin(picks, weights):
    o = Origin(
        time=T0,
        latitude=14.1,
        longitude=121.2,
        latitude_errors=QuantityError(uncertainty=0.0032),
        longitude_errors=QuantityError(uncertainty=0.0026),
        depth=2228.0,
        depth_errors=QuantityError(uncertainty=535.6, confidence_level=68.0),
        depth_type="from location",
        method_id="smi:openmeq/nlloc/passB",
        quality=OriginQuality(
            standard_error=0.04,
            azimuthal_gap=94.9,
            secondary_azimuthal_gap=132.2,
            used_phase_count=3,
            used_station_count=2,
            minimum_distance=1.3,
            median_distance=2.7,
            maximum_distance=6.4,
        ),
        origin_uncertainty=OriginUncertainty(
            min_horizontal_uncertainty=430.9,
            max_horizontal_uncertainty=548.4,
            azimuth_max_horizontal_uncertainty=162.9,
            confidence_ellipsoid=ConfidenceEllipsoid(
                semi_major_axis_length=1.08,
                semi_minor_axis_length=0.45,
                semi_intermediate_axis_length=0.62,
                major_axis_plunge=65.8,
                major_axis_azimuth=132.0,
                major_axis_rotation=114.3,
            ),
            confidence_level=68.0,
            preferred_description="confidence ellipsoid",
        ),
        creation_info=CreationInfo(author="NLLoc:v7.1.05", creation_time=UTCDateTime(2026, 1, 1)),
        comments=[
            Comment(text=_STATS),
            Comment(text='NLLOC "out.grid0" "LOCATED" "Location completed."'),
        ],
    )
    for p, w in zip(picks, weights, strict=True):
        o.arrivals.append(
            Arrival(
                pick_id=p.resource_id,
                phase=p.phase_hint,
                time_residual=0.01,
                time_weight=w,
                distance=2.0,
                azimuth=74.0,
                takeoff_angle=120.0,
            )
        )
    return o


@pytest.fixture
def catalog():
    ev1 = make_event("ev20180310T130917_295")
    ev1.picks = [
        _pick("MB02", "P", 1.0),
        _pick("MB02", "S", 1.8, cha="EHN"),
        _pick("MB03", "P", 1.2),
        _pick("MB03", "S", 2.0, cha="EHN", status="rejected"),
    ]
    o = _nlloc_origin(ev1.picks, [1.0, 1.0, 1.0, 0.0])
    ev1.origins.append(o)
    ev1.preferred_origin_id = o.resource_id
    mag = Magnitude(mag=1.23, magnitude_type="ML", station_count=2, origin_id=o.resource_id)
    ev1.magnitudes.append(mag)
    ev1.preferred_magnitude_id = mag.resource_id
    set_extra(ev1, "magnitude_n_stations_used", 2)
    set_extra(ev1, "magnitude_skip_reason", "none")

    ev2 = make_event("ev20180310T130839_810")
    ev2.picks = [_pick("MB02", "P", 0.5)]
    placeholder = Origin(method_id="smi:openmeq/nlloc/passA")
    ev2.origins.append(placeholder)
    ev2.preferred_origin_id = placeholder.resource_id
    reject_event(ev2, reason="nlloc_failed:2 pick(s)", step="locate")
    return Catalog(events=[ev1, ev2])


@pytest.fixture
def stations_csv(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text(
        "network,station,location,latitude,longitude,elevation_m,depth_m\n"
        "MB,MB02,,14.12,121.25,138.8,0.0\n"
        "MB,MB03,,14.10,121.23,275.4,0.0\n"
        "MB,MB09,,14.11,121.21,300.0,0.0\n"
    )
    return path


def test_events_table(tmp_path, catalog, stations_csv):
    res = export_catalog(catalog, tmp_path / "out", stem="cat", stations_csv=stations_csv)
    assert res.n_events == 2 and res.n_mapped == 1
    ev = pd.read_csv(tmp_path / "out" / "cat_events.csv", keep_default_na=False)
    acc = ev[ev.event_id == "ev20180310T130917_295"].iloc[0]
    assert acc.status == "accepted" and acc.reason == ""
    assert acc.time == "2018-03-10T13:09:18.250000Z"
    assert float(acc.depth_km) == pytest.approx(2.228)
    assert float(acc.err_z_km) == pytest.approx(0.5356)
    assert float(acc.err_lat_km) == pytest.approx(0.0032 * 111.19, rel=1e-3)
    assert float(acc.err_h_km) == pytest.approx(0.5484)  # falls back to the ellipse
    assert float(acc.ell_az_deg) == pytest.approx(162.9)
    assert (int(acc.n_p), int(acc.n_s), int(acc.n_picks)) == (2, 1, 4)
    assert float(acc.cov_zz) == pytest.approx(0.286899)
    assert float(acc.nll_x_km) == pytest.approx(1.712603)
    assert acc.nll_status == "LOCATED"
    assert float(acc.mag) == pytest.approx(1.23) and acc.mag_type == "ML"
    assert int(acc.magnitude_n_stations_used) == 2
    assert acc.n_phases == "3"  # nullable int, not "3.0"

    rej = ev[ev.event_id == "ev20180310T130839_810"].iloc[0]
    assert rej.status == "rejected"
    assert rej.reason == "locate:nlloc_failed:2 pick(s)"
    assert rej.lat == ""


def test_accepted_only(tmp_path, catalog):
    res = export_catalog(catalog, tmp_path, accepted_only=True, shapefiles=False)
    assert res.n_events == 1
    arr = pd.read_csv(tmp_path / "catalog_arrivals.csv")
    assert set(arr.event_id) == {"ev20180310T130917_295"}
    assert not (tmp_path / "catalog_events.shp").exists()


def test_arrivals_and_stations(tmp_path, catalog, stations_csv):
    export_catalog(catalog, tmp_path, stations_csv=stations_csv, shapefiles=False)
    arr = pd.read_csv(tmp_path / "catalog_arrivals.csv")
    ev1 = arr[arr.event_id == "ev20180310T130917_295"]
    assert len(ev1) == 4
    rej = ev1[ev1.pick_status == "rejected"].iloc[0]
    assert not rej.used and rej.weight == 0
    assert rej.pick_reason == "locate/passB:residual>0.3"
    p = ev1[(ev1.station == "MB02") & (ev1.phase == "P")].iloc[0]
    assert p.used and p.travel_time_s == pytest.approx(1.0) and p.probability == 0.9

    sta = pd.read_csv(tmp_path / "catalog_stations.csv").set_index("station")
    assert sta.loc["MB03", "n_p_used"] == 1 and sta.loc["MB03", "n_rejected"] == 1
    assert sta.loc["MB09", "n_p_used"] == 0


def test_hypodd_origin_uses_nlloc_arrivals(tmp_path, catalog):
    ev = catalog[0]
    dd = Origin(
        time=T0,
        latitude=14.1001,
        longitude=121.2001,
        depth=2200.0,
        method_id="smi:openmeq/hypodd",
        origin_uncertainty=OriginUncertainty(horizontal_uncertainty=3.1),
    )
    set_extra(dd, "hypodd_cluster_id", 1)
    set_extra(dd, "hypodd_rct", 0.066)
    ev.origins.append(dd)
    ev.preferred_origin_id = dd.resource_id
    export_catalog(catalog, tmp_path, shapefiles=True)
    row = pd.read_csv(tmp_path / "catalog_events.csv").iloc[0]
    assert row.method == "smi:openmeq/hypodd"
    assert (row.n_p, row.n_s) == (2, 1)
    assert row.err_h_km == pytest.approx(0.0031)
    assert row.hypodd_cluster_id == 1
    fields = {f[0]: f for f in shapefile.Reader(str(tmp_path / "catalog_events")).fields[1:]}
    assert fields["hyp_cluste"][3] == 0  # integer extra -> integer DBF field
    assert fields["hyp_rct"][3] == 6
    arr = pd.read_csv(tmp_path / "catalog_arrivals.csv")
    assert arr[arr.event_id == "ev20180310T130917_295"].used.sum() == 3


def test_shapefiles(tmp_path, catalog, stations_csv):
    export_catalog(catalog, tmp_path, stations_csv=stations_csv)
    for layer in ("events", "ellipses", "stations"):
        assert (tmp_path / f"catalog_{layer}.prj").read_text().startswith('GEOGCS["GCS_WGS_1984"')
        assert (tmp_path / f"catalog_{layer}.cpg").read_text() == "UTF-8"

    events = shapefile.Reader(str(tmp_path / "catalog_events"))
    assert events.shapeType == shapefile.POINT
    assert len(events) == 1  # the rejected placeholder origin has no coordinates
    assert events.shape(0).points[0] == pytest.approx((121.2, 14.1))
    rec = events.record(0).as_dict()
    assert rec["event_id"] == "ev20180310T130917_295" and rec["n_p"] == 2
    assert rec["ell_az"] == pytest.approx(162.9)
    assert all(len(f[0]) <= 10 for f in events.fields[1:])

    ellipses = shapefile.Reader(str(tmp_path / "catalog_ellipses"))
    assert ellipses.shapeType == shapefile.POLYGON and len(ellipses) == 1

    stations = shapefile.Reader(str(tmp_path / "catalog_stations"))
    assert len(stations) == 3

    fields = pd.read_csv(tmp_path / "catalog_fields.csv")
    assert {"events", "arrivals", "stations"} == set(fields.table)
    assert fields.shp_field.str.len().max() <= 10


def test_ellipse_ring_geometry():
    ring = _ellipse_ring(0.0, 0.0, a_km=2.0, b_km=1.0, az_deg=90.0)
    assert ring[0] == pytest.approx(ring[-1])
    km = 111.19492664455873
    # Semi-major axis points east for azimuth 90.
    assert ring[0][0] * km == pytest.approx(2.0) and ring[0][1] == pytest.approx(0.0, abs=1e-12)
    # Shoelace: clockwise rings (shapefile outer ring) have negative signed area.
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:], strict=False)) / 2
    assert area < 0
    assert abs(area) * km**2 == pytest.approx(math.pi * 2.0 * 1.0, rel=1e-2)


def test_cli_export(tmp_path, catalog, monkeypatch):
    from openmeq import cli
    from openmeq.paths import RunPaths

    cat_path = tmp_path / "run" / "7_location" / "catalog_auto.xml"
    write_catalog(catalog, cat_path)

    class _Cfg:
        pass

    monkeypatch.setattr(cli, "load_config", lambda path: _Cfg())
    monkeypatch.setattr(RunPaths, "__init__", lambda self, cfg: None)
    monkeypatch.setattr(RunPaths, "catalog_auto_xml", property(lambda self: cat_path))
    monkeypatch.setattr(
        RunPaths, "stations_csv", property(lambda self: tmp_path / "run" / "missing.csv")
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("project: {}\n")

    result = CliRunner().invoke(cli.main, ["export", "-c", str(cfg_file)])
    assert result.exit_code == 0, result.output
    out = cat_path.parent / "export"
    assert (out / "catalog_auto_events.csv").exists()
    assert (out / "catalog_auto_events.shp").exists()
    assert not (out / "catalog_auto_stations.csv").exists()
    assert "exported 2 events (1 with coordinates)" in result.output
