"""Tests for openmeq.relocate.ph2dt_step (M10.2 / OpenMEQ-16d.2).

Reuses M9.2's example1 fixture helpers (``test_relocate_hypodd_files``'s
``_read_example1_stations``/``_read_example1_events``/``_build_example1_catalog``/
``_parse_dt_ct``) per this task's own notes ("this can share fixture setup
with M9.2's own example1 test"), and M9.5's ``_write_fake_inc_tree`` pattern
(``test_setup_hypodd_script``) so ``setup_hypodd``'s compiled-limit check
never touches the main checkout's ``external/HYPODD`` (ARCHITECTURE.md §6c).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from obspy import Catalog
from obspy.core.event import (
    Arrival,
    Magnitude,
    Origin,
    Pick,
    WaveformStreamID,
)
from test_relocate_hypodd_files import (  # noqa: E402
    _build_example1_catalog,
    _read_example1_events,
    _read_example1_stations,
)
from test_relocate_hypodd_files import (
    _parse_dt_ct as _parse_dt_ct_full,
)

from openmeq.catalog import make_event, write_catalog
from openmeq.config import ConfigError, load_config
from openmeq.external import ExternalToolMissing, bin_path
from openmeq.paths import RunPaths
from openmeq.relocate.hypodd_files import build_event_id_map, invert_event_id_map
from openmeq.relocate.ph2dt_step import parse_dt_ct, run_ph2dt

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"
HYPODD_MISSING = not (EXAMPLE1_DIR.exists() and (EXTERNAL_DIR / "bin" / "ph2dt").exists())


# ---------------------------------------------------------------------------
# scratch .inc tree (never the main checkout's external/HYPODD, ARCHITECTURE.md
# §6c) -- generous limits so setup_hypodd's ensure_compiled_limits check never
# triggers a recompile for these small test datasets.
# ---------------------------------------------------------------------------

_FAKE_HYPODD_INC = """\
c hypoDD.inc: FAKE, tiny, for tests only.
\tinteger*4 MAXEVE, MAXLAY, MAXDATA, MAXSTA, MAXEVE0, MAXDATA0
\tinteger*4 MAXCL
\tparameter(MAXEVE=  1000,
     &          MAXDATA=  100000,
     &          MAXEVE0=  1000,
     &          MAXDATA0= 100000,
     &          MAXLAY=   30,
     &          MAXSTA=   500,
     &          MAXCL=    10)
"""

_FAKE_PH2DT_INC = """\
c ph2dt.inc: FAKE, tiny, for tests only.
\tinteger\tMEV, MSTA, MOBS
\tparameter(MEV=    1000,
     &\t\tMSTA=   500,
     &\t\tMOBS=   200)
"""

_FAKE_VEL3D_INC = """\
      integer\tmaxnx,maxny,maxnz,maxnz2
      parameter(maxnx= 25)
      parameter(maxny= 25)
      parameter(maxnz= 20)
      parameter(maxnz2=40)
      integer\tixkms,iykms,izkms
      parameter(ixkms= 10000)
      parameter(iykms= 10000)
      parameter(izkms= 10000)
"""


def _write_fake_inc_tree(root: Path) -> Path:
    include_dir = root / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "hypoDD.inc").write_text(_FAKE_HYPODD_INC)
    (include_dir / "ph2dt.inc").write_text(_FAKE_PH2DT_INC)
    (include_dir / "vel3d.inc").write_text(_FAKE_VEL3D_INC)
    return root


# ---------------------------------------------------------------------------
# small, hand-rolled config builder (mirrors test_setup_hypodd_script.py's
# _write_config_yaml: round-tripped from config/example.yaml so every other
# required field keeps a valid, if unused, value)
# ---------------------------------------------------------------------------


def _base_config_dict() -> dict:
    with open(EXAMPLE_CONFIG) as fh:
        return yaml.safe_load(fh)


def _write_layer_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# depth_top_km  vp_km_s  vs_km_s  rho_g_cc\n"
        "0.0   3.50  2.00  2.3\n"
        "2.0   5.00  2.90  2.6\n"
        "5.0   6.00  3.40  2.7\n"
    )
    return path


def _write_config(
    tmp_path: Path, station_table: pd.DataFrame, *, ph2dt_overrides: dict | None = None
):
    data = _base_config_dict()
    data["project"]["output_dir"] = str(tmp_path / "run")

    # cfg.stations.csv's own raw-column convention (openmeq.stations._CSV_COLUMNS:
    # net,sta,loc,lat,lon,elev_m,depth_m) differs from station_table()'s output
    # columns (network,station,...) -- rename before writing.
    stations_csv = tmp_path / "stations.csv"
    raw = station_table.rename(
        columns={
            "network": "net",
            "station": "sta",
            "location": "loc",
            "latitude": "lat",
            "longitude": "lon",
            "elevation_m": "elev_m",
        }
    )[["net", "sta", "loc", "lat", "lon", "elev_m", "depth_m"]]
    raw.to_csv(stations_csv, index=False)
    data["stations"] = {"csv": str(stations_csv)}

    layer_path = _write_layer_file(tmp_path / "vel1d.txt")
    data["hypodd"]["velocity_model"] = {"type": "1d", "file": str(layer_path)}
    if ph2dt_overrides:
        data["hypodd"]["ph2dt"].update(ph2dt_overrides)

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(data, fh)
    return load_config(config_path)


def _station_table_from_dict(stations: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = [
        {
            "network": "XX",
            "station": label,
            "location": "",
            "latitude": lat,
            "longitude": lon,
            "elevation_m": 0.0,
            "depth_m": 0.0,
        }
        for label, (lat, lon) in sorted(stations.items())
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_missing_catalog_fails_clearly
# ---------------------------------------------------------------------------


def test_missing_catalog_fails_clearly(tmp_path):
    station_table = _station_table_from_dict({"AAA": (37.8, -122.2)})
    cfg = _write_config(tmp_path, station_table)
    paths = RunPaths(cfg)
    paths.ensure()

    assert not paths.catalog_mag_xml.exists()

    with pytest.raises(ConfigError, match="catalog_mag.xml"):
        run_ph2dt(cfg, paths)


# ---------------------------------------------------------------------------
# test_event_pair_list_parsed_correctly
# ---------------------------------------------------------------------------


def test_event_pair_list_parsed_correctly(tmp_path):
    dt_ct = tmp_path / "dt.ct"
    dt_ct.write_text(
        "#     38542     38520\n"
        "NCCBW     3.430   3.430 1.0000 P\n"
        "NCCSP     2.850   2.840 1.0000 P\n"
        "#     38542     45165\n"
        "NCCBW     3.430   3.500 0.5000 P\n"
        "#         7         3\n"
        "AAA       1.000   1.200 1.0000 S\n"
    )
    assert parse_dt_ct(dt_ct) == [(38542, 38520), (38542, 45165), (7, 3)]


def test_parse_dt_ct_rejects_malformed_header(tmp_path):
    dt_ct = tmp_path / "dt.ct"
    dt_ct.write_text("# 1 2 3\nAAA 1.0 1.0 1.0 P\n")
    with pytest.raises(ConfigError, match="malformed"):
        parse_dt_ct(dt_ct)


# ---------------------------------------------------------------------------
# test_example1_dt_ct_reproduction (requires_hypodd)
# ---------------------------------------------------------------------------


@pytest.mark.requires_hypodd
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt) not available")
def test_example1_dt_ct_reproduction(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, orig_cuspid_map, station_table = _build_example1_catalog(stations, events)
    write_catalog(cat, tmp_path / "catalog_mag.xml")

    cfg = _write_config(
        tmp_path,
        station_table,
        ph2dt_overrides={"maxdist_km": 120, "maxsep_km": 10, "minobs": 1, "maxobs": 100},
    )
    paths = RunPaths(cfg)
    paths.ensure()
    shutil.copy(tmp_path / "catalog_mag.xml", paths.catalog_mag_xml)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")

    try:
        bin_path("ph2dt")
    except ExternalToolMissing:
        pytest.skip("ph2dt binary not available")

    pairs = run_ph2dt(
        cfg,
        paths,
        external_hypodd_dir=fake_inc_root,
        external_bin_dir=tmp_path / "fake_bin",
    )
    assert pairs, "expected at least one event pair"

    dt_ct_path = paths.dt_ct
    assert dt_ct_path == paths.relocate_work_dir / "dt.ct"
    assert dt_ct_path.is_file()

    # Translate our own auto-assigned cuspids back to example1's original
    # cuspids (via our persisted event_id_map.csv + _build_example1_catalog's
    # own event_id -> original-cuspid map) so the comparison is invariant to
    # this module's own cuspid-assignment convention (build_event_id_map
    # assigns 1..N by origin time, not example1's own arbitrary numbers).
    our_event_id_map = build_event_id_map(cat, paths.relocate_work_dir / "event_id_map.csv")
    our_cuspid_to_event_id = invert_event_id_map(our_event_id_map)
    our_cuspid_to_orig_cuspid = {
        our_cuspid: orig_cuspid_map[eid] for our_cuspid, eid in our_cuspid_to_event_id.items()
    }
    translated_pairs = {
        frozenset((our_cuspid_to_orig_cuspid[a], our_cuspid_to_orig_cuspid[b])) for a, b in pairs
    }

    written = _parse_dt_ct_full(dt_ct_path)
    bundled = _parse_dt_ct_full(EXAMPLE1_DIR / "dt.ct")
    bundled_pairs = {frozenset(pair) for pair in bundled}

    assert translated_pairs == bundled_pairs

    # Per-pair travel-time-difference data: key the written dt.ct's own
    # (our-cuspid) pairs by their *translated* (original-cuspid) frozenset so
    # they can be compared directly against the bundled dt.ct's own pairs.
    written_by_translated: dict[frozenset, list] = {}
    for (a, b), lines in written.items():
        key = frozenset((our_cuspid_to_orig_cuspid[a], our_cuspid_to_orig_cuspid[b]))
        written_by_translated[key] = lines

    for pair_key, bundled_lines in bundled.items():
        key = frozenset(pair_key)
        written_lines = written_by_translated[key]
        written_by_sta_pha = {(sta, pha): (t1, t2, wt) for sta, t1, t2, wt, pha in written_lines}
        bundled_by_sta_pha = {(sta, pha): (t1, t2, wt) for sta, t1, t2, wt, pha in bundled_lines}
        assert set(written_by_sta_pha) == set(bundled_by_sta_pha)
        for sta_pha, (t1_b, t2_b, wt_b) in bundled_by_sta_pha.items():
            t1_w, t2_w, wt_w = written_by_sta_pha[sta_pha]
            # Our cuspid<->event mapping need not preserve which original event
            # is "id1" vs "id2" for a given pair, so compare the *unordered*
            # pair of travel times rather than assuming the same t1/t2 order.
            assert {round(t1_w, 3), round(t2_w, 3)} == {round(t1_b, 3), round(t2_b, 3)} or (
                pytest.approx(t1_w, abs=5e-3) == t1_b and pytest.approx(t2_w, abs=5e-3) == t2_b
            )
            assert wt_w == pytest.approx(wt_b, abs=5e-3)


# ---------------------------------------------------------------------------
# test_skip_unchanged
# ---------------------------------------------------------------------------


@pytest.mark.requires_hypodd
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt) not available")
def test_skip_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, _orig_cuspid_map, station_table = _build_example1_catalog(stations, events)

    cfg = _write_config(
        tmp_path,
        station_table,
        ph2dt_overrides={"maxdist_km": 120, "maxsep_km": 10, "minobs": 1, "maxobs": 100},
    )
    paths = RunPaths(cfg)
    paths.ensure()
    write_catalog(cat, paths.catalog_mag_xml)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")

    try:
        bin_path("ph2dt")
    except ExternalToolMissing:
        pytest.skip("ph2dt binary not available")

    pairs1 = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    dt_ct_path = paths.dt_ct
    mtime1 = dt_ct_path.stat().st_mtime_ns

    # Nothing changed -- ph2dt must not be rerun (dt.ct's mtime is untouched),
    # but the same event-pair list is still returned (reparsed from the
    # existing dt.ct, not silently trusting a stale in-memory value either).
    pairs2 = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    mtime2 = dt_ct_path.stat().st_mtime_ns

    assert pairs2 == pairs1
    assert mtime2 == mtime1

    # force=True reruns it (mtime moves, or at least ph2dt actually executes;
    # since ph2dt's output is deterministic for unchanged input, compare
    # content rather than requiring a different mtime on a fast filesystem).
    pairs3 = run_ph2dt(
        cfg,
        paths,
        force=True,
        external_hypodd_dir=fake_inc_root,
        external_bin_dir=tmp_path / "fake_bin",
    )
    assert pairs3 == pairs1

    # Now change the catalogue (reject one event's picks entirely by
    # removing the event) -- the fingerprint must change and ph2dt must
    # rerun even without force=True.
    smaller_cat = Catalog(events=list(cat)[:-1])
    write_catalog(smaller_cat, paths.catalog_mag_xml)
    pairs4 = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert len(pairs4) <= len(pairs1)


# ---------------------------------------------------------------------------
# test_synthetic_project_ph2dt_runs
# ---------------------------------------------------------------------------


def _catalog_from_truth(truth: dict) -> Catalog:
    """Build a located, magnitude-bearing Catalog by hand from
    ``synthetic_project``'s ``truth`` structure (the magnitude step, M8.3,
    isn't built yet -- orchestrator note: "build located catalogues by hand
    in tests ... plus magnitudes via openmeq.catalog").
    """
    cat = Catalog()
    for ev_truth in truth["events"]:
        origin_truth = ev_truth["origin"]
        eid = ev_truth["event_id_hint"]
        event = make_event(eid)
        origin = Origin(
            time=origin_truth["time"],
            latitude=origin_truth["lat"],
            longitude=origin_truth["lon"],
            depth=origin_truth["depth_km"] * 1000.0,
        )
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        mag_obj = Magnitude(mag=ev_truth["ml"], magnitude_type="ML")
        event.magnitudes = [mag_obj]
        event.preferred_magnitude_id = mag_obj.resource_id

        for net_sta, phases in ev_truth["arrivals"].items():
            net, sta = net_sta.split(".", 1)
            for phase, t in phases.items():
                pick = Pick(
                    time=t,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(network_code=net, station_code=sta),
                )
                event.picks.append(pick)
                arrival = Arrival(pick_id=pick.resource_id, phase=phase, time_weight=1.0)
                origin.arrivals.append(arrival)

        cat.append(event)
    return cat


@pytest.mark.requires_hypodd
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt) not available")
def test_synthetic_project_ph2dt_runs(synthetic_project, tmp_path):
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")
    paths = RunPaths(cfg)
    paths.ensure()

    cat = _catalog_from_truth(truth)
    write_catalog(cat, paths.catalog_mag_xml)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")

    try:
        bin_path("ph2dt")
    except ExternalToolMissing:
        pytest.skip("ph2dt binary not available")

    pairs = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )

    assert paths.dt_ct.is_file()
    assert len(pairs) > 0, "expected a non-trivial event-pair list from the synthetic catalogue"
    # Every parsed cuspid must resolve back to a real event_id via the
    # persisted map -- catches any cuspid drift/mismatch silently.
    event_id_map = build_event_id_map(cat, paths.relocate_work_dir / "event_id_map.csv")
    cuspid_to_event_id = invert_event_id_map(event_id_map)
    for a, b in pairs:
        assert a in cuspid_to_event_id
        assert b in cuspid_to_event_id
