"""The example-1 reproduction test (M10.5 / OpenMEQ-16d.5, PLAN.md §10's
"the full pipeline must reproduce the example-1 hypoDD.reloc"): the
project's primary hypoDD-correctness gate.

Reconstructs ``external/HYPODD/examples/example1``'s own bundled inputs
(``El16.pha``, ``station.dat``, its own ``dt.cc``) as an OpenMEQ-native
catalogue (reusing ``test_relocate_hypodd_files``'s
``_build_example1_catalog``/``_read_example1_events``/
``_read_example1_stations`` fixture helpers -- M9.2's own test already
established that these round-trip through ``write_phase_dat``/
``write_event_dat`` correctly, and M10.2's own
``test_example1_dt_ct_reproduction`` already proved this project's own
``ph2dt`` invocation reproduces the bundled ``dt.ct`` exactly), then drives
``openmeq.relocate.run_relocate`` -- this task's own orchestration, not a
hand-assembled ad hoc script -- end to end through M9.2's writers, M10.2's
real ``ph2dt`` invocation and M10.4's real ``hypoDD`` invocation.

M10 has no raw waveforms for example1's events (only its own bundled
``dt.cc``, since example1 ships pre-computed cross-correlation results, not
seismograms) -- ``run_xcorr`` is monkeypatched (as imported into
``openmeq.relocate``'s own namespace) to copy the bundled ``dt.cc`` in
verbatim instead of actually cross-correlating, per this issue's own
pitfall note ("don't let this test's necessary use of example1's own dt.cc
create false confidence about M10.3's correctness -- M10.3 has its own
dedicated sign-convention tests"). Every other step runs for real.

**Weighting schedule / ISOLV / velocity model**: this test uses example1's
own original ``hypoDD.inp`` tuning (transcribed verbatim below from
``examples/example1/hypoDD.inp``), not this project's own small-aperture-
network-tuned defaults (``docs/hypodd_tuning.md``) -- the whole point of
this test is to reproduce example1's own original ``hypoDD.reloc``, which
was produced with example1's own tuning; applying this project's own
different-network-scale defaults to example1's own regional network would
make any discrepancy meaningless to interpret (a real bug vs. a
deliberately different, better-justified weighting scheme -- see this
issue's own design note). ``ph2dt`` overrides similarly reuse
``test_relocate_ph2dt_step.py``'s own working values, transcribed from
example1's own bundled ``ph2dt.inp``.

Because the persisted ``event_id_map.csv``/``station_label_map.csv``
(ARCHITECTURE.md §6c) are loaded-and-reused rather than regenerated
(`build_event_id_map`/`build_station_label_map`), pre-seeding
``event_id_map.csv`` with example1's own original cuspids (via
``_build_example1_catalog``'s returned mapping) before running means this
test's own ``ph2dt``/``hypoDD`` runs use example1's *exact* original
cuspids throughout -- so the bundled ``dt.cc`` (which references those same
cuspids) can be injected directly with no translation, and the resulting
``hypoDD.reloc``'s ``CUSPID`` column can be compared straight against
``examples/example1/hypoDD.reloc``'s own.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from obspy.geodetics import gps2dist_azimuth
from test_relocate_hypodd_files import (  # noqa: E402
    _build_example1_catalog,
    _read_example1_events,
    _read_example1_stations,
)
from test_relocate_ph2dt_step import _write_fake_inc_tree  # noqa: E402

from openmeq.catalog import event_id, get_extra, read_catalog, write_catalog
from openmeq.config import load_config
from openmeq.external import ExternalToolMissing, bin_path
from openmeq.paths import RunPaths
from openmeq.relocate import run_relocate
from openmeq.relocate.hypodd_files import build_event_id_map

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"
HYPODD_BIN_MISSING = not (
    (EXTERNAL_DIR / "bin" / "hypoDD").exists() and (EXTERNAL_DIR / "bin" / "ph2dt").exists()
)
EXAMPLE1_MISSING = not (EXAMPLE1_DIR / "dt.cc").exists()

# ---------------------------------------------------------------------------
# example1's own original hypoDD.inp tuning, transcribed verbatim (see the
# module docstring for why this test doesn't use this project's own
# defaults). NITER is a per-row *count* of iterations (hypoDD's own
# convention, confirmed against a real run's hypoDD.log: rows 1,3,3,3,3
# produced iteration ranges "1-1","2-4","5-7","8-10","11-13"), not a
# cumulative "last iteration" index.
# ---------------------------------------------------------------------------
_EXAMPLE1_WEIGHTING_ROWS = [
    {
        "niter": 1,
        "wtccp": 0.01,
        "wtccs": 0.005,
        "wrcc": -999,
        "wdcc": -999,
        "wtctp": 1.00,
        "wtcts": 0.5,
        "wrct": 20,
        "wdct": -999,
        "damp": 20,
    },
    {
        "niter": 3,
        "wtccp": 0.01,
        "wtccs": 0.005,
        "wrcc": -999,
        "wdcc": -999,
        "wtctp": 1.00,
        "wtcts": 0.5,
        "wrct": 10,
        "wdct": 4,
        "damp": 20,
    },
    {
        "niter": 3,
        "wtccp": 1,
        "wtccs": 0.5,
        "wrcc": 6,
        "wdcc": 3,
        "wtctp": 1.00,
        "wtcts": 0.5,
        "wrct": 10,
        "wdct": 3,
        "damp": 20,
    },
    {
        "niter": 3,
        "wtccp": 1,
        "wtccs": 0.5,
        "wrcc": 5,
        "wdcc": 2,
        "wtctp": 0.1,
        "wtcts": 0.05,
        "wrct": 9,
        "wdct": 2,
        "damp": 20,
    },
    {
        "niter": 3,
        "wtccp": 1,
        "wtccs": 0.5,
        "wrcc": 4,
        "wdcc": 1,
        "wtctp": 0.01,
        "wtcts": 0.005,
        "wrct": 8,
        "wdct": 2,
        "damp": 20,
    },
]

#: example1's own 1D model (``hypoDD.inp``'s active, uncommented IMOD=1
#: "variable vp/vs ratio" block): top depth (km), vp (km/s), constant
#: vp/vs ratio 1.73 (from which vs is derived for OpenMEQ's own layer-file
#: format, ARCHITECTURE.md §6a).
_EXAMPLE1_LAYER_TOP_KM = [0.00, 0.25, 1.50, 2.50, 3.50, 5.00, 6.00, 9.00, 15.00, 25.00]
_EXAMPLE1_LAYER_VP = [1.42, 3.24, 4.82, 5.36, 5.60, 5.65, 5.90, 6.15, 6.60, 8.00]
_EXAMPLE1_VP_VS_RATIO = 1.73

#: example1's own ph2dt.inp values (MINWGHT MAXDIST MAXSEP MAXNGH MINLNK
#: MINOBS MAXOBS: 0 120 10 10 8 1 100) -- maxngh/minlnk/minwght already
#: match this project's own defaults (config/example.yaml), only
#: maxdist_km/maxsep_km/minobs/maxobs differ. Same values
#: ``test_relocate_ph2dt_step.py::test_example1_dt_ct_reproduction`` already
#: proved reproduce the bundled ``dt.ct`` exactly.
_EXAMPLE1_PH2DT_OVERRIDES = {"maxdist_km": 120, "maxsep_km": 10, "minobs": 1, "maxobs": 100}

#: A "few tens of metres" (the issue's own phrasing) of relative-location
#: agreement, with headroom for float-formatting/rounding differences
#: between this project's own writers and hypoddpy's/example1's original
#: run -- see the test's own printed pass/fail summary for the actual
#: measured agreement. Measured in practice (real ph2dt/hypoDD binaries,
#: this project's own writers, example1's own bundled dt.cc): mean 3D
#: diff ~0.1 m, max ~0.7 m across all 16 relocated events -- essentially
#: bit-identical to example1's own original run, since this test uses
#: example1's own original tuning verbatim (see the module docstring) and
#: OpenMEQ's own ph2dt/phase.dat/event.dat writers already reproduce
#: example1's dt.ct exactly (test_relocate_ph2dt_step.py). 20 m keeps
#: ~30x headroom over that measurement for run-to-run/platform float
#: differences, while still being tight enough to catch a real regression.
_TOLERANCE_KM = 0.02


def _base_config_dict() -> dict:
    with open(EXAMPLE_CONFIG) as fh:
        return yaml.safe_load(fh)


def _write_example1_layer_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# depth_top_km  vp_km_s  vs_km_s"]
    for top, vp in zip(_EXAMPLE1_LAYER_TOP_KM, _EXAMPLE1_LAYER_VP, strict=True):
        vs = vp / _EXAMPLE1_VP_VS_RATIO
        lines.append(f"{top:.4f}  {vp:.4f}  {vs:.4f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_example1_config(tmp_path: Path, station_table: pd.DataFrame):
    data = _base_config_dict()
    data["project"]["output_dir"] = str(tmp_path / "run")

    # cfg.stations.csv's own raw-column convention (openmeq.stations._CSV_COLUMNS:
    # net,sta,loc,lat,lon,elev_m,depth_m) differs from station_table()'s
    # output columns (network,station,...).
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

    layer_path = _write_example1_layer_file(tmp_path / "vel1d_example1.txt")
    data["hypodd"]["velocity_model"] = {"type": "1d", "file": str(layer_path)}
    data["hypodd"]["ph2dt"].update(_EXAMPLE1_PH2DT_OVERRIDES)
    data["hypodd"]["isolv"] = 1  # example1's own SVD, not this project's own LSQR default
    data["hypodd"]["max_station_dist_km"] = 120  # example1's own IDAT/IPHA/DIST line
    data["hypodd"]["weighting_rows"] = _EXAMPLE1_WEIGHTING_ROWS

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(data, fh)
    return load_config(config_path)


def _parse_bundled_reloc(path: Path) -> dict[int, dict]:
    """Minimal, self-contained ``hypoDD.reloc`` parser for the bundled
    comparison file -- deliberately not
    `openmeq.relocate.hypodd_runner.read_reloc` (which needs an
    ``event_id_map`` and raises on an unmapped ``CUSPID``): this test wants
    every bundled row, keyed by its own raw cuspid, with no dependency on
    this run's own map.
    """
    rows: dict[int, dict] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            cuspid = int(parts[0])
            lat, lon, depth_km = float(parts[1]), float(parts[2]), float(parts[3])
            rows[cuspid] = {"lat": lat, "lon": lon, "depth_km": depth_km}
    return rows


def _dist_3d_km(
    lat1: float, lon1: float, depth_km1: float, lat2: float, lon2: float, depth_km2: float
) -> float:
    horiz_m, _az, _baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz_km = depth_km1 - depth_km2
    return math.sqrt((horiz_m / 1000.0) ** 2 + dz_km**2)


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(
    HYPODD_BIN_MISSING or EXAMPLE1_MISSING,
    reason="external/HYPODD (ph2dt/hypoDD binaries or examples/example1) not available",
)
def test_example1_reproduction(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    cat, orig_cuspid_map, station_table = _build_example1_catalog(stations, events)

    cfg = _write_example1_config(tmp_path, station_table)
    paths = RunPaths(cfg)
    paths.ensure()
    write_catalog(cat, paths.catalog_mag_xml)

    # Pre-seed the persisted event_id/cuspid map with example1's OWN
    # original cuspids (see the module docstring) so this run's own ph2dt/
    # hypoDD use example1's exact original numbering throughout.
    paths.relocate_work_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.relocate_work_dir / "event_id_map.csv", "w") as fh:
        fh.write("event_id,cuspid\n")
        for eid, cuspid in orig_cuspid_map.items():
            fh.write(f"{eid},{cuspid}\n")

    try:
        bin_path("ph2dt")
        bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("ph2dt/hypoDD binaries not available")

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")

    def _fake_run_xcorr(cfg, paths, pairs, *, force=False):
        shutil.copy(EXAMPLE1_DIR / "dt.cc", paths.dt_cc)
        return paths.dt_cc

    monkeypatch.setattr("openmeq.relocate.run_xcorr", _fake_run_xcorr)

    result = run_relocate(
        cfg,
        paths,
        external_hypodd_dir=fake_inc_root,
        external_bin_dir=tmp_path / "fake_bin",
    )
    assert result.hypodd.n_relocated > 0, "expected at least some example1 events to be relocated"

    merged = read_catalog(paths.catalog_dd_xml)
    bundled_reloc = _parse_bundled_reloc(EXAMPLE1_DIR / "hypoDD.reloc")
    our_event_id_map = build_event_id_map(cat, paths.relocate_work_dir / "event_id_map.csv")

    diffs_km: list[float] = []
    missing: list[int] = []
    outliers: list[tuple[int, float]] = []
    for event in merged:
        eid = event_id(event)
        if get_extra(event, "hypodd_status") != "relocated":
            continue
        cuspid = our_event_id_map[eid]
        bundled = bundled_reloc.get(cuspid)
        if bundled is None:
            missing.append(cuspid)
            continue
        origin = event.preferred_origin()
        d_km = _dist_3d_km(
            origin.latitude,
            origin.longitude,
            origin.depth / 1000.0,
            bundled["lat"],
            bundled["lon"],
            bundled["depth_km"],
        )
        diffs_km.append(d_km)
        if d_km > _TOLERANCE_KM:
            outliers.append((cuspid, d_km))

    assert diffs_km, (
        "expected at least one relocated event comparable against the bundled hypoDD.reloc"
    )
    mean_diff = sum(diffs_km) / len(diffs_km)
    max_diff = max(diffs_km)

    # Per this issue's own item 5: a clear pass/fail summary of the actual
    # numeric agreement, always printed (this is the project's primary
    # hypoDD-correctness gate and its output is valuable even when this
    # slow/requires_hypodd test isn't run as part of routine CI).
    with capsys.disabled():
        print(
            f"\nOpenMEQ-16d.5 example-1 reproduction: {len(diffs_km)}/{len(cat)} events "
            f"relocated and compared against examples/example1/hypoDD.reloc -- "
            f"mean 3D diff={mean_diff:.4f} km, max diff={max_diff:.4f} km, "
            f"{len(outliers)} event(s) outside the {_TOLERANCE_KM} km tolerance"
            + (f": {outliers}" if outliers else "")
        )

    assert not missing, (
        f"cuspid(s) relocated by OpenMEQ but absent from the bundled hypoDD.reloc: {missing}"
    )
    assert not outliers, (
        f"{len(outliers)}/{len(diffs_km)} event(s) diverged from example1's own hypoDD.reloc by "
        f"more than {_TOLERANCE_KM} km: {outliers}"
    )
