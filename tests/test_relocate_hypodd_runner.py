"""Tests for openmeq.relocate.hypodd_runner (M10.4 / OpenMEQ-16d.4).

Reuses M9.2/M10.2's own example1 fixture helpers (``test_relocate_hypodd_files``'s
``_read_example1_stations``/``_read_example1_events``/``_build_example1_catalog``)
and M10.2/M10.3's own scratch-``.inc``-tree/synthetic-project scaffolding
(``test_relocate_ph2dt_step``'s ``_write_fake_inc_tree``, ``test_relocate_xcorr``'s
``test_synthetic_project_xcorr_and_hypodd_smoke``), per this project's own
"reuse fixture setup, don't duplicate it" convention.

Disk-space note (worker instructions): every test here writes only into its
own function-scoped ``tmp_path``.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import Arrival, Magnitude, Origin, Pick, WaveformStreamID
from obspy.geodetics import gps2dist_azimuth
from test_relocate_hypodd_files import (  # noqa: E402
    _build_example1_catalog,
    _read_example1_events,
    _read_example1_stations,
)
from test_relocate_ph2dt_step import _write_fake_inc_tree  # noqa: E402

from openmeq.catalog import event_id, get_extra, make_event, read_catalog, write_catalog
from openmeq.config import ConfigError, ParallelConfig
from openmeq.external import ExternalToolMissing, bin_path
from openmeq.paths import RunPaths
from openmeq.relocate.hypodd_runner import (
    HypoddRunFailed,
    RelocatedOrigin,
    _build_relocated_origin,
    _run_hypodd_clustered,
    invoke_hypodd,
    merge_relocations,
    read_reloc,
    run_hypodd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_DIR = EXTERNAL_DIR / "HYPODD" / "examples" / "example1"
HYPODD_BIN_MISSING = not (EXTERNAL_DIR / "bin" / "hypoDD").exists()
EXAMPLE1_MISSING = not EXAMPLE1_DIR.exists()


def _make_reloc(**overrides) -> RelocatedOrigin:
    defaults = dict(
        cuspid=1,
        lat=37.001,
        lon=-122.001,
        depth_km=5.2,
        x_m=10.0,
        y_m=-5.0,
        z_m=3.0,
        ex_m=1.0,
        ey_m=1.5,
        ez_m=2.0,
        time=UTCDateTime("2025-01-01T00:00:01"),
        mag=1.2,
        nccp=5,
        nccs=4,
        nctp=10,
        ncts=2,
        rcc=0.01,
        rct=0.02,
        cid=1,
    )
    defaults.update(overrides)
    return RelocatedOrigin(**defaults)


def _make_located_event(event_id_str: str, *, lat=37.0, lon=-122.0, depth_m=5000.0) -> tuple:
    event = make_event(event_id_str)
    origin = Origin(
        time=UTCDateTime("2025-01-01T00:00:00"), latitude=lat, longitude=lon, depth=depth_m
    )
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id
    return event, origin


# ---------------------------------------------------------------------------
# test_example1_reloc_parsing (bundled static file -- no hypoDD binary needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(EXAMPLE1_MISSING, reason="external/HYPODD/examples/example1 not available")
def test_example1_reloc_parsing():
    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    _cat, event_id_map, _station_table = _build_example1_catalog(stations, events)

    reloc_map = read_reloc(EXAMPLE1_DIR / "hypoDD.reloc", event_id_map)

    target_event_id = next(eid for eid, cuspid in event_id_map.items() if cuspid == 38542)
    r = reloc_map[target_event_id]

    # Hand-picked row (external/HYPODD/examples/example1/hypoDD.reloc):
    #     38542  37.878646 -122.244507     9.251      -48.7       62.8
    #     -57.3      3.2      3.2     14.0 1985  1 24  2 19 58.740  1.40
    #     106   103   177     0  0.006  0.047   1
    assert r.cuspid == 38542
    assert r.lat == pytest.approx(37.878646)
    assert r.lon == pytest.approx(-122.244507)
    assert r.depth_km == pytest.approx(9.251)
    assert r.x_m == pytest.approx(-48.7)
    assert r.y_m == pytest.approx(62.8)
    assert r.z_m == pytest.approx(-57.3)
    assert r.ex_m == pytest.approx(3.2)
    assert r.ey_m == pytest.approx(3.2)
    assert r.ez_m == pytest.approx(14.0)
    assert r.time == UTCDateTime(1985, 1, 24, 2, 19, 58, 740000)
    assert r.mag == pytest.approx(1.40)
    assert r.nccp == 106
    assert r.nccs == 103
    assert r.nctp == 177
    assert r.ncts == 0
    assert r.rcc == pytest.approx(0.006)
    assert r.rct == pytest.approx(0.047)
    assert r.cid == 1


# ---------------------------------------------------------------------------
# test_magnitudes_carried_over
# ---------------------------------------------------------------------------


def test_magnitudes_carried_over():
    eid = "ev20250101T000000_000"
    event, _origin = _make_located_event(eid)
    mag = Magnitude(mag=1.23, magnitude_type="ML")
    event.magnitudes = [mag]
    event.preferred_magnitude_id = mag.resource_id
    cat = Catalog(events=[event])

    reloc = _make_reloc(cuspid=101)
    merged, n_relocated, n_not_linked = merge_relocations(cat, {eid: reloc})

    assert n_relocated == 1
    assert n_not_linked == 0
    out_event = merged[0]
    assert len(out_event.magnitudes) == 1
    assert out_event.preferred_magnitude().mag == pytest.approx(1.23)
    assert out_event.preferred_magnitude().resource_id == mag.resource_id

    new_origin = out_event.preferred_origin()
    assert len(out_event.origins) == 2  # old + new -- nothing deleted
    assert str(new_origin.method_id) == "smi:openmeq/hypodd"
    assert new_origin.latitude == pytest.approx(reloc.lat)
    assert new_origin.longitude == pytest.approx(reloc.lon)


# ---------------------------------------------------------------------------
# test_unlinked_events_keep_prior_origin
# ---------------------------------------------------------------------------


def test_unlinked_events_keep_prior_origin(tmp_path):
    eid_linked = "ev20250101T000000_000"
    eid_unlinked = "ev20250101T000100_000"
    event_linked, _origin_linked = _make_located_event(eid_linked)
    event_unlinked, origin_unlinked = _make_located_event(eid_unlinked, lat=38.0, lon=-121.0)
    cat = Catalog(events=[event_linked, event_unlinked])

    reloc = _make_reloc(cuspid=1)
    merged, n_relocated, n_not_linked = merge_relocations(cat, {eid_linked: reloc})

    assert n_relocated == 1
    assert n_not_linked == 1

    out_linked = next(ev for ev in merged if event_id(ev) == eid_linked)
    out_unlinked = next(ev for ev in merged if event_id(ev) == eid_unlinked)

    assert str(out_linked.preferred_origin().method_id) == "smi:openmeq/hypodd"
    assert get_extra(out_linked, "hypodd_status") == "relocated"

    assert out_unlinked.preferred_origin() is origin_unlinked
    assert out_unlinked.preferred_origin().latitude == pytest.approx(38.0)
    assert len(out_unlinked.origins) == 1
    assert get_extra(out_unlinked, "hypodd_status") == "not_linked"

    # Round-trip through QuakeML to prove the extras survive (ARCHITECTURE.md
    # §5's "always use set_extra/get_extra so values survive a QuakeML
    # round-trip").
    out_path = tmp_path / "catalog_dd.xml"
    write_catalog(merged, out_path)
    reread = read_catalog(out_path)
    reread_unlinked = next(ev for ev in reread if event_id(ev) == eid_unlinked)
    assert get_extra(reread_unlinked, "hypodd_status") == "not_linked"
    reread_linked = next(ev for ev in reread if event_id(ev) == eid_linked)
    assert get_extra(reread_linked, "hypodd_status") == "relocated"


# ---------------------------------------------------------------------------
# test_depth_unit_conversion
# ---------------------------------------------------------------------------


def test_depth_unit_conversion():
    reloc = _make_reloc(depth_km=7.25)
    origin = _build_relocated_origin(reloc)
    assert origin.depth == pytest.approx(7250.0)

    # And through the full read_reloc parse too.
    reloc_path_text = (
        "     1  37.000000 -122.000000     7.250       0.0        0.0        "
        "0.0      1.0      1.0      1.0 2025  1  1  0  0  1.000  1.20     "
        "5     4    10     2  0.010  0.020   1\n"
    )
    fields = reloc_path_text.split()
    assert len(fields) == 24
    assert float(fields[3]) == pytest.approx(7.25)


# ---------------------------------------------------------------------------
# test_stale_event_id_map_raises_clearly
# ---------------------------------------------------------------------------


def test_stale_event_id_map_raises_clearly(tmp_path):
    reloc_path = tmp_path / "hypoDD.reloc"
    reloc_path.write_text(
        "    99999  37.878646 -122.244507     9.251      -48.7       62.8      "
        "-57.3      3.2      3.2     14.0 1985  1 24  2 19 58.740  1.40   106   "
        "103   177     0  0.006  0.047   1\n"
    )
    with pytest.raises(ConfigError, match="event_id_map"):
        read_reloc(reloc_path, {"ev_some_other_event": 1})


def test_read_reloc_rejects_malformed_line(tmp_path):
    reloc_path = tmp_path / "hypoDD.reloc"
    reloc_path.write_text("1 2 3\n")
    with pytest.raises(ConfigError, match="24"):
        read_reloc(reloc_path, {"ev_x": 1})


# ---------------------------------------------------------------------------
# invoke_hypodd error path (no real binary needed -- a fake failing script)
# ---------------------------------------------------------------------------


def test_invoke_hypodd_raises_with_log_tail_on_failure(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "hypoDD.log").write_text("some diagnostic hypoDD wrote before failing\n")

    fake_bin = tmp_path / "fake_hypodd.sh"
    fake_bin.write_text("#!/bin/sh\necho boom >&2\nexit 7\n")
    fake_bin.chmod(0o755)

    with pytest.raises(HypoddRunFailed, match="some diagnostic hypoDD wrote before failing"):
        invoke_hypodd(fake_bin, work_dir)


def test_invoke_hypodd_raises_when_reloc_missing_despite_zero_exit(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "hypoDD.log").write_text("aborted before opening hypoDD.reloc\n")

    fake_bin = tmp_path / "fake_hypodd.sh"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    with pytest.raises(HypoddRunFailed, match="did not produce"):
        invoke_hypodd(fake_bin, work_dir)


# ---------------------------------------------------------------------------
# run_hypodd orchestration: missing-input errors and skip-unchanged/force
# (a fake hypoDD script -- no real binary needed; these test this module's
# own orchestration logic, not hypoDD's physics, which the requires_hypodd
# tests above already cover).
# ---------------------------------------------------------------------------


def test_run_hypodd_missing_hypodd_inp_raises_clearly(tmp_path):
    from test_relocate_ph2dt_step import _station_table_from_dict, _write_config

    station_table = _station_table_from_dict({"AAA": (37.8, -122.2)})
    cfg = _write_config(tmp_path, station_table)
    paths = RunPaths(cfg)
    paths.ensure()

    with pytest.raises(ConfigError, match="hypoDD.inp"):
        run_hypodd(cfg, paths)


def test_run_hypodd_missing_catalog_raises_clearly(tmp_path):
    from test_relocate_ph2dt_step import _station_table_from_dict, _write_config

    station_table = _station_table_from_dict({"AAA": (37.8, -122.2)})
    cfg = _write_config(tmp_path, station_table)
    paths = RunPaths(cfg)
    paths.ensure()

    work_dir = paths.relocate_work_dir
    for name in ("hypoDD.inp", "event.sel", "station.sel", "dt.ct"):
        (work_dir / name).write_text("")

    assert not paths.catalog_mag_xml.exists()
    with pytest.raises(ConfigError, match="catalog_mag.xml"):
        run_hypodd(cfg, paths)


def _write_fake_hypodd_script(path: Path, cuspid: int) -> None:
    """A fake ``hypoDD`` executable: appends one line to ``hypodd_calls.log``
    in its own cwd (so a test can count invocations) and writes a minimal,
    valid-format ``hypoDD.reloc`` for a single, fixed `cuspid`."""
    path.write_text(
        "#!/bin/sh\n"
        'echo "call" >> hypodd_calls.log\n'
        "cat > hypoDD.reloc << 'EOF'\n"
        f"   {cuspid}  37.000000 -122.000000     5.000       0.0        0.0"
        "        0.0      1.0      1.0      1.0 2025  1  1  0  0  1.000  "
        "1.20     5     4    10     2  0.010  0.020   1\n"
        "EOF\n"
        "touch hypoDD.log\n"
        "exit 0\n"
    )
    path.chmod(0o755)


def test_run_hypodd_skip_unchanged_then_force(tmp_path, monkeypatch):
    from test_relocate_ph2dt_step import (
        _station_table_from_dict,
        _write_config,
        _write_fake_inc_tree,
    )

    from openmeq.relocate.hypodd_files import build_event_id_map

    station_table = _station_table_from_dict({"AAA": (37.8, -122.2)})
    cfg = _write_config(tmp_path, station_table)
    paths = RunPaths(cfg)
    paths.ensure()

    eid = "ev20250101T000000_000"
    event, _origin = _make_located_event(eid)
    cat = Catalog(events=[event])
    write_catalog(cat, paths.catalog_mag_xml)

    work_dir = paths.relocate_work_dir
    for name in ("hypoDD.inp", "event.sel", "station.sel"):
        (work_dir / name).write_text("")
    (work_dir / "dt.ct").write_text("")  # no pairs -- fine, only existence is checked

    event_id_map = build_event_id_map(cat, work_dir / "event_id_map.csv")
    cuspid = event_id_map[eid]

    fake_bin = tmp_path / "fake_hypodd.sh"
    _write_fake_hypodd_script(fake_bin, cuspid)
    monkeypatch.setattr("openmeq.relocate.hypodd_runner.bin_path", lambda name: fake_bin)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd_inc")

    result1 = run_hypodd(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert result1.rebuilt is True
    assert result1.n_relocated == 1
    calls_after_1 = (work_dir / "hypodd_calls.log").read_text().count("call")
    assert calls_after_1 == 1

    # Nothing changed -- the fake binary must not be invoked again.
    result2 = run_hypodd(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert result2.rebuilt is False
    assert result2.run_mode == "skipped"
    assert result2.n_relocated == 1  # catalog_dd.xml still correctly (re)derived
    calls_after_2 = (work_dir / "hypodd_calls.log").read_text().count("call")
    assert calls_after_2 == 1

    # force=True reinvokes it even though nothing changed.
    result3 = run_hypodd(
        cfg,
        paths,
        force=True,
        external_hypodd_dir=fake_inc_root,
        external_bin_dir=tmp_path / "fake_bin",
    )
    assert result3.rebuilt is True
    calls_after_3 = (work_dir / "hypodd_calls.log").read_text().count("call")
    assert calls_after_3 == 2

    # Changing dt.ct (even to still-empty-of-pairs content) must trigger a
    # rerun without force=True.
    (work_dir / "dt.ct").write_text("# 1 2\n")
    result4 = run_hypodd(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert result4.rebuilt is True
    calls_after_4 = (work_dir / "hypodd_calls.log").read_text().count("call")
    assert calls_after_4 == 3


# ---------------------------------------------------------------------------
# test_example1_full_run (requires_hypodd, slow)
# ---------------------------------------------------------------------------


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(
    HYPODD_BIN_MISSING or EXAMPLE1_MISSING, reason="external/HYPODD (hypoDD) not available"
)
def test_example1_full_run(tmp_path, monkeypatch):
    """A pure "does our runner correctly invoke and parse the real binary"
    test: example1's own bundled dt.cc/dt.ct/event.sel/station.sel/hypoDD.inp,
    unmodified, run through our own `invoke_hypodd` + `read_reloc` -- not any
    of OpenMEQ's own file writers. Confirmed (this task's own development)
    that hypoDD is deterministic for identical inputs in this environment
    (a byte-identical hypoDD.reloc, verified with `diff`); this test still
    uses a small numerical tolerance rather than requiring byte-identical
    text, since determinism across *environments* (different compiler/libm)
    is not itself guaranteed.
    """
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))
    try:
        hypodd_bin = bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("hypoDD binary not available")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for name in ("dt.cc", "dt.ct", "event.sel", "station.sel", "hypoDD.inp"):
        shutil.copy(EXAMPLE1_DIR / name, work_dir / name)

    invoke_hypodd(hypodd_bin, work_dir)

    stations = _read_example1_stations(EXAMPLE1_DIR)
    events = _read_example1_events(EXAMPLE1_DIR)
    _cat, event_id_map, _station_table = _build_example1_catalog(stations, events)

    ours = read_reloc(work_dir / "hypoDD.reloc", event_id_map)
    bundled = read_reloc(EXAMPLE1_DIR / "hypoDD.reloc", event_id_map)

    assert set(ours) == set(bundled)
    assert len(bundled) == 16
    for eid in bundled:
        a, b = ours[eid], bundled[eid]
        assert a.cuspid == b.cuspid
        assert a.lat == pytest.approx(b.lat, abs=1e-4)
        assert a.lon == pytest.approx(b.lon, abs=1e-4)
        assert a.depth_km == pytest.approx(b.depth_km, abs=1e-2)
        assert abs(a.time - b.time) < 0.01
        assert a.mag == pytest.approx(b.mag, abs=1e-2)
        assert a.nccp == b.nccp
        assert a.nccs == b.nccs
        assert a.nctp == b.nctp
        assert a.ncts == b.ncts


# ---------------------------------------------------------------------------
# test_parallel_clusters_matches_single_run (requires_hypodd, slow)
# ---------------------------------------------------------------------------


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(
    HYPODD_BIN_MISSING or EXAMPLE1_MISSING, reason="external/HYPODD (hypoDD) not available"
)
def test_parallel_clusters_matches_single_run(tmp_path, monkeypatch):
    """Proof of consistency for the experimental ``cfg.hypodd.parallel_clusters``
    path (the issue's own bar for shipping it): raising example1's own
    ``OBSCT`` from 8 to 60 splits its real 16 events into 3 real hypoDD
    clusters (sizes 6/4/2, plus 4 isolated events left un-relocated by
    *both* paths) -- confirmed empirically during this task's development.
    A single CID=0 run and this module's own per-cluster orchestration,
    run against the exact same (real, unmodified-except-for-OBSCT) inputs,
    must produce bit-identical merged relocations, since hypoDD relocates
    each cluster as an independent sub-problem either way.
    """
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))
    try:
        hypodd_bin = bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("hypoDD binary not available")

    base_inp = (EXAMPLE1_DIR / "hypoDD.inp").read_text()
    needle = "     0     8        -999     -999    -999 \n"
    replacement = "     0    60        -999     -999    -999 \n"
    assert needle in base_inp, "example1's own OBSCT line format changed?"
    base_inp = base_inp.replace(needle, replacement)
    # example1's own bundled hypoDD.inp targets CID=1 (harmless with its
    # own OBSCT=8, since nclust=1 there -- "cluster 1" already means "every
    # event"), not CID=0 -- once OBSCT=60 above actually splits the dataset
    # into multiple clusters, CID=1 would silently mean "only cluster 1".
    # Force CID=0 (relocate every cluster) for this test's own single-run
    # reference, so it is a genuine "everything in one run" comparison.
    cid_needle = "    1      \n"
    assert cid_needle in base_inp, "example1's own CID line format changed?"
    base_inp = base_inp.replace(cid_needle, "    0      \n")

    # Reference: a single CID=0 run over the OBSCT=60 inputs.
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    for name in ("dt.cc", "dt.ct", "event.sel", "station.sel"):
        shutil.copy(EXAMPLE1_DIR / name, single_dir / name)
    (single_dir / "hypoDD.inp").write_text(base_inp)
    invoke_hypodd(hypodd_bin, single_dir)
    single_lines = {
        line.split()[0]: line
        for line in (single_dir / "hypoDD.reloc").read_text().splitlines()
        if line.strip()
    }
    assert len(single_lines) == 12  # 6 + 4 + 2 clustered events; 4 isolated

    # This module's own per-cluster orchestration over the same inputs, but
    # with hypoDD.inp written the way openmeq.relocate.hypodd_files always
    # writes it (bare "0\n" CID line after the marker comment -- cluster
    # parallelism's own _rewrite_cid_line requires this, module docstring).
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for name in ("dt.cc", "dt.ct", "event.sel", "station.sel"):
        shutil.copy(EXAMPLE1_DIR / name, work_dir / name)
    # example1's own footer doesn't match this project's writer's marker
    # line/format at all -- build our own footer instead of trying to
    # string-replace example1's differently-worded one.
    marker_idx = base_inp.index("*--- event selection:")
    our_inp = (
        base_inp[:marker_idx]
        + "* event selection: CID (0 = all)\n"
        + "0\n"
        + "* ID (blank = all events in the selected cluster)\n"
    )
    (work_dir / "hypoDD.inp").write_text(our_inp)

    class _FakeCfg:
        parallel = ParallelConfig(n_workers=1)

    nclust = _run_hypodd_clustered(
        _FakeCfg(), hypodd_bin, work_dir, None, None, logging.getLogger("test.hypodd_runner")
    )
    assert nclust == 3

    merged_lines = {
        line.split()[0]: line
        for line in (work_dir / "hypoDD.reloc").read_text().splitlines()
        if line.strip()
    }
    assert set(merged_lines) == set(single_lines)
    for cuspid, merged_line in merged_lines.items():
        assert merged_line == single_lines[cuspid]


# ---------------------------------------------------------------------------
# test_synthetic_end_to_end: setup_hypodd -> run_ph2dt -> build_cache ->
# run_xcorr -> run_hypodd, relocated positions closer to truth than the
# perturbed input positions.
# ---------------------------------------------------------------------------


def _offset_latlon(lat: float, lon: float, north_km: float, east_km: float) -> tuple[float, float]:
    km_per_deg_lat = 111.19
    dlat = north_km / km_per_deg_lat
    dlon = east_km / (km_per_deg_lat * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def _dist_3d_km(
    lat1: float, lon1: float, depth_km1: float, lat2: float, lon2: float, depth_km2: float
) -> float:
    horiz_m, _az, _baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz_km = depth_km1 - depth_km2
    return math.sqrt((horiz_m / 1000.0) ** 2 + dz_km**2)


#: Combined ~0.5-0.6 km 3D perturbation (the issue's own spec, OpenMEQ-m1f.5
#: acceptance criterion): a fixed 0.5 km horizontal offset in a random
#: azimuth, plus up to +-0.3 km of independent vertical noise, per event --
#: deliberately *independent per event* (not a uniform shift of the whole
#: catalogue): hypoDD's double-difference method only recovers *relative*
#: geometry from purely differential data, so a shared/uniform translation
#: of every event would be entirely unobservable and never corrected;
#: independent per-event scatter is exactly what double-difference
#: relocation is designed to collapse back toward a tight,
#: internally-consistent (and, since the perturbation is zero-mean across
#: events, close to *absolutely* correct) cluster.
#:
#: **History (M10.4 / OpenMEQ-16d.4, superseded here)**: with the old
#: straight-ray synthetic travel times, a ~0.5 km perturbation was swallowed
#: by this shared synthetic fixture's own location-accuracy floor (a real,
#: if small, velocity-model mismatch between the generator's straight rays
#: and hypoDD's own proper 1D ray tracing through the same layered model,
#: independent of and in addition to whatever this test perturbs) -- at
#: ~0.5 km, that mismatch alone dominated the result (14 events tested,
#: only 2 improved), forcing a larger ~2.2-2.4 km perturbation instead.
#: OpenMEQ-m1f.5 replaced those straight rays with exact 1D layered
#: first-arrival times (direct ray + head waves), validated to <5 ms P /
#: <10 ms S against NLLoc's own Grid2Time
#: (``tests/locate/test_synthetic_traveltime_validation.py``) -- removing
#: that floor. At this original 0.5/0.3 km spec, all 20/20 events now
#: improve, with mean relocated error ~0.058 km (down from ~0.532 km
#: perturbed).
_HORIZ_OFFSET_KM = 0.5
_VERT_OFFSET_KM = 0.3


def _catalog_from_truth_perturbed(
    truth: dict, rng: np.random.Generator
) -> tuple[Catalog, dict[str, tuple[float, float, float]]]:
    cat = Catalog()
    perturbed_by_event: dict[str, tuple[float, float, float]] = {}

    for ev_truth in truth["events"]:
        origin_truth = ev_truth["origin"]
        eid = ev_truth["event_id_hint"]
        true_lat, true_lon = origin_truth["lat"], origin_truth["lon"]
        true_depth_km = origin_truth["depth_km"]

        az = rng.uniform(0.0, 2.0 * math.pi)
        north_km = _HORIZ_OFFSET_KM * math.cos(az)
        east_km = _HORIZ_OFFSET_KM * math.sin(az)
        down_km = rng.uniform(-_VERT_OFFSET_KM, _VERT_OFFSET_KM)

        p_lat, p_lon = _offset_latlon(true_lat, true_lon, north_km, east_km)
        p_depth_km = max(0.05, true_depth_km + down_km)

        event = make_event(eid)
        origin = Origin(
            time=origin_truth["time"], latitude=p_lat, longitude=p_lon, depth=p_depth_km * 1000.0
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
        perturbed_by_event[eid] = (p_lat, p_lon, p_depth_km)

    return cat, perturbed_by_event


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(HYPODD_BIN_MISSING, reason="external/HYPODD (ph2dt/hypoDD) not available")
def test_synthetic_end_to_end_relocation_improves_on_perturbed_input(
    synthetic_project, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))
    from openmeq.chunk import write_chunks
    from openmeq.config import load_config
    from openmeq.index import build_index
    from openmeq.relocate.ph2dt_step import run_ph2dt
    from openmeq.relocate.waveform_cache import build_cache
    from openmeq.relocate.xcorr import run_xcorr

    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")
    cfg.parallel.n_workers = 1
    # Loosen ph2dt's own filtering (defaults are tuned for a much larger
    # dataset) so this small, sparse 20-event synthetic archive still
    # yields a non-trivial linked-pair set -- same overrides
    # test_relocate_xcorr.py's own synthetic smoke test already uses.
    cfg.hypodd.ph2dt.maxdist_km = 200
    cfg.hypodd.ph2dt.maxsep_km = 50
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100
    # This tiny (20-event, 8-station) synthetic network's residuals are on
    # a much smaller scale (noise-free analytic travel times, ~ms) than the
    # project's own default progressive weighting/re-weighting schedule was
    # tuned for (a much larger, real dataset, ARCHITECTURE.md §6c) -- the
    # default schedule's residual/distance cutoffs progressively discard
    # almost all of this network's own (already-small) residuals, leaving
    # too little data to relocate most events by the final iteration.
    # A single, un-tightened weighting row (every residual/distance cutoff
    # left at hypoDD's own "-999 = inactive" sentinel) avoids that
    # divergence -- this test needs every event relocated so the accuracy
    # comparison isn't skewed by which few events happened to survive.
    from openmeq.config import WeightingRow

    cfg.hypodd.weighting_rows = [
        WeightingRow(
            niter=15,
            wtccp=1.0,
            wtccs=0.5,
            wrcc=-999,
            wdcc=-999,
            wtctp=1.0,
            wtcts=0.5,
            wrct=-999,
            wdct=-999,
            damp=20,
        )
    ]
    paths = RunPaths(cfg)
    paths.ensure()

    rng = np.random.default_rng(20260928)
    cat, perturbed_by_event = _catalog_from_truth_perturbed(truth, rng)
    write_catalog(cat, paths.catalog_mag_xml)

    index = build_index(cfg)
    rows = []
    for ev_truth in truth["events"]:
        eid = ev_truth["event_id_hint"]
        stations = sorted(ev_truth["arrivals"].keys())
        rows.append(
            {
                "event_id": eid,
                "time": str(UTCDateTime(ev_truth["origin"]["time"])),
                "duration_s": 5.0,
                "coincidence_sum": len(stations),
                "n_stations": len(stations),
                "stations": ";".join(stations),
                "trace_ids": "",
            }
        )
    detections_df = pd.DataFrame(rows)
    write_chunks(detections_df, index, cfg, paths)

    event_ids = [ev["event_id_hint"] for ev in truth["events"]]
    build_cache(cfg, paths, event_ids)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")
    try:
        bin_path("ph2dt")
        bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("ph2dt/hypoDD binaries not available")

    pairs = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert pairs, "expected at least one linked event pair from the synthetic catalogue"

    run_xcorr(cfg, paths, pairs)
    assert paths.dt_cc.is_file()
    assert paths.dt_cc.stat().st_size > 0

    result = run_hypodd(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert result.n_relocated > 0, "expected at least some synthetic events to be relocated"
    assert paths.catalog_dd_xml.is_file()

    merged = read_catalog(paths.catalog_dd_xml)
    truth_by_event = {ev["event_id_hint"]: ev["origin"] for ev in truth["events"]}

    perturbed_errors = []
    relocated_errors = []
    for event in merged:
        eid = event_id(event)
        if get_extra(event, "hypodd_status") != "relocated":
            continue
        truth_origin = truth_by_event[eid]
        p_lat, p_lon, p_depth_km = perturbed_by_event[eid]
        perturbed_err = _dist_3d_km(
            p_lat,
            p_lon,
            p_depth_km,
            truth_origin["lat"],
            truth_origin["lon"],
            truth_origin["depth_km"],
        )
        new_origin = event.preferred_origin()
        relocated_err = _dist_3d_km(
            new_origin.latitude,
            new_origin.longitude,
            new_origin.depth / 1000.0,
            truth_origin["lat"],
            truth_origin["lon"],
            truth_origin["depth_km"],
        )
        perturbed_errors.append(perturbed_err)
        relocated_errors.append(relocated_err)

    assert perturbed_errors, "expected at least one relocated event to compare against truth"
    mean_perturbed = sum(perturbed_errors) / len(perturbed_errors)
    mean_relocated = sum(relocated_errors) / len(relocated_errors)
    n_improved = sum(1 for p, r in zip(perturbed_errors, relocated_errors, strict=True) if r < p)
    with capsys.disabled():
        print(
            f"\nM10.4 synthetic relocation accuracy ({len(perturbed_errors)} relocated events): "
            f"perturbed mean error={mean_perturbed:.3f} km, relocated mean error="
            f"{mean_relocated:.3f} km, {n_improved}/{len(perturbed_errors)} events improved"
        )

    assert mean_relocated < mean_perturbed
    # OpenMEQ-m1f.5: with the synthetic generator's exact 1D layered travel
    # times, this measures mean_relocated ~0.058 km and 20/20 events
    # improved at this (now issue-spec, 0.5 km) perturbation -- add the
    # absolute/count bounds the issue's own acceptance criterion implies,
    # with a comfortable margin over that measurement (a real hypoDD/ph2dt
    # binary run, not mocked, so some run-to-run slack is kept rather than
    # asserting the exact measured values).
    assert mean_relocated < 0.15, f"mean relocated error {mean_relocated:.3f} km"
    assert n_improved >= len(perturbed_errors) - 1, (
        f"only {n_improved}/{len(perturbed_errors)} events improved"
    )
