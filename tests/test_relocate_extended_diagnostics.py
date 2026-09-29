"""Tests for `openmeq.relocate.extended_diagnostics` (OpenMEQ-02o.7): the
``hypoDD.res`` parser, the dt.cc/dt.ct matcher, weighting-set mapping, and
``write_relocate_diagnostics``'s extended on/off gating and never-fail
behaviour. Pure unit tests -- no hypoDD binary needed.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import pytest
from obspy import Catalog
from obspy.core.event import Origin

from openmeq.catalog import make_event, write_catalog
from openmeq.config import WeightingRow, load_config
from openmeq.paths import RunPaths
from openmeq.relocate import extended_diagnostics as ext
from openmeq.relocate.diagnostics import (
    HypoddIterationRow,
    parse_hypodd_log_iterations,
    write_relocate_diagnostics,
)
from openmeq.relocate.extended_diagnostics import (
    iteration_weighting_sets,
    match_dtcc_dtct,
    parse_hypodd_res,
    write_relocate_extended_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
EXAMPLE1_RES = EXTERNAL_DIR / "HYPODD" / "examples" / "example1" / "hypoDD.res"

EXTENDED_FIGURES = {
    "dtcc_vs_dtct.png",
    "dtcc_vs_dtct.csv",
    "res_final_by_type.png",
    "iteration_shifts.png",
}


def _res_line(sta, dt, c1, c2, idx, qual, res_ms, wt, offs) -> str:
    """One line in hypoDD.f:1271-1272's exact Fortran format."""
    return (
        f"{sta:<7s} {dt:12.7f} {c1:9d} {c2:9d} {idx:1d} {qual:9.4f} {res_ms:12.6f} "
        f"{wt:11.6f} {offs:8.1f}\n"
    )


# ---------------------------------------------------------------------------
# parse_hypodd_res
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EXAMPLE1_RES.is_file(), reason="external/HYPODD example1 not available")
def test_parse_hypodd_res_example1():
    rows = parse_hypodd_res(EXAMPLE1_RES)
    assert len(rows) == 3409  # every line of the real file
    assert Counter(r.data_type for r in rows) == {
        "CT-P": 1863,
        "CC-P": 851,
        "CC-S": 669,
        "CT-S": 26,
    }
    first = rows[0]
    # "NCCAI      0.0151391     38542     38520 1    0.9600     1.392568    0.958827     58.3"
    assert first.sta == "NCCAI"
    assert first.dt_s == pytest.approx(0.0151391)
    assert (first.cuspid1, first.cuspid2, first.idx) == (38542, 38520, 1)
    assert first.qual == pytest.approx(0.96)
    assert first.res_ms == pytest.approx(1.392568)
    assert first.weight == pytest.approx(0.958827)
    assert first.offset_m == pytest.approx(58.3)
    assert all(math.isfinite(r.res_ms) for r in rows)


def test_parse_hypodd_res_handmade(tmp_path):
    good = _res_line("STA1", 0.0123, 1, 2, 3, 1.0, -4.5, 0.8, 120.0)
    overflow = _res_line("STA2", -0.01, 1, 2, 2, 0.9, 0.0, 0.0, 10.0)
    # Fortran prints '*' for an f12.6 overflow -- must become nan, not shift fields.
    overflow = overflow[:53] + "*" * 12 + overflow[65:]
    path = tmp_path / "hypoDD.res"
    path.write_text(good + "\n" + overflow + "garbage line that is not data\n")

    rows = parse_hypodd_res(path)
    assert len(rows) == 2
    assert rows[0].data_type == "CT-P"
    assert rows[0].res_ms == pytest.approx(-4.5)
    assert rows[0].offset_m == pytest.approx(120.0)
    assert rows[1].data_type == "CC-S"
    assert math.isnan(rows[1].res_ms)
    assert rows[1].weight == pytest.approx(0.0)


def test_parse_hypodd_res_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_hypodd_res(tmp_path / "nope.res")


# ---------------------------------------------------------------------------
# dt.cc vs dt.ct
# ---------------------------------------------------------------------------


def _write_dt_files(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "dt.ct").write_text(
        "#         1         2\n"
        "STA1      3.430   3.420 1.0000 P\n"
        "STA1      5.900   5.880 1.0000 S\n"
        "STA2      4.000   4.000 1.0000 P\n"
        "#         3         4\n"
        "STA1      2.000   2.100 1.0000 P\n"
        "#         5         6\n"
        "STA1      1.000   1.000 1.0000 P\n"
    )
    (work_dir / "dt.cc").write_text(
        "#         1         2   0.000000\n"
        "STA1        0.012000 0.9500 P\n"  # diff +2 ms
        "STA1        0.015000 0.9000 S\n"  # diff -5 ms
        "STA9        0.000000 0.9000 P\n"  # no dt.ct counterpart
        "#         4         3   0.050000\n"  # swapped order, non-zero OTC
        "STA1        0.150000 0.8000 P\n"  # (0.15-0.05) - (+0.1) = 0
        "#         5         6  -999.000000\n"  # hypoDD skips: no OTC
        "STA1        0.500000 0.8000 P\n"
    )


def test_match_dtcc_dtct(tmp_path):
    _write_dt_files(tmp_path)
    df = match_dtcc_dtct(tmp_path / "dt.ct", tmp_path / "dt.cc")
    assert len(df) == 3
    by_key = {(r.cuspid1, r.cuspid2, r.phase): r for r in df.itertuples()}
    assert by_key[(1, 2, "P")].diff_ms == pytest.approx(2.0)
    assert by_key[(1, 2, "S")].diff_ms == pytest.approx(-5.0)
    swapped = by_key[(4, 3, "P")]
    assert swapped.pair_swapped
    assert swapped.otc_s == pytest.approx(0.05)
    assert swapped.dt_ct_s == pytest.approx(0.1)
    assert swapped.diff_ms == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# weighting sets
# ---------------------------------------------------------------------------


def _wrow(niter: int, wtccp: float = 1.0) -> WeightingRow:
    return WeightingRow(
        niter=niter, wtccp=wtccp, wtccs=0.5, wrcc=-999, wdcc=-999,
        wtctp=1.0, wtcts=0.5, wrct=-999, wdct=-999, damp=70,
    )  # fmt: skip


def _irow(it: int, updated: int | None) -> HypoddIterationRow:
    return HypoddIterationRow(
        iteration=it, ev_pct=100, ct_pct=100, cc_pct=100, rmsct_ms=10, rmscc_ms=5,
        rmsst_ms=8, dx_m=10, dy_m=10, dz_m=10, dt_ms=1, os_m=5, aq=0, cnd=10,
        updated_iteration=updated,
    )  # fmt: skip


def test_iteration_weighting_sets_air_quake_shifts_boundary():
    weights = [_wrow(2), _wrow(2)]
    # iteration 2 removed air-quakes (blank str3): sets shift by one.
    rows = [_irow(1, 1), _irow(2, None), _irow(3, 2), _irow(4, 3), _irow(5, 4), _irow(6, 5)]
    assert iteration_weighting_sets(rows, weights) == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# write_relocate_diagnostics gating / never-fail
# ---------------------------------------------------------------------------

_LOG_HEADER = (
    "  IT   EV  CT  CC    RMSCT      RMSCC   RMSST   DX   DY   DZ   DT   OS  AQ  CND\n"
    "        %   %   %   ms     %    ms     %    ms    m    m    m   ms    m\n"
)


def _write_hypodd_log(path: Path) -> None:
    lines = [_LOG_HEADER]
    updated = 0
    for it in range(1, 7):
        if it == 3:  # air-quake iteration: str3 blank
            lines.append(f"{it:4d}    100 100 100  90 -10  40 -5  50  80  60  50  30  2 1 20\n")
            continue
        updated += 1
        lines.append(
            f"{it:4d} {updated:3d} 100 100 100  90 -10  40 -5  50  {80 - it} 60 50 30  2 0 20\n"
        )
    path.write_text("".join(lines))


def _project(tmp_path, *, extended: bool | None, step_override: bool | None = None):
    cfg = load_config(REPO_ROOT / "config" / "example.yaml")
    cfg.project.output_dir = tmp_path / "run"
    if extended is not None:
        cfg.diagnostics.extended = extended
    if step_override is not None:
        cfg.diagnostics.relocate.extended = step_override
    paths = RunPaths(cfg)
    paths.ensure()

    cat = Catalog()
    event = make_event("ev20250101T000000_000")
    origin = Origin(time="2025-01-01T00:00:00Z", latitude=37.0, longitude=-122.0, depth=1000.0)
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id
    cat.append(event)
    write_catalog(cat, paths.catalog_mag_xml)
    write_catalog(cat, paths.catalog_dd_xml)
    paths.relocate_work_dir.mkdir(parents=True, exist_ok=True)
    return cfg, paths


def _write_all_inputs(paths: RunPaths) -> None:
    work = paths.relocate_work_dir
    _write_dt_files(work)
    (work / "hypoDD.res").write_text(
        _res_line("STA1", 0.01, 1, 2, 1, 0.95, 1.5, 0.9, 50.0)
        + _res_line("STA1", 0.01, 1, 2, 2, 0.9, -2.5, 0.0, 50.0)
        + _res_line("STA1", 0.01, 1, 2, 3, 1.0, 12.0, 1.0, 50.0)
        + _res_line("STA2", 0.01, 1, 2, 4, 1.0, -20.0, 0.5, 50.0)
    )
    _write_hypodd_log(work / "hypoDD.log")


def test_parsed_log_rows_carry_updated_iteration(tmp_path):
    _write_hypodd_log(tmp_path / "hypoDD.log")
    rows = parse_hypodd_log_iterations(tmp_path / "hypoDD.log")
    assert [r.iteration for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r.updated_iteration for r in rows] == [1, 2, None, 3, 4, 5]
    assert rows[3].dx_m == pytest.approx(76.0)  # 80 - it


def test_extended_disabled_writes_nothing(tmp_path):
    cfg, paths = _project(tmp_path, extended=False)
    _write_all_inputs(paths)
    result = write_relocate_diagnostics(paths, cfg)
    assert result.extended_written == []
    assert not (paths.diagnostics_dir("relocate") / "extended").exists()
    assert (paths.diagnostics_dir("relocate") / "summary.csv").is_file()


def test_extended_step_override_disables(tmp_path):
    cfg, paths = _project(tmp_path, extended=True, step_override=False)
    _write_all_inputs(paths)
    assert write_relocate_diagnostics(paths, cfg).extended_written == []


def test_extended_enabled_writes_all(tmp_path):
    cfg, paths = _project(tmp_path, extended=None, step_override=True)
    _write_all_inputs(paths)
    result = write_relocate_diagnostics(paths, cfg)
    out = paths.diagnostics_dir("relocate") / "extended"
    assert set(result.extended_written) == EXTENDED_FIGURES
    for name in EXTENDED_FIGURES:
        assert (out / name).is_file() and (out / name).stat().st_size > 0
    import pandas as pd

    csv = pd.read_csv(out / "dtcc_vs_dtct.csv")
    assert len(csv) == 3
    # Baseline set is unchanged by enabling the extended one.
    assert (paths.diagnostics_dir("relocate") / "iteration_rms.png").is_file()


def test_no_cfg_means_baseline_only(tmp_path):
    _cfg, paths = _project(tmp_path, extended=True)
    _write_all_inputs(paths)
    assert write_relocate_diagnostics(paths).extended_written == []


def test_extended_missing_inputs_skip_with_warning(tmp_path, caplog):
    cfg, paths = _project(tmp_path, extended=True)
    with caplog.at_level("WARNING", logger="openmeq"):
        written = write_relocate_extended_diagnostics(cfg, paths)
    assert written == []
    messages = " ".join(rec.message for rec in caplog.records)
    for name in ("dtcc_vs_dtct", "res_final_by_type", "iteration_shifts"):
        assert name in messages


def test_extended_figure_failure_does_not_stop_others(tmp_path, caplog, monkeypatch):
    cfg, paths = _project(tmp_path, extended=True)
    _write_all_inputs(paths)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr(ext, "plot_res_final_by_type", _boom)
    with caplog.at_level("WARNING", logger="openmeq"):
        result = write_relocate_diagnostics(paths, cfg)
    assert set(result.extended_written) == EXTENDED_FIGURES - {"res_final_by_type.png"}
    assert any("synthetic plotting failure" in rec.message for rec in caplog.records)


def test_extended_res_falls_back_to_cluster_dirs(tmp_path):
    cfg, paths = _project(tmp_path, extended=True)
    work = paths.relocate_work_dir
    for cid, idx in ((1, 1), (2, 3)):
        cdir = work / f"_cluster_{cid:04d}"
        cdir.mkdir()
        (cdir / "hypoDD.res").write_text(_res_line("STA1", 0.0, cid, 9, idx, 1, 1.0, 1.0, 1.0))
    assert ext._hypodd_res_paths(work) == sorted(work.glob("_cluster_*/hypoDD.res"))
    assert "res_final_by_type.png" in write_relocate_extended_diagnostics(cfg, paths)
