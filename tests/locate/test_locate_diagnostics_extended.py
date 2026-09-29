"""OpenMEQ-02o.5: ``openmeq.locate.diagnostics_extended`` (locate extended QAQC).

Fast, NLLoc-free tests: a hand-built
:class:`~openmeq.locate.locate_step.LocateRunResult` (plain-data
``PassDiagnostics`` records, so no location is ever run) fed through the
real :func:`openmeq.locate.diagnostics.make_locate_diagnostics`, checking
the ``diagnostics.locate`` gate (extended dir present only when enabled),
the per-station residual CSV's contents, and that empty passes / missing
inputs still produce every file without raising. The end-to-end
``distances_km`` population from real NLLoc arrivals is covered in
``tests/locate/test_locate_diagnostics.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from openmeq.config import (
    Config,
    DataConfig,
    HypoddConfig,
    HypoddVelocityModelConfig,
    NllocConfig,
    NllocGridConfig,
    NllocVelocityModelConfig,
    ParallelConfig,
    ProjectConfig,
    StationsConfig,
)
from openmeq.locate import diagnostics_extended as ext
from openmeq.locate.diagnostics import make_locate_diagnostics
from openmeq.locate.locate_step import LocateRunResult, PassDiagnostics
from openmeq.paths import RunPaths

_STATIONS = [("XX", "STA1", 0.00, 0.00), ("XX", "STA2", 0.05, 0.00), ("XX", "STA3", 0.00, 0.05)]

_EXTENDED_FILES = [
    f"pass{p}_{name}"
    for p in ("A", "B")
    for name in (
        "res_vs_distance.png",
        "station_res_map.png",
        "station_residuals.csv",
        "depth_hist_grid.png",
    )
]


def _make_cfg(tmp_path: Path, *, extended: bool, model_type: str = "1d") -> Config:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stations_csv = tmp_path / "stations.csv"
    pd.DataFrame(
        [
            {"net": n, "sta": s, "loc": "00", "lat": lat, "lon": lon, "elev_m": 0.0, "depth_m": 0.0}
            for n, s, lat, lon in _STATIONS
        ]
    ).to_csv(stations_csv, index=False)
    vel = tmp_path / "vel.txt"
    vel.write_text("0.0 3.5 2.0\n1.0 4.5 2.6\n3.0 5.8 3.35\n")
    cfg = Config(
        project=ProjectConfig(name="test", output_dir=tmp_path / "run"),
        parallel=ParallelConfig(n_workers=1),
        data=DataConfig(waveform_dirs=[tmp_path], start=date(2025, 1, 1), end=date(2025, 1, 2)),
        stations=StationsConfig(csv=stations_csv),
        nlloc=NllocConfig(
            velocity_model=NllocVelocityModelConfig(type=model_type, file=vel),
            grid=NllocGridConfig(origin=(0.0, 0.0), size_km=(10.0, 10.0, 8.0), spacing_km=0.5),
        ),
        hypodd=HypoddConfig(velocity_model=HypoddVelocityModelConfig(type="1d", file=vel)),
    )
    cfg.diagnostics.locate.extended = extended
    return cfg


def _diag(
    event_id: str, *, ran: bool = True, depth_km: float | None = 2.0, **kw
) -> PassDiagnostics:
    return PassDiagnostics(
        event_id=event_id,
        ran=ran,
        status="ok" if ran else None,
        reason=None,
        n_iterations=1 if ran else 0,
        stabilized=True if ran else None,
        time=None,
        lat=0.02 if ran else None,
        lon=0.02 if ran else None,
        depth_km=depth_km if ran else None,
        rms=0.05 if ran else None,
        gap=90.0 if ran else None,
        err_h_km=0.5 if ran else None,
        err_z_km=0.8 if ran else None,
        **kw,
    )


def _populated_result() -> LocateRunResult:
    """Three events; STA1 P residuals +0.1/+0.2/+0.3, STA2 P -0.1 x3, S on
    STA1 only (pass B); STA3 never has a residual."""
    pass_a, pass_b = {}, {}
    for i, (r1, depth) in enumerate([(0.1, 1.0), (0.2, 2.9), (0.3, 7.9)]):
        eid = f"ev{i}"
        residuals = {"XX.STA1.00.HHZ": r1, "XX.STA2.00.HHZ": -0.1}
        phases = {"XX.STA1.00.HHZ": "P", "XX.STA2.00.HHZ": "P", "XX.STA1.00.HHN": "S"}
        dists = {"XX.STA1.00.HHZ": 2.0 + i, "XX.STA2.00.HHZ": 5.0 + i, "XX.STA1.00.HHN": 2.0 + i}
        pass_a[eid] = _diag(
            eid,
            depth_km=depth,
            residuals=residuals,
            phase_by_seed_id=phases,
            distances_km=dists,
        )
        pass_b[eid] = _diag(
            eid,
            depth_km=depth,
            residuals={**residuals, "XX.STA1.00.HHN": 0.05 * (i + 1)},
            phase_by_seed_id=phases,
            distances_km=dists,
        )
    return LocateRunResult(
        events=[], summary_rows=[], removal_records=[], pass_a=pass_a, pass_b=pass_b
    )


def _run(cfg: Config, result: LocateRunResult) -> Path:
    paths = RunPaths(cfg)
    paths.ensure()
    make_locate_diagnostics(paths, cfg, result)
    return ext.extended_dir(paths)


def test_extended_disabled_writes_no_extended_dir(tmp_path):
    cfg = _make_cfg(tmp_path, extended=False)
    out = _run(cfg, _populated_result())
    assert not out.exists()
    assert (out.parent / "passA_residuals_P.png").is_file()  # baseline unaffected


def test_extended_enabled_writes_figures_and_csv(tmp_path):
    cfg = _make_cfg(tmp_path, extended=True)
    out = _run(cfg, _populated_result())
    for name in _EXTENDED_FILES:
        assert (out / name).is_file(), name
        assert (out / name).stat().st_size > 0

    table = pd.read_csv(out / "passB_station_residuals.csv")
    assert list(table.columns) == ext.STATION_RESIDUAL_COLUMNS
    rows = {(r.station, r.phase): r for r in table.itertuples(index=False)}
    assert set(rows) == {("XX.STA1", "P"), ("XX.STA2", "P"), ("XX.STA1", "S")}
    sta1_p = rows[("XX.STA1", "P")]
    assert sta1_p.n == 3
    assert sta1_p.mean == pytest.approx(0.2)
    assert sta1_p.median == pytest.approx(0.2)
    assert sta1_p.std == pytest.approx(0.1)
    assert rows[("XX.STA2", "P")].mean == pytest.approx(-0.1)
    assert rows[("XX.STA1", "S")].mean == pytest.approx(0.1)

    # Pass A carries no S residuals here (its residuals dict has no S seed id).
    table_a = pd.read_csv(out / "passA_station_residuals.csv")
    assert set(table_a["phase"]) == {"P"}


def test_extended_enabled_by_global_switch(tmp_path):
    cfg = _make_cfg(tmp_path, extended=False)
    cfg.diagnostics.locate.extended = None
    cfg.diagnostics.extended = True
    assert _run(cfg, _populated_result()).is_dir()


def test_extended_empty_passes_do_not_raise(tmp_path):
    """No events at all, then an event pass B never ran for: every file is
    still written (header-only CSVs, "No data" figures)."""
    cfg = _make_cfg(tmp_path, extended=True)
    empty = LocateRunResult(events=[], summary_rows=[], removal_records=[], pass_a={}, pass_b={})
    out = _run(cfg, empty)
    for name in _EXTENDED_FILES:
        assert (out / name).is_file(), name
    assert pd.read_csv(out / "passA_station_residuals.csv").empty

    not_run = LocateRunResult(
        events=[],
        summary_rows=[],
        removal_records=[],
        pass_a={"ev0": _diag("ev0")},  # ran, but no residuals/distances at all
        pass_b={"ev0": _diag("ev0", ran=False)},
    )
    out = _run(cfg, not_run)
    for name in _EXTENDED_FILES:
        assert (out / name).is_file(), name


def test_extended_survives_missing_model_and_stations(tmp_path):
    """Unreadable velocity model and station table: warnings only, files
    still written; a 3D model also gets grid limits only."""
    cfg = _make_cfg(tmp_path, extended=True)
    cfg.nlloc.velocity_model.file.unlink()
    cfg.stations.csv.unlink()
    out = _run(cfg, _populated_result())
    for name in _EXTENDED_FILES:
        assert (out / name).is_file(), name

    cfg3d = _make_cfg(tmp_path / "3d", extended=True, model_type="3d")
    z_top, z_bottom, interfaces = ext._grid_and_interfaces(cfg3d, None)
    assert interfaces == []
    assert z_top is not None and z_top < 0.0 < z_bottom


def test_grid_and_interfaces_1d(tmp_path):
    cfg = _make_cfg(tmp_path, extended=True)
    z_top, z_bottom, interfaces = ext._grid_and_interfaces(cfg, None)
    assert interfaces == [1.0, 3.0]
    assert z_bottom == pytest.approx(8.0)
    assert z_top < 0.0


def test_one_figure_failing_does_not_stop_others(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ext, "plot_res_vs_distance", boom)
    cfg = _make_cfg(tmp_path, extended=True)
    out = _run(cfg, _populated_result())
    assert not (out / "passA_res_vs_distance.png").exists()
    assert (out / "passA_station_res_map.png").is_file()
    assert (out / "passB_depth_hist_grid.png").is_file()
