"""Tests for ``scripts/setup_hypodd.py`` / ``openmeq.relocate.setup`` (M9.5 /
OpenMEQ-6ld.5).

Builds small, hand-rolled configs (a CSV-only station table, a tiny 1D
layer file, and -- for 3D -- a minimal ``vel3D.dat`` header) rather than the
heavy ``synthetic_project`` fixture (~500 MB): this task is orchestration
only, and none of its acceptance criteria need a realistic dataset.

``ensure_compiled_limits``/``recompile_hypodd`` are exercised against a
scratch ``.inc`` tree (never the main checkout's ``external/HYPODD``, per
``docs/ARCHITECTURE.md`` §6c's testing rule), and the recompile path itself
is mocked -- these tests are about *orchestration* (was the right function
called, in the right order, with the right arguments), not about
`hypodd_limits`'s own logic (covered by ``tests/test_relocate_limits.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from openmeq.config import ConfigError, load_config
from openmeq.paths import RunPaths
from openmeq.relocate.setup import setup_hypodd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import setup_hypodd as setup_hypodd_script  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"

_DEFAULT_STATIONS = [
    {
        "net": "XX",
        "sta": "STA01",
        "loc": "",
        "lat": 37.773,
        "lon": -122.274,
        "elev_m": 615.0,
        "depth_m": 0.0,
    },
    {
        "net": "XX",
        "sta": "STA02",
        "loc": "",
        "lat": 37.789,
        "lon": -122.220,
        "elev_m": 555.0,
        "depth_m": 0.0,
    },
]

# The 3D model origin used below matches config/example.yaml's own
# nlloc.grid.origin ([37.80, -122.20]) for consistency, and the domain span
# matches tests/test_relocate_limits.py's own `_SMALL_VEL3D_GRID` fixture
# (a real small-aperture-network-sized model), so `_DEFAULT_STATIONS` above
# fall inside it (`check_vel3d_coverage` must not raise).
_ORIGIN_3D = (37.80, -122.20)
_VEL3D_XN = (-80.0, 80.0)
_VEL3D_YN = (-80.0, 80.0)
_VEL3D_ZN = (-2.0, 10.0)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _write_stations_csv(path: Path, rows: list[dict] | None = None) -> Path:
    pd.DataFrame(rows if rows is not None else _DEFAULT_STATIONS).to_csv(path, index=False)
    return path


def _write_layer_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# depth_top_km  vp_km_s  vs_km_s  rho_g_cc\n"
        "0.0   3.50  2.00  2.3\n"
        "2.0   5.00  2.90  2.6\n"
        "5.0   6.00  3.40  2.7\n"
    )
    return path


def _write_minimal_vel3d_dat(
    path: Path,
    *,
    bld: float = 0.1,
    xn: tuple[float, ...] = _VEL3D_XN,
    yn: tuple[float, ...] = _VEL3D_YN,
    zn: tuple[float, ...] = _VEL3D_ZN,
) -> Path:
    """Just enough of a SIMUL2000 ``vel3D.dat`` for
    `openmeq.relocate.hypodd_files.read_vel3d_header` (header + node
    coordinate lines only -- no velocity blocks, which nothing in this test
    file's code path reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{bld:g} {len(xn)} {len(yn)} {len(zn)}  test\n",
        " ".join(f"{v:.2f}" for v in xn) + "\n",
        " ".join(f"{v:.2f}" for v in yn) + "\n",
        " ".join(f"{v:.2f}" for v in zn) + "\n",
    ]
    path.write_text("".join(lines))
    return path


# Generous fake .inc tree (mirrors tests/test_relocate_limits.py's own
# _write_fake_inc_tree -- never the main checkout's real external/HYPODD).
_FAKE_HYPODD_INC = """\
c hypoDD.inc: FAKE, tiny, for tests only.
	integer*4 MAXEVE, MAXLAY, MAXDATA, MAXSTA, MAXEVE0, MAXDATA0
	integer*4 MAXCL
	parameter(MAXEVE=  {maxeve},
     &          MAXDATA=  {maxdata},
     &          MAXEVE0=  {maxeve},
     &          MAXDATA0= {maxdata},
     &          MAXLAY=   30,
     &          MAXSTA=   {maxsta},
     &          MAXCL=    10)
"""

_FAKE_PH2DT_INC = """\
c ph2dt.inc: FAKE, tiny, for tests only.
	integer	MEV, MSTA, MOBS
	parameter(MEV=    {mev},
     &		MSTA=   {msta},
     &		MOBS=   {mobs})
"""

_FAKE_VEL3D_INC = """\
      integer	maxnx,maxny,maxnz,maxnz2
      parameter(maxnx= {maxnx})
      parameter(maxny= {maxny})
      parameter(maxnz= {maxnz})
      parameter(ixkms={ixkms})
      parameter(iykms={iykms})
      parameter(izkms={izkms})
"""


def _write_fake_inc_tree(root: Path) -> Path:
    include_dir = root / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "hypoDD.inc").write_text(
        _FAKE_HYPODD_INC.format(maxeve=1000, maxdata=100000, maxsta=500)
    )
    (include_dir / "ph2dt.inc").write_text(_FAKE_PH2DT_INC.format(mev=1000, msta=500, mobs=200))
    (include_dir / "vel3d.inc").write_text(
        _FAKE_VEL3D_INC.format(maxnx=25, maxny=25, maxnz=20, ixkms=10000, iykms=10000, izkms=10000)
    )
    return root


def _base_config_dict() -> dict:
    with open(EXAMPLE_CONFIG) as fh:
        return yaml.safe_load(fh)


def _write_config_yaml(
    tmp_path: Path,
    *,
    velocity_type: str = "1d",
    vel3d_path: Path | None = None,
    nlloc_grid_root: Path | None = None,
    layer_path: Path | None = None,
    stations_csv: Path | None = None,
) -> Path:
    """A real config.yaml on disk (round-tripped from config/example.yaml
    via a plain dict, so every *other* required field -- data/nlloc/phasenet/
    etc. -- keeps a valid, if unused, value) with `project.output_dir`,
    `stations` and `hypodd.velocity_model` overridden for this test.
    """
    data = _base_config_dict()
    data["project"]["output_dir"] = str(tmp_path / "run")

    stations_csv = stations_csv or _write_stations_csv(tmp_path / "stations.csv")
    data["stations"] = {"csv": str(stations_csv)}

    if velocity_type == "1d":
        layer_path = layer_path or _write_layer_file(tmp_path / "vel1d.txt")
        data["hypodd"]["velocity_model"] = {"type": "1d", "file": str(layer_path)}
    else:
        vel3d_path = vel3d_path if vel3d_path is not None else tmp_path / "vel3D.dat"
        data["hypodd"]["velocity_model"] = {
            "type": "3d",
            "file": str(vel3d_path),
            "origin_3d": list(_ORIGIN_3D),
            "rotation_3d_deg": 0.0,
            "simul2000": {"iphase": 1},
        }
        if nlloc_grid_root is not None:
            data["nlloc"]["velocity_model"] = {"type": "3d", "file": str(nlloc_grid_root)}

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(data, fh)
    return config_path


# ---------------------------------------------------------------------------
# test_runs_with_no_catalogue_yet
# ---------------------------------------------------------------------------


def test_runs_with_no_catalogue_yet(tmp_path):
    """No magnitude/catalog_mag.xml yet (day-one run, PLAN.md §9): the CLI
    still writes station.dat/ph2dt.inp/hypoDD.inp and warns, rather than
    crashing, about skipping the compiled-limit check."""
    config_path = _write_config_yaml(tmp_path, velocity_type="1d")

    runner = CliRunner()
    result = runner.invoke(setup_hypodd_script.main, ["-c", str(config_path)])
    failure_detail = result.output + (str(result.exception) if result.exception else "")
    assert result.exit_code == 0, failure_detail

    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    assert not paths.catalog_mag_xml.exists()
    work = paths.relocate_work_dir
    assert (work / "station.dat").is_file()
    assert (work / "ph2dt.inp").is_file()
    assert (work / "hypoDD.inp").is_file()
    assert (work / "event_id_map.csv").is_file()
    assert (work / "station_label_map.csv").is_file()

    log_text = paths.log_file("hypodd_setup").read_text()
    assert "No catalogue found" in log_text
    assert "skipping" in log_text.lower() or "skip" in log_text.lower()


# ---------------------------------------------------------------------------
# test_1d_vs_3d_dispatch
# ---------------------------------------------------------------------------


def test_1d_vs_3d_dispatch_1d(tmp_path):
    config_path = _write_config_yaml(tmp_path, velocity_type="1d")
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    result = setup_hypodd(cfg, paths)
    assert result.rebuilt is True
    assert result.imod == 1

    text = (paths.relocate_work_dir / "hypoDD.inp").read_text()
    lines = text.splitlines()
    # The IMOD data line is the bare "1"/"9" right after the "* IMOD:" comment.
    imod_comment_idx = next(i for i, line in enumerate(lines) if line.startswith("* IMOD:"))
    assert lines[imod_comment_idx + 1].strip() == "1"
    assert "1 = 1D" in lines[imod_comment_idx]


def test_1d_vs_3d_dispatch_3d(tmp_path):
    vel3d_path = tmp_path / "vel3D.dat"
    _write_minimal_vel3d_dat(vel3d_path)
    config_path = _write_config_yaml(tmp_path, velocity_type="3d", vel3d_path=vel3d_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")
    result = setup_hypodd(
        cfg,
        paths,
        external_hypodd_dir=fake_root,
        external_bin_dir=tmp_path / "bin",
    )
    assert result.rebuilt is True
    assert result.imod == 9

    text = (paths.relocate_work_dir / "hypoDD.inp").read_text()
    lines = text.splitlines()
    imod_comment_idx = next(i for i, line in enumerate(lines) if line.startswith("* IMOD:"))
    assert lines[imod_comment_idx + 1].strip() == "9"
    assert "9 = 3D" in lines[imod_comment_idx]
    assert "vel3D.dat" in text


# ---------------------------------------------------------------------------
# test_nlloc_to_simul_flag_invoked
# ---------------------------------------------------------------------------


def test_nlloc_to_simul_flag_invoked(tmp_path, monkeypatch):
    """--nlloc-to-simul calls M9.4's converter (mocked here) before the rest
    of the sequence -- the mock's own side effect writes the vel3D.dat that
    every later step (write_hypodd_inp_3d, ensure_compiled_limits,
    check_vel3d_coverage) then depends on, so a call-order bug (running the
    rest of the sequence first) would make this test fail downstream, not
    just at the mock assertion.
    """
    nlloc_root = tmp_path / "nlloc_grid" / "layer"
    nlloc_root.parent.mkdir(parents=True, exist_ok=True)
    (nlloc_root.parent / "layer.P.mod.hdr").write_text("fake nlloc P grid header (mocked reader)")

    vel3d_path = tmp_path / "vel3D.dat"
    config_path = _write_config_yaml(
        tmp_path, velocity_type="3d", vel3d_path=vel3d_path, nlloc_grid_root=nlloc_root
    )
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    call_order: list[str] = []

    mock_read_grid = Mock(side_effect=lambda hdr: call_order.append("read_nlloc_grid") or Mock())

    def _fake_convert(nlloc_grid, output_path, lat_3d, lon_3d, rot_3d, **kwargs):
        call_order.append("nlloc_grid_to_vel3d")
        _write_minimal_vel3d_dat(Path(output_path))

    mock_convert = Mock(side_effect=_fake_convert)

    monkeypatch.setattr("openmeq.relocate.setup.read_nlloc_grid", mock_read_grid)
    monkeypatch.setattr("openmeq.relocate.setup.nlloc_grid_to_vel3d", mock_convert)

    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")
    result = setup_hypodd(
        cfg,
        paths,
        nlloc_to_simul=True,
        external_hypodd_dir=fake_root,
        external_bin_dir=tmp_path / "bin",
    )

    mock_convert.assert_called_once()
    assert result.nlloc_to_simul_ran is True
    assert result.imod == 9
    assert vel3d_path.is_file()

    # The converter ran before station.dat/ph2dt.inp/hypoDD.inp were written.
    assert call_order == ["read_nlloc_grid", "nlloc_grid_to_vel3d"]
    station_dat_mtime = (paths.relocate_work_dir / "station.dat").stat().st_mtime
    assert vel3d_path.stat().st_mtime <= station_dat_mtime


def test_nlloc_to_simul_rejected_for_1d_config(tmp_path):
    config_path = _write_config_yaml(tmp_path, velocity_type="1d")
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    with pytest.raises(ConfigError, match="3D"):
        setup_hypodd(cfg, paths, nlloc_to_simul=True)


# ---------------------------------------------------------------------------
# test_idempotent_rerun
# ---------------------------------------------------------------------------


def test_idempotent_rerun(tmp_path, monkeypatch):
    config_path = _write_config_yaml(tmp_path, velocity_type="1d")
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    mock_recompile = Mock()
    monkeypatch.setattr("openmeq.relocate.hypodd_limits.recompile_hypodd", mock_recompile)

    fake_root = _write_fake_inc_tree(tmp_path / "HYPODD")
    result1 = setup_hypodd(
        cfg, paths, external_hypodd_dir=fake_root, external_bin_dir=tmp_path / "bin"
    )
    assert result1.rebuilt is True
    hypodd_inp_path = paths.relocate_work_dir / "hypoDD.inp"
    mtime1 = hypodd_inp_path.stat().st_mtime
    station_dat_mtime1 = (paths.relocate_work_dir / "station.dat").stat().st_mtime

    result2 = setup_hypodd(
        cfg, paths, external_hypodd_dir=fake_root, external_bin_dir=tmp_path / "bin"
    )
    assert result2.rebuilt is False

    mock_recompile.assert_not_called()
    assert hypodd_inp_path.stat().st_mtime == mtime1
    assert (paths.relocate_work_dir / "station.dat").stat().st_mtime == station_dat_mtime1

    # --force always redoes the work, even though nothing changed.
    result3 = setup_hypodd(
        cfg,
        paths,
        force=True,
        external_hypodd_dir=fake_root,
        external_bin_dir=tmp_path / "bin",
    )
    assert result3.rebuilt is True


# ---------------------------------------------------------------------------
# test_clear_failure_on_missing_velocity_model_file
# ---------------------------------------------------------------------------


def test_clear_failure_on_missing_velocity_model_file(tmp_path):
    missing_layer = tmp_path / "does_not_exist.txt"
    config_path = _write_config_yaml(tmp_path, velocity_type="1d", layer_path=missing_layer)

    runner = CliRunner()
    result = runner.invoke(setup_hypodd_script.main, ["-c", str(config_path)])

    assert result.exit_code != 0
    message = str(result.exception)
    assert "does_not_exist.txt" in message


def test_clear_failure_on_missing_3d_velocity_model_file(tmp_path):
    missing_vel3d = tmp_path / "does_not_exist_vel3D.dat"
    config_path = _write_config_yaml(tmp_path, velocity_type="3d", vel3d_path=missing_vel3d)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    with pytest.raises(ConfigError, match="does not exist"):
        setup_hypodd(cfg, paths)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_setup_hypodd_script_help_exits_0():
    runner = CliRunner()
    result = runner.invoke(setup_hypodd_script.main, ["--help"])
    assert result.exit_code == 0
    assert "--nlloc-to-simul" in result.output
    assert "--force" in result.output
    assert "--config" in result.output
