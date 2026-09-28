"""Session-scoped fixtures for tests/locate.

``nlloc_grids`` (M6.1 / OpenMEQ-xdf.1) builds the synthetic network's NLLoc
grids exactly once per test session (``openmeq.nlloc_setup.setup_nlloc``),
so every test in this directory that calls
``openmeq.locate.nlloc_runner.locate_event`` reuses the same grids instead
of rebuilding them per test.

Unlike ``tests/test_nlloc_setup.py``'s per-test ``shutil.copytree`` of the
whole ~500 MB ``synthetic_project`` directory (necessary there because
those tests exercise ``build_1d_grid``/``run_grid2time`` themselves and
must not mutate the shared session directory), this fixture never touches
``synthetic_project``'s own directory at all: ``setup_nlloc`` only *reads*
small inputs from it (the 1D layer file, the station table) and writes the
NLLoc grids under ``cfg.project.output_dir``, which this fixture points at
a fresh, separate ``tmp_path_factory`` session directory instead of the
synthetic project's own ``./runs/synthetic``. No copy is needed, and the
grids this network/config produce are small (a few MB), so building them
once per session costs nothing worth avoiding further.

``chunked_project`` (M6.5 / OpenMEQ-xdf.5) extends the same idea one step
further, for ``tests/locate/test_locate_recovery.py``: a session-scoped project
with real ``openmeq detect``/``openmeq chunk`` outputs already on disk, so
that test only needs to run the (phasenet-dependent) ``pick``/``locate``
steps itself, inside its own skip-guarded test body.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from openmeq.cli import main
from openmeq.config import Config, load_config
from openmeq.nlloc_setup import setup_nlloc
from openmeq.paths import RunPaths
from openmeq.qc.flags import compute_channel_flags, write_channel_flags
from openmeq.stations import station_table as load_station_table


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    Mirrors ``tests/pick/conftest.py``'s identical helper -- duplicated
    here (not imported) to avoid cross-test-module coupling, the same
    convention that module's own docstring documents.
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


@pytest.fixture(scope="session")
def chunked_project(synthetic_project, synthetic_qc_tables, tmp_path_factory):
    """``(cfg, paths, config_path, truth)``, with real ``detect``/``chunk``
    outputs already on disk (nothing ``pick``/``locate``-related yet).

    Session-scoped, phasenet-independent (mirrors ``tests/pick/conftest.py``
    ``pick_ready_project``'s own identical fixture and rationale, duplicated
    -- not imported -- since pytest fixtures are not visible across sibling
    test directories): ``openmeq detect``/``openmeq chunk`` together cost
    ~35s against the full synthetic dataset, and
    ``tests/locate/test_locate_recovery.py`` is the only consumer here, but this
    still avoids re-running them if a second locate-recovery-style test is
    added later. ``openmeq qc`` itself is *not* re-run for real (~40s): its
    outputs are built directly from the already-computed, session-scoped
    ``synthetic_qc_tables`` fixture.
    """
    config_path, truth = synthetic_project
    _qc_cfg, idx, availability_df, daily_stats_df, monthly_df, _truth2 = synthetic_qc_tables

    cfg = load_config(config_path)
    outdir = tmp_path_factory.mktemp("chunked_project")
    cfg.project.output_dir = outdir / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    idx.write(paths.index_json)
    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    write_channel_flags(flags, paths.channel_flags_json)

    new_config_path = outdir / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

    runner = CliRunner()
    detect_result = runner.invoke(main, ["detect", "-c", str(new_config_path)])
    assert detect_result.exit_code == 0, detect_result.output

    chunk_result = runner.invoke(main, ["chunk", "-c", str(new_config_path)])
    assert chunk_result.exit_code == 0, chunk_result.output

    return cfg, paths, new_config_path, truth


@pytest.fixture(scope="session")
def nlloc_grids(synthetic_project, tmp_path_factory):
    """``(cfg, paths, truth, station_table)`` with this session's shared NLLoc
    grids already built under a fresh output directory.

    ``cfg`` is loaded from the shared ``synthetic_project`` fixture's
    config file (so every input path -- station metadata, 1D velocity
    model -- resolves into that read-only shared directory), but
    ``cfg.project.output_dir`` is overridden to this fixture's own
    session-scoped temp directory before anything is written.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path_factory.mktemp("nlloc_runner_run")

    paths = RunPaths(cfg)
    paths.ensure()
    setup_nlloc(cfg, paths)

    table = load_station_table(cfg.stations)
    return cfg, paths, truth, table
