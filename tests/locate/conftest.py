"""Session-scoped NLLoc grid fixture for tests/locate (M6.1 / OpenMEQ-xdf.1).

Builds the synthetic network's NLLoc grids exactly once per test session
(``openmeq.nlloc_setup.setup_nlloc``), so every test in this directory that
calls ``openmeq.locate.nlloc_runner.locate_event`` reuses the same grids
instead of rebuilding them per test.

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
"""

from __future__ import annotations

import pytest

from openmeq.config import load_config
from openmeq.nlloc_setup import setup_nlloc
from openmeq.paths import RunPaths
from openmeq.stations import station_table as load_station_table


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
