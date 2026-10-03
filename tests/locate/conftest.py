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

OpenMEQ-m1f.6: ``chunked_project`` used to run the real ``openmeq detect``
then ``openmeq chunk`` CLI commands itself (~35s). It now instead copies
those same real outputs out of the shared, session-scoped
``full_pipeline_run`` fixture (``tests/conftest.py``) -- mirrors
``tests/pick/conftest.py``'s ``pick_ready_project``, duplicated (not
imported) since pytest fixtures aren't visible across sibling test
directories; see that module's own docstring for the full rationale.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from openmeq.config import Config, load_config
from openmeq.nlloc_setup import setup_nlloc
from openmeq.paths import SETUP_DIR, STEP_DIRS, RunPaths
from openmeq.stations import station_table as load_station_table

#: The per-step output folders (ARCHITECTURE.md §3) that make up a
#: "detect+chunk done" project -- copied verbatim from the shared
#: ``full_pipeline_run`` fixture rather than recomputed (mirrors
#: ``tests/pick/conftest.py``'s identical constant).
#: Relative to the run root, so they follow ``openmeq.paths``' numbered layout.
_PRE_PICK_FOLDERS = (
    SETUP_DIR,
    STEP_DIRS["qc"],
    STEP_DIRS["detections"],
    STEP_DIRS["chunks"],
    STEP_DIRS["chunks_pn"],
)


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    Mirrors ``tests/pick/conftest.py``'s identical helper -- duplicated
    here (not imported) to avoid cross-test-module coupling, the same
    convention that module's own docstring documents.
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


@pytest.fixture(scope="session")
def chunked_project(synthetic_project, full_pipeline_run, tmp_path_factory):
    """``(cfg, paths, config_path, truth)``, with real ``detect``/``chunk``
    outputs already on disk (nothing ``pick``/``locate``-related yet).

    Session-scoped: ``tests/locate/test_locate_recovery.py`` is the only
    consumer here, but this still avoids re-running detect+chunk if a
    second locate-recovery-style test is added later. See module docstring
    for why this copies rather than re-runs detect+chunk.
    """
    config_path, truth = synthetic_project
    _src_cfg, source_paths, _src_truth, _result = full_pipeline_run

    cfg = load_config(config_path)
    outdir = tmp_path_factory.mktemp("chunked_project")
    cfg.project.output_dir = outdir / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    for folder in _PRE_PICK_FOLDERS:
        dest = paths.root / folder
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(source_paths.root / folder, dest)

    new_config_path = outdir / "config.yaml"
    _dump_config_copy(cfg, new_config_path)

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
