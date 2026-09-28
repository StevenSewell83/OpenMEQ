"""Shared fixtures for ``tests/pick/`` (M4.4, ``OpenMEQ-1m7.4``).

``pick_ready_project`` gives every test in ``tests/pick/`` a session-scoped
project with real ``openmeq detect``/``openmeq chunk`` output already on
disk (nothing ``pick``-related yet).

OpenMEQ-m1f.6: this used to run the real ``openmeq detect`` then ``openmeq
chunk`` CLI commands itself (~35s against the full synthetic dataset) into
its own output directory. It now instead **copies** those same real
outputs straight out of the shared, session-scoped ``full_pipeline_run``
fixture (``tests/conftest.py``) -- one real ``openmeq run`` chain computed
once per session -- rather than re-running detect+chunk for real a second
time. Every step's output lives entirely under its own top-level output
folder (``docs/ARCHITECTURE.md`` §3), so copying just ``inventory/``,
``qc/``, ``detections/``, ``chunks/`` and ``chunks_pn/`` (small, derived
per-event artifacts -- kilobytes to a few MB total, never the ~500 MB raw
archive) gives an equivalent "detect+chunk done" starting point. ``cfg``/
``config_path`` still resolve every *input* path (waveforms, station
metadata) into the shared, read-only ``synthetic_project`` directory, same
as before -- only those small already-computed output folders are copied
in from ``full_pipeline_run``'s own (separate) private project copy.

Session-scoped, not per-test: ``tests/pick/test_recovery.py`` and
``tests/pick/test_cli_pick.py`` both need a real, fully-chunked project to
run ``openmeq pick`` against, so computing/copying it once and sharing it
avoids paying that twice. Both of those test modules run ``openmeq pick``
itself (via ``CliRunner``) against this shared project -- whichever runs
first does the real, ~10s PhaseNet pass; the other's own ``openmeq pick``
invocation then hits ``StepManifest``'s skip-unchanged path (fast, and
still a valid exercise of "the CLI command produces the expected files",
since they're already on disk from the first run) -- this is safe because
neither test module mutates ``chunks_pn/``/``cfg.phasenet`` between them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from openmeq.config import Config, load_config
from openmeq.paths import RunPaths

#: The per-step output folders (ARCHITECTURE.md §3) that make up a
#: "detect+chunk done" project -- copied verbatim from the shared
#: ``full_pipeline_run`` fixture rather than recomputed.
_PRE_PICK_FOLDERS = ("inventory", "qc", "detections", "chunks", "chunks_pn")


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    See ``tests/integration/test_detect_integration.py``'s identical
    helper for the full rationale (duplicated here to avoid cross-test-
    module coupling).
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


@pytest.fixture(scope="session")
def pick_ready_project(synthetic_project, full_pipeline_run, tmp_path_factory):
    """``(cfg, paths, config_path, truth)``, with real ``detect``/``chunk``
    outputs already on disk (nothing ``pick``-related yet). Session-scoped
    -- see module docstring.
    """
    config_path, truth = synthetic_project
    _src_cfg, source_paths, _src_truth, _result = full_pipeline_run

    cfg = load_config(config_path)
    outdir = tmp_path_factory.mktemp("pick_ready_project")
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
