"""Shared fixtures for ``tests/pick/`` (M4.4, ``OpenMEQ-1m7.4``).

``pick_ready_project`` runs the real ``openmeq detect`` then ``openmeq
chunk`` (both via ``CliRunner``, per this task's own "Integration test via
CliRunner: qc, then detect, then chunk, then pick" instruction) against the
synthetic fixture, into one **session-scoped** output directory --
``openmeq qc`` itself is *not* re-run for real (~40s): its outputs
(``inventory/index.json``, ``qc/channel_flags.json``) are built directly
from the already-computed, session-scoped ``synthetic_qc_tables`` fixture,
exactly mirroring ``tests/integration/test_detect_integration.py``'s own
``detect_project`` fixture / ``_build_qc_outputs_from_tables`` helper
(duplicated here rather than imported, to avoid cross-test-module coupling
-- same convention ``tests/test_chunk_command.py`` already uses for its own
``_dump_config_copy``).

Session-scoped, not per-test: ``openmeq detect``/``openmeq chunk`` together
cost ~35s against the full synthetic dataset; ``tests/pick/test_recovery.py``
and ``tests/pick/test_cli_pick.py`` both need a real, fully-chunked project
to run ``openmeq pick`` against, so computing it once and sharing it avoids
paying that twice. Both of those test modules run ``openmeq pick`` itself
(via ``CliRunner``) against this shared project -- whichever runs first
does the real, ~10s PhaseNet pass; the other's own ``openmeq pick``
invocation then hits ``StepManifest``'s skip-unchanged path (fast, and
still a valid exercise of "the CLI command produces the expected files",
since they're already on disk from the first run) -- this is safe because
neither test module mutates ``chunks_pn/``/``cfg.phasenet`` between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from openmeq.cli import main
from openmeq.config import Config, load_config
from openmeq.paths import RunPaths
from openmeq.qc.flags import compute_channel_flags, write_channel_flags


def _dump_config_copy(cfg: Config, path: Path) -> None:
    """Write ``cfg`` to ``path`` as YAML, with every path already absolute.

    See ``tests/integration/test_detect_integration.py``'s identical
    helper for the full rationale (duplicated here to avoid cross-test-
    module coupling).
    """
    dumped = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(dumped, sort_keys=False))


@pytest.fixture(scope="session")
def pick_ready_project(synthetic_project, synthetic_qc_tables, tmp_path_factory):
    """``(cfg, paths, config_path, truth)``, with real ``detect``/``chunk``
    outputs already on disk (nothing ``pick``-related yet). Session-scoped
    -- see module docstring.
    """
    config_path, truth = synthetic_project
    _qc_cfg, idx, availability_df, daily_stats_df, monthly_df, _truth2 = synthetic_qc_tables

    cfg = load_config(config_path)
    outdir = tmp_path_factory.mktemp("pick_ready_project")
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
