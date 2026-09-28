"""Shared pytest fixtures.

``synthetic_project`` (M0.8) builds the deterministic synthetic test
dataset described in ``docs/ARCHITECTURE.md`` §8 and ``tests/data/make_synthetic.py``'s
module docstring, once per test session, and hands every test
``(config_path, truth)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tests/data/make_synthetic.py is a standalone script/module, not part of
# the installed openmeq package (ARCHITECTURE.md §8 / the M0.8 task scopes
# it under tests/data/, not openmeq/) -- import it by adding tests/data to
# sys.path rather than assuming tests/ is an importable package (it has no
# __init__.py, matching the rest of this test suite).
sys.path.insert(0, str(Path(__file__).parent / "data"))
import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory) -> tuple[Path, dict]:
    """The session-wide synthetic test dataset: ``(config_path, truth)``.

    Generates the dataset described in ``tests/data/make_synthetic.py``
    (~20 events, 8 stations, 2 days at 200 Hz, one gap/spike/flat anomaly
    each) **once** into a ``tmp_path_factory`` session temp directory, and
    returns the path to its generated ``config.yaml`` (loadable via
    ``openmeq.config.load_config``) alongside the ``truth`` structure (see
    ``make_synthetic``'s module docstring for its exact shape: per-event
    ``event_id_hint``/``origin``/``ml``/``arrivals``, plus
    ``truth["known_issues"]`` for the gap/spike/flat locations).

    **Usage pattern -- read before writing a test against this fixture**:
    this is session-scoped, so the same on-disk project directory is shared
    by every test in the run. A test that only *reads* the data (parses
    the config, indexes/reads waveforms, checks ``truth``) can use the
    fixture directly. A test that exercises a pipeline step -- which
    writes into ``cfg.project.output_dir``, or otherwise mutates files
    under the project directory -- must instead copy the whole project
    directory (``config_path.parent``) into its own function-scoped
    ``tmp_path`` first, and load a config pointed at *that copy*:

    .. code-block:: python

        def test_some_step(synthetic_project, tmp_path):
            config_path, truth = synthetic_project
            project_copy = tmp_path / "project"
            shutil.copytree(config_path.parent, project_copy)
            cfg = load_config(project_copy / "config.yaml")
            ...

    Mutating the shared session directory directly will corrupt it for
    every other test in the same run.
    """
    outdir = tmp_path_factory.mktemp("synthetic_project")
    truth = make_synthetic.generate(outdir)
    config_path = outdir / "config.yaml"
    return config_path, truth
