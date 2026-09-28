"""Smoke tests for the package skeleton itself (M0.1).

These guard the contract every other task depends on: the package installs,
every listed submodule imports cleanly, and the pytest markers used across
the test suite are registered so ``pytest --strict-markers`` never warns.
"""

import importlib
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import openmeq

REPO_ROOT = Path(__file__).resolve().parents[1]

SUBMODULES = [
    "openmeq.config",
    "openmeq.cli",
    "openmeq.index",
    "openmeq.stations",
    "openmeq.chunk",
    "openmeq.paths",
    "openmeq.log",
    "openmeq.parallel",
    "openmeq.manifest",
    "openmeq.external",
    "openmeq.catalog",
    "openmeq.qc",
    "openmeq.detect",
    "openmeq.pick",
    "openmeq.locate",
    "openmeq.review",
    "openmeq.magnitude",
    "openmeq.relocate",
    "openmeq.plotting",
]

EXPECTED_MARKERS = [
    "requires_nlloc",
    "requires_hypodd",
    "requires_phasenet",
    "requires_gui",
    "slow",
]


def test_version_is_importable():
    assert isinstance(openmeq.__version__, str)
    assert openmeq.__version__


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports_cleanly(name):
    module = importlib.import_module(name)
    assert module is not None


def test_pytest_markers_registered_in_pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    registered_names = [m.split(":", 1)[0].strip() for m in markers]

    for name in EXPECTED_MARKERS:
        assert name in registered_names
