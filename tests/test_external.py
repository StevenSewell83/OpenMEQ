"""Tests for openmeq.external."""

import logging
import os
import stat
import sys

import pytest

from openmeq.external import (
    ExternalToolError,
    ExternalToolMissing,
    bin_path,
    phasenet_dir,
    phasenet_model_dir,
    run,
)


@pytest.fixture
def external_root(tmp_path, monkeypatch):
    """A temp `external/` dir pointed to via OPENMEQ_EXTERNAL_DIR."""
    root = tmp_path / "external"
    (root / "bin").mkdir(parents=True)
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(root))
    return root


def test_bin_path_found_via_external_bin(external_root):
    tool = external_root / "bin" / "faketool"
    tool.write_text("#!/bin/sh\necho hi\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    found = bin_path("faketool")
    assert found == tool


def test_bin_path_missing_raises(external_root):
    with pytest.raises(ExternalToolMissing) as excinfo:
        bin_path("definitely_not_a_real_tool_xyz")
    assert "definitely_not_a_real_tool_xyz" in str(excinfo.value)


def test_bin_path_falls_back_to_path(external_root, monkeypatch, tmp_path):
    # Nothing under external/bin, but something on PATH.
    path_dir = tmp_path / "pathdir"
    path_dir.mkdir()
    tool = path_dir / "onpathtool"
    tool.write_text("#!/bin/sh\necho hi\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(path_dir) + os.pathsep + os.environ.get("PATH", ""))

    found = bin_path("onpathtool")
    assert found == tool


def test_phasenet_dir_and_model_dir(external_root):
    assert phasenet_dir() == external_root / "PhaseNet"
    assert phasenet_model_dir() == external_root / "PhaseNet" / "model" / "190703-214543"


def test_phasenet_model_dir_path(external_root):
    assert phasenet_model_dir() == phasenet_dir() / "model" / "190703-214543"


def test_run_success_logs_stdout(caplog):
    with caplog.at_level(logging.INFO, logger="openmeq.external"):
        proc = run([sys.executable, "-c", "print('hello from subprocess')"])
    assert proc.returncode == 0
    assert "hello from subprocess" in caplog.text


def test_run_logs_and_raises_on_nonzero_exit(caplog):
    logger = logging.getLogger("openmeq.external.test")
    with caplog.at_level(logging.ERROR, logger="openmeq.external.test"):
        with pytest.raises(ExternalToolError) as excinfo:
            run(
                [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
                log=logger,
            )
    assert excinfo.value.returncode == 3
    assert "boom" in excinfo.value.stderr
    assert "boom" in caplog.text


def test_run_timeout_raises_external_tool_error():
    with pytest.raises(ExternalToolError):
        run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.1)
