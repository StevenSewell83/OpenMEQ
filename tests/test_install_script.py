"""Tests for install.sh (M0.2).

Fast, static checks run under plain ``pytest`` on every CI run: the script
exists, is executable, is syntactically valid bash, and mentions the tools/
URLs it's supposed to install/build. These do not touch the network or build
anything, so they're cheap and safe to run anywhere.

A single end-to-end test that actually runs ``./install.sh`` is marked
``@pytest.mark.slow`` and is additionally gated behind the
``OPENMEQ_TEST_INSTALL_SH`` environment variable, since it clones PhaseNet/
NonLinLoc from the network, compiles NonLinLoc and hypoDD, and downloads a
~500 MB TensorFlow wheel -- far too slow/heavy for a normal test run.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"


def test_install_sh_exists():
    assert INSTALL_SH.is_file()


def test_install_sh_is_executable():
    mode = INSTALL_SH.stat().st_mode
    assert mode & stat.S_IXUSR, "install.sh must be chmod +x"


def test_install_sh_bash_syntax_ok():
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def install_sh_text():
    return INSTALL_SH.read_text()


@pytest.mark.parametrize(
    "pattern",
    [
        r"set\s+-euo\s+pipefail",
        r"uv\s+venv\s+--python\s+3\.10",
        r"github\.com/AI4EPS/PhaseNet",
        r"190703-214543",
        r"tensorflow==2\.11\.0",
        r"github\.com/alomax/NonLinLoc",
        r"\bcmake\b",
        r"\bNLLoc\b",
        r"\bVel2Grid\b",
        r"\bVel2Grid3D\b",
        r"\bGrid2Time\b",
        r"HYPODD\.zip",
        r"\bhypoDD\b",
        r"\bph2dt\b",
        r"--skip-external",
        r"external/bin",
    ],
)
def test_install_sh_mentions_expected_tools(install_sh_text, pattern):
    assert re.search(pattern, install_sh_text), f"install.sh missing expected pattern: {pattern}"


def test_install_sh_pins_phasenet_commit_sha(install_sh_text):
    """The PhaseNet clone must be pinned to a specific commit, not a branch HEAD."""
    match = re.search(r'PHASENET_COMMIT="([0-9a-f]{40})"', install_sh_text)
    assert match, "install.sh must pin PHASENET_COMMIT to a 40-char commit SHA"


def test_install_sh_pins_nlloc_commit_sha(install_sh_text):
    match = re.search(r'NLLOC_COMMIT="([0-9a-f]{40})"', install_sh_text)
    assert match, "install.sh must pin NLLOC_COMMIT to a 40-char commit SHA"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENMEQ_TEST_INSTALL_SH"),
    reason="set OPENMEQ_TEST_INSTALL_SH=1 to run the real (slow, network-using) install.sh",
)
def test_install_sh_runs_end_to_end():
    """Runs the real installer and checks it leaves the documented artifacts.

    Not run by default: clones PhaseNet/NonLinLoc, downloads TensorFlow, and
    compiles NonLinLoc + hypoDD. Opt in with:

        OPENMEQ_TEST_INSTALL_SH=1 pytest -m slow tests/test_install_script.py
    """
    result = subprocess.run(
        [str(INSTALL_SH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    assert venv_python.is_file()

    assert (REPO_ROOT / "external" / "PhaseNet" / "model" / "190703-214543").is_dir()

    bin_dir = REPO_ROOT / "external" / "bin"
    for name in ("NLLoc", "Vel2Grid", "Vel2Grid3D", "Grid2Time", "hypoDD", "ph2dt"):
        binary = bin_dir / name
        assert binary.is_file(), f"missing {binary}"
        assert os.access(binary, os.X_OK), f"{binary} is not executable"

    reloc = REPO_ROOT / "external" / "HYPODD" / "examples" / "example1" / "hypoDD.reloc"
    assert reloc.is_file() and reloc.stat().st_size > 0
