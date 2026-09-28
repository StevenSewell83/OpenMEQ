"""External tool discovery and subprocess helpers (PhaseNet, NonLinLoc, hypoDD).

This module is the single place every step that shells out to PhaseNet/
NLLoc/hypoDD/ph2dt goes through (ARCHITECTURE.md §7): it locates the
``external/`` folder, resolves tool binaries within it (falling back to
``PATH``), and runs subprocesses with consistent logging and error handling.

Does not import ``openmeq.config`` or ``openmeq.log`` (built concurrently);
uses the stdlib ``logging`` module directly, and `run` accepts an optional
logger parameter.

Implemented in M0.7.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import openmeq

_MODULE_LOGGER = logging.getLogger(__name__)

#: Env var overriding where the `external/` folder is found. Officially
#: supported for test/CI/non-standard layouts, and the escape hatch for any
#: install layout (e.g. a future non-editable install) where the
#: `__file__`-relative default below doesn't hold.
EXTERNAL_DIR_ENV = "OPENMEQ_EXTERNAL_DIR"

_PHASENET_MODEL_VERSION = "190703-214543"


class ExternalToolMissing(FileNotFoundError):
    """Raised when an external tool cannot be found in `external/bin` or PATH."""


class ExternalToolError(RuntimeError):
    """Raised by `run` when a subprocess exits non-zero or times out.

    Attributes
    ----------
    cmd : list[str]
        The command that was run.
    returncode : int or None
        The process's exit code, or `None` if it timed out.
    stdout : str
        Captured standard output (possibly empty).
    stderr : str
        Captured standard error (possibly empty).
    """

    def __init__(self, message: str, cmd, returncode, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def external_dir() -> Path:
    """Resolve the `external/` directory.

    Resolution order (ARCHITECTURE.md §2/§7):

    1. ``$OPENMEQ_EXTERNAL_DIR``, if set.
    2. ``<repo root>/external``, where the repo root is taken as
       ``Path(openmeq.__file__).parents[1]``. This holds for the standard
       ``pip install -e .`` layout (flat `openmeq/` package at the repo
       root, ARCHITECTURE.md §1) used throughout this project, but would
       break under a non-editable install where the package is copied
       elsewhere without `external/` alongside it — in that case, set
       ``$OPENMEQ_EXTERNAL_DIR`` explicitly. Tests should always set the env
       var to point at a temporary fixture rather than relying on this
       default or the real `external/` directory being present.

    Returns
    -------
    pathlib.Path
    """
    env_val = os.environ.get(EXTERNAL_DIR_ENV)
    if env_val:
        return Path(env_val)
    return Path(openmeq.__file__).resolve().parents[1] / "external"


def bin_path(name: str) -> Path:
    """Locate an external tool binary.

    Looks in ``<external_dir()>/bin/<name>`` first, then falls back to
    ``shutil.which(name)`` (i.e. `PATH`).

    Parameters
    ----------
    name : str
        The executable's name, e.g. ``"NLLoc"`` or ``"hypoDD"``.

    Returns
    -------
    pathlib.Path
        The resolved path to the executable.

    Raises
    ------
    ExternalToolMissing
        If `name` isn't found in `external/bin` or on `PATH`. The message
        includes the tool name and every path checked.
    """
    local = external_dir() / "bin" / name
    if local.is_file() and os.access(local, os.X_OK):
        return local

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)

    raise ExternalToolMissing(
        f"External tool {name!r} not found: checked {local} and PATH ({os.environ.get('PATH', '')})"
    )


def phasenet_dir() -> Path:
    """Return the PhaseNet installation directory (``external/PhaseNet``)."""
    return external_dir() / "PhaseNet"


def phasenet_model_dir() -> Path:
    """Return the pinned PhaseNet model directory.

    Returns
    -------
    pathlib.Path
        ``<phasenet_dir()>/model/190703-214543``.
    """
    return phasenet_dir() / "model" / _PHASENET_MODEL_VERSION


def run(
    cmd: list[str],
    cwd: Path | None = None,
    log: logging.Logger | None = None,
    timeout: float | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess, logging its output, raising on failure.

    Output is captured (not streamed live) and then logged in one shot per
    stream, which is simpler and sufficient for the batch tools this wraps.
    Decoded with ``errors="replace"`` (added M10.4 / OpenMEQ-16d.4): the
    installed ``hypoDD`` binary was observed, empirically, to occasionally
    emit a single non-UTF-8 byte on stdout (immediately after its startup
    timestamp line -- most likely an uninitialised-memory artefact in its
    Fortran ``write`` formatting, not a real encoding), which crashed a
    plain ``text=True`` decode nondeterministically (same input, same
    binary, different runs). Replacing an occasional stray byte with the
    Unicode replacement character only affects a cosmetic log line and
    never changes `run`'s success/failure semantics (exit code, or the
    exception raised) -- a purely defensive addition, backward compatible
    with every existing caller:
    stdout is logged at ``INFO`` level, and stderr at ``WARNING`` level on
    success or ``ERROR`` level on a non-zero exit. On a non-zero exit (or a
    timeout), an `ExternalToolError` is raised with the captured output
    attached, so every caller has one exception type to handle instead of
    inspecting a `CompletedProcess` return value themselves.

    Parameters
    ----------
    cmd : list[str]
        The command and its arguments.
    cwd : pathlib.Path or None, optional
        Working directory for the subprocess.
    log : logging.Logger or None, optional
        Logger to send output through. Defaults to this module's logger
        (``openmeq.external``) if not given — `openmeq.log`'s
        `configure_logging`/`get_logger` are not imported here (built
        concurrently), so callers running under the CLI should pass their
        own step logger explicitly to get it routed to the step's log file.
    timeout : float or None, optional
        Timeout in seconds. A `subprocess.TimeoutExpired` is caught and
        re-raised as an `ExternalToolError` (same exception family as a
        non-zero exit), rather than left as a separate exception type.
    env : dict or None, optional
        Extra/overriding environment variables for the subprocess, merged
        over a copy of the parent process's ``os.environ`` (so a caller only
        needs to name the variable(s) it wants to change, e.g.
        ``env={"CUDA_VISIBLE_DEVICES": ""}``, without losing `PATH` or
        anything else the subprocess needs). ``None`` (the default) leaves
        the subprocess's environment exactly as `subprocess.run` would with
        no `env` argument at all (inherits the parent's `os.environ`
        unchanged) — this is a one-off override for a single call, not a
        replacement for mutating `os.environ` globally. Added in M4.3 /
        ``OpenMEQ-1m7.3``; see ``docs/phasenet_interface.md`` §1 for the gap
        this closes (forcing `CUDA_VISIBLE_DEVICES` for a single PhaseNet
        subprocess call without touching the parent process's environment).

    Returns
    -------
    subprocess.CompletedProcess
        On success (exit code 0).

    Raises
    ------
    ExternalToolError
        If the process exits non-zero, or times out.
    """
    logger = log if log is not None else _MODULE_LOGGER
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info("running: %s", cmd_str)

    full_env = {**os.environ, **env} if env is not None else None

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=full_env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if stdout:
            logger.info(stdout)
        if stderr:
            logger.error(stderr)
        logger.error("command timed out after %ss: %s", timeout, cmd_str)
        raise ExternalToolError(
            f"Command timed out after {timeout}s: {cmd_str}",
            cmd=cmd,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
        ) from exc

    if proc.stdout:
        logger.info(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            logger.error(proc.stderr)
        logger.error("command failed (exit %s): %s", proc.returncode, cmd_str)
        raise ExternalToolError(
            f"Command failed (exit {proc.returncode}): {cmd_str}",
            cmd=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    elif proc.stderr:
        logger.warning(proc.stderr)

    return proc
