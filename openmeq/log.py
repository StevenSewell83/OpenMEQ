"""Logging setup (``configure_logging``, ``get_logger``).

Design (ARCHITECTURE.md §2)
----------------------------
There is a single top-level ``"openmeq"`` logger. Every module gets its own
logger by calling :func:`get_logger`, which returns
``logging.getLogger(f"openmeq.{name}")`` -- a *child* of ``"openmeq"`` with
no handlers of its own. A child logger's records propagate up to (and are
actually emitted by) the top-level ``"openmeq"`` logger's handlers, so
attaching handlers in exactly one place -- here -- is enough for every
module's logger to be routed correctly.

On import, this module attaches a console ``StreamHandler`` to the
``"openmeq"`` logger, so :func:`get_logger` is immediately usable -- with
console output only -- from library code, scripts and tests that never call
:func:`configure_logging` (e.g. a unit test importing a single function).
This is what ARCHITECTURE.md §2 means by "until ``configure_logging`` is
called it logs to the console only".

``openmeq/cli.py`` calls ``configure_logging(paths, step, level="INFO")``
once, before dispatching to a step's body. This additionally attaches a
``FileHandler`` writing to ``paths.log_file(step)``, so every message any
module logs via :func:`get_logger` during that step lands in both the
console and ``<output_dir>/logs/<step>.log``.

Re-entrancy and the "no duplicate handlers" guard
--------------------------------------------------
:func:`configure_logging` may run more than once in a single process --
e.g. a test harness invoking several CLI subcommands with click's
``CliRunner`` inside one pytest session, or (once M11.1 fills it in)
``openmeq run`` dispatching several steps' bodies in-process. Each call:

* reuses the single console handler -- it is created once, on import, and
  never duplicated;
* replaces the *previous* file handler with a new one for the new
  ``paths.log_file(step)``, unless that call repeats the exact same log path
  already configured, in which case the existing file handler is left in
  place (this is the "guard against a handler added twice" pitfall the beads
  issue calls out).

In other words, only the *most recently configured* step's log file
receives new records. That matches normal use: each ``openmeq <step>``
invocation is a separate process with its own log file; only test/chaining
code calls :func:`configure_logging` more than once per process, and each
such call is for a genuinely different step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmeq.paths import RunPaths

_LOGGER_NAME = "openmeq"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None
_file_handler_path: Path | None = None


def _root_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def _ensure_console_handler() -> None:
    """Attach the (single) console handler to the ``"openmeq"`` logger, if not already done."""
    global _console_handler
    logger = _root_logger()
    if _console_handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        _console_handler = handler
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


# Console-only logging is available as soon as this module is imported.
_ensure_console_handler()


def configure_logging(paths: RunPaths, step: str, level: str | int = "INFO") -> None:
    """Configure the ``"openmeq"`` logger for one step's run.

    Called once by the CLI, before a step's body runs. Sets the console
    handler (already attached on import) plus a ``FileHandler`` writing to
    ``paths.log_file(step)`` -- see the module docstring for the guard
    against duplicate/stale file handlers across repeated calls.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
        Used for ``paths.log_file(step)``.
    step : str
        The CLI subcommand name (``"qc"``, ``"detect"``, ...); forwarded to
        ``paths.log_file``.
    level : str or int, optional
        Logging level for the console and file handlers (and the
        ``"openmeq"`` logger itself). Default ``"INFO"``.
    """
    global _file_handler, _file_handler_path
    logger = _root_logger()
    _ensure_console_handler()
    logger.setLevel(level)
    if _console_handler is not None:
        _console_handler.setLevel(level)

    log_path = Path(paths.log_file(step))
    if _file_handler is not None and _file_handler_path == log_path:
        # Same (paths.root, step) reconfigured -- just refresh the level.
        _file_handler.setLevel(level)
        return

    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    logger.addHandler(handler)
    _file_handler = handler
    _file_handler_path = log_path


def get_logger(name: str) -> logging.Logger:
    """Return a named child of the ``"openmeq"`` logger.

    Parameters
    ----------
    name : str
        Usually a step name (``"qc"``, ``"detect"``, ...) or a module name.

    Returns
    -------
    logging.Logger
        ``logging.getLogger(f"openmeq.{name}")``. Its records propagate to
        the ``"openmeq"`` logger's handlers -- console-only until
        :func:`configure_logging` has been called in this process.
    """
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
