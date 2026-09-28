"""Tests for openmeq.log: configure_logging, get_logger (M0.4).

See docs/ARCHITECTURE.md §2: ``configure_logging(paths, step, level="INFO")``
sets up the console handler plus ``<output_dir>/logs/<step>.log``;
``get_logger(name)`` returns a child of the ``"openmeq"`` logger and, before
``configure_logging`` is called, logs to the console only.
"""

from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

import openmeq.log as log_mod
from openmeq.config import Config
from openmeq.log import configure_logging, get_logger
from openmeq.paths import RunPaths


@contextmanager
def _capture_console():
    """Redirect the module's single console handler to an in-memory buffer.

    Testing the *real* console (process stderr) through pytest's own output
    capturing is unreliable here: our handler is created once, at module
    import time, and holds a direct reference to whatever ``sys.stderr`` was
    at that moment, which does not necessarily match the stream object a
    given test's capture fixture swaps in later. Swapping the handler's
    ``.stream`` directly is a deterministic way to assert on exactly what
    the console handler writes, without depending on that plumbing.
    """
    handler = log_mod._console_handler
    assert handler is not None, "console handler should be attached on import"
    buf = io.StringIO()
    old_stream = handler.stream
    handler.stream = buf
    try:
        yield buf
    finally:
        handler.stream = old_stream


def _make_config(output_dir: Path) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "test", "output_dir": str(output_dir)},
            "data": {
                "waveform_dirs": [str(output_dir)],
                "start": "2025-01-01",
                "end": "2025-01-02",
            },
            "nlloc": {
                "velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")},
                "grid": {"origin": [0.0, 0.0], "size_km": [1.0, 1.0, 1.0], "spacing_km": 0.1},
            },
            "hypodd": {
                "velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")},
            },
        }
    )


@pytest.fixture(autouse=True)
def _reset_openmeq_logger():
    """Isolate each test: openmeq.log's module state (and the shared
    "openmeq" logger's handlers/level) persists across tests in the same
    process, so restore it after each test.
    """
    import openmeq.log as log_mod

    logger = logging.getLogger("openmeq")
    handlers_before = list(logger.handlers)
    level_before = logger.level
    console_before = log_mod._console_handler
    file_before = log_mod._file_handler
    file_path_before = log_mod._file_handler_path
    yield
    for h in list(logger.handlers):
        if h not in handlers_before:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(level_before)
    log_mod._console_handler = console_before
    log_mod._file_handler = file_before
    log_mod._file_handler_path = file_path_before


def test_get_logger_is_a_child_of_openmeq() -> None:
    logger = get_logger("qc")
    assert logger.name == "openmeq.qc"


def test_get_logger_works_without_configure_logging() -> None:
    """Console-only logging must work before configure_logging is ever called."""
    logger = get_logger("standalone")
    with _capture_console() as buf:
        logger.warning("hello console")
    assert "hello console" in buf.getvalue()


def test_configure_logging_writes_console_and_file(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "run")
    paths = RunPaths(cfg)
    paths.ensure()

    configure_logging(paths, "qc")
    logger = get_logger("qc")
    with _capture_console() as buf:
        logger.info("qc message one")
    assert "qc message one" in buf.getvalue()

    log_path = paths.log_file("qc")
    assert log_path.is_file()
    assert "qc message one" in log_path.read_text()


def test_configure_logging_repeated_same_step_no_duplicate_handlers(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "run")
    paths = RunPaths(cfg)
    paths.ensure()

    configure_logging(paths, "qc")
    n_handlers_first = len(logging.getLogger("openmeq").handlers)
    configure_logging(paths, "qc")
    n_handlers_second = len(logging.getLogger("openmeq").handlers)

    assert n_handlers_first == n_handlers_second

    logger = get_logger("qc")
    logger.info("only once each")
    log_path = paths.log_file("qc")
    assert log_path.read_text().count("only once each") == 1


def test_configure_logging_switches_step_log_file(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "run")
    paths = RunPaths(cfg)
    paths.ensure()

    configure_logging(paths, "qc")
    get_logger("qc").info("in qc log")

    configure_logging(paths, "detect")
    get_logger("detect").info("in detect log")

    qc_log = paths.log_file("qc").read_text()
    detect_log = paths.log_file("detect").read_text()
    assert "in qc log" in qc_log
    assert "in detect log" not in qc_log
    assert "in detect log" in detect_log
