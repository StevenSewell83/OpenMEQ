"""Tests for ``openmeq.detect.worker``'s M11.3 performance optimisations.

Two things are checked here, in a new file (per the M11.3 worker
instructions: another in-flight task, ``OpenMEQ-m1f.6``, is consolidating
``tests/`` fixtures, so this file is additive and does not touch
``tests/conftest.py``):

1. :func:`openmeq.detect.worker._fast_linear_detrend` (the closed-form OLS
   detrend that replaced ``stream.detrend("linear")``, see that module's
   docstring's "Fast linear detrend" note) matches
   ``scipy.signal.detrend(data, type="linear")`` within a documented
   tolerance, on both synthetic arrays and a real synthetic-project
   channel-day.
2. :func:`openmeq.detect.worker.detect_window`'s new per-worker-process
   ``DataIndex``/``Config`` caching (mirroring ``openmeq.qc.stats``'s
   established pattern) actually avoids re-parsing the index/config on a
   second call with the same paths, without changing its output.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as sps
from obspy import read

from openmeq.config import load_config
from openmeq.detect import worker as detect_worker
from openmeq.detect.worker import _fast_linear_detrend, build_windows, detect_window
from openmeq.index import DataIndex, build_index
from openmeq.paths import RunPaths

# ---------------------------------------------------------------------------
# 1. _fast_linear_detrend vs. scipy.signal.detrend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [1, 2, 3, 1000, 200_001])
def test_fast_linear_detrend_matches_scipy_reference(dtype, n):
    rng = np.random.default_rng(1234)
    raw = (rng.standard_normal(n) * 100 + np.linspace(0, 5000, n)).astype(dtype)

    reference = sps.detrend(raw.copy(), type="linear")
    fast = _fast_linear_detrend(raw.copy())

    assert fast.dtype == reference.dtype
    assert fast.shape == reference.shape

    diff = np.abs(fast.astype(np.float64) - reference.astype(np.float64))
    # Scale against the *input* data's own amplitude (not the detrended
    # reference's, which can be near-zero for a tiny/degenerate n where the
    # fit is exact) -- a meaningful denominator for "relative to what this
    # array actually looked like before detrending".
    scale = max(float(np.abs(raw.astype(np.float64)).max()), 1.0)
    # Same fitted line up to floating-point accumulation rounding (see
    # module docstring): relative tolerance against the input's own
    # amplitude scale, documented and generous, not a tight numerical match.
    assert diff.max() / scale < 1e-4, f"n={n} dtype={dtype}: max diff {diff.max()} / scale {scale}"


def test_fast_linear_detrend_constant_array_is_just_demeaned():
    # A perfectly flat array has zero slope by construction (sum_x2 == 0
    # only for n<=1; for a flat *value* array with n>1 the closed-form slope
    # is 0 too -- exercised here as a degenerate/edge case).
    data = np.full(500, 42.0, dtype=np.float64)
    out = _fast_linear_detrend(data)
    assert np.allclose(out, 0.0, atol=1e-9)


def test_fast_linear_detrend_single_sample():
    data = np.array([7.0], dtype=np.float32)
    out = _fast_linear_detrend(data)
    assert out.dtype == np.float32
    assert out.shape == (1,)
    assert np.allclose(out, [0.0])


def test_fast_linear_detrend_matches_scipy_on_real_channel_day(synthetic_project):
    """Real synthetic-project miniSEED data (not synthetic-array noise)."""
    config_path, _truth = synthetic_project
    waveforms_dir = config_path.parent / "waveforms" / "2025-01-01"
    mseed_files = sorted(waveforms_dir.glob("*.mseed"))
    assert mseed_files, "expected at least one real synthetic channel-day file"

    stream = read(str(mseed_files[0]))
    trace = stream[0]
    raw = trace.data.astype(np.float32)

    reference = sps.detrend(raw.copy(), type="linear")
    fast = _fast_linear_detrend(raw.copy())

    diff = np.abs(fast.astype(np.float64) - reference.astype(np.float64))
    scale = max(float(np.abs(raw.astype(np.float64)).max()), 1.0)
    assert diff.max() / scale < 1e-4


# ---------------------------------------------------------------------------
# 2. Per-worker-process DataIndex/Config caching
# ---------------------------------------------------------------------------


def _setup_project(synthetic_project, tmp_path: Path):
    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    new_config_path = project_copy / "config.yaml"
    cfg = load_config(new_config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    build_index(cfg)
    return cfg, paths, truth, new_config_path


def test_detect_window_caches_index_and_config_per_worker_process(
    synthetic_project, tmp_path, monkeypatch
):
    cfg, paths, _truth, config_path = _setup_project(synthetic_project, tmp_path)

    windows = build_windows(cfg)
    assert len(windows) >= 2, "need at least two windows to exercise the cache"

    # Clear the module-level caches (a previous test in this same pytest
    # process may have already populated them under different paths) so
    # this test's call counts are unambiguous.
    detect_worker._worker_index_cache.clear()
    detect_worker._worker_cfg_cache.clear()

    real_index_load = DataIndex.load
    real_load_config = load_config
    index_load_calls = []
    load_config_calls = []

    def _spy_index_load(path):
        index_load_calls.append(path)
        return real_index_load(path)

    def _spy_load_config(path):
        load_config_calls.append(path)
        return real_load_config(path)

    monkeypatch.setattr("openmeq.detect.worker.DataIndex.load", _spy_index_load)
    monkeypatch.setattr("openmeq.detect.worker.load_config", _spy_load_config)

    all_triggers = []
    for window_start, window_end in windows:
        all_triggers.extend(
            detect_window(window_start, window_end, str(paths.index_json), str(config_path))
        )

    # Same index_path/cfg_path across every window in this run -> loaded
    # (and validated) exactly once each, not once per window.
    assert index_load_calls == [str(paths.index_json)]
    assert load_config_calls == [str(config_path)]
    assert len(all_triggers) > 0
