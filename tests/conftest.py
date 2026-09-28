"""Shared pytest fixtures.

``synthetic_project`` (M0.8) builds the deterministic synthetic test
dataset described in ``docs/ARCHITECTURE.md`` §8 and ``tests/data/make_synthetic.py``'s
module docstring, once per test session, and hands every test
``(config_path, truth)``.

``synthetic_qc_tables`` (M1.9, ``OpenMEQ-5nl.9``) builds on top of that: the
full QC daily-stats computation for the whole synthetic archive (48
channel-days) took ~20-40s per test *module* before this fixture existed,
and two modules (``tests/test_qc_stats.py``, ``tests/test_qc_monthly_flags.py``)
each had their own module-scoped fixture recomputing it independently --
see docs/performance.md for the measured before/after suite runtime this
sharing gets.
"""

from __future__ import annotations

import sys
import time
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


@pytest.fixture(scope="session")
def synthetic_qc_tables(synthetic_project):
    """``(cfg, idx, availability_df, daily_stats_df, monthly_df, truth)`` for
    the whole synthetic archive, computed **once per test session** and
    shared by every QC test module (``tests/test_qc_stats.py``,
    ``tests/test_qc_monthly_flags.py``) -- OpenMEQ-5nl.9: both modules used
    to independently recompute the same ~48-channel-day daily stats in
    their own module-scoped fixture (~20-40s each), which dominated the
    full suite's runtime; see docs/performance.md for the measured
    before/after.

    Read-only against the shared ``synthetic_project`` fixture's waveform
    files (only ``inventory/index.json`` under its ``runs/synthetic/`` gets
    (re)written, deterministically -- same pattern ``tests/test_index.py``
    and the modules above already relied on for their own now-removed
    per-module fixtures), so it is safe to compute once and hand the same
    DataFrames to every test in the session.
    """
    from openmeq.config import load_config
    from openmeq.index import build_index
    from openmeq.qc.availability import compute_availability, date_range_inclusive
    from openmeq.qc.flags import compute_monthly_stats
    from openmeq.qc.stats import compute_daily_stats

    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    idx = build_index(cfg)
    seed_ids = idx.seed_ids()
    dates = date_range_inclusive(cfg.data.start, cfg.data.end)
    n = len(seed_ids) * len(dates)

    availability_df = compute_availability(idx, seed_ids, dates, cfg)

    t0 = time.perf_counter()
    daily_stats_df = compute_daily_stats(idx, seed_ids, dates, cfg)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[qc.stats throughput, shared synthetic_qc_tables fixture] "
        f"{n} channel-days in {elapsed:.2f}s ({n / elapsed:.1f} channel-days/s, "
        f"n_workers={cfg.parallel.n_workers})"
    )

    monthly_df = compute_monthly_stats(daily_stats_df, cfg)

    return cfg, idx, availability_df, daily_stats_df, monthly_df, truth


def pytest_sessionfinish(session, exitstatus):
    """Remove the per-checkout basetemp after a fully passing run.

    Synthetic datasets are ~500 MB each; with ``--basetemp`` pytest keeps every
    session fixture's data until the next run, so clean up once everything has
    passed (failed runs keep it for debugging).
    """
    if exitstatus != 0:
        return
    import shutil

    basetemp = session.config._tmp_path_factory.getbasetemp()
    shutil.rmtree(basetemp, ignore_errors=True)
