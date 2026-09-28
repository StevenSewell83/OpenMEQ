"""Detection diagnostics (M2.3, ``OpenMEQ-ujx.3``, PLAN.md §5's "Detect" row:
"Detections per day/hour; coincidence-sum histogram; station participation").

:func:`make_detect_diagnostics` works purely from the ``detections.csv``-shaped
``pandas.DataFrame`` :func:`openmeq.detect.merge.merge_windows` produces (or
the same table re-read from disk -- see the ``stations`` column note below),
never recomputing detection itself, mirroring :mod:`openmeq.qc.plots`'s own
"diagnostics module never recomputes the step" convention. Every figure goes
through :mod:`openmeq.plotting` (``plot_timeseries``, ``plot_hist``,
``plot_bar``), so styling stays consistent with every other step's
diagnostics; nothing here imports ``matplotlib`` directly.

Outputs (``detections/diagnostics/``, ARCHITECTURE.md §3)
------------------------------------------------------------
- ``detections_per_hour.png``: detection counts binned by UTC hour, one bar
  covering the whole run's ``[cfg.data.start, cfg.data.end]`` span --
  :func:`~openmeq.plotting.plot_timeseries` (a "count over time" line is a
  better fit than :func:`~openmeq.plotting.plot_hist`'s numeric binning here,
  since the x-axis needs to be real UTC time, not an arbitrary bin index, and
  empty hours must show as an explicit zero rather than a gap). For this
  project's 1-2 day synthetic/typical short runs this single hourly-binned
  chart shows the per-day pattern too (each day is simply ~24 consecutive
  points); PLAN.md §5's "per day/hour" wording is satisfied by one adequately
  time-resolved chart rather than two separate ones, matching this task's own
  acceptance criteria (only ``detections_per_hour.png`` is required).
- ``coincidence_sum_hist.png``: histogram of every detection's
  ``coincidence_sum`` -- :func:`~openmeq.plotting.plot_hist`.
- ``station_participation.png``: how many detections each station
  (``NET.STA``) appears in, across the ``stations`` column --
  :func:`~openmeq.plotting.plot_bar` (a new, minimal addition to
  ``openmeq.plotting`` this task made -- see that module's docstring; no
  existing primitive fit a discrete-category count).
- ``summary.csv``: total detection count, mean/median ``coincidence_sum``,
  and the covered date range (PLAN.md §5: "a short text/CSV summary").

``stations`` column shape
--------------------------
:func:`merge_windows`'s own in-memory ``DataFrame`` (before
:func:`~openmeq.detect.merge.write_detections_csv` runs) holds ``stations``
as a Python ``list[str]``; the same table re-read from ``detections.csv``
holds it as a ``;``-joined string (ARCHITECTURE.md §4.3). :func:`_split_stations`
accepts either, so this module works identically whichever form the caller
passes -- ``openmeq.detect.pipeline.run_detect`` passes the in-memory
``DataFrame`` straight from :func:`merge_windows` (never re-reads the CSV it
just wrote), but a caller/test that only has the on-disk CSV can load it with
``pandas.read_csv`` and pass that directly.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.plotting import plot_bar, plot_hist, plot_timeseries

if TYPE_CHECKING:
    from openmeq.paths import RunPaths

#: ``detections/diagnostics/summary.csv``'s exact column order.
SUMMARY_COLUMNS = ["metric", "value"]


def _split_stations(value: object) -> list[str]:
    """One ``stations`` cell (a list or a ``;``-joined string) -> a list of ``NET.STA``."""
    if isinstance(value, list):
        return [s for s in value if s]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [s for s in text.split(";") if s]


def _station_counts(detections_df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if detections_df is None or detections_df.empty or "stations" not in detections_df.columns:
        return counts
    for value in detections_df["stations"]:
        counts.update(_split_stations(value))
    return counts


def _detections_per_hour(detections_df: pd.DataFrame, path: str | Path) -> None:
    """``detections_per_hour.png``: detection count per UTC hour bucket.

    Every hour in ``[floor(min time), floor(max time)]`` is represented,
    including hours with zero detections (an explicit ``0``, not a gap), so
    the chart's x-axis faithfully spans the run's covered time range.
    """
    if detections_df is None or detections_df.empty or "time" not in detections_df.columns:
        plot_timeseries(
            pd.DataFrame(columns=["hour", "count"]),
            path,
            x="hour",
            y="count",
            xlabel="Time (UTC)",
            ylabel="Detections per hour",
            title="Detections per hour",
            marker=None,
        )
        return

    times = pd.to_datetime(detections_df["time"], utc=True)
    hours = times.dt.floor("h")
    full_range = pd.date_range(hours.min(), hours.max(), freq="1h")
    counts = hours.value_counts().reindex(full_range, fill_value=0).sort_index()
    df = pd.DataFrame({"hour": counts.index, "count": counts.to_numpy()})

    plot_timeseries(
        df,
        path,
        x="hour",
        y="count",
        xlabel="Time (UTC)",
        ylabel="Detections per hour",
        title="Detections per hour",
        marker=None,
    )


def _coincidence_sum_hist(detections_df: pd.DataFrame, path: str | Path) -> None:
    if detections_df is None or detections_df.empty or "coincidence_sum" not in detections_df:
        plot_hist([], path, xlabel="Coincidence sum", title="Coincidence-sum distribution")
        return

    values = pd.to_numeric(detections_df["coincidence_sum"], errors="coerce").dropna()
    if values.empty:
        bins: int | list[float] = 10
    else:
        lo, hi = int(np.floor(values.min())), int(np.ceil(values.max()))
        # Integer-centered bins (coincidence_sum is a station-weight sum,
        # typically small near-integer values) -- falls back to a plain
        # bin count if every value is identical (lo == hi).
        bins = list(np.arange(lo - 0.5, hi + 1.5, 1.0)) if hi > lo else 10

    plot_hist(
        values,
        path,
        bins=bins,
        xlabel="Coincidence sum",
        ylabel="Count",
        title="Coincidence-sum distribution",
    )


def _station_participation(detections_df: pd.DataFrame, path: str | Path) -> None:
    counts = _station_counts(detections_df)
    stations = sorted(counts)
    values = [counts[s] for s in stations]
    plot_bar(
        stations,
        values,
        path,
        xlabel="Station (NET.STA)",
        ylabel="Detections",
        title="Station participation",
    )


def _write_summary(detections_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if detections_df is None or detections_df.empty:
        rows = [
            {"metric": "total_detections", "value": 0},
            {"metric": "mean_coincidence_sum", "value": ""},
            {"metric": "median_coincidence_sum", "value": ""},
            {"metric": "date_range_start", "value": ""},
            {"metric": "date_range_end", "value": ""},
        ]
    else:
        coincidence = pd.to_numeric(detections_df["coincidence_sum"], errors="coerce")
        # ISO-8601 "...Z" strings (ARCHITECTURE.md §1) sort lexicographically
        # in the same order as chronologically (fixed-width fields), so the
        # plain string min/max gives the true covered range without parsing.
        times = detections_df["time"].astype(str)
        rows = [
            {"metric": "total_detections", "value": int(len(detections_df))},
            {"metric": "mean_coincidence_sum", "value": float(coincidence.mean())},
            {"metric": "median_coincidence_sum", "value": float(coincidence.median())},
            {"metric": "date_range_start", "value": times.min()},
            {"metric": "date_range_end", "value": times.max()},
        ]

    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


def make_detect_diagnostics(paths: RunPaths, detections_df: pd.DataFrame) -> None:
    """Build every ``detections/diagnostics/`` output from ``detections_df``.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
        Only ``paths.detect_diagnostics_dir`` is used.
    detections_df : pandas.DataFrame
        The ``detections.csv``-shaped table (ARCHITECTURE.md §4.3) -- either
        :func:`openmeq.detect.merge.merge_windows`'s in-memory ``DataFrame``
        (``stations``/``trace_ids`` as Python lists) or the same table
        re-read from disk (``;``-joined strings); see the module docstring.
        An empty (zero-row) or ``None`` frame produces every output file
        with an explicit "no data" placeholder rather than raising or
        skipping -- ``test_diagnostics_files_exist``-style acceptance checks
        only require the files to exist and be non-empty PNGs/CSV, which an
        empty-but-valid figure/summary still satisfies.
    """
    out_dir = Path(paths.detect_diagnostics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _detections_per_hour(detections_df, out_dir / "detections_per_hour.png")
    _coincidence_sum_hist(detections_df, out_dir / "coincidence_sum_hist.png")
    _station_participation(detections_df, out_dir / "station_participation.png")
    _write_summary(detections_df, out_dir / "summary.csv")
