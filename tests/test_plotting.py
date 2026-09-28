"""Tests for openmeq.plotting.

Covers every public function from openmeq/plotting/__init__.py: each gets a
synthetic-data smoke test asserting a non-empty PNG is produced, plus edge
cases (empty data, mismatched station data, figure-leak checks, and
multiprocessing safety).
"""

from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from openmeq.plotting import (
    plot_boxplot,
    plot_heatmap,
    plot_hist,
    plot_map_sections,
    plot_timeseries,
    save_fig,
    setup_style,
)


def _png_ok(path: Path) -> None:
    assert path.exists(), f"{path} was not created"
    assert path.stat().st_size > 0, f"{path} is empty"


# --------------------------------------------------------------------------
# setup_style
# --------------------------------------------------------------------------


def test_setup_style_sets_agg_backend_and_is_idempotent():
    setup_style()
    assert matplotlib.get_backend().lower() == "agg"
    # Calling it again must not raise and must leave the backend as Agg.
    setup_style()
    setup_style()
    assert matplotlib.get_backend().lower() == "agg"


# --------------------------------------------------------------------------
# save_fig
# --------------------------------------------------------------------------


def test_save_fig_creates_nested_dirs_and_closes_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    out = tmp_path / "a" / "b" / "c" / "plot.png"
    assert plt.fignum_exists(fig.number)

    save_fig(fig, out)

    _png_ok(out)
    assert not plt.fignum_exists(fig.number)


def test_save_fig_batch_leaves_no_open_figures(tmp_path):
    plt.close("all")
    for i in range(25):
        fig, ax = plt.subplots()
        ax.hist(np.random.default_rng(i).normal(size=50))
        save_fig(fig, tmp_path / f"batch_{i}.png")
    assert len(plt.get_fignums()) == 0


# --------------------------------------------------------------------------
# plot_map_sections
# --------------------------------------------------------------------------


@pytest.fixture
def events_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "lon": -122.0 + rng.normal(scale=0.05, size=n),
            "lat": 38.0 + rng.normal(scale=0.05, size=n),
            "depth_km": rng.uniform(0.5, 8.0, size=n),
            "rms": rng.uniform(0.01, 0.3, size=n),
        }
    )


@pytest.fixture
def stations_df():
    return pd.DataFrame(
        {
            "network": ["XX"] * 6,
            "station": [f"S{i:02d}" for i in range(6)],
            "location": [""] * 6,
            "latitude": 38.0 + np.linspace(-0.06, 0.06, 6),
            "longitude": -122.0 + np.linspace(-0.06, 0.06, 6),
            "elevation_m": [100.0] * 6,
            "depth_m": [0.0] * 6,
        }
    )


def test_plot_map_sections_basic(tmp_path, events_df, stations_df):
    out = tmp_path / "diagnostics" / "passA" / "map_sections.png"
    plot_map_sections(events_df, stations_df, out, color_by="rms", title="Pass A locations")
    _png_ok(out)


def test_plot_map_sections_no_color_by(tmp_path, events_df, stations_df):
    out = tmp_path / "map_sections_plain.png"
    plot_map_sections(events_df, stations_df, out)
    _png_ok(out)


@pytest.mark.parametrize(
    "bad_stations",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"foo": [1, 2, 3]}),
        pd.DataFrame({"latitude": [], "longitude": []}),
    ],
    ids=["none", "empty", "wrong-columns", "empty-with-columns"],
)
def test_plot_map_sections_handles_bad_station_data(tmp_path, events_df, bad_stations):
    out = tmp_path / "map_sections_bad_stations.png"
    # Must not raise for any of these mismatched/empty station inputs.
    plot_map_sections(events_df, bad_stations, out)
    _png_ok(out)


def test_plot_map_sections_handles_empty_events(tmp_path, stations_df):
    out = tmp_path / "map_sections_no_events.png"
    empty = pd.DataFrame(columns=["lon", "lat", "depth_km"])
    plot_map_sections(empty, stations_df, out)
    _png_ok(out)


# --------------------------------------------------------------------------
# plot_hist
# --------------------------------------------------------------------------


def test_plot_hist_basic(tmp_path):
    rng = np.random.default_rng(1)
    values = rng.normal(scale=0.1, size=500)
    out = tmp_path / "residuals.png"
    plot_hist(
        values,
        out,
        bins=40,
        xlabel="P residual (s)",
        title="P residuals, pass A",
        vlines={"threshold": 0.2, "-threshold": -0.2},
    )
    _png_ok(out)


def test_plot_hist_empty(tmp_path):
    out = tmp_path / "empty_hist.png"
    plot_hist([], out, xlabel="Residual (s)")
    _png_ok(out)


def test_plot_hist_with_nan_and_plain_vlines(tmp_path):
    values = [1.0, 2.0, float("nan"), 3.0, 3.0, 4.0]
    out = tmp_path / "hist_nan.png"
    plot_hist(values, out, vlines=[2.5])
    _png_ok(out)


# --------------------------------------------------------------------------
# plot_heatmap
# --------------------------------------------------------------------------


def test_plot_heatmap_basic(tmp_path):
    rng = np.random.default_rng(2)
    channels = [f"XX.S{i:02d}..HHZ" for i in range(10)]
    days = pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    matrix = pd.DataFrame(
        rng.uniform(0, 100, size=(len(channels), len(days))), index=channels, columns=days
    )
    out = tmp_path / "availability_heatmap.png"
    plot_heatmap(
        matrix,
        out,
        xlabel="Day",
        ylabel="Channel",
        cbar_label="Coverage (%)",
        title="QC availability",
    )
    _png_ok(out)


def test_plot_heatmap_array_input(tmp_path):
    matrix = np.zeros((3, 3))
    out = tmp_path / "heatmap_array.png"
    plot_heatmap(matrix, out)
    _png_ok(out)


# --------------------------------------------------------------------------
# plot_timeseries
# --------------------------------------------------------------------------


def test_plot_timeseries_grouped(tmp_path):
    rng = np.random.default_rng(3)
    rows = []
    for seed_id in ["XX.S01..HHZ", "XX.S02..HHZ", "XX.S03..HHZ"]:
        for day in pd.date_range("2025-01-01", periods=14):
            rows.append({"seed_id": seed_id, "date": day, "rms": rng.uniform(1, 10)})
    df = pd.DataFrame(rows)
    out = tmp_path / "daily_stats.png"
    plot_timeseries(
        df, out, x="date", y="rms", group_by="seed_id", ylabel="RMS", title="Daily RMS per channel"
    )
    _png_ok(out)


def test_plot_timeseries_single_series(tmp_path):
    df = pd.DataFrame(
        {"time": pd.date_range("2025-01-01", periods=20), "ml": np.linspace(0.5, 2.5, 20)}
    )
    out = tmp_path / "ml_through_time.png"
    plot_timeseries(df, out, x="time", y="ml", title="ML through time")
    _png_ok(out)


def test_plot_timeseries_empty(tmp_path):
    out = tmp_path / "ts_empty.png"
    plot_timeseries(pd.DataFrame(columns=["time", "value"]), out, x="time", y="value")
    _png_ok(out)


# --------------------------------------------------------------------------
# plot_boxplot
# --------------------------------------------------------------------------


def test_plot_boxplot_basic(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for station in [f"S{i:02d}" for i in range(8)]:
        for _ in range(30):
            rows.append({"station": station, "ratio": rng.normal(1.0, 0.2)})
    df = pd.DataFrame(rows)
    out = tmp_path / "monthly_boxplot.png"
    plot_boxplot(
        df,
        out,
        by="station",
        column="ratio",
        ylabel="RMS ratio to network median",
        title="Monthly QC",
    )
    _png_ok(out)


def test_plot_boxplot_empty(tmp_path):
    out = tmp_path / "boxplot_empty.png"
    plot_boxplot(pd.DataFrame(columns=["station", "ratio"]), out, by="station", column="ratio")
    _png_ok(out)


# --------------------------------------------------------------------------
# Multiprocessing / worker-process safety
# --------------------------------------------------------------------------
#
# openmeq.parallel.pmap (M0.5) is being built concurrently with this task and
# is still a stub as of this test; the test below skips gracefully rather
# than failing hard if it's not implemented yet, and a second test exercises
# the exact same requirement directly via ProcessPoolExecutor so the
# "safe under multiprocessing/Agg backend" guarantee is verified regardless.


def _worker_render_one(args):
    """Top-level, picklable worker: render one PNG of each kind."""
    kind, out_path = args
    out_path = Path(out_path)
    if kind == "hist":
        plot_hist([1, 2, 2, 3, 3, 3, 4], out_path, xlabel="x")
    elif kind == "heatmap":
        plot_heatmap(np.arange(12).reshape(3, 4).astype(float), out_path)
    elif kind == "timeseries":
        df = pd.DataFrame({"x": range(5), "y": [1.0, 2.0, 1.5, 3.0, 2.5]})
        plot_timeseries(df, out_path, x="x", y="y")
    elif kind == "boxplot":
        df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
        plot_boxplot(df, out_path, by="g", column="v")
    elif kind == "map_sections":
        df = pd.DataFrame({"lon": [-122.0, -122.1], "lat": [38.0, 38.1], "depth_km": [1.0, 2.0]})
        stations = pd.DataFrame({"latitude": [38.05], "longitude": [-122.05]})
        plot_map_sections(df, stations, out_path)
    else:
        raise ValueError(kind)
    return str(out_path)


def test_plotting_functions_safe_in_process_pool_executor(tmp_path):
    kinds = ["hist", "heatmap", "timeseries", "boxplot", "map_sections"]
    jobs = [(k, str(tmp_path / f"{k}.png")) for k in kinds]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=__import__("multiprocessing").get_context("spawn")
    ) as ex:
        results = list(ex.map(_worker_render_one, jobs))
    assert len(results) == len(kinds)
    for _kind, out_path in jobs:
        _png_ok(Path(out_path))


def test_plotting_functions_safe_via_openmeq_parallel_pmap(tmp_path):
    """Same guarantee, but via openmeq.parallel.pmap as the acceptance
    criterion specifies. Skips if pmap isn't implemented yet (it's a
    separate, concurrently-developed task, M0.5) rather than failing --
    once it lands this test starts exercising the real thing.
    """
    parallel = importlib.import_module("openmeq.parallel")
    pmap = getattr(parallel, "pmap", None)
    if pmap is None:
        pytest.skip("openmeq.parallel.pmap not implemented yet (M0.5)")

    kinds = ["hist", "heatmap", "timeseries", "boxplot", "map_sections"]
    jobs = [(k, str(tmp_path / f"pmap_{k}.png")) for k in kinds]

    class _PCfg:
        n_workers = 2
        threads_per_worker = 1

    results = pmap(_worker_render_one, jobs, _PCfg())
    assert len(results) == len(kinds)
    for _kind, out_path in jobs:
        _png_ok(Path(out_path))
