"""Opt-in extended magnitude QAQC (OpenMEQ-02o.6).

:func:`make_magnitude_extended_diagnostics` is called by
:func:`~openmeq.magnitude.pipeline.run_magnitude` only when
``cfg.diagnostics.extended_enabled("magnitude")`` is true; the always-on
baseline set (:mod:`openmeq.magnitude.diagnostics`) is unaffected. Like the
baseline, it works from the in-memory ``station_magnitudes.csv``-shaped
table and output `Catalog` -- except ``amp_window_*.png``, which re-runs
:func:`~openmeq.magnitude.amplitude.measure_event_amplitudes` (with its
``capture=`` hook) for the few sampled events, so the traces shown are the
exact ones the amplitudes were measured on, not a re-implementation.

Every figure is independently guarded: an exception is logged as a warning
and the next figure is attempted -- extended QAQC never fails the step.

Outputs (``magnitude/diagnostics/extended/``)
----------------------------------------------
- ``res_vs_distance.png``: included station ML minus event ML vs
  hypocentral distance, coloured by component (last channel-code
  character), with equal-count distance-binned median and IQR. A trend
  with distance means the ``-log A0`` attenuation relation
  (``cfg.magnitude.ml.relation``) does not fit the region.
- ``gr_mc_bvalue.png`` / ``gr_mc_bvalue.csv``: cumulative and
  non-cumulative magnitude-frequency distribution, Mc by maximum curvature,
  Aki/Utsu maximum-likelihood b-value, bootstrap uncertainty (see
  :func:`gr_statistics`); the CSV is the one-row summary of the same fit.
- ``amp_window_<event_id>.png``: for up to ``cfg.diagnostics.n_examples``
  evenly sampled measured events, one panel per station of the chosen
  channel's Wood-Anderson-simulated trace, with the measurement window
  shaded, the peak sample marked, and P/S pick lines.

Mc / b-value method (:func:`gr_statistics`)
-------------------------------------------
Magnitudes are binned to ``bin_width`` (0.1). **Mc** is the maximum-
curvature estimate -- the bin with the highest non-cumulative count --
plus a **+0.2 correction**, since MAXC systematically underestimates Mc for
gradually curved distributions (Woessner & Wiemer 2005, BSSA 95(2)).
**b** is Aki's (1965) maximum-likelihood estimator with Utsu's (1965)
correction for binned data, ``b = log10(e) / (mean(M >= Mc) - (Mc -
bin_width / 2))``, and ``a = log10 N(>= Mc) + b * Mc``. Uncertainties are
the standard deviations over ``n_bootstrap`` (200) resamples-with-
replacement of the whole catalogue, re-estimating Mc *and* b each time,
from a fixed-seed RNG so reruns are byte-for-byte deterministic. With fewer
than :data:`MIN_EVENTS_FOR_FIT` events at or above Mc no fit is attempted:
the data are still drawn and the figure/CSV say "insufficient events".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.catalog import event_id as get_event_id
from openmeq.log import get_logger
from openmeq.magnitude.amplitude import (
    WoodAndersonCapture,
    _find_pick,
    measure_event_amplitudes,
)
from openmeq.magnitude.diagnostics import _event_ml_table
from openmeq.plotting import save_fig, setup_style
from openmeq.plotting._style import COLORBLIND_PALETTE

if TYPE_CHECKING:
    from obspy import Catalog
    from obspy.core.event import Event

    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("magnitude")

#: ``magnitude/diagnostics/<EXTENDED_SUBDIR>/``.
EXTENDED_SUBDIR = "extended"

#: Magnitude bin width for Mc / b-value estimation.
GR_BIN_WIDTH = 0.1

#: Maximum-curvature Mc correction (Woessner & Wiemer 2005).
MC_MAXC_CORRECTION = 0.2

#: Bootstrap resamples for the b / Mc uncertainty.
N_BOOTSTRAP = 200

#: Fixed bootstrap RNG seed (deterministic reruns).
BOOTSTRAP_SEED = 20050401

#: Minimum events at or above Mc for a b-value fit to be attempted.
MIN_EVENTS_FOR_FIT = 20

#: Cap on per-event panels in ``amp_window_<event_id>.png`` (nearest first).
MAX_AMP_WINDOW_STATIONS = 12


# ---------------------------------------------------------------------------
# Gutenberg-Richter statistics (pure functions)
# ---------------------------------------------------------------------------


def _bin_magnitudes(mags: np.ndarray, bin_width: float) -> np.ndarray:
    return np.round(np.round(mags / bin_width) * bin_width, 6)


def mc_max_curvature(
    mags, *, bin_width: float = GR_BIN_WIDTH, correction: float = MC_MAXC_CORRECTION
) -> float:
    """Mc by maximum curvature plus ``correction`` (module docstring).

    ``NaN`` for an empty input.
    """
    arr = np.asarray(mags, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    binned = _bin_magnitudes(arr, bin_width)
    values, counts = np.unique(binned, return_counts=True)
    return float(round(values[int(np.argmax(counts))] + correction, 6))


def b_value_mle(mags, mc: float, *, bin_width: float = GR_BIN_WIDTH) -> float:
    """Aki (1965) / Utsu (1965) MLE b-value of ``mags >= mc`` (binned).

    ``NaN`` if fewer than two events reach ``mc`` or their spread is
    degenerate.
    """
    arr = np.asarray(mags, dtype=float)
    arr = arr[np.isfinite(arr)]
    binned = _bin_magnitudes(arr, bin_width)
    above = binned[binned >= mc - 1e-9]
    if above.size < 2:
        return float("nan")
    denom = float(np.mean(above)) - (mc - bin_width / 2.0)
    if denom <= 0:
        return float("nan")
    return math.log10(math.e) / denom


@dataclass
class GrStatistics:
    """:func:`gr_statistics`'s result -- also ``gr_mc_bvalue.csv``'s one row."""

    status: str
    n_events: int
    bin_width: float
    mc_correction: float
    mc: float
    mc_std: float
    n_above_mc: int
    b: float
    b_std: float
    a: float
    n_bootstrap: int
    bootstrap_seed: int
    min_events_for_fit: int


def gr_statistics(
    mags,
    *,
    bin_width: float = GR_BIN_WIDTH,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    min_events: int = MIN_EVENTS_FOR_FIT,
) -> tuple[GrStatistics, np.ndarray]:
    """Mc, b, a and bootstrap uncertainties (module docstring's method).

    Returns
    -------
    (GrStatistics, numpy.ndarray)
        The summary, and the ``(n_ok, 3)`` array of per-resample
        ``(mc, b, a)`` (empty unless ``status == "ok"``). ``status`` is
        ``"ok"``, ``"insufficient events"`` or ``"no data"``.
    """
    arr = np.asarray(mags, dtype=float)
    arr = arr[np.isfinite(arr)]
    nan = float("nan")
    base = dict(
        n_events=int(arr.size),
        bin_width=bin_width,
        mc_correction=MC_MAXC_CORRECTION,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=seed,
        min_events_for_fit=min_events,
    )
    if arr.size == 0:
        stats = GrStatistics(
            status="no data", mc=nan, mc_std=nan, n_above_mc=0, b=nan, b_std=nan, a=nan, **base
        )
        return stats, np.empty((0, 3))

    mc = mc_max_curvature(arr, bin_width=bin_width)
    binned = _bin_magnitudes(arr, bin_width)
    n_above = int(np.sum(binned >= mc - 1e-9))
    if n_above < min_events:
        stats = GrStatistics(
            status="insufficient events",
            mc=mc,
            mc_std=nan,
            n_above_mc=n_above,
            b=nan,
            b_std=nan,
            a=nan,
            **base,
        )
        return stats, np.empty((0, 3))

    b = b_value_mle(arr, mc, bin_width=bin_width)
    a = math.log10(n_above) + b * mc

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        mc_i = mc_max_curvature(sample, bin_width=bin_width)
        b_i = b_value_mle(sample, mc_i, bin_width=bin_width)
        n_i = int(np.sum(_bin_magnitudes(sample, bin_width) >= mc_i - 1e-9))
        if math.isfinite(b_i) and n_i > 0:
            boot.append((mc_i, b_i, math.log10(n_i) + b_i * mc_i))
    boot_arr = np.asarray(boot, dtype=float).reshape(-1, 3)
    mc_std = float(np.std(boot_arr[:, 0], ddof=1)) if len(boot_arr) > 1 else nan
    b_std = float(np.std(boot_arr[:, 1], ddof=1)) if len(boot_arr) > 1 else nan

    stats = GrStatistics(
        status="ok", mc=mc, mc_std=mc_std, n_above_mc=n_above, b=b, b_std=b_std, a=a, **base
    )
    return stats, boot_arr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _no_data(ax, text: str = "No data") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def _residual_table(station_df: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    cols = ["distance_km", "residual", "component"]
    if station_df is None or station_df.empty or event_table.empty:
        return pd.DataFrame(columns=cols)
    df = station_df[station_df["included"].astype(bool)].copy()
    df["station_ml"] = pd.to_numeric(df["station_ml"], errors="coerce")
    df = df.merge(event_table[["event_id", "ml"]], on="event_id", how="inner")
    df["residual"] = df["station_ml"] - df["ml"]
    df["component"] = df["component_used"].astype(str).str[-1:]
    return df.dropna(subset=["distance_km", "residual"])[cols]


def _binned_median_iqr(x: np.ndarray, y: np.ndarray, *, max_bins: int = 10, min_per_bin: int = 5):
    """Equal-count bins over ``x``: ``(x_median, y_q25, y_median, y_q75)`` arrays."""
    n_bins = min(max_bins, x.size // min_per_bin)
    if n_bins < 1:
        return (np.array([]),) * 4
    order = np.argsort(x)
    rows = [
        (np.median(x[idx]), *np.percentile(y[idx], [25, 50, 75]))
        for idx in np.array_split(order, n_bins)
    ]
    return tuple(np.asarray(col) for col in zip(*rows, strict=True))


def plot_residual_vs_distance(
    station_df: pd.DataFrame, event_table: pd.DataFrame, path: Path
) -> None:
    """``res_vs_distance.png`` (module docstring)."""
    setup_style()
    import matplotlib.pyplot as plt

    df = _residual_table(station_df, event_table)
    fig, ax = plt.subplots()
    if df.empty:
        _no_data(ax)
    else:
        for i, (comp, group) in enumerate(df.groupby("component", sort=True)):
            ax.scatter(
                group["distance_km"],
                group["residual"],
                s=14,
                alpha=0.6,
                edgecolor="none",
                color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)],
                label=f"*{comp} ({len(group)})",
            )
        xm, q25, q50, q75 = _binned_median_iqr(
            df["distance_km"].to_numpy(float), df["residual"].to_numpy(float)
        )
        if xm.size:
            ax.fill_between(xm, q25, q75, color="0.3", alpha=0.2, label="Binned IQR")
            ax.plot(xm, q50, "o-", color="0.1", lw=1.5, ms=4, label="Binned median")
        ax.axhline(0.0, color="0.4", lw=1, ls="--")
        ax.legend(loc="best")
    ax.set_xlabel("Hypocentral distance (km)")
    ax.set_ylabel("Station ML - event ML")
    ax.set_title("Station ML residual vs distance (a trend = attenuation misfit)")
    save_fig(fig, path)


def plot_gr_mc_bvalue(mags, png_path: Path, csv_path: Path) -> GrStatistics:
    """``gr_mc_bvalue.png`` + ``gr_mc_bvalue.csv`` (module docstring)."""
    stats, boot = gr_statistics(mags)
    pd.DataFrame([asdict(stats)]).to_csv(csv_path, index=False)

    setup_style()
    import matplotlib.pyplot as plt

    arr = np.asarray(mags, dtype=float)
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots()
    if arr.size == 0:
        _no_data(ax)
    else:
        bw = stats.bin_width
        binned = _bin_magnitudes(arr, bw)
        values, counts = np.unique(binned, return_counts=True)
        cumulative = np.cumsum(counts[::-1])[::-1]
        ax.bar(values, counts, width=bw * 0.9, color="#56B4E9", alpha=0.7, label="Non-cumulative")
        ax.plot(values, cumulative, "s", color="#0072B2", ms=4, label="Cumulative N(>= M)")
        ax.axvline(stats.mc, color="#D55E00", ls="--", lw=1.2, label=f"Mc = {stats.mc:.1f}")
        ax.set_yscale("log")

        if stats.status == "ok":
            m_grid = np.linspace(stats.mc, float(values.max()) + bw / 2, 50)
            ax.plot(
                m_grid,
                10 ** (stats.a - stats.b * m_grid),
                color="#D55E00",
                lw=1.5,
                label=f"b = {stats.b:.2f}",
            )
            if len(boot) > 1:
                curves = 10 ** (boot[:, 2][:, None] - boot[:, 1][:, None] * m_grid[None, :])
                lo, hi = np.percentile(curves, [16, 84], axis=0)
                ax.fill_between(
                    m_grid, lo, hi, color="#D55E00", alpha=0.15, label="Bootstrap 16-84%"
                )
            text = (
                f"b = {stats.b:.2f} +/- {stats.b_std:.2f}\n"
                f"Mc = {stats.mc:.2f} +/- {stats.mc_std:.2f} (MAXC+{stats.mc_correction:.1f})\n"
                f"a = {stats.a:.2f}\n"
                f"N >= Mc = {stats.n_above_mc} / {stats.n_events}"
            )
        else:
            text = (
                f"Insufficient events for a b-value fit\n"
                f"(N >= Mc = {stats.n_above_mc} < {stats.min_events_for_fit})"
            )
        ax.text(
            0.97,
            0.97,
            text,
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.legend(loc="lower left")
    ax.set_xlabel("Magnitude (ML)")
    ax.set_ylabel("Number of events")
    ax.set_title("Gutenberg-Richter: Mc (max curvature) and MLE b-value")
    save_fig(fig, png_path)
    return stats


def _sample_event_ids(station_df: pd.DataFrame, n: int) -> list[str]:
    """Up to ``n`` evenly spaced event ids among events with a measurement."""
    if station_df is None or station_df.empty:
        return []
    ids = list(dict.fromkeys(station_df["event_id"].astype(str)))
    if len(ids) <= n:
        return ids
    idx = np.unique(np.linspace(0, len(ids) - 1, n).round().astype(int))
    return [ids[i] for i in idx]


def plot_amp_window(
    event: Event,
    captures: dict[tuple[str, str], WoodAndersonCapture],
    distances: dict[tuple[str, str], float],
    path: Path,
) -> None:
    """``amp_window_<event_id>.png``: one panel per captured station."""
    setup_style()
    import matplotlib.pyplot as plt

    origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
    keys = sorted(captures, key=lambda k: distances.get(k, math.inf))[:MAX_AMP_WINDOW_STATIONS]
    n = max(len(keys), 1)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 1.6 * n + 1.0), sharex=True, squeeze=False)
    axes = axes[:, 0]
    if not keys:
        _no_data(axes[0], "No station amplitudes")

    ref = origin.time if origin is not None else None
    for ax, key in zip(axes, keys, strict=False):
        cap = captures[key]
        ref_t = ref if ref is not None else cap.trace.stats.starttime
        p_pick = _find_pick(event, cap.net, cap.sta, "P")
        s_pick = _find_pick(event, cap.net, cap.sta, "S")
        t_lo = min(cap.window_start, p_pick.time if p_pick else cap.window_start) - 3.0
        t_hi = cap.window_end + 5.0
        tr = cap.trace.slice(t_lo, t_hi)
        t = tr.times(reftime=ref_t)
        ax.plot(t, np.asarray(tr.data) * 1e3, color="0.2", lw=0.7)
        ax.axvspan(cap.window_start - ref_t, cap.window_end - ref_t, color="#E69F00", alpha=0.25)
        i_peak = int(
            round((cap.peak_time - cap.trace.stats.starttime) * cap.trace.stats.sampling_rate)
        )
        peak_val = float(cap.trace.data[i_peak]) * 1e3
        ax.plot(cap.peak_time - ref_t, peak_val, "o", color="#D55E00", ms=5)
        for pick, colour, label in ((p_pick, "#0072B2", "P"), (s_pick, "#009E73", "S")):
            if pick is not None:
                ax.axvline(pick.time - ref_t, color=colour, lw=1.0)
                ax.text(
                    pick.time - ref_t,
                    0.04,
                    f" {label}",
                    color=colour,
                    transform=ax.get_xaxis_transform(),
                    va="bottom",
                    fontweight="bold",
                )
        ax.set_ylabel("WA (mm)", fontsize=8)
        ax.text(
            0.01,
            0.95,
            f"{cap.net}.{cap.sta}.{cap.channel}  {distances.get(key, float('nan')):.1f} km  "
            f"A = {cap.amplitude_m * 1e3:.3g} mm",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
            zorder=5,
        )
    axes[-1].set_xlabel("Time since origin (s)" if ref is not None else "Time (s)")
    fig.suptitle(f"{get_event_id(event)}: Wood-Anderson traces, window and peak")
    save_fig(fig, path)


def _amp_window_figures(
    paths: RunPaths, station_df: pd.DataFrame, catalog: Catalog, cfg: Config, out_dir: Path
) -> None:
    from obspy import read

    from openmeq.magnitude.pipeline import _cached_worker_inventory

    event_ids = _sample_event_ids(station_df, cfg.diagnostics.n_examples)
    if not event_ids:
        return
    events = {get_event_id(ev): ev for ev in catalog}
    inventory = _cached_worker_inventory(cfg)
    for eid in event_ids:
        try:
            event = events[eid]
            stream = read(str(paths.chunk_file(eid)))
            captures: dict[tuple[str, str], WoodAndersonCapture] = {}
            amps = measure_event_amplitudes(
                event, stream, inventory, cfg.magnitude, capture=captures
            )
            distances = {(sa.net, sa.sta): sa.distance_km for sa in amps}
            plot_amp_window(event, captures, distances, out_dir / f"amp_window_{eid}.png")
        except Exception as exc:  # noqa: BLE001 - extended QAQC never fails the step
            logger.warning("magnitude extended QAQC: amp_window_%s.png failed: %s", eid, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def make_magnitude_extended_diagnostics(
    paths: RunPaths, station_df: pd.DataFrame, catalog: Catalog, cfg: Config
) -> None:
    """Build every ``magnitude/diagnostics/extended/`` output (module docstring).

    The caller gates on ``cfg.diagnostics.extended_enabled("magnitude")``.
    Never raises: each figure's failure is logged as a warning.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    station_df : pandas.DataFrame
        :data:`openmeq.magnitude.pipeline.STATION_MAGNITUDES_COLUMNS`-shaped.
    catalog : obspy.Catalog
        The output catalogue (with this run's Magnitudes).
    cfg : openmeq.config.Config
        Reads ``cfg.diagnostics.n_examples``, plus ``cfg.magnitude`` and
        ``cfg.stations`` to re-measure the sampled events' traces.
    """
    out_dir = Path(paths.diagnostics_dir("magnitude")) / EXTENDED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    event_table = _event_ml_table(catalog)

    figures = (
        (
            "res_vs_distance.png",
            lambda: plot_residual_vs_distance(
                station_df, event_table, out_dir / "res_vs_distance.png"
            ),
        ),
        (
            "gr_mc_bvalue.png",
            lambda: plot_gr_mc_bvalue(
                event_table["ml"].to_numpy(dtype=float),
                out_dir / "gr_mc_bvalue.png",
                out_dir / "gr_mc_bvalue.csv",
            ),
        ),
        (
            "amp_window_<event_id>.png",
            lambda: _amp_window_figures(paths, station_df, catalog, cfg, out_dir),
        ),
    )
    for name, build in figures:
        try:
            build()
        except Exception as exc:  # noqa: BLE001 - extended QAQC never fails the step
            logger.warning("magnitude extended QAQC: %s failed: %s", name, exc)
