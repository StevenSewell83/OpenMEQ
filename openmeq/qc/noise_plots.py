"""Per-instrument-group noise figures (OpenMEQ-h8j), MicroQuake ``noise_daily`` style.

Reads the hourly noise columns of ``qc/daily_stats.csv`` (:mod:`openmeq.qc.noise`)
and the instrument table (:mod:`openmeq.qc.instruments`), writes
``qc/noise_daily.csv`` (levels converted to dB re 1 m/s where a sensitivity is
known) and, under ``qc/diagnostics/noise/``, for each instrument group:

- ``<group>_heatmap.png`` -- daily noise level (median hourly RMS) and daily
  transient spread (p95 - p5 of hourly RMS), station x day.
- ``<group>_strips.png`` -- one panel per station: median with a p5-p95 band.
- ``<group>_overlay.png`` -- every station of the group on one axis.

Each group gets its own colour and axis range, because levels are only
comparable between like instruments. ``impulsiveness.png`` puts every station
on one figure (hourly kurtosis and crest factor, p95 per day): both are
scale-invariant, so no response is needed to compare them.

Vertical channels only (the component code's last character is ``Z``), as in
the MicroQuake figures. Station labels carry a dagger when the sensitivity
was assumed or missing, and a double dagger when the group was set in
``cfg.qc.instrument_groups`` rather than read from StationXML.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from openmeq.log import get_logger
from openmeq.plotting import save_fig, setup_style
from openmeq.plotting._style import COLORBLIND_PALETTE

logger = get_logger("qc.noise_plots")

NOISE_DAILY_COLUMNS = [
    "date",
    "seed_id",
    "station",
    "channel",
    "group",
    "group_source",
    "sensitivity",
    "sens_flag",
    "units",
    "completeness",
    "n_windows",
    "rms_db_median",
    "rms_db_p5",
    "rms_db_p95",
    "spread_db",
    "ratio_low_core_db",
    "kurt_median",
    "kurt_p95",
    "crest_db_median",
    "crest_db_p95",
    "dead_frac",
]

_MISSING_GREY = "#d9d9d9"
_DASHES = ["-", (0, (5, 1.6)), (0, (1.4, 1.4))]
_LEVEL_COLS = ("rms_db_median", "rms_db_p5", "rms_db_p95")


def noise_table(daily_stats_df: pd.DataFrame, instruments_df: pd.DataFrame) -> pd.DataFrame:
    """Join the daily noise columns with the instrument table; convert levels to m/s.

    Dividing a waveform by its scalar sensitivity shifts every dB level by
    ``-20 log10(sensitivity)``; ratio, spread, kurtosis and crest factor are
    unchanged. Channels with no sensitivity stay in dB re 1 count.
    """
    noise_cols = [c for c in daily_stats_df.columns if c.startswith("noise_")]
    df = daily_stats_df[["seed_id", "date", *noise_cols]].rename(
        columns={c: c[len("noise_") :] for c in noise_cols}
    )
    df = df.merge(instruments_df, on="seed_id", how="left")
    parts = df["seed_id"].str.split(".", expand=True)
    df["station"] = df["station"].fillna(parts[1])
    df["channel"] = df["channel"].fillna(parts[3])
    df["group"] = df["group"].fillna("unknown")
    df["group_source"] = df["group_source"].fillna("unknown")
    df["sens_flag"] = df["sens_flag"].fillna("missing")
    has_sens = df["sensitivity"].notna() & (df["sensitivity"] > 0)
    shift = np.where(has_sens, 20.0 * np.log10(df["sensitivity"].where(has_sens, 1.0)), 0.0)
    for col in _LEVEL_COLS:
        df[col] = df[col] - shift
    units = df["sensitivity_units"].fillna("").str.lower()
    df["units"] = np.where(has_sens, "dB re 1 " + units.where(units != "", "m/s"), "dB re 1 count")
    return df[NOISE_DAILY_COLUMNS].sort_values(["group", "station", "seed_id", "date"])


def write_noise_daily_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=NOISE_DAILY_COLUMNS, float_format="%.4g")


def _slug(group: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")
    return slug or "group"


def _row_labels(sub: pd.DataFrame) -> dict[str, str]:
    """seed_id -> label: the station, or STA.LOC when a station has several Z channels."""
    per_station = sub.groupby("station")["seed_id"].nunique()
    labels = {}
    for seed_id, station, flag, source in (
        sub[["seed_id", "station", "sens_flag", "group_source"]]
        .drop_duplicates("seed_id")
        .itertuples(index=False)
    ):
        label = station if per_station[station] == 1 else f"{station}.{seed_id.split('.')[2]}"
        if flag != "ok":
            label += " †"
        if source == "config":
            label += " ‡"
        labels[seed_id] = label
    return labels


def _pivot(
    sub: pd.DataFrame, col: str, seed_ids: list[str], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    return sub.pivot_table(index="seed_id", columns="date", values=col, aggfunc="last").reindex(
        index=seed_ids, columns=dates
    )


def _robust_range(values: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    return float(vmin), float(vmax)


def _unit_label(sub: pd.DataFrame) -> str:
    units = sub["units"].dropna().unique()
    return units[0] if len(units) == 1 else " / ".join(sorted(units))


def _heatmap_panel(ax, fig, mat, dates, labels, *, cmap, label, title, norm=None, vrange=None):
    ax.set_facecolor(_MISSING_GREY)
    ax.grid(False)  # the shared style's grid would draw over the grey missing cells
    if np.isfinite(mat.values).any():
        kwargs = {"norm": norm} if norm is not None else {"vmin": vrange[0], "vmax": vrange[1]}
        im = ax.pcolormesh(
            dates, np.arange(len(labels)), mat.values, cmap=cmap, shading="nearest", **kwargs
        )
        fig.colorbar(im, ax=ax, pad=0.01, label=label)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _plot_group_heatmap(sub, seed_ids, label_of, dates, group, title, out_path, noise_cfg) -> None:
    labels = [label_of[s] for s in seed_ids]
    level = _pivot(sub, "rms_db_median", seed_ids, dates)
    comp = _pivot(sub, "completeness", seed_ids, dates)
    dead = _pivot(sub, "dead_frac", seed_ids, dates)
    level = level.mask(comp < 0.5)
    spread = _pivot(sub, "spread_db", seed_ids, dates)
    band = f"{noise_cfg.core_band[0]:g}–{noise_cfg.core_band[1]:g} Hz"
    unit = _unit_label(sub)

    height = max(4.0, 1.6 + 0.32 * len(seed_ids))
    fig, axes = plt.subplots(2, 1, figsize=(14, 2 * height), sharex=True)
    for ax, mat, label, ttl in (
        (axes[0], level, f"median hourly RMS {band} ({unit})", "Daily noise level"),
        (axes[1], spread, "p95 − p5 of hourly RMS (dB)", "Daily transient spread"),
    ):
        vrange = _robust_range(mat.values) or (0.0, 1.0)
        _heatmap_panel(
            ax, fig, mat, dates, labels, cmap="coolwarm", label=label, title=ttl, vrange=vrange
        )
        bad = (dead > 0.2) & comp.notna()
        ys, xs = np.where(bad.values)
        if len(xs):
            ax.scatter(dates[xs], ys, marker="x", s=12, color="black", linewidths=1)
    notes = ["grey = missing/<50% complete", "black x = dead windows >20%"]
    if any("†" in v for v in labels):
        notes.insert(0, "† sensitivity assumed/missing")
    if any("‡" in v for v in labels):
        notes.insert(0, "‡ group set in config")
    axes[0].text(
        1.0,
        1.02,
        " · ".join(notes),
        ha="right",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#555",
    )
    fig.suptitle(f"{title} — {group} daily noise statistics (vertical channels)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_path)


def _plot_group_strips(sub, seed_ids, label_of, dates, group, title, out_path, noise_cfg) -> None:
    ncols = min(4, len(seed_ids))
    nrows = int(np.ceil(len(seed_ids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.3 * nrows + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    yrange = _robust_range(sub["rms_db_median"].to_numpy(dtype=float), 0.5, 99.5)
    line, fill = COLORBLIND_PALETTE[0], "#9ecae1"
    for ax, seed_id in zip(axes.flat[: len(seed_ids)], seed_ids, strict=True):
        d = sub[sub.seed_id == seed_id].set_index("date").reindex(dates)
        ax.fill_between(d.index, d.rms_db_p5, d.rms_db_p95, color=fill, alpha=0.6, lw=0)
        ax.plot(d.index, d.rms_db_median, color=line, lw=1.2)
        ax.set_title(label_of[seed_id], fontsize=9, loc="left", color=line)
        ax.grid(alpha=0.25)
        if yrange:
            ax.set_ylim(yrange[0] - 3, yrange[1] + 3)
    for ax in axes.flat[len(seed_ids) :]:
        ax.set_visible(False)
    # sharex hides tick labels on all but the last row; a part-filled last row
    # leaves columns whose lowest visible panel is a row up -- relabel those.
    for col in range(ncols):
        for row in range(nrows - 1, -1, -1):
            ax = axes[row, col]
            if ax.get_visible():
                ax.tick_params(labelbottom=True)
                ax.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
                )
                break
    band = f"{noise_cfg.core_band[0]:g}–{noise_cfg.core_band[1]:g} Hz"
    fig.suptitle(
        f"{title} — {group}: daily median hourly RMS {band}, {_unit_label(sub)}"
        " (band = p5–p95 across hours)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_path)


def _plot_group_overlay(sub, seed_ids, label_of, dates, group, title, out_path, noise_cfg) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ends = []
    for i, seed_id in enumerate(seed_ids):
        colour = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        dash = _DASHES[(i // len(COLORBLIND_PALETTE)) % len(_DASHES)]
        d = sub[sub.seed_id == seed_id].set_index("date").reindex(dates)["rms_db_median"]
        ax.plot(d.index, d.values, color=colour, ls=dash, lw=1.4, label=label_of[seed_id])
        last = d.dropna()
        if len(last):
            ends.append((last.index[-1], float(last.iloc[-1]), label_of[seed_id], colour))
    for x, y, text, colour in ends:
        ax.plot([x], [y], "o", color=colour, ms=4)
        ax.annotate(
            text,
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=colour,
        )
    yrange = _robust_range(sub["rms_db_median"].to_numpy(dtype=float), 0.5, 99.5)
    if yrange:
        ax.set_ylim(yrange[0] - 3, yrange[1] + 3)
    band = f"{noise_cfg.core_band[0]:g}–{noise_cfg.core_band[1]:g} Hz"
    ax.set_ylabel(f"median hourly RMS {band} ({_unit_label(sub)})")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if len(seed_ids) > 1:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(8, len(seed_ids)),
            fontsize=8,
            frameon=False,
        )
    ax.set_title(f"{title} — {group}: daily median hourly RMS, one line per station", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_path)


def _plot_impulsiveness(dfz, dates, title, out_path) -> None:
    order = dfz.drop_duplicates("seed_id").sort_values(["group", "station", "seed_id"])
    seed_ids = order.seed_id.tolist()
    labels_by_group = {}
    for _group, grp in dfz.groupby("group"):
        labels_by_group.update(_row_labels(grp))
    labels = [f"{labels_by_group[s]}  [{g}]" for s, g in zip(seed_ids, order.group, strict=True)]

    kurt = _pivot(dfz, "kurt_p95", seed_ids, dates)
    crest = _pivot(dfz, "crest_db_p95", seed_ids, dates)
    height = max(4.0, 1.6 + 0.3 * len(seed_ids))
    fig, axes = plt.subplots(2, 1, figsize=(14, 2 * height), sharex=True)
    kvals = kurt.values[np.isfinite(kurt.values)]
    kmax = max(float(np.percentile(kvals, 98)), 10.0) if kvals.size else 10.0
    _heatmap_panel(
        axes[0],
        fig,
        kurt,
        dates,
        labels,
        cmap="coolwarm",
        label="kurtosis (log scale)",
        title="Impulsiveness: p95 of hourly kurtosis (Gaussian noise = 3, log scale)",
        norm=LogNorm(3.0, kmax),
    )
    _heatmap_panel(
        axes[1],
        fig,
        crest,
        dates,
        labels,
        cmap="coolwarm",
        label="crest factor (dB)",
        title="Impulsiveness: p95 of hourly crest factor (peak/RMS; Gaussian hour ≈ 12–13 dB)",
        vrange=_robust_range(crest.values) or (10.0, 30.0),
    )
    # Thin lines between instrument groups.
    boundaries = np.flatnonzero(order.group.to_numpy()[1:] != order.group.to_numpy()[:-1]) + 0.5
    for ax in axes:
        for b in boundaries:
            ax.axhline(b, color="white", lw=1.5)
    fig.suptitle(
        f"{title} spiky/transient noise"
        " (scale-invariant — gain cancels, so all instruments share one scale)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_path)


def make_noise_diagnostics(
    noise_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    start: date,
    end: date,
    noise_cfg,
    title: str = "",
) -> list[Path]:
    """Write every noise figure for ``noise_df`` (:func:`noise_table`'s output) into ``out_dir``.

    Returns the paths written. Groups with no finite level are skipped.
    """
    setup_style()
    out_dir = Path(out_dir)
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    dfz = noise_df[noise_df.channel.str.endswith("Z")].copy()
    dfz["date"] = pd.to_datetime(dfz["date"])
    written: list[Path] = []
    if dfz.empty:
        logger.info("noise plots: no vertical channels, nothing to plot")
        return written

    for group, sub in dfz.groupby("group", sort=True):
        if not np.isfinite(sub["rms_db_median"].to_numpy(dtype=float)).any():
            logger.info("noise plots: group %s has no usable noise levels, skipped", group)
            continue
        seed_ids = (
            sub.drop_duplicates("seed_id").sort_values(["station", "seed_id"]).seed_id.tolist()
        )
        label_of = _row_labels(sub)
        for kind, plot in (
            ("heatmap", _plot_group_heatmap),
            ("strips", _plot_group_strips),
            ("overlay", _plot_group_overlay),
        ):
            path = out_dir / f"{_slug(group)}_{kind}.png"
            plot(sub, seed_ids, label_of, dates, group, title, path, noise_cfg)
            written.append(path)

    path = out_dir / "impulsiveness.png"
    _plot_impulsiveness(dfz, dates, title, path)
    written.append(path)
    logger.info("noise plots: wrote %d figure(s) to %s", len(written), out_dir)
    return written
