"""QC diagnostic plots (M1.6, ``OpenMEQ-5nl.6``).

Renders PLAN.md §5's QC diagnostics row ("Availability heatmap; daily
statistics time series per channel; monthly station box plots and
rankings") plus a flag overview, from the QC tables already produced by
:mod:`openmeq.qc.availability` (M1.3), :mod:`openmeq.qc.stats` (M1.4) and
:mod:`openmeq.qc.flags` (M1.5) -- this module never recomputes QC, it only
visualises it. A short ``summary.csv`` (also PLAN.md §5: "a short text/CSV
summary ... generated automatically at the end of the step") is written
alongside the plots.

.. note::
   **Deviation from the original beads spec (documented per the worker
   instructions).** ``OpenMEQ-5nl.6``'s original text calls for
   ``openmeq/qc/diagnostics.py`` with a signature taking a ``RunPaths`` and
   a full ``openmeq.config.Config``. The build orchestrator's task note for
   this worker superseded that: the module lives at ``openmeq/qc/plots.py``
   and works **only from plain DataFrames/dicts and an output directory**,
   with no ``openmeq.config``/``openmeq.paths`` import at all. This mirrors
   the existing precedent in ``tests/test_qc_monthly_flags.py`` (a full
   ``Config`` requires unrelated required sections -- ``project``, ``data``,
   ``nlloc``, ``hypodd`` -- with no bearing on this module) and keeps this
   module's tests fast and independent of the concurrently-changing
   ``openmeq.qc.stats``/``openmeq.qc.flags`` column sets. ``OpenMEQ-5nl.7``
   (CLI wiring) is the natural place to adapt ``cfg``/``paths`` into the
   plain arguments here (``paths.qc_diagnostics_dir``, a plain list of
   stat names from ``cfg.qc.daily_stats``, the loaded DataFrames/dict).

Every figure goes through :mod:`openmeq.plotting`'s helpers (``setup_style``,
``plot_grouped_heatmap``) -- nothing here imports ``matplotlib`` directly,
so styling stays consistent with every other step's diagnostics.

Rows, labels and instrument groups (OpenMEQ-gyb)
-------------------------------------------------
Every figure is a heatmap with one labelled row per channel (``STA.CHA``,
or ``STA.LOC.CHA`` when a station has several location codes), rows grouped
by instrument group (``instruments``: the per-channel ``group`` and
``sensitivity`` from :func:`openmeq.qc.noise_plots.noise_table`, i.e. the
StationXML sensor model or ``cfg.qc.instrument_groups``), with a white line
and the group name between groups. Without ``instruments`` every channel is
in one ``all`` group.

Statistics whose value scales with the instrument's gain
(:data:`GAIN_DEPENDENT_STATS`: ``rms``, ``mean_abs``) are never put on one
colour scale across instruments:

- they get no ``daily_<stat>.png`` -- the per-group, sensitivity-corrected
  noise heatmaps in ``diagnostics/noise/`` already show amplitude per day;
- their monthly ratio is computed on ``value / sensitivity`` against the
  median of the channel's own instrument group; a group with a single
  station (no peers) is compared with its own median over the run instead
  (rows marked ``*``).

Gain-independent statistics (kurtosis, spike fraction, flat/clip counts)
share one figure across every instrument, with network-wide ratios.

Outputs (``qc/diagnostics/``, ARCHITECTURE.md §3)
--------------------------------------------------
- ``availability_heatmap.png``: channel x date ``coverage_pct`` (0-100).
- ``daily_<stat>.png``: channel x date, one per gain-independent stat;
  ``spike_count`` is shown as the spike *fraction* (``spike_count / npts``)
  and ``kurtosis`` as Pearson kurtosis (excess + 3; Gaussian = 3), both on
  log colour scales.
- ``monthly_<stat>.png``: channel x month ratio to the median (log colour
  scale centred on 1x, value written in each cell), see above.
- ``flag_overview.png``: channel x date, each day coloured by its
  highest-priority flag reason (:data:`FLAG_CATEGORIES`) or ok.
- ``summary.csv``: ``seed_id, pct_days_ok, worst_reason, n_days_flagged``,
  one row per ``seed_id`` present in ``flags``.

:func:`make_qc_diagnostics` returns the PNG paths it wrote, so the caller can
remove figures an earlier version wrote (e.g. ``daily_rms.png``).

Tolerance to changing stat columns
-----------------------------------
Two other in-flight tasks touch the stats this module plots
(``OpenMEQ-5nl.8`` and ``OpenMEQ-5nl.9``, which may redefine
``spike_count`` and add a longest-flat-run column). This module never
hardcodes a stat name: the default stat list is *discovered* from whatever
columns/``stat`` values the input DataFrames actually have, and an
explicit ``stats`` argument that names a column/stat missing from a given
DataFrame is skipped with a warning rather than raising.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import plot_grouped_heatmap, setup_style

logger = get_logger("qc.plots")

#: Columns of ``qc/daily_stats.csv`` that are never themselves a "stat" to
#: plot (ARCHITECTURE.md §3).
_DAILY_STATS_NON_STAT_COLUMNS = frozenset({"seed_id", "date", "npts"})

#: ``qc/diagnostics/summary.csv``'s exact column order.
SUMMARY_COLUMNS = ["seed_id", "pct_days_ok", "worst_reason", "n_days_flagged"]

#: Daily stats whose value scales with instrument gain (module docstring).
GAIN_DEPENDENT_STATS = frozenset({"rms", "mean_abs"})

#: Flag overview categories ``(value, colour, label, reason prefix)``,
#: highest priority first after ``ok``; a day with several reasons shows
#: the first match.
FLAG_CATEGORIES = [
    (0, "#a1d99b", "ok", None),
    (1, "#252525", "excluded by config", "excluded_by_config"),
    (2, "#3182bd", "low availability", "availability"),
    (3, "#756bb1", "flat", "flat"),
    (4, "#e7298a", "clipped", "clipped"),
    (5, "#de2d26", "spikes", "spikes"),
    (6, "#fdae6b", "network outlier", "network_outlier"),
    (7, "#969696", "other", ""),
]

#: Monthly ratio colour range: log10(ratio) in [-1, 1] = 0.1x .. 10x.
_RATIO_TICKS = {-1.0: "0.1×", -0.5: "0.3×", 0.0: "1×", 0.5: "3×", 1.0: "10×"}


def _default_daily_stats(daily_stats_df: pd.DataFrame) -> list[str]:
    if daily_stats_df is None or daily_stats_df.empty:
        return []
    return [c for c in daily_stats_df.columns if c not in _DAILY_STATS_NON_STAT_COLUMNS]


def _default_monthly_stats(monthly_df: pd.DataFrame) -> list[str]:
    if monthly_df is None or monthly_df.empty or "stat" not in monthly_df.columns:
        return []
    return sorted(monthly_df["stat"].dropna().unique().tolist())


def _row_table(seed_ids, instruments: pd.DataFrame | None) -> pd.DataFrame:
    """One row per channel in display order: ``seed_id, station, group,
    sensitivity, label`` (module docstring's "Rows, labels and instrument
    groups"). Groups are ordered largest first, then by name."""
    columns = ["seed_id", "station", "location", "channel", "group", "sensitivity", "label"]
    if not seed_ids:
        return pd.DataFrame(columns=columns)
    rows = pd.DataFrame({"seed_id": sorted({str(s) for s in seed_ids})})
    parts = rows["seed_id"].str.split(".", expand=True).reindex(columns=range(4))
    rows["station"], rows["location"], rows["channel"] = parts[1], parts[2].fillna(""), parts[3]
    rows["group"], rows["sensitivity"] = "all", np.nan
    if instruments is not None and not instruments.empty and "group" in instruments:
        meta = instruments.drop_duplicates("seed_id").set_index("seed_id")
        rows["group"] = rows["seed_id"].map(meta["group"]).fillna("unknown").astype(str)
        if "sensitivity" in meta:
            rows["sensitivity"] = rows["seed_id"].map(meta["sensitivity"]).astype(float)
    multi_loc = rows.groupby("station")["location"].transform("nunique") > 1
    rows["label"] = np.where(
        multi_loc,
        rows["station"] + "." + rows["location"] + "." + rows["channel"],
        rows["station"] + "." + rows["channel"],
    )
    size = rows.groupby("group")["seed_id"].transform("size")
    rows = rows.assign(_size=-size).sort_values(
        ["_size", "group", "station", "location", "channel"]
    )
    return rows.drop(columns="_size").reset_index(drop=True)


def _title(prefix: str, text: str) -> str:
    return f"{prefix} — {text}" if prefix else text


def _date_pivot(df: pd.DataFrame, value: str, rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` x every date in ``df`` (a ``DatetimeIndex``) matrix of ``value``."""
    sub = df[["seed_id", "date", value]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    pivot = sub.pivot_table(index="seed_id", columns="date", values=value, aggfunc="mean")
    dates = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
    return pivot.reindex(index=rows["seed_id"], columns=dates)


def _heatmap(matrix: pd.DataFrame, rows: pd.DataFrame, path: Path, labels=None, **kwargs) -> None:
    plot_grouped_heatmap(
        matrix,
        path,
        row_labels=list(rows["label"]) if labels is None else labels,
        row_groups=list(rows["group"]),
        **kwargs,
    )


def _availability_heatmap(availability_df, rows, out_dir: Path, title: str) -> Path:
    path = out_dir / "availability_heatmap.png"
    _heatmap(
        _date_pivot(availability_df, "coverage_pct", rows),
        rows,
        path,
        title=_title(title, "data availability per channel-day"),
        cmap="viridis",
        vmin=0.0,
        vmax=100.0,
        cbar_label="coverage (%)",
    )
    return path


def _daily_stat_heatmap(daily_stats_df, stat: str, rows, out_dir: Path, title: str) -> Path:
    path = out_dir / f"daily_{stat}.png"
    df = daily_stats_df.copy()
    log, label, text = False, stat, f"daily {stat}"
    if stat == "spike_count" and "npts" in df.columns:
        df[stat] = df[stat] / df["npts"].where(df["npts"] > 0)
        log, label = True, "spike fraction (log scale)"
        text = "daily spike fraction (spike_count / samples)"
    elif stat == "kurtosis":
        df[stat] = df[stat] + 3.0
        log, label = True, "kurtosis (log scale)"
        text = "daily kurtosis (Pearson; Gaussian noise = 3)"
    _heatmap(
        _date_pivot(df, stat, rows),
        rows,
        path,
        title=_title(title, text),
        note="gain-independent: one scale for all instruments",
        cmap="magma_r" if log else "viridis",
        log=log,
        cbar_label=label,
    )
    return path


def _monthly_ratios(monthly_df: pd.DataFrame, stat: str, rows: pd.DataFrame):
    """``(seed_id x month ratio matrix, own-history seed ids)`` for one stat."""
    sub = monthly_df[monthly_df["stat"] == stat][["seed_id", "month", "value", "ratio"]].copy()
    sub["month"] = sub["month"].astype(str)
    own = set()
    if stat in GAIN_DEPENDENT_STATS:
        meta = rows.set_index("seed_id")
        sens = sub["seed_id"].map(meta["sensitivity"]).astype(float)
        sub["corr"] = sub["value"] / sens.where(sens > 0, 1.0)
        sub["group"] = sub["seed_id"].map(meta["group"]).fillna("unknown")
        n_sta = rows.groupby("group")["station"].nunique()
        peers = sub["group"].map(n_sta).fillna(0) >= 2
        group_median = sub.groupby(["group", "month"])["corr"].transform("median")
        own_median = sub.groupby("seed_id")["corr"].transform("median")
        sub["ratio"] = sub["corr"] / group_median.where(peers, own_median)
        own = set(sub.loc[~peers, "seed_id"])
    matrix = sub.pivot_table(index="seed_id", columns="month", values="ratio", aggfunc="mean")
    return matrix.reindex(index=rows["seed_id"], columns=sorted(matrix.columns)), own


def _monthly_stat_heatmap(monthly_df, stat: str, rows, out_dir: Path, title: str) -> Path:
    path = out_dir / f"monthly_{stat}.png"
    if monthly_df is None or monthly_df.empty or "stat" not in monthly_df.columns:
        ratio, own = pd.DataFrame(index=rows["seed_id"]), set()
    else:
        ratio, own = _monthly_ratios(monthly_df, stat, rows)
    values = ratio.to_numpy(dtype=float)
    log_ratio = np.log10(np.where(values > 0, values, np.nan))
    annotate = [[f"{v:.2g}" if np.isfinite(v) and v > 0 else "" for v in row] for row in values]
    labels = [
        f"{lab} *" if s in own else lab
        for s, lab in zip(rows["seed_id"], rows["label"], strict=True)
    ]
    if stat in GAIN_DEPENDENT_STATS:
        text = f"monthly {stat} (sensitivity-corrected) vs instrument-group median"
        note = "* single-station group: vs its own median over the run"
    else:
        text = f"monthly {stat} vs network median"
        note = "gain-independent: all instruments compared"
    _heatmap(
        pd.DataFrame(log_ratio, index=ratio.index, columns=ratio.columns),
        rows,
        path,
        labels=labels,
        title=_title(title, text),
        note=note if own or stat not in GAIN_DEPENDENT_STATS else None,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        cbar_label="ratio to median (log scale)",
        cbar_ticks=_RATIO_TICKS,
        annotate=annotate,
    )
    return path


def _flag_category(info: dict) -> int:
    if info.get("ok"):
        return 0
    reasons = info.get("reasons", [])
    for value, _colour, _label, prefix in FLAG_CATEGORIES[1:-1]:
        if any(r.startswith(prefix) for r in reasons):
            return value
    return FLAG_CATEGORIES[-1][0]


def _flag_overview(flags: dict, rows, out_dir: Path, title: str) -> Path:
    path = out_dir / "flag_overview.png"
    long_df = pd.DataFrame(
        [
            {"seed_id": seed_id, "date": date, "category": _flag_category(info)}
            for seed_id, days in (flags or {}).items()
            for date, info in days.items()
        ],
        columns=["seed_id", "date", "category"],
    )
    matrix = (
        _date_pivot(long_df, "category", rows)
        if not long_df.empty
        else pd.DataFrame(index=rows["seed_id"])
    )
    _heatmap(
        matrix,
        rows,
        path,
        title=_title(title, "QC flags per channel-day"),
        note="colour = highest-priority reason; counts in summary.csv",
        categories=[(v, c, lab) for v, c, lab, _p in FLAG_CATEGORIES],
    )
    return path


def summarize_flags(flags: dict) -> pd.DataFrame:
    """Build the ``summary.csv`` DataFrame from a ``channel_flags.json``-shaped
    ``flags`` dict (ARCHITECTURE.md §4.2's ``"flags"`` inner mapping --
    :func:`openmeq.qc.flags.compute_channel_flags`'s return value, *not* the
    on-disk envelope with the ``"version"`` key).

    One row per ``seed_id`` present in ``flags``:

    - ``pct_days_ok``: 100 * (days with ``ok: true``) / (total days present
      for that ``seed_id``), in ``[0, 100]``. ``nan`` if the channel has no
      days at all (should not happen in practice -- ``flags`` only ever
      gets a ``seed_id`` key when it has at least one day).
    - ``worst_reason``: the most frequent single reason string among that
      channel's flagged (``ok: false``) days; ``"none"`` if the channel is
      never flagged (not ``""``/``NaN`` -- those round-trip ambiguously
      through a CSV re-read, e.g. ``pandas.read_csv``'s default
      ``keep_default_na=True`` reads an empty cell back as ``NaN``, which
      would misleadingly suggest missing data rather than "no reason").
      Ties broken alphabetically for determinism.
    - ``n_days_flagged``: count of days with ``ok: false``.

    Parameters
    ----------
    flags : dict
        ``{seed_id: {date: {"ok": bool, "reasons": [str, ...]}}}``.

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`SUMMARY_COLUMNS`, sorted by ``seed_id``.
    """
    rows = []
    for seed_id, days in (flags or {}).items():
        n_total = len(days)
        n_ok = sum(1 for info in days.values() if info.get("ok"))
        n_flagged = n_total - n_ok
        pct_days_ok = 100.0 * n_ok / n_total if n_total > 0 else float("nan")

        reason_counts: Counter[str] = Counter()
        for info in days.values():
            if not info.get("ok"):
                reason_counts.update(info.get("reasons", []))
        if reason_counts:
            max_count = max(reason_counts.values())
            worst_reason = sorted(r for r, c in reason_counts.items() if c == max_count)[0]
        else:
            worst_reason = "none"

        rows.append(
            {
                "seed_id": seed_id,
                "pct_days_ok": pct_days_ok,
                "worst_reason": worst_reason,
                "n_days_flagged": n_flagged,
            }
        )

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return df.sort_values("seed_id", kind="stable").reset_index(drop=True)


def make_qc_diagnostics(
    availability_df: pd.DataFrame,
    daily_stats_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    flags: dict,
    out_dir: str | Path,
    *,
    stats: Sequence[str] | None = None,
    instruments: pd.DataFrame | None = None,
    title: str = "",
) -> list[Path]:
    """Render every QC diagnostic plot plus ``summary.csv`` into ``out_dir``.

    Parameters
    ----------
    availability_df : pandas.DataFrame
        ``qc/availability.csv``-shaped: at least ``seed_id``, ``date``,
        ``coverage_pct`` (:data:`openmeq.qc.availability.COLUMNS`).
    daily_stats_df : pandas.DataFrame
        ``qc/daily_stats.csv``-shaped: at least ``seed_id``, ``date``, plus
        one column per daily stat (:data:`openmeq.qc.stats.DAILY_STATS_COLUMNS`).
    monthly_df : pandas.DataFrame
        ``qc/monthly_stats.csv``-shaped: ``seed_id, month, stat, value,
        network_median, ratio, rank`` (:data:`openmeq.qc.flags.MONTHLY_STATS_COLUMNS`).
    flags : dict
        ``qc/channel_flags.json``'s ``"flags"`` inner mapping (i.e.
        :func:`openmeq.qc.flags.compute_channel_flags`'s return value --
        *not* the on-disk envelope with the top-level ``"version"`` key).
    out_dir : str or pathlib.Path
        Output directory (``qc/diagnostics/``, ARCHITECTURE.md §3). Created
        if missing.
    stats : sequence of str, optional
        Which statistics to render ``daily_<stat>.png`` (gain-independent
        stats only) and ``monthly_<stat>.png`` for (e.g. ``cfg.qc.daily_stats``).
        Defaults to every stat column in ``daily_stats_df`` / every ``stat``
        in ``monthly_df``. A stat missing from a DataFrame is skipped with a
        warning.
    instruments : pandas.DataFrame, optional
        Per-channel ``seed_id``, ``group`` and ``sensitivity`` (e.g.
        ``noise_daily.csv``), for the instrument grouping (module docstring).
    title : str, optional
        Prefix for every figure title (the project name).

    Returns
    -------
    list of pathlib.Path
        The PNGs written.
    """
    setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_stats = list(stats) if stats is not None else _default_daily_stats(daily_stats_df)
    monthly_stats = list(stats) if stats is not None else _default_monthly_stats(monthly_df)
    seed_ids = set()
    for df in (availability_df, daily_stats_df):
        if df is not None and not df.empty:
            seed_ids |= set(df["seed_id"].astype(str))
    seed_ids |= set(flags or {})
    rows = _row_table(seed_ids, instruments)

    written = []
    if availability_df is not None and not availability_df.empty:
        written.append(_availability_heatmap(availability_df, rows, out_dir, title))
    for stat in daily_stats:
        if stat in GAIN_DEPENDENT_STATS:
            logger.info("qc.plots: no daily_%s.png (gain-dependent; see diagnostics/noise/)", stat)
        elif daily_stats_df is None or stat not in getattr(daily_stats_df, "columns", []):
            logger.warning("qc.plots: stat %r not in daily_stats_df -- daily plot skipped", stat)
        elif not daily_stats_df.empty:
            written.append(_daily_stat_heatmap(daily_stats_df, stat, rows, out_dir, title))
    present = set(_default_monthly_stats(monthly_df))
    for stat in monthly_stats:
        if stat not in present:
            logger.warning("qc.plots: stat %r not in monthly_df -- monthly plot skipped", stat)
            continue
        written.append(_monthly_stat_heatmap(monthly_df, stat, rows, out_dir, title))
    if flags:
        written.append(_flag_overview(flags, rows, out_dir, title))

    summary_df = summarize_flags(flags)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, columns=SUMMARY_COLUMNS)
    return written


__all__ = [
    "FLAG_CATEGORIES",
    "GAIN_DEPENDENT_STATS",
    "SUMMARY_COLUMNS",
    "make_qc_diagnostics",
    "summarize_flags",
]
