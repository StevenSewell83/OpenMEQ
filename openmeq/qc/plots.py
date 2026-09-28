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
``plot_heatmap``, ``plot_timeseries``, ``plot_boxplot``, ``save_fig``) --
nothing here imports ``matplotlib``/``matplotlib.pyplot`` directly, so
styling stays consistent with every other step's diagnostics.

Outputs (``qc/diagnostics/``, ARCHITECTURE.md §3)
--------------------------------------------------
- ``availability_heatmap.png``: ``seed_id`` x ``date`` matrix of
  ``coverage_pct`` (0-100), from ``availability_df``.
- ``daily_<stat>.png``: one time series per stat (default: every column of
  ``daily_stats_df`` other than ``seed_id``/``date``/``npts``, or the
  ``stats`` argument if given), one line per ``seed_id``.
- ``monthly_<stat>_boxplot.png``: one box plot per stat (default: every
  ``stat`` value present in ``monthly_df``, or ``stats`` if given), one box
  per station of that station's monthly ``ratio`` (value / network median)
  across every month in ``monthly_df``. **Rankings are folded into this
  same plot** (per the beads issue's design note: "don't over-engineer
  this into a separate table format") via the box plots' left-to-right
  *order*: stations are ordered by their most recent month's ``rank``
  (rank 1, the most anomalous station that month, drawn first), not
  alphabetically -- so the figure shows both the distribution (box plot)
  and the current ranking (x-axis order) in one image. A station with no
  row in the latest month is placed after every ranked station, in
  ``seed_id`` order.
- ``flag_overview.png``: ``seed_id`` x ``date`` matrix of ``ok`` (1.0) /
  flagged (0.0) from ``flags`` -- a companion to the availability heatmap
  that reflects *every* flag reason (availability, spikes, flat, clipped,
  excluded-by-config, network-outlier), not just raw coverage.
- ``summary.csv``: ``seed_id, pct_days_ok, worst_reason, n_days_flagged``,
  one row per ``seed_id`` present in ``flags``.

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

import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import plot_boxplot, plot_heatmap, plot_timeseries, setup_style

logger = get_logger("qc.plots")

#: Columns of ``qc/daily_stats.csv`` that are never themselves a "stat" to
#: plot a time series of (ARCHITECTURE.md §3).
_DAILY_STATS_NON_STAT_COLUMNS = frozenset({"seed_id", "date", "npts"})

#: ``qc/diagnostics/summary.csv``'s exact column order.
SUMMARY_COLUMNS = ["seed_id", "pct_days_ok", "worst_reason", "n_days_flagged"]


def _default_daily_stats(daily_stats_df: pd.DataFrame) -> list[str]:
    if daily_stats_df is None or daily_stats_df.empty:
        return []
    return [c for c in daily_stats_df.columns if c not in _DAILY_STATS_NON_STAT_COLUMNS]


def _default_monthly_stats(monthly_df: pd.DataFrame) -> list[str]:
    if monthly_df is None or monthly_df.empty or "stat" not in monthly_df.columns:
        return []
    return sorted(monthly_df["stat"].dropna().unique().tolist())


def _availability_heatmap(availability_df: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "availability_heatmap.png"
    if availability_df is None or availability_df.empty:
        pivot = pd.DataFrame()
    else:
        pivot = availability_df.pivot_table(
            index="seed_id", columns="date", values="coverage_pct", aggfunc="mean"
        ).sort_index()
        pivot = pivot[sorted(pivot.columns)]
    plot_heatmap(
        pivot,
        path,
        xlabel="Date",
        ylabel="Seed ID",
        title="QC availability (% coverage)",
        cbar_label="Coverage (%)",
        vmin=0.0,
        vmax=100.0,
    )


def _daily_stat_timeseries(
    daily_stats_df: pd.DataFrame, stats: Sequence[str], out_dir: Path
) -> None:
    for stat in stats:
        path = out_dir / f"daily_{stat}.png"
        if daily_stats_df is None or daily_stats_df.empty or stat not in daily_stats_df.columns:
            logger.warning(
                "qc.plots: stat %r not present in daily_stats_df -- writing an empty "
                "'daily_%s.png' rather than skipping it (a concurrently-changing M1.4/M1.8/"
                "M1.9 daily_stats.csv may legitimately omit or rename it).",
                stat,
                stat,
            )
            plot_timeseries(pd.DataFrame(columns=["date", stat]), path, x="date", y=stat)
            continue
        sub = daily_stats_df[["seed_id", "date", stat]].copy()
        # A true datetime x-axis (rather than one categorical tick per
        # ISO-string date) is what keeps this legible at e.g. 365 days --
        # matplotlib's date locator thins ticks automatically, whereas the
        # shared plot_timeseries primitive does not thin a categorical axis.
        sub["date"] = pd.to_datetime(sub["date"])
        plot_timeseries(
            sub,
            path,
            x="date",
            y=stat,
            group_by="seed_id",
            xlabel="Date",
            ylabel=stat,
            title=f"Daily {stat} per channel",
            marker=None,
        )


def _monthly_rank_order(sub: pd.DataFrame) -> list[str]:
    """Station order for a stat's box plot: most anomalous (lowest ``rank``
    in the most recent month) first -- see the module docstring's "Rankings
    are folded into this same plot" note. Stations absent from the latest
    month are appended afterwards, in ``seed_id`` order.
    """
    all_seed_ids = sorted(sub["seed_id"].dropna().unique().tolist())
    if sub.empty or "month" not in sub.columns or "rank" not in sub.columns:
        return all_seed_ids
    latest_month = sub["month"].max()
    latest_rank = sub.loc[sub["month"] == latest_month].set_index("seed_id")["rank"]

    def sort_key(seed_id: str) -> tuple:
        rank = latest_rank.get(seed_id) if seed_id in latest_rank.index else None
        if rank is None or pd.isna(rank):
            return (1, 0, seed_id)
        return (0, float(rank), seed_id)

    return sorted(all_seed_ids, key=sort_key)


def _monthly_stat_boxplots(monthly_df: pd.DataFrame, stats: Sequence[str], out_dir: Path) -> None:
    for stat in stats:
        path = out_dir / f"monthly_{stat}_boxplot.png"
        if monthly_df is None or monthly_df.empty or "stat" not in monthly_df.columns:
            plot_boxplot(
                pd.DataFrame(columns=["seed_id", "ratio"]), path, by="seed_id", column="ratio"
            )
            continue
        sub = monthly_df[monthly_df["stat"] == stat]
        if sub.empty:
            logger.warning(
                "qc.plots: stat %r not present in monthly_df -- writing an empty "
                "'monthly_%s_boxplot.png' rather than skipping it.",
                stat,
                stat,
            )
            plot_boxplot(
                pd.DataFrame(columns=["seed_id", "ratio"]), path, by="seed_id", column="ratio"
            )
            continue
        order = _monthly_rank_order(sub)
        plot_boxplot(
            sub,
            path,
            by="seed_id",
            column="ratio",
            order=order,
            xlabel="Station (ranked: most anomalous first)",
            ylabel=f"{stat} ratio to network median",
            title=f"Monthly {stat}: station ratio to network median",
        )


def _flag_overview(flags: dict, out_dir: Path) -> None:
    path = out_dir / "flag_overview.png"
    rows = [
        {"seed_id": seed_id, "date": date, "ok": 1.0 if info.get("ok") else 0.0}
        for seed_id, days in (flags or {}).items()
        for date, info in days.items()
    ]
    if not rows:
        pivot = pd.DataFrame()
    else:
        long_df = pd.DataFrame(rows)
        pivot = long_df.pivot_table(
            index="seed_id", columns="date", values="ok", aggfunc="mean"
        ).sort_index()
        pivot = pivot[sorted(pivot.columns)]
    plot_heatmap(
        pivot,
        path,
        xlabel="Date",
        ylabel="Seed ID",
        title="QC flag overview (1 = ok, 0 = flagged)",
        cbar_label="OK",
        vmin=0.0,
        vmax=1.0,
    )


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
) -> None:
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
        Which daily/monthly statistics to render ``daily_<stat>.png`` /
        ``monthly_<stat>_boxplot.png`` for (e.g. a plain list built from
        ``cfg.qc.daily_stats`` by the caller -- this module takes no
        ``openmeq.config.Config``). Defaults to every stat column present
        in ``daily_stats_df`` (for the daily plots) / every ``stat`` value
        present in ``monthly_df`` (for the monthly plots) -- see the
        module docstring's "Tolerance to changing stat columns". A stat
        named here but absent from a given DataFrame is still rendered as
        an empty placeholder PNG (logged as a warning) rather than being
        silently dropped, so the expected file always exists.
    """
    setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_stats = list(stats) if stats is not None else _default_daily_stats(daily_stats_df)
    monthly_stats = list(stats) if stats is not None else _default_monthly_stats(monthly_df)

    _availability_heatmap(availability_df, out_dir)
    _daily_stat_timeseries(daily_stats_df, daily_stats, out_dir)
    _monthly_stat_boxplots(monthly_df, monthly_stats, out_dir)
    _flag_overview(flags, out_dir)

    summary_df = summarize_flags(flags)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, columns=SUMMARY_COLUMNS)


__all__ = ["make_qc_diagnostics", "summarize_flags", "SUMMARY_COLUMNS"]
