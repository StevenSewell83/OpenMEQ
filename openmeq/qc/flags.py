"""Monthly cross-station QC comparison and per-channel-day flags (M1.5, ``OpenMEQ-5nl.5``).

Completes PLAN.md §6 Step 1's "Monthly cross-station comparison" and its
required output, ``qc/channel_flags.json`` (ARCHITECTURE.md §4.2), consumed
directly by detection (M2.1) to exclude bad channels. This module works
purely from the two QC CSVs/DataFrames M1.3 (``qc/availability.csv``,
:mod:`openmeq.qc.availability`) and M1.4 (``qc/daily_stats.csv``,
:mod:`openmeq.qc.stats`) already produced -- it never reads waveforms, so it
stays fast even at production scale.

.. note::
   Per the build orchestrator's placement for this task, both the monthly
   comparison (:func:`compute_monthly_stats`) and the per-channel-day flags
   (:func:`compute_channel_flags`) live in this one module rather than being
   split across ``monthly.py``/``flags.py`` -- the two are small and tightly
   coupled (flags consumes the monthly comparison's output directly), and
   nothing else in the codebase imports a ``openmeq.qc.monthly`` name.

Two outputs, ARCHITECTURE.md §3
--------------------------------
- ``qc/monthly_stats.csv``: ``seed_id,month,stat,value,network_median,ratio,
  rank`` -- one row per (station, month, statistic), :func:`compute_monthly_stats`
  / :func:`write_monthly_stats_csv`.
- ``qc/channel_flags.json``: ``{"version": 1, "flags": {seed_id: {date:
  {"ok": bool, "reasons": [...]}}}}`` (ARCHITECTURE.md §4.2),
  :func:`compute_channel_flags` / :func:`write_channel_flags`.

Why combine several statistics rather than trust kurtosis alone
------------------------------------------------------------------
M1.4's own tests (``tests/test_qc_stats.py::test_known_spiky_channel_flagged_by_stats``)
found that on this project's synthetic dataset, a large real seismic event
can give an otherwise-ordinary channel/day a higher per-day kurtosis than
the deliberately spiky channel's -- so per-day kurtosis alone is not a
reliable cross-*channel* discriminator here. This module never flags on a
single statistic in isolation:

- The four day-level thresholds (``spikes``, ``flat``, ``clipped``, plus
  ``availability``) each look at a robust, purpose-built count
  (MAD-threshold impulse count, run-length flat/clip counts, coverage
  percentage) rather than kurtosis.
- The monthly cross-station comparison (:func:`compute_monthly_stats`)
  additionally aggregates *every* configured ``cfg.qc.daily_stats`` entry
  (by default ``rms``, ``mean_abs``, ``spike_count`` and ``kurtosis``) and
  compares each station's monthly mean against the network's monthly
  median for that same statistic. :func:`compute_channel_flags`'s
  ``"network_outlier"`` reason fires if *any one* of those statistics
  deviates far enough from the network -- so a channel that is anomalous on
  ``spike_count`` (a robust discriminator) still gets flagged even on a
  day/month where its raw kurtosis alone would not stand out, and
  vice versa. This is the "combine several statistics" mitigation called
  for by the M1.4 handoff note.

Monthly ratio / rank conventions (documented -- ARCHITECTURE.md does not
specify these beyond the column names)
--------------------------------------------------------------------------
- ``network_median`` is the median, across every station reporting that
  statistic that month, of each station's own monthly mean value.
- ``ratio = value / network_median``, **except** when ``network_median ==
  0``: dividing by zero is undefined, so as a documented edge-case
  convention, ``ratio = value`` in that case (not ``inf``/``nan``), unless
  ``value`` is *also* zero, in which case ``ratio = nan`` (both numerator
  and denominator vanish -- "no deviation" is not a well-defined ratio
  here, but the station's raw value is also uninformative on its own, so it
  is not simply reported as the ratio).
- ``rank`` is 1-based, by *descending* ``abs(log(ratio))`` -- so rank 1 is
  the station whose monthly value deviates furthest from the network
  median that month, in *either* direction (both far above and far below
  are anomalous). A non-positive or non-finite ``ratio`` (``<= 0`` or
  ``nan``) has an undefined real logarithm; such rows are ranked *last*
  within their (month, stat) group (documented choice -- see the pitfall
  about NaN-safe sorting), except ``ratio == 0`` exactly, whose ``log`` is
  ``-inf`` and is therefore (correctly) treated as maximally anomalous
  (rank 1): a station reporting a value of zero while the rest of the
  network does not is a legitimate, informative deviation.

Only the stats named in ``cfg.qc.daily_stats`` are ever aggregated -- a
stat named in config but absent from ``daily_stats_df``'s columns (e.g. a
concurrently-narrowed M1.4 output) is skipped rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from openmeq.config import Config
from openmeq.log import get_logger

logger = get_logger("qc.flags")

#: ARCHITECTURE.md §3's exact column order for qc/monthly_stats.csv.
MONTHLY_STATS_COLUMNS = ["seed_id", "month", "stat", "value", "network_median", "ratio", "rank"]


# ---------------------------------------------------------------------------
# Monthly cross-station comparison
# ---------------------------------------------------------------------------


def _format_threshold(value: float) -> str:
    """Render a threshold for a reason string (``"availability<80"``, not
    ``"availability<80.0"``) -- an integer-valued float prints without a
    decimal point; anything else prints via ``repr`` of the float with
    trailing zeros trimmed.
    """
    if float(value).is_integer():
        return str(int(value))
    text = f"{float(value):g}"
    return text


def compute_monthly_stats(daily_stats_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Each station's monthly statistics against the network median.

    For every stat named in ``cfg.qc.daily_stats`` (PLAN.md §3 default
    ``[rms, mean_abs, spike_count, kurtosis]``) that is actually a column of
    ``daily_stats_df``, groups by ``(seed_id, month)`` (``month`` is a
    ``YYYY-MM`` string, sliced from the ``date`` column) and takes the mean
    of that stat's daily values as the station's monthly value. Separately,
    for each ``(month, stat)``, computes ``network_median`` = the median of
    ``value`` across every station reporting that month/stat.

    See the module docstring for the ``ratio``/``rank`` conventions
    (division-by-zero guard, ``abs(log(ratio))`` ranking, and how ``nan``
    ratios are ordered).

    Parameters
    ----------
    daily_stats_df : pandas.DataFrame
        ``qc/daily_stats.csv``-shaped: at least ``seed_id``, ``date``
        (``YYYY-MM-DD`` strings), plus one column per configured stat.
    cfg : openmeq.config.Config
        Only ``cfg.qc.daily_stats`` is used.

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`MONTHLY_STATS_COLUMNS`, sorted by
        ``seed_id``, ``month``, ``stat``.
    """
    stats = list(cfg.qc.daily_stats)
    if daily_stats_df.empty or not stats:
        return pd.DataFrame(columns=MONTHLY_STATS_COLUMNS)

    df = daily_stats_df.copy()
    df["month"] = df["date"].astype(str).str.slice(0, 7)

    long_frames = []
    for stat in stats:
        if stat not in df.columns:
            logger.warning(
                "qc.daily_stats names %r, which is not a column of daily_stats_df -- skipping "
                "it for the monthly comparison rather than raising (a concurrently-narrowed "
                "M1.4 output may legitimately omit it).",
                stat,
            )
            continue
        grouped = df.groupby(["seed_id", "month"], as_index=False)[stat].mean()
        grouped = grouped.rename(columns={stat: "value"})
        grouped["stat"] = stat
        long_frames.append(grouped)

    if not long_frames:
        return pd.DataFrame(columns=MONTHLY_STATS_COLUMNS)

    long_df = pd.concat(long_frames, ignore_index=True)

    medians = long_df.groupby(["month", "stat"])["value"].median().rename("network_median")
    long_df = long_df.merge(medians, on=["month", "stat"], how="left")

    def _ratio(value: float, median: float) -> float:
        if median == 0:
            if value == 0:
                return float("nan")
            return float(value)
        return float(value) / float(median)

    long_df["ratio"] = [
        _ratio(v, m) for v, m in zip(long_df["value"], long_df["network_median"], strict=True)
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_log_ratio = np.abs(np.log(long_df["ratio"].to_numpy(dtype=float)))
    long_df["_abs_log_ratio"] = abs_log_ratio

    # Deterministic tie-break: sort by seed_id before ranking with
    # method="first", so equal-deviation rows rank in seed_id order
    # regardless of the input's row order.
    long_df = long_df.sort_values(["month", "stat", "seed_id"], kind="stable").reset_index(
        drop=True
    )
    long_df["rank"] = (
        long_df.groupby(["month", "stat"])["_abs_log_ratio"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype(int)
    )

    long_df = long_df.drop(columns=["_abs_log_ratio"])
    long_df = long_df.sort_values(["seed_id", "month", "stat"], kind="stable").reset_index(
        drop=True
    )
    return long_df[MONTHLY_STATS_COLUMNS]


def write_monthly_stats_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``df`` to ``path`` as ``qc/monthly_stats.csv`` (ARCHITECTURE.md §3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, columns=MONTHLY_STATS_COLUMNS)


# ---------------------------------------------------------------------------
# Per-channel-day flags
# ---------------------------------------------------------------------------


def _month_of(date_str: str) -> str:
    return str(date_str)[:7]


def compute_channel_flags(
    availability_df: pd.DataFrame,
    daily_stats_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    cfg: Config,
) -> dict:
    """Per-``(seed_id, date)`` QC pass/fail flags (ARCHITECTURE.md §4.2).

    For every ``(seed_id, date)`` row present in ``availability_df`` or
    ``daily_stats_df``, evaluates each reason below and sets ``ok =
    len(reasons) == 0``:

    - ``"availability<{min_availability_pct}"`` if ``coverage_pct <
      cfg.qc.min_availability_pct``.
    - ``"spikes"`` if ``spike_count / npts > cfg.qc.max_spike_fraction``.
    - ``"flat"`` if ``flat_count / npts > cfg.qc.max_flat_fraction``.
    - ``"clipped"`` if ``clip_count > cfg.qc.max_clip_count``.
    - ``"excluded_by_config"`` if ``seed_id`` is in
      ``cfg.qc.exclude_channels`` -- a hard override applied to every date
      for that channel, regardless of how clean its stats otherwise are.
    - ``"network_outlier"`` applied to every date within a ``(seed_id,
      month)`` where that channel's monthly ``ratio`` (from ``monthly_df``)
      for *any* stat in ``cfg.qc.daily_stats`` exceeds
      ``cfg.qc.monthly_outlier_ratio`` or is below its reciprocal.

    A ``(seed_id, date)`` with ``coverage_pct == 0`` (no data at all) is
    omitted entirely from the output -- ARCHITECTURE.md §4.2: "a channel/day
    with no entry counts as ``ok: false``" -- rather than written out with a
    reason. This omission is unconditional (it happens even for a channel
    in ``cfg.qc.exclude_channels``): there is nothing to report reasons
    about on a day with no data.

    Parameters
    ----------
    availability_df : pandas.DataFrame
        ``qc/availability.csv``-shaped (:data:`openmeq.qc.availability.COLUMNS`).
    daily_stats_df : pandas.DataFrame
        ``qc/daily_stats.csv``-shaped
        (:data:`openmeq.qc.stats.DAILY_STATS_COLUMNS`).
    monthly_df : pandas.DataFrame
        :func:`compute_monthly_stats`'s output.
    cfg : openmeq.config.Config

    Returns
    -------
    dict
        ``{seed_id: {date: {"ok": bool, "reasons": [str, ...]}}}`` -- the
        *inner* value of ARCHITECTURE.md §4.2's ``"flags"`` key.
        :func:`write_channel_flags` adds the ``"version"``/``"flags"``
        envelope around this when writing to disk.
    """
    qc = cfg.qc

    avail_cols = availability_df[["seed_id", "date", "coverage_pct"]].copy()
    stats_cols = daily_stats_df[
        ["seed_id", "date", "spike_count", "npts", "flat_count", "clip_count"]
    ].copy()

    merged = pd.merge(avail_cols, stats_cols, on=["seed_id", "date"], how="outer")

    # ARCHITECTURE.md §4.2: a channel/day with no data at all (coverage_pct
    # == 0) is simply omitted, unconditionally -- see the docstring.
    merged = merged[~(merged["coverage_pct"] == 0)]

    exclude_channels = set(qc.exclude_channels)

    # (seed_id, month) pairs where any configured stat's monthly ratio
    # exceeds monthly_outlier_ratio or is below its reciprocal.
    outlier_pairs: set[tuple[str, str]] = set()
    if not monthly_df.empty:
        configured_stats = set(qc.daily_stats)
        ratio_threshold = qc.monthly_outlier_ratio
        reciprocal = 1.0 / ratio_threshold if ratio_threshold != 0 else float("inf")
        candidate = monthly_df[monthly_df["stat"].isin(configured_stats)]
        with np.errstate(invalid="ignore"):
            is_outlier = (candidate["ratio"] > ratio_threshold) | (candidate["ratio"] < reciprocal)
        is_outlier = is_outlier.fillna(False)
        outlier_seed_ids = candidate.loc[is_outlier, "seed_id"]
        outlier_months = candidate.loc[is_outlier, "month"]
        for seed_id, month in zip(outlier_seed_ids, outlier_months, strict=True):
            outlier_pairs.add((seed_id, month))

    flags: dict[str, dict[str, dict]] = {}

    for row in merged.itertuples(index=False):
        seed_id = row.seed_id
        date = row.date
        coverage_pct = row.coverage_pct
        spike_count = row.spike_count
        npts = row.npts
        flat_count = row.flat_count
        clip_count = row.clip_count

        reasons: list[str] = []

        if pd.notna(coverage_pct) and coverage_pct < qc.min_availability_pct:
            reasons.append(f"availability<{_format_threshold(qc.min_availability_pct)}")

        if pd.notna(spike_count) and pd.notna(npts) and npts > 0:
            if (spike_count / npts) > qc.max_spike_fraction:
                reasons.append("spikes")

        if pd.notna(flat_count) and pd.notna(npts) and npts > 0:
            if (flat_count / npts) > qc.max_flat_fraction:
                reasons.append("flat")

        if pd.notna(clip_count) and clip_count > qc.max_clip_count:
            reasons.append("clipped")

        if seed_id in exclude_channels:
            reasons.append("excluded_by_config")

        if (seed_id, _month_of(date)) in outlier_pairs:
            reasons.append("network_outlier")

        flags.setdefault(seed_id, {})[date] = {"ok": len(reasons) == 0, "reasons": reasons}

    return flags


def write_channel_flags(flags: dict, path: str | Path) -> None:
    """Write ``flags`` to ``path`` as ``qc/channel_flags.json``, exactly per
    ARCHITECTURE.md §4.2: ``{"version": 1, "flags": {...}}``.

    Parameters
    ----------
    flags : dict
        :func:`compute_channel_flags`'s output (the *inner* mapping --
        this function adds the ``"version"``/``"flags"`` envelope).
    path : str or pathlib.Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "flags": flags}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
