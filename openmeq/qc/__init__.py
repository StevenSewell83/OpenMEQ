"""Data quality control: index, station metadata, availability, noise stats, channel flags.

Implements ``openmeq qc`` (M1.7, ``OpenMEQ-5nl.7``): the single step that
sequences M1.1-M1.6's already-built pieces into one pass over the archive,
per PLAN.md §6 Step 1 and ARCHITECTURE.md §3's ``inventory/``/``qc/``
output layout. Every later step (starting with M2 detection) depends on
this step's outputs existing on disk.

``run(cfg, paths, **opts)`` is this module's registration hook
(ARCHITECTURE.md §2's ``openmeq.cli`` row): ``openmeq/cli.py`` looks it up
by name after lazily importing this module -- no change to ``cli.py``
beyond adding the ``qc`` subcommand's own ``--force`` option is needed to
wire a step in.

Pipeline order (strictly linear -- each stage's inputs are an earlier
stage's outputs, per the design note in this task's beads issue)
-------------------------------------------------------------------
0. Convert raw data if ``data.convert`` is set (``openmeq.ingest``,
   OpenMEQ-6zu) -> miniSEED day files under ``data.convert.output_dir``
   (skipped when its inputs are unchanged).
1. Build/update the miniSEED index (``openmeq.index.build_index``) ->
   ``inventory/index.json``.
2. Merge station metadata (``openmeq.stations``) -> ``inventory/
   stations.xml``/``.csv``, and cross-check it against the index's seed
   ids -> ``inventory/metadata_report.json``.
3. Compute per-channel-day availability (``openmeq.qc.availability``) ->
   ``qc/availability.csv``.
4. Compute per-channel-day daily noise statistics (``openmeq.qc.stats``) ->
   ``qc/daily_stats.csv``.
5. Compute the monthly cross-station comparison and per-channel-day flags
   (``openmeq.qc.flags``) -> ``qc/monthly_stats.csv``, ``qc/
   channel_flags.json``.
6. Render QC diagnostic plots (``openmeq.qc.plots``) -> ``qc/
   diagnostics/``.
7. Optional PPSD per channel (``cfg.qc.ppsd.enabled``, ``openmeq.qc.ppsd``,
   OpenMEQ-itl) -> ``qc/ppsd/`` and ``qc/diagnostics/ppsd/``; also run when
   the rest of QC is up to date.

Plot-only station exclusion (``cfg.qc.plot_exclude_stations``, OpenMEQ-v3i)
----------------------------------------------------------------------------
Stations listed there (``STA`` or ``NET.STA``) are dropped from every
``qc/diagnostics/`` figure and ``summary.csv``; the tables in ``qc/`` keep
them. Because the monthly comparison is against the network median, the
plots' monthly stats and flags are recomputed without those stations. The
option is left out of this step's fingerprint (it's ``exclude=True`` in the
config model), so changing it never recomputes QC: :func:`replot` (``openmeq
diagnostics qc``) redraws the figures from the existing ``qc/`` tables.

Skip-unchanged (whole-step manifest)
-------------------------------------
A single ``openmeq.manifest.StepManifest`` entry covers the *whole* qc step
(one fingerprint, not six per-substep ones -- PLAN.md §4 only requires
skipping "outputs already produced for the same inputs and config", and one
fingerprint keeps ``run`` simple and matches how ``--force`` is expected to
behave: re-run everything). The fingerprint covers ``cfg.data``,
``cfg.stations``, ``cfg.qc`` (the three config sections this step reads)
plus every on-disk file mtimes/sizes over ``cfg.data.waveform_dirs``
(recursively, matching ``build_index``'s own file discovery) and
``cfg.stations.stationxml``/``stationxml_dir``/``csv`` -- so new miniSEED
files arriving, or station metadata changing, retrigger a rerun even with
byte-identical YAML (the M1.7 design note: fingerprinting only the config
objects would miss this).

``force=True`` bypasses the ``is_done`` check but still calls
``mark_done`` at the end, so a subsequent unforced run skips correctly
(the CLI's ``--force`` flag, forwarded here as a keyword option). A
failure partway through (e.g. the stations merge raises) propagates
without ever calling ``mark_done`` -- a resumed run redoes everything
rather than silently treating a partial output set as complete.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from obspy import UTCDateTime

from openmeq.index import build_index
from openmeq.ingest import convert_raw_data
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.qc.availability import (
    compute_availability,
    date_range_inclusive,
    write_availability_csv,
)
from openmeq.qc.flags import (
    compute_channel_flags,
    compute_monthly_stats,
    write_channel_flags,
    write_monthly_stats_csv,
)
from openmeq.qc.instruments import instrument_table
from openmeq.qc.noise_plots import make_noise_diagnostics, noise_table, write_noise_daily_csv
from openmeq.qc.plots import make_qc_diagnostics
from openmeq.qc.ppsd import run_ppsd
from openmeq.qc.stats import compute_daily_stats, write_daily_stats_csv
from openmeq.stations import (
    check_coverage,
    load_inventory,
    write_merged_inventory,
    write_metadata_report,
)

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("qc")


def _station_excluded(seed_ids: pd.Series, exclude: list[str]) -> pd.Series:
    """Mask of ``seed_ids`` whose ``STA`` or ``NET.STA`` is in ``exclude``."""
    parts = seed_ids.astype(str).str.split(".")
    sta = parts.str[1]
    net_sta = parts.str[0] + "." + sta
    return sta.isin(exclude) | net_sta.isin(exclude)


def _drop_plot_stations(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """``df`` without ``cfg.qc.plot_exclude_stations``' rows (module docstring)."""
    exclude = cfg.qc.plot_exclude_stations
    if not exclude or df.empty:
        return df
    return df[~_station_excluded(df["seed_id"], exclude)].reset_index(drop=True)


def _make_plots(
    cfg: Config,
    paths: RunPaths,
    availability_df: pd.DataFrame,
    daily_stats_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    flags: dict,
    noise_df: pd.DataFrame | None,
) -> None:
    """Render ``qc/diagnostics/`` (and ``noise/``), honouring ``plot_exclude_stations``."""
    if cfg.qc.plot_exclude_stations:
        n_before = availability_df["seed_id"].nunique()
        availability_df = _drop_plot_stations(availability_df, cfg)
        daily_stats_df = _drop_plot_stations(daily_stats_df, cfg)
        monthly_df = compute_monthly_stats(daily_stats_df, cfg)
        flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
        logger.info(
            "qc: plots exclude stations %s (%d of %d channel(s) dropped)",
            cfg.qc.plot_exclude_stations,
            n_before - availability_df["seed_id"].nunique(),
            n_before,
        )
    written = make_qc_diagnostics(
        availability_df,
        daily_stats_df,
        monthly_df,
        flags,
        paths.qc_diagnostics_dir,
        stats=cfg.qc.daily_stats,
        instruments=noise_df,
        title=cfg.project.name,
    )
    # Figures an earlier version or setting wrote (e.g. daily_rms.png,
    # monthly_*_boxplot.png) would otherwise sit next to the new set.
    keep = {Path(w).name for w in written}
    for old in Path(paths.qc_diagnostics_dir).glob("*.png"):
        if old.name not in keep:
            old.unlink()
    if noise_df is None:
        return
    try:
        noise_dir = paths.qc_noise_diagnostics_dir
        written = make_noise_diagnostics(
            _drop_plot_stations(noise_df, cfg),
            noise_dir,
            start=cfg.data.start,
            end=cfg.data.end,
            noise_cfg=cfg.qc.noise,
            title=cfg.project.name,
        )
        # A group with no stations left (e.g. all excluded) writes nothing:
        # remove its figures from an earlier run rather than leave them stale.
        keep = {Path(w).name for w in written}
        for old in noise_dir.glob("*.png"):
            if old.name not in keep:
                old.unlink()
    except Exception:  # noqa: BLE001 - report figures must never fail the step
        logger.exception("qc: noise diagnostics failed (QC outputs are unaffected)")


def _noise_table(cfg: Config, paths: RunPaths, inv, seed_ids, daily_stats_df):
    """Per-instrument noise table (OpenMEQ-h8j), written to ``noise_daily.csv``.

    A failure here is logged, never raised (returns ``None``): the noise
    figures are a report and must not stop QC's own outputs (already
    written) from being marked done.
    """
    try:
        mid = cfg.data.start + (cfg.data.end - cfg.data.start) / 2
        instruments = instrument_table(
            inv, seed_ids, UTCDateTime(mid.isoformat()), cfg.qc.instrument_groups
        )
        noise_df = noise_table(daily_stats_df, instruments)
        write_noise_daily_csv(noise_df, paths.noise_daily_csv)
        return noise_df
    except Exception:  # noqa: BLE001 - report figures must never fail the step
        logger.exception("qc: noise table failed (QC outputs are unaffected)")
        return None


def _ppsd(cfg: Config, paths: RunPaths, *, force: bool, instruments=None) -> None:
    """Optional PPSD (``cfg.qc.ppsd.enabled``, :mod:`openmeq.qc.ppsd`).

    Runs after the manifest is marked done, and also when QC itself is up to
    date (PPSD tracks its own settings), so enabling it never recomputes QC.
    A failure is logged, never raised: QC's own outputs are already complete.
    """
    if not cfg.qc.ppsd.enabled:
        return
    try:
        run_ppsd(cfg, paths, force=force, instruments=instruments)
    except Exception:  # noqa: BLE001 - an optional report must never fail the step
        logger.exception("qc: PPSD failed (QC outputs are unaffected)")


def replot(cfg: Config, paths: RunPaths) -> None:
    """Redraw ``qc/diagnostics/`` from an existing run's ``qc/`` tables.

    ``openmeq diagnostics qc``: no waveforms are read and no table is
    rewritten, so a plot-only change (``cfg.qc.plot_exclude_stations``,
    ``daily_stats``) takes effect without recomputing QC.

    Raises
    ------
    FileNotFoundError
        If a required ``qc/`` table is missing (run ``openmeq qc`` first).
    """
    required = [
        paths.availability_csv,
        paths.daily_stats_csv,
        paths.monthly_stats_csv,
        paths.channel_flags_json,
    ]
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"QC table(s) missing, run `openmeq qc` first: {missing}")

    def read(path):
        return pd.read_csv(path, dtype={"seed_id": str, "date": str, "month": str})

    availability_df = read(paths.availability_csv)
    daily_stats_df = read(paths.daily_stats_csv)
    monthly_df = read(paths.monthly_stats_csv)
    with open(paths.channel_flags_json, encoding="utf-8") as f:
        flags = json.load(f)["flags"]
    noise_df = read(paths.noise_daily_csv) if Path(paths.noise_daily_csv).is_file() else None
    _make_plots(cfg, paths, availability_df, daily_stats_df, monthly_df, flags, noise_df)
    logger.info("qc: diagnostics redrawn in %s", paths.qc_diagnostics_dir)


#: Step key StepManifest is keyed on (ARCHITECTURE.md §2/§3's whole-step
#: manifest; see the module docstring's "Skip-unchanged" section).
_MANIFEST_KEY = "qc"


def _waveform_files(cfg: Config) -> list[Path]:
    """Every on-disk file ``build_index`` would scan, for fingerprinting.

    Mirrors ``openmeq.index.build_index``'s own file discovery exactly
    (recursive glob of ``cfg.data.file_pattern`` under each of
    ``cfg.data.waveform_dirs``), so a new/changed/removed miniSEED file
    changes the qc step's fingerprint even when the YAML config is
    byte-identical (the M1.7 design note).
    """
    files: list[Path] = []
    for root in cfg.data.waveform_dirs:
        files.extend(sorted(Path(root).rglob(cfg.data.file_pattern)))
    return files


def _stations_files(cfg: Config) -> list[Path]:
    """Every station-metadata source file, for fingerprinting.

    A single ``stationxml`` file and/or ``csv`` are included directly;
    ``stationxml_dir`` is expanded to its individual ``*.xml``/
    ``*.stationxml`` members (mirrors ``openmeq.stations.load_inventory``'s
    own glob) so a file added to/removed from that folder is detected too,
    not just the folder's own mtime.
    """
    files: list[Path] = []
    stations_cfg = cfg.stations
    if stations_cfg.stationxml:
        files.append(Path(stations_cfg.stationxml))
    if stations_cfg.stationxml_dir:
        xml_dir = Path(stations_cfg.stationxml_dir)
        files.extend(sorted(set(xml_dir.glob("*.xml")) | set(xml_dir.glob("*.stationxml"))))
    if stations_cfg.csv:
        files.append(Path(stations_cfg.csv))
    return files


def _input_fingerprint(cfg: Config) -> str:
    """A stable hash of every input this qc step reads (see module docstring)."""
    return fingerprint(cfg.data, cfg.stations, cfg.qc, _waveform_files(cfg), _stations_files(cfg))


def run(cfg: Config, paths: RunPaths, *, force: bool = False, **_opts: object) -> None:
    """Run the whole ``openmeq qc`` step (the CLI's ``qc`` subcommand body).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
        Already ``.ensure()``-d by ``openmeq/cli.py`` before this is called.
    force : bool, optional
        Bypass the "skip unchanged" check and recompute every output
        (``openmeq qc --force``). Default ``False``.
    """
    t_start = time.perf_counter()

    # 0. Raw-data conversion (data.convert, OpenMEQ-6zu) -- before the
    # fingerprint, so newly converted files count as qc inputs this run.
    convert_raw_data(cfg, paths, force=force)

    manifest = StepManifest(paths, _MANIFEST_KEY)
    input_fp = _input_fingerprint(cfg)
    if not force and manifest.is_done(_MANIFEST_KEY, input_fp):
        logger.info("qc: up to date, skipping")
        _ppsd(cfg, paths, force=False)
        return

    # 1. Index (M1.1) -- writes inventory/index.json itself.
    index = build_index(cfg)
    seed_ids = index.seed_ids()

    # 2. Station metadata merge + coverage report (M1.2).
    inv = load_inventory(cfg.stations)
    write_merged_inventory(inv, paths, cfg.stations)
    coverage_report = check_coverage(seed_ids, inv)
    write_metadata_report(coverage_report, paths)

    # 3. Availability (M1.3).
    dates = date_range_inclusive(cfg.data.start, cfg.data.end)
    availability_df = compute_availability(index, seed_ids, dates, cfg)
    write_availability_csv(availability_df, paths.availability_csv)

    # 4. Daily noise statistics (M1.4/M1.9).
    daily_stats_df = compute_daily_stats(index, seed_ids, dates, cfg)
    write_daily_stats_csv(daily_stats_df, paths.daily_stats_csv)

    # 5. Monthly cross-station comparison + per-channel-day flags (M1.5).
    monthly_df = compute_monthly_stats(daily_stats_df, cfg)
    write_monthly_stats_csv(monthly_df, paths.monthly_stats_csv)
    flags = compute_channel_flags(availability_df, daily_stats_df, monthly_df, cfg)
    write_channel_flags(flags, paths.channel_flags_json)

    # 6. Diagnostics (M1.6).
    noise_df = _noise_table(cfg, paths, inv, seed_ids, daily_stats_df)
    _make_plots(cfg, paths, availability_df, daily_stats_df, monthly_df, flags, noise_df)

    manifest.mark_done(_MANIFEST_KEY, input_fp)
    _ppsd(cfg, paths, force=force, instruments=noise_df)

    elapsed = time.perf_counter() - t_start
    logger.info("qc: completed in %.2fs", elapsed)
