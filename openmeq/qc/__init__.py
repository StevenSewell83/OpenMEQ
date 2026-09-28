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

import time
from pathlib import Path
from typing import TYPE_CHECKING

from openmeq.index import build_index
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
from openmeq.qc.plots import make_qc_diagnostics
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


def run(cfg: Config, paths: RunPaths, *, force: bool = False) -> None:
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

    manifest = StepManifest(paths, _MANIFEST_KEY)
    input_fp = _input_fingerprint(cfg)
    if not force and manifest.is_done(_MANIFEST_KEY, input_fp):
        logger.info("qc: up to date, skipping")
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
    make_qc_diagnostics(
        availability_df,
        daily_stats_df,
        monthly_df,
        flags,
        paths.qc_diagnostics_dir,
        stats=cfg.qc.daily_stats,
    )

    manifest.mark_done(_MANIFEST_KEY, input_fp)

    elapsed = time.perf_counter() - t_start
    logger.info("qc: completed in %.2fs", elapsed)
