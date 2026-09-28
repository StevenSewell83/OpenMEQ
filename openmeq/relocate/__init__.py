"""hypoDD cross-correlation and relocation: orchestration, diagnostics and
the ``openmeq relocate`` CLI command (M10.5 / OpenMEQ-16d.5).

Plan key: M10.5, the final M10 task. Chains M10.1-M10.4's already-built
steps -- each already gated by its own ``openmeq.manifest.StepManifest``
key sharing ``relocate/manifest.json`` (ARCHITECTURE.md §6c-2/§6f/§6g/§6h)
-- into one function, `run_relocate`, registered below as this package's
``run(cfg, paths, **opts)`` (the ``openmeq.cli`` step-module contract,
ARCHITECTURE.md §2):

1. `openmeq.relocate.waveform_cache.build_cache` (M10.1): pre-filtered
   waveform cache, one ``.npz`` per accepted event.
2. `openmeq.relocate.ph2dt_step.run_ph2dt` (M10.2): writes
   ``phase.dat``/``event.dat``, runs ``ph2dt`` (which itself calls
   `openmeq.relocate.setup.setup_hypodd`, M9.5, first), returns the
   linked-pair list.
3. `openmeq.relocate.xcorr.run_xcorr` (M10.3): cross-correlates every
   linked pair, writes ``dt.cc``.
4. `openmeq.relocate.hypodd_runner.run_hypodd` (M10.4): runs the real
   ``hypoDD`` binary and merges its relocations into
   ``relocate/catalog_dd.xml``.
5. `openmeq.relocate.diagnostics.write_relocate_diagnostics` (this task):
   before/after maps, dt.cc CC distribution, per-iteration residual/
   connectivity/AQ/CND trends, relative-shift histogram, ``summary.csv``.

Every step above already requires ``magnitude/catalog_mag.xml`` and raises
a clear `openmeq.config.ConfigError` itself if it's missing; `run_relocate`
also checks it up front (before doing any work at all, e.g. building the
whole waveform cache) so the "run openmeq magnitude first" message is the
very first thing a user sees rather than one buried after a partial run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from openmeq.catalog import accepted, event_id, read_catalog
from openmeq.log import get_logger
from openmeq.relocate.diagnostics import RelocateDiagnosticsResult, write_relocate_diagnostics
from openmeq.relocate.hypodd_runner import HypoddRunResult, run_hypodd
from openmeq.relocate.ph2dt_step import run_ph2dt
from openmeq.relocate.waveform_cache import build_cache
from openmeq.relocate.xcorr import run_xcorr

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate")


@dataclass
class RelocateResult:
    """What one `run_relocate` call did.

    Attributes
    ----------
    pairs : list[tuple[int, int]]
        The ``(cuspid1, cuspid2)`` linked-pair list from `run_ph2dt`.
    hypodd : openmeq.relocate.hypodd_runner.HypoddRunResult
    diagnostics : openmeq.relocate.diagnostics.RelocateDiagnosticsResult
    """

    pairs: list[tuple[int, int]]
    hypodd: HypoddRunResult
    diagnostics: RelocateDiagnosticsResult


def _require_catalog_mag(paths: RunPaths) -> None:
    """Raises `click.ClickException` (not `openmeq.config.ConfigError`) if
    ``magnitude/catalog_mag.xml`` doesn't exist -- matching
    `openmeq.magnitude.pipeline.run_magnitude`'s own established convention
    for this class of "hard upstream-dependency missing" error (a plain,
    clearly-worded CLI failure rather than a traceback; ``tests/test_cli.py``'s
    ``test_step_runs_or_fails_clearly_against_example_config`` and
    ``scripts/benchmark.py``'s own stage-error handling both rely on this
    for every implemented step, not just this one).
    """
    catalog_path = Path(paths.catalog_mag_xml)
    if not catalog_path.is_file():
        message = (
            f"openmeq relocate: input catalogue not found: {catalog_path} -- `openmeq relocate` "
            "needs magnitude/catalog_mag.xml (M8.3's output) as its starting catalogue: it "
            "supplies the picks/origins ph2dt and hypoDD need, the magnitudes carried over into "
            "hypoDD.reloc (PLAN.md §6 Step 7), and the catalogue every relocation is merged back "
            "into. Run `openmeq magnitude` first."
        )
        logger.error(message)
        raise click.ClickException(message)


def run_relocate(
    cfg: Config,
    paths: RunPaths,
    *,
    force: bool = False,
    external_hypodd_dir: Path | str | None = None,
    external_bin_dir: Path | str | None = None,
) -> RelocateResult:
    """Run the full M10.1 -> M10.4 relocation chain, then write M10.5's
    own diagnostics.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Forwarded to `run_ph2dt` and `run_hypodd` (each rebuilds/reruns
        regardless of its own manifest fingerprint) and to `run_xcorr`
        (discards any resumed ``xcorr_results.jsonl`` first, so every
        pair is recomputed). `build_cache` has no ``force`` of its own
        (ARCHITECTURE.md §6f) -- its per-event cache is always rebuilt if
        its own fingerprint (picks/chunk file/``cfg.hypodd.xcorr``)
        changed, `force` or not.
    external_hypodd_dir, external_bin_dir : pathlib.Path or str or None, keyword-only
        Forwarded to `run_ph2dt`/`run_hypodd`'s own compiled-limit checks
        (default: the real installed ``external/HYPODD``/``external/bin``
        tree -- tests should always pass a scratch directory instead,
        ARCHITECTURE.md §6c).

    Returns
    -------
    RelocateResult

    Raises
    ------
    click.ClickException
        If ``magnitude/catalog_mag.xml`` doesn't exist (`_require_catalog_mag`).
    openmeq.config.ConfigError
        If any chained step raises one (a missing input, an unresolvable
        station-label collision, a stale ``event_id_map``, ...).
    openmeq.relocate.hypodd_runner.HypoddRunFailed
        If the real ``hypoDD`` binary fails outright.
    """
    _require_catalog_mag(paths)

    catalog = read_catalog(paths.catalog_mag_xml)
    event_ids = [event_id(ev) for ev in accepted(catalog)]

    logger.info("relocate: building the waveform cache for %d accepted event(s)...", len(event_ids))
    build_cache(cfg, paths, event_ids)

    logger.info("relocate: running ph2dt...")
    pairs = run_ph2dt(
        cfg,
        paths,
        force=force,
        external_hypodd_dir=external_hypodd_dir,
        external_bin_dir=external_bin_dir,
    )

    logger.info("relocate: cross-correlating %d linked pair(s)...", len(pairs))
    run_xcorr(cfg, paths, pairs, force=force)

    logger.info("relocate: running hypoDD...")
    hypodd_result = run_hypodd(
        cfg,
        paths,
        force=force,
        external_hypodd_dir=external_hypodd_dir,
        external_bin_dir=external_bin_dir,
    )

    logger.info("relocate: writing diagnostics...")
    diagnostics = write_relocate_diagnostics(paths)

    logger.info(
        "relocate: done (%d relocated, %d not linked, run_mode=%s).",
        hypodd_result.n_relocated,
        hypodd_result.n_not_linked,
        hypodd_result.run_mode,
    )
    return RelocateResult(pairs=pairs, hypodd=hypodd_result, diagnostics=diagnostics)


def run(cfg: Config, paths: RunPaths, *, force: bool = False, **_ignored: Any) -> RelocateResult:
    """``openmeq relocate`` step entry point -- see `run_relocate`.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only, optional
        ``--force``: rerun every sub-step regardless of its own manifest.
    **_ignored
        Absorbs any other step-generic kwargs ``openmeq.cli``'s
        ``_dispatch`` forwards (ARCHITECTURE.md §2's
        ``run(cfg, paths, **opts)`` contract), e.g. ``config_path``.
    """
    return run_relocate(cfg, paths, force=force)
