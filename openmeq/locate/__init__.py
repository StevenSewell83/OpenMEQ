"""Two-pass NonLinLoc location and filtering.

``openmeq.locate.nlloc_runner`` (M6.1) is the per-event NLLoc runner.
``openmeq.locate.pass_a``/``pass_b`` (M6.2/M6.3) implement the iterative
P-only, then P+S, relocate-and-filter passes. ``openmeq.locate.finalize``
(M6.4) owns the final per-event accept/reject decision and the
``locate/summary.csv``/``locate/removal_log.csv`` row shapes.
``openmeq.locate.locate_step.run_locate`` (M6.4) is the per-event
orchestration -- pass A -> pass B -> finalize, parallelised over events --
that writes the three M6 output artifacts (ARCHITECTURE.md §3):
``locate/catalog_auto.xml``, ``locate/summary.csv``, ``locate/removal_log.csv``.
``openmeq.locate.diagnostics`` (M6.5) builds ``locate/diagnostics/`` from
``run_locate``'s own result.

This module's ``run(cfg, paths, **opts)`` is what ``openmeq/cli.py`` looks
for (ARCHITECTURE.md §2's ``openmeq.cli`` contract) -- registering the
``openmeq locate`` step needs no change to ``cli.py`` itself, mirroring
every other step's own ``run`` entry point.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import click

from openmeq.locate.diagnostics import make_locate_diagnostics
from openmeq.locate.locate_step import run_locate
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.nlloc_setup import setup_nlloc

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("locate")

#: ``RunPaths.diagnostics_dir``/``manifest_json``'s step-folder-name argument.
STEP_NAME = "locate"

#: The single :class:`~openmeq.manifest.StepManifest` entry key for this
#: step -- ``run_locate`` always relocates the whole ``pick/catalog_picks.xml``
#: catalogue in one pass (per-event parallelism happens *inside* it, via
#: ``openmeq.parallel.pmap``), so there is no finer-grained unit to skip
#: within one ``openmeq locate`` run (mirrors ``openmeq.pick``'s own single
#: ``MANIFEST_KEY`` convention, for the same reason).
MANIFEST_KEY = "locate"


def _require_pick_outputs(paths: RunPaths) -> None:
    """Raise a clear, actionable error if ``pick/catalog_picks.xml`` is missing.

    Mirrors ``openmeq.pick._require_chunk_outputs``'s identical convention:
    a fresh run directory, or one where an earlier step but not ``pick``
    has run, is the expected first-run-out-of-order mistake.
    """
    if paths.catalog_picks_xml.is_file():
        return
    message = (
        f"openmeq locate: missing required input: {paths.catalog_picks_xml}. "
        "Run `openmeq pick` first."
    )
    logger.error(message)
    raise click.ClickException(message)


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    force: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq locate`` step entry point: NLLoc grid setup -> pass A/B ->
    finalize -> diagnostics.

    1. Requires ``pick/catalog_picks.xml`` to already exist
       (:func:`_require_pick_outputs`) -- otherwise fails clearly, telling
       the user to run ``openmeq pick`` first.
    2. :func:`openmeq.nlloc_setup.setup_nlloc` (M5.2/M5.3): builds (or
       refreshes) the NLLoc travel-time grids under ``nlloc/`` if
       ``cfg.nlloc``/the station table/the velocity model input changed
       since the last grid build -- this call always happens (grid setup
       has its own manifest, ``nlloc/manifest.json``, entirely separate
       from this step's own ``locate/manifest.json`` below), so a fresh
       project (grids never built) or a config change that only touches
       ``cfg.nlloc.velocity_model``/``cfg.stations`` is picked up
       automatically without a separate ``scripts/setup_nlloc.py`` call.
    3. Skips the relocation itself (unless ``force``) if ``cfg.nlloc`` and
       ``pick/catalog_picks.xml``'s own mtime/size both match the last
       recorded fingerprint (:class:`openmeq.manifest.StepManifest`) and
       ``locate/catalog_auto.xml`` still exists.
    4. :func:`openmeq.locate.locate_step.run_locate` (M6.4): pass A -> pass
       B -> finalize per event -> ``locate/catalog_auto.xml`` /
       ``locate/summary.csv`` / ``locate/removal_log.csv``.
    5. :func:`openmeq.locate.diagnostics.make_locate_diagnostics` (M6.5) ->
       ``locate/diagnostics/``.
    6. Marks the manifest entry done and logs the elapsed time.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        Unused by this step (mirrors ``openmeq.pick.run``) -- accepted
        only because ``openmeq/cli.py``'s dispatcher always supplies it
        (ARCHITECTURE.md §2's ``run(cfg, paths, **opts)`` contract).
    force : bool, keyword-only, optional
        ``--force``: rerun even if the manifest says nothing changed. Does
        not force a grid rebuild (``setup_nlloc`` decides that on its own,
        via its own manifest) -- pass ``scripts/setup_nlloc.py --force``
        separately if a forced grid rebuild is also wanted.
    **_ignored
        Absorbs any other step-generic kwargs the dispatcher forwards.
    """
    del config_path  # unused -- see docstring
    t0 = time.perf_counter()
    paths.ensure()
    step_logger = get_logger("locate")

    _require_pick_outputs(paths)

    setup_nlloc(cfg, paths)

    fp = fingerprint(cfg.nlloc, paths.catalog_picks_xml, cfg.nlloc.station_delays_csv or "")
    manifest = StepManifest(paths, STEP_NAME)
    if not force and manifest.is_done(MANIFEST_KEY, fp) and paths.catalog_auto_xml.is_file():
        step_logger.info(
            "locate: unchanged since the last run (cfg.nlloc and pick/catalog_picks.xml both "
            "match the last recorded fingerprint) -- skipping (use --force to rerun)."
        )
        return

    result = run_locate(paths, cfg)
    make_locate_diagnostics(paths, cfg, result)

    manifest.mark_done(MANIFEST_KEY, fp)

    elapsed = time.perf_counter() - t0
    n_accepted = sum(1 for row in result.summary_rows if row.status == "accepted")
    step_logger.info(
        "locate: %d event(s) processed, %d accepted (%.2fs).",
        len(result.summary_rows),
        n_accepted,
        elapsed,
    )


__all__ = ["MANIFEST_KEY", "STEP_NAME", "run"]
