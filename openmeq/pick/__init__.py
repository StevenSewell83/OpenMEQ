"""PhaseNet prep, running PhaseNet, and converting picks to QuakeML.

``openmeq.pick.prep`` (M4.2, ``OpenMEQ-1m7.2``) prepares ``chunks_pn/``.
``openmeq.pick.run_phasenet`` (M4.3, ``OpenMEQ-1m7.3``) runs PhaseNet's
``predict.py`` once over the whole batch. ``openmeq.pick.to_quakeml`` and
``openmeq.pick.diagnostics`` (M4.4, ``OpenMEQ-1m7.4``) convert
``picks.csv`` to ``pick/catalog_picks.xml`` and build
``pick/diagnostics/``. This module's ``run(cfg, paths, **opts)`` is what
``openmeq/cli.py`` looks for (ARCHITECTURE.md §2's ``openmeq.cli`` contract)
-- registering the ``openmeq pick`` step needs no change to ``cli.py``
itself, mirroring ``openmeq.chunk``/``openmeq.detect``'s own ``run`` entry
points.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import click

from openmeq.chunk_diagnostics import make_chunk_pn_extended_diagnostics
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.pick.diagnostics import make_pick_diagnostics
from openmeq.pick.prep import prepare_all
from openmeq.pick.run_phasenet import run_phasenet
from openmeq.pick.to_quakeml import build_catalog, write_catalog_picks
from openmeq.stations import station_table as compute_station_table

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("pick")

#: ``RunPaths.diagnostics_dir``/``manifest_json``'s step-folder-name argument.
STEP_NAME = "pick"

#: The single :class:`~openmeq.manifest.StepManifest` entry key for this
#: step -- one whole-run unit of work (mirrors ``openmeq.chunk``'s
#: ``MANIFEST_KEY`` convention): PhaseNet's ``predict.py`` is always invoked
#: as a single batch call over every prepared event
#: (``openmeq.pick.run_phasenet``'s own module docstring), so there is no
#: finer-grained unit to skip within one ``openmeq pick`` run.
MANIFEST_KEY = "pick"


def _require_chunk_outputs(paths: RunPaths) -> None:
    """Raise a clear, actionable error if ``openmeq chunk``'s outputs are missing.

    ``openmeq pick`` reads raw waveform chunks from ``chunks/<event_id>/
    <event_id>.mseed`` (written by ``openmeq chunk``, M3.1/M3.2) -- a fresh
    run directory, or one where only ``qc``/``detect`` has run, is the
    expected first-run-out-of-order mistake; mirrors
    ``openmeq.chunk._require_detect_outputs``'s identical pattern.

    Raises
    ------
    click.ClickException
        Naming the missing directory and telling the user to run
        ``openmeq chunk`` first.
    """
    if paths.chunk_dir.is_dir() and any(paths.chunk_dir.glob("*/*.mseed")):
        return
    message = (
        f"openmeq pick: no chunk files found under {paths.chunk_dir}. Run `openmeq chunk` first."
    )
    logger.error(message)
    raise click.ClickException(message)


def _event_ids_from_chunks(paths: RunPaths) -> list[str]:
    """Every event with an actual raw chunk file on disk (``chunks/<event_id>/<event_id>.mseed``).

    Read from the ``chunks/`` directory listing itself, not
    ``chunks/completeness.csv`` -- that CSV can carry a zero-fraction row
    for an event whose chunk file was never written at all
    (``openmeq.chunk.chunk_day_group``'s "no chunk written" case,
    ``openmeq.chunk_diagnostics``'s own "zero-data rows are never omitted"
    convention), which :func:`~openmeq.pick.prep.prepare_chunk` cannot read.
    """
    return sorted(p.parent.name for p in paths.chunk_dir.glob("*/*.mseed"))


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    force: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq pick`` step entry point: prep -> PhaseNet -> QuakeML -> diagnostics.

    1. Requires ``chunks/<event_id>/<event_id>.mseed`` to already exist
       (:func:`_require_chunk_outputs`).
    2. Skips (unless ``force``) if ``cfg.phasenet``/``cfg.stations``, the
       set of chunked event ids, and every chunk file's mtime/size all
       match the last recorded fingerprint
       (:class:`openmeq.manifest.StepManifest`) and
       ``pick/catalog_picks.xml`` still exists.
    3. :func:`openmeq.pick.prep.prepare_all` (M4.2): ``chunks_pn/`` +
       ``data_list.csv``.
    4. :func:`openmeq.pick.run_phasenet.run_phasenet` (M4.3): PhaseNet's
       ``predict.py``, once, over the whole batch -> ``picks.csv``.
    5. :func:`openmeq.pick.to_quakeml.build_catalog` /
       :func:`~openmeq.pick.to_quakeml.write_catalog_picks` (M4.4) ->
       ``pick/catalog_picks.xml``.
    6. :func:`openmeq.pick.diagnostics.make_pick_diagnostics` (M4.4) ->
       ``pick/diagnostics/``.
    7. Marks the manifest entry done and logs the elapsed time.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        Unused by this step (mirrors ``openmeq.chunk.run``) -- accepted
        only because ``openmeq/cli.py``'s dispatcher always supplies it
        (ARCHITECTURE.md §2's ``run(cfg, paths, **opts)`` contract).
    force : bool, keyword-only, optional
        ``--force``: rerun even if the manifest says nothing changed.
    **_ignored
        Absorbs any other step-generic kwargs the dispatcher forwards.
    """
    del config_path  # unused -- see docstring
    t0 = time.perf_counter()
    paths.ensure()
    step_logger = get_logger("pick")

    _require_chunk_outputs(paths)

    event_ids = _event_ids_from_chunks(paths)
    if not event_ids:  # pragma: no cover - _require_chunk_outputs already guards this
        step_logger.info("pick: no chunks found under %s -- nothing to do.", paths.chunk_dir)
        return

    fp = fingerprint(
        cfg.phasenet,
        cfg.stations,
        event_ids,
        *[paths.chunk_file(event_id) for event_id in event_ids],
    )
    manifest = StepManifest(paths, STEP_NAME)
    if not force and manifest.is_done(MANIFEST_KEY, fp) and paths.catalog_picks_xml.is_file():
        step_logger.info(
            "pick: unchanged since the last run (cfg.phasenet, cfg.stations and every "
            "chunks/<event_id>/<event_id>.mseed all match the last recorded fingerprint) -- "
            "skipping (use --force to rerun)."
        )
        return

    station_table = compute_station_table(cfg.stations)

    prepare_all(cfg, paths, event_ids, station_table=station_table)
    # Chunk-step extended QAQC that needs chunks_pn/ (gated by diagnostics.chunk).
    make_chunk_pn_extended_diagnostics(cfg, paths, event_ids)
    picks_csv = run_phasenet(paths, cfg)
    catalog = build_catalog(picks_csv, paths, cfg, station_table=station_table)
    write_catalog_picks(catalog, paths)
    make_pick_diagnostics(paths, catalog)

    manifest.mark_done(MANIFEST_KEY, fp)

    elapsed = time.perf_counter() - t0
    step_logger.info(
        "pick: %d event(s) prepared, %d event(s) with picks (%.2fs).",
        len(event_ids),
        len(catalog.events),
        elapsed,
    )


__all__ = ["MANIFEST_KEY", "STEP_NAME", "run"]
