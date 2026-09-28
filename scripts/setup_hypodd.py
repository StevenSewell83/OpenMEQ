#!/usr/bin/env python
"""CLI wrapper around ``openmeq.relocate.setup`` (M9.5 / OpenMEQ-6ld.5;
PLAN.md §7).

Prepares hypoDD/ph2dt's static run environment: writes ``station.dat``,
``ph2dt.inp`` and ``hypoDD.inp`` (1D, ``IMOD=1``, or 3D SIMUL2000,
``IMOD=9``, per ``cfg.hypodd.velocity_model.type``), refreshes the
persisted event-id/station-label maps (``relocate/work/*.csv``), checks the
dataset against hypoDD/ph2dt's currently *compiled* array limits and
recompiles with headroom if they're exceeded, and -- for a 3D run --
verifies every station/event actually falls inside the 3D model's node
domain (a real memory-safety hazard otherwise, docs/hypodd_3d_notes.md).
All of the actual logic lives in ``openmeq.relocate.setup.setup_hypodd``
(and the modules it composes: ``openmeq.relocate.hypodd_files``,
``openmeq.relocate.hypodd_limits``, ``openmeq.relocate.nlloc_to_simul``) --
this script is a thin wrapper, so M10.2/M10.4 (the actual relocation
pipeline step) can ``import openmeq.relocate.setup`` and call the same
function directly instead of shelling out to this CLI.

Deliberately separate from ``openmeq relocate``'s own pipeline step:
PLAN.md §9 allows M9 to be built and *run* independently of the rest of the
pipeline, including on day one of a project before any events have been
detected -- re-running this script (e.g. after changing the velocity model)
never requires re-running detection/picking/location, and it degrades
gracefully (a logged warning, not a crash) when
``magnitude/catalog_mag.xml`` doesn't exist yet, simply skipping the
compiled-limit check until it does.

Usage
-----
::

    python scripts/setup_hypodd.py -c config.yaml [--nlloc-to-simul] [--force]

``--nlloc-to-simul`` converts the configured NLLoc 3D grid
(``cfg.nlloc.velocity_model.file``) into hypoDD's SIMUL2000 ``vel3D.dat``
format (written to ``cfg.hypodd.velocity_model.file``) before the rest of
the sequence runs -- only valid when ``cfg.hypodd.velocity_model.type ==
"3d"``. Omit it to use a user-supplied ``vel3D.dat`` (or for a 1D run,
where it doesn't apply).
"""

from __future__ import annotations

import click

from openmeq.config import load_config
from openmeq.log import configure_logging, get_logger
from openmeq.paths import RunPaths
from openmeq.relocate.setup import setup_hypodd


@click.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml.",
)
@click.option(
    "--nlloc-to-simul",
    "nlloc_to_simul",
    is_flag=True,
    default=False,
    help=(
        "Convert the configured NLLoc 3D grid (cfg.nlloc.velocity_model.file) into a "
        "SIMUL2000 vel3D.dat (cfg.hypodd.velocity_model.file) before proceeding. Only valid "
        "when cfg.hypodd.velocity_model.type == '3d'."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Rewrite every file and re-check compiled limits even if nothing changed since last run.",
)
def main(config_path: str, nlloc_to_simul: bool, force: bool) -> None:
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    configure_logging(paths, "hypodd_setup")
    logger = get_logger("setup_hypodd")

    result = setup_hypodd(cfg, paths, force=force, nlloc_to_simul=nlloc_to_simul)

    if not result.rebuilt:
        logger.info(
            "hypoDD setup unchanged since the last run -- skipped (use --force to rebuild)."
        )
        return

    logger.info(
        "hypoDD setup complete: IMOD=%d, --nlloc-to-simul %s, compiled-limit check %s.",
        result.imod,
        "ran" if result.nlloc_to_simul_ran else "not requested",
        (
            "skipped (no catalogue yet)"
            if result.limit_check is None
            else ("recompiled (limits exceeded)" if result.recompiled else "OK")
        ),
    )


if __name__ == "__main__":
    main()
