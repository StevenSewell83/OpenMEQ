#!/usr/bin/env python
"""CLI wrapper around ``openmeq.nlloc_setup`` (M5.2 / OpenMEQ-mev.2, extended
for 3D model grids by M5.3 / OpenMEQ-mev.3; PLAN.md §7).

Builds the shared velocity/travel-time NLLoc grids for every configured
station -- ``cfg.nlloc.velocity_model.type == "1d"`` runs ``Vel2Grid`` once
from the project's layer file; ``"3d"`` validates (coverage + TRANS match)
and links in a user-supplied NLLoc grid instead -- then runs ``Grid2Time``
per (station, phase) in parallel via ``openmeq.parallel.pmap``, and
refreshes the ObsPyck model link. ``openmeq.nlloc_setup.setup_nlloc`` (all
of the actual logic lives in ``openmeq/nlloc_setup.py``) dispatches between
the two based on that config field; this script is a thin wrapper so M6.1
(the per-event NLLoc runner, OpenMEQ-xdf.1) can ``import openmeq.nlloc_setup``
and reuse its control-file generation directly, without going through this
CLI.

Usage
-----
::

    python scripts/setup_nlloc.py -c config.yaml [--force]
"""

from __future__ import annotations

import click

from openmeq.config import load_config
from openmeq.log import configure_logging, get_logger
from openmeq.nlloc_setup import setup_nlloc
from openmeq.paths import RunPaths


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
    "--force",
    is_flag=True,
    default=False,
    help="Rebuild the grids and ObsPyck link even if nothing changed since the last run.",
)
def main(config_path: str, force: bool) -> None:
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    configure_logging(paths, "nlloc")
    logger = get_logger("nlloc_setup")

    rebuilt = setup_nlloc(cfg, paths, force=force)
    if rebuilt:
        logger.info("NLLoc setup rebuilt: velocity/travel-time grids and ObsPyck model link.")
    else:
        logger.info("NLLoc setup unchanged since the last run -- skipped (use --force to rebuild).")


if __name__ == "__main__":
    main()
