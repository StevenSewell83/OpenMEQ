"""Local magnitude (ML) calculation.

Implemented in M8. ``openmeq.magnitude.amplitude`` (M8.1) implements the
per-event Wood-Anderson amplitude measurement:
:func:`~openmeq.magnitude.amplitude.measure_event_amplitudes`.
``openmeq.magnitude.relations`` (M8.2) implements the attenuation-relation
layer: :func:`~openmeq.magnitude.relations.station_ml` and its supporting
loaders. ``openmeq.magnitude.pipeline`` (M8.3) implements event-level ML
aggregation, catalogue selection, QuakeML/CSV writing, diagnostics and the
``openmeq magnitude`` CLI command: :func:`~openmeq.magnitude.pipeline.run_magnitude`,
wired up below as this package's ``run(cfg, paths, **opts)`` (the
``openmeq.cli`` step-module contract, ARCHITECTURE.md §2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from openmeq.magnitude.pipeline import run_magnitude

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    catalog: str | Path | None = None,
    force: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq magnitude`` step entry point -- see
    :func:`~openmeq.magnitude.pipeline.run_magnitude`.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        The YAML config path ``openmeq/cli.py``'s ``_dispatch`` loaded
        ``cfg`` from -- used to resolve a relative ``catalog`` override
        against the config file's own directory
        (:func:`~openmeq.magnitude.pipeline.select_catalog_path`).
    catalog : str, pathlib.Path or None, keyword-only, optional
        ``--catalog``: override the input catalogue (default: see
        :func:`~openmeq.magnitude.pipeline.select_catalog_path`).
    force : bool, keyword-only, optional
        ``--force``: recompute even if the manifest says nothing changed.
    **_ignored
        Absorbs any other step-generic kwargs ``_dispatch`` forwards
        (ARCHITECTURE.md §2's ``run(cfg, paths, **opts)`` contract).
    """
    run_magnitude(
        cfg,
        paths,
        Path(catalog) if catalog is not None else None,
        config_path=config_path,
        force=force,
    )
