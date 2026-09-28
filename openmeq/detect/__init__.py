"""Network coincidence detection.

``openmeq.detect.worker`` (M2.1, ``OpenMEQ-ujx.1``) has the per-window/
per-day detection core (:func:`~openmeq.detect.worker.detect_window`,
:func:`~openmeq.detect.worker.build_windows`). ``openmeq.detect.merge``
(M2.2, ``OpenMEQ-ujx.2``) merges/de-duplicates triggers across windows and
assigns event ids. ``openmeq.detect.diagnostics`` and
``openmeq.detect.pipeline`` (M2.3, ``OpenMEQ-ujx.3``) wire those pieces
together into the real ``openmeq detect`` CLI step via this module's
``run(cfg, paths, **opts)`` (the ``openmeq.cli`` step-module contract,
ARCHITECTURE.md §2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openmeq.detect.pipeline import run_detect

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    force: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq detect`` step entry point -- see :func:`~openmeq.detect.pipeline.run_detect`.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        The YAML config path ``openmeq/cli.py``'s ``_dispatch`` loaded
        ``cfg`` from -- forwarded to :func:`~openmeq.detect.pipeline.run_detect`,
        which needs it to hand each parallel window worker a reloadable
        config path (see that function's docstring). ``_dispatch`` always
        supplies it for a real CLI invocation; optional here only so a
        caller that skips straight past the point it's needed (e.g. an
        already-unchanged manifest, or a harness that never reaches that
        point) doesn't have to supply it.
    force : bool, keyword-only, optional
        ``--force``: rerun even if the manifest says nothing changed.
    **_ignored
        Absorbs any other step-generic kwargs ``_dispatch`` (or a caller
        like ``scripts/benchmark.py``) forwards (ARCHITECTURE.md §2's
        ``run(cfg, paths, **opts)`` contract).
    """
    run_detect(cfg, paths, config_path=config_path, force=force)
