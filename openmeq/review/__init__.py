"""ObsPyck batch review and compile step.

``run(cfg, paths, **opts)`` (the ``openmeq.cli`` step-module contract,
``docs/ARCHITECTURE.md`` §2) is this package's CLI entry point:
``--compile`` dispatches to M7.4's compile step (not yet built -- logs "not
implemented" like every other not-yet-built step, matching
``openmeq/cli.py``'s own convention); otherwise, if ``cfg.review.enabled``
(``config/example.yaml``: "true to pause the pipeline for manual ObsPyck
review"), it runs M7.3's batch review loop,
:func:`openmeq.review.batch.run_review`; if not, it logs and prints a
one-line "nothing to do" message and returns (this step is documented as
optional, PLAN.md/``openmeq/cli.py``'s own ``review`` command docstring:
"Manual review, ObsPyck batch mode (optional)").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openmeq.log import get_logger
from openmeq.review.batch import run_review

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("review")


def run(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    compile: bool = False,  # noqa: A002 - matches the CLI's own `--compile` option name
    reset: tuple[str, ...] = (),
    reset_all: bool = False,
    **_ignored: Any,
) -> None:
    """``openmeq review`` step entry point.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        The YAML config path ``openmeq/cli.py``'s ``_dispatch`` loaded
        ``cfg`` from -- forwarded to :func:`openmeq.review.batch.run_review`,
        which needs it to launch ObsPyck in a subprocess (see that module's
        docstring for why). Optional here (unlike ``run_review``'s own
        required keyword) so a caller that never reaches the batch loop
        (``compile=True``, or ``cfg.review.enabled`` false -- e.g.
        ``scripts/benchmark.py``, which calls every step's ``run(cfg,
        paths, **opts)`` directly against a dataset that never enables
        review) doesn't have to supply it; a ``ValueError`` is raised if
        it's still missing once the batch loop is actually about to run.
    compile : bool, keyword-only, optional
        ``--compile``: merge reviewed events into
        ``review/catalog_final.xml`` instead of running the batch loop.
        M7.4's job -- not yet implemented.
    reset, reset_all
        ``--reset``/``--reset-all``: forwarded to
        :func:`openmeq.review.batch.run_review`.
    **_ignored
        Absorbs any other step-generic kwargs ``_dispatch`` forwards
        (ARCHITECTURE.md §2's ``run(cfg, paths, **opts)`` contract).
    """
    if compile:
        message = "openmeq review --compile: not implemented (see M7.4)"
        logger.info(message)
        print(message)
        return

    if not cfg.review.enabled:
        message = (
            "openmeq review: cfg.review.enabled is false, nothing to do "
            "(set review.enabled: true in the config to pause the pipeline "
            "for manual ObsPyck review)"
        )
        logger.info(message)
        print(message)
        return

    if config_path is None:
        raise ValueError(
            "openmeq.review.run(): config_path is required when cfg.review.enabled "
            "is true -- the batch loop launches ObsPyck in a subprocess that reloads "
            "this same OpenMEQ config file (see openmeq.review._launch_obspyck)"
        )

    result = run_review(
        cfg,
        paths,
        config_path=config_path,
        reset=reset,
        reset_all=reset_all,
    )
    logger.info(
        "openmeq review: %d event(s) processed (%d done, %d skipped, %d rejected)%s",
        result.processed,
        result.done,
        result.skipped,
        result.rejected,
        ", quit early" if result.quit else "",
    )
