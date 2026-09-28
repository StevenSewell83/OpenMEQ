"""OpenMEQ command-line interface (``openmeq <step>``).

``openmeq`` is a ``click`` group with one subcommand per PLAN.md §6/§8
pipeline step (``qc``, ``detect``, ``chunk``, ``pick``, ``locate``,
``review``, ``magnitude``, ``relocate``) plus ``run`` (the whole chain,
fully implemented only in M11.1). Every subcommand takes ``-c``/``--config``
(PLAN.md §3: ``openmeq <step> -c config.yaml``).

Registering a step's real implementation
-----------------------------------------
This task (M0.4) only wires the CLI end to end -- no subcommand does real
work yet. Each subcommand:

1. loads the config (:func:`openmeq.config.load_config`);
2. builds :class:`openmeq.paths.RunPaths` and calls ``.ensure()``;
3. calls :func:`openmeq.log.configure_logging`;
4. lazily imports that step's module (``openmeq.qc``, ``openmeq.detect``,
   ...) and looks for a module-level ``run(cfg, paths, **opts)`` function.

A later milestone "fills in" a step purely by adding that ``run`` function
to its module -- **no change to this file is needed**. Until a step defines
``run``, its subcommand logs (and prints) "not implemented (see <plan key>)"
and exits 0, which is what M1-M10 replace. The import is lazy (done inside
the command, not at module scope) precisely so ``openmeq --help`` -- and
every other subcommand -- keeps working before any step module exists or
while one is mid-build and temporarily broken.

``run`` (``openmeq run``) has no ``openmeq.run`` module to import: its real
chaining behaviour is M11.1's own job, built directly in this file once that
task starts; here it is a stub that only logs "not implemented".
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click

from openmeq.config import ConfigError, load_config
from openmeq.log import configure_logging, get_logger
from openmeq.paths import RunPaths

#: subcommand name -> (module to lazily import, plan key shown in the
#: "not implemented" message). Keys are ``openmeq.<name>`` except where the
#: step's implementation lives at a flat module (``chunk``), matching
#: docs/PLAN.md §8's repository layout.
_STEP_MODULES: dict[str, tuple[str, str]] = {
    "qc": ("openmeq.qc", "M1.7"),
    "detect": ("openmeq.detect", "M2.3"),
    "chunk": ("openmeq.chunk", "M3.2"),
    "pick": ("openmeq.pick", "M4.4"),
    "locate": ("openmeq.locate", "M6.5"),
    "review": ("openmeq.review", "M7.x"),
    "magnitude": ("openmeq.magnitude", "M8.3"),
    "relocate": ("openmeq.relocate", "M10.5"),
}

_CONFIG_OPTION = click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the YAML config file (see config/example.yaml).",
)


def _dispatch(step: str, config_path: str, **opts: Any) -> None:
    """Common body for every step subcommand: setup, then hand off to the step module.

    Parameters
    ----------
    step : str
        The CLI subcommand name; also used as the ``openmeq.log`` step name
        (``logs/<step>.log``).
    config_path : str
        ``-c``/``--config`` value.
    **opts
        Extra, step-specific options (e.g. ``review --compile``,
        ``magnitude --catalog``), forwarded to the step's ``run`` once one
        exists.
    """
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    paths = RunPaths(cfg)
    paths.ensure()
    configure_logging(paths, step)
    logger = get_logger(step)

    module_name, plan_key = _STEP_MODULES[step]
    module = importlib.import_module(module_name)
    run_func = getattr(module, "run", None)
    if run_func is None:
        message = f"openmeq {step}: not implemented (see {plan_key})"
        logger.info(message)
        click.echo(message)
        return

    logger.info("openmeq %s: starting (config=%s)", step, config_path)
    # Every step module's `run` follows the `run(cfg, paths, **opts)`
    # contract (ARCHITECTURE.md §2), so passing `config_path` here is always
    # safe even for a step whose own `run` doesn't use it -- it just lands
    # in that `run`'s own `**opts`. M7.3 (`openmeq.review`) is the first
    # step that actually needs it (its batch loop launches ObsPyck in a
    # subprocess, which must reload the same config file).
    run_func(cfg, paths, config_path=str(Path(config_path).resolve()), **opts)
    logger.info("openmeq %s: done", step)


@click.group()
def main() -> None:
    """OpenMEQ: pipeline for microearthquake detection, location and relocation.

    Run ``openmeq <step> -c config.yaml`` for one pipeline step, or
    ``openmeq run -c config.yaml`` for the whole chain.
    """


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Recompute every QC output even if inputs haven't changed.",
)
def qc(config_path: str, force: bool) -> None:
    """Data QC: index, station metadata, availability, noise stats, channel flags."""
    _dispatch("qc", config_path, force=force)


@main.command()
@_CONFIG_OPTION
def detect(config_path: str) -> None:
    """Network coincidence detection."""
    _dispatch("detect", config_path)


@main.command()
@_CONFIG_OPTION
def chunk(config_path: str) -> None:
    """Per-event waveform chunk writing."""
    _dispatch("chunk", config_path)


@main.command()
@_CONFIG_OPTION
def pick(config_path: str) -> None:
    """PhaseNet preparation, running PhaseNet, and converting picks to QuakeML."""
    _dispatch("pick", config_path)


@main.command()
@_CONFIG_OPTION
def locate(config_path: str) -> None:
    """Two-pass NonLinLoc location and residual filtering."""
    _dispatch("locate", config_path)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--compile",
    "compile_",
    is_flag=True,
    default=False,
    help="Merge reviewed events over the automatic catalogue into review/catalog_final.xml.",
)
@click.option(
    "--reset",
    "reset_events",
    metavar="EVENT_ID",
    multiple=True,
    help="Clear this event's review/state.json entry so it is offered again. May be repeated.",
)
@click.option(
    "--reset-all",
    "reset_all",
    is_flag=True,
    default=False,
    help="Clear every event's review/state.json entry so all are offered again.",
)
def review(
    config_path: str, compile_: bool, reset_events: tuple[str, ...], reset_all: bool
) -> None:
    """Manual review, ObsPyck batch mode (optional)."""
    _dispatch("review", config_path, compile=compile_, reset=reset_events, reset_all=reset_all)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--catalog",
    "catalog",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Override the input catalogue (default: review/catalog_final.xml if review "
    "was run, else locate/catalog_auto.xml).",
)
def magnitude(config_path: str, catalog: str | None) -> None:
    """Local magnitude (ML) calculation."""
    _dispatch("magnitude", config_path, catalog=catalog)


@main.command()
@_CONFIG_OPTION
def relocate(config_path: str) -> None:
    """hypoDD cross-correlation and relocation."""
    _dispatch("relocate", config_path)


@main.command()
@_CONFIG_OPTION
def run(config_path: str) -> None:
    """Run the whole chain (stub; the real chaining behaviour is M11.1's job)."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    paths = RunPaths(cfg)
    paths.ensure()
    configure_logging(paths, "run")
    logger = get_logger("run")
    message = "openmeq run: not implemented (see M11.1)"
    logger.info(message)
    click.echo(message)


if __name__ == "__main__":
    main()
