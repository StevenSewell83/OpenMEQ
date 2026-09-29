"""OpenMEQ command-line interface (``openmeq <step>``).

``openmeq`` is a ``click`` group with one subcommand per PLAN.md §6/§8
pipeline step (``qc``, ``detect``, ``chunk``, ``pick``, ``locate``,
``review``, ``magnitude``, ``relocate``) plus ``run`` (the whole chain,
M11.1). Every subcommand takes ``-c``/``--config`` (PLAN.md §3: ``openmeq
<step> -c config.yaml``).

Registering a step's real implementation
-----------------------------------------
Each subcommand:

1. loads the config (:func:`openmeq.config.load_config`);
2. builds :class:`openmeq.paths.RunPaths` and calls ``.ensure()``;
3. calls :func:`openmeq.log.configure_logging`;
4. lazily imports that step's module (``openmeq.qc``, ``openmeq.detect``,
   ...) and looks for a module-level ``run(cfg, paths, **opts)`` function.

A milestone "fills in" a step purely by adding that ``run`` function to its
module -- **no change to this file is needed**. Until a step defines
``run``, its subcommand logs (and prints) "not implemented (see <plan key>)"
and exits 0. The import is lazy (done inside the command, not at module
scope) precisely so ``openmeq --help`` -- and every other subcommand --
keeps working before any step module exists or while one is mid-build and
temporarily broken.

``run`` (``openmeq run``, M11.1) chains every step's own registered
``run(cfg, paths, **opts)`` in process, sharing one loaded
:class:`~openmeq.config.Config`/:class:`~openmeq.paths.RunPaths` pair across
all of them (only ``openmeq.log.configure_logging`` is re-invoked per step,
same as ``_dispatch``, so each step still gets its own ``logs/<step>.log``
and its own manifest-based "skip unchanged" check applies exactly as it
would standalone). :func:`run_all` is the plain, CliRunner-independent
entry point the ``run`` command's body calls -- see its own docstring for
the step sequence, the ``review`` branching rule and failure handling.

``diagnostics funnel`` (OpenMEQ-02o.8) writes the run-level
``diagnostics/event_funnel.{png,csv}`` (:mod:`openmeq.run_diagnostics`) on
demand; ``openmeq run`` also writes it at the end of every run (successful
or not) when ``cfg.diagnostics.extended_enabled("run")``.
"""

from __future__ import annotations

import dataclasses
import importlib
import time
from pathlib import Path
from typing import Any

import click

from openmeq.config import Config, ConfigError, load_config
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


def _load_cfg_and_paths(config_path: str) -> tuple[Config, RunPaths]:
    """Load the config and build+``ensure()`` its :class:`RunPaths`.

    Shared by every subcommand (via :func:`_dispatch`) and by ``run``
    (:func:`run_all`'s caller), so a chained ``openmeq run`` loads the
    config and creates the run directory tree exactly once instead of once
    per step (ARCHITECTURE.md §2 / this task's own design note).
    """
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, paths


def _run_step(step: str, cfg: Config, paths: RunPaths, config_path: str, **opts: Any) -> Any:
    """Configure logging for ``step``, then hand off to its registered module.

    The common body every step subcommand needs *after* config/paths are
    already available -- factored out of ``_dispatch`` so :func:`run_all`
    can call it once per chained step without reloading the config or
    re-running :meth:`RunPaths.ensure` each time.

    Parameters
    ----------
    step : str
        The CLI subcommand name; also used as the ``openmeq.log`` step name
        (``logs/<step>.log``).
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
        Already ``.ensure()``-d.
    config_path : str
        ``-c``/``--config`` value (forwarded to the step's own ``run`` as
        ``config_path``, resolved to an absolute path).
    **opts
        Extra, step-specific options (e.g. ``review --compile``,
        ``magnitude --catalog``), forwarded to the step's ``run`` once one
        exists.

    Returns
    -------
    Any
        Whatever the step's own ``run`` returns (``None`` for most steps;
        e.g. ``relocate``'s returns a ``RelocateResult``). ``None`` if the
        step is still a stub.
    """
    configure_logging(paths, step)
    logger = get_logger(step)

    module_name, plan_key = _STEP_MODULES[step]
    module = importlib.import_module(module_name)
    run_func = getattr(module, "run", None)
    if run_func is None:
        message = f"openmeq {step}: not implemented (see {plan_key})"
        logger.info(message)
        click.echo(message)
        return None

    logger.info("openmeq %s: starting (config=%s)", step, config_path)
    # Every step module's `run` follows the `run(cfg, paths, **opts)`
    # contract (ARCHITECTURE.md §2), so passing `config_path` here is always
    # safe even for a step whose own `run` doesn't use it -- it just lands
    # in that `run`'s own `**opts`. M7.3 (`openmeq.review`) is the first
    # step that actually needs it (its batch loop launches ObsPyck in a
    # subprocess, which must reload the same config file).
    result = run_func(cfg, paths, config_path=str(Path(config_path).resolve()), **opts)
    logger.info("openmeq %s: done", step)
    return result


def _dispatch(step: str, config_path: str, **opts: Any) -> None:
    """Common body for every single-step subcommand: setup, then hand off to the step module.

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
    cfg, paths = _load_cfg_and_paths(config_path)
    _run_step(step, cfg, paths, config_path, **opts)


# ---------------------------------------------------------------------------
# `openmeq run` (M11.1): chain every step in order, sharing one Config/RunPaths.
# ---------------------------------------------------------------------------

#: The 8 pipeline steps in dependency order (PLAN.md §6/§9), matching
#: ``_STEP_MODULES``'s keys and every step's own CLI subcommand name --
#: used for both the fixed execution order and the ``--from``/``--to``
#: sub-range options.
STEP_ORDER: tuple[str, ...] = (
    "qc",
    "detect",
    "chunk",
    "pick",
    "locate",
    "review",
    "magnitude",
    "relocate",
)


@dataclasses.dataclass
class StepOutcome:
    """One chained step's result, as reported in ``run_all``'s summary table."""

    step: str
    status: str  #: ``"ok"``, ``"skipped"`` or ``"failed"``.
    detail: str
    elapsed_s: float


@dataclasses.dataclass
class RunAllResult:
    """What one :func:`run_all` call did: every step's :class:`StepOutcome`,
    in order, plus which step (if any) stopped the chain."""

    outcomes: list[StepOutcome]
    failed_step: str | None = None

    @property
    def ok(self) -> bool:
        return self.failed_step is None


class StepFailed(Exception):
    """Raised by :func:`run_all` when a chained step raises.

    Carries the failing step's name and a message naming it and how to
    resume (``openmeq run --from <step>``); ``openmeq/cli.py``'s ``run``
    command turns this into a `click.ClickException` (clean message, no
    traceback, non-zero exit).
    """

    def __init__(self, step: str, config_path: str, cause: BaseException) -> None:
        self.step = step
        self.cause = cause
        message = (
            f"openmeq run: step '{step}' failed: {cause}. "
            f"Resume with `openmeq run -c {config_path} --from {step}`."
        )
        super().__init__(message)


def _step_range(from_step: str | None, to_step: str | None) -> list[str]:
    """Resolve ``--from``/``--to`` into the ordered sub-list of :data:`STEP_ORDER` to run."""
    for name, value in (("--from", from_step), ("--to", to_step)):
        if value is not None and value not in STEP_ORDER:
            raise click.ClickException(
                f"openmeq run: unknown {name} step {value!r} (choices: {', '.join(STEP_ORDER)})"
            )
    start = STEP_ORDER.index(from_step) if from_step is not None else 0
    end = STEP_ORDER.index(to_step) if to_step is not None else len(STEP_ORDER) - 1
    if start > end:
        raise click.ClickException(
            f"openmeq run: --from {from_step!r} comes after --to {to_step!r} in the pipeline "
            f"order ({', '.join(STEP_ORDER)})"
        )
    return list(STEP_ORDER[start : end + 1])


def _run_review_step(
    cfg: Config, paths: RunPaths, *, config_path: str, force: bool, skip_review: bool
) -> tuple[str, str]:
    """The ``review`` link in the chain -- see :func:`run_all`'s own docstring
    for the full rationale (ARCHITECTURE.md §1's ``requires_gui`` marker is
    the reason this never runs the interactive batch loop itself).

    Returns
    -------
    (status, detail) : tuple[str, str]
    """
    logger = get_logger("review")

    if skip_review:
        detail = "skipped (--skip-review)"
        logger.info("openmeq run: review: %s", detail)
        return "skipped", detail

    if not cfg.review.enabled:
        # Safe no-op: `openmeq.review.run` itself just logs "nothing to do"
        # and returns without touching the (interactive) batch loop at all.
        _run_step("review", cfg, paths, config_path, compile=False, force=force)
        return "skipped", "cfg.review.enabled is false"

    reviewed_events = (
        any(paths.review_events_dir.glob("*.xml")) if paths.review_events_dir.is_dir() else False
    )
    if paths.catalog_final_xml.is_file() or reviewed_events:
        # A prior (separate, interactive) `openmeq review` session already
        # produced review/events/*.xml and/or review/catalog_final.xml --
        # (re)compiling here is always safe (never launches ObsPyck) and
        # keeps catalog_final.xml current with anything reviewed since the
        # last compile; it is itself skip-unchanged (its own StepManifest).
        _run_step("review", cfg, paths, config_path, compile=True, force=force)
        return "ok", "compiled review/catalog_final.xml"

    message = (
        "openmeq run: review.enabled is true but no events have been reviewed yet "
        f"(no {paths.review_events_dir} entries) and {paths.catalog_final_xml} does not "
        "exist -- review is interactive (ARCHITECTURE.md §1's requires_gui marker), so "
        "`openmeq run` never launches it automatically. Run `openmeq review -c "
        f"{config_path}` to review events, then `openmeq review -c {config_path} --compile` "
        "(or just re-run `openmeq run`, which will pick up review/catalog_final.xml "
        "automatically next time). Continuing this run with locate/catalog_auto.xml."
    )
    logger.info(message)
    click.echo(message)
    return "skipped", "pending manual review; using locate/catalog_auto.xml"


def run_all(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str,
    force: bool = False,
    from_step: str | None = None,
    to_step: str | None = None,
    skip_review: bool = False,
) -> RunAllResult:
    """Run the whole pipeline (or a ``--from``/``--to`` sub-range) in process.

    Calls each step's own registered ``run(cfg, paths, **opts)``
    (ARCHITECTURE.md §2's step-module contract) in PLAN.md §6/§9 order --
    ``qc -> detect -> chunk -> pick -> locate -> review -> magnitude ->
    relocate`` -- sharing one already-loaded ``cfg``/``paths`` across all of
    them (only ``openmeq.log.configure_logging`` is re-invoked per step, via
    :func:`_run_step`). Every step's own :class:`openmeq.manifest.StepManifest`
    "skip unchanged" check applies exactly as it does standalone -- this
    function implements no change-detection of its own.

    ``review`` branching (:func:`_run_review_step`)
    -------------------------------------------------
    ``openmeq review``'s batch loop launches ObsPyck (a GUI subprocess) per
    event and is documented as interactive (ARCHITECTURE.md §1's
    ``requires_gui`` marker) -- ``openmeq run`` never launches it silently.
    Instead:

    * ``--skip-review``, or ``cfg.review.enabled`` false: nothing to do,
      ``magnitude`` falls through to ``locate/catalog_auto.xml``.
    * ``cfg.review.enabled`` true and a prior (separate, interactive)
      ``openmeq review`` session already produced ``review/events/*.xml``
      and/or ``review/catalog_final.xml``: safely (re)compile it
      (``openmeq review --compile``, never interactive) so it's current.
    * ``cfg.review.enabled`` true but nothing has been reviewed yet: log a
      clear message naming ``openmeq review -c <config>`` as the separate
      command to run, and continue with ``locate/catalog_auto.xml``.

    ``magnitude`` is always called with ``catalog=None`` (never an
    override), so its own documented catalogue-selection rule
    (:func:`openmeq.magnitude.pipeline.select_catalog_path` --
    ``catalog_final.xml`` if it exists, else ``catalog_auto.xml``) is the
    one and only place that logic lives.

    Failure handling
    ------------------
    If a step raises, the chain stops immediately (later steps are never
    attempted) and :class:`StepFailed` is raised, naming the failing step
    and the ``--from`` option to resume with.

    Either way, the run-level event funnel (:mod:`openmeq.run_diagnostics`)
    is written last when ``cfg.diagnostics.extended_enabled("run")``; it
    never raises, so it cannot change the outcome.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
        Already ``.ensure()``-d.
    config_path : str, keyword-only
        ``-c``/``--config`` value, forwarded to every step and used in the
        ``StepFailed``/review-pending messages.
    force : bool, keyword-only, optional
        ``--force``: forwarded to every step (steps that don't take
        ``force`` absorb it via their own ``**_ignored``).
    from_step, to_step : str or None, keyword-only, optional
        ``--from``/``--to``: run only the inclusive sub-range of
        :data:`STEP_ORDER` between them (default: the whole chain).
    skip_review : bool, keyword-only, optional
        ``--skip-review``: never touch the ``review`` step at all, even if
        ``cfg.review.enabled`` is true.

    Returns
    -------
    RunAllResult

    Raises
    ------
    StepFailed
        On the first step that raises; ``state.outcomes`` up to that point
        are not returned (the caller should catch this and print its own
        partial summary -- ``openmeq/cli.py``'s ``run`` command does).
    """
    steps = _step_range(from_step, to_step)
    run_logger = get_logger("run")
    configure_logging(paths, "run")
    run_logger.info(
        "openmeq run: starting %s (config=%s, force=%s, skip_review=%s)",
        " -> ".join(steps),
        config_path,
        force,
        skip_review,
    )

    outcomes: list[StepOutcome] = []
    for step in steps:
        t0 = time.perf_counter()
        try:
            if step == "review":
                status, detail = _run_review_step(
                    cfg, paths, config_path=config_path, force=force, skip_review=skip_review
                )
            else:
                opts: dict[str, Any] = {"force": force}
                if step == "magnitude":
                    opts["catalog"] = None
                _run_step(step, cfg, paths, config_path, **opts)
                status, detail = "ok", "done"
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any step failure stops the chain
            elapsed = time.perf_counter() - t0
            outcomes.append(StepOutcome(step, "failed", str(exc), elapsed))
            configure_logging(paths, "run")
            run_logger.error("openmeq run: step %r failed after %.2fs: %s", step, elapsed, exc)
            click.echo(_format_summary_table(outcomes))
            _maybe_run_diagnostics(cfg, paths)
            raise StepFailed(step, config_path, exc) from exc
        elapsed = time.perf_counter() - t0
        outcomes.append(StepOutcome(step, status, detail, elapsed))
        configure_logging(paths, "run")

    total = sum(o.elapsed_s for o in outcomes)
    run_logger.info("openmeq run: done (%d step(s), %.2fs total)", len(outcomes), total)
    click.echo(_format_summary_table(outcomes))
    _maybe_run_diagnostics(cfg, paths)
    return RunAllResult(outcomes=outcomes)


def _maybe_run_diagnostics(cfg: Config, paths: RunPaths) -> None:
    """Write the run-level event funnel (OpenMEQ-02o.8) if
    ``cfg.diagnostics.extended_enabled("run")`` -- after a full run and after
    a failed partial one alike. Never raises: a diagnostic can't change the
    run's outcome (:func:`openmeq.run_diagnostics.make_event_funnel` already
    swallows its own errors; this also guards the import)."""
    if not cfg.diagnostics.extended_enabled("run"):
        return
    try:
        from openmeq.run_diagnostics import make_event_funnel

        make_event_funnel(cfg, paths)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the run
        get_logger("run").warning("openmeq run: event funnel diagnostic failed: %s", exc)


def _format_summary_table(outcomes: list[StepOutcome]) -> str:
    """Render the final ``step | status | wall time`` summary table."""
    header = f"{'step':<10} {'status':<8} {'wall time':>10}  detail"
    lines = [header, "-" * len(header)]
    for o in outcomes:
        lines.append(f"{o.step:<10} {o.status:<8} {o.elapsed_s:>9.2f}s  {o.detail}")
    total = sum(o.elapsed_s for o in outcomes)
    lines.append("-" * len(header))
    lines.append(f"{'total':<10} {'':<8} {total:>9.2f}s")
    return "\n".join(lines)


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
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Rerun even if qc/channel_flags.json, inventory/index.json and cfg.detection "
    "are unchanged since the last run.",
)
def detect(config_path: str, force: bool) -> None:
    """Network coincidence detection."""
    _dispatch("detect", config_path, force=force)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Rerun even if chunks/completeness.csv, detections/detections.csv and "
    "inventory/index.json are unchanged since the last run.",
)
def chunk(config_path: str, force: bool) -> None:
    """Per-event waveform chunk writing."""
    _dispatch("chunk", config_path, force=force)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Rerun even if cfg.phasenet, cfg.stations and every chunks/<event_id>/<event_id>.mseed "
    "are unchanged since the last run.",
)
def pick(config_path: str, force: bool) -> None:
    """PhaseNet preparation, running PhaseNet, and converting picks to QuakeML."""
    _dispatch("pick", config_path, force=force)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Rerun even if cfg.nlloc and pick/catalog_picks.xml are unchanged since the last run.",
)
def locate(config_path: str, force: bool) -> None:
    """Two-pass NonLinLoc location and residual filtering."""
    _dispatch("locate", config_path, force=force)


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
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Recompute magnitudes even if the input catalogue and cfg.magnitude are "
    "unchanged since the last run.",
)
def magnitude(config_path: str, catalog: str | None, force: bool) -> None:
    """Local magnitude (ML) calculation."""
    _dispatch("magnitude", config_path, catalog=catalog, force=force)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Rerun every relocate sub-step (ph2dt, cross-correlation, hypoDD) even if "
    "relocate/manifest.json says nothing changed.",
)
def relocate(config_path: str, force: bool) -> None:
    """hypoDD cross-correlation and relocation."""
    _dispatch("relocate", config_path, force=force)


@main.command()
@_CONFIG_OPTION
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help="Forward --force to every chained step (recompute even if unchanged).",
)
@click.option(
    "--from",
    "from_step",
    type=click.Choice(STEP_ORDER),
    default=None,
    help="Start the chain at this step instead of qc (e.g. --from locate to resume).",
)
@click.option(
    "--to",
    "to_step",
    type=click.Choice(STEP_ORDER),
    default=None,
    help="Stop the chain after this step instead of relocate (e.g. --to locate for a partial run).",
)
@click.option(
    "--skip-review",
    "skip_review",
    is_flag=True,
    default=False,
    help="Never touch the review step, even if cfg.review.enabled is true.",
)
def run(
    config_path: str,
    force: bool,
    from_step: str | None,
    to_step: str | None,
    skip_review: bool,
) -> None:
    """Run the whole chain: qc -> detect -> chunk -> pick -> locate -> [review] -> magnitude ->
    relocate.

    Each step is called via its own registered ``run(cfg, paths, **opts)``
    (ARCHITECTURE.md §2), so every step's manifest-based skip-unchanged
    check applies exactly as it would running that step standalone. See
    :func:`run_all` for the full behaviour, especially the ``review`` step's
    branching rule (it is interactive and is never launched automatically).
    """
    cfg, paths = _load_cfg_and_paths(config_path)
    try:
        run_all(
            cfg,
            paths,
            config_path=str(Path(config_path).resolve()),
            force=force,
            from_step=from_step,
            to_step=to_step,
            skip_review=skip_review,
        )
    except StepFailed as exc:
        raise click.ClickException(str(exc)) from exc


@main.group()
def diagnostics() -> None:
    """Run-level diagnostics, generated standalone from a run's on-disk outputs."""


@diagnostics.command("funnel")
@_CONFIG_OPTION
def diagnostics_funnel(config_path: str) -> None:
    """Event counts per stage -> diagnostics/event_funnel.{png,csv}.

    Always written when asked for here (``cfg.diagnostics.run`` only gates
    the automatic copy at the end of ``openmeq run``).
    """
    from openmeq.run_diagnostics import make_event_funnel

    cfg, paths = _load_cfg_and_paths(config_path)
    configure_logging(paths, "run")
    png = make_event_funnel(cfg, paths)
    if png is None:
        raise click.ClickException(
            f"openmeq diagnostics funnel: failed (see {paths.log_file('run')})"
        )
    click.echo(f"wrote {png} and {png.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
