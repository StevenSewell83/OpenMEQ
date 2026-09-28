#!/usr/bin/env python3
"""OpenMEQ pipeline benchmark harness (M0.9, docs/PLAN.md §4 last bullet).

Times each *implemented* ``openmeq`` CLI step (docs/ARCHITECTURE.md §2's
``openmeq.cli`` table: ``qc``, ``detect``, ``chunk``, ``pick``, ``locate``,
``review``, ``magnitude``, ``relocate``) against the synthetic test dataset
(``tests/data/make_synthetic.py``), and writes a timestamped JSON + Markdown
report under ``benchmarks/results/``.

Usage
-----
::

    # Generate a fresh synthetic dataset, run every step openmeq.cli knows
    # about that has a real ``run()`` yet (others are reported "skipped"),
    # and write benchmarks/results/<timestamp>.{json,md}.
    python scripts/benchmark.py

    # Reuse an existing project instead of generating one (skips the
    # "generate_synthetic_dataset" stage):
    python scripts/benchmark.py --config /path/to/project/config.yaml

    # Only time a subset of the pipeline steps:
    python scripts/benchmark.py --steps qc,detect

    # Pin the worker count (default: whatever the config says, normally
    # "auto" = all cores this process is allowed to use -- see
    # openmeq.parallel; do not hardcode a core count here, GitHub-hosted
    # runners have far fewer cores than a dev workstation):
    python scripts/benchmark.py --n-workers 2

Why this script exists (M11.3)
-------------------------------
This is deliberately a thin, self-contained harness, not a second pipeline
runner: every pipeline-step stage below calls the *exact same*
``<module>.run(cfg, paths, **opts)`` entry point ``openmeq.cli`` calls (via
``openmeq.cli._STEP_MODULES``, imported directly rather than duplicated
here), so a benchmark number reflects what a user actually runs. M0-M10 land
steps one at a time; a step with no ``run`` function yet is reported
``"skipped"`` rather than making the script fail, so this harness keeps
working, unmodified, as the pipeline grows. M11.3 (performance profiling and
optimisation) extends the ``STAGES`` registry below once every step exists,
and can plot trends across the JSON files this script writes over time
(``git_commit`` + ``timestamp`` make that ordering possible).

The ``STAGES`` registry: this script's one-line extension point
------------------------------------------------------------------
``STAGES`` (near the bottom of this module) is an ordered
``list[tuple[str, Callable[[BenchmarkContext], str]]]`` -- a stage name and
a callable that does the work, given the shared :class:`BenchmarkContext`,
and returns a short human-readable note (or raises :class:`StageSkipped` if
there's nothing to time yet). Nothing about *how* stages are timed,
recorded or reported needs to change to add a new one: append one
``(name, callable)`` tuple. This is what lets M1-M10 "automatically" start
being timed the moment their step module gains a ``run`` function -- the
``qc``/``detect``/.../``relocate`` entries already exist below and do
nothing but check for that function today.

Report schema (JSON)
---------------------
See the module-level ``SCHEMA_VERSION`` and :func:`_build_report`'s
docstring for the authoritative field list. In short: ``schema_version``,
``generated_at``, ``git_commit``, ``python_version``, ``platform``,
``argv``, ``n_workers_requested``/``n_workers_resolved``/
``threads_per_worker``, ``dataset`` (source/seed/counts), and ``stages``
(one entry per stage: ``name``, ``status`` -- ``"ran"``/``"skipped"``/
``"error"`` --, ``duration_s``, ``peak_rss_kb``, ``peak_rss_kb_delta``,
``note``).

Peak memory units and caveats
------------------------------
``peak_rss_kb`` comes from ``resource.getrusage(RUSAGE_SELF).ru_maxrss``,
which on **Linux** (this project's only supported platform, per
ARCHITECTURE.md §1) is already in **kilobytes** (it is bytes on macOS --
irrelevant here, but a classic gotcha if this script is ever ported).
``ru_maxrss`` is a *whole-process, monotonically non-decreasing* high-water
mark, not a per-stage measurement: ``peak_rss_kb_delta`` (this stage's value
minus the previous stage's) is only an approximation of that stage's own
contribution, and is ``0`` whenever a stage's transient memory use never
exceeds an earlier stage's peak (even if that earlier peak had since been
freed). If ``psutil`` is installed, a more accurate *sampled* peak RSS is
additionally recorded per stage (a background thread polls
``psutil.Process().memory_info().rss`` while the stage runs); ``psutil`` is
optional (not a project dependency) and this script runs fine without it.

Results are not committed
--------------------------
``benchmarks/results/`` is git-ignored (machine- and run-dependent, not
meaningful to diff) -- only this script and ``benchmarks/README.md`` (the
documented schema) are committed. See that file for a standalone schema
reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# tests/data/make_synthetic.py is a standalone module under tests/data/, not
# part of the installed openmeq package (see tests/conftest.py, which does
# the same sys.path trick for the same reason) -- this script calls its
# generate() function directly rather than going through the pytest
# `synthetic_project` fixture, per this task's design note (no dependency on
# pytest fixture machinery from a standalone script).
sys.path.insert(0, str(REPO_ROOT / "tests" / "data"))
import make_synthetic  # noqa: E402

from openmeq.cli import _STEP_MODULES  # noqa: E402
from openmeq.config import Config, ConfigError, load_config  # noqa: E402
from openmeq.log import configure_logging  # noqa: E402
from openmeq.parallel import _resolve_n_workers  # noqa: E402
from openmeq.paths import RunPaths  # noqa: E402

SCHEMA_VERSION = 1

#: Extra keyword arguments each CLI step's ``run(cfg, paths, **opts)`` takes
#: beyond ``cfg``/``paths`` (mirrors the options ``openmeq.cli``'s
#: subcommands pass through -- see that module's ``review``/``magnitude``
#: commands). A step absent here gets no extra kwargs.
_EXTRA_STEP_OPTS: dict[str, dict[str, Any]] = {
    "review": {"compile": False},
    "magnitude": {"catalog": None},
}

try:
    import psutil
except ImportError:  # optional; see module docstring
    psutil = None


class StageSkipped(Exception):
    """Raised by a stage callable to report "nothing to time yet" (not an error)."""


@dataclass
class BenchmarkContext:
    """State threaded through and mutated by successive stage callables.

    A stage only needs the attributes earlier stages have already filled
    in; e.g. ``qc``'s stage callable reads ``cfg``/``paths`` (set by the
    ``load_config`` stage) and does not care whether ``config_path`` came
    from ``--config`` or from ``generate_synthetic_dataset``.
    """

    config_path: Path | None
    seed: int
    n_workers_override: int | str | None
    cfg: Config | None = None
    paths: RunPaths | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    name: str
    status: str  # "ran" | "skipped" | "error"
    duration_s: float | None
    peak_rss_kb: float | None
    peak_rss_kb_delta: float | None
    peak_rss_bytes_sampled: int | None
    note: str | None


def _current_ru_maxrss_kb() -> float:
    """Current whole-process ``ru_maxrss`` high-water mark, in KB (Linux)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class _RssSampler:
    """Background-thread peak-RSS sampler used when ``psutil`` is available.

    ``ru_maxrss`` alone can miss a stage's true peak if a *later* stage
    never exceeds it (see module docstring); polling the live RSS while a
    stage runs gives a tighter, stage-local peak. Sampling (not tracing)
    keeps overhead negligible, which matters since this literally measures
    the stage it's watching.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self._interval_s = interval_s
        self._process = psutil.Process() if psutil is not None else None
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _RssSampler:
        if self._process is None:
            return self
        self._peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
            except Exception:  # pragma: no cover - process introspection is best-effort
                break
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self._interval_s)

    def __exit__(self, *exc_info: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)

    @property
    def peak_bytes(self) -> int | None:
        return self._peak_bytes if self._process is not None else None


def _run_stage(
    name: str, fn: Callable[[BenchmarkContext], str | None], ctx: BenchmarkContext
) -> StageResult:
    """Time one stage, catching :class:`StageSkipped` and any other exception.

    A stage that raises anything other than :class:`StageSkipped` is
    recorded with ``status="error"`` (with its exception message as the
    note) rather than aborting the whole run, so one broken/half-built step
    doesn't prevent the report from covering every other stage -- important
    since this script is run *while* the pipeline is still being built.
    """
    mem_before = _current_ru_maxrss_kb()
    t0 = time.perf_counter()
    try:
        with _RssSampler() as sampler:
            note = fn(ctx)
        duration_s = time.perf_counter() - t0
    except StageSkipped as exc:
        return StageResult(name, "skipped", None, None, None, None, str(exc) or None)
    except Exception as exc:
        duration_s = time.perf_counter() - t0
        mem_after = _current_ru_maxrss_kb()
        delta = max(mem_after - mem_before, 0.0)
        return StageResult(
            name,
            "error",
            round(duration_s, 6),
            mem_after,
            delta,
            sampler.peak_bytes,
            f"{type(exc).__name__}: {exc}",
        )
    mem_after = _current_ru_maxrss_kb()
    delta = max(mem_after - mem_before, 0.0)
    return StageResult(
        name, "ran", round(duration_s, 6), mem_after, delta, sampler.peak_bytes, note
    )


# ---------------------------------------------------------------------------
# Stage callables
# ---------------------------------------------------------------------------


def _stage_generate_dataset(ctx: BenchmarkContext) -> str:
    """Build a fresh synthetic project via ``make_synthetic.generate`` directly.

    Skipped when ``--config`` was given: the caller already has a project
    to benchmark against, and generation (~500 MB, ~8 s) is exactly the
    thing they're opting out of.
    """
    if ctx.config_path is not None:
        raise StageSkipped(f"dataset provided via --config ({ctx.config_path}); not generated")

    scratch_dir = Path(tempfile.mkdtemp(prefix="openmeq_benchmark_"))
    truth = make_synthetic.generate(scratch_dir, seed=ctx.seed)
    ctx.config_path = scratch_dir / "config.yaml"
    ctx.dataset_info = {
        "source": "generated",
        "scratch_dir": str(scratch_dir),
        "seed": truth["seed"],
        "n_events": len(truth["events"]),
        "n_stations": len(truth["stations"]),
    }
    return (
        f"generated {len(truth['events'])} events, {len(truth['stations'])} stations "
        f"(seed={truth['seed']}) at {scratch_dir}"
    )


def _stage_load_config(ctx: BenchmarkContext) -> str:
    """Load and validate the config, build+ensure ``RunPaths``.

    Applies ``--n-workers`` (if given) to ``cfg.parallel.n_workers`` right
    here, before any pipeline-step stage reads it.
    """
    if ctx.config_path is None:  # pragma: no cover - defensive; CLI always sets this
        raise StageSkipped("no config path available (dataset generation was skipped or failed)")
    cfg = load_config(ctx.config_path)
    if ctx.n_workers_override is not None:
        cfg.parallel.n_workers = ctx.n_workers_override
    ctx.cfg = cfg
    ctx.paths = RunPaths(cfg)
    ctx.paths.ensure()
    if not ctx.dataset_info:
        ctx.dataset_info = {"source": "config", "config_path": str(ctx.config_path)}
    return f"loaded {ctx.config_path}"


def _make_cli_step_stage(step_name: str) -> Callable[[BenchmarkContext], str]:
    """Build a stage callable for one ``openmeq.cli`` pipeline step.

    Mirrors ``openmeq.cli._dispatch`` exactly (config logging, then lazily
    import the step's module and call its ``run(cfg, paths, **opts)`` if
    present, always forwarding ``config_path`` -- ARCHITECTURE.md §2: "the
    dispatcher passes common options such as `config_path` to every step")
    -- reusing ``openmeq.cli._STEP_MODULES`` as the single source of truth
    for module names / plan keys, rather than a second, divergent copy of
    that mapping (see the "pitfalls" design note this task shipped with).
    """
    module_name, plan_key = _STEP_MODULES[step_name]

    def _stage(ctx: BenchmarkContext) -> str:
        if ctx.cfg is None or ctx.paths is None:
            raise StageSkipped("config not loaded (load_config stage was skipped or failed)")
        import importlib

        module = importlib.import_module(module_name)
        run_func = getattr(module, "run", None)
        if run_func is None:
            raise StageSkipped(f"not implemented yet (see {plan_key})")
        configure_logging(ctx.paths, step_name)
        opts = _EXTRA_STEP_OPTS.get(step_name, {})
        run_func(ctx.cfg, ctx.paths, config_path=str(ctx.config_path), **opts)
        return f"{step_name}.run(cfg, paths) completed"

    _stage.__name__ = f"stage_{step_name}"
    return _stage


#: The registry. See the module docstring's "one-line extension point"
#: section: appending a new ``(name, callable)`` pair is the only change
#: needed to time a new stage. The ``qc``..``relocate`` entries are in
#: ``openmeq.cli``'s own step order and start actually running (instead of
#: reporting "skipped") the moment their module defines ``run``.
STAGES: list[tuple[str, Callable[[BenchmarkContext], str | None]]] = [
    ("generate_synthetic_dataset", _stage_generate_dataset),
    ("load_config", _stage_load_config),
    *[(name, _make_cli_step_stage(name)) for name in _STEP_MODULES],
]

#: Stage names that are always run regardless of ``--steps`` (they set up
#: ``ctx.cfg``/``ctx.paths`` that every pipeline-step stage depends on).
_ALWAYS_RUN_STAGES = {"generate_synthetic_dataset", "load_config"}

#: The subset of STAGES names ``--steps`` may filter (i.e. everything else).
_FILTERABLE_STAGE_NAMES = [name for name, _ in STAGES if name not in _ALWAYS_RUN_STAGES]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _build_report(
    stage_results: list[StageResult],
    ctx: BenchmarkContext,
    n_workers_requested: int | str,
    argv: list[str],
) -> dict[str, Any]:
    """Assemble the JSON-serializable report dict.

    Schema (``schema_version=1``)
    ------------------------------
    - ``schema_version`` : int
    - ``generated_at`` : ISO-8601 UTC timestamp string
    - ``git_commit`` : str or null (working tree's ``HEAD``, or null if git
      is unavailable)
    - ``python_version`` / ``platform`` : ``sys.version`` / ``platform.platform()``
    - ``argv`` : the CLI arguments this run was invoked with
    - ``n_workers_requested`` : ``"auto"`` or int (``--n-workers``, or the
      config's own ``parallel.n_workers`` if not overridden)
    - ``n_workers_resolved`` : int, the actual worker count ``"auto"``
      resolved to on this machine (``openmeq.parallel``'s own resolution,
      reused here rather than duplicated)
    - ``dataset`` : ``{"source": "generated"|"config", "seed"?, "n_events"?,
      "n_stations"?, "scratch_dir"?, "config_path"?}``
    - ``stages`` : list of ``{"name", "status", "duration_s",
      "peak_rss_kb", "peak_rss_kb_delta", "peak_rss_bytes_sampled", "note"}``
      -- see the module docstring for units/caveats.
    """
    resolved = None
    if ctx.cfg is not None:
        resolved = _resolve_n_workers(ctx.cfg.parallel.n_workers)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "argv": argv,
        "n_workers_requested": n_workers_requested,
        "n_workers_resolved": resolved,
        "threads_per_worker": ctx.cfg.parallel.threads_per_worker if ctx.cfg else None,
        "dataset": ctx.dataset_info,
        "stages": [asdict(r) for r in stage_results],
    }


def _write_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OpenMEQ benchmark report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- git_commit: {report['git_commit']}",
        f"- python_version: {report['python_version'].splitlines()[0]}",
        f"- platform: {report['platform']}",
        f"- n_workers: requested={report['n_workers_requested']} "
        f"resolved={report['n_workers_resolved']}",
        f"- dataset: {json.dumps(report['dataset'])}",
        "",
        "| stage | status | duration_s | peak_rss_kb | note |",
        "|---|---|---|---|---|",
    ]
    for stage in report["stages"]:
        duration = "-" if stage["duration_s"] is None else f"{stage['duration_s']:.3f}"
        peak = "-" if stage["peak_rss_kb"] is None else f"{stage['peak_rss_kb']:.0f}"
        note = stage["note"] or ""
        lines.append(f"| {stage['name']} | {stage['status']} | {duration} | {peak} | {note} |")
    lines.append("")
    return "\n".join(lines)


def _parse_n_workers(value: str) -> int | str:
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be 'auto' or an integer, got {value!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an existing project's config.yaml to benchmark against. "
        "Default: generate a fresh synthetic dataset (tests/data/make_synthetic.py) "
        "into a temporary directory and time that generation as a stage.",
    )
    parser.add_argument(
        "--n-workers",
        type=_parse_n_workers,
        default=None,
        metavar="{auto,N}",
        help="Override cfg.parallel.n_workers for every pipeline-step stage. "
        "Default: whatever the config says (normally 'auto', which resolves to "
        "every core this process may use -- do not hardcode a specific machine's "
        "core count).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        metavar="STEP[,STEP...]",
        help="Comma-separated subset of pipeline steps to time: "
        f"{{{','.join(_FILTERABLE_STAGE_NAMES)}}}. Default: all of them. "
        "'generate_synthetic_dataset' and 'load_config' always run (they set up "
        "the config/paths every pipeline step needs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=make_synthetic.DEFAULT_SEED,
        help="Seed for a freshly generated dataset (ignored with --config). "
        f"Default: {make_synthetic.DEFAULT_SEED} (make_synthetic.DEFAULT_SEED).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="Directory to write the timestamped JSON+Markdown report into. "
        "Default: benchmarks/results/ (git-ignored -- see benchmarks/README.md).",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Don't delete a freshly generated dataset's scratch directory afterwards "
        "(~500 MB; useful for inspecting output). Has no effect with --config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.steps is not None:
        requested = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = sorted(set(requested) - set(_FILTERABLE_STAGE_NAMES))
        if unknown:
            parser.error(f"unknown step(s) {unknown}; choose from {_FILTERABLE_STAGE_NAMES}")
        selected = set(requested)
    else:
        selected = set(_FILTERABLE_STAGE_NAMES)

    config_path = args.config.resolve() if args.config is not None else None
    if config_path is not None and not config_path.is_file():
        parser.error(f"--config: no such file: {config_path}")

    ctx = BenchmarkContext(
        config_path=config_path,
        seed=args.seed,
        n_workers_override=args.n_workers,
    )

    stage_results: list[StageResult] = []
    for name, fn in STAGES:
        if name not in _ALWAYS_RUN_STAGES and name not in selected:
            continue
        result = _run_stage(name, fn, ctx)
        stage_results.append(result)
        marker = {"ran": "ran   ", "skipped": "skip  ", "error": "ERROR "}[result.status]
        duration = "-" if result.duration_s is None else f"{result.duration_s:8.3f}s"
        print(f"[{marker}] {name:32s} {duration}  {result.note or ''}")

    n_workers_requested = (
        args.n_workers
        if args.n_workers is not None
        else (ctx.cfg.parallel.n_workers if ctx.cfg is not None else "auto")
    )
    report = _build_report(stage_results, ctx, n_workers_requested, argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = args.output_dir / f"{timestamp}.json"
    md_path = args.output_dir / f"{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_write_markdown_report(report))
    print(f"\nWrote {json_path}\nWrote {md_path}")

    scratch_dir = ctx.dataset_info.get("scratch_dir")
    if scratch_dir is not None and not args.keep_scratch:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return 1 if any(r.status == "error" for r in stage_results) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:  # pragma: no cover - user-facing config errors
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
