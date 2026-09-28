"""Batch review loop: resume, skip and reject (M7.3 / OpenMEQ-c3l.3).

:func:`run_review` is this module's one public entry point, backing
``openmeq review`` (``openmeq/cli.py``). It is the actual manual-review
batch loop the M7 epic exists for: launch ObsPyck once per not-yet-reviewed
accepted event, wait for it, detect whether it saved, and file the result --
resumable (a killed/interrupted session picks up where it left off), skippable
and rejectable (see ``docs/ARCHITECTURE.md`` §3a for ``review/state.json``'s
exact semantics).

Everything empirical this module relies on -- the in-process launch
requirement, the save-detection mechanism, the observed exit codes, the
headless/Xvfb finding -- comes from the OpenMEQ-c3l.1 spike
(``docs/obspyck_notes.md``); this module's own job is to wire that up into
a resumable loop, not re-derive any of it.

Launching ObsPyck
------------------
Every event is launched as its own subprocess, in its own
``paths.review_work_dir(event_id)`` cwd::

    python -m openmeq.review._launch_obspyck \\
        -c <obspyck_cfg> -t <chunk_start> -d <chunk_duration> \\
        --event <event>.xml <chunk>.mseed <stations>.xml

**Not** the pip-installed ``obspyck`` console script -- the in-process
launcher module (:mod:`openmeq.review._launch_obspyck`) is required so the
two ObsPyck 0.8.2 bugfix monkeypatches
(:mod:`openmeq.review._obspyck_compat`) actually take effect (a patch
applied in this parent process has no effect on a separate ``obspyck``
child interpreter that imports the unpatched module fresh). The OpenMEQ run
config travels to the launcher via the ``OPENMEQ_CONFIG`` environment
variable (see that module's docstring for why not argv).

``-t``/``-d`` are read directly off the actual raw chunk file's own header
(``obspy.read(..., headonly=True)``), not derived from
``cfg.chunks.pre_s``/``post_s``: this is the exact window the chunk step
(M3) actually wrote to disk, so it can never be shorter than what
``--event``'s picks reference or what ObsPyck's own local-file ingestion
(``-d`` shorter than the chunk truncates it, ``docs/obspyck_notes.md``
pitfall in ``openmeq.review.obspyck_config``'s own module docstring) would
otherwise silently cut off, and it stays correct even if M3's own
trigger-time-vs-origin-time windowing convention ever changes.

Headless operation
-------------------
A real manual review session needs the reviewer to actually *see* the
window, so this module never forces ``QT_QPA_PLATFORM=offscreen`` when a
real ``$DISPLAY`` is already attached (:func:`_resolve_launch_wrapper`).
Only when there is genuinely no display does it choose a headless launch
strategy -- preferring whatever the caller already arranged
(``QT_QPA_PLATFORM=offscreen``, e.g. set by a test, per
``docs/obspyck_notes.md`` §1's finding that this is bit-identical to and
needs no extra system package compared to ``xvfb-run``) over spawning
``xvfb-run`` itself, and only falling back to ``xvfb-run`` -- the M7 epic
design's own documented headless fallback, also confirmed working in the
spike -- when neither is available. If nothing can render at all, the whole
``openmeq review`` run fails immediately with a clear error rather than
hanging on a GUI that can never appear (the design's own explicit
requirement).

Detecting the save (never trust the exit code alone)
-------------------------------------------------------
``docs/obspyck_notes.md`` §9 empirically confirmed the M7 epic design's
prediction: exit code ``0`` means *either* "closed after a save" *or*
"closed with no save at all" -- genuinely indistinguishable by exit code.
So the only way to tell is to snapshot ``work_dir.glob("obspyck_*.xml")``
before launching and diff it against the same glob after ``proc.wait()``
returns (:data:`_SAVE_GLOB`). A *nonzero* exit code, by contrast, **is**
distinguishable (confirmed: it means a startup/config error, raised before
``QApplication`` is even constructed) and is treated as a hard failure --
this aborts the **whole** batch run (:class:`ReviewLaunchError`) rather than
being silently recorded as a review decision, so an event that can never be
reviewed due to bad data is never mistaken for one the reviewer
deliberately skipped (``docs/obspyck_notes.md`` summary item 7).

When a save is detected, the new file is moved into
``paths.review_event_file(event_id)`` using **this loop's own** ``event_id``
for both the filename and the re-stamped ``Event.resource_id`` --
``ObsPyck.setXMLEventID()`` (called by the real "save Event locally" button,
confirmed directly against the installed ``obspyck`` source) unconditionally
overwrites ``Event.resource_id`` with its own save-timestamp-based id, which
would otherwise violate ``docs/ARCHITECTURE.md`` §4.4's invariant that an
event's ``resource_id`` "is created in detection and never changes
afterwards" for every event flowing through ``review/events/*.xml`` into
M7.4's compile step.

Interactive prompt / non-interactive test mode
------------------------------------------------
When no save is detected, the reviewer is prompted at the terminal
(``input_func``, defaulting to the builtin ``input`` -- injectable so tests
can drive the loop without real stdin, per this issue's own requirement).
``non_interactive=True`` (used only by automated tests, never by the real
CLI) skips the prompt and applies the documented default, "skip", instead --
see the module-level ``_prompt_choice`` docstring for the exact prompt text
and accepted answers.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from obspy import Catalog
from obspy import read as obspy_read

from openmeq.catalog import accepted, make_event, read_catalog, write_catalog
from openmeq.catalog import event_id as event_id_of
from openmeq.log import get_logger
from openmeq.review._launch_obspyck import OPENMEQ_CONFIG_ENV
from openmeq.review.obspyck_config import generate_obspyck_config

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("review")

#: The launcher module this loop always shells out to (never the pip-installed
#: ``obspyck`` console script -- see the module docstring's "Launching
#: ObsPyck" section).
_LAUNCHER_MODULE = "openmeq.review._launch_obspyck"

#: The glob :func:`run_review` diffs before/after every launch to detect a
#: save (``ObsPyck.save_event_locally``'s observed filename pattern,
#: ``docs/obspyck_notes.md`` §8: ``"obspyck_" + <timestamp> + ".xml"``).
_SAVE_GLOB = "obspyck_*.xml"


class ReviewLaunchError(RuntimeError):
    """A hard failure launching or preparing ObsPyck for one event.

    Raised for anything that means the batch run cannot safely continue --
    a nonzero ObsPyck exit code (a startup/config error, distinguishable
    from a normal close per ``docs/obspyck_notes.md`` §9), a missing input
    file, or no way to render the GUI at all -- **never** for a normal
    save/skip/reject outcome. Propagating this aborts the whole
    :func:`run_review` call; ``state.json`` already reflects every event
    completed before the failure (written after each one, see the module
    docstring), so a rerun resumes correctly.
    """


@dataclasses.dataclass
class BatchReviewResult:
    """Summary of one :func:`run_review` call, for the CLI/tests to report.

    Attributes
    ----------
    processed : int
        Number of not-yet-reviewed events actually launched this run.
    done, skipped, rejected : int
        How each of those was resolved.
    quit : bool
        Whether the reviewer chose ``[q]uit`` (stopping before every
        remaining event was offered -- not an error, just an early, clean
        stop; ``state.json`` up to that point is already persisted).
    """

    processed: int = 0
    done: int = 0
    skipped: int = 0
    rejected: int = 0
    quit: bool = False


def _load_state(paths: RunPaths) -> dict[str, str]:
    path = paths.review_state_json
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_state(paths: RunPaths, state: dict[str, str]) -> None:
    """Persist ``state.json`` atomically (write-temp-then-replace), matching
    ``openmeq.manifest.StepManifest``'s own convention, so a process killed
    mid-write never corrupts the file."""
    path = paths.review_state_json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _apply_reset(state: dict[str, str], reset: Iterable[str], reset_all: bool) -> dict[str, str]:
    """Return a new state dict with the requested entries cleared.

    ``reset_all=True`` clears everything (``reset`` is then ignored);
    otherwise each event id in ``reset`` is dropped if present (silently
    ignored if it has no entry -- clearing something already clear is not
    an error).
    """
    if reset_all:
        return {}
    state = dict(state)
    for eid in reset:
        state.pop(eid, None)
    return state


def _resolve_launch_wrapper(env: dict[str, str]) -> tuple[list[str], str]:
    """Decide how (if at all) to wrap the launch command so ObsPyck can
    render, without ever hiding a real display from a real reviewer.

    See the module docstring's "Headless operation" section. Returns
    ``(wrapper_argv, launch_mode)``; ``wrapper_argv`` is prepended to the
    launch command (empty when no wrapping is needed), and ``launch_mode``
    is logged so a review session's log always says which path was used.

    An **explicitly set** ``QT_QPA_PLATFORM`` always wins, even over a real
    ``$DISPLAY`` -- this dev machine (and many CI runners) has a desktop
    display attached (worker environment note), but a test must never open
    a visible window (this project's own hard rule); a test sets
    ``QT_QPA_PLATFORM=offscreen`` explicitly for exactly this reason, and
    that must be honoured regardless of ``$DISPLAY``. Only when
    ``QT_QPA_PLATFORM`` is unset at all does ``$DISPLAY`` decide -- so a
    real, unmodified reviewer environment (which never sets
    ``QT_QPA_PLATFORM`` itself) always gets the real, visible GUI.
    """
    if env.get("QT_QPA_PLATFORM"):
        return [], f"env QT_QPA_PLATFORM={env['QT_QPA_PLATFORM']}"
    if env.get("DISPLAY"):
        return [], "display"
    xvfb = shutil.which("xvfb-run")
    if xvfb:
        return [xvfb, "-a"], "xvfb-run"
    raise ReviewLaunchError(
        "No $DISPLAY is set, QT_QPA_PLATFORM is not set, and 'xvfb-run' is not "
        "on PATH -- ObsPyck cannot render (docs/obspyck_notes.md §1/§7). Set "
        "QT_QPA_PLATFORM=offscreen (tests/CI), install xvfb, or run "
        "openmeq review with a display attached."
    )


def _clean_stale_saves(work_dir: Path) -> None:
    """Remove any ``obspyck_*.xml`` left over from a previous *crashed*
    attempt at this same event, so the before/after diff isn't polluted by
    a leftover file (the design's own documented pitfall). A fresh work dir
    has nothing to clean; this is purely defensive."""
    stale = sorted(work_dir.glob(_SAVE_GLOB))
    if not stale:
        return
    logger.warning(
        "review/work/%s: removing %d stale obspyck_*.xml file(s) left over "
        "from a previous crashed attempt at this event",
        work_dir.name,
        len(stale),
    )
    for f in stale:
        f.unlink()


def _chunk_time_bounds(chunk_mseed: Path) -> tuple:
    """Read the raw chunk's actual start time and duration from its own
    header (``headonly=True`` -- no need to decode sample data), rather
    than re-deriving them from ``cfg.chunks.pre_s``/``post_s`` (see the
    module docstring's "Launching ObsPyck" section for why)."""
    st = obspy_read(str(chunk_mseed), headonly=True)
    start = min(tr.stats.starttime for tr in st)
    end = max(tr.stats.endtime for tr in st)
    return start, float(end - start)


def _prompt_choice(event_id: str, input_func: Callable[[str], str]) -> str:
    """Prompt the reviewer at the terminal when no save was detected.

    Returns ``"s"`` (skip), ``"r"`` (reject) or ``"q"`` (quit). Accepts
    either the single letter or the full word (case-insensitive), and
    re-prompts on anything else -- ``input_func`` is injectable
    specifically so tests can drive this without real stdin (e.g. an
    iterator or a canned ``list.pop(0)``-style stand-in), per this issue's
    own requirement.
    """
    prompt = f"Event {event_id}: no save detected. [s]kip / [r]eject / [q]uit batch? "
    while True:
        raw = input_func(prompt).strip().lower()
        if raw in ("s", "skip"):
            return "s"
        if raw in ("r", "reject"):
            return "r"
        if raw in ("q", "quit"):
            return "q"
        print(f"unrecognized choice {raw!r}; enter s, r, or q")


def _launch_obspyck_subprocess(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    """Thin, dedicated seam around ``subprocess.Popen`` for launching ObsPyck.

    A test mocking process launches must not simply monkeypatch the global
    ``subprocess.Popen`` -- that also intercepts unrelated internal
    ``subprocess.Popen`` calls made deep inside this same call stack (e.g.
    ``platform.architecture()``, called from
    ``openmeq.nlloc_setup.link_obspyck_model`` while generating the ObsPyck
    pluginpath, itself shells out via ``subprocess.check_output``) with an
    incompatible fake, breaking code that has nothing to do with ObsPyck.
    Tests instead monkeypatch this one dedicated function
    (``openmeq.review.batch._launch_obspyck_subprocess``), which is called
    from exactly one place (:func:`run_review`) and nowhere else in this
    process.
    """
    return subprocess.Popen(cmd, cwd=str(cwd), env=env)


def run_review(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | Path,
    reset: Iterable[str] = (),
    reset_all: bool = False,
    non_interactive: bool = False,
    input_func: Callable[[str], str] = input,
) -> BatchReviewResult:
    """Run (or resume) one batch review session.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or pathlib.Path, keyword-only
        Path to the OpenMEQ YAML config file that produced ``cfg`` --
        resolved to an absolute path and forwarded to every launched
        subprocess via the ``OPENMEQ_CONFIG`` environment variable (the
        launcher module reloads it there; see
        ``openmeq.review._launch_obspyck``'s docstring for why this can't
        just be part of ``cfg``/``paths`` themselves: the launcher runs in
        a separate process).
    reset : iterable of str, keyword-only, optional
        Event ids whose ``state.json`` entry should be cleared before this
        run starts, so they are offered again (``openmeq review --reset``).
    reset_all : bool, keyword-only, optional
        Clear every ``state.json`` entry before this run starts
        (``openmeq review --reset-all``). Takes priority over ``reset``.
    non_interactive : bool, keyword-only, optional
        Skip the terminal prompt on a no-save outcome and record the
        documented default ("skip") instead. For automated tests only --
        the real CLI never sets this (see the module docstring).
    input_func : callable, keyword-only, optional
        ``input``-compatible callable used for the interactive prompt.
        Defaults to the builtin ``input``; tests inject their own.

    Returns
    -------
    BatchReviewResult

    Raises
    ------
    ReviewLaunchError
        On any hard failure (nonzero ObsPyck exit code, a missing input
        file, or no way to render the GUI at all). ``state.json`` reflects
        every event completed before the failure.
    """
    config_path = Path(config_path).resolve()
    paths.review_events_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(paths)
    if reset_all or list(reset):
        cleared = list(reset)
        state = _apply_reset(state, cleared, reset_all)
        _save_state(paths, state)
        logger.info(
            "review: cleared state.json for %s",
            "ALL events" if reset_all else cleared,
        )

    generate_obspyck_config(cfg, paths)
    obspyck_cfg = Path(cfg.review.obspyck_config)

    catalog = accepted(read_catalog(paths.catalog_auto_xml))
    events = sorted(catalog, key=lambda ev: ev.preferred_origin().time)

    result = BatchReviewResult()

    for event in events:
        eid = event_id_of(event)
        if eid in state:
            continue

        result.processed += 1
        started = time.monotonic()
        logger.info("event %s: starting review", eid)

        origin = event.preferred_origin()
        if origin is None:
            raise ReviewLaunchError(f"event {eid}: has no preferred origin, cannot review")

        chunk_mseed = paths.chunk_file(eid)
        if not chunk_mseed.exists():
            raise ReviewLaunchError(f"event {eid}: raw chunk not found: {chunk_mseed}")
        stations_xml = paths.stations_xml
        if not stations_xml.exists():
            raise ReviewLaunchError(f"event {eid}: merged StationXML not found: {stations_xml}")

        chunk_start, chunk_duration = _chunk_time_bounds(chunk_mseed)

        work_dir = paths.review_work_dir(eid)
        work_dir.mkdir(parents=True, exist_ok=True)
        _clean_stale_saves(work_dir)

        # Single-event QuakeML for ObsPyck's --event argument: never the
        # whole multi-event locate/catalog_auto.xml (docs/obspyck_notes.md
        # pitfall re: setEventFromCatalog picking whichever event it likes).
        event_xml = work_dir / f"{eid}_auto.xml"
        write_catalog(Catalog(events=[event]), event_xml)

        env = os.environ.copy()
        env[OPENMEQ_CONFIG_ENV] = str(config_path)
        wrapper, launch_mode = _resolve_launch_wrapper(env)

        cmd = [
            *wrapper,
            sys.executable,
            "-m",
            _LAUNCHER_MODULE,
            "-c",
            str(obspyck_cfg),
            "-t",
            str(chunk_start),
            "-d",
            str(int(round(chunk_duration))),
            "--event",
            str(event_xml),
            str(chunk_mseed),
            str(stations_xml),
        ]

        before = set(work_dir.glob(_SAVE_GLOB))
        logger.info("event %s: launching ObsPyck (launch mode: %s)", eid, launch_mode)
        proc = _launch_obspyck_subprocess(cmd, work_dir, env)
        returncode = proc.wait()
        elapsed = time.monotonic() - started

        if returncode != 0:
            logger.error(
                "event %s: ObsPyck exited %s after %.1fs -- treated as a startup/config "
                "error (docs/obspyck_notes.md §9), not a review decision",
                eid,
                returncode,
                elapsed,
            )
            raise ReviewLaunchError(
                f"event {eid}: ObsPyck launcher exited with code {returncode} "
                "(startup/config error); aborting the batch review run"
            )

        after = set(work_dir.glob(_SAVE_GLOB))
        new_files = sorted(after - before)

        if new_files:
            if len(new_files) > 1:
                logger.warning(
                    "event %s: %d new save files detected (expected 1); using the earliest: %s",
                    eid,
                    len(new_files),
                    new_files,
                )
            saved_path = new_files[0]
            saved_cat = read_catalog(saved_path)
            saved_event = saved_cat[0]
            # ObsPyck's setXMLEventID() (called by the real "save Event
            # locally" button) overwrites Event.resource_id with its own
            # save-timestamp-based id -- restore OpenMEQ's own convention
            # (ARCHITECTURE.md §4.4: "created in detection and never
            # changes afterwards") before filing it.
            saved_event.resource_id = make_event(eid).resource_id
            write_catalog(Catalog(events=[saved_event]), paths.review_event_file(eid))
            state[eid] = "done"
            result.done += 1
            logger.info("event %s: done (saved, %.1fs)", eid, elapsed)
        else:
            if non_interactive:
                choice = "s"
                logger.info("event %s: no save detected; non-interactive mode -> skip", eid)
            else:
                choice = _prompt_choice(eid, input_func)

            if choice == "q":
                logger.info("event %s: reviewer quit the batch run (%.1fs)", eid, elapsed)
                result.quit = True
                _save_state(paths, state)
                return result
            if choice == "r":
                state[eid] = "rejected"
                result.rejected += 1
                logger.info("event %s: rejected (no save, %.1fs)", eid, elapsed)
            else:
                state[eid] = "skipped"
                result.skipped += 1
                logger.info("event %s: skipped (no save, %.1fs)", eid, elapsed)

        _save_state(paths, state)

    return result
