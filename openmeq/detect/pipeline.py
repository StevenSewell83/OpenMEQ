"""Detection pipeline orchestration (M2.3, ``OpenMEQ-ujx.3``): :func:`run_detect`,
the real implementation behind ``openmeq detect`` (registered as ``run`` in
``openmeq/detect/__init__.py``, ARCHITECTURE.md §2's ``openmeq.cli`` contract).

Mirrors ``openmeq.nlloc_setup.setup_nlloc``'s and ``openmeq.relocate.setup
.setup_hypodd``'s own "require inputs, fingerprint, skip-unchanged, do the
work, mark done, log elapsed time" shape (this task's own design note: "same
shape ... for consistency across steps"), adapted for M2's own parallel unit
of work (one :func:`~openmeq.detect.worker.detect_window` call per
:func:`~openmeq.detect.worker.build_windows` window, via
:func:`openmeq.parallel.pmap`, PLAN.md §4's "Detect: day (all stations)" row)
rather than a single whole-run unit.

Manifest step-folder name (ARCHITECTURE.md §2's own clarifying note)
----------------------------------------------------------------------
ARCHITECTURE.md §2 is explicit that ``RunPaths.diagnostics_dir(step)``/
``manifest_json(step)`` take the step's **output-folder name**, "not the CLI
command name", and calls out this *exact* step as the example: "the
``detect`` subcommand's outputs live under ``detections/``". This task's own
beads notes sketch ``StepManifest(paths, "detect")``, which would resolve to
a nonexistent ``detect/manifest.json`` rather than
``detections/manifest.json`` -- ARCHITECTURE.md wins per the worker
instructions ("ARCHITECTURE.md ... wins over the spec if they conflict"), so
:data:`STEP_NAME` here is ``"detections"``, matching
``paths.detect_diagnostics_dir``'s own ``diagnostics_dir("detections")``.

Extended QAQC (``OpenMEQ-02o.2``)
----------------------------------
When ``cfg.diagnostics.extended_enabled("detect")`` is true, each window
runs :func:`~openmeq.detect.worker.detect_window_with_station_triggers`
instead of :func:`~openmeq.detect.worker.detect_window` (same detections,
plus every single-station trigger), ``detections/station_triggers.csv`` is
written, and :func:`openmeq.detect.extended.make_detect_extended_diagnostics`
builds ``detections/diagnostics/extended/``. When it is false nothing about
the step changes: the plain worker runs, and a ``station_triggers.csv`` left
over from an earlier extended run is deleted on a real (non-skipped) run so
it can never sit next to a ``detections.csv`` it no longer matches.

The fingerprint deliberately does *not* include ``cfg.diagnostics`` --
toggling a diagnostics switch must not invalidate the detections themselves.
Instead the skip-unchanged check additionally requires
``station_triggers.csv`` to exist when extended QAQC is on (so switching it
on for an already-detected run triggers one rerun, the only way to obtain
per-station triggers), and a skipped run with extended QAQC on but no
``extended/`` directory yet rebuilds those figures from the on-disk CSVs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import click
import pandas as pd

from openmeq.detect.diagnostics import make_detect_diagnostics
from openmeq.detect.extended import (
    extended_dir,
    make_detect_extended_diagnostics,
    station_triggers_csv,
    station_triggers_frame,
    write_station_triggers_csv,
)
from openmeq.detect.merge import merge_windows, write_detections_csv
from openmeq.detect.worker import (
    build_windows,
    detect_window,
    detect_window_with_station_triggers,
)
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("detect.pipeline")

#: ``RunPaths.manifest_json``/``diagnostics_dir``'s step-folder-name argument
#: -- the output folder (``detections/``), not the CLI command (``detect``).
#: See the module docstring's "Manifest step-folder name" note.
STEP_NAME = "detections"

#: The single :class:`~openmeq.manifest.StepManifest` entry key for this
#: step -- one whole-run unit of work (unlike e.g. ``chunks_pn``'s
#: per-event keys), mirroring ``openmeq.nlloc_setup.setup_nlloc``'s own
#: single ``"setup"`` key.
MANIFEST_KEY = "detect"


def _require_qc_outputs(paths: RunPaths) -> None:
    """Raise a clear, actionable error if ``openmeq qc``'s outputs are missing.

    ``openmeq detect`` reads ``inventory/index.json`` (to find/read the
    waveform files, :mod:`openmeq.index`) and ``qc/channel_flags.json`` (to
    exclude bad channel/days, ARCHITECTURE.md §4.2) -- both written by
    ``openmeq qc`` (M1.7). A fresh run directory with either missing is the
    expected first-run-out-of-order mistake; this is a deliberate,
    CLI-level hard dependency check (this task's own design note: "not just
    an implicit crash if index.json/channel_flags.json happen to be
    missing"), not an incidental ``FileNotFoundError`` raised deeper inside
    :func:`~openmeq.detect.worker.detect_window`.

    Raises
    ------
    click.ClickException
        Naming every missing path and telling the user to run ``openmeq
        qc`` first. ``click.ClickException`` (rather than a bare
        ``RuntimeError``) is used deliberately so a real CLI invocation
        prints a clean ``"Error: ..."`` message instead of a raw traceback
        (click's own top-level handler catches it), while still being a
        plain ``Exception`` subclass any caller (tests, a future ``openmeq
        run``) can catch directly.
    """
    missing = [p for p in (paths.index_json, paths.channel_flags_json) if not p.is_file()]
    if not missing:
        return
    names = ", ".join(str(p) for p in missing)
    message = (
        f"openmeq detect: missing required input(s): {names}. Run `openmeq qc` first "
        "to build inventory/index.json and qc/channel_flags.json."
    )
    logger.error(message)
    raise click.ClickException(message)


def _detect_window_item(item: tuple[str, str, str, str, str]) -> list[dict]:
    """``openmeq.parallel.pmap`` worker: unpack one window's args.

    :func:`~openmeq.detect.worker.detect_window` takes five separate
    positional arguments, not a single tuple, but ``pmap`` calls
    ``func(item)`` with the whole item -- this thin, top-level, picklable
    wrapper does the unpacking, the same pattern
    :mod:`openmeq.pick.prep`'s own ``_prepare_chunk_worker`` uses for the
    same reason.
    """
    window_start, window_end, index_path, flags_path, cfg_path = item
    return detect_window(window_start, window_end, index_path, flags_path, cfg_path)


def _detect_window_extended_item(
    item: tuple[str, str, str, str, str],
) -> tuple[list[dict], list[dict]]:
    """:func:`_detect_window_item`'s extended-QAQC twin (also returns station triggers)."""
    window_start, window_end, index_path, flags_path, cfg_path = item
    return detect_window_with_station_triggers(
        window_start, window_end, index_path, flags_path, cfg_path
    )


def _rebuild_extended_from_disk(cfg: Config, paths: RunPaths) -> None:
    """Skip-unchanged path: build ``extended/`` from the CSVs already on disk."""
    try:
        detections_df = pd.read_csv(paths.detections_csv)
        station_df = pd.read_csv(station_triggers_csv(paths))
    except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
        get_logger("detect").warning("detect extended QAQC: could not re-read CSVs: %s", exc)
        return
    make_detect_extended_diagnostics(cfg, paths, detections_df, station_df)


def run_detect(
    cfg: Config,
    paths: RunPaths,
    *,
    config_path: str | None = None,
    force: bool = False,
) -> None:
    """Run network coincidence detection end to end (``openmeq detect``).

    1. Requires ``qc/channel_flags.json``/``inventory/index.json`` to
       already exist (:func:`_require_qc_outputs`).
    2. Skips (unless ``force``) if ``cfg.detection`` and both input files'
       mtimes/sizes match the last recorded fingerprint
       (:class:`openmeq.manifest.StepManifest`) and ``detections/
       detections.csv`` still exists.
    3. :func:`~openmeq.detect.worker.build_windows`, then
       :func:`~openmeq.detect.worker.detect_window` per window via
       :func:`openmeq.parallel.pmap` (PLAN.md §4's "Detect: day (all
       stations)" row -- this is where that per-window parallelism is
       actually exercised).
    4. :func:`~openmeq.detect.merge.merge_windows` /
       :func:`~openmeq.detect.merge.write_detections_csv`.
    5. :func:`~openmeq.detect.diagnostics.make_detect_diagnostics`, plus
       (only if ``cfg.diagnostics.extended_enabled("detect")``)
       ``station_triggers.csv`` and the ``extended/`` figures -- see the
       module docstring's "Extended QAQC" note.
    6. Marks the manifest entry done and logs the elapsed time.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    config_path : str or None, keyword-only, optional
        Path to the YAML config file on disk. Each
        :func:`~openmeq.detect.worker.detect_window` call runs in a
        separate worker process (:func:`openmeq.parallel.pmap`) and reloads
        the config from this path itself (a ``Config`` object is not
        picklable/passed directly, per :mod:`openmeq.detect.worker`'s own
        design). Optional here (mirroring :func:`openmeq.review.run`'s own
        ``config_path``, ARCHITECTURE.md §2's ``run(cfg, paths, **opts)``
        contract -- a caller that never reaches the point where it's
        actually needed, e.g. ``scripts/benchmark.py``, which calls every
        step's ``run(cfg, paths, **opts)`` without it, doesn't have to
        supply it) -- a clear ``ValueError`` is raised below only once
        detection is actually about to run (never on a skip-unchanged
        return, which needs no config path at all).
    force : bool, keyword-only, optional
        Rerun even if the manifest says nothing changed.
    """
    t0 = time.perf_counter()
    paths.ensure()
    step_logger = get_logger("detect")

    _require_qc_outputs(paths)

    fp = fingerprint(cfg.detection, paths.channel_flags_json, paths.index_json)
    manifest = StepManifest(paths, STEP_NAME)
    extended = cfg.diagnostics.extended_enabled("detect")
    station_csv = station_triggers_csv(paths)
    if (
        not force
        and manifest.is_done(MANIFEST_KEY, fp)
        and paths.detections_csv.is_file()
        and (not extended or station_csv.is_file())
    ):
        step_logger.info(
            "detect: unchanged since the last run (cfg.detection, qc/channel_flags.json and "
            "inventory/index.json all match the last recorded fingerprint) -- skipping "
            "(use --force to rerun)."
        )
        if extended and not extended_dir(paths).is_dir():
            _rebuild_extended_from_disk(cfg, paths)
        return

    if config_path is None:
        raise ValueError(
            "openmeq.detect.pipeline.run_detect(): config_path is required to actually run "
            "detection -- each openmeq.parallel.pmap worker process reloads the OpenMEQ config "
            "from this path (a Config object itself is not passed to worker processes). Only "
            "omit it when this step is unchanged (skipped above)."
        )

    windows = build_windows(cfg)
    items = [
        (
            window_start,
            window_end,
            str(paths.index_json),
            str(paths.channel_flags_json),
            str(config_path),
        )
        for window_start, window_end in windows
    ]
    if extended:
        results = pmap(_detect_window_extended_item, items, cfg.parallel, desc="detect")
        per_window_results = [triggers for triggers, _ in results]
        station_df = station_triggers_frame([station for _, station in results])
    else:
        per_window_results = pmap(_detect_window_item, items, cfg.parallel, desc="detect")
        station_df = None

    df = merge_windows(per_window_results, cfg)
    write_detections_csv(df, paths.detections_csv)

    make_detect_diagnostics(paths, df)

    if station_df is not None:
        write_station_triggers_csv(station_df, station_csv)
        make_detect_extended_diagnostics(cfg, paths, df, station_df)
    elif station_csv.is_file():
        # Stale from an earlier extended run -- see the module docstring.
        station_csv.unlink()

    manifest.mark_done(MANIFEST_KEY, fp)

    elapsed = time.perf_counter() - t0
    n_raw = sum(len(w) for w in per_window_results)
    step_logger.info(
        "detect: %d window(s), %d raw trigger(s), %d detection(s) after de-dup (%.2fs).",
        len(windows),
        n_raw,
        len(df),
        elapsed,
    )
