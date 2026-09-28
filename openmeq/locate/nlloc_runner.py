"""Per-event NLLoc runner (M6.1 / OpenMEQ-xdf.1).

:func:`locate_event` is the single reusable primitive M6.2 (pass A) and
M6.3 (pass B) call repeatedly, once per event per iteration: write an
``NLLOC_OBS`` phase file for an explicit, already-filtered list of `Pick`
objects, run ``NLLoc`` against the shared grids :func:`openmeq.nlloc_setup.
setup_nlloc` already built, and parse the resulting ``.hyp`` file back into
an `obspy.core.event.Origin` whose arrivals resolve to the *same* `Pick`
objects passed in.

This module never decides which picks to use -- ``picks_to_use`` is built
by the caller (M6.2/M6.3: "accepted, correct phase type for this pass, this
iteration") and passed in verbatim. It also never re-derives NLLoc's shared
``CONTROL``/``TRANS`` boilerplate or the ``LOCGRID`` bounds:
:func:`openmeq.nlloc_setup.write_control_common` and
:func:`openmeq.nlloc_setup.location_grid_bounds` are the single sources of
truth for those (ARCHITECTURE.md §6d), used here exactly as M5.2 built
them.

Every claim about obspy's ``NLLOC_OBS``/``NLLOC_HYP`` I/O and NLLoc's own
control-file/stdout/output-filename behaviour below is confirmed in
docs/nlloc_interface.md (the M5.1 spike) against the installed obspy
version and the installed NonLinLoc v7.1.05 build; re-derive nothing here
that spike already verified.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from obspy import Catalog, read_events
from obspy.core.event import Event, Pick
from obspy.geodetics import degrees2kilometers

from openmeq.catalog import event_id as get_event_id
from openmeq.config import ConfigError
from openmeq.external import ExternalToolError, bin_path, run
from openmeq.nlloc_setup import (
    LocationGridBounds,
    location_grid_bounds,
    read_grid3d_hdr,
    write_control_common,
)

if TYPE_CHECKING:
    from obspy.core.event import Origin

    from openmeq.config import Config
    from openmeq.paths import RunPaths

#: NLLoc's own "location succeeded" stdout line (confirmed verbatim against
#: the installed source, ``external/NonLinLoc/src/NLLoc1.c:612``:
#: ``sprintf(MsgStr, "Finished location: %s.grid0.loc.hyp", fn_root_out);``,
#: printed via ``nll_putmsg`` -- ``util.c:117-121`` -- which writes exactly
#: ``"%s\n"``, no extra prefix). ``fn_root_out`` already carries the
#: ``.<YYYYMMDD>.<HHMMSS>`` suffix NLLoc derives from the earliest
#: observation time (docs/nlloc_interface.md §4's ``LOCHYPOUT`` note), so
#: this is the single most reliable way to find the exact ``.hyp`` file
#: this specific run wrote, without guessing the suffix.
_FINISHED_LOCATION_RE = re.compile(r"^Finished location: (?P<path>.+)$", re.MULTILINE)

#: Root name (relative to the per-iteration work dir) NLLoc's ``LOCFILES``/
#: ``LOCHYPOUT`` write every output file under (``<out_root>.<suffix>...``).
_OUT_ROOT = "out"

#: Tolerance (km) for the LOCGRID/travel-time-grid z-coverage check below --
#: purely to absorb floating point round-off in the bounds/grid arithmetic,
#: not a real margin (docs/nlloc_interface.md's LOCGRID note: NLLoc's own
#: coverage check has no tolerance, so neither does this one).
_COVERAGE_EPS_KM = 1e-6

#: Number of trailing lines of NLLoc's stdout/stderr kept in
#: :class:`NLLocFailed`'s ``stderr_tail`` -- enough to show the actual
#: error/warning without dumping an entire run's verbose log.
_TAIL_LINES = 20


class NLLocFailed(RuntimeError):
    """Raised when NLLoc did not produce exactly one location for this event.

    Covers both a non-zero NLLoc exit (a genuine subprocess failure) and
    the silent "N events read, 0 events located" case (exit code 0, no
    ``.hyp`` file written -- e.g. too few phases passed ``LOCMETH``'s
    ``minNumberPhases``, or an observation rejected for a grid-coverage
    mismatch, docs/nlloc_interface.md's ``LOCGRID``/``LOCMETH`` notes). The
    caller (M6.2/M6.3) is expected to treat this as "0 accepted picks this
    iteration" for the affected event, not to propagate a location for it.
    """

    def __init__(self, event_id: str, pass_name: str, iteration: int, stderr_tail: str):
        message = (
            f"NLLoc did not produce a location for event {event_id!r} "
            f"({pass_name}, iteration {iteration}): {stderr_tail}"
        )
        super().__init__(message)
        self.event_id = event_id
        self.pass_name = pass_name
        self.iteration = iteration
        self.stderr_tail = stderr_tail


class DuplicatePickError(ValueError):
    """Raised when ``picks_to_use`` has two picks sharing (station_code, phase_hint).

    ``obspy.read_events(..., format="NLLOC_HYP", picks=...)`` re-links each
    parsed arrival to the matching original `Pick` by ``station_code`` +
    ``phase_hint``, **first match wins** (docs/nlloc_interface.md §7) -- a
    duplicate would silently attribute the wrong `Pick`'s residual/weight,
    so this is checked and raised *before* any NLLoc subprocess is launched.
    """


def _tail(text: str | None, n_lines: int = _TAIL_LINES) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:])


def _bool01(value: bool) -> str:
    """NLLoc renders a boolean control-statement field as a literal ``1``/``0``
    (docs/nlloc_interface.md §4, ``LOCSEARCH``/``LOCMETH`` notes), not
    Python's ``True``/``False``."""
    return "1" if value else "0"


def _check_no_duplicate_station_phase(picks_to_use: list[Pick], event_id: str) -> None:
    seen: set[tuple[str, str]] = set()
    dupes: set[tuple[str, str]] = set()
    for pick in picks_to_use:
        key = (pick.waveform_id.station_code, pick.phase_hint)
        if key in seen:
            dupes.add(key)
        seen.add(key)
    if dupes:
        raise DuplicatePickError(
            f"event {event_id!r}: picks_to_use has duplicate (station_code, phase_hint) pairs "
            f"{sorted(dupes)} -- obspy's read_nlloc_hyp picks= re-linking matches "
            "first-match-wins on exactly this pair (docs/nlloc_interface.md §7), so a duplicate "
            "would silently corrupt residual/weight attribution between the two picks."
        )


def _check_uncertainties_set(picks_to_use: list[Pick], event_id: str) -> None:
    missing = [
        f"{p.waveform_id.station_code}.{p.phase_hint}"
        for p in picks_to_use
        if p.time_errors is None or p.time_errors.uncertainty is None
    ]
    if missing:
        raise ValueError(
            f"event {event_id!r}: pick(s) {missing} have no time_errors.uncertainty set. "
            "obspy's NLLOC_OBS writer silently defaults this to 0.0 with a UserWarning "
            "(docs/nlloc_interface.md §7) rather than raising -- upstream pick-producing steps "
            "must always set this (ARCHITECTURE.md §5); treat a missing value as a bug to fix "
            "there, not something to swallow here."
        )


@dataclass(frozen=True)
class _RunnerContext:
    """Per-``(cfg, paths)`` state cheap to reuse across many :func:`locate_event`
    calls (built once, cached by :func:`_get_context` -- see its docstring)."""

    control_common: str
    grid_root: str
    bounds: LocationGridBounds
    #: (station_code, phase) -> (z_min_km, z_max_km) for that station's own
    #: Grid2Time output, read once from its ``.hdr`` file.
    time_grid_z_range: dict[tuple[str, str], tuple[float, float]]


#: Cache keyed by ``str(paths.root)`` (stable across repeated calls within
#: one worker process even when ``cfg``/``paths`` are freshly unpickled per
#: call, e.g. under ``openmeq.parallel.pmap`` with ``n_workers > 1`` --
#: unlike ``id(cfg)``, which changes on every unpickle). Assumes
#: ``cfg.nlloc``/the station table don't change for a given
#: ``cfg.project.output_dir`` within one process's lifetime, which holds
#: for one pipeline run (or one test session's shared grid fixture).
_CONTEXT_CACHE: dict[str, _RunnerContext] = {}


def _get_context(cfg: Config, paths: RunPaths) -> _RunnerContext:
    """Build (once per ``paths.root``, per process) and cache everything
    :func:`locate_event` needs that doesn't vary per event/iteration:
    the shared ``CONTROL``/``TRANS`` boilerplate, the ``LOCGRID`` bounds,
    and every station's own travel-time grid z-range (for the coverage
    check below) -- so a hot loop over many events/iterations (the whole
    point of this function, per the issue's "make it fast") does not
    re-read the station table or every grid ``.hdr`` file on every call.
    """
    cache_key = str(paths.root)
    ctx = _CONTEXT_CACHE.get(cache_key)
    if ctx is not None:
        return ctx

    from openmeq.stations import station_table as load_station_table

    table = load_station_table(cfg.stations)
    bounds = location_grid_bounds(cfg, table)
    grid_root = paths.nlloc_grid_root

    z_ranges: dict[tuple[str, str], tuple[float, float]] = {}
    grouped = table.drop_duplicates(subset=["network", "station"])
    for row in grouped.itertuples(index=False):
        for phase in ("P", "S"):
            hdr_path = grid_root.parent / f"{grid_root.name}.{phase}.{row.station}.time.hdr"
            if not hdr_path.exists():
                continue
            hdr = read_grid3d_hdr(hdr_path)
            z_min = hdr.origz
            z_max = hdr.origz + hdr.dz * (hdr.numz - 1)
            z_ranges[(row.station, phase)] = (z_min, z_max)

    ctx = _RunnerContext(
        control_common=write_control_common(cfg, paths),
        grid_root=str(grid_root),
        bounds=bounds,
        time_grid_z_range=z_ranges,
    )
    _CONTEXT_CACHE[cache_key] = ctx
    return ctx


def _check_locgrid_coverage(
    ctx: _RunnerContext, stations_used: set[str], phases_used: set[str], event_id: str
) -> None:
    """Verify every (station, phase) pair actually used by this event's own
    Grid2Time travel-time grid vertically spans the ``LOCGRID`` z-range
    (docs/nlloc_interface.md's ``LOCGRID`` note): NLLoc silently rejects
    (and locates 0 events, exit code 0, no error) any observation whose
    station's time grid doesn't fully cover the initial location search
    grid's z-range -- checked here up front rather than discovered as a
    mysterious :class:`NLLocFailed`.
    """
    shortfalls = []
    for station in sorted(stations_used):
        for phase in sorted(phases_used):
            z_range = ctx.time_grid_z_range.get((station, phase))
            if z_range is None:
                shortfalls.append(
                    f"{station} {phase}: no travel-time grid found under {ctx.grid_root}"
                )
                continue
            z_min, z_max = z_range
            if z_min > ctx.bounds.z_min_km + _COVERAGE_EPS_KM or (
                z_max < ctx.bounds.z_max_km - _COVERAGE_EPS_KM
            ):
                shortfalls.append(
                    f"{station} {phase}: travel-time grid z-range [{z_min:.3f}, {z_max:.3f}] km "
                    f"does not cover LOCGRID z-range [{ctx.bounds.z_min_km:.3f}, "
                    f"{ctx.bounds.z_max_km:.3f}] km"
                )
    if shortfalls:
        raise ConfigError(
            f"event {event_id!r}: LOCGRID is not fully covered by every used station/phase's own "
            "travel-time grid (docs/nlloc_interface.md's LOCGRID note: NLLoc would otherwise "
            "silently reject these observations and locate 0 events, exit code 0): "
            + "; ".join(shortfalls)
        )


def _render_control(cfg: Config, ctx: _RunnerContext, obs_filename: str, out_root: str) -> str:
    """The full per-event NLLoc control file: shared ``CONTROL``/``TRANS``
    (:func:`openmeq.nlloc_setup.write_control_common`, never re-derived)
    plus ``LOCFILES``/``LOCHYPOUT``/``LOCSEARCH``/``LOCGRID`` (from
    :func:`openmeq.nlloc_setup.location_grid_bounds`, never re-derived)/
    ``LOCMETH``/``LOCGAU``/``LOCGAU2``/``LOCPHASEID``/``LOCQUAL2ERR``.
    """
    bounds = ctx.bounds
    spacing = cfg.nlloc.grid.spacing_km
    x_num = max(2, math.ceil((bounds.x_max_km - bounds.x_min_km) / spacing) + 1)
    y_num = max(2, math.ceil((bounds.y_max_km - bounds.y_min_km) / spacing) + 1)
    z_num = max(2, math.ceil((bounds.z_max_km - bounds.z_min_km) / spacing) + 1)

    ls = cfg.nlloc.locsearch
    lg = cfg.nlloc.locgau
    lg2 = cfg.nlloc.locgau2
    lm = cfg.nlloc.locmeth
    locqual2err = " ".join(str(v) for v in cfg.nlloc.locqual2err_thresholds_s)

    return (
        ctx.control_common + "\n"
        f"LOCFILES  {obs_filename}  NLLOC_OBS  {ctx.grid_root}  {out_root}\n"
        "\n"
        "LOCHYPOUT SAVE_NLLOC_ALL\n"
        "\n"
        f"LOCSEARCH  OCT {ls.init_num_cells_x} {ls.init_num_cells_y} {ls.init_num_cells_z} "
        f"{ls.min_node_size} {ls.max_num_nodes} {ls.num_scatter} "
        f"{_bool01(ls.use_stations_density)} {_bool01(ls.stop_on_min_node_size)}\n"
        "\n"
        f"LOCGRID  {x_num} {y_num} {z_num}  "
        f"{bounds.x_min_km} {bounds.y_min_km} {bounds.z_min_km}  "
        f"{spacing} {spacing} {spacing}  PROB_DENSITY SAVE\n"
        "\n"
        f"LOCMETH {lm.method} {lm.max_dist_sta_grid} {lm.min_number_phases} "
        f"{lm.max_number_phases} {lm.min_number_s_phases} {lm.vp_vs_ratio} "
        f"{lm.max_num_3d_grid_memory} {lm.min_dist_sta_grid} "
        f"{_bool01(lm.reject_duplicate_arrivals)}\n"
        "\n"
        f"LOCGAU {lg.sigma_time_s} {lg.corr_len_km}\n"
        f"LOCGAU2 {lg2.sigma_time_fraction} {lg2.sigma_time_min_s} {lg2.sigma_time_max_s}\n"
        "\n"
        "LOCPHASEID  P   P\n"
        "LOCPHASEID  S   S\n"
        "\n"
        f"LOCQUAL2ERR {locqual2err}\n"
        "\n"
        "LOCANGLES ANGLES_NO 5\n"
    )


def _find_hyp_file(stdout: str | None, workdir: Path, out_root: str) -> Path | None:
    """Locate the per-event ``.hyp`` file this specific run wrote.

    Prefers NLLoc's own ``"Finished location: <path>"`` stdout line
    (:data:`_FINISHED_LOCATION_RE`); falls back to globbing
    ``<out_root>.*.*.grid0.loc.hyp`` (docs/nlloc_interface.md §4's
    confirmed naming pattern) if that line isn't present or the path it
    names doesn't exist. Returns ``None`` (never guesses) if neither
    approach finds exactly one file -- the caller raises :class:`NLLocFailed`.
    """
    all_matches = list(_FINISHED_LOCATION_RE.finditer(stdout or ""))
    if all_matches:
        # Keep the last match; a single-event obs file locates at most once.
        candidate = Path(all_matches[-1].group("path").strip())
        if not candidate.is_absolute():
            candidate = workdir / candidate
        if candidate.exists():
            return candidate

    matches = sorted(glob.glob(str(workdir / f"{out_root}.*.*.grid0.loc.hyp")))
    if len(matches) == 1:
        return Path(matches[0])
    return None


def locate_event(
    event: Event,
    picks_to_use: list[Pick],
    paths: RunPaths,
    cfg: Config,
    pass_name: str,
    iteration: int,
) -> Origin:
    """Locate one event with NLLoc from an explicit, already-filtered pick list.

    ``picks_to_use`` is built entirely by the caller (M6.2's pass-A filter,
    M6.3's pass-B filter): "accepted, correct phase type for this pass,
    this iteration". This function never decides inclusion itself -- it
    writes exactly (and only) ``picks_to_use`` to the ``NLLOC_OBS`` file
    (obspy's writer has no rejection-status filtering of its own,
    docs/nlloc_interface.md §7, so the caller's filtering is the only thing
    standing between a rejected pick and a spurious relocation input).

    Parameters
    ----------
    event : obspy.core.event.Event
        The event being located; only its ``resource_id`` (via
        ``openmeq.catalog.event_id``) is used, for the work directory and
        error messages.
    picks_to_use : list[obspy.core.event.Pick]
        The exact picks to send to NLLoc for this iteration. No two may
        share the same ``(station_code, phase_hint)`` (see
        :class:`DuplicatePickError`), and every one must have
        ``time_errors.uncertainty`` set.
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config
    pass_name : str
        ``"passA"`` or ``"passB"`` (``openmeq.paths.RunPaths.locate_pass_dir``).
    iteration : int
        This pass's iteration number; the work directory is
        ``paths.locate_pass_dir(pass_name, event_id) / f"iter_{iteration:02d}"``.

    Returns
    -------
    obspy.core.event.Origin
        Exactly one `Origin`, parsed from NLLoc's ``.hyp`` output via
        ``obspy.read_events(..., format="NLLOC_HYP", picks=picks_to_use)``,
        so ``Arrival.pick_id`` resolves back to the exact `Pick` objects
        passed in. Each arrival's ``distance`` is converted from obspy's
        QuakeML-standard degrees to km (``obspy.geodetics.degrees2kilometers``),
        per ARCHITECTURE.md §1's km convention for the location step
        (ARCHITECTURE.md §5 updated in this commit to say so).

    Raises
    ------
    DuplicatePickError
        If two picks in ``picks_to_use`` share ``(station_code, phase_hint)``
        -- checked before any file is written or subprocess launched.
    ValueError
        If any pick in ``picks_to_use`` has no ``time_errors.uncertainty``.
    openmeq.config.ConfigError
        If the ``LOCGRID`` search volume is not fully covered by every used
        station/phase's own travel-time grid (docs/nlloc_interface.md's
        ``LOCGRID`` note).
    NLLocFailed
        On a non-zero NLLoc exit, or if NLLoc exited 0 but produced no
        per-event ``.hyp`` file (e.g. too few phases passed ``LOCMETH``'s
        minimum-phase-count filter -- "0 events located").
    """
    eid = get_event_id(event)

    _check_no_duplicate_station_phase(picks_to_use, eid)
    _check_uncertainties_set(picks_to_use, eid)

    ctx = _get_context(cfg, paths)
    stations_used = {p.waveform_id.station_code for p in picks_to_use}
    phases_used = {p.phase_hint for p in picks_to_use}
    _check_locgrid_coverage(ctx, stations_used, phases_used, eid)

    workdir = paths.locate_pass_dir(pass_name, eid) / f"iter_{iteration:02d}"
    workdir.mkdir(parents=True, exist_ok=True)

    throwaway_event = Event()
    throwaway_event.picks = list(picks_to_use)
    Catalog(events=[throwaway_event]).write(str(workdir / "obs.txt"), format="NLLOC_OBS")

    control_text = _render_control(cfg, ctx, "obs.txt", _OUT_ROOT)
    ctrl_path = workdir / "run.in"
    ctrl_path.write_text(control_text)

    try:
        proc = run([str(bin_path("NLLoc")), ctrl_path.name], cwd=workdir)
    except ExternalToolError as exc:
        raise NLLocFailed(
            eid, pass_name, iteration, _tail(exc.stderr) or _tail(exc.stdout)
        ) from exc

    hyp_path = _find_hyp_file(proc.stdout, workdir, _OUT_ROOT)
    if hyp_path is None:
        raise NLLocFailed(
            eid,
            pass_name,
            iteration,
            _tail(proc.stdout)
            or "NLLoc exited 0 but produced no per-event .hyp file (likely 0 phases passed "
            "LOCMETH's minimum-phase-count filter, or an observation was rejected for a "
            "grid-coverage mismatch)",
        )

    result_catalog = read_events(str(hyp_path), format="NLLOC_HYP", picks=picks_to_use)
    if len(result_catalog) != 1:
        raise NLLocFailed(
            eid,
            pass_name,
            iteration,
            f"expected exactly one Event from {hyp_path}, got {len(result_catalog)}",
        )
    origins = result_catalog[0].origins
    if len(origins) != 1:
        raise NLLocFailed(
            eid,
            pass_name,
            iteration,
            f"expected exactly one Origin from {hyp_path}, got {len(origins)}",
        )
    origin = origins[0]

    for arrival in origin.arrivals:
        if arrival.distance is not None:
            arrival.distance = degrees2kilometers(arrival.distance)

    return origin
