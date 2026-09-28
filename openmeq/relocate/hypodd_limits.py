"""hypoDD/ph2dt compiled-array-limit checking and recompilation.

Plan key: M9.3 / OpenMEQ-6ld.3 (PLAN.md §7's "setup_hypodd.py ... checks
event/station/pair counts and 3D grid dimensions against the compiled
limits ... and recompiles with larger limits if exceeded"). This module
implements that logic as a library; wiring it into a CLI is M9.5's
``scripts/setup_hypodd.py`` (out of scope here).

Ground truth for the compiled limits and the recompile mechanism -- see
``bd show OpenMEQ-6ld`` (the M9 epic's design note) and
``docs/ARCHITECTURE.md`` §6c/§7:

- ``external/HYPODD/include/hypoDD.inc``: ``MAXEVE``, ``MAXDATA``,
  ``MAXEVE0``, ``MAXDATA0``, ``MAXLAY``, ``MAXSTA``, ``MAXCL``.
- ``external/HYPODD/include/ph2dt.inc``: ``MEV``, ``MSTA``, ``MOBS``.
- ``external/HYPODD/include/vel3d.inc``: ``maxnx``, ``maxny``, ``maxnz``.
- Compiled values are **parsed from the `.inc` files at check time**, never
  hardcoded as the shipped defaults above: a previous recompile may already
  have changed them (idempotency -- work item 5).
- Recompile mechanism: ``src/Makefile``'s ``SUBDIRS = hypoDD ncsn2pha
  ph2dt``; ``make clean && make`` in ``<hypodd tree>/src`` (via
  `openmeq.external.run`) rebuilds all three, landing binaries at
  ``src/hypoDD/hypoDD`` and ``src/ph2dt/ph2dt`` -- the same mechanism
  ``install.sh`` (M0.2) runs once at install time with the shipped
  defaults. Both subdirectory Makefiles build with plain gfortran, no
  special flags beyond ``-O -I../../include``; verified against a real
  edited ``.inc`` file (not just assumed) during this module's development
  -- a full ``make clean && make`` of the whole tree takes under 2 seconds.

**Never point this module's recompile functions at the main checkout's**
``external/HYPODD``/``external/bin`` **from a test**: copy the source tree
to a scratch directory, patch the copy's include files, build there, and
install to a scratch bin directory. `recompile_hypodd` itself is
directory-agnostic (it takes both directories as arguments); enforcing the
"never touch the main tree in tests" rule is the caller's job.

**3D (IMOD=9) coverage/registration check** (`check_vel3d_coverage`, M9.6 /
OpenMEQ-6ld.6): a *different* failure mode from the compiled-array-limit
checks above -- a genuine memory-safety bug in the bundled hypoDD 2.1
Fortran, not a size limit. Root-caused with a debug build (``gfortran -g
-fcheck=all -fbacktrace``, never done to the main checkout's
``external/HYPODD`` -- see this task's report for the scratch-build
procedure) against a small-aperture (~15 km, borehole+surface) network:

- ``src/hypoDD/ray_3d.f``'s ``intmap`` subroutine (called from ``vel3``,
  itself called from every point along every trial ray path the ``IMOD=9``
  simulps ray tracer builds) computes a node-grid lookup index directly
  from a query point's raw Cartesian coordinate --
  ``ip=int((x+xl)/bld); ip=ixloc(ip)`` (``ray_3d.f:585-586``, similarly for
  ``jp``/``iyloc`` and ``kp``/``izloc``) -- with **no bounds check** before
  indexing the ``ixloc``/``iyloc``/``izloc`` lookup arrays
  (``vel3d.inc``'s ``ixkms``/``iykms``/``izkms``, each sized 10000). Any
  query point (an event or station, projected into the 3D model's own
  Cartesian frame) that falls outside the ``vel3D.dat``'s node domain
  produces an out-of-range index: caught as a clean
  ``Fortran runtime error: Index ... below lower bound of 1`` by a
  ``-fcheck=all`` debug build, but undefined behaviour -- in practice a
  ``SIGSEGV`` -- in the production build (``install.sh``'s plain ``-O``,
  no bounds checking).
- Confirmed by reproduction: a small-aperture network's events/stations,
  correctly registered against a ``vel3D.dat`` that covers them (matching
  ``LAT_3D``/``LON_3D``/``ROT_3D`` and adequate node-grid extent, e.g. via
  `openmeq.relocate.nlloc_to_simul.nlloc_grid_to_vel3d`), relocates
  successfully with no crash. The *same* network with a **mismatched**
  ``origin_3d`` (simulating a ``vel3D.dat`` copied from a different
  survey/frame, or any other registration mistake) reproduces the exact
  crash above with a full backtrace through ``vel3_``/``intmap_`` --
  confirming the M9 epic design note's leading hypothesis over the M9.4
  worker's "pre-existing Fortran fragility" guess. See
  ``docs/hypodd_3d_notes.md`` for the full backtrace and reproduction
  recipe.
- **Fix**: since this is an input problem (nothing in the pipeline
  previously checked that a supplied ``vel3D.dat``'s node domain, under
  hypoDD.inp's declared ``LAT_3D``/``LON_3D``/``ROT_3D``, actually covers
  the real event/station geometry -- `nlloc_to_simul`'s own
  ``_check_coordinate_frame`` only validates this for the one path that
  builds a ``vel3D.dat`` *from* an NLLoc grid, not for a user-supplied
  file), `check_vel3d_coverage` projects every accepted event's and every
  station's location into the model's own frame (a direct port of hypoDD's
  own runtime projection, ``setorg.f``/``dist.f`` -- **not**
  ``setorg_3d.f``/``disto.f``, which are unused dead code: ``hypoDD.f``'s
  ``IMOD.eq.9`` branch, lines ~550-563, sets ``sdc0_lat/lon=lat_3d/lon_3d``
  and calls the ordinary ``setorg``/``SDC2`` machinery with ``rot_3d`` as
  the rotation) and raises `openmeq.config.ConfigError`, naming every
  offending station/event and which axis it falls outside of, before
  hypoDD ever runs. Patching the Fortran itself (adding a bounds check in
  ``intmap``) was considered and rejected for this task: the compiled
  binaries are shared build artifacts (``external/bin``, built once by
  ``install.sh``), a source patch would need to flow through that install
  path for every environment, and a clean *input* validation catches the
  same failure earlier, with a far more actionable message, without ever
  touching the vendored Fortran.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from obspy import Catalog

from openmeq.catalog import accepted, event_id
from openmeq.config import Config
from openmeq.config import ConfigError as _ConfigError
from openmeq.external import external_dir
from openmeq.external import run as _run_external
from openmeq.log import get_logger
from openmeq.relocate.hypodd_files import Vel3DGrid, depth_m_to_km

logger = get_logger("relocate.hypodd_limits")

#: Which `.inc` file each compiled-limit parameter lives in -- used both by
#: `check_compiled_limits` (which file to parse for a given quantity) and
#: `recompile_hypodd` (which file to edit for a given `new_limits` key).
HYPODD_INC_PARAMS = ("MAXEVE", "MAXDATA", "MAXEVE0", "MAXDATA0", "MAXLAY", "MAXSTA", "MAXCL")
PH2DT_INC_PARAMS = ("MEV", "MSTA", "MOBS")
VEL3D_INC_PARAMS = ("maxnx", "maxny", "maxnz")

_INC_FILE_PARAMS = {
    "hypoDD.inc": HYPODD_INC_PARAMS,
    "ph2dt.inc": PH2DT_INC_PARAMS,
    "vel3d.inc": VEL3D_INC_PARAMS,
}

#: Default headroom multiplier `ensure_compiled_limits` applies to the
#: current dataset value when computing a new compiled limit, per the M9
#: epic design note's "Pitfalls" recommendation ("choose new limits
#: generously ... so a small dataset growth doesn't force another
#: recompile immediately").
DEFAULT_HEADROOM_FACTOR = 1.5

#: Maps a `LimitCheck.name` to the compiled-limit parameter(s) that must be
#: raised if that check is exceeded. `event_count`/`station_count` are
#: checked against the smaller of hypoDD's own limit and ph2dt's (both
#: binaries process the same quantity), so both underlying parameters are
#: raised together, to the same new value, rather than letting them drift
#: apart for the same logical quantity.
_CHECK_NAME_TO_PARAMS = {
    "event_count": ("MAXEVE", "MEV"),
    "station_count": ("MAXSTA", "MSTA"),
    "observation_count": ("MAXDATA",),
    "phases_per_event": ("MOBS",),
    "grid_nx": ("maxnx",),
    "grid_ny": ("maxny",),
    "grid_nz": ("maxnz",),
}


# ---------------------------------------------------------------------------
# .inc file parsing
# ---------------------------------------------------------------------------


def _parse_fortran_int_params(path: Path, names: tuple[str, ...]) -> dict[str, int]:
    """Parse ``parameter(NAME= value, ...)`` integers out of a hypoDD
    ``.inc`` file, skipping fixed-form Fortran comment lines (a
    ``c``/``C``/``*``/``!`` in column 1).

    ``hypoDD.inc`` in particular ships a second, entirely commented-out
    "small size problems" parameter block right after the active one
    (``include/hypoDD.inc``'s own ``cc parameters for small size
    problems`` section) -- skipping comment lines isn't just defensive
    styling here, it's required for correctness once *any* value in the
    active block is edited to no longer be the smallest/most distinctive
    number in the file.

    Parameters
    ----------
    path : pathlib.Path
        The ``.inc`` file.
    names : tuple[str, ...]
        Parameter names to look up (e.g. `HYPODD_INC_PARAMS`).

    Returns
    -------
    dict[str, int]

    Raises
    ------
    openmeq.config.ConfigError
        If any requested name isn't found in the active parameter block.
    """
    text = path.read_text()
    code_lines = [ln for ln in text.splitlines() if ln[:1] not in ("c", "C", "*", "!")]
    code = "\n".join(code_lines)
    values: dict[str, int] = {}
    for name in names:
        m = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)", code)
        if m is None:
            raise _ConfigError(
                f"{path}: compiled-limit parameter {name!r} not found in the active "
                "(non-comment) parameter block"
            )
        values[name] = int(m.group(1))
    return values


def _substitute_inc_params(text: str, updates: dict[str, int]) -> tuple[str, dict[str, int]]:
    """Replace ``NAME= value`` integers in the *active* (non-comment)
    parameter block of a hypoDD ``.inc`` file's text with new values from
    `updates`, leaving comment lines (a ``c``/``C``/``*``/``!`` in column 1
    -- including any commented-out alternate block) untouched.

    A regex-based rewrite of just the numeric values (rather than a
    wholesale rewrite of the ``parameter(...)`` statement) preserves
    Fortran continuation-line syntax and comment formatting regardless of
    how many digits the new values have, per the M9 epic design note's
    "Pitfalls" section -- confirmed safe against the real ``.inc`` files by
    `openmeq.relocate.hypodd_limits`'s own ``requires_hypodd`` integration
    test, which recompiles a file edited by this exact function.

    Parameters
    ----------
    text : str
        The `.inc` file's current contents.
    updates : dict[str, int]
        ``{parameter_name: new_value}``.

    Returns
    -------
    (str, dict[str, int])
        The new file contents, and ``{parameter_name: old_value}`` for
        every substitution actually made.

    Raises
    ------
    openmeq.config.ConfigError
        If any name in `updates` isn't found in the active parameter block.
    """
    lines = text.splitlines(keepends=True)
    old_values: dict[str, int] = {}
    remaining = dict(updates)
    out_lines = []
    for line in lines:
        if not remaining or line[:1] in ("c", "C", "*", "!"):
            out_lines.append(line)
            continue
        new_line = line
        for name in list(remaining):
            pattern = re.compile(rf"(\b{re.escape(name)}\s*=\s*)(-?\d+)")
            match = pattern.search(new_line)
            if match is None:
                continue
            old_values[name] = int(match.group(2))
            new_value = remaining.pop(name)
            new_line = pattern.sub(lambda m, v=new_value: f"{m.group(1)}{v}", new_line, count=1)
        out_lines.append(new_line)

    if remaining:
        raise _ConfigError(
            f"parameter(s) {sorted(remaining)} not found in the active (non-comment) "
            "parameter block"
        )
    return "".join(out_lines), old_values


# ---------------------------------------------------------------------------
# limit check
# ---------------------------------------------------------------------------


@dataclass
class LimitCheck:
    """One checked quantity: the current dataset value vs. the compiled
    (or configured, for the 3D grid) limit.

    Attributes
    ----------
    name : str
        One of ``"event_count"``, ``"station_count"``,
        ``"observation_count"``, ``"phases_per_event"``, or (only when a
        3D grid is checked) ``"grid_nx"``/``"grid_ny"``/``"grid_nz"``.
    current : int
        The current dataset's value for this quantity.
    limit : int
        The effective compiled limit checked against (for
        ``event_count``/``station_count``, already the smaller of hypoDD's
        own limit and ph2dt's -- see `check_compiled_limits`).
    exceeded : bool
        ``current > limit``.
    """

    name: str
    current: int
    limit: int
    exceeded: bool


@dataclass
class LimitCheckResult:
    """Every `LimitCheck` from one `check_compiled_limits` call."""

    checks: list[LimitCheck] = field(default_factory=list)

    @property
    def any_exceeded(self) -> bool:
        """``True`` if any check in `checks` is exceeded."""
        return any(c.exceeded for c in self.checks)

    def exceeded_checks(self) -> list[LimitCheck]:
        """The subset of `checks` that are exceeded."""
        return [c for c in self.checks if c.exceeded]

    def by_name(self, name: str) -> LimitCheck:
        """The single `LimitCheck` named `name` (see `LimitCheck.name`).

        Raises
        ------
        KeyError
            If no check with that name exists.
        """
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


def _accepted_event_obs_counts(catalog: Catalog) -> list[int]:
    """Per accepted event, the count of non-rejected picks with an arrival
    on the event's preferred origin -- i.e. the phase.dat/event count
    `openmeq.relocate.hypodd_files.write_phase_dat` would write for that
    event (rejected picks excluded per ARCHITECTURE.md §5/§6c, same
    ``pick.evaluation_status != "rejected"`` check).
    """
    counts = []
    for event in accepted(catalog):
        origin = event.preferred_origin()
        if origin is None:
            origin = event.origins[0] if event.origins else None
        n = 0
        if origin is not None:
            picks_by_id = {str(p.resource_id): p for p in event.picks}
            for arrival in origin.arrivals:
                pick = picks_by_id.get(str(arrival.pick_id))
                if pick is None or pick.evaluation_status == "rejected":
                    continue
                n += 1
        counts.append(n)
    return counts


def check_compiled_limits(
    cfg: Config,
    catalog: Catalog,
    station_table: pd.DataFrame,
    vel3d_grid: Vel3DGrid | None = None,
    *,
    xcorr_obs_count: int = 0,
    include_dir: Path | str | None = None,
) -> LimitCheckResult:
    """Check a dataset's event/station/observation counts (and, if a 3D
    model is configured, its grid dimensions) against hypoDD/ph2dt's
    *currently compiled* array limits.

    Checks performed (see the M9 epic design note, quoted in this module's
    docstring, for the parameter-name ground truth):

    - ``event_count``: accepted event count vs. ``min(MAXEVE, MEV)`` --
      both hypoDD's own limit and ph2dt's matter, since both binaries
      process the full event set (ph2dt before hypoDD ever sees a
      clustered/filtered subset).
    - ``station_count``: station count vs. ``min(MAXSTA, MSTA)``.
    - ``observation_count``: total phase-pick observation count (summed
      across accepted events, rejected picks excluded) plus
      `xcorr_obs_count`, vs. ``MAXDATA``. `xcorr_obs_count` defaults to
      ``0`` since cross-correlation (M10.1) hasn't run yet when this is
      first called in the pipeline -- re-run this check with the real
      ``dt.cc`` pair count once it exists (the M9 epic design note's
      "Pitfalls": re-check after every relevant upstream step).
    - ``phases_per_event``: the single largest per-event phase-pick count,
      vs. ``MOBS`` (ph2dt-only limit; catalog phases only, not xcorr).
    - ``grid_nx``/``grid_ny``/``grid_nz`` (only if `vel3d_grid` is given):
      each grid axis's node count vs. ``vel3d.inc``'s ``maxnx``/``maxny``/
      ``maxnz``.

    Parameters
    ----------
    cfg : openmeq.config.Config
        Accepted for interface symmetry with the file writers and so a
        future check can read config fields (e.g. a configured headroom
        policy); not otherwise used directly by this function today.
    catalog : obspy.Catalog
        The full catalogue; rejected events/picks are excluded via
        `openmeq.catalog.accepted` and `Pick.evaluation_status`,
        respectively (ARCHITECTURE.md §5/§6c).
    station_table : pandas.DataFrame
        `openmeq.stations.station_table`'s contract; only the row count is
        used.
    vel3d_grid : openmeq.relocate.hypodd_files.Vel3DGrid or None, optional
        From `openmeq.relocate.hypodd_files.read_vel3d_header`. When
        given, the three grid-node-count checks are added.
    xcorr_obs_count : int, optional
        Cross-correlation (``dt.cc``) observation count to add to
        ``observation_count``'s ``MAXDATA`` check; ``0`` by default (see
        above).
    include_dir : pathlib.Path or str or None, optional
        Directory containing ``hypoDD.inc``/``ph2dt.inc``/``vel3d.inc``.
        Defaults to ``<external_dir()>/HYPODD/include`` (the real,
        installed tree; `openmeq.external.external_dir`) -- tests pass a
        scratch directory with tiny fake `.inc` files instead, so this
        function never needs a real compiler/tree to unit-test the
        counting logic itself.

    Returns
    -------
    LimitCheckResult
    """
    include_dir = (
        Path(include_dir) if include_dir is not None else external_dir() / "HYPODD" / "include"
    )

    hypodd_inc = _parse_fortran_int_params(include_dir / "hypoDD.inc", HYPODD_INC_PARAMS)
    ph2dt_inc = _parse_fortran_int_params(include_dir / "ph2dt.inc", PH2DT_INC_PARAMS)

    n_events = len(accepted(catalog))
    n_stations = len(station_table)
    per_event_obs = _accepted_event_obs_counts(catalog)
    n_obs_total = sum(per_event_obs) + xcorr_obs_count
    max_obs_per_event = max(per_event_obs, default=0)

    event_limit = min(hypodd_inc["MAXEVE"], ph2dt_inc["MEV"])
    station_limit = min(hypodd_inc["MAXSTA"], ph2dt_inc["MSTA"])

    checks = [
        LimitCheck(
            name="event_count",
            current=n_events,
            limit=event_limit,
            exceeded=n_events > event_limit,
        ),
        LimitCheck(
            name="station_count",
            current=n_stations,
            limit=station_limit,
            exceeded=n_stations > station_limit,
        ),
        LimitCheck(
            name="observation_count",
            current=n_obs_total,
            limit=hypodd_inc["MAXDATA"],
            exceeded=n_obs_total > hypodd_inc["MAXDATA"],
        ),
        LimitCheck(
            name="phases_per_event",
            current=max_obs_per_event,
            limit=ph2dt_inc["MOBS"],
            exceeded=max_obs_per_event > ph2dt_inc["MOBS"],
        ),
    ]

    if vel3d_grid is not None:
        vel3d_inc = _parse_fortran_int_params(include_dir / "vel3d.inc", VEL3D_INC_PARAMS)
        for axis, current in (
            ("nx", vel3d_grid.nx),
            ("ny", vel3d_grid.ny),
            ("nz", vel3d_grid.nz),
        ):
            limit = vel3d_inc[f"max{axis}"]
            checks.append(
                LimitCheck(
                    name=f"grid_{axis}",
                    current=current,
                    limit=limit,
                    exceeded=current > limit,
                )
            )

    return LimitCheckResult(checks=checks)


# ---------------------------------------------------------------------------
# recompile
# ---------------------------------------------------------------------------


def recompile_hypodd(
    new_limits: dict[str, int],
    external_hypodd_dir: Path | str,
    external_bin_dir: Path | str,
    log=None,
) -> None:
    """Rewrite the compiled-array-limit ``.inc`` files under
    ``<external_hypodd_dir>/include`` for every parameter in `new_limits`,
    rebuild hypoDD/ph2dt (``make clean && make`` in
    ``<external_hypodd_dir>/src``, via `openmeq.external.run`), and install
    the rebuilt binaries into `external_bin_dir`, overwriting whatever was
    there.

    This always rebuilds when called (it does no exceeded-limit comparison
    of its own -- see `ensure_compiled_limits` for the idempotent,
    check-then-recompile-only-if-needed wrapper work item 5 calls for).

    Parameters
    ----------
    new_limits : dict[str, int]
        ``{parameter_name: new_value}``, e.g. ``{"MAXEVE": 16000, "MEV":
        24000}``. Every name must belong to one of `HYPODD_INC_PARAMS`,
        `PH2DT_INC_PARAMS` or `VEL3D_INC_PARAMS`; only the `.inc` file(s)
        containing at least one requested parameter are edited.
    external_hypodd_dir : pathlib.Path or str
        The hypoDD source tree root (containing ``include/`` and ``src/``).
        **Never the main checkout's `external/HYPODD` from a test** -- a
        scratch copy.
    external_bin_dir : pathlib.Path or str
        Where the rebuilt ``hypoDD``/``ph2dt`` binaries are installed. In
        production, ``external/bin``; in a test, a scratch directory.
    log : logging.Logger or None, optional
        Passed through to `openmeq.external.run`, and used to log the
        old/new limit values before rebuilding (at ``WARNING``, since a
        recompile is a significant, somewhat slow side effect worth
        surfacing).

    Raises
    ------
    openmeq.config.ConfigError
        If a name in `new_limits` isn't found in any of the three `.inc`
        files' active parameter blocks.
    openmeq.external.ExternalToolError
        If ``make clean`` or ``make`` exits non-zero or times out.
    RuntimeError
        If the build succeeds (exit 0) but doesn't produce the expected
        binary paths (shouldn't happen with the bundled Makefiles; a
        defensive check rather than silently installing nothing new).
    """
    external_hypodd_dir = Path(external_hypodd_dir)
    external_bin_dir = Path(external_bin_dir)
    include_dir = external_hypodd_dir / "include"
    active_log = log if log is not None else logger

    old_values: dict[str, int] = {}
    changed_files: list[str] = []
    for filename, param_names in _INC_FILE_PARAMS.items():
        updates = {k: v for k, v in new_limits.items() if k in param_names}
        if not updates:
            continue
        inc_path = include_dir / filename
        new_text, replaced = _substitute_inc_params(inc_path.read_text(), updates)
        inc_path.write_text(new_text)
        old_values.update(replaced)
        changed_files.append(filename)

    unknown = set(new_limits) - set(old_values)
    if unknown:
        known = sorted(HYPODD_INC_PARAMS + PH2DT_INC_PARAMS + VEL3D_INC_PARAMS)
        raise _ConfigError(
            f"recompile_hypodd: unknown compiled-limit parameter(s) {sorted(unknown)} "
            f"(known: {known})"
        )

    active_log.warning(
        "Recompiling hypoDD/ph2dt with larger compiled limits (%s changed) -- old values: "
        "%s, new values: %s",
        ", ".join(changed_files) or "(none)",
        old_values,
        new_limits,
    )

    src_dir = external_hypodd_dir / "src"
    _run_external(["make", "clean"], cwd=src_dir, log=active_log)
    _run_external(["make"], cwd=src_dir, log=active_log, timeout=600)

    hypodd_bin = src_dir / "hypoDD" / "hypoDD"
    ph2dt_bin = src_dir / "ph2dt" / "ph2dt"
    if not hypodd_bin.is_file():
        raise RuntimeError(f"recompile_hypodd: build did not produce {hypodd_bin}")
    if not ph2dt_bin.is_file():
        raise RuntimeError(f"recompile_hypodd: build did not produce {ph2dt_bin}")

    external_bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(hypodd_bin, external_bin_dir / "hypoDD")
    shutil.copy(ph2dt_bin, external_bin_dir / "ph2dt")
    active_log.info(
        "recompile_hypodd: installed rebuilt hypoDD/ph2dt to %s (old limits: %s, new: %s)",
        external_bin_dir,
        old_values,
        new_limits,
    )


# ---------------------------------------------------------------------------
# check + recompile-if-needed
# ---------------------------------------------------------------------------


def ensure_compiled_limits(
    cfg: Config,
    catalog: Catalog,
    station_table: pd.DataFrame,
    external_hypodd_dir: Path | str,
    external_bin_dir: Path | str,
    vel3d_grid: Vel3DGrid | None = None,
    *,
    xcorr_obs_count: int = 0,
    headroom_factor: float = DEFAULT_HEADROOM_FACTOR,
    log=None,
) -> LimitCheckResult:
    """`check_compiled_limits`, then `recompile_hypodd` only if it found
    something exceeded (work item 5's idempotency requirement -- a dataset
    that already fits never triggers a recompile).

    When a recompile is needed, every exceeded check's underlying compiled
    parameter(s) (`_CHECK_NAME_TO_PARAMS`) are raised to
    ``ceil(current * headroom_factor)`` (never less than ``current + 1``),
    per the M9 epic design note's headroom recommendation -- generous
    enough that a small further dataset growth doesn't immediately force
    another recompile.

    Parameters
    ----------
    cfg, catalog, station_table, vel3d_grid, xcorr_obs_count
        As `check_compiled_limits`.
    external_hypodd_dir, external_bin_dir, log
        As `recompile_hypodd`. `include_dir` for the check is derived as
        ``<external_hypodd_dir>/include``, so the check and any recompile
        always agree on which tree they're looking at.
    headroom_factor : float, optional
        See above; default `DEFAULT_HEADROOM_FACTOR`.

    Returns
    -------
    LimitCheckResult
        The check performed *before* any recompile (so callers can log/
        report what triggered it). The compiled limits on disk and in
        `external_bin_dir` reflect the recompile, if one happened.
    """
    include_dir = Path(external_hypodd_dir) / "include"
    result = check_compiled_limits(
        cfg,
        catalog,
        station_table,
        vel3d_grid,
        xcorr_obs_count=xcorr_obs_count,
        include_dir=include_dir,
    )
    if not result.any_exceeded:
        return result

    new_limits: dict[str, int] = {}
    for check in result.exceeded_checks():
        new_value = max(math.ceil(check.current * headroom_factor), check.current + 1)
        for param in _CHECK_NAME_TO_PARAMS[check.name]:
            new_limits[param] = max(new_limits.get(param, 0), new_value)

    recompile_hypodd(new_limits, external_hypodd_dir, external_bin_dir, log=log)
    return result


# ---------------------------------------------------------------------------
# 3D (IMOD=9) model coverage / registration check (M9.6 / OpenMEQ-6ld.6)
# ---------------------------------------------------------------------------

#: WGS72 ellipsoid constants ``setorg.f``/``dist.f`` use for their short
#: distance conversion (SDC) -- ``REARTH``/``ELLIP`` in ``setorg.f:47-48``.
_SDC_REARTH_KM = 6378.135
_SDC_ELLIP = 298.26


def _hypodd_sdc_xy_km(
    lat_deg: float, lon_deg: float, orlat_deg: float, orlon_deg: float, rot_deg: float
) -> tuple[float, float]:
    """A direct Python port of ``src/hypoDD/setorg.f`` + ``dist.f``'s short
    distance conversion (SDC) -- the *exact* projection hypoDD's own runtime
    uses to turn an event's/station's lat/lon into the local Cartesian
    ``(x east-km, y north-km)`` frame the ``IMOD=9`` ray tracer
    (``ray_3d.f``/``get_vel3d.f``) queries the 3D model in.

    Ground truth (confirmed directly against source, not re-derived from
    generic geodesy, per this project's own convention): for ``IMOD=9``,
    ``hypoDD.f``'s ``imod.eq.9`` branch (~line 550-563) sets
    ``sdc0_lat=lat_3d, sdc0_lon=lon_3d`` and calls
    ``setorg(sdc0_lat, sdc0_lon, rot_3d, 0)``, then every event's/station's
    Cartesian position is `` call SDC2(x, y, lat, lon, -1)`` (`partials_3d.f`
    /``hypoDD.f``), which dispatches to ``dist.f`` for direction ``-1``.
    **Not** ``setorg_3d.f``/``disto.f`` -- despite the ``_3d`` name, those
    two subroutines are unused dead code in this hypoDD build (confirmed by
    grepping every call site in ``src/hypoDD/*.f``: `get_vel3d.f`'s own
    ``call setorg_3d(...)`` is commented out, and ``setorg_3d``/``disto``
    are never called from anywhere else).

    Empirically verified (not just transcribed) against a real debug
    hypoDD run's ``hypoDD.loc`` output (its ``x``/``y`` columns, metres,
    match this function's ``x_km``/``y_km`` * 1000 to within about 1 m at a
    6 km offset -- the small residual is single- vs. double-precision
    rounding in hypoDD's own ``real`` arithmetic, not a formula
    discrepancy) -- see ``docs/hypodd_3d_notes.md``.

    Parameters
    ----------
    lat_deg, lon_deg : float
        The point to project (degrees, standard signed longitude,
        negative west).
    orlat_deg, orlon_deg : float
        The projection origin -- hypoDD.inp's ``LAT_3D``/``LON_3D``
        (`openmeq.config.HypoddVelocityModelConfig.origin_3d`).
    rot_deg : float
        hypoDD.inp's ``ROT_3D`` (anticlockwise-positive degrees,
        `openmeq.config.HypoddVelocityModelConfig.rotation_3d_deg`).

    Returns
    -------
    (float, float)
        ``(x_km, y_km)`` in the 3D model's own rotated Cartesian frame --
        directly comparable to a `Vel3DGrid`'s ``xn``/``yn`` node
        coordinates.
    """
    rad = math.radians(1.0)
    olat = orlat_deg * 60.0
    olon = orlon_deg * 60.0

    phi = math.radians(orlat_deg)
    beta = phi - math.sin(2.0 * phi) / _SDC_ELLIP
    rlatc = 1.0 if phi == 0.0 else math.tan(beta) / math.tan(phi)

    lat1_o = math.atan(rlatc * math.tan(olat * rad / 60.0))
    lat2_o = math.atan(rlatc * math.tan((olat + 1.0) * rad / 60.0))
    dela = lat2_o - lat1_o
    r = _SDC_REARTH_KM * (1.0 - math.sin(lat1_o) ** 2 / _SDC_ELLIP)
    aa = r * dela
    delb = math.acos(math.sin(lat1_o) ** 2 + math.cos(rad / 60.0) * math.cos(lat1_o) ** 2)
    bc = r * delb
    bb = bc / math.cos(lat1_o)

    rotate = math.radians(rot_deg)
    sint = math.sin(rotate)
    cost = math.cos(rotate)

    q = 60.0 * lat_deg - olat
    yp = q + olat
    lat1 = math.atan(rlatc * math.tan(rad * yp / 60.0))
    lat2 = math.atan(rlatc * math.tan(rad * olat / 60.0))
    lat3 = (lat1 + lat2) / 2.0
    xx = 60.0 * lon_deg - olon
    q = q * aa
    xx = xx * bb * math.cos(lat3)
    if rotate != 0.0:
        yp2 = cost * q + sint * xx
        xx = cost * xx - sint * q
        q = yp2

    return xx, q


def _preferred_origin_lat_lon_depth_km(event) -> tuple[float, float, float]:
    origin = event.preferred_origin()
    if origin is None:
        if not event.origins:
            raise ValueError(f"Event {event.resource_id} has no origins")
        origin = event.origins[0]
    depth_km = depth_m_to_km(origin.depth) if origin.depth is not None else 0.0
    return origin.latitude, origin.longitude, depth_km


def check_vel3d_coverage(
    vel3d_grid: Vel3DGrid,
    lat_3d: float,
    lon_3d: float,
    rot_3d: float,
    catalog: Catalog,
    station_table: pd.DataFrame,
) -> None:
    """Verify every accepted event's and every station's location, projected
    into the 3D model's own Cartesian frame exactly as hypoDD's own runtime
    would (`_hypodd_sdc_xy_km`, driven by hypoDD.inp's ``LAT_3D``/
    ``LON_3D``/``ROT_3D``), falls within `vel3d_grid`'s own node domain
    (``xn[0]..xn[-1]``, ``yn[0]..yn[-1]``, ``zn[0]..zn[-1]``).

    This is a memory-safety check, not a cosmetic one: any query point
    outside this domain sends ``ray_3d.f``'s ``intmap`` subroutine an
    out-of-bounds index into the ``ixloc``/``iyloc``/``izloc`` compiled
    lookup arrays (this module's own docstring has the full citation
    trail), which segfaults hypoDD's production build (or aborts cleanly
    only under a ``-fcheck=all`` debug build). Call this **before** running
    ``hypoDD`` with ``IMOD=9`` -- there is currently no other place in the
    pipeline this is checked (`openmeq.relocate.nlloc_to_simul`'s own
    ``_check_coordinate_frame`` only validates the *building* of a
    ``vel3D.dat`` from an NLLoc grid, not a user-supplied file, and does not
    check individual event/station coverage at all).

    A station's z-coordinate is its ARCHITECTURE.md §6 *effective*
    elevation (``elevation_m - depth_m``, km, negative for a downhole
    borehole sensor) -- but clamped to ``0.0`` whenever that would be
    negative, mirroring ``getdata.f:275-278``'s own silent clamp for any
    ``IMOD`` other than ``5`` (which includes ``IMOD=9``): this check
    validates the coordinate hypoDD will *actually* query the model at, not
    the pre-clamp value (`openmeq.relocate.hypodd_files.write_station_dat`
    logs the pre-clamp negative-elevation warning separately).

    Parameters
    ----------
    vel3d_grid : Vel3DGrid
        From `openmeq.relocate.hypodd_files.read_vel3d_header`.
    lat_3d, lon_3d, rot_3d : float
        The values that will be written to hypoDD.inp's ``LAT_3D LON_3D
        ROT_3D`` line (`openmeq.relocate.hypodd_files.write_hypodd_inp_3d`,
        `openmeq.config.HypoddVelocityModelConfig`).
    catalog : obspy.Catalog
        The full catalogue; rejected events are excluded via
        `openmeq.catalog.accepted`.
    station_table : pandas.DataFrame
        `openmeq.stations.station_table`'s contract.

    Raises
    ------
    openmeq.config.ConfigError
        Naming every offending station/event, which axis (x/y/z) it falls
        outside of, by how much, and the domain it was checked against --
        if any accepted event or any station falls outside `vel3d_grid`'s
        node domain.
    """
    x_lo, x_hi = min(vel3d_grid.xn), max(vel3d_grid.xn)
    y_lo, y_hi = min(vel3d_grid.yn), max(vel3d_grid.yn)
    z_lo, z_hi = min(vel3d_grid.zn), max(vel3d_grid.zn)

    def _axis_problems(x: float, y: float, z: float) -> list[str]:
        out = []
        if not (x_lo <= x <= x_hi):
            out.append(f"x={x:.3f} km outside [{x_lo:.3f}, {x_hi:.3f}]")
        if not (y_lo <= y <= y_hi):
            out.append(f"y={y:.3f} km outside [{y_lo:.3f}, {y_hi:.3f}]")
        if not (z_lo <= z <= z_hi):
            out.append(f"z={z:.3f} km outside [{z_lo:.3f}, {z_hi:.3f}]")
        return out

    problems: list[str] = []

    for _, row in station_table.iterrows():
        net_sta = f"{row['network']}.{row['station']}"
        elevation_km = (float(row["elevation_m"]) - float(row["depth_m"])) / 1000.0
        z = 0.0 if elevation_km < 0 else elevation_km
        x, y = _hypodd_sdc_xy_km(
            float(row["latitude"]), float(row["longitude"]), lat_3d, lon_3d, rot_3d
        )
        axis_problems = _axis_problems(x, y, z)
        if axis_problems:
            problems.append(f"station {net_sta}: " + "; ".join(axis_problems))

    for event in accepted(catalog):
        eid = event_id(event)
        lat, lon, depth_km = _preferred_origin_lat_lon_depth_km(event)
        x, y = _hypodd_sdc_xy_km(lat, lon, lat_3d, lon_3d, rot_3d)
        axis_problems = _axis_problems(x, y, depth_km)
        if axis_problems:
            problems.append(f"event {eid}: " + "; ".join(axis_problems))

    if problems:
        raise _ConfigError(
            "3D model (vel3D.dat) coverage check failed: the following stations/events "
            "project (under the given LAT_3D/LON_3D/ROT_3D) outside the vel3D.dat node "
            "domain -- running hypoDD's IMOD=9 ray tracer with this input segfaults "
            "(ray_3d.f's intmap subroutine indexes its ixloc/iyloc/izloc lookup arrays "
            "with no bounds check, per this module's own docstring). Either fix "
            "LAT_3D/LON_3D/ROT_3D (cfg.hypodd.velocity_model.origin_3d/rotation_3d_deg) so "
            "they match the vel3D.dat's own registration, or rebuild the 3D model so its "
            "node grid actually covers this network "
            "(openmeq.relocate.nlloc_to_simul.nlloc_grid_to_vel3d): " + " | ".join(problems)
        )
