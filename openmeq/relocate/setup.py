"""``scripts/setup_hypodd.py``'s importable orchestration (M9.5 / OpenMEQ-6ld.5).

Plan key: M9.5 (PLAN.md §7's ``scripts/setup_hypodd.py -c config.yaml``).
Ties together every other M9 module -- the 1D/3D file writers (M9.2/M9.3,
``openmeq.relocate.hypodd_files``), the compiled-array-limit checker/
recompiler (M9.3, ``openmeq.relocate.hypodd_limits``), the NLLoc-to-SIMUL2000
converter (M9.4, ``openmeq.relocate.nlloc_to_simul``) and the 3D coverage
check (M9.6, ``openmeq.relocate.hypodd_limits.check_vel3d_coverage``) -- into
one importable function, following M5.2's ``openmeq.nlloc_setup.setup_nlloc``
pattern so both ``scripts/setup_hypodd.py`` (a thin CLI) and, later, M10.2/
M10.4's own pipeline step can call it directly. See ``docs/ARCHITECTURE.md``
§6c-2 for the full contract.

This module does *not* write ``phase.dat``/``event.dat`` (the per-event
observation files -- those depend on the catalogue's actual picks/origins
and are M10.2's job, re-derived every relocation run); it prepares hypoDD's
*static* environment: ``station.dat``, ``ph2dt.inp``, ``hypoDD.inp`` and,
for a 3D run, ``vel3D.dat``. Per PLAN.md §9 ("M5 and M9 can be built/run at
any time after M0"), this must succeed even before any events have been
detected -- ``setup_hypodd`` degrades gracefully (a logged ``WARNING``, not
a crash) when ``magnitude/catalog_mag.xml`` doesn't exist yet, skipping only
the parts of the sequence that genuinely need a populated catalogue (the
event-id-map refresh falls back to an empty catalogue -- a harmless no-op
the first time -- and the compiled-limit check is skipped outright).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from obspy import Catalog

from openmeq.catalog import read_catalog
from openmeq.config import Config
from openmeq.config import ConfigError as _ConfigError
from openmeq.external import external_dir
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.relocate.hypodd_files import (
    build_event_id_map,
    build_station_label_map,
    read_vel3d_header,
    write_hypodd_inp_1d,
    write_hypodd_inp_3d,
    write_ph2dt_inp,
    write_station_dat,
)
from openmeq.relocate.hypodd_limits import (
    LimitCheckResult,
    check_vel3d_coverage,
    compiled_vel3d_node_limits,
    ensure_compiled_limits,
)
from openmeq.relocate.nlloc_to_simul import nlloc_grid_to_vel3d, read_nlloc_grid
from openmeq.stations import station_table as load_station_table
from openmeq.velocity import read_layers

if TYPE_CHECKING:
    from openmeq.paths import RunPaths

logger = get_logger("relocate.setup")

#: `openmeq.manifest.StepManifest` step name -- shares `relocate/manifest.json`
#: with M10.2/M10.4's own relocation-run manifest entries (the manifest
#: contract is per-key, not per-file; `MANIFEST_KEY` below is this module's
#: own, distinct key).
STEP_NAME = "relocate"
MANIFEST_KEY = "hypodd_setup"


@dataclass
class SetupHypoddResult:
    """What one `setup_hypodd` call did -- returned for the CLI's own
    logging and for callers (M10.2/M10.4) that need to know, e.g., which
    ``IMOD`` was written.

    Attributes
    ----------
    rebuilt : bool
        ``True`` if the setup was (re)built, ``False`` if skipped as
        unchanged (`openmeq.manifest.StepManifest`, see `setup_hypodd`).
    imod : int or None
        ``1`` (1D) or ``9`` (3D), whichever was written. ``None`` if
        `rebuilt` is ``False`` (skipped -- the previous run's `hypoDD.inp`
        is left as-is).
    nlloc_to_simul_ran : bool
        Whether the ``--nlloc-to-simul`` converter actually ran.
    limit_check : openmeq.relocate.hypodd_limits.LimitCheckResult or None
        The compiled-limit check's result, or ``None`` if it was skipped
        (no catalogue available yet, or `rebuilt` is ``False``).
    """

    rebuilt: bool
    imod: int | None
    nlloc_to_simul_ran: bool
    limit_check: LimitCheckResult | None

    @property
    def recompiled(self) -> bool:
        """``True`` if `limit_check` found something exceeded (and
        `ensure_compiled_limits` therefore recompiled hypoDD/ph2dt)."""
        return self.limit_check is not None and self.limit_check.any_exceeded


def _nlloc_grid_paths(nlloc_root: Path | str) -> tuple[Path, Path | None]:
    """``<nlloc_root>.P.mod.hdr`` (always) and ``.S.mod.hdr`` (only if it
    exists) -- `openmeq.nlloc_setup`'s own 3D grid convention.
    """
    root = str(nlloc_root)
    p_hdr = Path(f"{root}.P.mod.hdr")
    s_hdr = Path(f"{root}.S.mod.hdr")
    return p_hdr, (s_hdr if s_hdr.is_file() else None)


def _run_nlloc_to_simul(cfg: Config, external_hypodd_dir: Path) -> None:
    """``--nlloc-to-simul``: convert the configured NLLoc 3D grid
    (``cfg.nlloc.velocity_model.file``) into hypoDD's SIMUL2000 ``vel3D.dat``
    format, writing it to ``cfg.hypodd.velocity_model.file`` (M9.4 /
    OpenMEQ-6ld.4's `nlloc_grid_to_vel3d`, from M9.5's own notes: "source
    --nlloc-to-simul inputs from cfg.hypodd.velocity_model").

    Raises
    ------
    openmeq.config.ConfigError
        If `cfg.hypodd.velocity_model`'s 3D fields aren't set, the NLLoc
        grid's ``.P.mod.hdr`` is missing, or an S grid is missing while
        ``simul2000.iphase != 1`` (S velocities are required).
    """
    vm = cfg.hypodd.velocity_model
    if vm.origin_3d is None or vm.rotation_3d_deg is None or vm.simul2000 is None:
        raise _ConfigError(
            "--nlloc-to-simul requires cfg.hypodd.velocity_model.origin_3d, "
            ".rotation_3d_deg and .simul2000 to all be set (the LAT_3D/LON_3D/ROT_3D and "
            f"IMOD=9 ray-tracing parameters the converted vel3D.dat is registered against) -- "
            f"got origin_3d={vm.origin_3d!r}, rotation_3d_deg={vm.rotation_3d_deg!r}, "
            f"simul2000={vm.simul2000!r}"
        )

    nlloc_root = cfg.nlloc.velocity_model.file
    p_hdr, s_hdr = _nlloc_grid_paths(nlloc_root)
    expected_s_hdr = Path(f"{nlloc_root}.S.mod.hdr")
    if not p_hdr.is_file():
        raise _ConfigError(
            f"--nlloc-to-simul: NLLoc P-velocity grid header not found at {p_hdr} -- "
            f"cfg.nlloc.velocity_model.file ({nlloc_root!r}) must be a 3D grid root name with "
            "'<root>.P.mod.hdr'/'.buf' already present (run scripts/setup_nlloc.py first, and "
            "check cfg.nlloc.velocity_model.type == '3d')."
        )
    p_grid = read_nlloc_grid(p_hdr)
    s_grid = read_nlloc_grid(s_hdr) if s_hdr is not None else None

    if s_grid is None and vm.simul2000.iphase != 1:
        raise _ConfigError(
            f"--nlloc-to-simul: no NLLoc S-velocity grid found (expected {expected_s_hdr}), but "
            f"cfg.hypodd.velocity_model.simul2000.iphase={vm.simul2000.iphase} requires S "
            "velocities/a Vp-Vs ratio (IPHA=1 is P-only) -- either supply an NLLoc S grid "
            "alongside the P grid, or set simul2000.iphase to 1."
        )

    max_nx, max_ny, max_nz = compiled_vel3d_node_limits(
        include_dir=Path(external_hypodd_dir) / "include"
    )

    lat_3d, lon_3d = vm.origin_3d
    output_path = Path(vm.file)
    nlloc_grid_to_vel3d(
        p_grid,
        output_path,
        lat_3d,
        lon_3d,
        vm.rotation_3d_deg,
        s_grid=s_grid,
        max_nx=max_nx,
        max_ny=max_ny,
        max_nz=max_nz,
    )
    logger.info(
        "nlloc_to_simul: wrote %s from NLLoc grid %s (%s), compiled node limits nx=%d ny=%d nz=%d.",
        output_path,
        p_hdr,
        f"with S grid {s_hdr}" if s_grid is not None else "P-only, no S grid",
        max_nx,
        max_ny,
        max_nz,
    )


def _dependency_paths(cfg: Config, *, is_3d: bool, nlloc_to_simul: bool) -> list[Path]:
    """Input file(s) `setup_hypodd`'s own output depends on, for
    `openmeq.manifest.fingerprint` -- so touching any of them (without
    changing `cfg` itself) still triggers a rebuild.
    """
    vm = cfg.hypodd.velocity_model
    if is_3d and nlloc_to_simul:
        root = cfg.nlloc.velocity_model.file
        p_hdr, s_hdr = _nlloc_grid_paths(root)
        deps = [p_hdr, p_hdr.with_suffix(".buf")]
        if s_hdr is not None:
            deps += [s_hdr, s_hdr.with_suffix(".buf")]
        return deps
    return [Path(vm.file)]


def setup_hypodd(
    cfg: Config,
    paths: RunPaths,
    *,
    force: bool = False,
    nlloc_to_simul: bool = False,
    external_hypodd_dir: Path | str | None = None,
    external_bin_dir: Path | str | None = None,
) -> SetupHypoddResult:
    """Run the full hypoDD/ph2dt setup sequence, skipping if nothing
    relevant changed since the last run.

    Sequence (docs/ARCHITECTURE.md §6c-2):

    1. If `nlloc_to_simul`, convert the configured NLLoc 3D grid into
       ``vel3D.dat`` (`_run_nlloc_to_simul`).
    2. Refresh ``relocate/work/event_id_map.csv``/``station_label_map.csv``
       (`openmeq.relocate.hypodd_files.build_event_id_map`/
       `build_station_label_map`) from whatever catalogue currently exists
       at `paths.catalog_mag_xml` -- an empty one if it doesn't yet.
    3. Write ``station.dat`` and ``ph2dt.inp``.
    4. Write ``hypoDD.inp`` -- ``IMOD=1`` (1D) or ``IMOD=9`` (3D) per
       ``cfg.hypodd.velocity_model.type``. For a 3D run, the ``vel3D.dat``
       named by ``cfg.hypodd.velocity_model.file`` must already exist on
       disk (from step 1, or user-supplied) -- checked explicitly, since
       `openmeq.relocate.hypodd_files.write_hypodd_inp_3d` itself only
       writes its *filename* into ``hypoDD.inp`` and never touches the
       file.
    5. If a catalogue is available, `openmeq.relocate.hypodd_limits.
       ensure_compiled_limits` (check, recompile only if exceeded). If not,
       log a ``WARNING`` and skip -- never a hard failure (PLAN.md §9: this
       script must be runnable before any events exist).
    6. For a 3D run, `openmeq.relocate.hypodd_limits.check_vel3d_coverage`
       immediately after, regardless of catalogue availability (station
       registration is worth checking from day one).

    Fingerprinted (`openmeq.manifest.fingerprint`) on `cfg.hypodd`,
    `cfg.stations`, the merged station table's own contents, the velocity
    model input file(s) (the layer file for 1D; the NLLoc grid files for
    `nlloc_to_simul`, else the ``vel3D.dat`` itself), `paths.catalog_mag_xml`
    (so a newly-appeared or changed catalogue always re-triggers the limit
    check even if nothing else did), and `nlloc_to_simul` itself.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Rebuild even if the manifest says nothing changed.
    nlloc_to_simul : bool, keyword-only
        Run the NLLoc-to-SIMUL2000 converter first (only valid when
        ``cfg.hypodd.velocity_model.type == "3d"``).
    external_hypodd_dir, external_bin_dir : pathlib.Path or str or None, keyword-only
        Passed through to `ensure_compiled_limits`/`compiled_vel3d_node_limits`.
        Default to ``<external_dir()>/HYPODD``/``<external_dir()>/bin`` (the
        real, installed tree) -- tests should always pass a scratch
        directory instead (docs/ARCHITECTURE.md §6c: never the main
        checkout's tree).

    Returns
    -------
    SetupHypoddResult

    Raises
    ------
    openmeq.config.ConfigError
        On any hard failure: a missing/unreadable velocity model file, a
        3D config missing its ``origin_3d``/``rotation_3d_deg``/
        ``simul2000`` fields, a missing NLLoc grid for ``--nlloc-to-simul``,
        an unresolvable station-label collision, or a failed 3D coverage
        check.
    openmeq.external.ExternalToolError
        If a required recompile's ``make`` step fails.
    """
    vm = cfg.hypodd.velocity_model
    is_3d = vm.type == "3d"

    if nlloc_to_simul and not is_3d:
        raise _ConfigError(
            "--nlloc-to-simul only applies to a 3D hypoDD velocity model "
            f"(cfg.hypodd.velocity_model.type == '3d'), got "
            f"cfg.hypodd.velocity_model.type={vm.type!r}"
        )

    external_hypodd_dir = (
        Path(external_hypodd_dir) if external_hypodd_dir is not None else external_dir() / "HYPODD"
    )
    external_bin_dir = (
        Path(external_bin_dir) if external_bin_dir is not None else external_dir() / "bin"
    )

    station_table = load_station_table(cfg.stations)

    dep_paths = _dependency_paths(cfg, is_3d=is_3d, nlloc_to_simul=nlloc_to_simul)
    fp = fingerprint(
        cfg.hypodd,
        cfg.stations,
        station_table.to_dict("records"),
        dep_paths,
        paths.catalog_mag_xml,
        nlloc_to_simul,
    )
    manifest = StepManifest(paths, STEP_NAME)
    if not force and manifest.is_done(MANIFEST_KEY, fp):
        logger.info(
            "hypoDD setup unchanged since the last run -- skipped (use --force to rebuild)."
        )
        return SetupHypoddResult(
            rebuilt=False, imod=None, nlloc_to_simul_ran=False, limit_check=None
        )

    work_dir = paths.relocate_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    nlloc_to_simul_ran = False
    if nlloc_to_simul:
        _run_nlloc_to_simul(cfg, external_hypodd_dir)
        nlloc_to_simul_ran = True

    catalog_path = Path(paths.catalog_mag_xml)
    catalog_available = catalog_path.is_file()
    if catalog_available:
        catalog: Catalog = read_catalog(catalog_path)
        logger.info(
            "Loaded catalogue %s (%d events) for the event-id-map refresh and compiled-limit "
            "check.",
            catalog_path,
            len(catalog),
        )
    else:
        catalog = Catalog()
        logger.warning(
            "No catalogue found at %s yet -- this is expected before magnitude (M8) or review "
            "(M7) has run. Proceeding with an empty catalogue for the event-id-map refresh, and "
            "skipping the compiled-limit check (it will run on the next invocation once a "
            "catalogue exists).",
            catalog_path,
        )

    event_id_map_path = work_dir / "event_id_map.csv"
    station_label_map_path = work_dir / "station_label_map.csv"
    event_id_map = build_event_id_map(catalog, event_id_map_path)
    label_map = build_station_label_map(station_table, station_label_map_path)
    logger.info(
        "Refreshed %s (%d events) and %s (%d stations).",
        event_id_map_path,
        len(event_id_map),
        station_label_map_path,
        len(label_map),
    )

    write_station_dat(work_dir / "station.dat", station_table, label_map, cfg)
    write_ph2dt_inp(work_dir / "ph2dt.inp", cfg)
    logger.info("Wrote %s and %s.", work_dir / "station.dat", work_dir / "ph2dt.inp")

    vel3d_grid = None
    hypodd_inp_path = work_dir / "hypoDD.inp"
    if is_3d:
        vel3d_path = Path(vm.file)
        if not vel3d_path.is_file():
            raise _ConfigError(
                f"cfg.hypodd.velocity_model.file={vel3d_path} (the SIMUL2000 vel3D.dat) does "
                "not exist -- supply one, or pass --nlloc-to-simul to build it from the "
                "configured NLLoc 3D grid (cfg.nlloc.velocity_model.file)."
            )
        write_hypodd_inp_3d(hypodd_inp_path, cfg, cfg.hypodd.weighting_rows, vel3d_path)
        imod = 9
        vel3d_grid = read_vel3d_header(vel3d_path)
        logger.info("Wrote %s (IMOD=9, 3D SIMUL2000 model %s).", hypodd_inp_path, vel3d_path)
    else:
        try:
            model = read_layers(vm.file)
        except OSError as exc:
            raise _ConfigError(
                f"cfg.hypodd.velocity_model.file={vm.file} (the 1D layer file) could not be "
                f"read: {exc}"
            ) from exc
        write_hypodd_inp_1d(hypodd_inp_path, cfg, cfg.hypodd.weighting_rows, model)
        imod = 1
        logger.info("Wrote %s (IMOD=1, 1D model %s).", hypodd_inp_path, vm.file)

    limit_check: LimitCheckResult | None = None
    if catalog_available:
        limit_check = ensure_compiled_limits(
            cfg,
            catalog,
            station_table,
            external_hypodd_dir,
            external_bin_dir,
            vel3d_grid=vel3d_grid,
            log=logger,
        )
        if limit_check.any_exceeded:
            logger.warning(
                "Compiled hypoDD/ph2dt limits were exceeded and hypoDD/ph2dt have been "
                "recompiled with headroom: %s",
                ", ".join(c.name for c in limit_check.exceeded_checks()),
            )
        else:
            logger.info(
                "Compiled hypoDD/ph2dt limits OK (%d checks, none exceeded).",
                len(limit_check.checks),
            )

    if is_3d:
        lat_3d, lon_3d = vm.origin_3d
        check_vel3d_coverage(vel3d_grid, lat_3d, lon_3d, vm.rotation_3d_deg, catalog, station_table)
        logger.info(
            "3D model (vel3D.dat) coverage check passed for %d station(s)%s.",
            len(station_table),
            f" and {len(catalog)} catalogued event(s)" if catalog_available else "",
        )

    manifest.mark_done(MANIFEST_KEY, fp)
    return SetupHypoddResult(
        rebuilt=True,
        imod=imod,
        nlloc_to_simul_ran=nlloc_to_simul_ran,
        limit_check=limit_check,
    )
