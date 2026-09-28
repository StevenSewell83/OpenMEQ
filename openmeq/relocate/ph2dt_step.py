"""``ph2dt`` run: ``dt.ct`` and the event-pair list (M10.2 / OpenMEQ-16d.2).

Plan key: M10.2 (PLAN.md §6 Step 8's "write ``phase.dat`` (with magnitudes)
from the catalogue, then run ``ph2dt`` -> ``dt.ct``"). This module is the
thin orchestration step that:

1. Reuses ``openmeq.relocate.setup.setup_hypodd`` (M9.5) to (re)write
   ``station.dat``/``ph2dt.inp``/``hypoDD.inp`` and refresh the persisted
   ``event_id_map.csv``/``station_label_map.csv`` (M9.2, §6c).
2. Writes ``phase.dat``/``event.dat`` from ``magnitude/catalog_mag.xml``'s
   *accepted* events and non-rejected picks (M9.2's
   ``openmeq.relocate.hypodd_files.write_phase_dat``/``write_event_dat`` --
   this module's own job per the M9.5 note: "M9.5 ... writes station.dat,
   ph2dt.inp and hypoDD.inp plus the event/station maps, but not
   phase.dat or event.dat; this issue writes those from the catalogue").
3. Runs the real ``ph2dt`` binary (``openmeq.external``) in
   ``relocate/work/`` (hypoDD's own convention: every input/output file
   ``hypoDD.inp`` references is a bare, cwd-relative filename, §6c).
4. Parses the resulting ``dt.ct`` into the event-pair list M10.3's
   cross-correlation will consume -- see ``docs/ARCHITECTURE.md`` §3 for
   the exact ``dt.ct`` format and this list's contract.

Per ``docs/hypoddpy_audit.md`` §6 defect #6 ("never trust a stale on-disk
cache"): this module always re-derives ``phase.dat``/``event.dat`` from the
*current* catalogue and gates re-running ``ph2dt`` itself on a
``openmeq.manifest.StepManifest`` fingerprint that includes
``magnitude/catalog_mag.xml``'s own mtime/size, not bare file existence --
a changed catalogue always re-triggers a rerun, even if ``dt.ct`` happens to
already exist on disk from a previous run.

Does not fall back to ``locate/catalog_auto.xml`` if
``magnitude/catalog_mag.xml`` is missing (see ``run_ph2dt``'s docstring):
the pipeline's own dependency order (M10 depends on M8.3) guarantees it
should exist by the time this step runs, and magnitudes are supposed to
"appear in hypoDD.reloc" (PLAN.md §6 Step 7) -- silently substituting an
un-magnitude'd catalogue would break that guarantee without any warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openmeq.catalog import accepted, read_catalog
from openmeq.config import Config
from openmeq.config import ConfigError as _ConfigError
from openmeq.external import bin_path, run
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.relocate.hypodd_files import (
    build_event_id_map,
    build_station_label_map,
    write_event_dat,
    write_phase_dat,
)
from openmeq.relocate.setup import setup_hypodd
from openmeq.stations import station_table as load_station_table

if TYPE_CHECKING:
    from openmeq.paths import RunPaths

logger = get_logger("relocate.ph2dt")

#: ``openmeq.manifest.StepManifest`` step name -- shares ``relocate/manifest.json``
#: with ``openmeq.relocate.setup``'s own ``"hypodd_setup"`` key (the manifest
#: contract is per-key, not per-file, ARCHITECTURE.md §6c-2).
STEP_NAME = "relocate"
MANIFEST_KEY = "ph2dt"


def parse_dt_ct(path: Path | str) -> list[tuple[int, int]]:
    """Parse ``dt.ct``'s ``#``-prefixed header lines into the event-pair list.

    ``dt.ct``'s format (confirmed against ``ph2dt.f`` and
    ``examples/example1/dt.ct``, ARCHITECTURE.md §3)::

        # CUSPID1 CUSPID2
        STA  TT1  TT2  WGHT  PHA
        ...

    A rewritten version of the hypoddpy template's ``get_event_id_pairs``
    pattern (confirmed correct in ``docs/hypoddpy_audit.md`` §3/§6): only
    the header lines matter here, since ``ph2dt`` has already applied its
    own ``MAXSEP``/``MAXNGH``/``MINLNK``/etc. filtering by the time
    ``dt.ct`` exists.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a ``dt.ct``-format file.

    Returns
    -------
    list[tuple[int, int]]
        ``(cuspid1, cuspid2)`` for every ``#`` header line, in file order
        (not deduplicated or sorted -- exactly ``ph2dt``'s own order).

    Raises
    ------
    openmeq.config.ConfigError
        If a ``#`` header line doesn't have exactly the two expected
        integer fields.
    """
    path = Path(path)
    pairs: list[tuple[int, int]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.startswith("#"):
                continue
            fields = line.split()[1:]
            if len(fields) != 2:
                raise _ConfigError(
                    f"{path}:{lineno}: malformed dt.ct header line (expected '# CUSPID1 "
                    f"CUSPID2'), got {line!r}"
                )
            try:
                pairs.append((int(fields[0]), int(fields[1])))
            except ValueError as exc:
                raise _ConfigError(
                    f"{path}:{lineno}: malformed dt.ct header line (non-integer cuspid), "
                    f"got {line!r}"
                ) from exc
    return pairs


def run_ph2dt(
    cfg: Config,
    paths: RunPaths,
    *,
    force: bool = False,
    external_hypodd_dir: Path | str | None = None,
    external_bin_dir: Path | str | None = None,
) -> list[tuple[int, int]]:
    """Write ``phase.dat``/``event.dat``, run ``ph2dt``, and return the
    event-pair list parsed from the resulting ``dt.ct``.

    Sequence (ARCHITECTURE.md §3, §6c):

    1. Require ``paths.catalog_mag_xml`` to exist -- raises `ConfigError`
       otherwise (never a silent fallback to ``locate/catalog_auto.xml``,
       see the module docstring).
    2. `openmeq.relocate.setup.setup_hypodd` (M9.5): (re)writes
       ``station.dat``/``ph2dt.inp``/``hypoDD.inp`` and refreshes
       ``relocate/work/event_id_map.csv``/``station_label_map.csv`` (its
       own skip-unchanged manifest key, ``"hypodd_setup"``, is independent
       of this function's own ``"ph2dt"`` key below).
    3. Skip re-running ``ph2dt`` (but still return the event-pair list,
       reparsed from the existing ``dt.ct``) if nothing this step depends
       on (``cfg.hypodd.ph2dt``, the catalogue file, the station table)
       has changed since the last run and ``dt.ct`` still exists --
       `openmeq.manifest.StepManifest`, key ``"ph2dt"``.
    4. `openmeq.relocate.hypodd_files.build_event_id_map`/
       `build_station_label_map` (idempotent re-reads of the maps step 2
       just refreshed) then `write_phase_dat`/`write_event_dat` from
       `magnitude/catalog_mag.xml`'s accepted events -- non-rejected picks
       only (ARCHITECTURE.md §5/§6c).
    5. Run the real ``ph2dt`` binary (`openmeq.external.bin_path`/`run`)
       with `paths.relocate_work_dir` as ``cwd`` (hypoDD's own convention:
       every file ``hypoDD.inp``/``ph2dt.inp`` reference is a bare,
       cwd-relative filename).
    6. `parse_dt_ct` the resulting ``dt.ct`` into the returned event-pair
       list, and log a one-line summary (accepted event/station/pair
       counts) via `openmeq.log`; ``ph2dt``'s own stdout (which already
       reports per-run counts, PLAN.md §6) is captured and logged by
       `openmeq.external.run` itself.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Rerun ``ph2dt`` even if the manifest says nothing changed.
    external_hypodd_dir, external_bin_dir : pathlib.Path or str or None, keyword-only
        Forwarded to `setup_hypodd`'s own compiled-limit check (see its
        docstring). Tests should always pass a scratch directory rather
        than relying on the real installed tree (ARCHITECTURE.md §6c).

    Returns
    -------
    list[tuple[int, int]]
        ``(cuspid1, cuspid2)`` pairs parsed from ``dt.ct``, in file order --
        see ``docs/ARCHITECTURE.md`` §3 for the exact contract.

    Raises
    ------
    openmeq.config.ConfigError
        If ``paths.catalog_mag_xml`` doesn't exist, or any of the
        underlying writers/`setup_hypodd` raise (e.g. an unresolvable
        station-label collision).
    openmeq.external.ExternalToolMissing
        If the ``ph2dt`` binary can't be found.
    openmeq.external.ExternalToolError
        If ``ph2dt`` exits non-zero.
    """
    catalog_path = Path(paths.catalog_mag_xml)
    if not catalog_path.is_file():
        raise _ConfigError(
            f"{catalog_path} does not exist -- ph2dt needs magnitude/catalog_mag.xml (M8.3's "
            "output) as its input catalogue. OpenMEQ's pipeline dependency order guarantees "
            "this file exists by the time relocate (M10) runs; this step never silently falls "
            "back to locate/catalog_auto.xml, since magnitudes are supposed to appear in "
            "hypoDD.reloc (PLAN.md §6 Step 7) -- run `openmeq magnitude` first."
        )

    catalog = read_catalog(catalog_path)
    station_table = load_station_table(cfg.stations)

    work_dir = paths.relocate_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # M9.5: station.dat / ph2dt.inp / hypoDD.inp + the persisted id/label
    # maps -- has its own skip-unchanged key, distinct from this function's.
    setup_hypodd(
        cfg,
        paths,
        force=force,
        external_hypodd_dir=external_hypodd_dir,
        external_bin_dir=external_bin_dir,
    )

    event_id_map = build_event_id_map(catalog, work_dir / "event_id_map.csv")
    label_map = build_station_label_map(station_table, work_dir / "station_label_map.csv")

    dt_ct_path = work_dir / "dt.ct"
    fp = fingerprint(cfg.hypodd.ph2dt, catalog_path, station_table.to_dict("records"))
    manifest = StepManifest(paths, STEP_NAME)
    accepted_events = accepted(catalog)

    if not force and manifest.is_done(MANIFEST_KEY, fp) and dt_ct_path.is_file():
        logger.info(
            "ph2dt unchanged since the last run (catalogue/ph2dt config/station table all "
            "match the last recorded fingerprint) -- reusing %s instead of rerunning.",
            dt_ct_path,
        )
        return parse_dt_ct(dt_ct_path)

    write_phase_dat(work_dir / "phase.dat", catalog, event_id_map, label_map)
    write_event_dat(work_dir / "event.dat", catalog, event_id_map)
    logger.info(
        "Wrote %s and %s from %d accepted event(s) (%d total in %s).",
        work_dir / "phase.dat",
        work_dir / "event.dat",
        len(accepted_events),
        len(catalog),
        catalog_path,
    )

    ph2dt = bin_path("ph2dt")
    run([str(ph2dt), "ph2dt.inp"], cwd=work_dir, log=logger)

    if not dt_ct_path.is_file():
        logger.warning(
            "ph2dt reported success but did not produce %s -- treating this as zero event "
            "pairs (this can happen with fewer than 2 accepted events).",
            dt_ct_path,
        )
        pairs: list[tuple[int, int]] = []
    else:
        pairs = parse_dt_ct(dt_ct_path)

    logger.info(
        "ph2dt: %d accepted event(s), %d station(s) -> %d event pair(s) (%s).",
        len(accepted_events),
        len(label_map),
        len(pairs),
        dt_ct_path,
    )

    manifest.mark_done(MANIFEST_KEY, fp)
    return pairs
