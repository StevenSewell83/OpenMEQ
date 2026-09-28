"""hypoDD runner: invoke the real binary and read relocations into catalog_dd.xml.

Plan key: M10.4 / ``OpenMEQ-16d.4``. Runs the real ``hypoDD`` binary against
everything M9/M10.1-M10.3 prepared in ``relocate/work/`` (``station.dat``/
``event.sel``/``station.sel``/``hypoDD.inp``, ``dt.ct``, ``dt.cc``,
``relocate/work/event_id_map.csv``) and reads ``hypoDD.reloc`` back into a
new QuakeML catalogue (``relocate/catalog_dd.xml``), per
``docs/ARCHITECTURE.md`` §5/§6h.

Fixes the defects ``docs/hypoddpy_audit.md`` §4/§6 assign to this task
(master defect table, items carried "port-rewritten" into M10.4):

- The ``hypoDD.reloc`` reader is rebuilt on the persisted, append-only
  ``event_id_map.csv`` (``openmeq.relocate.hypodd_files.build_event_id_map``/
  ``invert_event_id_map``) -- never on ObsPy's global ``ResourceIdentifier``
  registry (the hypoddpy template's fragile ``getReferredObject()`` pattern,
  which cannot survive OpenMEQ's own step-boundary process split: the
  catalogue is written, a separate ``hypoDD`` subprocess runs, and *this*
  process reads the catalogue back fresh -- there is no shared in-memory
  registry to resolve against).
- Every one of the 24 fields is extracted (:class:`RelocatedOrigin`),
  including ``MAG``/``NCCP``/``NCCS``/``NCTP``/``NCTS``/``RCC``/``RCT``,
  which the hypoddpy template discarded.
- The seconds-rounding-to-60 quirk (``hypoDD.f``'s ``f6.3`` ``SEC`` format
  can round up to ``"60.000"``) is handled by construction, and with *full*
  sub-second precision kept (an improvement over the hypoddpy template's
  own fix, which truncated to whole seconds) -- see :func:`_reloc_time`.

Also implements the two M9/M10 notes appended to this issue:

- **M9.3**: :func:`run_hypodd` re-runs
  ``openmeq.relocate.hypodd_limits.ensure_compiled_limits`` with the real
  ``xcorr_obs_count`` counted from ``dt.cc`` (which did not exist yet the
  first time M9.5's ``setup_hypodd`` ran this same check).
  ``xcorr_obs_count`` is *always available* -- it's re-run before every
  relocation, not just the first time.
- **6ld.6**: :func:`run_hypodd` calls
  ``openmeq.relocate.hypodd_limits.check_vel3d_coverage`` before every 3D
  (``IMOD=9``) run, including each cluster of the optional
  ``cfg.hypodd.parallel_clusters`` path (see below).

Pipeline shape
--------------
::

    relocate/work/{station.dat, ph2dt.inp, hypoDD.inp}   (M9.5, setup_hypodd)
    relocate/work/{event.sel, station.sel, dt.ct}         (M10.2, run_ph2dt)
    relocate/work/dt.cc            (M10.3, run_xcorr; optional -- IDAT=2 doesn't need it)
    relocate/work/event_id_map.csv                        (M9.2, persisted)
        -> invoke_hypodd (this module: runs the real `hypoDD` binary,
                           cwd=relocate/work/, per this task's own contract)
        -> relocate/work/hypoDD.reloc
        -> read_reloc (this module: the 24-field parser)
        -> merge_relocations (this module: into magnitude/catalog_mag.xml's
                               events, carrying magnitudes over unchanged)
        -> relocate/catalog_dd.xml

Origin uncertainty mapping (a genuine design decision, per the issue's own
notes -- documented here and in ``docs/ARCHITECTURE.md`` §6h since it
establishes this project's convention for hypoDD-sourced origin
uncertainties)
--------------------------------------------------------------------------
``hypoDD.reloc``'s ``EX``/``EY``/``EZ`` are 1-sigma formal errors (metres)
in the *cluster's local Cartesian frame* (``setorg.f``'s projection,
rotation depending on the run's own ``LAT_3D``/``LON_3D``/``ROT_3D`` for a
3D run, or an internal centroid-based frame for 1D) -- not directly
north/east/depth-aligned uncertainties in general. Rather than a
covariance-respecting full ellipse treatment (not warranted given the
formal errors alone don't carry the frame's rotation), this module uses the
simplest defensible mapping:

- ``Origin.origin_uncertainty.horizontal_uncertainty = sqrt(EX**2 + EY**2)``
  (metres, QuakeML's own convention for this field) -- a conservative
  combination of the two horizontal formal errors, not a true confidence
  ellipse semi-axis.
- ``Origin.depth_errors.uncertainty = abs(EZ)`` (metres).

Cluster parallelism (``cfg.hypodd.parallel_clusters``, default ``False``,
opt-in/experimental per the issue)
------------------------------------------------------------------------
hypoDD's own event clustering (``OBSCC``/``OBSCT`` in ``hypoDD.inp``,
``src/hypoDD/cluster1.f``, called from ``hypoDD.f`` right before its "big
loop over clusters" -- confirmed directly against the installed source,
*not* documented anywhere as a ``ph2dt`` feature) is what actually forms
hypoDD's own clusters, not ``ph2dt`` (``ph2dt.log`` has no clustering output
at all -- confirmed empirically against ``external/HYPODD/examples/example1``;
this corrects the issue text's "from ph2dt's own clustering, surfaced via
its log/output" the same way ``docs/hypoddpy_audit.md`` §7 corrected the M9
epic note's IAQ claim).

Rather than reimplementing ``cluster1.f``'s union-find algorithm in Python
(a real correctness risk: hypoDD's own internal cluster *numbering* -- which
``CID`` value a valid subprocess run must be given to select a specific
cluster -- depends on that algorithm's exact node-processing order, which a
from-scratch Python port could subtly disagree with), this module instead
lets the real ``hypoDD`` binary determine its own cluster count cheaply:
:func:`_probe_cluster_count` runs ``hypoDD`` once, in a scratch copy of the
run directory, with ``hypoDD.inp``'s ``CID`` line rewritten to a
deliberately out-of-range sentinel. hypoDD's own clustering *always* runs
first (before the expensive per-cluster relocation solve), so this reaches
hypoDD's own ``"invalid cluster number ... must be between 1 and nclust
(N)"`` abort (confirmed, empirically, to still exit code ``0`` -- a bare
Fortran ``STOP`` -- so this parses stdout, never the exit code) cheaply,
without paying for any cluster's actual relocation.

Given ``nclust`` from the probe, this module runs ``hypoDD`` once per
cluster (``CID=1..nclust``, via :func:`openmeq.parallel.pmap`), each in its
own scratch subdirectory but against the *same*, full, unfiltered
``event.sel``/``station.sel``/``dt.ct``/``dt.cc`` (never a Python-filtered
per-cluster subset) -- since hypoDD's own clustering is deterministic given
identical inputs, every subprocess independently (and cheaply) recomputes
the *same* clustering and then only relocates its own assigned cluster.
This sidesteps needing to replicate hypoDD's internal numbering at all:
each subprocess asks hypoDD itself "what is your own cluster N", for every
``N`` from 1 to the real ``nclust`` -- correctness follows directly from
hypoDD's own determinism, not from this module's bookkeeping.

**Proof of consistency** (the issue's own bar for shipping this
experimental path): ``tests/test_relocate_hypodd_runner.py``'s
``test_parallel_clusters_matches_single_run`` reproduces exactly this
against the real ``hypoDD`` binary and ``external/HYPODD/examples/example1``'s
own bundled ``dt.ct``/``dt.cc``/``event.sel``/``station.sel`` (with
``OBSCT`` raised from 8 to 60, which splits example1's 16 events into 3
real clusters of sizes 6/4/2 plus 4 isolated events -- confirmed empirically
during this task): the merged per-cluster relocations are *bit-identical*
(not merely "close") to a single ``CID=0`` run over the same inputs, since
each cluster's relocation is an independent sub-problem in hypoDD's own
algorithm either way.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from obspy import Catalog, UTCDateTime
from obspy.core.event import Origin, OriginUncertainty
from obspy.core.event.base import QuantityError, ResourceIdentifier

from openmeq.catalog import accepted, event_id, read_catalog, set_extra, write_catalog
from openmeq.config import ConfigError as _ConfigError
from openmeq.external import ExternalToolError, bin_path, external_dir
from openmeq.external import run as _run_external
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap
from openmeq.relocate.hypodd_files import build_event_id_map, invert_event_id_map, read_vel3d_header
from openmeq.relocate.hypodd_limits import (
    LimitCheckResult,
    check_vel3d_coverage,
    ensure_compiled_limits,
)
from openmeq.stations import station_table as load_station_table

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate.hypodd_runner")

#: ARCHITECTURE.md §5's Origin.method_id for every relocated origin this
#: module writes.
METHOD_ID = "smi:openmeq/hypodd"

#: openmeq.manifest.StepManifest step/key -- shares relocate/manifest.json
#: with M9.5's "hypodd_setup" and M10.2's "ph2dt" keys (per-key contract,
#: ARCHITECTURE.md §6c-2).
STEP_NAME = "relocate"
MANIFEST_KEY = "hypodd_run"

#: openmeq.catalog.set_extra keys this module writes (ARCHITECTURE.md §6h).
EXTRA_STATUS = "hypodd_status"
STATUS_RELOCATED = "relocated"
STATUS_NOT_LINKED = "not_linked"

_RELOC_N_FIELDS = 24

#: hypoDD.inp's CID line marker -- see openmeq.relocate.hypodd_files.
#: _write_hypodd_inp_footer, which always writes exactly this comment line
#: immediately before a bare "<CID>\n" line. Cluster parallelism (see the
#: module docstring) rewrites that one line; it therefore only supports a
#: hypoDD.inp written by this project's own writer, not an arbitrary/
#: hand-edited one -- documented limitation, since cfg.hypodd.parallel_clusters
#: is opt-in/experimental.
_CID_COMMENT_LINE = "* event selection: CID (0 = all)\n"

#: Deliberately out-of-range CID used to probe hypoDD's own cluster count
#: (module docstring, "Cluster parallelism").
_INVALID_CID_SENTINEL = 999_999_999

_CLUSTER_COUNT_RE = re.compile(r"#\s*clusters:\s*(\d+)")


class HypoddRunFailed(RuntimeError):
    """Raised when the real ``hypoDD`` binary fails outright (non-zero exit,
    or a missing ``hypoDD.reloc`` despite a zero exit), or when cluster
    parallelism's own bookkeeping is violated. Always includes
    ``hypoDD.log``'s own tail in the message, per the issue's own pitfall
    ("surface hypoDD.log's content in the raised error for debuggability").
    """


# ---------------------------------------------------------------------------
# RelocatedOrigin + read_reloc: the 24-field hypoDD.reloc parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelocatedOrigin:
    """One parsed ``hypoDD.reloc`` line (hypoDD.f:1283-1298's 24-field
    write): ``CUSPID LAT LON DEPTH_KM X_M Y_M Z_M EX_M EY_M EZ_M YR MO DY HR
    MI SEC MAG NCCP NCCS NCTP NCTS RCC RCT CID``.

    ``x_m``/``y_m``/``z_m`` are Cartesian coordinates relative to the
    cluster centroid (informative -- M10.5's before/after plots can use them
    directly); ``lat``/``lon``/``depth_km`` are the authoritative absolute
    location (ARCHITECTURE.md §6h).
    """

    cuspid: int
    lat: float
    lon: float
    depth_km: float
    x_m: float
    y_m: float
    z_m: float
    ex_m: float
    ey_m: float
    ez_m: float
    time: UTCDateTime
    mag: float
    nccp: int
    nccs: int
    nctp: int
    ncts: int
    rcc: float
    rct: float
    cid: int


def _reloc_time(yr: int, mo: int, dy: int, hr: int, mi: int, sec: float) -> UTCDateTime:
    """Build the origin time from hypoDD.reloc's unpacked ``YR MO DY HR MI
    SEC`` fields, handling the documented ``SEC`` rounding-to-``60.000``
    quirk (hypoDD.f's ``f6.3`` write format; docs/hypoddpy_audit.md §4,
    "the seconds-rounding-to-60 workaround ... a correct, worth-keeping
    defensive fix").

    Improves on the hypoddpy template's own fix (which truncated to whole
    seconds before checking the ``>= 60`` condition, discarding sub-second
    precision): building the "hh:mm:00" instant first and then adding the
    *full* ``sec`` float lets ``UTCDateTime``'s own arithmetic carry any
    overflow (``sec >= 60.0``) into the next minute correctly, with no
    precision lost and no manual carry logic needed at all.
    """
    return UTCDateTime(yr, mo, dy, hr, mi, 0) + sec


def read_reloc(path: Path | str, event_id_map: dict[str, int]) -> dict[str, RelocatedOrigin]:
    """Parse a ``hypoDD.reloc`` file into ``{event_id: RelocatedOrigin}``.

    Looks up each line's ``CUSPID`` via the *persisted* ``event_id_map``
    (``{event_id: cuspid}``, `openmeq.relocate.hypodd_files.build_event_id_map`'s
    contract -- inverted here) rather than ObsPy's global resource-id
    registry (docs/hypoddpy_audit.md §4's confirmed fragile-pattern finding
    this task must not port).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a ``hypoDD.reloc``-format file (the final, 24-field output
        -- **not** a per-iteration ``mdat.reloc.NNN.NNN`` scratch file,
        18 fields, this task's own documented pitfall).
    event_id_map : dict[str, int]
        ``{event_id: cuspid}``, as persisted in
        ``relocate/work/event_id_map.csv`` (`build_event_id_map`).

    Returns
    -------
    dict[str, RelocatedOrigin]
        One entry per line, keyed by ``event_id``.

    Raises
    ------
    openmeq.config.ConfigError
        If a line doesn't have exactly 24 whitespace-separated fields, or
        if a line's ``CUSPID`` is not present in `event_id_map` (a stale/
        out-of-date map -- this task's own explicit "raise a clear error,
        not a silent skip" requirement).
    """
    path = Path(path)
    cuspid_to_event_id = invert_event_id_map(event_id_map)
    result: dict[str, RelocatedOrigin] = {}

    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != _RELOC_N_FIELDS:
                raise _ConfigError(
                    f"{path}:{lineno}: expected {_RELOC_N_FIELDS} whitespace-separated fields "
                    "(hypoDD.f:1283-1298's CUSPID LAT LON DEPTH_KM X_M Y_M Z_M EX_M EY_M EZ_M "
                    "YR MO DY HR MI SEC MAG NCCP NCCS NCTP NCTS RCC RCT CID layout -- confirm "
                    "this is hypoDD's final hypoDD.reloc, not a per-iteration mdat.reloc.NNN.NNN "
                    f"scratch file, which has only 18 fields), got {len(fields)}: {line!r}"
                )

            cuspid = int(fields[0])
            if cuspid not in cuspid_to_event_id:
                raise _ConfigError(
                    f"{path}:{lineno}: hypoDD.reloc references CUSPID {cuspid}, which is not in "
                    "the persisted event_id_map (relocate/work/event_id_map.csv) -- this should "
                    "never happen if the map is properly append-only/persisted "
                    "(openmeq.relocate.hypodd_files.build_event_id_map); a stale or hand-edited "
                    "map produces exactly this symptom. Refusing to silently skip this event."
                )
            eid = cuspid_to_event_id[cuspid]

            (
                _cuspid,
                lat,
                lon,
                depth_km,
                x_m,
                y_m,
                z_m,
                ex_m,
                ey_m,
                ez_m,
                yr,
                mo,
                dy,
                hr,
                mi,
                sec,
                mag,
                nccp,
                nccs,
                nctp,
                ncts,
                rcc,
                rct,
                cid,
            ) = fields

            result[eid] = RelocatedOrigin(
                cuspid=cuspid,
                lat=float(lat),
                lon=float(lon),
                depth_km=float(depth_km),
                x_m=float(x_m),
                y_m=float(y_m),
                z_m=float(z_m),
                ex_m=float(ex_m),
                ey_m=float(ey_m),
                ez_m=float(ez_m),
                time=_reloc_time(int(yr), int(mo), int(dy), int(hr), int(mi), float(sec)),
                mag=float(mag),
                nccp=int(nccp),
                nccs=int(nccs),
                nctp=int(nctp),
                ncts=int(ncts),
                rcc=float(rcc),
                rct=float(rct),
                cid=int(cid),
            )

    return result


# ---------------------------------------------------------------------------
# merge into catalog_dd.xml
# ---------------------------------------------------------------------------


def _build_relocated_origin(reloc: RelocatedOrigin) -> Origin:
    """One new `Origin` (ARCHITECTURE.md §5/§6h) from a parsed
    `RelocatedOrigin`. See the module docstring's "Origin uncertainty
    mapping" section for the ``EX``/``EY``/``EZ`` -> QuakeML fields
    convention.
    """
    origin = Origin(
        time=reloc.time,
        latitude=reloc.lat,
        longitude=reloc.lon,
        depth=reloc.depth_km * 1000.0,  # hypoDD.reloc's km -> QuakeML metres
        method_id=ResourceIdentifier(METHOD_ID),
    )
    origin.origin_uncertainty = OriginUncertainty(
        horizontal_uncertainty=math.hypot(reloc.ex_m, reloc.ey_m)
    )
    origin.depth_errors = QuantityError(uncertainty=abs(reloc.ez_m))

    # Per-event QC counts/diagnostics fields carried through (audit-fixed --
    # see the module docstring) but not turned into new Magnitude/Amplitude
    # objects: the issue is explicit that the event's *existing* magnitude
    # objects (magnitude/catalog_mag.xml) stay untouched. hypodd_mag is
    # hypoDD's own echoed-back magnitude value, kept only for M10.5's
    # diagnostics/QC cross-checking against the real (untouched) magnitude.
    set_extra(origin, "hypodd_mag", reloc.mag)
    set_extra(origin, "hypodd_nccp", reloc.nccp)
    set_extra(origin, "hypodd_nccs", reloc.nccs)
    set_extra(origin, "hypodd_nctp", reloc.nctp)
    set_extra(origin, "hypodd_ncts", reloc.ncts)
    set_extra(origin, "hypodd_rcc", reloc.rcc)
    set_extra(origin, "hypodd_rct", reloc.rct)
    set_extra(origin, "hypodd_cluster_id", reloc.cid)
    set_extra(origin, "hypodd_x_m", reloc.x_m)
    set_extra(origin, "hypodd_y_m", reloc.y_m)
    set_extra(origin, "hypodd_z_m", reloc.z_m)
    return origin


def merge_relocations(
    catalog: Catalog, reloc_by_event: dict[str, RelocatedOrigin]
) -> tuple[Catalog, int, int]:
    """Merge hypoDD's relocations into a catalogue, per ARCHITECTURE.md §5/§6h.

    For every event in `catalog`:

    - If its ``event_id`` is a key of `reloc_by_event`: append a new
      `Origin` (`_build_relocated_origin`, ``method_id`` ``smi:openmeq/hypodd``),
      make it the preferred origin, and keep every existing origin
      (nothing is edited in place, ARCHITECTURE.md §5) and every existing
      `Magnitude`/`StationMagnitude`/`Amplitude` untouched (the issue's own
      "carrying magnitudes over" requirement). Records
      ``openmeq.catalog.set_extra(event, "hypodd_status", "relocated")``.
    - Otherwise (an event ph2dt's own clustering/linking excluded, e.g. too
      few links, or a rejected event never sent to hypoDD at all): the
      event's prior preferred origin is left untouched, and
      ``set_extra(event, "hypodd_status", "not_linked")`` records that
      hypoDD relocation was attempted for this run but this particular
      event wasn't linked -- so downstream consumers/diagnostics (M10.5)
      can distinguish "relocated" from "not linked" without having to
      re-derive it themselves.

    Parameters
    ----------
    catalog : obspy.Catalog
        The full catalogue (``magnitude/catalog_mag.xml``) -- every event,
        including any already-rejected ones (which are always "not linked",
        since `openmeq.catalog.accepted` excludes them from ever reaching
        ``phase.dat``/``event.dat`` in the first place).
    reloc_by_event : dict[str, RelocatedOrigin]
        From `read_reloc`.

    Returns
    -------
    (obspy.Catalog, int, int)
        The merged catalogue (a new `Catalog`, same catalogue-level
        metadata as `catalog`), the number of events relocated, and the
        number of events left as "not linked".
    """
    out = Catalog(events=[], resource_id=catalog.resource_id, description=catalog.description)
    n_relocated = 0
    n_not_linked = 0

    for ev in catalog:
        eid = event_id(ev)
        reloc = reloc_by_event.get(eid)
        if reloc is not None:
            new_origin = _build_relocated_origin(reloc)
            ev.origins.append(new_origin)
            ev.preferred_origin_id = new_origin.resource_id
            set_extra(ev, EXTRA_STATUS, STATUS_RELOCATED)
            n_relocated += 1
        else:
            set_extra(ev, EXTRA_STATUS, STATUS_NOT_LINKED)
            n_not_linked += 1
        out.append(ev)

    return out, n_relocated, n_not_linked


# ---------------------------------------------------------------------------
# invoking the real hypoDD binary
# ---------------------------------------------------------------------------


def _hypodd_log_tail(work_dir: Path, max_chars: int = 4000) -> str:
    log_path = work_dir / "hypoDD.log"
    if not log_path.is_file():
        return "(no hypoDD.log found in the run directory)"
    text = log_path.read_text(errors="replace")
    return text[-max_chars:]


def invoke_hypodd(
    hypodd_bin: Path | str,
    work_dir: Path | str,
    inp_name: str = "hypoDD.inp",
    *,
    log=None,
) -> None:
    """Run the real ``hypoDD`` binary once, with `work_dir` as its working
    directory (this task's own contract: every file ``hypoDD.inp``
    references is a bare, cwd-relative filename, ARCHITECTURE.md §6c),
    and verify it actually produced output before returning.

    This is the one reusable primitive both `run_hypodd`'s single-run path
    and cluster parallelism's per-cluster subprocess calls use -- and the
    one a test can call directly against a hand-assembled work directory
    (e.g. ``external/HYPODD/examples/example1``'s own bundled files,
    without going through any of this project's own file writers at all).

    Parameters
    ----------
    hypodd_bin : pathlib.Path or str
        Path to the ``hypoDD`` executable (`openmeq.external.bin_path`).
    work_dir : pathlib.Path or str
        The run directory (``hypoDD.inp``, ``event.sel``, ``station.sel``,
        ``dt.ct``, and optionally ``dt.cc``/``vel3D.dat`` must already be
        there).
    inp_name : str, optional
        The control-file name to pass on hypoDD's command line (default
        ``"hypoDD.inp"``).
    log : logging.Logger or None, optional
        Forwarded to `openmeq.external.run`.

    Raises
    ------
    HypoddRunFailed
        If hypoDD exits non-zero, or exits 0 but leaves no ``hypoDD.reloc``
        at all -- both cases include ``hypoDD.log``'s own tail in the
        message (this task's own pitfall: "surface hypoDD.log's content in
        the raised error for debuggability").
    """
    work_dir = Path(work_dir)
    active_log = log if log is not None else logger
    try:
        _run_external([str(hypodd_bin), inp_name], cwd=work_dir, log=active_log)
    except ExternalToolError as exc:
        raise HypoddRunFailed(
            f"hypoDD exited with code {exc.returncode} in {work_dir} (running {inp_name}). "
            f"hypoDD.log tail:\n{_hypodd_log_tail(work_dir)}"
        ) from exc

    reloc_path = work_dir / "hypoDD.reloc"
    if not reloc_path.is_file():
        raise HypoddRunFailed(
            f"hypoDD exited 0 in {work_dir} but did not produce {reloc_path} at all -- this "
            "can happen on a data problem hypoDD itself treats as non-fatal (e.g. a singular "
            f"system) that still aborts before opening its own output file. hypoDD.log tail:\n"
            f"{_hypodd_log_tail(work_dir)}"
        )


def _count_dt_cc_observations(path: Path) -> int:
    """The real ``dt.cc`` observation count (M9.3's note: re-run
    ``check_compiled_limits``/``ensure_compiled_limits`` with this, now that
    ``dt.cc`` actually exists) -- every non-``#``, non-blank line."""
    if not Path(path).is_file():
        return 0
    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            n += 1
    return n


# ---------------------------------------------------------------------------
# cluster parallelism (cfg.hypodd.parallel_clusters, opt-in/experimental)
# ---------------------------------------------------------------------------


def _rewrite_cid_line(hypodd_inp_text: str, new_cid: int) -> str:
    """Rewrite ``hypoDD.inp``'s ``CID`` line (see `_CID_COMMENT_LINE`) to
    `new_cid`, keeping everything else byte-identical.

    Raises
    ------
    openmeq.config.ConfigError
        If `hypodd_inp_text` doesn't contain the exact marker line
        ``openmeq.relocate.hypodd_files._write_hypodd_inp_footer`` always
        writes -- cluster parallelism only supports a ``hypoDD.inp`` written
        by this project's own writer (module docstring).
    """
    lines = hypodd_inp_text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line == _CID_COMMENT_LINE:
            if i + 1 >= len(lines):
                raise _ConfigError(f"hypoDD.inp ends right after {_CID_COMMENT_LINE!r}")
            lines[i + 1] = f"{new_cid}\n"
            return "".join(lines)
    raise _ConfigError(
        f"hypoDD.inp does not contain the expected marker line {_CID_COMMENT_LINE!r} written by "
        "openmeq.relocate.hypodd_files -- cfg.hypodd.parallel_clusters requires a hypoDD.inp "
        "written by this project's own writer, not a hand-edited or externally-supplied one."
    )


def _copy_static_inputs(src_dir: Path, dst_dir: Path, vel3d_filename: str | None) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("event.sel", "station.sel", "dt.ct"):
        shutil.copy(src_dir / name, dst_dir / name)
    if (src_dir / "dt.cc").is_file():
        shutil.copy(src_dir / "dt.cc", dst_dir / "dt.cc")
    if vel3d_filename is not None and (src_dir / vel3d_filename).is_file():
        shutil.copy(src_dir / vel3d_filename, dst_dir / vel3d_filename)


def _probe_cluster_count(
    hypodd_bin: Path, work_dir: Path, base_inp_text: str, vel3d_filename: str | None, log
) -> int:
    """Learn hypoDD's own cluster count cheaply, without paying for any
    cluster's relocation. See the module docstring's "Cluster parallelism"
    section for the full rationale and the empirical confirmation this is
    based on.

    Returns ``1`` (a safe fallback -- `_run_hypodd_clustered` then falls
    back to a single ``CID=0`` run) if the probe's expected stdout marker
    isn't found, logged at ``WARNING`` (e.g. a different hypoDD build's
    message wording).
    """
    with tempfile.TemporaryDirectory(prefix="openmeq_hypodd_cluster_probe_") as tmp:
        probe_dir = Path(tmp)
        _copy_static_inputs(work_dir, probe_dir, vel3d_filename)
        (probe_dir / "hypoDD.inp").write_text(
            _rewrite_cid_line(base_inp_text, _INVALID_CID_SENTINEL)
        )
        try:
            proc = _run_external([str(hypodd_bin), "hypoDD.inp"], cwd=probe_dir, log=log)
            text = (proc.stdout or "") + (proc.stderr or "")
        except ExternalToolError as exc:
            text = (exc.stdout or "") + (exc.stderr or "")

    match = _CLUSTER_COUNT_RE.search(text)
    if match is None:
        log.warning(
            "parallel_clusters: could not find a '# clusters: N' line in the out-of-range-CID "
            "probe's output (this hypoDD build's log wording may differ from the one this "
            "feature was confirmed against) -- falling back to a single CID=0 run."
        )
        return 1
    return int(match.group(1))


def _run_cluster_item(item: tuple[str, str]) -> None:
    """`openmeq.parallel.pmap`-compatible, top-level, picklable per-cluster
    worker: runs `invoke_hypodd` in one cluster's own scratch directory."""
    hypodd_bin, cluster_dir = item
    invoke_hypodd(Path(hypodd_bin), Path(cluster_dir), log=get_logger("relocate.hypodd_runner"))


def _run_hypodd_clustered(
    cfg: Config,
    hypodd_bin: Path,
    work_dir: Path,
    vel3d_filename: str | None,
    vel3d_coverage_args: tuple | None,
    log,
) -> int:
    """``cfg.hypodd.parallel_clusters=True`` path (module docstring).

    Leaves the merged result at ``work_dir / "hypoDD.reloc"`` (so the rest
    of `run_hypodd` never needs two code paths past this point) and cleans
    up every per-cluster scratch subdirectory before returning.

    Parameters
    ----------
    vel3d_coverage_args : tuple or None
        ``(vel3d_grid, lat_3d, lon_3d, rot_3d, catalog, station_table)`` for
        `check_vel3d_coverage`, or ``None`` for a 1D run. Called once per
        cluster (6ld.6's note: "before every 3D run, including each
        cluster") -- every cluster subprocess sees the identical, full,
        unfiltered input set, so this is the same check every time; still
        called per cluster rather than once up front, both to satisfy the
        note literally and to stay correct if a future revision starts
        filtering per-cluster subsets (cheap either way -- pure Python, no
        subprocess).

    Returns
    -------
    int
        ``nclust`` actually used (``1`` means the probe found nothing to
        parallelize, and this ran a single plain ``CID=0`` relocation
        instead -- callers should treat that exactly like the non-parallel
        path).
    """
    hypodd_inp_path = work_dir / "hypoDD.inp"
    base_text = hypodd_inp_path.read_text()

    nclust = _probe_cluster_count(hypodd_bin, work_dir, base_text, vel3d_filename, log)
    if nclust <= 1:
        log.info(
            "parallel_clusters requested, but hypoDD's own clustering found only %d cluster -- "
            "running a single CID=0 relocation instead (no benefit to splitting).",
            nclust,
        )
        if vel3d_coverage_args is not None:
            check_vel3d_coverage(*vel3d_coverage_args)
        invoke_hypodd(hypodd_bin, work_dir, log=log)
        return nclust

    log.info(
        "parallel_clusters: hypoDD's own clustering found %d cluster(s) -- relocating each "
        "independently.",
        nclust,
    )
    cluster_dirs: list[Path] = []
    for cid in range(1, nclust + 1):
        if vel3d_coverage_args is not None:
            check_vel3d_coverage(*vel3d_coverage_args)
        cdir = work_dir / f"_cluster_{cid:04d}"
        if cdir.exists():
            shutil.rmtree(cdir)
        _copy_static_inputs(work_dir, cdir, vel3d_filename)
        (cdir / "hypoDD.inp").write_text(_rewrite_cid_line(base_text, cid))
        cluster_dirs.append(cdir)

    items = [(str(hypodd_bin), str(cdir)) for cdir in cluster_dirs]
    pmap(_run_cluster_item, items, cfg.parallel, desc="hypodd-clusters")

    # Merge every cluster's own hypoDD.reloc -- cluster membership is a
    # partition of the linked event set by construction (hypoDD's own
    # clustering), so every CUSPID must appear in at most one cluster's
    # output; a repeat indicates this probe/rewrite scheme has desynced
    # from hypoDD's own internal numbering (fail loudly, never silently
    # keep whichever copy happened to be written last).
    combined_lines: list[str] = []
    seen_cuspids: set[str] = set()
    for cdir in cluster_dirs:
        reloc_path = cdir / "hypoDD.reloc"
        if not reloc_path.is_file():
            continue
        for line in reloc_path.read_text().splitlines(keepends=True):
            if not line.strip():
                continue
            cuspid = line.split()[0]
            if cuspid in seen_cuspids:
                raise HypoddRunFailed(
                    f"cluster parallelism produced CUSPID {cuspid} in more than one cluster's "
                    "own hypoDD.reloc -- this should be impossible (hypoDD's own clustering is "
                    "a partition of the linked event set) and indicates the per-cluster CID "
                    "probe/rewrite scheme has desynced from hypoDD's own internal cluster "
                    "numbering. Set cfg.hypodd.parallel_clusters: false and use the single-run "
                    "path instead."
                )
            seen_cuspids.add(cuspid)
            combined_lines.append(line)

    (work_dir / "hypoDD.reloc").write_text("".join(combined_lines))
    for cdir in cluster_dirs:
        shutil.rmtree(cdir, ignore_errors=True)
    return nclust


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class HypoddRunResult:
    """What one `run_hypodd` call did.

    Attributes
    ----------
    rebuilt : bool
        ``True`` if hypoDD was actually (re)invoked; ``False`` if skipped as
        unchanged (`openmeq.manifest.StepManifest`, keyed on
        ``dt.ct``+``dt.cc``+``hypoDD.inp``'s own fingerprint -- the merge
        into ``catalog_dd.xml`` itself is always redone, even when
        `rebuilt` is ``False``, so a ``magnitude/catalog_mag.xml`` change
        alone -- e.g. a re-run magnitude step -- still refreshes
        ``catalog_dd.xml`` without needlessly re-invoking hypoDD).
    run_mode : str
        ``"single"``, ``"parallel_clusters"``, ``"single (parallel_clusters:
        only 1 cluster found)"``, or ``"skipped"``.
    n_relocated, n_not_linked : int
        Event counts from `merge_relocations`.
    nclust : int or None
        Cluster count actually used by the parallel path, or ``None`` for
        the single-run/skipped paths.
    limit_check : openmeq.relocate.hypodd_limits.LimitCheckResult or None
        The compiled-limit check's result (re-run with the real ``dt.cc``
        observation count, M9.3's note), or ``None`` if the run was
        skipped.
    catalog_dd_xml : pathlib.Path
        ``paths.catalog_dd_xml`` (always (re)written, even when `rebuilt`
        is ``False``).
    """

    rebuilt: bool
    run_mode: str
    n_relocated: int
    n_not_linked: int
    nclust: int | None
    limit_check: LimitCheckResult | None
    catalog_dd_xml: Path


def run_hypodd(
    cfg: Config,
    paths: RunPaths,
    *,
    force: bool = False,
    external_hypodd_dir: Path | str | None = None,
    external_bin_dir: Path | str | None = None,
) -> HypoddRunResult:
    """Run the real hypoDD binary (once, or once per cluster --
    ``cfg.hypodd.parallel_clusters``) and merge its relocations into
    ``relocate/catalog_dd.xml``.

    Sequence (module docstring; ARCHITECTURE.md §5/§6h):

    1. Require ``relocate/work/{hypoDD.inp, event.sel, station.sel, dt.ct}``
       and ``magnitude/catalog_mag.xml`` to already exist (M9.5/M10.2's
       output) -- never a silent fallback.
    2. Re-read the persisted ``event_id_map.csv`` (never regenerated ad
       hoc, ARCHITECTURE.md §6c).
    3. M9.3's note: re-run ``ensure_compiled_limits`` with the real
       ``dt.cc`` observation count.
    4. 6ld.6's note: for a 3D (``IMOD=9``) run, ``check_vel3d_coverage``
       before every invocation (including each cluster of the parallel
       path).
    5. Skip re-invoking hypoDD (but still re-derive ``catalog_dd.xml``)
       if ``dt.ct``+``dt.cc``+``hypoDD.inp`` are unchanged since the last
       run and ``hypoDD.reloc`` already exists (work item 8's
       ``StepManifest`` skip-unchanged).
    6. Invoke hypoDD (`invoke_hypodd`, or `_run_hypodd_clustered` if
       ``cfg.hypodd.parallel_clusters``).
    7. `read_reloc`, then `merge_relocations` against
       ``magnitude/catalog_mag.xml``, then write ``relocate/catalog_dd.xml``.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Re-invoke hypoDD even if the manifest says nothing changed.
    external_hypodd_dir, external_bin_dir : pathlib.Path or str or None, keyword-only
        Forwarded to `ensure_compiled_limits`. Default to
        ``<external_dir()>/HYPODD``/``<external_dir()>/bin`` (the real,
        installed tree); tests should always pass a scratch directory
        instead (ARCHITECTURE.md §6c).

    Returns
    -------
    HypoddRunResult

    Raises
    ------
    openmeq.config.ConfigError
        If required input files are missing, or a stale ``event_id_map``
        is detected while parsing ``hypoDD.reloc`` (`read_reloc`).
    HypoddRunFailed
        If hypoDD itself fails (non-zero exit, or a missing
        ``hypoDD.reloc``), or (parallel path) its own per-cluster
        bookkeeping is violated.

    Notes
    -----
    Deviation from the issue's own minimal ``run_hypodd(cfg, paths) -> None``
    sketch: this returns `HypoddRunResult` rather than ``None``, matching
    every other orchestration function in this package
    (``openmeq.relocate.setup.setup_hypodd``, ``openmeq.relocate.ph2dt_step.
    run_ph2dt``) -- M10.5's diagnostics/CLI step needs exactly this
    information (relocated/not-linked counts, whether a recompile happened,
    which run mode was used) and re-deriving it by re-parsing
    ``catalog_dd.xml`` afterward would be redundant and slower.
    """
    work_dir = paths.relocate_work_dir
    hypodd_inp_path = work_dir / "hypoDD.inp"
    dt_ct_path = Path(paths.dt_ct)
    dt_cc_path = Path(paths.dt_cc)

    for required in (hypodd_inp_path, work_dir / "event.sel", work_dir / "station.sel", dt_ct_path):
        if not required.is_file():
            raise _ConfigError(
                f"{required} does not exist -- run_hypodd needs setup_hypodd's and run_ph2dt's "
                "output (station.dat/ph2dt.inp/hypoDD.inp, event.sel/station.sel/dt.ct) to "
                "already exist in relocate/work/. Run openmeq.relocate.setup.setup_hypodd and "
                "openmeq.relocate.ph2dt_step.run_ph2dt (and, if used, "
                "openmeq.relocate.xcorr.run_xcorr) first."
            )

    catalog_path = Path(paths.catalog_mag_xml)
    if not catalog_path.is_file():
        raise _ConfigError(
            f"{catalog_path} does not exist -- run_hypodd needs magnitude/catalog_mag.xml (M8.3's "
            "output) to merge relocations into and to carry magnitudes over from. Run "
            "`openmeq magnitude` first."
        )

    external_hypodd_dir = (
        Path(external_hypodd_dir) if external_hypodd_dir is not None else external_dir() / "HYPODD"
    )
    external_bin_dir = (
        Path(external_bin_dir) if external_bin_dir is not None else external_dir() / "bin"
    )

    catalog = read_catalog(catalog_path)
    station_table = load_station_table(cfg.stations)
    event_id_map = build_event_id_map(catalog, work_dir / "event_id_map.csv")

    vm = cfg.hypodd.velocity_model
    is_3d = vm.type == "3d"
    vel3d_grid = read_vel3d_header(vm.file) if is_3d else None
    vel3d_filename = Path(vm.file).name if is_3d else None

    # M9.3 note: re-run check_compiled_limits with the real xcorr_obs_count
    # from dt.cc, now that it exists.
    xcorr_obs_count = _count_dt_cc_observations(dt_cc_path)
    limit_check = ensure_compiled_limits(
        cfg,
        catalog,
        station_table,
        external_hypodd_dir,
        external_bin_dir,
        vel3d_grid=vel3d_grid,
        xcorr_obs_count=xcorr_obs_count,
        log=logger,
    )
    if limit_check.any_exceeded:
        logger.warning(
            "hypoDD/ph2dt recompiled with larger limits before this relocation run (the real "
            "dt.cc observation count exceeded a previously-checked limit): %s",
            ", ".join(c.name for c in limit_check.exceeded_checks()),
        )

    vel3d_coverage_args: tuple | None = None
    if is_3d:
        lat_3d, lon_3d = vm.origin_3d
        rot_3d = vm.rotation_3d_deg
        vel3d_coverage_args = (vel3d_grid, lat_3d, lon_3d, rot_3d, catalog, station_table)
        # 6ld.6's note: before every 3D run.
        check_vel3d_coverage(*vel3d_coverage_args)

    fp = fingerprint(
        dt_ct_path,
        dt_cc_path,
        hypodd_inp_path,
        cfg.hypodd.parallel_clusters,
    )
    manifest = StepManifest(paths, STEP_NAME)
    reloc_path = work_dir / "hypoDD.reloc"
    skip_run = not force and manifest.is_done(MANIFEST_KEY, fp) and reloc_path.is_file()

    hypodd_bin = bin_path("hypoDD")
    nclust_used: int | None = None
    if skip_run:
        logger.info(
            "hypoDD run unchanged since the last run (dt.ct/dt.cc/hypoDD.inp fingerprint "
            "matches) -- reusing %s instead of rerunning.",
            reloc_path,
        )
        run_mode = "skipped"
    elif cfg.hypodd.parallel_clusters:
        nclust_used = _run_hypodd_clustered(
            cfg, hypodd_bin, work_dir, vel3d_filename, vel3d_coverage_args, logger
        )
        run_mode = (
            "parallel_clusters"
            if nclust_used > 1
            else "single (parallel_clusters: only 1 cluster found)"
        )
    else:
        invoke_hypodd(hypodd_bin, work_dir, log=logger)
        run_mode = "single"

    reloc_map = read_reloc(reloc_path, event_id_map)

    n_accepted = len(accepted(catalog))
    n_pairs = sum(1 for line in dt_ct_path.read_text().splitlines() if line.startswith("#"))
    if not skip_run and not reloc_map and n_accepted >= 2 and n_pairs > 0:
        # A genuinely empty result despite accepted events *and* at least
        # one ph2dt-linked pair is surprising, but not provably wrong on its
        # own (a strict enough OBSCC/OBSCT clustering configuration can
        # legitimately disconnect every event, confirmed empirically during
        # this task) -- so this is a loud WARNING with hypoDD.log's tail
        # attached, not a hard failure (the issue's own pitfall is about not
        # *silently* declaring success, not about treating every legitimate
        # "clustering left nothing connected" outcome as fatal).
        logger.warning(
            "hypoDD relocated 0 events despite %d accepted event(s) and %d ph2dt-linked pair(s) "
            "-- this can be a legitimate outcome of a strict OBSCC/OBSCT clustering "
            "configuration, or a genuine data problem (a singular system, or every cluster "
            "disconnected) -- check hypoDD.log. Tail:\n%s",
            n_accepted,
            n_pairs,
            _hypodd_log_tail(work_dir),
        )

    merged, n_relocated, n_not_linked = merge_relocations(catalog, reloc_map)
    write_catalog(merged, paths.catalog_dd_xml)

    if not skip_run:
        manifest.mark_done(MANIFEST_KEY, fp)

    logger.info(
        "hypoDD relocation (%s): %d event(s) relocated, %d not linked (kept prior origin) -> %s",
        run_mode,
        n_relocated,
        n_not_linked,
        paths.catalog_dd_xml,
    )
    return HypoddRunResult(
        rebuilt=not skip_run,
        run_mode=run_mode,
        n_relocated=n_relocated,
        n_not_linked=n_not_linked,
        nclust=nclust_used,
        limit_check=limit_check,
        catalog_dd_xml=paths.catalog_dd_xml,
    )
