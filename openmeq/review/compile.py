"""Review compile step: merge reviewed events over the automatic catalogue
(M7.4 / OpenMEQ-c3l.4).

:func:`compile_review` is this module's one public entry point, backing
``openmeq review --compile`` (``openmeq/review/__init__.py``). It reads
``locate/catalog_auto.xml`` (M6.4's output) and ``review/state.json`` +
``review/events/<event_id>.xml`` (M7.3's batch loop output, see
``docs/ARCHITECTURE.md`` §3a) and writes ``review/catalog_final.xml``, the
catalogue M8 (magnitude) prefers when it exists (``docs/PLAN.md`` §6 Step
7).

Merge rule (per event, keyed by the ``review/events/<event_id>.xml``
filename stem -- the *authoritative* event id, ARCHITECTURE.md §3a / M7.3's
own notes, not whatever ``Event.resource_id`` a hand-built or otherwise
produced file happens to carry):

* ``review/events/<event_id>.xml`` exists -> the reviewed event replaces
  the automatic one **wholesale** (origin, picks, magnitudes if any) -- the
  reviewed QuakeML from ObsPyck is a complete re-pick + relocation, not a
  patch, so nothing is merged field-by-field (:func:`_merge_reviewed`). The
  automatic origin is kept alongside it as a non-preferred `Origin`, purely
  for comparison (this module's own diagnostics use it, and so can anyone
  reading ``catalog_final.xml`` directly).
* ``state.json[event_id] == "rejected"`` -> the automatic event is kept,
  but flagged via ``openmeq.catalog.reject_event`` -- M7.3 only *persists*
  the reviewer's decision (ARCHITECTURE.md §3a), this is the step that
  actually applies the pipeline's rejection convention (ARCHITECTURE.md
  §5).
* otherwise (``"skipped"``, or no ``state.json`` entry at all -- never
  offered for review, or the batch loop hasn't reached it yet) -> the
  automatic event passes through completely unchanged.

Translating ObsPyck's own extra-namespace marker
--------------------------------------------------
Every event ObsPyck saves carries its own ``evaluationMode``/``public``
extra fields under the ``http://erdbeben-in-bayern.de/xmlns/0.1`` namespace
(``obspyck.obspyck.ObsPyck.get_QUAKEML_string``, confirmed directly against
the installed ``obspyck`` 0.8.2 source: ``extra.evaluationMode = {'value':
"manual", 'namespace': NAMESPACE}``, set unconditionally on the `Event`,
never on `Pick`/`Origin`) -- ObsPyck's own convention for "this was manually
reviewed", not OpenMEQ's. This module never leaves that foreign-namespace
marker sitting unexplained in ``catalog_final.xml``: :func:`_translate_obspyck_extra`
copies each key it finds into OpenMEQ's own extra namespace
(``openmeq.catalog.set_extra``, ARCHITECTURE.md §5), prefixed
``"obspyck_"``, then deletes the foreign-namespace entry -- translated, not
silently dropped *or* silently kept. Separately (and regardless of whether
that marker was present -- a hand-built reviewed file need not include it),
:func:`_apply_manual_conventions` sets OpenMEQ's own standard-field
convention (``evaluation_mode="manual"``, ``method_id
smi:openmeq/manual/obspyck``) on every reviewed pick and the reviewed
origin, since ObsPyck's own output never sets those standard QuakeML fields
itself -- only ever its own foreign extra marker.

Diagnostics
------------
:func:`_write_diagnostics` writes ``review/diagnostics/`` (PLAN.md §5's
"Review" row: "Auto vs manual pick-time differences; location shifts"):
pick-time-difference histograms per phase (matched by station + phase
between the automatic and reviewed pick sets -- *not* list position, since a
reviewer may add/remove picks entirely, this module's own "Pitfalls" note),
horizontal/depth location-shift histograms and a map coloured by horizontal
shift, and a ``summary.csv`` with one row per reviewed event.

StepManifest skip-unchanged
-----------------------------
Gated on ``locate/catalog_auto.xml`` + every ``review/events/*.xml`` file's
own on-disk identity (size/mtime, ``openmeq.manifest.fingerprint``'s `Path`
handling) + ``review/state.json`` -- covers every input this step reads, so
any reviewer action (a new save, a reset, a fresh rejection/skip) always
retriggers a rewrite, and re-running with nothing new is a no-op (this
issue's own ``test_skip_unchanged`` acceptance criterion).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd
from obspy import Catalog
from obspy.core.event import Event, Origin, Pick
from obspy.core.event.base import ResourceIdentifier
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import event_id as event_id_of
from openmeq.catalog import make_event, read_catalog, reject_event, set_extra, write_catalog
from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint
from openmeq.plotting import plot_hist, plot_map_sections

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("review")

#: ARCHITECTURE.md §5's manual-review method id convention.
_MANUAL_METHOD_ID = "smi:openmeq/manual/obspyck"

#: The step name ``reject_event`` records for a review-stage rejection.
_REJECT_STEP = "review"

#: ObsPyck 0.8.2's own foreign extra namespace (``obspyck/obspyck.py``:
#: ``NAMESPACE = "http://erdbeben-in-bayern.de/xmlns/0.1"``), set
#: unconditionally on the `Event` by ``get_QUAKEML_string`` whenever an
#: event is saved locally.
_OBSPYCK_NAMESPACE = "http://erdbeben-in-bayern.de/xmlns/0.1"

#: The two extra keys ObsPyck sets under that namespace.
_OBSPYCK_EXTRA_KEYS = ("evaluationMode", "public")

_SUMMARY_COLUMNS = [
    "event_id",
    "lon",
    "lat",
    "depth_km",
    "horizontal_shift_km",
    "depth_shift_km",
    "n_p_pairs",
    "n_s_pairs",
    "mean_dt_p_s",
    "mean_dt_s_s",
]


@dataclasses.dataclass
class _EventDiag:
    """One reviewed event's diagnostic data, gathered while merging."""

    event_id: str
    lon: float | None
    lat: float | None
    depth_km: float | None
    horizontal_shift_km: float | None
    depth_shift_km: float | None
    pick_diffs: dict[str, list[float]]


def _load_state(paths: RunPaths) -> dict[str, str]:
    """Read ``review/state.json`` (``{}`` if it doesn't exist yet -- a
    ``--compile`` run against a review session that never wrote anything is
    valid: every event just passes through unchanged)."""
    path = paths.review_state_json
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _reviewed_files(events_dir: Path) -> list[Path]:
    """Sorted list of every ``review/events/*.xml`` file, for the manifest
    fingerprint (module docstring) -- deterministic order, and empty (not
    an error) when the directory doesn't exist yet."""
    if not events_dir.exists():
        return []
    return sorted(events_dir.glob("*.xml"))


def _picks_by_station_phase(event: Event) -> dict[tuple[str, str], Pick]:
    """Index ``event.picks`` by ``(NET.STA, phase_hint)``.

    Matching is by station + phase, never list position or channel -- a
    reviewer may pick a different (or additional) channel for the same
    phase, and picks may be added/removed entirely between the automatic
    and reviewed sets (this module's own "Pitfalls" note, from the issue's
    DESIGN section). A duplicate ``(station, phase)`` within one event's own
    picks (shouldn't normally happen) keeps the first one seen.
    """
    index: dict[tuple[str, str], Pick] = {}
    for pick in event.picks:
        phase = pick.phase_hint
        wid = pick.waveform_id
        if phase is None or wid is None or wid.station_code is None:
            continue
        key = (f"{wid.network_code}.{wid.station_code}", phase)
        index.setdefault(key, pick)
    return index


def _apply_manual_conventions(event: Event, origin: Origin) -> None:
    """Set OpenMEQ's own ``evaluation_mode="manual"`` / ``method_id`` on the
    reviewed origin and every reviewed pick, if not already set.

    ObsPyck's own saved output never sets these standard QuakeML fields
    itself (module docstring: it only ever sets its own foreign-namespace
    extra marker) -- so in practice this always applies them for a real
    ObsPyck-produced file. It only *skips* a field already set, so a
    hand-built reviewed file that already sets these explicitly (e.g. a
    test fixture, or a future non-ObsPyck review tool) is respected as-is.
    """
    if origin.evaluation_mode is None:
        origin.evaluation_mode = "manual"
    if origin.method_id is None:
        origin.method_id = ResourceIdentifier(_MANUAL_METHOD_ID)
    for pick in event.picks:
        if pick.evaluation_mode is None:
            pick.evaluation_mode = "manual"
        if pick.method_id is None:
            pick.method_id = ResourceIdentifier(_MANUAL_METHOD_ID)


def _translate_obspyck_extra(event: Event) -> None:
    """Translate ObsPyck's own erdbeben-in-bayern.de extra markers into
    OpenMEQ's own extra namespace, then remove the foreign-namespace entry.

    See the module docstring's "Translating ObsPyck's own extra-namespace
    marker" section. A no-op if ``event.extra`` has neither key, or if a key
    of the same name exists under a different namespace (left untouched --
    not ObsPyck's marker).
    """
    extra = getattr(event, "extra", None)
    if not extra:
        return
    for key in _OBSPYCK_EXTRA_KEYS:
        entry = extra.get(key)
        if entry is None:
            continue
        namespace = entry.get("namespace") if hasattr(entry, "get") else None
        if namespace != _OBSPYCK_NAMESPACE:
            continue
        value = entry.get("value") if hasattr(entry, "get") else None
        set_extra(event, f"obspyck_{key}", value)
        del extra[key]


def _merge_reviewed(auto_event: Event, reviewed_path: Path, eid: str) -> tuple[Event, _EventDiag]:
    """Build the final `Event` for one reviewed event, plus its diagnostic data.

    Parameters
    ----------
    auto_event : obspy.core.event.Event
        This event's entry from ``locate/catalog_auto.xml``.
    reviewed_path : pathlib.Path
        ``review/events/<event_id>.xml``.
    eid : str
        The event id (the filename stem -- authoritative, ARCHITECTURE.md
        §3a).

    Returns
    -------
    (event, diag) : (obspy.core.event.Event, _EventDiag)
    """
    reviewed_cat = read_catalog(reviewed_path)
    if len(reviewed_cat) != 1:
        raise ValueError(
            f"event {eid!r}: {reviewed_path} must contain exactly one Event "
            f"(M7.3 saves one event per file), found {len(reviewed_cat)}"
        )
    reviewed_event = reviewed_cat[0]

    # The filename stem is authoritative (ARCHITECTURE.md §3a) -- re-stamp
    # defensively even though M7.3's batch loop already restamps a real
    # ObsPyck save (obspyck's own setXMLEventID overwrites resource_id with
    # its own timestamp-based id), so a reviewed file produced any other way
    # (e.g. hand-built) is never filed under the wrong event_id.
    reviewed_event.resource_id = make_event(eid).resource_id

    reviewed_origin = reviewed_event.preferred_origin()
    if reviewed_origin is None and reviewed_event.origins:
        reviewed_origin = reviewed_event.origins[-1]
    if reviewed_origin is None:
        raise ValueError(f"event {eid!r}: reviewed file {reviewed_path} has no origin")
    reviewed_event.preferred_origin_id = reviewed_origin.resource_id

    _apply_manual_conventions(reviewed_event, reviewed_origin)
    _translate_obspyck_extra(reviewed_event)

    # Keep the automatic origin too, as a non-preferred Origin, purely for
    # comparison (design section: "keep the automatic origins as
    # non-preferred origins"). A deep copy: the same Origin instance must
    # not be shared between the discarded auto_event and the kept
    # reviewed_event.
    auto_origin = auto_event.preferred_origin()
    if auto_origin is not None:
        reviewed_event.origins.append(copy.deepcopy(auto_origin))

    diag = _EventDiag(
        event_id=eid,
        lon=reviewed_origin.longitude,
        lat=reviewed_origin.latitude,
        depth_km=(reviewed_origin.depth / 1000.0) if reviewed_origin.depth is not None else None,
        horizontal_shift_km=None,
        depth_shift_km=None,
        pick_diffs=_pick_time_diffs(auto_event, reviewed_event),
    )
    if (
        auto_origin is not None
        and auto_origin.latitude is not None
        and auto_origin.longitude is not None
        and reviewed_origin.latitude is not None
        and reviewed_origin.longitude is not None
    ):
        dist_m, _, _ = gps2dist_azimuth(
            auto_origin.latitude,
            auto_origin.longitude,
            reviewed_origin.latitude,
            reviewed_origin.longitude,
        )
        diag.horizontal_shift_km = dist_m / 1000.0
        if auto_origin.depth is not None and reviewed_origin.depth is not None:
            diag.depth_shift_km = (reviewed_origin.depth - auto_origin.depth) / 1000.0

    return reviewed_event, diag


def _pick_time_diffs(auto_event: Event, reviewed_event: Event) -> dict[str, list[float]]:
    """``manual_time - auto_time`` (seconds), per phase, for every
    (station, phase) pair present in both events' own pick sets."""
    auto_picks = _picks_by_station_phase(auto_event)
    reviewed_picks = _picks_by_station_phase(reviewed_event)
    diffs: dict[str, list[float]] = {"P": [], "S": []}
    for key, auto_pick in auto_picks.items():
        reviewed_pick = reviewed_picks.get(key)
        if reviewed_pick is None:
            continue
        _station, phase = key
        if phase not in diffs:
            continue
        diffs[phase].append(float(reviewed_pick.time - auto_pick.time))
    return diffs


def _write_diagnostics(diag_rows: list[_EventDiag], paths: RunPaths) -> None:
    """Write ``review/diagnostics/`` (PLAN.md §5): pick-time-difference and
    location-shift plots, plus ``summary.csv``."""
    diagnostics_dir = paths.diagnostics_dir("review")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    all_dt: dict[str, list[float]] = {"P": [], "S": []}
    records = []
    for d in diag_rows:
        for phase in ("P", "S"):
            all_dt[phase].extend(d.pick_diffs.get(phase, []))
        p_vals = d.pick_diffs.get("P", [])
        s_vals = d.pick_diffs.get("S", [])
        records.append(
            {
                "event_id": d.event_id,
                "lon": d.lon,
                "lat": d.lat,
                "depth_km": d.depth_km,
                "horizontal_shift_km": d.horizontal_shift_km,
                "depth_shift_km": d.depth_shift_km,
                "n_p_pairs": len(p_vals),
                "n_s_pairs": len(s_vals),
                "mean_dt_p_s": (sum(p_vals) / len(p_vals)) if p_vals else None,
                "mean_dt_s_s": (sum(s_vals) / len(s_vals)) if s_vals else None,
            }
        )

    for phase in ("P", "S"):
        plot_hist(
            all_dt[phase],
            diagnostics_dir / f"pick_time_diff_{phase}.png",
            xlabel=f"manual - auto {phase} pick time (s)",
            title=f"Auto vs manual {phase} pick-time difference",
        )

    horizontal = [r["horizontal_shift_km"] for r in records if r["horizontal_shift_km"] is not None]
    depth = [r["depth_shift_km"] for r in records if r["depth_shift_km"] is not None]
    plot_hist(
        horizontal,
        diagnostics_dir / "location_shift_horizontal_km.png",
        xlabel="Horizontal shift, auto -> manual (km)",
        title="Auto vs manual location shift (horizontal)",
    )
    plot_hist(
        depth,
        diagnostics_dir / "location_shift_depth_km.png",
        xlabel="Depth shift, auto -> manual (km)",
        title="Auto vs manual location shift (depth)",
    )

    df = pd.DataFrame(records, columns=_SUMMARY_COLUMNS)

    stations_df = None
    if paths.stations_csv.exists():
        stations_df = pd.read_csv(paths.stations_csv)
    plot_map_sections(
        df,
        stations_df,
        diagnostics_dir / "location_shift_map.png",
        color_by="horizontal_shift_km",
        cbar_label="Horizontal shift (km)",
        title="Reviewed event locations (colour: horizontal shift, auto -> manual)",
    )

    df.to_csv(diagnostics_dir / "summary.csv", index=False, columns=_SUMMARY_COLUMNS)


def _require_catalog_auto(paths: RunPaths) -> None:
    """Raise a clear, actionable error if ``locate/catalog_auto.xml`` is
    missing.

    Mirrors ``openmeq.detect.pipeline._require_qc_outputs``'s own
    documented convention: a fresh run directory with no ``locate`` output
    yet is the expected first-run-out-of-order mistake (``openmeq review
    --compile`` before ``openmeq locate`` has ever run), so this is a
    deliberate, CLI-level hard-dependency check with a clear message,
    rather than an incidental ``FileNotFoundError`` raised deeper inside
    ``openmeq.catalog.read_catalog``.
    """
    if paths.catalog_auto_xml.is_file():
        return
    message = (
        f"openmeq review --compile: missing required input: {paths.catalog_auto_xml}. "
        "Run `openmeq locate` first to build locate/catalog_auto.xml."
    )
    logger.error(message)
    raise click.ClickException(message)


def compile_review(cfg: Config, paths: RunPaths) -> Catalog:
    """Merge ``review/events/*.xml`` over ``locate/catalog_auto.xml`` into
    ``review/catalog_final.xml`` and write ``review/diagnostics/``.

    See the module docstring for the exact merge rule. Idempotent and
    skip-unchanged (``openmeq.manifest.StepManifest``, keyed on
    ``locate/catalog_auto.xml`` + every ``review/events/*.xml`` file +
    ``review/state.json``): re-running with nothing new does not rewrite
    ``catalog_final.xml`` or the diagnostics.

    Parameters
    ----------
    cfg : openmeq.config.Config
        Unused directly (accepted for the same reason every other
        ``compile_*``/``setup_*`` top-level orchestrator in this project
        takes ``cfg`` even when it only reads paths -- consistency, and
        headroom for a future config-driven diagnostic option).
    paths : openmeq.paths.RunPaths

    Returns
    -------
    obspy.Catalog
        The compiled catalogue (whether freshly written or, on a
        skip-unchanged run, read back from the existing
        ``catalog_final.xml``).

    Raises
    ------
    click.ClickException
        If ``locate/catalog_auto.xml`` doesn't exist yet (see
        :func:`_require_catalog_auto`).
    ValueError
        If a ``review/events/<event_id>.xml`` file contains zero or more
        than one `Event` (the design section's own "Pitfalls" note: read
        defensively rather than silently pick the wrong event).
    """
    del cfg  # not currently used; kept for the orchestration-function contract, see docstring

    _require_catalog_auto(paths)

    events_dir = paths.review_events_dir
    fp = fingerprint(paths.catalog_auto_xml, _reviewed_files(events_dir), paths.review_state_json)
    manifest = StepManifest(paths, "review")

    if paths.catalog_final_xml.exists() and manifest.is_done("compile", fp):
        logger.info(
            "review --compile: catalog_auto.xml/review/events/state.json unchanged, skipping"
        )
        return read_catalog(paths.catalog_final_xml)

    auto_catalog = read_catalog(paths.catalog_auto_xml)
    state = _load_state(paths)

    final_events: list[Event] = []
    diag_rows: list[_EventDiag] = []
    n_reviewed = n_rejected = n_unchanged = 0

    for auto_event in auto_catalog:
        eid = event_id_of(auto_event)
        reviewed_path = paths.review_event_file(eid)
        status = state.get(eid)

        if reviewed_path.exists():
            final_event, diag = _merge_reviewed(auto_event, reviewed_path, eid)
            final_events.append(final_event)
            diag_rows.append(diag)
            n_reviewed += 1
        elif status == "rejected":
            reject_event(auto_event, reason="manual review", step=_REJECT_STEP)
            final_events.append(auto_event)
            n_rejected += 1
        else:
            # "skipped", or no state.json entry at all (never attempted) --
            # both pass through unchanged (module docstring / ARCHITECTURE.md §3a).
            final_events.append(auto_event)
            n_unchanged += 1

    final_catalog = Catalog(
        events=final_events,
        resource_id=auto_catalog.resource_id,
        description=auto_catalog.description,
    )
    write_catalog(final_catalog, paths.catalog_final_xml)
    _write_diagnostics(diag_rows, paths)

    manifest.mark_done("compile", fp)
    logger.info(
        "review --compile: %d event(s) (%d reviewed, %d rejected, %d unchanged) -> %s",
        len(final_events),
        n_reviewed,
        n_rejected,
        n_unchanged,
        paths.catalog_final_xml,
    )
    return final_catalog
