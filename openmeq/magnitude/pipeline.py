"""Event ML orchestration: catalogue selection, per-event amplitude+ML,
QuakeML/CSV writing, diagnostics and skip-unchanged (M8.3 / OpenMEQ-625.3).

:func:`run_magnitude` is the real implementation behind ``openmeq
magnitude`` (registered as ``run`` in ``openmeq/magnitude/__init__.py``,
``docs/ARCHITECTURE.md`` §2's ``openmeq.cli`` contract). It wires M8.1
(:func:`openmeq.magnitude.amplitude.measure_event_amplitudes`) and M8.2
(:func:`openmeq.magnitude.relations.station_ml`) together:

1. **Catalogue selection** (:func:`select_catalog_path`): ``--catalog`` if
   given, else ``review/catalog_final.xml`` if it exists, else
   ``locate/catalog_auto.xml``.
2. Reads that catalogue and processes only :func:`openmeq.catalog.accepted`
   events through :func:`openmeq.parallel.pmap`, one item per event
   (:func:`_magnitude_one_event`) -- rejected events pass through **unchanged**
   into the output catalogue (never dropped, ``docs/ARCHITECTURE.md`` §5's
   "never delete, always flag" rule; this also matches ``docs/ARCHITECTURE.md``
   §6c's note that ``magnitude/catalog_mag.xml`` still holds every event, with
   downstream ph2dt/hypoDD writers filtering by ``accepted()`` again
   themselves).
3. Each per-event worker loads that event's raw chunk
   (``chunks/<event_id>/<event_id>.mseed``, :meth:`openmeq.paths.RunPaths.chunk_file`)
   and the merged station inventory (cached per worker process, see
   :func:`_cached_worker_inventory`), calls
   :func:`~openmeq.magnitude.amplitude.measure_event_amplitudes` then
   :func:`~openmeq.magnitude.relations.station_ml` per station, aggregates
   to an event ML (``cfg.magnitude.statistic``), and writes ``Amplitude``/
   ``StationMagnitude``/``Magnitude`` QuakeML objects directly onto the
   `Event` it returns (``docs/ARCHITECTURE.md`` §5).
4. Writes ``magnitude/catalog_mag.xml`` and ``magnitude/station_magnitudes.csv``,
   then :func:`openmeq.magnitude.diagnostics.make_magnitude_diagnostics`, and
   -- only when ``cfg.diagnostics.extended_enabled("magnitude")`` --
   :func:`openmeq.magnitude.diagnostics_extended.make_magnitude_extended_diagnostics`
   (``magnitude/diagnostics/extended/``; never fails the step).
5. :class:`openmeq.manifest.StepManifest` skip-unchanged, fingerprinted on
   ``cfg.magnitude`` plus the selected input catalogue file's identity (so a
   ``--catalog`` override, or the catalogue selection itself switching from
   ``catalog_auto.xml`` to a newly-appeared ``catalog_final.xml``, both
   correctly bust the cache). When extended QAQC is enabled, its settings
   (``n_examples``) are appended to the fingerprint, so switching it on
   reruns the step; the disabled-case fingerprint is unchanged from before.

Idempotent re-processing (design decision, this task)
-------------------------------------------------------
Every accepted event's ``amplitudes``/``station_magnitudes``/``magnitudes``
lists (and ``preferred_magnitude_id``) are reset to empty/``None`` at the
start of that event's worker call, before being repopulated from this run's
own measurements. This makes a magnitude re-run idempotent even when the
*input* catalogue already carries prior magnitude objects forward -- the
one documented case is ``docs/ARCHITECTURE.md`` §6h: re-running ``openmeq
magnitude --catalog relocate/catalog_dd.xml`` after hypoDD, whose own
``merge_relocations`` step explicitly carries the prior ``Magnitude``/
``StationMagnitude``/``Amplitude`` objects over onto the new, relocated
origin. Without this reset, re-running magnitude on that catalogue would
accumulate duplicate magnitude objects rather than replacing them with
freshly-computed ones against the (now more accurate) relocated hypocentre
-- origins/picks/other event content are left completely untouched, only
the magnitude-specific lists are cleared and rebuilt.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
import pandas as pd
from obspy import Catalog, read
from obspy.core.event import (
    Amplitude,
    Magnitude,
    StationMagnitude,
    StationMagnitudeContribution,
    TimeWindow,
    WaveformStreamID,
)
from obspy.core.event.base import ResourceIdentifier

from openmeq.catalog import accepted, read_catalog, set_extra, write_catalog
from openmeq.catalog import event_id as get_event_id
from openmeq.log import get_logger
from openmeq.magnitude.amplitude import StationAmplitude, _find_pick, measure_event_amplitudes
from openmeq.magnitude.diagnostics import make_magnitude_diagnostics
from openmeq.magnitude.diagnostics_extended import make_magnitude_extended_diagnostics
from openmeq.magnitude.relations import get_clamp_counter, station_ml
from openmeq.manifest import StepManifest, fingerprint
from openmeq.parallel import pmap
from openmeq.stations import load_inventory

if TYPE_CHECKING:
    from obspy import Stream
    from obspy.core.event import Event, Pick
    from obspy.core.inventory import Inventory

    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("magnitude")

#: ``RunPaths.manifest_json``/``diagnostics_dir``'s step-folder-name argument
#: (ARCHITECTURE.md §2: the output folder, ``magnitude/`` -- same as the CLI
#: command name here, unlike e.g. ``detect``/``detections``).
STEP_NAME = "magnitude"

#: Single whole-run :class:`~openmeq.manifest.StepManifest` entry key
#: (mirroring ``openmeq.detect.pipeline``'s own single-key convention).
MANIFEST_KEY = "magnitude"

#: ``cfg.magnitude.ml.relation`` -> the QuakeML ``Magnitude``/
#: ``StationMagnitude``/``Amplitude`` ``method_id`` (ARCHITECTURE.md §5).
METHOD_IDS: dict[str, str] = {
    "hutton_boore": "smi:openmeq/ml/hutton_boore",
    "local": "smi:openmeq/ml/local",
}

#: ``magnitude/station_magnitudes.csv``'s exact column order (this task's
#: own design note lists ``event_id,net,sta,amplitude_m,distance_km,
#: component_used,station_ml,clamped`` plus "an extra included boolean
#: column" -- ``included``), plus ``excluded`` (OpenMEQ-tch: the station is in
#: ``cfg.magnitude.exclude_stations``; its ``station_ml`` is still reported).
STATION_MAGNITUDES_COLUMNS = [
    "event_id",
    "net",
    "sta",
    "amplitude_m",
    "distance_km",
    "component_used",
    "station_ml",
    "clamped",
    "included",
    "excluded",
]


def _is_excluded(net: str, sta: str, exclude_stations: list[str]) -> bool:
    """``cfg.magnitude.exclude_stations`` match on ``STA`` or ``NET.STA``."""
    return sta in exclude_stations or f"{net}.{sta}" in exclude_stations


# ---------------------------------------------------------------------------
# Catalogue selection
# ---------------------------------------------------------------------------


def select_catalog_path(
    paths: RunPaths,
    catalog_override: Path | None = None,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Pick the input catalogue: ``--catalog`` override, else
    ``review/catalog_final.xml`` if it exists, else ``locate/catalog_auto.xml``.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    catalog_override : pathlib.Path or None
        The ``--catalog`` CLI value, if given. A relative path is resolved
        against ``config_path``'s own directory (ARCHITECTURE.md §1's
        relative-path convention -- the same rule ``openmeq.config.load_config``
        applies to every path field in the YAML itself); an absolute path
        is used unchanged.
    config_path : str, pathlib.Path or None, keyword-only, optional
        The YAML config file's own path, needed only to resolve a relative
        ``catalog_override``.

    Returns
    -------
    pathlib.Path
    """
    if catalog_override is not None:
        path = Path(catalog_override)
        if not path.is_absolute() and config_path is not None:
            path = (Path(config_path).resolve().parent / path).resolve()
        return path
    if paths.catalog_final_xml.is_file():
        return paths.catalog_final_xml
    return paths.catalog_auto_xml


# ---------------------------------------------------------------------------
# Per-worker inventory cache (mirrors amplitude.py's own response cache and
# relations.py's own calibration-table cache: each pmap worker process gets
# its own copy of this module, and therefore its own cache -- no
# cross-process sharing, per openmeq.parallel's spawn-based worker model).
# ---------------------------------------------------------------------------

_worker_inventory_cache: dict[tuple, Inventory] = {}


def _inventory_cache_key(stations_cfg) -> tuple:
    return (
        str(stations_cfg.stationxml) if stations_cfg.stationxml else None,
        str(stations_cfg.stationxml_dir) if stations_cfg.stationxml_dir else None,
        str(stations_cfg.csv) if stations_cfg.csv else None,
    )


def _cached_worker_inventory(cfg: Config) -> Inventory:
    """Load (and cache, per worker process) the merged station inventory.

    Loading/merging StationXML+CSV (``openmeq.stations.load_inventory``) is
    the expensive, per-process-constant part of every event this worker
    processes; caching it here means only the first event handled by a
    given worker pays that cost, matching this task's own "caching
    responses per worker" design note (the response cache inside
    ``openmeq.magnitude.amplitude`` then further caches individual
    ``Response`` lookups *within* that one already-loaded `Inventory`, per
    that module's own docstring).
    """
    key = _inventory_cache_key(cfg.stations)
    inv = _worker_inventory_cache.get(key)
    if inv is None:
        inv = load_inventory(cfg.stations)
        _worker_inventory_cache[key] = inv
    return inv


def clear_worker_inventory_cache() -> None:
    """Clear the module-level inventory cache (tests only)."""
    _worker_inventory_cache.clear()


# ---------------------------------------------------------------------------
# Per-event worker
# ---------------------------------------------------------------------------


@dataclass
class MagnitudeEventResult:
    """One event's :func:`_magnitude_one_event` result.

    Attributes
    ----------
    event : obspy.core.event.Event
        The same `Event` passed in, mutated in place with fresh
        ``Amplitude``/``StationMagnitude``/``Magnitude`` objects (see the
        module docstring's "Idempotent re-processing" section) and returned
        for the caller to collect into the output catalogue.
    rows : list of dict
        One :data:`STATION_MAGNITUDES_COLUMNS`-shaped row per station
        actually measured by M8.1 (may be empty).
    status : str
        ``"ok"``, ``"missing_chunk"`` or ``"unreadable_chunk"`` -- for the
        caller's own summary log line; a non-``"ok"`` status still returns
        a valid (magnitude-less) `Event`, never raises.
    """

    event: Event
    rows: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"


def _location_code(stream: Stream, net: str, sta: str, channel: str) -> str:
    """The location code of the first trace in ``stream`` matching
    ``net``/``sta``/``channel`` -- :class:`~openmeq.magnitude.amplitude.StationAmplitude`
    itself doesn't carry a location code (ARCHITECTURE.md §6: "location
    codes are ignored when two [stations] share a site"), but a QuakeML
    ``WaveformStreamID`` needs one, so this looks it up from the same chunk
    stream the amplitude was measured from. Returns ``""`` if no matching
    trace is found (should not happen for a channel M8.1 already measured,
    but defensive)."""
    for tr in stream:
        if tr.stats.network == net and tr.stats.station == sta and tr.stats.channel == channel:
            return tr.stats.location
    return ""


def _pick_used_for_window(event: Event, net: str, sta: str) -> Pick | None:
    """The pick that determined M8.1's measurement window for ``net.sta``.

    Reuses :func:`openmeq.magnitude.amplitude._find_pick` directly (same
    S-preferred-else-P lookup order as
    :func:`openmeq.magnitude.amplitude._pick_window`) rather than
    re-deriving the same logic independently, so this can never drift out
    of sync with which pick M8.1 actually used to build the window.
    """
    s_pick = _find_pick(event, net, sta, "S")
    if s_pick is not None:
        return s_pick
    return _find_pick(event, net, sta, "P")


def _make_amplitude(
    event_id: str, sa: StationAmplitude, location_code: str, pick: Pick | None, method_id: str
) -> Amplitude:
    """One station's QuakeML ``Amplitude`` (type ``"AML"``).

    **Unit boundary** (this task's own design note, easy to get backwards):
    ``generic_amplitude`` is set to ``sa.amplitude_m`` -- **metres**, M8.1's
    native unit and ARCHITECTURE.md §5's contract ("generic amplitude in
    metres, Wood-Anderson") -- never the millimetre value
    ``openmeq.magnitude.relations.station_ml`` converts to internally for
    its own log10(A) formula.
    """
    amp = Amplitude(
        resource_id=ResourceIdentifier(f"smi:openmeq/amplitude/{event_id}/{sa.net}.{sa.sta}"),
        generic_amplitude=sa.amplitude_m,
        type="AML",
        unit="m",
        waveform_id=WaveformStreamID(
            network_code=sa.net,
            station_code=sa.sta,
            location_code=location_code,
            channel_code=sa.component_used,
        ),
        time_window=TimeWindow(
            reference=sa.window_start,
            begin=0.0,
            end=float(sa.window_end - sa.window_start),
        ),
        method_id=ResourceIdentifier(method_id),
        evaluation_mode="automatic",
    )
    if pick is not None:
        amp.pick_id = pick.resource_id
    return amp


def _make_station_magnitude(
    event_id: str,
    sa: StationAmplitude,
    ml_value: float,
    amplitude: Amplitude,
    origin_id: ResourceIdentifier | None,
    method_id: str,
) -> StationMagnitude:
    return StationMagnitude(
        resource_id=ResourceIdentifier(
            f"smi:openmeq/stationmagnitude/{event_id}/{sa.net}.{sa.sta}"
        ),
        origin_id=origin_id,
        mag=ml_value,
        station_magnitude_type="ML",
        amplitude_id=amplitude.resource_id,
        waveform_id=amplitude.waveform_id,
        method_id=ResourceIdentifier(method_id),
    )


def _aggregate(values: list[float], statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def _magnitude_one_event(item: tuple[Event, RunPaths, Config]) -> MagnitudeEventResult:
    """``openmeq.parallel.pmap`` worker: amplitude + ML for one accepted event.

    Loads the event's raw chunk and the (per-worker-cached) merged
    inventory, calls M8.1 then M8.2 per station, aggregates to an event ML,
    and writes the QuakeML ``Amplitude``/``StationMagnitude``/``Magnitude``
    objects directly onto ``event`` (module docstring). Never raises for a
    per-event data problem (missing/unreadable chunk) -- only a genuine
    configuration problem (``openmeq.config.ConfigError``, e.g. an unknown
    ``magnitude.ml.relation``) propagates, since that indicates every event
    in this run would fail identically and should stop the run rather than
    be silently logged away event by event.
    """
    event, paths, cfg = item
    event_id = get_event_id(event)
    ml_cfg = cfg.magnitude.ml
    method_id = METHOD_IDS[ml_cfg.relation]

    # Idempotent re-processing (module docstring): start every event from a
    # clean magnitude slate, never accumulating onto whatever the input
    # catalogue already carried (e.g. relocate/catalog_dd.xml's carried-over
    # magnitude objects, ARCHITECTURE.md §6h).
    event.amplitudes = []
    event.station_magnitudes = []
    event.magnitudes = []
    event.preferred_magnitude_id = None

    chunk_path = Path(paths.chunk_file(event_id))
    if not chunk_path.is_file():
        logger.warning("%s: missing chunk file %s; no amplitude measurement", event_id, chunk_path)
        return MagnitudeEventResult(event=event, rows=[], status="missing_chunk")

    try:
        stream = read(str(chunk_path))
    except Exception as exc:  # noqa: BLE001 - a corrupt chunk must not abort the whole run
        logger.warning("%s: unreadable chunk %s: %s", event_id, chunk_path, exc)
        return MagnitudeEventResult(event=event, rows=[], status="unreadable_chunk")

    inventory = _cached_worker_inventory(cfg)

    origin = event.preferred_origin()
    if origin is None:
        origin = event.origins[0] if event.origins else None
    origin_id = origin.resource_id if origin is not None else None

    station_amps = measure_event_amplitudes(event, stream, inventory, cfg.magnitude)

    rows: list[dict[str, Any]] = []
    included_mls: list[float] = []

    for sa in station_amps:
        location_code = _location_code(stream, sa.net, sa.sta, sa.component_used)
        pick = _pick_used_for_window(event, sa.net, sa.sta)

        clamp_before = get_clamp_counter()
        included = True
        try:
            ml_value = station_ml(
                amplitude_m=sa.amplitude_m,
                distance_km=sa.distance_km,
                cfg=cfg,
                net=sa.net,
                sta=sa.sta,
            )
            if not math.isfinite(ml_value):
                included = False
        except ValueError as exc:
            # E.g. a math-domain error from log10 of a non-positive
            # amplitude -- a genuine per-station data problem, not a config
            # problem (ConfigError, which is not a ValueError and always
            # propagates), so this one station is excluded, logged, and the
            # rest of the event still proceeds.
            logger.warning(
                "%s.%s: station_ml failed for event %s: %s", sa.net, sa.sta, event_id, exc
            )
            ml_value = float("nan")
            included = False
        clamped = get_clamp_counter() > clamp_before
        # OpenMEQ-tch: measured and reported, but never in the event ML.
        excluded = _is_excluded(sa.net, sa.sta, cfg.magnitude.exclude_stations)
        computed = included
        included = included and not excluded

        amp = _make_amplitude(event_id, sa, location_code, pick, method_id)
        set_extra(amp, "clamped", clamped)
        set_extra(amp, "included_in_event_ml", included)
        if excluded:
            set_extra(amp, "excluded_by_config", True)
        event.amplitudes.append(amp)

        if included:
            station_mag = _make_station_magnitude(event_id, sa, ml_value, amp, origin_id, method_id)
            event.station_magnitudes.append(station_mag)
            included_mls.append(ml_value)

        rows.append(
            {
                "event_id": event_id,
                "net": sa.net,
                "sta": sa.sta,
                "amplitude_m": sa.amplitude_m,
                "distance_km": sa.distance_km,
                "component_used": sa.component_used,
                "station_ml": ml_value if computed else float("nan"),
                "clamped": clamped,
                "included": included,
                "excluded": excluded,
            }
        )

    n_measured = len(station_amps)
    n_used = len(included_mls)
    set_extra(event, "magnitude_n_stations_measured", n_measured)
    set_extra(event, "magnitude_n_stations_used", n_used)

    if n_used >= cfg.magnitude.min_stations:
        event_ml = _aggregate(included_mls, cfg.magnitude.statistic)
        magnitude = Magnitude(
            resource_id=ResourceIdentifier(f"smi:openmeq/magnitude/{event_id}/ml"),
            mag=event_ml,
            magnitude_type="ML",
            origin_id=origin_id,
            method_id=ResourceIdentifier(method_id),
            station_count=n_used,
            evaluation_mode="automatic",
        )
        magnitude.station_magnitude_contributions = [
            StationMagnitudeContribution(
                station_magnitude_id=sm.resource_id, residual=sm.mag - event_ml
            )
            for sm in event.station_magnitudes
        ]
        event.magnitudes.append(magnitude)
        event.preferred_magnitude_id = magnitude.resource_id
    else:
        logger.info(
            "%s: only %d station ML(s) (< min_stations=%d); no event Magnitude written "
            "(event kept in the output catalogue per ARCHITECTURE.md's never-drop rule)",
            event_id,
            n_used,
            cfg.magnitude.min_stations,
        )
        set_extra(event, "magnitude_skip_reason", "min_stations")

    return MagnitudeEventResult(event=event, rows=rows, status="ok")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_magnitude(
    cfg: Config,
    paths: RunPaths,
    catalog_override: Path | None = None,
    *,
    config_path: str | Path | None = None,
    force: bool = False,
) -> None:
    """Run event ML end to end (``openmeq magnitude``).

    See the module docstring for the full sequence.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    catalog_override : pathlib.Path or None, optional
        ``--catalog``: forwarded to :func:`select_catalog_path`.
    config_path : str, pathlib.Path or None, keyword-only, optional
        The YAML config file's own path, used to resolve a relative
        ``catalog_override`` (:func:`select_catalog_path`).
    force : bool, keyword-only, optional
        ``--force``: recompute even if the manifest says nothing changed.

    Raises
    ------
    click.ClickException
        If the selected input catalogue does not exist on disk.
    """
    t0 = time.perf_counter()
    paths.ensure()
    step_logger = get_logger("magnitude")

    catalog_path = select_catalog_path(paths, catalog_override, config_path=config_path)
    if not catalog_path.is_file():
        message = (
            f"openmeq magnitude: input catalogue not found: {catalog_path}. Run `openmeq locate` "
            "(and optionally `openmeq review --compile`) first, or pass --catalog."
        )
        step_logger.error(message)
        raise click.ClickException(message)

    extended = cfg.diagnostics.extended_enabled(STEP_NAME)
    if extended:
        fp = fingerprint(
            cfg.magnitude,
            catalog_path,
            {"extended": True, "n_examples": cfg.diagnostics.n_examples},
        )
    else:
        fp = fingerprint(cfg.magnitude, catalog_path)
    manifest = StepManifest(paths, STEP_NAME)
    if not force and manifest.is_done(MANIFEST_KEY, fp) and paths.catalog_mag_xml.is_file():
        step_logger.info(
            "magnitude: unchanged since the last run (cfg.magnitude and %s match the last "
            "recorded fingerprint) -- skipping (use --force to rerun).",
            catalog_path,
        )
        return

    input_catalog = read_catalog(catalog_path)
    accepted_events = list(accepted(input_catalog))
    accepted_ids = {get_event_id(ev) for ev in accepted_events}

    items = [(ev, paths, cfg) for ev in accepted_events]
    results = pmap(_magnitude_one_event, items, cfg.parallel, desc="magnitude")
    results_by_id = {get_event_id(r.event): r for r in results}

    out_events: list[Event] = []
    rows: list[dict[str, Any]] = []
    n_with_magnitude = n_missing_chunk = n_unreadable = 0
    for ev in input_catalog:
        eid = get_event_id(ev)
        if eid in accepted_ids:
            result = results_by_id[eid]
            out_events.append(result.event)
            rows.extend(result.rows)
            if result.status == "missing_chunk":
                n_missing_chunk += 1
            elif result.status == "unreadable_chunk":
                n_unreadable += 1
            if result.event.preferred_magnitude_id is not None:
                n_with_magnitude += 1
        else:
            # Rejected event: passed through completely unchanged
            # (ARCHITECTURE.md §5's "never delete, always flag" rule).
            out_events.append(ev)

    out_catalog = Catalog(
        events=out_events,
        resource_id=input_catalog.resource_id,
        description=input_catalog.description,
    )
    write_catalog(out_catalog, paths.catalog_mag_xml)

    df = pd.DataFrame(rows, columns=STATION_MAGNITUDES_COLUMNS)
    paths.station_magnitudes_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.station_magnitudes_csv, index=False)

    make_magnitude_diagnostics(paths, df, out_catalog)
    if extended:
        make_magnitude_extended_diagnostics(paths, df, out_catalog, cfg)

    manifest.mark_done(MANIFEST_KEY, fp)

    n_clamped = int(df["clamped"].sum()) if not df.empty else 0
    elapsed = time.perf_counter() - t0
    step_logger.info(
        "magnitude: %d accepted event(s) processed (%d with a Magnitude, %d missing chunk, "
        "%d unreadable chunk); %d/%d station measurement(s) table-clamped; %d rejected event(s) "
        "passed through unchanged; done in %.2fs",
        len(accepted_events),
        n_with_magnitude,
        n_missing_chunk,
        n_unreadable,
        n_clamped,
        len(df),
        len(out_events) - len(accepted_events),
        elapsed,
    )
