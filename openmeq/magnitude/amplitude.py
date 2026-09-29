"""Wood-Anderson amplitude measurement per event (M8.1 / OpenMEQ-625.1).

:func:`measure_event_amplitudes` is the lowest-level piece of the ML step
(PLAN.md §6 Step 7 / ARCHITECTURE.md §5): given one event's `Origin` and
`Pick`\\ s, its raw waveform chunk, and the merged station inventory, it
returns one :class:`StationAmplitude` per usable station -- a Wood-Anderson
zero-to-peak amplitude, **in metres of WA displacement**
(ARCHITECTURE.md §5: "Amplitude type AML, in metres of WA displacement"),
plus the hypocentral distance and the window/component used to measure it.

This module does no file I/O beyond what is passed in (the `Event`,
`Stream` and `Inventory` are already-parsed objects) and never raises for a
single bad station -- every skip condition (§ "Skips", below) is logged and
the station is simply left out of the returned list, so one bad channel
never aborts a whole event's amplitude measurement inside `openmeq.parallel
.pmap` (M8.3 wires that; see the module's own performance note there).

Downstream (M8.2 `openmeq.magnitude.relations.station_ml`, M8.3): this
module's `StationAmplitude.amplitude_m` stays in **metres** -- the mm
conversion for the log10(A) formula happens once, by name, in
`relations.station_ml`, not here (the SI-unit boundary this task's design
note asks for).

Algorithm (per station, per candidate channel)
------------------------------------------------
1. **Window**: ``[S_pick - window_before_s, S_pick + window_after_s]`` if an
   S pick exists for that station; else the fallback coda window
   ``[P_pick + 1.0, P_pick + 8.0]`` (a warning is logged for the fallback).
   A rejected pick (``evaluation_status == "rejected"``,
   ARCHITECTURE.md §5) is never used to build a window.
2. The raw (unprocessed) trace is sliced to that window first, to cheaply
   reject a channel with no data covering the window, a data gap in the
   window (a masked sample), or a flat-lined window (``np.ptp == 0``) --
   all logged, none raised -- *before* spending the cost of a response
   removal on unusable data.
3. **2-step Wood-Anderson simulation**, applied to the *whole* available
   trace (not just the window -- these are FFT-based filters, and applying
   them over a short window would introduce edge artifacts right where the
   window itself needs to be accurate):
   a. demean, linear-detrend, then a ``cfg.taper_pct`` cosine taper
      (raw chunks are unprocessed -- ARCHITECTURE.md §3 -- so this step is
      not optional).
   b. ``trace.remove_response(output="DISP", water_level=60,
      pre_filt=cfg.pre_filt)`` -- deconvolves the instrument response to
      recover true ground displacement in metres. Rather than passing
      ``inventory=`` (which makes ObsPy re-walk the whole `Inventory` tree
      for a matching channel/epoch on every call), the resolved `Response`
      is looked up once per seed id through the module-level cache (see
      "Response cache" below) and attached directly to
      ``trace.stats.response``, which `Trace.remove_response` uses when no
      ``inventory`` is given -- functionally identical, cheaper across the
      many calls one event's amplitude measurement makes.
   c. ``trace.simulate(paz_simulate=WOOD_ANDERSON_PAZ | {"sensitivity":
      cfg.wa_sensitivity})`` -- convolves the recovered ground displacement
      with the (uncalibrated-gain, ``gain=1.0``) Wood-Anderson poles/zeros
      and multiplies by ``wa_sensitivity`` (the historical WA magnification
      constant), producing the displacement trace a real Wood-Anderson
      seismograph would have recorded. Deliberately the literal PAZ from
      the M8 epic design note (**not** ``obspy.signal.invsim.WOODANDERSON``,
      which has a different --and, per several published discussions,
      arguably non-standard-- zero count: one zero at the origin rather
      than the classical two; see :data:`WOOD_ANDERSON_PAZ`'s own docstring).
4. The now WA-simulated trace is sliced to the window and
   ``max(abs(window_samples))`` is the zero-to-peak amplitude, in metres.
5. **Component selection** (``cfg.component == "horizontal"``, the only
   value the schema currently allows): every horizontal channel present
   (last character ``N``/``E``/``1``/``2`` -- ARCHITECTURE.md §6, unoriented
   borehole ``1``/``2`` channels count as horizontal here) is measured
   independently through steps 2-4 above; whichever has the larger peak
   amplitude in the window is used. If no horizontal channel is usable
   (missing/no response, no data, flat, or simply absent), the vertical
   (``Z``) channel is used instead, logged as a fallback.
6. **Distance**: hypocentral, 3-D (ARCHITECTURE.md §6/§8), combining the
   epicentral distance (`obspy.geodetics.gps2dist_azimuth` on the origin
   and the *chosen* channel's station coordinates) with the vertical
   separation between the origin's depth and the station's effective
   elevation (``elevation_m - depth_m``, ARCHITECTURE.md §6 -- the
   channel's own ``depth`` field in the `Inventory`, i.e. sensor depth
   below the surface for a borehole) via Pythagoras.

Response cache
--------------
``_response_cache: dict[str, Response | None]``, keyed by seed id
(``"NET.STA.LOC.CHA"``), populated lazily the first time a channel is seen.
Per ``openmeq.parallel``'s worker model (ARCHITECTURE.md §2: `pmap` spawns
persistent worker processes, one `Inventory` parse per call from the
caller, not per worker), this module-level cache stays warm across every
event a given worker processes in one run -- it does *not* re-parse or
re-select from the `Inventory` (already parsed once by the caller and
passed to every call); it only saves ObsPy's own per-call linear search
over ``inventory.networks`` / ``.stations`` / ``.channels`` for a matching
seed id and epoch. A seed id that fails the lookup (channel absent from the
`Inventory` -- e.g. a station known only from the stations CSV, with no
response, per ``openmeq.stations``' merge precedence) is cached as ``None``
so the (also-failing) lookup is not repeated for a later event either --
see :func:`_get_cached_response`. Like `openmeq.magnitude.relations`'s
calibration-table cache, each `pmap` worker process gets its own copy of
this module and therefore its own cache (spawn start method, no
cross-process sharing).

Skips (logged, never raised)
-----------------------------
- No P or S pick at all for a station present in the stream.
- No response in the `Inventory` for the channel(s) attempted (including a
  station known only from the CSV metadata source, which carries no
  channels/response at all -- ``openmeq.stations.load_inventory``).
- No waveform data (or a data gap) covering the measurement window.
- A flat-lined window (``np.ptp(window_data) == 0``).
- No usable horizontal *or* vertical channel at all for a station.

Trace capture (extended QAQC)
-----------------------------
``measure_event_amplitudes(..., capture={})`` additionally records, per
station, the chosen channel's already-computed WA-simulated trace, window
and peak-sample time as a :class:`WoodAndersonCapture` (used by
``magnitude/diagnostics/extended/amp_window_<event_id>.png``). It reuses
the very trace the measurement was made on -- no second simulation -- and
is ``None`` (nothing retained) by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.core.event import Event
from obspy.core.inventory import Inventory
from obspy.geodetics import gps2dist_azimuth

from openmeq.config import MagnitudeConfig, MlConfig
from openmeq.log import get_logger

logger = get_logger("magnitude")

#: Wood-Anderson poles/zeros (M8 epic design note), gain normalized to 1.0 --
#: the overall magnification is applied separately via
#: ``paz_simulate["sensitivity"]`` (``cfg.magnitude.ml.wa_sensitivity``,
#: default 2080, the IASPEI/Uhrhammer & Collins 1990 value; see
#: `openmeq.config.MlConfig`). Deliberately **not**
#: ``obspy.signal.invsim.WOODANDERSON``: that constant has only one zero at
#: the origin (``zeros=[0j]``) rather than the classical two
#: (``zeros=[0j, 0j]``) used here, and a poles value rounded to 3 rather
#: than 4 decimal places -- both close enough to be easy to miss, but this
#: task's design note specifies the values below explicitly rather than
#: reusing that constant.
WOOD_ANDERSON_PAZ: dict[str, object] = {
    "poles": [complex(-6.2832, -4.7124), complex(-6.2832, 4.7124)],
    "zeros": [0j, 0j],
    "gain": 1.0,
}

#: Channel-code last characters counted as "horizontal" (ARCHITECTURE.md §6:
#: oriented N/E, or unoriented borehole 1/2, which count as horizontal for
#: amplitude-component selection even though their orientation is unknown).
_HORIZONTAL_LAST_CHARS = ("N", "E", "1", "2")

#: Sample-count tolerance used when checking a sliced trace actually covers
#: a requested window (rounding/edge-sample slack around ``Trace.slice``).
_COVERAGE_TOL_SAMPLES = 1.5

#: module-level response cache (per-process; see module docstring).
_response_cache: dict[str, object] = {}


def clear_response_cache() -> None:
    """Clear the module-level response cache (tests only).

    A long-running worker process never needs this -- a `seed_id` is a
    stable cache key for the process's whole lifetime.
    """
    _response_cache.clear()


@dataclass
class StationAmplitude:
    """One station's Wood-Anderson amplitude measurement for one event.

    Attributes
    ----------
    net, sta : str
        Network and station codes (ARCHITECTURE.md §6 ``NET.STA``).
    amplitude_m : float
        Zero-to-peak Wood-Anderson-simulated displacement amplitude, in
        **metres** (not mm -- the mm conversion is
        `openmeq.magnitude.relations.station_ml`'s job).
    distance_km : float
        Hypocentral (3-D) distance from the origin, km.
    component_used : str
        The full channel code (e.g. ``"HHE"``, ``"DP1"``, ``"HHZ"``) whose
        measurement was used.
    window_start, window_end : obspy.UTCDateTime
        The measurement window actually used (S-pick-relative, or the
        P-derived fallback coda window).
    """

    net: str
    sta: str
    amplitude_m: float
    distance_km: float
    component_used: str
    window_start: UTCDateTime
    window_end: UTCDateTime


@dataclass
class WoodAndersonCapture:
    """The chosen channel's processed trace behind one :class:`StationAmplitude`.

    Only produced when :func:`measure_event_amplitudes` is given a
    ``capture`` dict (extended QAQC); see the module docstring.

    Attributes
    ----------
    net, sta, channel : str
    trace : obspy.Trace
        The full-extent WA-simulated displacement trace (metres) the
        amplitude was measured on.
    window_start, window_end : obspy.UTCDateTime
    peak_time : obspy.UTCDateTime
        Time of the ``max(abs(...))`` sample inside the window.
    amplitude_m : float
    """

    net: str
    sta: str
    channel: str
    trace: Trace
    window_start: UTCDateTime
    window_end: UTCDateTime
    peak_time: UTCDateTime
    amplitude_m: float


# ---------------------------------------------------------------------------
# response cache
# ---------------------------------------------------------------------------


def _get_cached_response(inventory: Inventory, seed_id: str, time: UTCDateTime):
    """Return (and cache) the `Response` for ``seed_id``, or ``None``.

    See the module docstring's "Response cache" section. ``None`` is cached
    too (a seed id absent from ``inventory``, e.g. CSV-only station
    metadata), so a station that will never have a response is not
    re-searched for on every later event.
    """
    if seed_id not in _response_cache:
        try:
            _response_cache[seed_id] = inventory.get_response(seed_id, time)
        except Exception:  # noqa: BLE001 - any lookup failure means "no response"
            _response_cache[seed_id] = None
    return _response_cache[seed_id]


# ---------------------------------------------------------------------------
# picks / window
# ---------------------------------------------------------------------------


def _find_pick(event: Event, net: str, sta: str, phase: str):
    """First non-rejected pick for ``net.sta`` with the given ``phase_hint``."""
    for pick in event.picks:
        if pick.evaluation_status == "rejected":
            continue
        wid = pick.waveform_id
        if wid is None:
            continue
        if wid.network_code == net and wid.station_code == sta and pick.phase_hint == phase:
            return pick
    return None


def _pick_window(
    event: Event, net: str, sta: str, ml_cfg: MlConfig
) -> tuple[UTCDateTime, UTCDateTime] | None:
    """The measurement window for ``net.sta``, or ``None`` if no P/S pick exists.

    ``[S - window_before_s, S + window_after_s]`` if an S pick exists;
    otherwise the fallback ``[P + 1.0, P + 8.0]`` coda window (logged), per
    this task's design.
    """
    s_pick = _find_pick(event, net, sta, "S")
    if s_pick is not None:
        return (
            s_pick.time - ml_cfg.window_before_s,
            s_pick.time + ml_cfg.window_after_s,
        )

    p_pick = _find_pick(event, net, sta, "P")
    if p_pick is not None:
        logger.warning(
            "%s.%s: no S pick; using P-derived coda window [P+1, P+8] as a fallback",
            net,
            sta,
        )
        return p_pick.time + 1.0, p_pick.time + 8.0

    return None


# ---------------------------------------------------------------------------
# per-channel measurement
# ---------------------------------------------------------------------------


def _covers_window(sliced: Trace, window_start: UTCDateTime, window_end: UTCDateTime) -> bool:
    if sliced.stats.npts == 0:
        return False
    tol = _COVERAGE_TOL_SAMPLES / sliced.stats.sampling_rate
    if sliced.stats.starttime > window_start + tol or sliced.stats.endtime < window_end - tol:
        return False
    return not np.ma.is_masked(sliced.data)


def _simulate_wood_anderson(trace: Trace, response, ml_cfg: MlConfig) -> Trace:
    """The 2-step Wood-Anderson simulation (module docstring, steps 3a-3c).

    Applied to a *copy* of ``trace`` over its full extent (not sliced to any
    window -- these are FFT-based filters; see the module docstring).
    """
    out = trace.copy()
    out.detrend("demean")
    out.detrend("linear")
    out.taper(max_percentage=ml_cfg.taper_pct)
    out.stats.response = response
    out.remove_response(output="DISP", water_level=60, pre_filt=tuple(ml_cfg.pre_filt))
    paz_wa = dict(WOOD_ANDERSON_PAZ)
    paz_wa["sensitivity"] = ml_cfg.wa_sensitivity
    out.simulate(paz_simulate=paz_wa, simulate_sensitivity=True)
    return out


def _measure_one_channel(
    trace: Trace,
    inventory: Inventory,
    window_start: UTCDateTime,
    window_end: UTCDateTime,
    ml_cfg: MlConfig,
) -> float | None:
    """Measure one channel's WA zero-to-peak amplitude in the window.

    Returns ``None`` (logging why) for any of this task's "skip" conditions
    (module docstring): no coverage, a gap, a flat window, or no response.
    """
    measured = _measure_one_channel_full(trace, inventory, window_start, window_end, ml_cfg)
    return None if measured is None else measured[0]


def _measure_one_channel_full(
    trace: Trace,
    inventory: Inventory,
    window_start: UTCDateTime,
    window_end: UTCDateTime,
    ml_cfg: MlConfig,
) -> tuple[float, Trace, UTCDateTime] | None:
    """:func:`_measure_one_channel`, also returning the processed trace and
    the peak sample's time: ``(amplitude_m, wa_trace, peak_time)``."""
    seed_id = trace.id

    raw_window = trace.slice(window_start, window_end)
    if not _covers_window(raw_window, window_start, window_end):
        logger.warning(
            "%s: no waveform data covering window [%s, %s]; skipping",
            seed_id,
            window_start,
            window_end,
        )
        return None

    if np.ptp(np.asarray(raw_window.data)) == 0:
        logger.warning(
            "%s: flat-lined window [%s, %s]; skipping",
            seed_id,
            window_start,
            window_end,
        )
        return None

    response = _get_cached_response(inventory, seed_id, trace.stats.starttime)
    if response is None:
        logger.warning("%s: no response in inventory; skipping", seed_id)
        return None

    processed = _simulate_wood_anderson(trace, response, ml_cfg)
    windowed = processed.slice(window_start, window_end)
    if windowed.stats.npts == 0:
        return None
    abs_data = np.abs(np.asarray(windowed.data))
    i_peak = int(np.argmax(abs_data))
    peak_time = windowed.stats.starttime + i_peak * windowed.stats.delta
    return float(abs_data[i_peak]), processed, peak_time


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


def _find_station_and_channel(inventory: Inventory, net: str, sta: str, loc: str, cha: str):
    """``(channel, station)`` `Inventory` objects for a seed id, or ``(None, None)``."""
    for network in inventory.networks:
        if network.code != net:
            continue
        for station in network.stations:
            if station.code != sta:
                continue
            for channel in station.channels:
                if channel.location_code == loc and channel.code == cha:
                    return channel, station
            return None, station
    return None, None


def _hypocentral_distance_km(origin, inventory: Inventory, trace: Trace) -> float | None:
    """3-D hypocentral distance (km) from ``origin`` to ``trace``'s station.

    Combines the epicentral distance (`gps2dist_azimuth`) with the vertical
    separation between the origin's depth and the station's *effective*
    elevation (``elevation_m - depth_m``, ARCHITECTURE.md §6) via
    Pythagoras. ``None`` if the station/channel can't be found in
    ``inventory`` (should not happen for a channel that already passed
    :func:`_measure_one_channel`'s response check, but defensive).
    """
    channel, station = _find_station_and_channel(
        inventory,
        trace.stats.network,
        trace.stats.station,
        trace.stats.location,
        trace.stats.channel,
    )
    if station is None or station.latitude is None or station.longitude is None:
        return None

    depth_m = channel.depth if channel is not None and channel.depth is not None else 0.0
    elevation_m = station.elevation if station.elevation is not None else 0.0
    effective_elevation_m = elevation_m - depth_m

    horizontal_m, _az, _baz = gps2dist_azimuth(
        origin.latitude, origin.longitude, station.latitude, station.longitude
    )
    horizontal_km = horizontal_m / 1000.0
    vertical_km = (origin.depth / 1000.0) + (effective_elevation_m / 1000.0)
    return math.hypot(horizontal_km, vertical_km)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def _group_by_station(stream: Stream) -> dict[tuple[str, str], list[Trace]]:
    groups: dict[tuple[str, str], list[Trace]] = {}
    for trace in stream:
        key = (trace.stats.network, trace.stats.station)
        groups.setdefault(key, []).append(trace)
    return groups


def measure_event_amplitudes(
    event: Event,
    stream: Stream,
    inventory: Inventory,
    cfg: MagnitudeConfig,
    *,
    capture: dict[tuple[str, str], WoodAndersonCapture] | None = None,
) -> list[StationAmplitude]:
    """Measure per-station Wood-Anderson amplitudes for one event.

    Pure function: no file I/O beyond what ``stream``/``inventory`` already
    hold, so it is directly unit-testable and safe to call once per event
    inside `openmeq.parallel.pmap` (M8.3 wires that).

    Parameters
    ----------
    event : obspy.core.event.Event
        The event, with its (preferred, or only) `Origin` and its `Pick`\\ s
        -- from whichever catalogue the caller selects (M8.3; e.g.
        ``review/catalog_final.xml`` or ``locate/catalog_auto.xml``, per
        PLAN.md §6 Step 7).
    stream : obspy.Stream
        The event's raw waveform chunk (``chunks/<event_id>/<event_id>
        .mseed``, ARCHITECTURE.md §3 -- all channels, original sampling,
        unprocessed).
    inventory : obspy.Inventory
        The merged station inventory with full instrument responses
        (``openmeq.stations.load_inventory``); parsed once by the caller and
        reused across every event a worker processes (see the module
        docstring's "Response cache" section).
    cfg : openmeq.config.MagnitudeConfig
        ``cfg.magnitude`` (this function reads ``cfg.ml.*``).
    capture : dict or None, keyword-only, optional
        If given, filled in place with one :class:`WoodAndersonCapture` per
        returned station, keyed ``(net, sta)`` (extended QAQC only; see the
        module docstring's "Trace capture" section).

    Returns
    -------
    list of StationAmplitude
        One entry per station with a usable measurement. Stations with no
        pick, no response, no data covering the window, or a flat window
        are skipped (logged, never raised) -- see the module docstring's
        "Skips" section.
    """
    ml_cfg = cfg.ml

    origin = event.preferred_origin()
    if origin is None:
        origin = event.origins[0] if event.origins else None
    if origin is None:
        logger.warning("Event %s has no origin; cannot measure amplitudes", event.resource_id)
        return []

    results: list[StationAmplitude] = []

    for (net, sta), traces in _group_by_station(stream).items():
        window = _pick_window(event, net, sta, ml_cfg)
        if window is None:
            logger.warning("%s.%s: no P or S pick; skipping amplitude measurement", net, sta)
            continue
        window_start, window_end = window

        horizontal_traces = [t for t in traces if t.stats.channel[-1] in _HORIZONTAL_LAST_CHARS]
        vertical_traces = [t for t in traces if t.stats.channel[-1] == "Z"]

        # (trace, amplitude_m, wa_trace, peak_time) per usable channel.
        horizontal_hits: list[tuple[Trace, float, Trace, UTCDateTime]] = []
        for trace in horizontal_traces:
            measured = _measure_one_channel_full(trace, inventory, window_start, window_end, ml_cfg)
            if measured is not None:
                horizontal_hits.append((trace, *measured))

        vertical_hit: tuple[Trace, float, Trace, UTCDateTime] | None = None
        for trace in vertical_traces:
            measured = _measure_one_channel_full(trace, inventory, window_start, window_end, ml_cfg)
            if measured is not None:
                vertical_hit = (trace, *measured)
                break

        if horizontal_hits:
            chosen_trace, amplitude_m, wa_trace, peak_time = max(
                horizontal_hits, key=lambda hit: hit[1]
            )
        elif vertical_hit is not None:
            chosen_trace, amplitude_m, wa_trace, peak_time = vertical_hit
            if horizontal_traces:
                logger.warning(
                    "%s.%s: no usable horizontal component in window; falling back to Z (%s)",
                    net,
                    sta,
                    chosen_trace.stats.channel,
                )
            else:
                logger.info(
                    "%s.%s: no horizontal channels present; using Z (%s) as a fallback",
                    net,
                    sta,
                    chosen_trace.stats.channel,
                )
        else:
            logger.warning("%s.%s: no usable component; skipping", net, sta)
            continue

        distance_km = _hypocentral_distance_km(origin, inventory, chosen_trace)
        if distance_km is None:
            logger.warning(
                "%s.%s: station not found in inventory for distance calculation; skipping",
                net,
                sta,
            )
            continue

        results.append(
            StationAmplitude(
                net=net,
                sta=sta,
                amplitude_m=amplitude_m,
                distance_km=distance_km,
                component_used=chosen_trace.stats.channel,
                window_start=window_start,
                window_end=window_end,
            )
        )
        if capture is not None:
            capture[(net, sta)] = WoodAndersonCapture(
                net=net,
                sta=sta,
                channel=chosen_trace.stats.channel,
                trace=wa_trace,
                window_start=window_start,
                window_end=window_end,
                peak_time=peak_time,
                amplitude_m=amplitude_m,
            )

    return results
