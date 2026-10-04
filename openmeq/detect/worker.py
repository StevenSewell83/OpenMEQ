"""Per-day network detection worker (M2.1, ``OpenMEQ-ujx.1``,
ARCHITECTURE.md §2/§4.3, PLAN.md §6 Step 2).

:func:`detect_window` is the top-level, picklable unit of parallel work
PLAN.md §4's "Detect" row describes ("day (all stations)"): given one time
window (usually one calendar day, per ``cfg.parallel.chunk_days``), it reads
that window's data (plus a symmetric pad, ``cfg.detection.day_overlap_s``,
so a trigger straddling the window boundary is not lost) from every indexed
channel except ``cfg.qc.exclude_channels`` -- QC flags are a report and do
not gate detection (OpenMEQ-7c4) -- filters, runs recursive STA/LTA, and returns ObsPy's
:func:`~obspy.signal.trigger.coincidence_trigger` result across stations as a
list of plain, JSON-serializable dicts.

:func:`build_windows` splits ``[cfg.data.start, cfg.data.end]`` (inclusive,
per ``config/example.yaml``'s convention) into ``cfg.parallel.chunk_days``-day
windows -- the list of ``(window_start, window_end)`` ISO strings that
``openmeq.parallel.pmap``/M2.3's CLI will eventually map :func:`detect_window`
over.

Out of scope (M2.2, ``OpenMEQ-ujx.2``): merging/de-duplicating triggers
across adjacent windows and assigning event ids. This module's output is
deliberately raw and unsorted-across-windows; a trigger near a window
boundary legitimately appears in *both* of its adjacent windows' outputs
(that redundancy is exactly what proves the overlap pad works -- see
:func:`detect_window`'s "Overlap padding" note below -- and de-duplicating
it is the next task's job).

Design notes
------------
**No QC gating (OpenMEQ-7c4).** Detection reads every channel the index
has data for, across the whole padded window in one
:meth:`DataIndex.read` call, dropping only ``cfg.qc.exclude_channels``.
``qc/channel_flags.json`` is a report for the user, not an input here: a
per-day "ok" gate turned out to drop whole real-network runs when a QC
threshold didn't suit the data (RK-NM 2026 Q1: every channel-day flagged).
See :func:`_read_window`.

**Overlap padding size (``cfg.detection.day_overlap_s``, default 60.0 s).**
Must comfortably exceed ``cfg.detection.lta`` plus a trigger's typical
on-to-off duration, so that an event straddling a window boundary is fully
contained -- STA/LTA warm-up included -- within *both* adjacent windows'
padded reads. With the PLAN.md §3 defaults (``lta: 1.0`` s, and coincidence
triggers for small local events typically lasting at most a second or two
once ``thr_off`` is crossed), 60 s leaves roughly two orders of magnitude of
headroom; the true lower bound needed is on the order of a few seconds, so
this default is intentionally generous rather than tightly tuned, since the
padded read is cheap relative to the rest of the step.

**``stream.split()`` before filtering.** :meth:`openmeq.index.DataIndex.read`
merges with ``method=0`` (ObsPy's default), which produces masked arrays
across any gap rather than filling it. Recursive STA/LTA and the bandpass
filter cannot operate on a masked array (see the pitfall this module works
around: a naive ``stream.filter()`` on a still-masked trace either raises or
silently corrupts the characteristic function across the gap). Calling
``stream.split()`` first breaks every gappy trace into its contiguous,
unmasked segments, so a gap simply produces two independent
trigger-candidate segments instead.

**Flagged-channel verification.** ``test_flagged_channel_excluded``
(``tests/test_detect_worker.py``) verifies exclusion by spying on the
``Stream`` object actually passed into
:func:`obspy.signal.trigger.coincidence_trigger` (via monkeypatching), not
merely by inspecting the final trigger list -- a flagged-but-otherwise-quiet
station might never appear in a trigger list anyway, which would make that
weaker check pass for the wrong reason.

**``coincidence_trigger``'s ``trace_ids`` weighting parameter (verified
against the installed ObsPy 1.5.1 source,
``obspy/signal/trigger.py:coincidence_trigger``, not assumed from training
data or from PLAN.md §3's config sketch).** ``trace_ids``, when given as a
dict, must be keyed by each trace's **full** SEED id (``tr.id``, i.e.
``NET.STA.LOC.CHA``) -- *not* by ``NET.STA`` -- because the function's inner
loop does ``if tr.id not in trace_ids: continue`` (with a ``UserWarning``)
before any triggering happens for that trace. ``cfg.detection.station_weights``
is documented (``config/example.yaml``) as an optional ``{"NET.STA": weight}``
mapping, so passing it straight through as ``trace_ids`` would silently
disregard *every* trace in the stream whose full seed id isn't itself a key
of that dict -- i.e. every trace, in practice, since ``NET.STA`` is never a
valid ``tr.id``. :func:`_build_trace_ids` instead expands
``station_weights`` into a full ``{tr.id: weight}`` dict covering *every*
trace actually present in the (already flag-filtered) stream, defaulting to
weight ``1.0`` for any trace whose station has no entry in
``station_weights`` -- this preserves "every present, unflagged trace counts
toward the coincidence sum" (``coincidence_trigger``'s own documented
default behaviour for ``trace_ids=None``) while still letting
``station_weights`` up- or down-weight specific stations. When
``station_weights`` is empty (the default), ``None`` is passed straight
through, which is cheaper and behaviourally identical (every present trace,
weight 1).

**Zero usable channels -> ``[]``, never a raise.** If every channel is
flagged not-``ok`` for every date the (padded) window covers, the read
returns an empty stream, and this function returns ``[]`` immediately
without calling ``coincidence_trigger`` at all (an empty stream would also
make ``coincidence_trigger`` return ``[]`` on its own, per its source, but
returning early avoids a wasted filter/trigger pass).

**Performance.** Filtering happens exactly once per (already gap-split)
trace, via one ``stream.filter(...)`` call over the whole stream (ObsPy
dispatches this to one call per trace internally) -- never per-window
duplicate filtering of the same samples. Trace data is cast to
``numpy.float32`` immediately after ``stream.split()`` (``detrend``/
``bandpass``/etc. all accept and preserve a ``float32`` input, per direct
testing against the installed ObsPy/SciPy: ``scipy.signal.detrend``
preserves the input dtype, and even though ``obspy.signal.filter.bandpass``
itself promotes its *output* to ``float64`` -- and
:func:`~obspy.signal.trigger.recursive_sta_lta` always internally converts
its input to ``float64`` via ``numpy.ascontiguousarray`` regardless of what
we pass it, so a filtered float32 stream and a float64 one are numerically
identical to it), so the trace data actually held in memory across the
merge/split/detrend/filter chain is float32 rather than the float64 default,
roughly halving peak memory for that chain on a typically many-channel,
day-long stream. The filtered stream is cast back to float32 (from
filter's own float64 output) before ``coincidence_trigger`` for the same
reason -- ``coincidence_trigger`` triggers on a **copy** of each trace
internally, so this halves the *steady-state* memory the still-referenced,
filtered ``stream`` object occupies while that per-trace triggering loop
runs.

**Fast linear detrend (M11.3, ``OpenMEQ-m1f.3``).** cProfile against one
full day-window (25 traces, 200 Hz synthetic data) showed
``stream.detrend("linear")`` costing **~49% of the whole
:func:`detect_window` call** (~7.0s of 14.4s) -- almost entirely
``scipy.signal.detrend``'s own ``scipy.linalg.lstsq`` (SVD-based) call, a
generic solver built for arbitrary-column design matrices and multiple
breakpoints, neither of which this call site ever uses (ObsPy's
``Trace.detrend("linear")`` always calls it with zero breakpoints, i.e. one
single-segment ordinary-least-squares fit of a straight line against the
sample index). :func:`_fast_linear_detrend` replaces it with a closed-form
OLS solve (two ``float64``-accumulated reductions -- ``sum(x*y)`` and
``sum(y)``, via a mean-centred index so ``sum(x) == 0`` and the normal
equations collapse to ``slope = sum(x*y) / sum(x**2)``,
``intercept = mean(y)`` -- no matrix factorization at all). Least-squares
linear regression is invariant to an affine reparametrization of its single
predictor, so this fits the mathematically *identical* line
``scipy.signal.detrend(data, type="linear")`` does; measured against it
directly (``tests/test_detect_fast_detrend.py``) the two agree to within
~1e-5 relative amplitude on a full 17.28M-sample float32 channel-day (the
residual is float64-accumulation rounding, not an algorithm difference), a
downstream-irrelevant difference against a bandpass/STA-LTA pipeline that
already discards DC/near-DC content. Measured ~2-4x faster than
``scipy.signal.detrend`` at this array size (float32/float64
respectively) -- see ``docs/performance.md`` for the full before/after
``detect_window`` numbers.

**Per-worker-process caching of ``DataIndex``/``Config`` (M11.3).** Mirrors
``openmeq.qc.stats``'s and ``openmeq.chunk``'s identical
``_cached_index``/``_cached_*_cfg`` pattern (see either module's docstring
for the full rationale): a long-lived ``ProcessPoolExecutor`` worker handles
many :func:`detect_window` calls (one per window) over its lifetime, all
sharing the same ``index_path``/``cfg_path`` -- without caching, every
single window re-parsed the whole ``inventory/index.json`` and
re-validated/re-loaded the entire project YAML from scratch. This was not
yet applied to this module before M11.3 (unlike ``qc``/``chunk``, which
already had it); at production scale (e.g. one window per day over a
multi-month survey) this is a per-run fixed cost proportional to the number
of windows, not to the data volume, so it does not show up in a
per-channel-day profile but does shorten real wall-clock time.

**Per-station triggers for extended QAQC (``OpenMEQ-02o.2``).**
:func:`detect_window_with_station_triggers` runs exactly the same
read/preprocess/``coincidence_trigger`` pass as :func:`detect_window` (both
delegate to :func:`_detect`), and additionally returns every *single-station*
STA/LTA trigger -- the per-trace on/off intervals ``coincidence_trigger``
computes internally but never exposes. They are recomputed by
:func:`_station_triggers` in a separate per-trace pass (the characteristic
function via :func:`characteristic_function`, i.e. the very same
``Trace.trigger("recstalta", sta=..., lta=...)`` call ``coincidence_trigger``
makes on its own internal copy, then ``trigger_onset`` with
``coincidence_trigger``'s own default ``max_len``/``max_len_delete``), rather
than by handing ``coincidence_trigger`` a precomputed-CF stream: the
``coincidence_trigger`` call itself stays unchanged, so the merged detections
are identical whether or not extended diagnostics are on, and only one
trace's CF is held in memory at a time. The extra cost (one recursive
STA/LTA -- a C loop -- per trace; the expensive detrend/filter pass is
shared) is only paid when ``cfg.diagnostics.extended_enabled("detect")``. A
station trigger is kept only if its on-time falls inside the *unpadded*
``[window_start, window_end)`` window, so adjacent windows' overlap pads
never duplicate one: unlike the coincidence triggers, the station-trigger
list needs no downstream de-duplication.
"""

from __future__ import annotations

import time

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.signal.trigger import coincidence_trigger, trigger_onset

from openmeq.catalog import iso_z
from openmeq.config import Config, load_config
from openmeq.index import DataIndex
from openmeq.log import get_logger

logger = get_logger("detect.worker")


def _iso_z(t: UTCDateTime) -> str:
    """``UTCDateTime`` -> ISO-8601 string with a ``Z`` suffix (ARCHITECTURE.md §1)."""
    return iso_z(t)


def _read_window(
    index: DataIndex,
    exclude_channels: list[str],
    t1: UTCDateTime,
    t2: UTCDateTime,
    pad: float,
) -> Stream:
    """Read ``[t1-pad, t2+pad]`` for every indexed channel except ``exclude_channels``.

    QC flags do not gate detection (OpenMEQ-7c4): every channel with data
    in the window is used, and only the user's own ``cfg.qc.exclude_channels``
    list is dropped. Seed ids are sorted for a deterministic read order.
    """
    excluded = set(exclude_channels)
    seed_ids = sorted(sid for sid in index.seed_ids() if sid not in excluded)
    if not seed_ids:
        return Stream()
    return index.read(seed_ids, t1, t2, pad=pad)


def _fast_linear_detrend(data: np.ndarray) -> np.ndarray:
    """Closed-form single-segment linear-regression detrend (M11.3 optimisation).

    Drop-in replacement for ``scipy.signal.detrend(data, type="linear")``
    (what ``obspy.core.trace.Trace.detrend("linear")`` calls internally with
    zero breakpoints) -- see the module docstring's "Fast linear detrend"
    section for the full profiling rationale and measured speedup/tolerance.

    Fits ``y = slope * x + intercept`` by ordinary least squares against the
    sample index ``x = arange(n)``, mean-centred so ``sum(x) == 0`` (which
    both simplifies the normal equations to two reductions and avoids the
    large-magnitude cancellation a raw ``arange(n)`` parametrization would
    hit for a multi-million-sample array), then returns ``data - trend``.
    Least-squares regression against a single predictor is invariant to any
    affine reparametrization of that predictor, so the fitted line -- and
    therefore the detrended output -- is mathematically identical to
    scipy's own ``lstsq``-based fit; the two differ only by
    floating-point-accumulation rounding (see
    ``tests/test_detect_fast_detrend.py``).

    Parameters
    ----------
    data : numpy.ndarray
        1-D array (a ``Trace.data`` array). Not modified in place; a new
        array of the same dtype is returned (matching
        ``scipy.signal.detrend``'s own return-a-new-array contract, which
        ``obspy.core.trace.Trace.detrend`` then assigns back to
        ``self.data``).

    Returns
    -------
    numpy.ndarray
        Detrended data, same shape and dtype as ``data``. A constant
        (zero-variance index, i.e. ``n <= 1``) array is returned demeaned
        only (slope forced to 0), matching scipy's own behaviour for a
        single-sample segment.
    """
    orig_dtype = data.dtype
    n = data.shape[0]
    if n == 0:
        return data.copy()
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    sum_x2 = np.dot(x, x)
    sum_y = float(data.sum(dtype=np.float64))
    slope = 0.0
    if sum_x2 != 0.0:
        sum_xy = np.dot(x, data.astype(np.float64, copy=False))
        slope = sum_xy / sum_x2
    intercept = sum_y / n
    trend = (slope * x + intercept).astype(orig_dtype, copy=False)
    return data - trend


def _apply_filter(stream: Stream, filter_cfg) -> None:
    """Filter ``stream`` in place per ``cfg.detection.filter`` (one call, one pass per trace).

    ``obspy.core.stream.Stream.filter``/``Trace.filter`` dispatch on
    ``type`` to a different function signature per ObsPy's own filter
    registry: ``bandpass``/``bandstop`` take ``freqmin``/``freqmax``;
    ``highpass``/``lowpass`` take a single ``freq``.
    ``openmeq.config.DetectionFilterConfig`` only carries ``freqmin``/
    ``freqmax`` (matching PLAN.md §3's sketch and the ``bandpass`` default),
    so for the two single-corner types this maps ``freqmin`` (highpass) or
    ``freqmax`` (lowpass) onto ``freq``.
    """
    if filter_cfg.type in ("bandpass", "bandstop"):
        stream.filter(filter_cfg.type, freqmin=filter_cfg.freqmin, freqmax=filter_cfg.freqmax)
    elif filter_cfg.type == "highpass":
        stream.filter(filter_cfg.type, freq=filter_cfg.freqmin)
    else:  # lowpass
        stream.filter(filter_cfg.type, freq=filter_cfg.freqmax)


def _build_trace_ids(stream: Stream, station_weights: dict[str, float]) -> dict[str, float] | None:
    """``coincidence_trigger``'s ``trace_ids`` dict, keyed by full seed id.

    See the module docstring's "``coincidence_trigger``'s ``trace_ids``"
    design note for why this expansion (from ``{"NET.STA": weight}`` to
    ``{seed_id: weight}``, covering every trace currently in ``stream``,
    default weight ``1.0``) is required rather than passing
    ``station_weights`` straight through. Returns ``None`` (ObsPy's own
    "use every trace, weight 1" default) when ``station_weights`` is empty,
    which is both cheaper and behaviourally identical.
    """
    if not station_weights:
        return None
    return {
        tr.id: station_weights.get(f"{tr.stats.network}.{tr.stats.station}", 1.0) for tr in stream
    }


def _normalize_trigger(trig: dict) -> dict:
    """One ``coincidence_trigger`` result dict -> the ARCHITECTURE.md §4.3 shape.

    ``coincidence_trigger``'s own ``event["stations"]`` field (verified
    against the installed ObsPy source) is only the bare station *code*
    (``tr_id.split(".")[1]``, no network) and is not deduplicated (two
    channels of the same station triggering both add an entry) -- neither
    matches this task's required ``"NET.STA"`` / deduplicated-sorted-list
    contract, so both ``stations`` and ``n_stations`` are rederived here
    from ``trace_ids`` instead of trusted from the raw trigger dict.
    """
    trace_ids = sorted(trig["trace_ids"])
    stations = sorted({".".join(tid.split(".")[:2]) for tid in trace_ids})
    return {
        "time": _iso_z(trig["time"]),
        "duration_s": float(trig["duration"]),
        "coincidence_sum": float(trig["coincidence_sum"]),
        "n_stations": len(stations),
        "stations": stations,
        "trace_ids": trace_ids,
    }


#: Per-worker-*process* caches, keyed by the path string each worker was
#: called with -- see the module docstring's "Per-worker-process caching"
#: (M11.3) note, and ``openmeq.qc.stats``'s identical caches for the full
#: rationale.
_worker_index_cache: dict[str, DataIndex] = {}
_worker_cfg_cache: dict[str, Config] = {}


def _cached_index(index_path: str) -> DataIndex:
    index = _worker_index_cache.get(index_path)
    if index is None:
        index = DataIndex.load(index_path)
        _worker_index_cache[index_path] = index
    return index


def _cached_cfg(cfg_path: str) -> Config:
    cfg = _worker_cfg_cache.get(cfg_path)
    if cfg is None:
        cfg = load_config(cfg_path)
        _worker_cfg_cache[cfg_path] = cfg
    return cfg


#: ``coincidence_trigger``'s own ``max_trigger_length`` default (seconds,
#: verified against the installed ObsPy 1.5.1 signature) -- reused by
#: :func:`_station_triggers` so a station trigger is cut exactly where
#: ``coincidence_trigger`` itself would cut it.
_COINCIDENCE_MAX_TRIGGER_LENGTH_S = 1e6


def preprocess_stream(stream: Stream, det_cfg) -> Stream:
    """Merge, gap-split, detrend and filter ``stream`` exactly as detection does.

    The single preprocessing path shared by :func:`detect_window` and the
    extended-QAQC ``cf_examples`` figures (:mod:`openmeq.detect.extended`),
    so the plotted filtered trace/STA-LTA ratio cannot drift from what the
    detector actually triggered on. See the module docstring's
    "``stream.split()`` before filtering", "Performance" and "Fast linear
    detrend" notes for why each step is done the way it is.

    Parameters
    ----------
    stream : obspy.Stream
        Raw stream (as returned by :func:`_read_window`); consumed.
    det_cfg : openmeq.config.DetectionConfig

    Returns
    -------
    obspy.Stream
        The filtered, float32, gap-free-per-trace stream (possibly empty).
    """
    stream.merge(method=0)
    stream = stream.split()
    if len(stream) == 0:
        return stream

    for tr in stream:
        tr.data = tr.data.astype(np.float32)

    # M11.3: closed-form OLS detrend (see module docstring's "Fast linear
    # detrend" note) instead of ``stream.detrend("linear")`` (ObsPy ->
    # ``scipy.signal.detrend`` -> ``scipy.linalg.lstsq``), which cProfile
    # showed costing ~49% of this function's total time.
    for tr in stream:
        tr.data = _fast_linear_detrend(tr.data)
    _apply_filter(stream, det_cfg.filter)

    for tr in stream:
        tr.data = tr.data.astype(np.float32)
    return stream


def characteristic_function(tr, det_cfg) -> np.ndarray:
    """Recursive STA/LTA ratio of one preprocessed trace, as the detector computes it.

    Exactly the call ``coincidence_trigger("recstalta", ..., sta=, lta=)``
    makes on its internal copy of each trace (``Trace.trigger``, which
    converts ``sta``/``lta`` seconds to ``int(seconds * sampling_rate)``
    samples). ``tr`` itself is not modified.
    """
    return tr.copy().trigger("recstalta", sta=det_cfg.sta, lta=det_cfg.lta).data


def _station_triggers(stream: Stream, det_cfg, t1: UTCDateTime, t2: UTCDateTime) -> list[dict]:
    """Every single-trace STA/LTA trigger whose on-time is in ``[t1, t2)``.

    See the module docstring's "Per-station triggers" note. One dict per
    trigger: ``{"seed_id", "station" ("NET.STA"), "on_time", "off_time"
    (ISO-Z strings), "duration_s", "peak_ratio"}`` -- ``peak_ratio`` being
    the STA/LTA maximum over ``[on, off)`` (``coincidence_trigger``'s own
    ``cft_peak``). Sorted by ``(on_time, seed_id)`` for determinism.
    """
    rows = []
    for tr in stream:
        cft = characteristic_function(tr, det_cfg)
        sr = tr.stats.sampling_rate
        onsets = trigger_onset(
            cft,
            det_cfg.trigger_on,
            det_cfg.trigger_off,
            max_len=int(_COINCIDENCE_MAX_TRIGGER_LENGTH_S * sr + 0.5),
            max_len_delete=False,
        )
        for on, off in onsets:
            on_t = tr.stats.starttime + float(on) / sr
            if not (t1 <= on_t < t2):
                continue
            off_t = tr.stats.starttime + float(off) / sr
            peak = float(cft[on:off].max()) if off > on else float(cft[on])
            rows.append(
                {
                    "seed_id": tr.id,
                    "station": f"{tr.stats.network}.{tr.stats.station}",
                    "on_time": _iso_z(on_t),
                    "off_time": _iso_z(off_t),
                    "duration_s": float(off_t - on_t),
                    "peak_ratio": peak,
                }
            )
    rows.sort(key=lambda r: (r["on_time"], r["seed_id"]))
    return rows


def _detect(
    window_start: str,
    window_end: str,
    index_path: str,
    cfg_path: str,
    *,
    with_station_triggers: bool,
) -> tuple[list[dict], list[dict]]:
    """Shared body of :func:`detect_window`/:func:`detect_window_with_station_triggers`.

    Returns ``(triggers, station_triggers)``; ``station_triggers`` is always
    ``[]`` unless ``with_station_triggers``.
    """
    t0 = time.perf_counter()
    t1 = UTCDateTime(window_start)
    t2 = UTCDateTime(window_end)

    index = _cached_index(index_path)
    cfg = _cached_cfg(cfg_path)
    det_cfg = cfg.detection

    stream = _read_window(index, cfg.qc.exclude_channels, t1, t2, pad=det_cfg.day_overlap_s)
    if len(stream) == 0:
        logger.info(
            "detect_window %s..%s: no usable data (%.2fs)",
            window_start,
            window_end,
            time.perf_counter() - t0,
        )
        return [], []

    stream = preprocess_stream(stream, det_cfg)
    if len(stream) == 0:
        return [], []

    trace_ids = _build_trace_ids(stream, det_cfg.station_weights)

    raw_triggers = coincidence_trigger(
        "recstalta",
        det_cfg.trigger_on,
        det_cfg.trigger_off,
        stream,
        det_cfg.min_trace_coincidence,
        trace_ids=trace_ids,
        sta=det_cfg.sta,
        lta=det_cfg.lta,
        details=True,
    )
    triggers = [_normalize_trigger(t) for t in raw_triggers]
    # OpenMEQ-8a0: min_trace_coincidence counts traces; this is the station count.
    triggers = [t for t in triggers if t["n_stations"] >= det_cfg.min_stations]
    station_triggers = _station_triggers(stream, det_cfg, t1, t2) if with_station_triggers else []

    logger.info(
        "detect_window %s..%s: %d trace(s), %d trigger(s)%s (%.2fs)",
        window_start,
        window_end,
        len(stream),
        len(triggers),
        f", {len(station_triggers)} station trigger(s)" if with_station_triggers else "",
        time.perf_counter() - t0,
    )
    return triggers, station_triggers


def detect_window_with_station_triggers(
    window_start: str,
    window_end: str,
    index_path: str,
    cfg_path: str,
) -> tuple[list[dict], list[dict]]:
    """:func:`detect_window` plus every single-station STA/LTA trigger (extended QAQC).

    Same arguments as :func:`detect_window`; top-level and picklable. The
    first element is exactly :func:`detect_window`'s return value; the
    second is :func:`_station_triggers`'s list for the unpadded window (see
    the module docstring's "Per-station triggers" note).
    """
    return _detect(window_start, window_end, index_path, cfg_path, with_station_triggers=True)


def detect_window(
    window_start: str,
    window_end: str,
    index_path: str,
    cfg_path: str,
) -> list[dict]:
    """Detect triggers in one time window (top-level, picklable -- ``openmeq.parallel`` worker).

    Parameters
    ----------
    window_start, window_end : str
        ISO-8601 window bounds (``obspy.UTCDateTime``-parseable), *before*
        ``cfg.detection.day_overlap_s`` padding -- typically one
        :func:`build_windows` entry.
    index_path : str
        Path to ``inventory/index.json`` (:meth:`DataIndex.load`).
    cfg_path : str
        Path to the project's YAML config (:func:`openmeq.config.load_config`).

    Returns
    -------
    list of dict
        One dict per raw trigger (unsorted, undeduplicated across windows --
        that's M2.2's job), each ``{"time": iso_z_string, "duration_s": float,
        "coincidence_sum": float, "n_stations": int, "stations": [...],
        "trace_ids": [...]}`` (ARCHITECTURE.md §4.3). ``[]`` if no channel has
        data in this window (or every one is in ``cfg.qc.exclude_channels``)
        or no trigger is found.
    """
    triggers, _ = _detect(
        window_start, window_end, index_path, cfg_path, with_station_triggers=False
    )
    return triggers


def build_windows(cfg: Config) -> list[tuple[str, str]]:
    """Split ``[cfg.data.start, cfg.data.end]`` into ``cfg.parallel.chunk_days``-day windows.

    ``cfg.data.end`` is an inclusive calendar date (``config/example.yaml``:
    "inclusive, UTC calendar date"), so the covered range runs through the
    end of that day, i.e. up to (but not including) midnight the day after
    ``cfg.data.end``.

    Returns
    -------
    list of (str, str)
        ``(window_start, window_end)`` ISO-8601 ``...Z`` strings, contiguous
        and non-overlapping, each ``cfg.parallel.chunk_days`` days wide
        (the last one shorter if the range doesn't divide evenly).
    """
    step_s = cfg.parallel.chunk_days * 86400.0
    start = UTCDateTime(cfg.data.start)
    end = UTCDateTime(cfg.data.end) + 86400.0

    windows: list[tuple[str, str]] = []
    t = start
    while t < end:
        t_next = min(t + step_s, end)
        windows.append((_iso_z(t), _iso_z(t_next)))
        t = t_next
    return windows
