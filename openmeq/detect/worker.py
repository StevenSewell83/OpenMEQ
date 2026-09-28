"""Per-day network detection worker (M2.1, ``OpenMEQ-ujx.1``,
ARCHITECTURE.md §2/§4.3, PLAN.md §6 Step 2).

:func:`detect_window` is the top-level, picklable unit of parallel work
PLAN.md §4's "Detect" row describes ("day (all stations)"): given one time
window (usually one calendar day, per ``cfg.parallel.chunk_days``), it reads
that window's data (plus a symmetric pad, ``cfg.detection.day_overlap_s``,
so a trigger straddling the window boundary is not lost), excludes any
channel flagged not-``ok`` for its calendar date in ``qc/channel_flags.json``,
filters, runs recursive STA/LTA, and returns ObsPy's
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
**Per-day flagged-channel exclusion across a possibly multi-day, padded
read.** A window (plus its pad) can cover more than one calendar date --
either because ``cfg.parallel.chunk_days > 1``, or simply because the pad
pushes a few seconds into the previous/next day at the boundary of a
1-day window. ``qc/channel_flags.json`` (ARCHITECTURE.md §4.2) flags a
channel *per calendar day*, and a channel bad on one day but good on an
adjacent day of the same window must still contribute its good day's data
(this task's design note, ARCHITECTURE.md §4.2: "A channel/day with no
entry counts as ``ok: false``"). This module resolves that by reading one
sub-window *per calendar date* covered by the padded read, each with that
date's own usable-seed_id subset, and concatenating (unioning) the
resulting per-date streams before merging -- rather than reading one
superset stream and trying to retroactively cut per-channel per-date
pieces out of it. See :func:`_read_window` for the implementation.

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
"""

from __future__ import annotations

import json
import time
from datetime import timedelta

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.signal.trigger import coincidence_trigger

from openmeq.catalog import iso_z
from openmeq.config import Config, load_config
from openmeq.index import DataIndex
from openmeq.log import get_logger

logger = get_logger("detect.worker")


def _iso_z(t: UTCDateTime) -> str:
    """``UTCDateTime`` -> ISO-8601 string with a ``Z`` suffix (ARCHITECTURE.md §1)."""
    return iso_z(t)


def _dates_covered(t1: UTCDateTime, t2: UTCDateTime) -> list[str]:
    """Every calendar date (``"YYYY-MM-DD"``, UTC) that ``[t1, t2]`` touches."""
    d = t1.date
    end_d = t2.date
    dates = []
    while d <= end_d:
        dates.append(d.isoformat())
        d = d + timedelta(days=1)
    return dates


def _usable_seed_ids(all_seed_ids: list[str], flags: dict, date_str: str) -> list[str]:
    """Seed ids usable on ``date_str`` per ``channel_flags["flags"]``.

    ARCHITECTURE.md §4.2: "A channel/day with no entry counts as
    ``ok: false`` (no data)" -- so a seed id absent from ``flags``, or
    present but missing ``date_str``, or present with ``ok`` anything other
    than ``True``, is excluded. Sorted for deterministic read order.
    """
    usable = [
        seed_id
        for seed_id in all_seed_ids
        if flags.get(seed_id, {}).get(date_str, {}).get("ok") is True
    ]
    return sorted(usable)


def _read_window(
    index: DataIndex,
    flags: dict,
    t1: UTCDateTime,
    t2: UTCDateTime,
    pad: float,
) -> Stream:
    """Read ``[t1-pad, t2+pad]`` with per-calendar-date flagged-channel exclusion.

    See the module docstring's "Per-day flagged-channel exclusion" design
    note: one :meth:`DataIndex.read` call per calendar date the padded
    window covers, each restricted to that date's own usable seed ids, then
    concatenated. A date with zero usable channels contributes nothing (no
    wasted read).
    """
    read_start = t1 - pad
    read_end = t2 + pad

    all_seed_ids = index.seed_ids()
    stream = Stream()
    for date_str in _dates_covered(read_start, read_end):
        day_start = UTCDateTime(date_str)
        day_end = day_start + 86400.0
        sub_start = max(read_start, day_start)
        sub_end = min(read_end, day_end)
        if sub_start >= sub_end:
            continue
        usable = _usable_seed_ids(all_seed_ids, flags, date_str)
        if not usable:
            continue
        stream += index.read(usable, sub_start, sub_end, pad=0.0)

    return stream


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


def detect_window(
    window_start: str,
    window_end: str,
    index_path: str,
    flags_path: str,
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
    flags_path : str
        Path to ``qc/channel_flags.json`` (ARCHITECTURE.md §4.2).
    cfg_path : str
        Path to the project's YAML config (:func:`openmeq.config.load_config`).

    Returns
    -------
    list of dict
        One dict per raw trigger (unsorted, undeduplicated across windows --
        that's M2.2's job), each ``{"time": iso_z_string, "duration_s": float,
        "coincidence_sum": float, "n_stations": int, "stations": [...],
        "trace_ids": [...]}`` (ARCHITECTURE.md §4.3). ``[]`` if no channel is
        usable for this window (every candidate flagged not-``ok``) or no
        trigger is found.
    """
    t0 = time.perf_counter()
    t1 = UTCDateTime(window_start)
    t2 = UTCDateTime(window_end)

    index = _cached_index(index_path)
    with open(flags_path, encoding="utf-8") as f:
        channel_flags = json.load(f)
    flags = channel_flags.get("flags", {})
    cfg = _cached_cfg(cfg_path)
    det_cfg = cfg.detection

    stream = _read_window(index, flags, t1, t2, pad=det_cfg.day_overlap_s)
    if len(stream) == 0:
        logger.info(
            "detect_window %s..%s: no usable data (%.2fs)",
            window_start,
            window_end,
            time.perf_counter() - t0,
        )
        return []

    stream.merge(method=0)
    stream = stream.split()
    if len(stream) == 0:
        return []

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

    trace_ids = _build_trace_ids(stream, det_cfg.station_weights)

    raw_triggers = coincidence_trigger(
        "recstalta",
        det_cfg.trigger_on,
        det_cfg.trigger_off,
        stream,
        det_cfg.coincidence_sum,
        trace_ids=trace_ids,
        sta=det_cfg.sta,
        lta=det_cfg.lta,
        details=True,
    )
    triggers = [_normalize_trigger(t) for t in raw_triggers]

    logger.info(
        "detect_window %s..%s: %d trace(s), %d trigger(s) (%.2fs)",
        window_start,
        window_end,
        len(stream),
        len(triggers),
        time.perf_counter() - t0,
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
