"""Parallel cross-correlation to ``dt.cc`` (M10.3 / ``OpenMEQ-16d.3``).

Plan key: M10.3 (PLAN.md §6 Step 8's "cross-correlation differential times
(dt.cc)"). This is the heaviest, most correctness-sensitive step in the
whole build (PLAN.md §4: "xcorr (dt.cc): event pair (batched) -- The
heaviest step") and a from-scratch rewrite of the hypoddpy template's
cross-correlation loop, guided by (not copied from) that template, fixing
every defect ``docs/hypoddpy_audit.md`` §2/§6 assigns to this task.

Pipeline shape
--------------
::

    relocate/work/dt.ct  (M10.2, ph2dt's linked-pair list + per-pair
                           per-station/phase detail -- re-read here, not
                           duplicated: `openmeq.relocate.ph2dt_step.run_ph2dt`
                           returns only the pair identities)
    relocate/cache/<event_id>.npz  (M10.1, pre-filtered/resampled windows)
        -> cross_correlate_pair (this module, one pure worker call per pair)
        -> relocate/work/xcorr_results.jsonl  (openmeq.parallel.pmap_incremental,
                                                resumable)
        -> relocate/work/dt.cc  (assembled from the results file)

The dt.cc sign convention (the confirmed defect this task exists to fix)
-------------------------------------------------------------------------
``docs/hypoddpy_audit.md`` §2.1 (ground-truthed against
``external/HYPODD/src/hypoDD/getdata.f:369-402`` and
``hypoDD2_UserGuide.pdf`` §B.3.2, "DT = T1-T2"): for a pair header
``# ID1 ID2 OTC``, hypoDD reads the ``dt.cc`` phase line's differential-time
field **verbatim** (only subtracting ``OTC``) -- it never recomputes the
sign the way it does for ``dt.ct`` (``dt_dt(i) = dt1 - dt2``,
``getdata.f:485``, matching the header's ``ID1, ID2`` order exactly). So the
value this module writes for header ``# i j`` must already equal
``TT(event i) - TT(event j)``, the same physical quantity/sign as ``dt.ct``'s
own ``dt1 - dt2``, for the two data types to be consistent double-difference
inputs.

``obspy.signal.cross_correlation.xcorr_pick_correction``'s own docstring
gives the exact recipe: *"To get a corrected differential travel time using
origin times for both events calculate: ((pick2 + pick2_corr - ot2) -
(pick1 - ot1))"* when called as ``xcorr_pick_correction(pick1, trace1, pick2,
trace2, ...)``. Calling it (or this module's own faster equivalent, see
below) with **event i's pick first**, the correct dt.cc value is::

    dt = (pick_i - origin_i) - (pick_j + pick2_corr - origin_j)   # TT_i - TT_j_corrected

The hypoddpy template computes the negated value (confirmed
``docs/hypoddpy_audit.md`` §2.1, live code at
``hypodd_relocator_parallel.py:1542-1544``, with the *correct* formula
immediately below, commented out -- strong evidence the sign was flipped at
some point after the correct version was written). :func:`cross_correlate_pair`
below implements the correct formula and :mod:`tests.test_relocate_xcorr`
proves it with a synthetic pair and a known, deliberately-introduced lag
(``test_sign_convention_recovers_known_shift``) -- a flipped-sign
implementation fails that test, not merely "looks plausible".

Canonical pair ordering
------------------------
Per this task's design note: ``i``/``j`` above are exactly the
``(cuspid1, cuspid2)`` order ``ph2dt`` itself wrote in ``dt.ct``'s ``#``
header lines (``openmeq.relocate.ph2dt_step.run_ph2dt``'s returned pair
list, in file order) -- this module never reorders a pair (e.g. "always
lower cuspid first"); it reuses whatever order ``dt.ct`` already
established, so ``dt.cc``'s header for a given pair matches ``dt.ct``'s
header for the same pair exactly (same ``i``/``j`` identity), which is what
makes the two files consistent double-difference inputs for the same pair.

Per-station/phase detail: re-read from ``dt.ct``, not re-derived
-------------------------------------------------------------------
``run_ph2dt`` returns only the pair identities (``(cuspid1, cuspid2)``
tuples); the *set of station+phase observations* linked for each pair is
``ph2dt``'s own (``MAXSEP``/``MAXNGH``/``MINLNK``-filtered) judgement of
which station+phase combinations are shared and useful for that pair, and
is re-parsed here directly from ``dt.ct`` (:func:`parse_dt_ct_detail`) --
this is more correct (matches exactly what ``ph2dt`` itself linked, not an
independently-reconstructed "picks present at both events" set that could
disagree with ``ph2dt``'s own ``MINLNK``/etc. filtering) and cheaper (no
per-pair re-scan of the whole catalogue).

P/S channel selection and the multi-channel S combination (audit defect #9)
-----------------------------------------------------------------------------
P is always cross-correlated on ``Z`` (``cfg.hypodd.xcorr.components["P"]``,
typically ``["Z"]``); S is tried independently on every configured
horizontal (``components["S"]``, typically ``1``/``2``/``N``/``E`` --
whichever the station's own mapped channels happen to be, per
``openmeq.stations.apply_channel_map``). Per this task's design note and
``docs/hypoddpy_audit.md`` §2.4/§6 defect #9: the hypoddpy template combines
multiple horizontal channels' independently-measured lags/coefficients via a
**fixed per-channel weighted average**, which can drag a good lag estimate
down (or bias it) using a poorly-correlating channel's independent
measurement. This module instead runs the cross-correlation on every
available channel and keeps the **single best-correlating channel's**
``(dt, cc)`` outright (the ``max`` over ``cc`` in
:func:`cross_correlate_pair`'s inner loop) -- simpler, standard practice, and
avoids blending two physically-independent lag measurements (a horizontal
channel's lag reflects that channel's own orientation-dependent waveform,
not a shared "true" lag with independent noise) as though they were noisy
estimates of one quantity. The discarded channels are silently dropped, not
averaged in.

Origin-time correction (OTC): always ``0.0``
----------------------------------------------
Every ``dt.cc`` header this module writes uses ``OTC = 0.0``
(:data:`_OTC`). ``docs/hypoddpy_audit.md`` §2.5's confirmed-correct reading
of ``getdata.f:398-399`` shows ``OTC`` exists to correct a per-pair
*origin-time* bias baked into the phase-line differential times; since every
``dt_s`` this module writes already subtracts each event's own origin time
independently (the sign-convention formula above), no additional per-pair
origin-time bias remains to correct, **provided** each event's own origin
time is itself accurate (true here -- origins come from NLLoc/hypoDD-quality
locations, not naive picks, per this task's design note). This matches the
hypoddpy template's own choice, which the audit found defensible under this
condition -- written here as an explicit, tested decision, not a silent copy.

Cross-correlation implementation: benchmarked, not assumed
-------------------------------------------------------------
This task called out a choice between ``obspy.signal.cross_correlation.
xcorr_pick_correction`` and a faster vectorised ``correlate`` +
``xcorr_max``-style parabolic sub-sample fit. Both were benchmarked directly
(synthetic 200 Hz windows, a known 0.037 s lag, 2000 calls each,
``.venv`` from this task): ``xcorr_pick_correction`` recovered ``0.0362 s``
at ``2725 calls/s``; this module's own :func:`_lag_and_cc` (three-point
parabolic fit around ``correlate``'s integer-sample peak, no `Trace` object
construction/copy, no `warnings.warn` machinery, `method="direct"` matching
`xcorr_pick_correction`'s own hardcoded choice for these small shift
values -- `correlate`'s docstring: "only for small values of shift
(approximately less than 100) direct time domain cross-correlation might
save some time") recovered ``0.0359 s`` at ``36887 calls/s`` -- a **13.5x
speedup at equal accuracy** (both within one sample, 0.005 s at 200 Hz, of
the true lag and of each other). Given this is explicitly the heaviest
step in the pipeline, :func:`_lag_and_cc` is what
:func:`cross_correlate_pair` actually calls; ``xcorr_pick_correction``
remains available from ``obspy`` for ad hoc/manual verification (and this
module's own test suite cross-checks :func:`_lag_and_cc` against it
directly) but is not on the production path. One deliberate deviation from
``xcorr_pick_correction``'s own default: that function's internal
``xcorr_max`` call uses ``abs_max=True`` (picks the *largest-magnitude*
correlation peak, positive or negative, and only warns if it turns out
negative); :func:`_lag_and_cc` instead always takes the largest **positive**
peak. A large negative (anti-correlated) peak is not a better match for
"the same phase arrival on both events" just because its magnitude is
large, and ``dt.cc``'s ``WGHT`` column is documented (user guide §B.3.2,
``docs/hypoddpy_audit.md`` §2.5) as a non-negative coherency-like weight in
``[0, 1]`` -- picking a large-magnitude anti-correlation would both violate
that convention and select a nonsensical (opposite-polarity) lag. In
practice ``cfg.hypodd.xcorr.min_cc`` (default 0.75) filters out any low or
negative-cc result either way.

The xcorr filter band (``cfg.hypodd.xcorr.filter``): kept at 2-20 Hz
------------------------------------------------------------------------
M10.1 shipped this key defaulting to 2-20 Hz with an open ``TODO`` ("confirm
these corners suit the project's actual cross-correlation frequency band");
this task's own notes suggest 5-40 Hz may suit microseismic events better.
Resolved here empirically rather than by assumption: this project's own
synthetic fixture (``tests/data/make_synthetic.py``) injects P arrivals as a
10 Hz-dominant damped sinusoid and S arrivals as a 6 Hz-dominant one
(``P_DOM_FREQ_HZ``/``S_DOM_FREQ_HZ``, 6 cycles). A Monte-Carlo comparison
(300 trials/phase, realistic amplitude/noise ratio, this task's own
:func:`_lag_and_cc`) found **no meaningful difference** between a 2-20 Hz
and a 5-40 Hz band for this dataset's P energy (both ~0.27 ms lag-recovery
std, cc ~0.985-0.988), but a 5-40 Hz band sits right at the low edge of (and
would attenuate) the S pulse's 6 Hz-centred energy for any real deployment
whose S corner frequency is similarly low (small/microseismic sources still
commonly have S energy well below 40 Hz, e.g. this project's own S pulses).
Since 2-20 Hz comfortably covers both this project's validated P and S
content with no measured accuracy cost, **the default stays 2-20 Hz**;
``XcorrFilterConfig`` remains fully project-overridable in
``config/example.yaml`` for a deployment with a genuinely higher-frequency
source spectrum (e.g. a shallow, high-corner-frequency hydraulic-fracturing
microseismic source), where 5-40 Hz (or higher) may indeed be the better
choice -- this is a per-project tuning decision, not a universal constant.

Parallelism and resume
------------------------
One :func:`cross_correlate_pair` call per ph2dt-linked pair, batched across
workers via :func:`openmeq.parallel.pmap_incremental` (``relocate/work/
xcorr_results.jsonl``, keyed by ``"{cuspid_i}_{cuspid_j}"`` -- a plain tuple
key, as the issue's own pseudocode names, is not itself a valid resumable
key: ``pmap_incremental``'s JSON-lines file round-trips a key through
``json.dumps``/``json.loads``, and a JSON array decodes back to an
unhashable Python ``list``, so this module uses a string key with the same
identity instead). A killed/interrupted run resumes without recomputing
already-written pairs. Each worker process keeps a small cache of every
event's decompressed ``.npz`` arrays it has already loaded
(:func:`_get_event_arrays`, keyed by ``(cache_dir, event_id)`` so it can
never leak between two different runs/tests sharing one process) --
matching ``openmeq.magnitude.amplitude``'s per-worker response-cache pattern
(ARCHITECTURE.md §6e) -- since one event commonly appears in many linked
pairs and re-opening/re-decompressing its ``.npz`` file every time would be
wasteful for the heaviest step in the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate

from openmeq.catalog import accepted, event_id, read_catalog
from openmeq.config import ConfigError as _ConfigError
from openmeq.log import get_logger
from openmeq.parallel import pmap_incremental
from openmeq.relocate.hypodd_files import build_event_id_map, build_station_label_map
from openmeq.relocate.hypodd_files import invert_event_id_map as _invert_event_id_map
from openmeq.stations import station_table as load_station_table

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate.xcorr")

#: Origin-time correction written in every dt.cc header -- always 0.0. See
#: the module docstring's "Origin-time correction (OTC)" section.
_OTC = 0.0

#: Bumped whenever the correlation itself changes, so results resumed from an
#: older ``xcorr_results.jsonl`` are discarded (see :func:`_reset_stale_results`).
#: 2 = OpenMEQ-6y1 window fix.
XCORR_ALGORITHM_VERSION = 2


def obspy_window(window: tuple[float, float]) -> tuple[float, float]:
    """``cfg.hypodd.xcorr.window_p``/``window_s`` (signed ``[start, end]``
    seconds relative to the pick, e.g. ``(-0.05, 0.2)``) as
    ``xcorr_pick_correction``'s ``(t_before, t_after)``, where ``t_before``
    counts *backwards* from the pick (``0.05``).

    OpenMEQ-6y1: the start offset used to be passed through unnegated, so the
    window began after the pick (pick + 0.025 s for the defaults) and missed
    the onset.
    """
    return -window[0], window[1]


# ---------------------------------------------------------------------------
# dt.ct re-parsing: per-pair station/phase detail (not returned by run_ph2dt)
# ---------------------------------------------------------------------------


def parse_dt_ct_detail(path: Path | str) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """Parse ``dt.ct``'s per-pair ``(station_label, phase)`` observation list.

    ``run_ph2dt`` (M10.2) returns only the ordered ``(cuspid1, cuspid2)``
    pair list; this function re-reads ``dt.ct`` itself for the per-pair
    detail this task needs (which station+phase combinations ``ph2dt``
    itself linked for that pair) -- see the module docstring's "Per-station/
    phase detail" section. ``dt.ct``'s own travel-time/weight fields
    (``TT1``/``TT2``/``WGHT``) are not needed here -- this module recomputes
    its own cross-correlation-based differential time and weight from the
    catalogue's actual pick times, not ``ph2dt``'s catalog-derived travel
    times -- so only the station label and phase are kept.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a ``dt.ct``-format file (``docs/ARCHITECTURE.md`` §3).

    Returns
    -------
    dict[tuple[int, int], list[tuple[str, str]]]
        ``{(cuspid1, cuspid2): [(station_label, phase), ...]}``, one entry
        per ``#`` header line, in file order, with the station/phase lines
        immediately following it (up to the next ``#`` line or EOF).

    Raises
    ------
    openmeq.config.ConfigError
        On a malformed header line, or a non-header line encountered before
        the first header line.
    """
    path = Path(path)
    pairs: dict[tuple[int, int], list[tuple[str, str]]] = {}
    current: tuple[int, int] | None = None
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.startswith("#"):
                fields = line.split()[1:]
                if len(fields) != 2:
                    raise _ConfigError(
                        f"{path}:{lineno}: malformed dt.ct header line (expected "
                        f"'# CUSPID1 CUSPID2'), got {line!r}"
                    )
                try:
                    current = (int(fields[0]), int(fields[1]))
                except ValueError as exc:
                    raise _ConfigError(
                        f"{path}:{lineno}: malformed dt.ct header line (non-integer "
                        f"cuspid), got {line!r}"
                    ) from exc
                pairs[current] = []
                continue
            if not line.strip():
                continue
            if current is None:
                raise _ConfigError(
                    f"{path}:{lineno}: dt.ct data line before any '#' header line: {line!r}"
                )
            parts = line.split()
            if len(parts) < 5:
                raise _ConfigError(
                    f"{path}:{lineno}: malformed dt.ct data line (expected 'STA TT1 TT2 "
                    f"WGHT PHA'), got {line!r}"
                )
            sta, phase = parts[0], parts[4]
            pairs[current].append((sta, phase))
    return pairs


# ---------------------------------------------------------------------------
# per-pair worker: types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DtccLine:
    """One ``dt.cc`` phase line: ``STA DT WEIGHT PHASE`` (getdata.f:398)."""

    sta: str
    dt_s: float
    cc: float
    phase: str


@dataclass(frozen=True)
class PairXcorrParams:
    """Everything :func:`cross_correlate_pair` needs besides the pair
    identity and the two events' pick/origin data -- the ``cfg`` argument in
    the issue's own ``cross_correlate_pair(pair, cache_dir, picks_by_event,
    cfg)`` signature. Built fresh, per pair, by :func:`run_xcorr` (cheap:
    every field but ``station_phase_list``/``cuspid_to_event_id`` is a small
    constant shared across all pairs; the latter two are restricted to just
    this one pair, not the whole run, to keep each ``pmap_incremental`` item
    small -- see the module docstring's "Parallelism and resume" section).
    """

    window_p: tuple[float, float]
    window_s: tuple[float, float]
    min_cc: float
    maxlag_s: float
    components: dict[str, list[str]]
    label_to_net_sta: dict[str, str]
    station_phase_list: list[tuple[str, str]]
    cuspid_to_event_id: dict[int, str]
    reject_edge_lags: bool = True
    s_maxlag_s: float | None = None  # OpenMEQ-suj: S max lag; None = maxlag_s


# ---------------------------------------------------------------------------
# per-worker-process caches (mirror openmeq.magnitude.amplitude's response
# cache / openmeq.relocate.waveform_cache's cfg cache pattern) -- keyed by
# (cache_dir, event_id) so a stale entry can never leak between two
# different runs/tests that happen to share one process (n_workers=1 keeps
# everything in-process, including across pytest tests in the same session).
# ---------------------------------------------------------------------------

_worker_npz_cache: dict[tuple[str, str], dict[str, Any] | None] = {}


def _get_event_arrays(event_id_: str, cache_dir: str) -> dict[str, Any] | None:
    """Load (and per-worker-process cache) one event's ``.npz`` cache file.

    Returns ``None`` (logged) if the file is missing or unreadable -- never
    raises; the caller treats this exactly like "no data for this event"
    (the same missing-phase-at-one-event pitfall this module must not
    crash on, module docstring).
    """
    key = (str(cache_dir), event_id_)
    if key in _worker_npz_cache:
        return _worker_npz_cache[key]

    path = Path(cache_dir) / f"{event_id_}.npz"
    if not path.exists():
        logger.debug("xcorr: no waveform cache for event %s (%s); skipping.", event_id_, path)
        _worker_npz_cache[key] = None
        return None

    try:
        arrays: dict[str, Any] = {}
        with np.load(path) as npz:
            seed_ids = [str(s) for s in npz["seed_ids"]]
            sampling_rate = float(npz["sampling_rate"])
            for seed_id in seed_ids:
                arrays[seed_id] = (npz[f"{seed_id}__data"], float(npz[f"{seed_id}__start"]))
        arrays["seed_ids"] = seed_ids
        arrays["sampling_rate"] = sampling_rate
    except Exception as exc:  # noqa: BLE001 -- a corrupt cache file must not abort the whole run
        logger.warning("xcorr: failed to read waveform cache %s: %s", path, exc)
        _worker_npz_cache[key] = None
        return None

    _worker_npz_cache[key] = arrays
    return arrays


def _find_seed_id(seed_ids: list[str], net_sta: str, comp: str) -> str | None:
    """The one cached seed id (``NET.STA.LOC.CHA``) for ``net_sta`` whose
    mapped channel's last character is ``comp`` (``openmeq.stations.
    apply_channel_map``'s convention, ARCHITECTURE.md §6), or ``None``."""
    for seed_id in seed_ids:
        parts = seed_id.split(".")
        if len(parts) != 4:
            continue
        if f"{parts[0]}.{parts[1]}" == net_sta and parts[3] and parts[3][-1] == comp:
            return seed_id
    return None


def _extract_window(
    arrays: dict[str, Any],
    seed_id: str,
    pick_ts: float,
    t_before: float,
    t_after: float,
    maxlag_s: float,
    sr: float,
) -> np.ndarray | None:
    """The fixed-length correlation window around ``pick_ts`` for ``seed_id``.

    Matches ``xcorr_pick_correction``'s own slice bounds exactly (``t -
    t_before - maxlag/2`` to ``t + t_after + maxlag/2``), computed from a
    single shared ``window_len`` (rather than independent per-event
    start/end rounding) so both events' windows always have exactly the
    same sample count -- required by ``correlate`` (unequal lengths would
    silently realign around each array's own middle, ``correlate``'s own
    docstring) and by the fixed-``shift`` cross-correlation below. Returns
    ``None`` if the seed id isn't cached or the window would run off the
    edge of the cached array (should not normally happen given M10.1's own
    ``maxlag_s``-aware padding, but a hand-built/misconfigured cache must
    not crash this worker).
    """
    entry = arrays.get(seed_id)
    if entry is None:
        return None
    data, start = entry
    pad = maxlag_s / 2.0
    window_len = int(round((t_before + t_after + maxlag_s) * sr)) + 1
    i0 = int(round((pick_ts - start - t_before - pad) * sr))
    i1 = i0 + window_len
    if i0 < 0 or i1 > len(data):
        return None
    return data[i0:i1]


def _at_lag_edge(pick2_corr: float, maxlag_s: float, sr: float) -> bool:
    """Whether a lag sits within half a sample of the ``+/- maxlag_s`` search
    edge (``shift = round(maxlag_s * sr)`` samples, as in :func:`_lag_and_cc`)
    -- i.e. the correlation maximum is at, or fitted onto, the boundary rather
    than at an interior peak (OpenMEQ-9fh)."""
    shift = int(round(maxlag_s * sr))
    return abs(pick2_corr) * sr >= shift - 0.5


def _lag_and_cc(
    w1: np.ndarray, w2: np.ndarray, sr: float, maxlag_s: float
) -> tuple[float, float] | None:
    """Vectorised replacement for ``xcorr_pick_correction``'s lag/coefficient
    computation -- see the module docstring's "benchmarked, not assumed"
    section for the measured 13.5x speedup and the accuracy cross-check.

    Returns ``(pick2_corr, cc)`` with exactly ``xcorr_pick_correction``'s own
    semantics (``pick2_corr`` is *added* to pick 2's time to align it with
    pick 1's waveform; a corrected differential travel time is then
    ``(pick2 + pick2_corr - origin2) - (pick1 - origin1)``, ``obspy``'s own
    documented convention) -- or ``None`` if the windows are unusable (too
    short for the requested ``maxlag_s``, mismatched lengths, silent/flat
    data, or a non-finite correlation).
    """
    shift = int(round(maxlag_s * sr))
    if shift < 1 or len(w1) != len(w2) or len(w1) < 2 * shift + 1:
        return None
    a = np.asarray(w1, dtype=np.float64)
    b = np.asarray(w2, dtype=np.float64)
    if not (np.any(a) and np.any(b)):
        return None

    cc = correlate(a, b, shift, demean=True, normalize="naive", method="direct")
    if cc.size == 0 or not np.all(np.isfinite(cc)):
        return None

    idx = int(np.argmax(cc))
    mid = (len(cc) - 1) / 2.0
    if idx == 0 or idx == len(cc) - 1:
        # Peak sits at the edge of the search window -- no neighbour on one
        # side to fit a parabola through; fall back to the integer-sample
        # shift (matches xcorr_pick_correction's own "fitting at maximum
        # lag" warning path, which also cannot improve on this).
        lag_samples = idx - mid
        coeff = float(cc[idx])
    else:
        y0, y1, y2 = float(cc[idx - 1]), float(cc[idx]), float(cc[idx + 1])
        denom = y0 - 2 * y1 + y2
        if denom == 0:
            lag_samples = idx - mid
            coeff = y1
        else:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-1.0, min(1.0, delta))  # bound the sub-sample correction to +/-1 sample
            lag_samples = (idx + delta) - mid
            coeff = y1 - 0.25 * (y0 - y2) * delta

    pick2_corr = -lag_samples / sr
    coeff = max(-1.0, min(1.0, coeff))
    return pick2_corr, coeff


# ---------------------------------------------------------------------------
# per-pair worker
# ---------------------------------------------------------------------------


def cross_correlate_pair(
    pair: tuple[int, int],
    cache_dir: str,
    picks_by_event: dict[str, dict[str, Any]],
    cfg: PairXcorrParams,
) -> list[DtccLine]:
    """Cross-correlate every shared station+phase for one ph2dt-linked pair.

    Pure, top-level, picklable per-pair worker function (this module's
    docstring, "Parallelism and resume"). ``pair`` is ``(cuspid_i,
    cuspid_j)`` in ``dt.ct``'s own header order for this pair (module
    docstring's "Canonical pair ordering"); the returned ``dt_s`` for every
    line is ``TT_i - TT_j_corrected``, matching a ``# {cuspid_i} {cuspid_j}
    ...`` header.

    Parameters
    ----------
    pair : tuple[int, int]
        ``(cuspid_i, cuspid_j)``.
    cache_dir : str
        ``relocate/cache/`` (``RunPaths.relocate_cache_dir``); each event's
        ``<event_id>.npz`` is loaded from here (:func:`_get_event_arrays`).
    picks_by_event : dict[str, dict]
        ``{event_id: {"origin_time": iso_str, "picks": {net_sta:
        {"P"|"S": iso_str}}}}`` -- restricted by the caller to just this
        pair's two events (not the whole catalogue), keyed by
        ``cfg.cuspid_to_event_id``'s values.
    cfg : PairXcorrParams
        This pair's station/phase list plus the shared xcorr settings.

    Returns
    -------
    list[DtccLine]
        One line per station+phase that had a pick at both events, a
        cached channel at both events, and a cross-correlation with
        ``cc >= cfg.min_cc`` and ``|lag| <=`` the phase's max lag (plus a small
        one-sample tolerance for the sub-sample fit). Empty if nothing
        qualified -- never raises for a missing phase/channel/event (module
        docstring's pitfalls).
    """
    cuspid_i, cuspid_j = pair
    event_i = cfg.cuspid_to_event_id.get(cuspid_i)
    event_j = cfg.cuspid_to_event_id.get(cuspid_j)
    if event_i is None or event_j is None:
        return []

    data_i = picks_by_event.get(event_i)
    data_j = picks_by_event.get(event_j)
    if data_i is None or data_j is None:
        return []

    arrays_i = _get_event_arrays(event_i, cache_dir)
    arrays_j = _get_event_arrays(event_j, cache_dir)
    if arrays_i is None or arrays_j is None:
        return []

    origin_i = UTCDateTime(data_i["origin_time"])
    origin_j = UTCDateTime(data_j["origin_time"])

    lines: list[DtccLine] = []
    for station_label, phase in cfg.station_phase_list:
        net_sta = cfg.label_to_net_sta.get(station_label)
        if net_sta is None:
            continue

        pick_i_iso = data_i["picks"].get(net_sta, {}).get(phase)
        pick_j_iso = data_j["picks"].get(net_sta, {}).get(phase)
        if pick_i_iso is None or pick_j_iso is None:
            # Missing phase at one event of the pair -- skip this
            # station+phase, do not crash (audit-confirmed template
            # behaviour to keep, module docstring's pitfalls).
            continue

        pick_i_t = UTCDateTime(pick_i_iso)
        pick_j_t = UTCDateTime(pick_j_iso)
        t_before, t_after = obspy_window(cfg.window_p if phase == "P" else cfg.window_s)
        maxlag = cfg.s_maxlag_s if phase == "S" and cfg.s_maxlag_s is not None else cfg.maxlag_s
        comps = cfg.components.get(phase, [])

        best: tuple[float, float, float] | None = None  # (cc, pick2_corr, sr)
        for comp in comps:
            seed_i = _find_seed_id(arrays_i["seed_ids"], net_sta, comp)
            seed_j = _find_seed_id(arrays_j["seed_ids"], net_sta, comp)
            if seed_i is None or seed_j is None:
                continue
            sr_i = arrays_i["sampling_rate"]
            sr_j = arrays_j["sampling_rate"]
            if abs(sr_i - sr_j) > 1e-6:
                logger.warning(
                    "xcorr: sampling-rate mismatch for %s %s (%s vs %s Hz) -- skipping this "
                    "channel (M10.1's cache should guarantee one shared project-wide rate).",
                    net_sta,
                    comp,
                    sr_i,
                    sr_j,
                )
                continue
            sr = sr_i
            w_i = _extract_window(
                arrays_i, seed_i, pick_i_t.timestamp, t_before, t_after, maxlag, sr
            )
            w_j = _extract_window(
                arrays_j, seed_j, pick_j_t.timestamp, t_before, t_after, maxlag, sr
            )
            if w_i is None or w_j is None:
                continue
            result = _lag_and_cc(w_i, w_j, sr, maxlag)
            if result is None:
                continue
            pick2_corr, coeff = result
            if cfg.reject_edge_lags and _at_lag_edge(pick2_corr, maxlag, sr):
                # Best lag pinned to the search edge: not a real peak. Another
                # channel (S) may still give an interior one.
                continue
            if best is None or coeff > best[0]:
                best = (coeff, pick2_corr, sr)

        if best is None:
            continue
        coeff, pick2_corr, sr = best
        if coeff < cfg.min_cc:
            continue
        if abs(pick2_corr) > maxlag + 1.0 / sr:
            continue

        dt_s = (pick_i_t - origin_i) - (pick_j_t + pick2_corr - origin_j)
        lines.append(DtccLine(sta=station_label, dt_s=dt_s, cc=coeff, phase=phase))

    return lines


def _cross_correlate_pair_item(
    item: tuple[tuple[int, int], str, dict[str, dict[str, Any]], PairXcorrParams],
) -> dict[str, Any]:
    """``pmap_incremental``-compatible single-argument adapter around
    :func:`cross_correlate_pair` (mirrors ``openmeq.relocate.waveform_cache
    ._build_event_cache_item``'s pattern). Converts the returned
    ``list[DtccLine]`` into a plain, JSON-serializable dict, since
    ``pmap_incremental`` writes each result as one JSON-lines record."""
    pair, cache_dir, picks_by_event, cfg = item
    lines = cross_correlate_pair(pair, cache_dir, picks_by_event, cfg)
    return {"lines": [asdict(line) for line in lines]}


def _pair_item_key(item: tuple[tuple[int, int], str, dict, PairXcorrParams]) -> str:
    """A stable, JSON-round-trippable, hashable resume key for one pair.

    The issue's own pseudocode names ``key=lambda pair: pair`` -- but
    ``pmap_incremental`` round-trips every key through ``json.dumps``/
    ``json.loads`` (``openmeq.parallel``'s JSON-lines results file), and a
    JSON array decodes back to an unhashable Python ``list``, so a bare
    tuple cannot actually serve as this dict key. This uses a string with
    the same identity instead (module docstring's "Parallelism and resume").
    """
    (cuspid_i, cuspid_j), *_rest = item
    return f"{cuspid_i}_{cuspid_j}"


# ---------------------------------------------------------------------------
# catalogue -> per-event pick/origin data (main process only)
# ---------------------------------------------------------------------------


def _event_pick_data(event) -> dict[str, Any] | None:
    """``{"origin_time": iso_str, "picks": {net_sta: {"P"|"S": iso_str}}}``
    for one event, or ``None`` if it has no origin at all.

    Non-rejected P/S picks only (ARCHITECTURE.md §5's rejection convention),
    mirroring ``openmeq.relocate.waveform_cache._event_picks``'s filtering
    but organised by station for O(1) lookup in :func:`cross_correlate_pair`,
    and including the origin time :func:`cross_correlate_pair`'s sign-
    convention formula needs.
    """
    origin = event.preferred_origin()
    if origin is None:
        if not event.origins:
            return None
        origin = event.origins[0]

    picks: dict[str, dict[str, str]] = {}
    for pick in event.picks:
        if pick.evaluation_status == "rejected":
            continue
        phase = pick.phase_hint
        if phase not in ("P", "S"):
            continue
        wf = pick.waveform_id
        if wf is None or not wf.network_code or not wf.station_code:
            continue
        net_sta = f"{wf.network_code}.{wf.station_code}"
        picks.setdefault(net_sta, {})[phase] = str(pick.time)

    return {"origin_time": str(origin.time), "picks": picks}


# ---------------------------------------------------------------------------
# dt.cc assembly
# ---------------------------------------------------------------------------


def write_dt_cc(
    path: Path | str,
    pairs: list[tuple[int, int]],
    pair_lines: list[list[DtccLine]],
    *,
    otc: float = _OTC,
) -> int:
    """Write ``relocate/work/dt.cc``: one ``# ID1 ID2 OTC`` header per pair
    with at least one surviving observation, followed by its ``STA DT WEIGHT
    PHASE`` lines (free-format, ``getdata.f:369-402`` -- confirmed against
    the real ``external/HYPODD/examples/example1/dt.cc``).

    A pair with zero surviving lines (every station+phase missing, below
    ``min_cc``, or filtered) is omitted entirely -- an empty header block
    would be a harmless no-op for hypoDD but is simply not written, since
    there is nothing to report for that pair.

    Parameters
    ----------
    path : pathlib.Path or str
        Output path, ``relocate/work/dt.cc`` (``RunPaths.dt_cc``).
    pairs : list[tuple[int, int]]
        ``(cuspid_i, cuspid_j)`` in ``dt.ct``'s own header order (module
        docstring's "Canonical pair ordering").
    pair_lines : list[list[DtccLine]]
        One entry per ``pairs`` element (same order, same length).
    otc : float, keyword-only
        Origin-time correction written in every header -- always ``0.0``
        by this module's own convention (module docstring, "Origin-time
        correction").

    Returns
    -------
    int
        The number of pairs actually written (i.e. with >= 1 line).

    Notes
    -----
    M10.5 / OpenMEQ-16d.5's note: writes are idempotent -- `run_xcorr`
    calls this on *every* run (even when every pair was resumed rather
    than recomputed, `openmeq.parallel.pmap_incremental`'s own resume
    behaviour), so writing unconditionally would give this file a fresh
    mtime every single ``openmeq relocate`` invocation. `run_hypodd`'s own
    ``StepManifest`` skip-unchanged fingerprint (ARCHITECTURE.md §6h)
    includes this file's mtime/size, so an unconditional write would force
    an unnecessary hypoDD re-invocation on every run regardless of whether
    anything actually changed, breaking the "re-running with nothing
    changed re-runs nothing" contract the whole M10.1-M10.5 chain is
    supposed to provide end to end. The file is therefore only actually
    (re)written when its assembled content differs from what's already on
    disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines_out: list[str] = []
    n_written = 0
    for (cuspid_i, cuspid_j), lines in zip(pairs, pair_lines, strict=True):
        if not lines:
            continue
        lines_out.append(f"# {cuspid_i:9d} {cuspid_j:9d} {otc:10.6f}\n")
        for line in lines:
            lines_out.append(f"{line.sta:<7s} {line.dt_s:12.6f} {line.cc:6.4f} {line.phase}\n")
        n_written += 1
    content = "".join(lines_out)

    if path.is_file():
        try:
            if path.read_text() == content:
                return n_written
        except OSError:
            pass

    with open(path, "w") as fh:
        fh.write(content)
    return n_written


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _reset_stale_results(out_path: Path, xcorr_cfg) -> None:
    """Delete ``xcorr_results.jsonl`` if it was computed with different
    ``hypodd.xcorr`` settings or an older :data:`XCORR_ALGORITHM_VERSION`.

    ``pmap_incremental`` resumes by pair key alone, so without this a settings
    change (or the OpenMEQ-6y1 window fix) would silently reuse old per-pair
    results. The stamp lives next to the results file
    (``xcorr_results.settings.json``); a results file with no stamp predates
    it and is treated as stale.
    """
    import json

    stamp_path = out_path.with_name(out_path.stem + ".settings.json")
    stamp = {"version": XCORR_ALGORITHM_VERSION, "xcorr": xcorr_cfg.model_dump(mode="json")}
    try:
        old = json.loads(stamp_path.read_text()) if stamp_path.is_file() else None
    except (OSError, ValueError):
        old = None
    if old != stamp:
        if out_path.is_file():
            logger.info(
                "xcorr: settings/algorithm changed since %s was written; recomputing every pair.",
                out_path,
            )
        out_path.unlink(missing_ok=True)
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(stamp, indent=1))


def run_xcorr(
    cfg: Config,
    paths: RunPaths,
    pairs: list[tuple[int, int]],
    *,
    force: bool = False,
) -> Path:
    """Cross-correlate every ``pairs`` entry in parallel and write ``dt.cc``.

    Sequence:

    1. Require ``paths.catalog_mag_xml`` and ``paths.dt_ct`` to already
       exist (raises `openmeq.config.ConfigError` otherwise -- this step
       never silently substitutes a different catalogue/pair source, same
       discipline as `openmeq.relocate.ph2dt_step.run_ph2dt`).
    2. Re-read the persisted event-id/station-label maps (M9.2, §6c) --
       never regenerated ad hoc (this module's own docstring).
    3. `parse_dt_ct_detail` for the per-pair station/phase lists.
    4. Build one small, picklable work item per pair (`PairXcorrParams` +
       a 2-event-only ``picks_by_event`` subset) and run them through
       `openmeq.parallel.pmap_incremental` (resumable, ``relocate/work/
       xcorr_results.jsonl``).
    5. Assemble `write_dt_cc` from the (possibly partly-resumed) results.

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    pairs : list[tuple[int, int]]
        ``(cuspid1, cuspid2)`` pairs, e.g. from
        `openmeq.relocate.ph2dt_step.run_ph2dt`'s return value, in ``dt.ct``'s
        own header order (module docstring's "Canonical pair ordering").
    force : bool, keyword-only
        If True, discard any existing ``xcorr_results.jsonl`` first, so
        every pair is recomputed rather than resumed from a prior run.

    Returns
    -------
    pathlib.Path
        ``paths.dt_cc``.

    Raises
    ------
    openmeq.config.ConfigError
        If ``paths.catalog_mag_xml`` or ``paths.dt_ct`` doesn't exist.
    """
    catalog_path = Path(paths.catalog_mag_xml)
    if not catalog_path.is_file():
        raise _ConfigError(
            f"{catalog_path} does not exist -- cross-correlation needs magnitude/catalog_mag.xml "
            "(M8.3's output) for pick times and origins. Run `openmeq magnitude` first."
        )
    dt_ct_path = Path(paths.dt_ct)
    if not dt_ct_path.is_file():
        raise _ConfigError(
            f"{dt_ct_path} does not exist -- cross-correlation needs ph2dt's linked-pair list. "
            "Run `openmeq.relocate.ph2dt_step.run_ph2dt` first."
        )

    catalog = read_catalog(catalog_path)
    station_table = load_station_table(cfg.stations)
    work_dir = paths.relocate_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    event_id_map = build_event_id_map(catalog, work_dir / "event_id_map.csv")
    label_map = build_station_label_map(station_table, work_dir / "station_label_map.csv")
    cuspid_to_event_id_full = _invert_event_id_map(event_id_map)
    label_to_net_sta = {label: net_sta for net_sta, label in label_map.items()}

    pair_detail = parse_dt_ct_detail(dt_ct_path)
    xcorr_cfg = cfg.hypodd.xcorr

    wanted_cuspids = {c for pair in pairs for c in pair}
    wanted_event_ids = {
        cuspid_to_event_id_full[c] for c in wanted_cuspids if c in cuspid_to_event_id_full
    }
    event_pick_data: dict[str, dict[str, Any]] = {}
    for event in accepted(catalog):
        eid = event_id(event)
        if eid in wanted_event_ids:
            data = _event_pick_data(event)
            if data is not None:
                event_pick_data[eid] = data

    cache_dir = str(paths.relocate_cache_dir)
    items: list[tuple[tuple[int, int], str, dict[str, dict[str, Any]], PairXcorrParams]] = []
    for pair in pairs:
        cuspid_i, cuspid_j = pair
        event_i = cuspid_to_event_id_full.get(cuspid_i)
        event_j = cuspid_to_event_id_full.get(cuspid_j)
        params = PairXcorrParams(
            window_p=xcorr_cfg.window_p,
            window_s=xcorr_cfg.window_s,
            min_cc=xcorr_cfg.min_cc,
            maxlag_s=xcorr_cfg.maxlag_s,
            components=xcorr_cfg.components,
            label_to_net_sta=label_to_net_sta,
            station_phase_list=pair_detail.get(pair, []),
            cuspid_to_event_id={cuspid_i: event_i, cuspid_j: event_j},
            reject_edge_lags=xcorr_cfg.reject_edge_lags,
            s_maxlag_s=xcorr_cfg.s_maxlag_s,
        )
        picks_subset: dict[str, dict[str, Any]] = {}
        if event_i in event_pick_data:
            picks_subset[event_i] = event_pick_data[event_i]
        if event_j in event_pick_data:
            picks_subset[event_j] = event_pick_data[event_j]
        items.append((pair, cache_dir, picks_subset, params))

    out_path = paths.xcorr_results_jsonl
    if force:
        Path(out_path).unlink(missing_ok=True)
    _reset_stale_results(Path(out_path), xcorr_cfg)

    t0 = time.monotonic()
    raw_results = pmap_incremental(
        _cross_correlate_pair_item,
        items,
        cfg.parallel,
        out_path,
        key=_pair_item_key,
        desc="xcorr",
    )
    elapsed = max(time.monotonic() - t0, 1e-9)

    pair_lines: list[list[DtccLine]] = []
    n_obs = 0
    for raw in raw_results:
        lines = [DtccLine(**line) for line in raw["lines"]]
        pair_lines.append(lines)
        n_obs += len(lines)

    n_pairs_written = write_dt_cc(paths.dt_cc, pairs, pair_lines, otc=_OTC)

    logger.info(
        "xcorr: %d pair(s) processed -> %d written with >=1 observation, %d total "
        "observation(s) in %.2fs (%.1f pairs/s, %s).",
        len(pairs),
        n_pairs_written,
        n_obs,
        elapsed,
        len(items) / elapsed,
        paths.dt_cc,
    )
    return paths.dt_cc
