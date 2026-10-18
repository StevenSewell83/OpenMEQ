"""PhaseNet ``result.h5`` probability curves: reading, station columns, peak width.

Shared by :mod:`openmeq.pick.to_quakeml` (pick uncertainties, OpenMEQ-9kn)
and the pick diagnostics (probability overlays).

``result.h5`` layout
---------------------
``run_phasenet`` always passes ``--save_prob``, so ``pick/phasenet_raw/
result.h5`` holds one dataset per chunk under ``predict.py``'s ``/prob``
group, keyed by its ``data_list.csv`` ``fname``
(``prob/<event_id>/<event_id>.mseed``), shape ``(nt, nx, 3)``: sample,
station, ``[noise, P, S]`` softmax (``docs/phasenet_interface.md`` §6).
Sample 0 is the chunk's earliest trace start.

Station columns
----------------
PhaseNet's station order along ``nx`` is not part of its documented
contract. The vendored ``read_mseed`` uses ``station_keys =
sorted(station_ids)`` (the file's distinct ``trace.id[:-1]`` values), so
:func:`station_columns` assumes that order **and verifies it**: the column
count must equal the sorted key count, and every supplied check pick must
read back its own probability (within :data:`PROB_CHECK_TOLERANCE`) from its
mapped column at its sample index. Any mismatch returns ``{}`` so callers
never use another station's curve.

Peak width
-----------
:func:`peak_half_width_s` is the half-width of the contiguous region around
a pick where the curve stays at or above ``frac * peak`` (``frac=0.5``: half
the FWHM), with linear interpolation of both crossings for sub-sample
precision. PhaseNet was trained on Gaussian-shaped labels, so a sharp onset
gives a narrow peak and an emergent one a broad peak; the half-width
therefore tracks how well the onset time is constrained.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime

from openmeq.log import get_logger

logger = get_logger("pick.probability")

#: ``result.h5`` channel index for each phase (index 0 is noise).
PHASE_CHANNEL = {"P": 1, "S": 2}

#: Max allowed ``|result.h5 value - pick probability|`` at a pick's own
#: sample when verifying the station-column mapping; ``picks.csv`` rounds
#: ``phase_score`` to 3 decimals.
PROB_CHECK_TOLERANCE = 0.05


def result_h5_key(event_id: str) -> str:
    """``result.h5`` dataset key for one chunk (``/prob`` + its ``data_list.csv`` ``fname``)."""
    return f"prob/{event_id}/{event_id}.mseed"


def read_result_h5_probs(result_h5_path: Path, key: str) -> np.ndarray | None:
    """One chunk's ``(nt, nx, 3)`` probability array, or ``None`` if unavailable."""
    try:
        import h5py
    except ImportError:  # pragma: no cover - h5py ships with tensorflow's own deps here
        return None
    try:
        with h5py.File(result_h5_path, "r") as f:
            if key not in f:
                return None
            return np.asarray(f[key])
    except Exception as exc:  # noqa: BLE001 - a bad/partial result.h5 must not abort the caller
        logger.warning("could not read %s from %s: %s", key, result_h5_path, exc)
        return None


def sample_index(time: UTCDateTime, begin: UTCDateTime, sr: float) -> int:
    """Nearest ``result.h5`` sample index for ``time``."""
    return int(round((time - begin) * sr))


def station_columns(
    stream: Stream,
    prob: np.ndarray | None,
    checks: Iterable[tuple[str, str, UTCDateTime, float | None]],
    begin: UTCDateTime,
    sr: float,
) -> dict[str, int]:
    """``{station_key: result.h5 column}``, or ``{}`` if unverifiable (module docstring).

    Parameters
    ----------
    stream : obspy.Stream
        The chunk's ``chunks_pn`` traces.
    prob : numpy.ndarray or None
        The chunk's ``result.h5`` array.
    checks : iterable of (station_key, phase, time, probability)
        Picks whose probability must be reproduced by their mapped column;
        entries with an unknown station, ``None`` probability or a time
        outside the array are skipped.
    begin : obspy.UTCDateTime
        Time of sample 0 (the chunk's earliest trace start).
    sr : float
        The chunk's sampling rate (Hz).
    """
    if prob is None or prob.ndim != 3 or prob.shape[2] < 3:
        return {}
    keys = sorted({tr.id[:-1] for tr in stream})
    if len(keys) != prob.shape[1]:
        logger.warning(
            "result.h5 has %d station column(s) but chunks_pn has %d station(s) -- "
            "not using its probability curves",
            prob.shape[1],
            len(keys),
        )
        return {}
    cols = {key: i for i, key in enumerate(keys)}
    for key, phase, time, p in checks:
        idx = sample_index(time, begin, sr)
        if p is None or key not in cols or phase not in PHASE_CHANNEL:
            continue
        if not 0 <= idx < prob.shape[0]:
            continue
        value = float(prob[idx, cols[key], PHASE_CHANNEL[phase]])
        if abs(value - float(p)) > PROB_CHECK_TOLERANCE:
            logger.warning(
                "result.h5 column for %s does not reproduce its %s pick probability "
                "(%.3f vs %.3f) -- not using its probability curves for this event",
                key,
                phase,
                value,
                float(p),
            )
            return {}
    return cols


def peak_half_width_s(curve: np.ndarray, idx: int, sr: float, frac: float = 0.5) -> float | None:
    """Half-width (s) of ``curve``'s peak at ``idx`` measured at ``frac * peak``.

    The peak value is ``curve[idx]``, or the larger of its immediate
    neighbours if the pick time rounded one sample off the maximum. A side
    that never drops below the level before the array edge ends at the edge.
    Returns ``None`` if ``idx`` is outside ``curve`` or the peak is not
    positive.
    """
    curve = np.asarray(curve, dtype=float)
    n = curve.size
    if not 0 <= idx < n:
        return None
    lo, hi = max(idx - 1, 0), min(idx + 2, n)
    idx = lo + int(np.argmax(curve[lo:hi]))
    peak = curve[idx]
    if not peak > 0:
        return None
    level = frac * peak

    i = idx
    while i > 0 and curve[i - 1] >= level:
        i -= 1
    if i > 0:
        # Crossing between i-1 (below) and i (at/above).
        left = i - (curve[i] - level) / (curve[i] - curve[i - 1])
    else:
        left = 0.0

    j = idx
    while j < n - 1 and curve[j + 1] >= level:
        j += 1
    if j < n - 1:
        right = j + (curve[j] - level) / (curve[j] - curve[j + 1])
    else:
        right = float(n - 1)

    return (right - left) / (2.0 * sr)


__all__ = [
    "PHASE_CHANNEL",
    "PROB_CHECK_TOLERANCE",
    "peak_half_width_s",
    "read_result_h5_probs",
    "result_h5_key",
    "sample_index",
    "station_columns",
]
