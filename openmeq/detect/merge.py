"""Merge per-window raw triggers into the final detections catalogue (M2.2,
``OpenMEQ-ujx.2``, ARCHITECTURE.md §4.3/§4.4, PLAN.md §6 Step 2).

:func:`openmeq.detect.worker.detect_window` (M2.1) returns one raw trigger
list per time window, padded by ``cfg.detection.day_overlap_s`` on each
side, so a trigger near a window boundary legitimately appears -- often with
a slightly different time/coincidence_sum/station set -- in *both* of its
adjacent windows' outputs. This module flattens every window's output into
one list, de-duplicates those boundary-induced (and other near-)duplicates
into a single detection per underlying physical event, assigns each
surviving detection its permanent ``event_id``
(:func:`openmeq.catalog.event_id_from_time`), and writes the result as
``detections/detections.csv`` (ARCHITECTURE.md §4.3).

Design notes
------------
**De-duplication rule (``cfg.detection.dedup_window_s``).** Two triggers are
considered the same underlying event if they are within
``dedup_window_s`` seconds of each other. Sort all flattened triggers by
time, then walk the sorted list greedily, growing a cluster while each next
trigger is within ``dedup_window_s`` of the *cluster's current
representative time* -- not the cluster's first member's time, and not a
fixed-width time bin. This means:

- A chain of triggers each within ``dedup_window_s`` of its immediate
  neighbour, but spanning more than ``dedup_window_s`` end-to-end, is still
  merged into one cluster (this is the deliberate "greedy chaining" choice
  called out in this issue's design notes -- a fixed-width bin would
  incorrectly split a cluster straddling a bin edge).
- The representative time is updated to the *strongest* member's time each
  time a stronger trigger (higher ``coincidence_sum``) is folded into the
  cluster, so the chain can only extend from whichever time currently best
  represents the cluster.

**Known limitation.** Because de-duplication is purely time-based, two
*genuinely distinct* events that happen to occur within ``dedup_window_s``
of each other (physically possible in a dense microseismic sequence,
although not deliberately present in the synthetic dataset used by this
module's tests) will be incorrectly merged into a single detection. This is
a documented limitation of the design, not a bug: distinguishing "one event,
seen twice, near a window boundary" from "two real events that happen to be
close together" from time alone is not possible in general; a future step
could use waveform similarity or station-arrival-pattern differences to
split such cases, but that is out of scope here.

**Cluster -> single detection.** Within a cluster:

- ``time`` / ``coincidence_sum`` are taken from the member with the highest
  ``coincidence_sum`` (ties broken by earliest time) -- the window run that
  saw the trigger most completely (most of its stations counted, not
  clipped by a window edge) is presumed to have the more reliable
  time/coincidence_sum, and using that member's own actual trigger time
  keeps the reported time physically meaningful (a real trigger-on time
  from one run), rather than e.g. an artificial average across members.
- ``stations`` / ``trace_ids`` are the sorted union across *all* members of
  the cluster (not just the representative), since a trigger seen only
  partially by one window (clipped near its edge) and more fully by the
  adjacent window should end up with the fuller station list.
- ``n_stations`` is ``len(stations)`` (recomputed from the merged union, not
  copied from any single member).
- ``duration_s`` is the max across all members.

**Event id collisions.** ``event_id_from_time`` truncates to millisecond
precision, so two distinct (post-merge) detections could in principle round
to the same id if they happen to land within the same millisecond -- e.g. a
pathological cluster split that produces two representative times a
millisecond apart is *not* expected in practice (they would have been in the
same dedup cluster), but a defensive, deterministic tie-break is applied
regardless: detections are assigned ids in ascending-time order, and if the
freshly-computed id has already been used by an earlier (necessarily
earlier-in-time-or-equal) detection, a ``_dup<n>`` suffix (``n`` starting at
1) is appended, incrementing until the id is unique. This keeps ids
deterministic given the same input (the assignment order is always
time-sorted) without silently dropping a detection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

from openmeq.catalog import event_id_from_time, iso_z
from openmeq.config import Config

#: ARCHITECTURE.md §4.3 column order -- also the CSV column order.
DETECTIONS_COLUMNS = [
    "event_id",
    "time",
    "duration_s",
    "coincidence_sum",
    "n_stations",
    "stations",
    "trace_ids",
]


def _cluster_triggers(triggers: list[dict], dedup_window_s: float) -> list[list[dict]]:
    """Greedily chain-cluster time-sorted ``triggers`` within ``dedup_window_s``.

    ``triggers`` must already be sorted by time. See the module docstring's
    "De-duplication rule" note for the exact semantics: a cluster grows
    while the next trigger is within ``dedup_window_s`` of the cluster's
    *current representative* time, and the representative is re-picked
    (highest ``coincidence_sum``, ties to the earliest time) every time a
    trigger is added.
    """
    clusters: list[list[dict]] = []
    current: list[dict] = []
    rep_time: UTCDateTime | None = None

    for trig in triggers:
        t = UTCDateTime(trig["time"])
        if current and (t - rep_time) <= dedup_window_s:
            current.append(trig)
        else:
            if current:
                clusters.append(current)
            current = [trig]
        rep_time = _representative(current)[1]

    if current:
        clusters.append(current)
    return clusters


def _representative(cluster: list[dict]) -> tuple[dict, UTCDateTime]:
    """The cluster member used for ``time``/``coincidence_sum``: highest
    ``coincidence_sum``, ties broken by earliest time."""
    best = min(
        cluster,
        key=lambda trig: (-trig["coincidence_sum"], UTCDateTime(trig["time"])),
    )
    return best, UTCDateTime(best["time"])


def _merge_cluster(cluster: list[dict]) -> dict:
    """One de-duplicated cluster of raw trigger dicts -> one detection row (dict)."""
    rep, rep_time = _representative(cluster)
    stations = sorted({s for trig in cluster for s in trig["stations"]})
    trace_ids = sorted({tid for trig in cluster for tid in trig["trace_ids"]})
    duration_s = max(trig["duration_s"] for trig in cluster)
    return {
        "time": rep_time,
        "duration_s": float(duration_s),
        "coincidence_sum": float(rep["coincidence_sum"]),
        "n_stations": len(stations),
        "stations": stations,
        "trace_ids": trace_ids,
    }


def _assign_event_ids(detections: list[dict]) -> list[str]:
    """Deterministic, unique ``event_id`` per detection.

    ``detections`` must already be sorted by time. See the module
    docstring's "Event id collisions" note: ids are assigned in ascending
    time order, and a millisecond-rounding collision with an
    already-assigned id is broken with a ``_dup<n>`` suffix, incrementing
    until unique.
    """
    seen: dict[str, int] = {}
    ids = []
    for det in detections:
        base = event_id_from_time(det["time"])
        if base not in seen:
            seen[base] = 0
            ids.append(base)
        else:
            n = seen[base] + 1
            candidate = f"{base}_dup{n}"
            while candidate in seen:
                n += 1
                candidate = f"{base}_dup{n}"
            seen[base] += 1
            seen[candidate] = 0
            ids.append(candidate)
    return ids


def merge_windows(per_window_results: list[list[dict]], cfg: Config) -> pd.DataFrame:
    """Flatten, de-duplicate and assign event ids to per-window raw triggers.

    Parameters
    ----------
    per_window_results : list of list of dict
        One raw trigger list per window, each in the shape
        :func:`openmeq.detect.worker.detect_window` returns (deliberately
        overlapping/duplicated near window boundaries).
    cfg : openmeq.config.Config
        Only ``cfg.detection.dedup_window_s`` is used.

    Returns
    -------
    pandas.DataFrame
        Exactly the ARCHITECTURE.md §4.3 columns
        (:data:`DETECTIONS_COLUMNS`), sorted by ``time``. ``stations`` and
        ``trace_ids`` are Python lists of strings at this stage (joined into
        ``;``-separated strings only in :func:`write_detections_csv`).
    """
    flattened = [trig for window in per_window_results for trig in window]
    flattened.sort(key=lambda trig: UTCDateTime(trig["time"]))

    clusters = _cluster_triggers(flattened, cfg.detection.dedup_window_s)
    detections = [_merge_cluster(cluster) for cluster in clusters]
    detections.sort(key=lambda det: det["time"])

    event_ids = _assign_event_ids(detections)

    rows = []
    for event_id, det in zip(event_ids, detections, strict=True):
        rows.append(
            {
                "event_id": event_id,
                "time": iso_z(det["time"]),
                "duration_s": det["duration_s"],
                "coincidence_sum": det["coincidence_sum"],
                "n_stations": det["n_stations"],
                "stations": det["stations"],
                "trace_ids": det["trace_ids"],
            }
        )

    df = pd.DataFrame(rows, columns=DETECTIONS_COLUMNS)
    return df


def write_detections_csv(df: pd.DataFrame, path) -> None:
    """Write ``df`` (as produced by :func:`merge_windows`) to ``path``.

    ``stations``/``trace_ids`` (Python lists in ``df``) are ``;``-joined
    into strings (ARCHITECTURE.md §4.3), never written as Python ``list``
    ``repr()`` strings. The parent directory is created if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have exactly :data:`DETECTIONS_COLUMNS`.
    path : str or pathlib.Path
        Destination CSV path, e.g. ``RunPaths.detections_csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["stations"] = out["stations"].apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    out["trace_ids"] = out["trace_ids"].apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    out.to_csv(path, index=False, columns=DETECTIONS_COLUMNS)
