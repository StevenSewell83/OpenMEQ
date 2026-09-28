"""Chunk completeness report and diagnostics (M3.2, ``OpenMEQ-rq8.2``).

Implements ARCHITECTURE.md §3's ``chunks/completeness.csv`` and PLAN.md
§5's Chunk diagnostics row ("Chunk completeness per station -- fraction of
the window present"), completing the M3 epic started by M3.1
(``openmeq/chunk.py``'s ``write_chunks``).

Decoupled recomputation from written files
---------------------------------------------
:func:`compute_completeness` recomputes completeness by **re-reading the
written chunk files** (headonly, via ``obspy.read``), not by trusting
``write_chunks``'s own in-memory ``{"event_id", "seed_id", "npts_present"}``
return value. This keeps chunk writing (M3.1) and completeness reporting
(M3.2) independently testable, and matches ARCHITECTURE.md §5's general
"every step reads the previous step's actual on-disk output" convention
(there written for the QuakeML chain, applied here to a non-QuakeML
output).

``expected_npts`` formula
----------------------------
``expected_npts = round((cfg.chunks.pre_s + cfg.chunks.post_s) *
sampling_rate) + 1`` -- inclusive of both the window's start and end
sample (the beads issue's own design note: "pick this convention and use
it consistently"). ``sampling_rate`` is read directly off the present
trace(s) for that ``(event_id, seed_id)`` (never re-derived from
``inventory/index.json``), since a channel actually present in the chunk
file is unambiguous ground truth for its own native rate. A
``(event_id, seed_id)`` with **no** data at all needs no ``sampling_rate``
to reach ``fraction_present = 0.0`` -- ``0 / expected_npts`` is ``0.0``
for any positive ``expected_npts``, so this module never has to guess a
missing channel's sampling rate from index metadata.

Zero-data rows are never omitted
------------------------------------
A ``(event_id, seed_id)`` pair with no trace at all in
``chunks/<event_id>/<event_id>.mseed`` (the station had zero data in that
event's window -- down, out of range, or the whole event's chunk file is
itself missing because *no* channel had any data, ``openmeq.chunk
.chunk_day_group``'s own "no chunk written" case) still gets a row with
``fraction_present = 0.0``, for every ``seed_id`` in the **wider**
``inventory/index.json`` (:meth:`openmeq.index.DataIndex.seed_ids`), not
just whatever happens to already be in the chunk file. This is the design
note's explicit requirement: absence must be visible in
``completeness.csv``/the heatmap as a clear ``0``, never a silently
omitted row that could be misread as "not applicable".

Clamping
-----------
``fraction_present`` is always ``min(1.0, actual_npts / expected_npts)``
and never negative (``actual_npts >= 0`` by construction) -- so an
off-by-one in ``expected_npts`` or a trace that is very slightly longer
than the nominal window (a real possibility right at a sample-rate/window
boundary) can never push a value past ``1.0`` or below ``0.0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from obspy import read

from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.plotting import plot_boxplot, plot_heatmap

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("chunk.diagnostics")

#: ``chunks/completeness.csv``'s exact column order (ARCHITECTURE.md §3).
COMPLETENESS_COLUMNS = ["event_id", "seed_id", "fraction_present"]

#: ``chunks/diagnostics/summary.csv``'s exact column order.
SUMMARY_COLUMNS = ["metric", "value"]

#: Rows below this fraction count as "incomplete" in the summary.
_INCOMPLETE_THRESHOLD = 0.99


def _npts_and_rate_by_seed_id(chunk_path: Path) -> tuple[dict[str, int], dict[str, float]]:
    """Header-only read of one written chunk file -> per-seed-id summed ``npts``/native rate.

    Sums ``npts`` across every trace segment sharing a ``seed_id`` (a
    multi-segment/gappy channel, from M3.1's ``Stream.split()`` around a
    known gap, must not have only its first segment counted -- the beads
    issue's own "Pitfalls" note). Returns ``({}, {})`` for a chunk file
    that does not exist (the whole event had zero data in its window,
    ``openmeq.chunk.chunk_day_group``'s "no chunk written" case) or cannot
    be read (logged, never raised -- one bad/corrupt file must not abort
    completeness reporting for every other event).
    """
    if not chunk_path.is_file():
        return {}, {}
    try:
        stream = read(str(chunk_path), headonly=True)
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole report
        logger.warning("Could not read chunk file %s for completeness: %s", chunk_path, exc)
        return {}, {}

    npts_by_seed_id: dict[str, int] = {}
    rate_by_seed_id: dict[str, float] = {}
    for tr in stream:
        seed_id = tr.get_id()
        npts_by_seed_id[seed_id] = npts_by_seed_id.get(seed_id, 0) + tr.stats.npts
        rate_by_seed_id.setdefault(seed_id, float(tr.stats.sampling_rate))
    return npts_by_seed_id, rate_by_seed_id


def compute_completeness(detections_df: pd.DataFrame, cfg: Config, paths: RunPaths) -> pd.DataFrame:
    """Recompute chunk completeness for every ``(event_id, seed_id)`` pair.

    Parameters
    ----------
    detections_df : pandas.DataFrame
        ``detections/detections.csv``'s columns (ARCHITECTURE.md §4.3);
        only ``event_id`` is used.
    cfg : openmeq.config.Config
        Only ``cfg.chunks.pre_s``/``post_s`` are used (the expected-window
        duration).
    paths : openmeq.paths.RunPaths
        ``paths.chunk_file(event_id)`` locates each event's written chunk;
        ``paths.index_json`` gives the wider set of ``seed_id``\\ s every
        event must report a row for (see the module docstring).

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`COMPLETENESS_COLUMNS`
        (``event_id,seed_id,fraction_present``), one row per
        ``(event_id, seed_id)`` for every event in ``detections_df`` x
        every ``seed_id`` in ``inventory/index.json``.
    """
    index = DataIndex.load(paths.index_json)
    all_seed_ids = sorted(index.seed_ids())
    pre_s = cfg.chunks.pre_s
    post_s = cfg.chunks.post_s

    rows: list[dict] = []
    if detections_df is None or detections_df.empty:
        return pd.DataFrame(rows, columns=COMPLETENESS_COLUMNS)

    for event_id in detections_df["event_id"]:
        chunk_path = paths.chunk_file(event_id)
        npts_by_seed_id, rate_by_seed_id = _npts_and_rate_by_seed_id(chunk_path)

        for seed_id in all_seed_ids:
            actual_npts = npts_by_seed_id.get(seed_id, 0)
            if actual_npts <= 0:
                fraction_present = 0.0
            else:
                sampling_rate = rate_by_seed_id[seed_id]
                expected_npts = round((pre_s + post_s) * sampling_rate) + 1
                fraction_present = (
                    min(1.0, actual_npts / expected_npts) if expected_npts > 0 else 0.0
                )
            rows.append(
                {
                    "event_id": event_id,
                    "seed_id": seed_id,
                    "fraction_present": round(max(0.0, fraction_present), 6),
                }
            )

    return pd.DataFrame(rows, columns=COMPLETENESS_COLUMNS)


def write_completeness_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write ``df`` to ``path`` (``chunks/completeness.csv``, ARCHITECTURE.md §3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame(columns=COMPLETENESS_COLUMNS)
    df.to_csv(path, index=False, columns=COMPLETENESS_COLUMNS)


def _completeness_heatmap(completeness_df: pd.DataFrame, path: Path) -> None:
    if completeness_df is None or completeness_df.empty:
        plot_heatmap(
            pd.DataFrame(),
            path,
            xlabel="Seed id",
            ylabel="Event id",
            title="Chunk completeness (fraction of window present)",
            vmin=0.0,
            vmax=1.0,
        )
        return

    pivot = completeness_df.pivot_table(
        index="event_id", columns="seed_id", values="fraction_present", aggfunc="first"
    )
    plot_heatmap(
        pivot,
        path,
        xlabel="Seed id",
        ylabel="Event id",
        title="Chunk completeness (fraction of window present)",
        cbar_label="fraction_present",
        vmin=0.0,
        vmax=1.0,
    )


def _completeness_boxplot(completeness_df: pd.DataFrame, path: Path) -> None:
    if completeness_df is None or completeness_df.empty:
        plot_boxplot(
            pd.DataFrame(columns=["seed_id", "fraction_present"]),
            path,
            by="seed_id",
            column="fraction_present",
            xlabel="Seed id",
            ylabel="Fraction of window present",
            title="Chunk completeness per station",
        )
        return

    order = sorted(completeness_df["seed_id"].unique())
    plot_boxplot(
        completeness_df,
        path,
        by="seed_id",
        column="fraction_present",
        order=order,
        xlabel="Seed id",
        ylabel="Fraction of window present",
        title="Chunk completeness per station",
    )


def _write_summary(completeness_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if completeness_df is None or completeness_df.empty:
        rows = [
            {"metric": "n_rows", "value": 0},
            {"metric": "n_events", "value": 0},
            {"metric": "n_seed_ids", "value": 0},
            {"metric": "mean_fraction_present", "value": ""},
            {"metric": "min_fraction_present", "value": ""},
            {"metric": "n_incomplete_rows", "value": 0},
            {"metric": "n_zero_rows", "value": 0},
        ]
    else:
        fraction = completeness_df["fraction_present"]
        rows = [
            {"metric": "n_rows", "value": int(len(completeness_df))},
            {"metric": "n_events", "value": int(completeness_df["event_id"].nunique())},
            {"metric": "n_seed_ids", "value": int(completeness_df["seed_id"].nunique())},
            {"metric": "mean_fraction_present", "value": float(fraction.mean())},
            {"metric": "min_fraction_present", "value": float(fraction.min())},
            {
                "metric": "n_incomplete_rows",
                "value": int((fraction < _INCOMPLETE_THRESHOLD).sum()),
            },
            {"metric": "n_zero_rows", "value": int((fraction == 0.0).sum())},
        ]

    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


def make_chunk_diagnostics(paths: RunPaths, completeness_df: pd.DataFrame) -> None:
    """Build every ``chunks/diagnostics/`` output from ``completeness_df``.

    PLAN.md §5's Chunk diagnostics row ("Chunk completeness per station --
    fraction of the window present"), via :mod:`openmeq.plotting` (nothing
    here imports ``matplotlib`` directly).

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
        Only ``paths.diagnostics_dir("chunks")`` is used.
    completeness_df : pandas.DataFrame
        :func:`compute_completeness`'s output. An empty (zero-row) or
        ``None`` frame still produces every output file with an explicit
        "no data" placeholder, matching :mod:`openmeq.detect.diagnostics`'s
        own convention.
    """
    out_dir = Path(paths.diagnostics_dir("chunks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _completeness_heatmap(completeness_df, out_dir / "completeness_heatmap.png")
    _completeness_boxplot(completeness_df, out_dir / "completeness_boxplot.png")
    _write_summary(completeness_df, out_dir / "summary.csv")
