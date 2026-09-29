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

import numpy as np
import pandas as pd
from obspy import read

from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.plotting import plot_boxplot, plot_heatmap, save_fig, setup_style

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


# ---------------------------------------------------------------------------
# Extended QAQC (``OpenMEQ-02o.3``): component coverage + raw-vs-PN examples
# ---------------------------------------------------------------------------
#
# Written to ``chunks/diagnostics/extended/`` only when
# ``cfg.diagnostics.extended_enabled("chunk")`` is true. Every output is
# wrapped by :func:`_safe_extended` -- an extended-QAQC failure logs a
# warning and never fails the step.
#
# Where the hooks live
# --------------------
# ``chunks_pn/`` is written by the *pick* step's prep
# (:func:`openmeq.pick.prep.prepare_all`), not by ``openmeq chunk``, so the
# chunk extended set is split in two:
#
# * :func:`make_chunk_extended_diagnostics` -- called from
#   :func:`openmeq.chunk.run` right after the baseline diagnostics. Writes
#   ``component_coverage.{csv,png}`` from the raw chunks only (any existing
#   ``chunks_pn/`` is ignored: the chunks were just rewritten, so it may be
#   stale).
# * :func:`make_chunk_pn_extended_diagnostics` -- called from
#   :func:`openmeq.pick.run` right after ``prepare_all``. Rewrites
#   ``component_coverage.{csv,png}`` with both raw and ``chunks_pn``
#   columns, and writes ``raw_vs_pn_<event_id>.png`` for up to
#   ``cfg.diagnostics.n_examples`` evenly-sampled events.
#
# Both are gated by the *chunk* diagnostics switch and write into the chunk
# diagnostics extended dir even though the second runs during ``openmeq
# pick``: the figures are about chunk contents, so they live in one place.

#: ``chunks/diagnostics/extended/component_coverage.csv``'s column order.
#: ``n_pn``/``frac_pn`` are empty until the pick-step hook has run.
COMPONENT_COVERAGE_COLUMNS = [
    "station",
    "component",
    "n_events",
    "n_raw",
    "frac_raw",
    "n_pn",
    "frac_pn",
]

#: Preferred component column order in the coverage matrix; anything else
#: seen (an unmapped channel's own last character -- usually a
#: ``channel_map`` gap) is appended alphabetically.
_COMPONENT_ORDER = ["Z", "1", "2", "N", "E"]

#: A raw-vs-PN figure shows all three components (Z/1/2 columns) only up
#: to this many stations; above it, Z only.
_COMPACT_MAX_STATIONS = 8

#: raw-vs-PN horizontal slots: PN's rotated ``N``/``E`` share the ``1``/``2``
#: slots with the unoriented raw horizontals (the legend names the channel).
_HORIZONTAL_SLOT = {"1": "1", "N": "1", "2": "2", "E": "2"}


def _extended_dir(paths: RunPaths) -> Path:
    return Path(paths.diagnostics_dir("chunks")) / "extended"


def _extended_on(cfg: Config) -> bool:
    diagnostics = getattr(cfg, "diagnostics", None)
    return bool(diagnostics is not None and diagnostics.extended_enabled("chunk"))


def _safe_extended(name: str, fn, *args, **kwargs) -> None:
    """Run one extended-QAQC output; log (never raise) on failure."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
        logger.warning("chunk extended diagnostics: %s failed: %s", name, exc)


def _mapped_component(channel: str, channel_map: dict[str, str]) -> str:
    """Last channel character after ``cfg.stations.channel_map`` (no rotation)."""
    if channel in channel_map:
        return channel_map[channel]
    return channel[-1:] if channel else "?"


def _raw_components(chunk_path: Path, channel_map: dict[str, str]) -> set[tuple[str, str]]:
    """``{(NET.STA, component)}`` with any data in one raw chunk (headonly)."""
    if not chunk_path.is_file():
        return set()
    stream = read(str(chunk_path), headonly=True)
    return {
        (f"{tr.stats.network}.{tr.stats.station}", _mapped_component(tr.stats.channel, channel_map))
        for tr in stream
        if tr.stats.npts > 0
    }


def _pn_components(pn_path: Path) -> set[tuple[str, str]]:
    """``{(NET.STA, component)}`` with real (not zero-filled) data in one PN chunk."""
    if not pn_path.is_file():
        return set()
    stream = read(str(pn_path))
    return {
        (f"{tr.stats.network}.{tr.stats.station}", tr.stats.channel[-1:])
        for tr in stream
        if tr.stats.npts > 0 and np.any(np.asarray(tr.data) != 0)
    }


def compute_component_coverage(
    cfg: Config, paths: RunPaths, event_ids: list[str], *, include_pn: bool
) -> pd.DataFrame:
    """Fraction of events with each ``(station, component)`` present.

    Stations are ``NET.STA`` (location codes collapsed, matching PhaseNet
    prep's own location collation). Raw components are each channel's last
    character after ``cfg.stations.channel_map`` (no rotation); ``chunks_pn``
    components are the written channel's last character, so rotated
    ``N``/``E`` show as their own components there. A zero-filled
    ``chunks_pn`` component (prep's "exactly 3 components" padding) does
    **not** count as present. Every seed id in ``inventory/index.json``
    (when it exists) is added to the station/component universe, and the
    result is the full station x component grid (always including
    ``Z``/``1``/``2``), so a component that never reaches any chunk -- e.g.
    a missing horizontal -- still shows as a ``0`` row.

    Parameters
    ----------
    cfg : openmeq.config.Config
        Only ``cfg.stations.channel_map`` is used.
    paths : openmeq.paths.RunPaths
    event_ids : list of str
        The denominator: every event counted (a missing chunk counts as
        "absent" for every component).
    include_pn : bool, keyword-only
        Also read ``chunks_pn/<event_id>/<event_id>.mseed``.

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`COMPONENT_COVERAGE_COLUMNS`; ``n_pn``/
        ``frac_pn`` are NaN when ``include_pn`` is false.
    """
    channel_map = dict(cfg.stations.channel_map)
    raw_counts: dict[tuple[str, str], int] = {}
    pn_counts: dict[tuple[str, str], int] = {}
    universe: set[tuple[str, str]] = set()

    if Path(paths.index_json).is_file():
        for seed_id in DataIndex.load(paths.index_json).seed_ids():
            net, sta, _loc, cha = seed_id.split(".")
            universe.add((f"{net}.{sta}", _mapped_component(cha, channel_map)))

    for event_id in event_ids:
        for key in _raw_components(paths.chunk_file(event_id), channel_map):
            raw_counts[key] = raw_counts.get(key, 0) + 1
        if include_pn:
            for key in _pn_components(paths.chunk_pn_file(event_id)):
                pn_counts[key] = pn_counts.get(key, 0) + 1
    universe |= set(raw_counts) | set(pn_counts)
    # Full station x component grid (always including Z/1/2), so a missing
    # horizontal is an explicit 0 row, never an absent one.
    stations = {station for station, _ in universe}
    components = {"Z", "1", "2"} | {component for _, component in universe}
    universe = {(s, c) for s in stations for c in components}

    n_events = len(event_ids)
    rows = []
    for station, component in sorted(universe):
        n_raw = raw_counts.get((station, component), 0)
        n_pn = pn_counts.get((station, component), 0) if include_pn else np.nan
        rows.append(
            {
                "station": station,
                "component": component,
                "n_events": n_events,
                "n_raw": n_raw,
                "frac_raw": n_raw / n_events if n_events else 0.0,
                "n_pn": n_pn,
                "frac_pn": (n_pn / n_events if n_events else 0.0) if include_pn else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=COMPONENT_COVERAGE_COLUMNS)


def _component_sort_key(component: str) -> tuple[int, str]:
    if component in _COMPONENT_ORDER:
        return (_COMPONENT_ORDER.index(component), component)
    return (len(_COMPONENT_ORDER), component)


def _write_component_coverage(coverage_df: pd.DataFrame, out_dir: Path, include_pn: bool) -> None:
    """``component_coverage.csv`` + a station x ``source:component`` heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(
        out_dir / "component_coverage.csv", index=False, columns=COMPONENT_COVERAGE_COLUMNS
    )

    if coverage_df.empty:
        matrix = pd.DataFrame()
    else:
        components = sorted(coverage_df["component"].unique(), key=_component_sort_key)
        sources = [("raw", "frac_raw")] + ([("pn", "frac_pn")] if include_pn else [])
        blocks = []
        for label, column in sources:
            block = coverage_df.pivot_table(
                index="station", columns="component", values=column, aggfunc="first"
            ).reindex(columns=components)
            block.columns = [f"{label}:{c}" for c in components]
            blocks.append(block)
        matrix = pd.concat(blocks, axis=1).fillna(0.0)

    plot_heatmap(
        matrix,
        out_dir / "component_coverage.png",
        xlabel="Source : component (after channel_map)",
        ylabel="Station",
        title="Chunk component coverage (fraction of events present)",
        cbar_label="fraction of events",
        vmin=0.0,
        vmax=1.0,
    )


def _sample_evenly(items: list[str], n: int) -> list[str]:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n)
    return [items[int(round(i))] for i in idx]


def _normalised(data) -> np.ndarray:
    """Demeaned, peak-normalised copy (an all-zero trace stays all-zero)."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    data = data - data.mean()
    peak = np.max(np.abs(data))
    return data / peak if peak > 0 else data


def _slot_traces(stream, component_of) -> dict[tuple[str, str], object]:
    """``{(NET.STA, slot): trace}``, slot one of ``Z``/``1``/``2`` (first trace wins)."""
    out: dict[tuple[str, str], object] = {}
    for tr in sorted(stream, key=lambda t: t.id):
        comp = component_of(tr)
        slot = "Z" if comp == "Z" else _HORIZONTAL_SLOT.get(comp)
        if slot is None:
            continue
        out.setdefault((f"{tr.stats.network}.{tr.stats.station}", slot), tr)
    return out


def _plot_raw_vs_pn(cfg: Config, paths: RunPaths, event_id: str, path: Path) -> None:
    """One event: normalised raw chunk (grey) overlaid with its PN chunk (blue).

    One row per station, one column per component (Z/1/2 when at most
    :data:`_COMPACT_MAX_STATIONS` stations, else Z only). A zero-filled PN
    component is drawn dotted and labelled as such.
    """
    setup_style()
    import matplotlib.pyplot as plt

    channel_map = dict(cfg.stations.channel_map)
    raw = read(str(paths.chunk_file(event_id)))
    raw.merge(method=1, fill_value=0)
    pn = read(str(paths.chunk_pn_file(event_id)))

    raw_slots = _slot_traces(raw, lambda tr: _mapped_component(tr.stats.channel, channel_map))
    pn_slots = _slot_traces(pn, lambda tr: tr.stats.channel[-1:])
    stations = sorted({s for s, _ in raw_slots} | {s for s, _ in pn_slots})
    slots = ["Z", "1", "2"] if len(stations) <= _COMPACT_MAX_STATIONS else ["Z"]

    all_traces = list(raw_slots.values()) + list(pn_slots.values())
    t0 = min((tr.stats.starttime for tr in all_traces), default=None)

    n_rows = max(1, len(stations))
    fig, axes = plt.subplots(
        n_rows,
        len(slots),
        figsize=(4.0 * len(slots) + 1.0, max(2.0, 1.1 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    for i, station in enumerate(stations):
        for j, slot in enumerate(slots):
            ax = axes[i, j]
            raw_tr = raw_slots.get((station, slot))
            pn_tr = pn_slots.get((station, slot))
            if raw_tr is not None:
                ax.plot(
                    raw_tr.times(reftime=t0),
                    _normalised(raw_tr.data),
                    color="0.55",
                    linewidth=0.6,
                    label=f"raw {raw_tr.stats.channel}",
                )
            if pn_tr is not None:
                zero_filled = not np.any(np.asarray(pn_tr.data) != 0)
                ax.plot(
                    pn_tr.times(reftime=t0),
                    _normalised(pn_tr.data),
                    color="#0072B2",
                    linewidth=0.6,
                    linestyle=":" if zero_filled else "-",
                    label=f"pn {pn_tr.stats.channel}" + (" (zero-filled)" if zero_filled else ""),
                )
                ax.legend(loc="upper right", fontsize=5, frameon=False)
            elif raw_tr is not None:
                ax.legend(loc="upper right", fontsize=5, frameon=False)
            else:
                ax.text(0.5, 0.5, "absent", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(station, rotation=0, ha="right", va="center", fontsize=8)
            if i == 0:
                ax.set_title(f"component {slot}", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time since chunk start (s)")

    fig.suptitle(f"{event_id}: raw vs PhaseNet-prepped chunk (each trace peak-normalised)")
    fig.tight_layout()
    save_fig(fig, path)


def make_chunk_extended_diagnostics(cfg: Config, paths: RunPaths, event_ids: list[str]) -> None:
    """Chunk-step extended QAQC: raw-only ``component_coverage.{csv,png}``.

    No-op unless ``cfg.diagnostics.extended_enabled("chunk")``. Called from
    :func:`openmeq.chunk.run`. ``raw_vs_pn_<event_id>.png`` needs
    ``chunks_pn/`` and is deferred to
    :func:`make_chunk_pn_extended_diagnostics` (see the section comment
    above). Never raises.
    """
    if not _extended_on(cfg):
        return
    out_dir = _extended_dir(paths)
    event_ids = list(event_ids)

    def _coverage() -> None:
        df = compute_component_coverage(cfg, paths, event_ids, include_pn=False)
        _write_component_coverage(df, out_dir, include_pn=False)

    _safe_extended("component_coverage", _coverage)
    logger.info(
        "chunk extended diagnostics: raw_vs_pn deferred until chunks_pn/ exists "
        "(written by the pick step's PhaseNet prep)."
    )


def make_chunk_pn_extended_diagnostics(cfg: Config, paths: RunPaths, event_ids: list[str]) -> None:
    """Post-PhaseNet-prep chunk extended QAQC (raw + ``chunks_pn``).

    No-op unless ``cfg.diagnostics.extended_enabled("chunk")``. Called from
    :func:`openmeq.pick.run` after :func:`openmeq.pick.prep.prepare_all`.
    Rewrites ``component_coverage.{csv,png}`` with ``chunks_pn`` columns and
    writes ``raw_vs_pn_<event_id>.png`` for up to
    ``cfg.diagnostics.n_examples`` events, evenly sampled (in the given
    order) from those with both a raw and a PN chunk; with no PN chunk at
    all the examples are skipped with a log message. Never raises.
    """
    if not _extended_on(cfg):
        return
    out_dir = _extended_dir(paths)
    event_ids = list(event_ids)

    def _coverage() -> None:
        df = compute_component_coverage(cfg, paths, event_ids, include_pn=True)
        _write_component_coverage(df, out_dir, include_pn=True)

    _safe_extended("component_coverage", _coverage)

    both = [
        e for e in event_ids if paths.chunk_file(e).is_file() and paths.chunk_pn_file(e).is_file()
    ]
    if not both:
        logger.info("chunk extended diagnostics: no chunks_pn/ chunks found -- skipping raw_vs_pn.")
        return
    for event_id in _sample_evenly(both, cfg.diagnostics.n_examples):
        _safe_extended(
            f"raw_vs_pn_{event_id}",
            _plot_raw_vs_pn,
            cfg,
            paths,
            event_id,
            out_dir / f"raw_vs_pn_{event_id}.png",
        )
