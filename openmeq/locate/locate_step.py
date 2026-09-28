"""Locate step orchestration (M6.4 / OpenMEQ-xdf.4): pass A -> pass B ->
finalize, per event, parallelised over events.

:func:`run_locate` is the function ``openmeq locate`` (M6.5, not registered
here -- see the module docstring's "Out of scope" note below) will call.
Reads ``pick/catalog_picks.xml`` (M4.4's output: events with automatic
picks, no origins yet), runs :func:`openmeq.locate.pass_a.run_pass_a`,
:func:`openmeq.locate.pass_b.run_pass_b` and
:func:`openmeq.locate.finalize.finalize_event` for every event, and writes
the three M6 output artifacts (ARCHITECTURE.md §3):
``locate/catalog_auto.xml``, ``locate/summary.csv``,
``locate/removal_log.csv``.

**Per-event parallelism.** Per the M6 epic's cross-cutting design note (also
restated in this issue's own "DESIGN" section), the ``openmeq.parallel.pmap``
unit function (:func:`_locate_one_event`) wraps all three steps -- pass A,
pass B, finalize -- together for *one* event, rather than parallelising each
step separately (as ``openmeq.locate.pass_b.run_pass_b_many`` alone would).
This halves the number of process-pool round trips per event (one dispatch
instead of two) and keeps each worker's per-event NLLoc grid/context cache
(``openmeq.locate.nlloc_runner``'s own per-process cache) warm across all
three steps for that event, rather than potentially cold-starting it twice
if pass A and pass B landed in different worker processes.

Out of scope (per the issue): diagnostic plots and the ``openmeq locate``
CLI command itself (both M6.5) -- ``run_locate`` is not registered as
``openmeq.locate.run`` (the ``openmeq.cli`` dispatch convention,
ARCHITECTURE.md §2) in this task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from obspy import Catalog

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import read_catalog, write_catalog
from openmeq.locate.finalize import (
    SummaryRow,
    finalize_event,
    write_removal_log_csv,
    write_summary_csv,
)
from openmeq.locate.pass_a import RemovalRecord, run_pass_a
from openmeq.locate.pass_b import run_pass_b
from openmeq.parallel import pmap

if TYPE_CHECKING:
    from obspy.core.event import Event

    from openmeq.config import Config
    from openmeq.paths import RunPaths


def _locate_one_event(
    item: tuple[Event, RunPaths, Config],
) -> tuple[Event, SummaryRow, list[RemovalRecord]]:
    """``openmeq.parallel.pmap``-compatible per-event unit: pass A -> pass B
    -> finalize, for one event (module docstring's "per-event parallelism"
    note)."""
    event, paths, cfg = item

    pass_a_result = run_pass_a(event, paths, cfg)
    if pass_a_result.status == "rejected":
        # PLAN.md's step ordering places the min_p_phases gate before pass B
        # begins (openmeq.locate.pass_b.run_pass_b's own early-return would
        # do the same if called, but skipping the call entirely avoids a
        # pointless S-prediction lookup against an already-rejected event).
        finalized = finalize_event(event, pass_a_result, None, cfg)
        removed = list(pass_a_result.removed)
    else:
        pass_b_result = run_pass_b(event, pass_a_result, paths, cfg)
        finalized = finalize_event(event, pass_a_result, pass_b_result, cfg)
        removed = list(pass_a_result.removed) + list(pass_b_result.removed)

    return finalized.event, finalized.summary, removed


def run_locate(paths: RunPaths, cfg: Config) -> None:
    """Locate every event in ``pick/catalog_picks.xml`` and write the three
    M6 output artifacts.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config

    Writes
    ------
    ``locate/catalog_auto.xml``
        Every event from the input catalogue, exactly once, each carrying
        its final `Origin` (pass A's, pass B's, or a rejection-only
        placeholder, ``openmeq.locate.finalize``) -- rejected events are
        flagged (``openmeq.catalog.reject_event``), never dropped
        (ARCHITECTURE.md §5).
    ``locate/summary.csv``
        One row per event, ``openmeq.locate.finalize.SUMMARY_COLUMNS``.
    ``locate/removal_log.csv``
        One row per pick removed by either pass (residual filtering) or
        pass B's S-prediction screening,
        ``openmeq.locate.finalize.REMOVAL_LOG_COLUMNS``.
    """
    input_catalog = read_catalog(paths.catalog_picks_xml)

    items = [(event, paths, cfg) for event in input_catalog]
    results = pmap(
        _locate_one_event, items, cfg.parallel, desc="locate: pass A + pass B + finalize"
    )

    events: list[Event] = []
    summary_rows: list[SummaryRow] = []
    removal_records: list[RemovalRecord] = []
    for event, summary_row, removed in results:
        events.append(event)
        summary_rows.append(summary_row)
        removal_records.extend(removed)

    out_catalog = Catalog(
        events=events, resource_id=input_catalog.resource_id, description=input_catalog.description
    )
    write_catalog(out_catalog, paths.catalog_auto_xml)
    write_summary_csv(summary_rows, paths.locate_summary_csv)
    write_removal_log_csv(removal_records, paths.removal_log_csv)

    # Defensive: every input event_id appears exactly once in summary.csv
    # (tests/locate/test_locate_step.py's own acceptance criterion) -- a
    # cheap, deterministic check here fails loudly rather than shipping a
    # silently-mismatched file.
    input_ids = [get_event_id(ev) for ev in input_catalog]
    summary_ids = [row.event_id for row in summary_rows]
    if sorted(summary_ids) != sorted(input_ids):
        raise AssertionError(
            "locate/summary.csv event_ids do not reconcile with pick/catalog_picks.xml: "
            f"{len(summary_ids)} summary rows vs {len(input_ids)} input events"
        )
