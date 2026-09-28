"""``chunks_pn/data_list.csv`` writer (M4.2, ``OpenMEQ-1m7.2``).

Per ``docs/phasenet_interface.md`` §3 (M4.1's confirmed
``predict.py --format=mseed`` contract): a single ``fname`` column, one row
per event, whose value is ``<event_id>/<event_id>.mseed`` -- relative to
``chunks_pn/`` (M4.3's ``--data_dir=chunks_pn``), matching the PhaseNet-ready
chunk :func:`openmeq.pick.prep.prepare_chunk` writes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmeq.paths import RunPaths

#: ``data_list.csv``'s one and only column (predict.py's mseed data reader
#: ignores every other column -- docs/phasenet_interface.md §3).
DATA_LIST_COLUMNS = ["fname"]


def write_data_list(paths: RunPaths, event_ids: list[str]) -> Path:
    """Write ``chunks_pn/data_list.csv``: one row per event.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    event_ids : list of str
        Every event with a prepared chunk (``chunks_pn/<event_id>/<event_id>.mseed``).
        Written in the given order -- callers wanting a deterministic file
        should pass a pre-sorted list.

    Returns
    -------
    pathlib.Path
        ``paths.data_list_csv``.
    """
    out_path = paths.data_list_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(DATA_LIST_COLUMNS)
        for event_id in event_ids:
            writer.writerow([f"{event_id}/{event_id}.mseed"])
    return out_path


__all__ = ["DATA_LIST_COLUMNS", "write_data_list"]
