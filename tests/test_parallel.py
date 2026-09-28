"""Tests for openmeq.parallel: pmap, pmap_incremental, thread limits.

Functions used as ``func`` with ``n_workers > 1`` must be top-level and
picklable (see openmeq/parallel.py's module docstring), so the worker
functions below are defined at module level rather than as closures.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from openmeq.parallel import ParallelSettings, pmap, pmap_incremental

# ---------------------------------------------------------------------------
# top-level worker functions (must be picklable for n_workers > 1 tests)
# ---------------------------------------------------------------------------


def _square(x: int) -> int:
    return x * x


def _sleep_inversely_and_return(x: int) -> int:
    """Sleep longer for *earlier* items, so parallel workers finish out of order."""
    time.sleep(0.05 * (5 - x) if x < 5 else 0.0)
    return x


def _read_omp_num_threads(_x: int) -> str | None:
    return os.environ.get("OMP_NUM_THREADS")


def _boom(x: int) -> int:
    if x == 3:
        raise ValueError("boom on 3")
    return x


def _identity(x: int) -> int:
    return x


# ---------------------------------------------------------------------------
# pmap
# ---------------------------------------------------------------------------


def test_pmap_serial_n_workers_1():
    """n_workers=1 runs in-process: correct, ordered results, no pickling needed."""
    calls = []

    def unpicklable_closure(x):
        # A closure over a mutable local is not picklable; if pmap tried to
        # send this through a multiprocessing queue it would raise. Success
        # here proves the n_workers=1 path never pickles func.
        calls.append(x)
        return x * 2

    pcfg = ParallelSettings(n_workers=1, threads_per_worker=1)
    items = [1, 2, 3, 4, 5]
    result = pmap(unpicklable_closure, items, pcfg)

    assert result == [2, 4, 6, 8, 10]
    assert calls == items  # processed in-process, in order


def test_pmap_parallel_matches_serial():
    pcfg_serial = ParallelSettings(n_workers=1, threads_per_worker=1)
    pcfg_parallel = ParallelSettings(n_workers=2, threads_per_worker=1)
    items = list(range(10))

    serial_result = pmap(_square, items, pcfg_serial)
    parallel_result = pmap(_square, items, pcfg_parallel)

    assert sorted(serial_result) == sorted(parallel_result)
    assert serial_result == [x * x for x in items]
    assert parallel_result == [x * x for x in items]


def test_pmap_ordered_preserves_order():
    pcfg = ParallelSettings(n_workers=3, threads_per_worker=1)
    items = list(range(6))  # items 0..4 sleep decreasingly, item 5 doesn't sleep

    result = pmap(_sleep_inversely_and_return, items, pcfg, ordered=True)

    assert result == items


def test_pmap_unordered_returns_same_set():
    pcfg = ParallelSettings(n_workers=3, threads_per_worker=1)
    items = list(range(6))

    result = pmap(_sleep_inversely_and_return, items, pcfg, ordered=False)

    assert sorted(result) == sorted(items)


def test_pmap_propagates_exception():
    pcfg = ParallelSettings(n_workers=2, threads_per_worker=1)
    items = list(range(6))

    with pytest.raises(ValueError, match="boom on 3"):
        pmap(_boom, items, pcfg)


def test_pmap_propagates_exception_serial():
    pcfg = ParallelSettings(n_workers=1, threads_per_worker=1)
    items = list(range(6))

    with pytest.raises(ValueError, match="boom on 3"):
        pmap(_boom, items, pcfg)


def test_pmap_auto_resolves_to_positive_int():
    pcfg = ParallelSettings(n_workers="auto", threads_per_worker=1)
    result = pmap(_identity, [1, 2, 3], pcfg)
    assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# worker thread env vars
# ---------------------------------------------------------------------------


def test_worker_thread_env_vars_set():
    pcfg = ParallelSettings(n_workers=2, threads_per_worker=3)
    items = list(range(4))

    result = pmap(_read_omp_num_threads, items, pcfg)

    assert result == ["3"] * len(items)


# ---------------------------------------------------------------------------
# pmap_incremental
# ---------------------------------------------------------------------------


def test_pmap_incremental_resumes(tmp_path):
    out_path = tmp_path / "results.jsonl"
    call_counts: dict[int, int] = {}

    def counted(x: int) -> int:
        call_counts[x] = call_counts.get(x, 0) + 1
        return x * 10

    # n_workers=1 (in-process) here: this test is about the resume/skip
    # bookkeeping in pmap_incremental itself, not about multiprocessing
    # correctness (that's covered by test_pmap_parallel_matches_serial and
    # friends), so a plain in-process call counter is the simplest reliable
    # side channel.
    pcfg = ParallelSettings(n_workers=1, threads_per_worker=1)
    items = list(range(10))

    # First, partial run over a subset only (simulates a run that was
    # killed after finishing the first half).
    pmap_incremental(counted, items[:4], pcfg, out_path, key=str)
    assert call_counts == {i: 1 for i in range(4)}

    lines_after_partial = out_path.read_text().splitlines()
    assert len(lines_after_partial) == 4

    # Resume with the *full* item list: already-done items must not be
    # recomputed.
    result = pmap_incremental(counted, items, pcfg, out_path, key=str)

    assert call_counts == {i: 1 for i in items}  # no item recomputed
    assert result == [i * 10 for i in items]

    lines_after_full = out_path.read_text().splitlines()
    assert len(lines_after_full) == 10
    keys = [json.loads(line)["key"] for line in lines_after_full]
    assert sorted(keys) == sorted(str(i) for i in items)
    assert len(set(keys)) == 10  # unique-keyed lines


def test_pmap_incremental_parallel(tmp_path):
    out_path = tmp_path / "results_parallel.jsonl"
    pcfg = ParallelSettings(n_workers=2, threads_per_worker=1)
    items = list(range(8))

    result = pmap_incremental(_square, items, pcfg, out_path, key=str)

    assert result == [x * x for x in items]
    lines = out_path.read_text().splitlines()
    assert len(lines) == 8
    for line in lines:
        record = json.loads(line)
        assert record["result"] == int(record["key"]) ** 2


def test_pmap_incremental_tolerates_truncated_last_line(tmp_path):
    out_path = tmp_path / "results.jsonl"
    out_path.write_text(
        json.dumps({"key": "0", "result": 0}) + "\n" + '{"key": "1", "resul'  # truncated
    )
    pcfg = ParallelSettings(n_workers=1, threads_per_worker=1)

    result = pmap_incremental(_identity, [0, 1, 2], pcfg, out_path, key=str)

    # Tolerated: item 0's valid line is kept as-done; the truncated line for
    # item 1 is unreadable and so item 1 (and item 2, never written before)
    # are (re)computed and appended as fresh, valid lines. The result is
    # correct even though the file still contains that one unparseable
    # leftover line from the killed run.
    assert result == [0, 1, 2]
    lines = out_path.read_text().splitlines()
    parsed_keys = []
    for line in lines:
        try:
            parsed_keys.append(json.loads(line)["key"])
        except json.JSONDecodeError:
            continue  # the leftover truncated line from before
    assert sorted(parsed_keys, key=lambda k: (len(k), k)) == ["0", "1", "2"]
