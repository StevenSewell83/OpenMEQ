"""Process-pool helper: worker init, thread limits, progress, resume.

``openmeq.parallel`` is the single parallelism primitive every heavy step in
the pipeline uses (index scanning, QC, detection, chunk writing, PhaseNet
prep/run, location, magnitude, waveform cache / cross-correlation, ...). It
exposes two entry points:

- :func:`pmap` -- a ``concurrent.futures``-flavoured parallel map.
- :func:`pmap_incremental` -- the same, but appending results to a
  JSON-lines file as each item completes, and skipping items whose key is
  already present in that file, so a killed/resumed run doesn't redo work.

Configuration
-------------
Both functions take a ``pcfg`` object with two attributes:

- ``n_workers``: an ``int``, or the string ``"auto"``.
- ``threads_per_worker``: an ``int``.

Any object or namespace with those two attributes works (duck typing) --
this module intentionally does **not** import ``openmeq.config``, to avoid a
circular import (M0.3's ``openmeq.config`` is what actually constructs the
``pcfg`` passed in as ``cfg.parallel``). :class:`ParallelSettings` is a
small concrete dataclass provided here for callers/tests that just need a
``pcfg``-shaped object without pulling in the config module.

``func`` must be a top-level, picklable function
-------------------------------------------------
When ``n_workers > 1``, ``func`` and every item are sent to worker
processes through a ``multiprocessing`` queue, which pickles them. That
means:

- ``func`` must be defined at module level (no lambdas, no closures, no
  bound methods on an unpicklable object).
- ``items`` must themselves be picklable (paths/parameters, not e.g. open
  file handles or ObsPy ``Stream`` objects held open across processes).

This is a standard ``multiprocessing`` limitation, not something this
module can paper over. ``n_workers == 1`` runs ``func`` serially, in the
calling process, with a plain loop -- no pickling occurs on that path,
which is why it is also useful for tests that pass unpicklable closures.

Because worker processes use the ``spawn`` start method (see below), any
top-level script (as opposed to the installed ``openmeq`` console script,
or ``pytest``, both of which are already fine) that calls ``pmap``/
``pmap_incremental`` with ``n_workers > 1`` at module scope must guard that
call with ``if __name__ == "__main__":`` -- otherwise spawning a worker
re-imports and re-executes the script, which is a standard Python
``multiprocessing`` requirement, not specific to this module.

Thread limits and the ``spawn`` start method
---------------------------------------------
Per ``docs/ARCHITECTURE.md`` (S2), worker processes are created with the
``spawn`` start method (not the Linux default, ``fork``), specifically so
that the thread-limiting environment variables below are guaranteed to be
set *before* NumPy/SciPy/BLAS get imported in the child:

- ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` are set
  to ``pcfg.threads_per_worker`` unconditionally (BLAS-based code always
  benefits from this).
- If TensorFlow is importable, ``tf.config.threading.
  set_intra_op_parallelism_threads``/``set_inter_op_parallelism_threads``
  are also set. The import is guarded with ``try/except`` so this module
  does not hard-require TensorFlow -- only PhaseNet-running steps need it.

Why ``spawn`` matters here: with the (Linux) default ``fork`` start method,
a worker process is a copy-on-write clone of the *parent's* address space,
taken at pool-creation time. If the parent process had already imported
NumPy/BLAS (e.g. because something else in the same interpreter imported
it earlier), that BLAS library had already read the thread-count env vars
and initialized its internal thread pool *in the parent* -- forking after
that point copies an already-initialized thread pool, and setting
``os.environ`` in a post-fork initializer is too late to change it. With
``spawn``, each worker is a *fresh* interpreter: nothing has imported NumPy
yet when it starts, so as long as the env vars are set before ``func``'s
module (and therefore NumPy, transitively) gets imported -- which is
exactly what ``_worker_init`` guarantees, because
``concurrent.futures.process`` runs the pool ``initializer`` before it ever
unpickles a submitted call item -- the child's BLAS libraries see the
right values the first time they read them.

Caveat: this guarantee covers *worker* processes. It does not retroactively
fix an already-oversubscribed *parent* process. That's why, per
ARCHITECTURE.md, :func:`pmap` and :func:`pmap_incremental` also export the
same env vars in the calling process before creating the pool (belt and
braces: workers inherit them via the environment at spawn time *and* set
them again explicitly in ``_worker_init``). If oversubscription is ever
observed in profiling (M11.3), the fix is to set these env vars even
earlier -- e.g. in the CLI entry point, before any heavy import happens in
the parent at all -- since the guarantee above only starts once
``pmap``/``pmap_incremental`` is called.

Physical vs. logical cores
---------------------------
``n_workers="auto"`` resolves to ``len(os.sched_getaffinity(0))`` (the
number of CPUs this process is allowed to run on -- Linux-only, which is
fine since this is a Linux-targeted project), falling back to
``os.cpu_count()`` if ``sched_getaffinity`` is unavailable. This is the
*logical* core count on hyperthreaded machines (not physical cores), but
it is the pragmatic choice: it respects cgroup/affinity restrictions (e.g.
running inside a container with a CPU quota) and needs no extra
dependency (``psutil`` was considered and deliberately not added). If this
turns out to oversubscribe hyperthreaded machines in practice, that's a
question for the M11.3 performance pass, not this module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@runtime_checkable
class ParallelConfigLike(Protocol):
    """Structural type for the ``pcfg`` argument (duck typing, no import of ``openmeq.config``)."""

    n_workers: int | str
    threads_per_worker: int


@dataclass
class ParallelSettings:
    """Minimal concrete ``pcfg``-shaped object.

    ``openmeq.config`` (M0.3) will eventually define the real
    ``ParallelConfig`` used as ``cfg.parallel``; this dataclass exists so
    this module -- and its tests -- do not need to import that module.
    Any object with ``n_workers``/``threads_per_worker`` attributes works
    equally well.
    """

    n_workers: int | str = "auto"
    threads_per_worker: int = 1


def _set_thread_env(threads_per_worker: int) -> None:
    """Set the BLAS/OpenMP thread-count env vars in the *current* process."""
    value = str(int(threads_per_worker))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = value


def _worker_init(threads_per_worker: int) -> None:
    """``ProcessPoolExecutor`` initializer: cap BLAS/OpenMP (and TF) threads.

    Runs once per worker process, before that worker unpickles and executes
    any submitted call -- see the module docstring for why that ordering,
    combined with the ``spawn`` start method, is what makes this reliable.
    """
    _set_thread_env(threads_per_worker)
    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(threads_per_worker)
    except ImportError:
        pass  # TensorFlow is optional; only PhaseNet-running steps need it.
    except Exception:
        # TF's threading knobs can only be set once, before any TF op runs;
        # if something else in this fresh worker already touched TF (it
        # shouldn't have, but be defensive), don't crash the worker over it.
        pass


def _resolve_n_workers(n_workers: int | str) -> int:
    """Resolve ``pcfg.n_workers`` ("auto" or an int) to a concrete worker count."""
    if isinstance(n_workers, str):
        if n_workers != "auto":
            raise ValueError(f"n_workers must be an int or 'auto', got {n_workers!r}")
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is Linux-only; fall back elsewhere.
            return os.cpu_count() or 1
    n = int(n_workers)
    if n < 1:
        raise ValueError(f"n_workers must be >= 1, got {n}")
    return n


def _spawn_context():
    import multiprocessing as mp

    return mp.get_context("spawn")


def pmap(
    func: Callable[[T], R],
    items: Iterable[T],
    pcfg: ParallelConfigLike,
    desc: str = "",
    chunksize: int = 1,
    ordered: bool = True,
) -> list[R]:
    """Parallel map of ``func`` over ``items``, sized and thread-capped by ``pcfg``.

    Parameters
    ----------
    func : callable
        A top-level, picklable function (see module docstring) taking one
        item and returning one result.
    items : iterable
        The items to process. Consumed into a list internally (progress
        reporting and ``ordered=True`` both need the length up front).
    pcfg : ParallelConfigLike
        Anything with ``n_workers`` (``int`` or ``"auto"``) and
        ``threads_per_worker`` (``int``) attributes.
    desc : str
        tqdm progress bar description.
    chunksize : int
        Passed to ``ProcessPoolExecutor.map`` when ``ordered=True`` (a
        chunksize > 1 amortizes IPC overhead for many small, fast items).
        Ignored when ``ordered=False`` (each item is submitted
        individually so results can be drained via ``as_completed``).
    ordered : bool
        If True (default), the returned list is in the same order as
        ``items`` (like ``executor.map``). If False, results are returned
        in arrival/completion order (like ``as_completed``), which can be
        faster when items take very different amounts of time and callers
        don't care about order.

    Returns
    -------
    list
        One result per item. A worker exception propagates (raised from
        this call) rather than being swallowed or silently producing a
        short list.
    """
    items = list(items)
    n_workers = _resolve_n_workers(pcfg.n_workers)
    _set_thread_env(pcfg.threads_per_worker)  # also cap this (parent) process's own threads

    if n_workers == 1:
        return [func(item) for item in tqdm(items, total=len(items), desc=desc, disable=None)]

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=_spawn_context(),
        initializer=_worker_init,
        initargs=(pcfg.threads_per_worker,),
    ) as executor:
        if ordered:
            results_iter = executor.map(func, items, chunksize=chunksize)
            return list(tqdm(results_iter, total=len(items), desc=desc, disable=None))
        else:
            futures = [executor.submit(func, item) for item in items]
            results: list[R] = []
            for future in tqdm(as_completed(futures), total=len(items), desc=desc, disable=None):
                results.append(future.result())
            return results


def _read_done(out_path: Path) -> dict[str, Any]:
    """Read an existing JSON-lines results file into ``{key: result}``.

    Tolerates a truncated/partial final line (e.g. the process was killed
    mid-write): that line is skipped rather than raising, so a resumed run
    can always read a file left by a previous, interrupted run.
    """
    done: dict[str, Any] = {}
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Partial last line from a killed process -- tolerate it;
                # it will simply be recomputed and rewritten on this run.
                continue
            done[record["key"]] = record["result"]
    return done


def pmap_incremental(
    func: Callable[[T], R],
    items: Iterable[T],
    pcfg: ParallelConfigLike,
    out_path: Path,
    *,
    key: Callable[[T], str],
    desc: str = "",
) -> list[R]:
    """Resumable parallel map: appends JSON-lines results to ``out_path`` as they finish.

    ``key(item)`` identifies each item. On entry, ``out_path`` (if it
    exists) is read as JSON-lines, and any item whose key is already
    present is skipped -- only the remaining items are (re)computed. Each
    result is appended to ``out_path`` as one JSON line, flushed to disk,
    *as soon as that item completes* (not batched at the end), so a killed
    process leaves a valid, resumable partial file.

    ``func``'s return value must be JSON-serializable (it is written as
    ``{"key": ..., "result": ...}`` per line). Combine with a wrapper
    function if ``func`` naturally returns something richer.

    This is the primary design target for e.g. M10.3 (cross-correlation to
    ``dt.cc``): tens of thousands of items, a long-running job, safe to
    Ctrl-C and resume.

    Parameters
    ----------
    func, pcfg, desc : see :func:`pmap`.
    items : iterable
        The full item list (including any already-done ones -- they are
        filtered out here).
    out_path : Path
        JSON-lines results file. Opened in append mode; created (with
        parent directories) if it doesn't exist.
    key : callable, keyword-only
        ``item -> str``, a stable identifier for resumability.

    Returns
    -------
    list
        One result per item in ``out_path``, at exactly this call: results already on disk
        come from the file; newly-computed ones come from this run.
    """
    items = list(items)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _read_done(out_path)
    remaining = [item for item in items if key(item) not in done]

    # If a previous run was killed mid-line, the file may not end with a
    # newline; append one now so our appends start on a fresh line instead
    # of silently concatenating onto -- and further corrupting -- that
    # already-unreadable partial line.
    if out_path.exists():
        existing = out_path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n")

    n_workers = _resolve_n_workers(pcfg.n_workers)
    _set_thread_env(pcfg.threads_per_worker)

    progress = tqdm(total=len(items), initial=len(items) - len(remaining), desc=desc, disable=None)
    try:
        with open(out_path, "a", encoding="utf-8") as f:

            def _record(item: T, result: R) -> None:
                item_key = key(item)
                done[item_key] = result
                f.write(json.dumps({"key": item_key, "result": result}) + "\n")
                f.flush()
                progress.update(1)

            if not remaining:
                pass
            elif n_workers == 1:
                for item in remaining:
                    _record(item, func(item))
            else:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=_spawn_context(),
                    initializer=_worker_init,
                    initargs=(pcfg.threads_per_worker,),
                ) as executor:
                    futures = {executor.submit(func, item): item for item in remaining}
                    for future in as_completed(futures):
                        item = futures[future]
                        _record(item, future.result())
    finally:
        progress.close()

    return [done[key(item)] for item in items]
