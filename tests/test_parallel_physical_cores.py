"""Tests for ``openmeq.parallel``'s M11.3 physical-core ``"auto"`` default.

New file (per the M11.3 worker instructions: ``OpenMEQ-m1f.6`` is
concurrently consolidating ``tests/`` fixtures, so this is additive and does
not touch ``tests/conftest.py``). Covers :func:`openmeq.parallel._physical_core_count`
against synthetic ``/proc/cpuinfo``-format content (so it never depends on
the actual test machine's topology) and :func:`openmeq.parallel._resolve_n_workers`
end to end.
"""

from __future__ import annotations

from pathlib import Path

from openmeq import parallel as parallel_module
from openmeq.parallel import (
    ParallelSettings,
    _logical_core_count,
    _physical_core_count,
    _resolve_n_workers,
)

# A hybrid performance/efficiency-core layout (this project's own dev
# machine's real shape, per docs/performance.md): 8 hyperthreaded
# "performance" cores (2 logical CPUs each, sharing a core id) + 12
# single-threaded "efficiency" cores -- 28 logical, 20 physical. Deliberately
# includes the hybrid case, not just plain 2x hyperthreading, since a naive
# ``logical // 2`` would get this wrong (14, not 20).
_HYBRID_CPUINFO = "\n\n".join(
    f"processor\t: {p}\nphysical id\t: 0\ncore id\t\t: {core_id}\ncpu cores\t: 20\n"
    for p, core_id in (
        [(p, p // 2) for p in range(16)]  # 8 P-cores x 2 logical CPUs each
        + [(p, p - 8) for p in range(16, 28)]  # 12 E-cores x 1 logical CPU each
    )
)


def _write_cpuinfo(tmp_path: Path, content: str) -> str:
    path = tmp_path / "cpuinfo"
    path.write_text(content)
    return str(path)


def test_physical_core_count_hybrid_topology_not_naive_half_of_logical(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_module.os, "sched_getaffinity", lambda pid: set(range(28)))
    cpuinfo_path = _write_cpuinfo(tmp_path, _HYBRID_CPUINFO)
    assert _physical_core_count(cpuinfo_path) == 20  # not 28 // 2 == 14


def test_physical_core_count_respects_affinity_restriction(tmp_path, monkeypatch):
    # Restricted to the first 4 logical CPUs (2 performance-core siblings,
    # i.e. processors 0-3, which are 2 physical cores per _HYBRID_CPUINFO's
    # construction above) -- e.g. a cgroup CPU-quota restriction.
    monkeypatch.setattr(parallel_module.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    cpuinfo_path = _write_cpuinfo(tmp_path, _HYBRID_CPUINFO)
    assert _physical_core_count(cpuinfo_path) == 2


def test_physical_core_count_missing_file_returns_none(monkeypatch):
    monkeypatch.setattr(parallel_module.os, "sched_getaffinity", lambda pid: set(range(4)))
    assert _physical_core_count("/nonexistent/path/cpuinfo") is None


def test_physical_core_count_malformed_content_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_module.os, "sched_getaffinity", lambda pid: set(range(4)))
    cpuinfo_path = _write_cpuinfo(tmp_path, "not really cpuinfo content\nvendor_id: GenuineIntel\n")
    assert _physical_core_count(cpuinfo_path) is None


def test_resolve_n_workers_auto_falls_back_to_logical_when_physical_unknown(monkeypatch):
    monkeypatch.setattr(parallel_module, "_physical_core_count", lambda: None)
    monkeypatch.setattr(parallel_module, "_logical_core_count", lambda: 7)
    assert _resolve_n_workers("auto") == 7


def test_resolve_n_workers_auto_uses_physical_when_available(monkeypatch):
    monkeypatch.setattr(parallel_module, "_physical_core_count", lambda: 5)
    assert _resolve_n_workers("auto") == 5


def test_resolve_n_workers_explicit_int_unaffected_by_physical_detection(monkeypatch):
    monkeypatch.setattr(parallel_module, "_physical_core_count", lambda: 5)
    assert _resolve_n_workers(3) == 3


def test_logical_core_count_matches_sched_getaffinity(monkeypatch):
    monkeypatch.setattr(parallel_module.os, "sched_getaffinity", lambda pid: {0, 1, 2})
    assert _logical_core_count() == 3


def test_auto_resolves_to_a_positive_int_on_this_real_machine():
    # End-to-end sanity check against the real, unmocked machine (mirrors
    # tests/test_parallel.py::test_pmap_auto_resolves_to_positive_int, but
    # also checks the physical <= logical invariant that motivated M11.3's
    # change in the first place).
    pcfg = ParallelSettings(n_workers="auto", threads_per_worker=1)
    resolved = _resolve_n_workers(pcfg.n_workers)
    assert isinstance(resolved, int)
    assert resolved >= 1
    assert resolved <= _logical_core_count()
