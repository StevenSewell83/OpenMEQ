"""Tests for openmeq.manifest: fingerprint, StepManifest (M0.4).

See docs/ARCHITECTURE.md §2 (``openmeq.manifest``) and PLAN.md §4's "skip
what hasn't changed" bullet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openmeq.config import Config, QcConfig
from openmeq.manifest import StepManifest, fingerprint
from openmeq.paths import RunPaths


def _make_config(output_dir: Path) -> Config:
    return Config.model_validate(
        {
            "project": {"name": "test", "output_dir": str(output_dir)},
            "data": {
                "waveform_dirs": [str(output_dir)],
                "start": "2025-01-01",
                "end": "2025-01-02",
            },
            "nlloc": {
                "velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")},
                "grid": {"origin": [0.0, 0.0], "size_km": [1.0, 1.0, 1.0], "spacing_km": 0.1},
            },
            "hypodd": {
                "velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")},
            },
        }
    )


# ---------------------------------------------------------------------------
# fingerprint()
# ---------------------------------------------------------------------------


def test_fingerprint_deterministic_for_scalars() -> None:
    assert fingerprint("a", 1, 2.5, True) == fingerprint("a", 1, 2.5, True)


def test_fingerprint_changes_with_scalar_input() -> None:
    assert fingerprint("a", 1) != fingerprint("a", 2)


def test_fingerprint_deterministic_for_pydantic_model() -> None:
    cfg = QcConfig(min_availability_pct=80.0)
    assert fingerprint(cfg) == fingerprint(cfg)


def test_fingerprint_changes_with_config_section() -> None:
    cfg_a = QcConfig(min_availability_pct=80.0)
    cfg_b = QcConfig(min_availability_pct=90.0)
    assert fingerprint(cfg_a) != fingerprint(cfg_b)


def test_fingerprint_changes_with_touched_file(tmp_path: Path) -> None:
    f = tmp_path / "input.mseed"
    f.write_bytes(b"original bytes")
    fp_before = fingerprint(f)

    # Touching mtime (and/or size) must change the fingerprint even if the
    # path string is identical.
    time.sleep(0.01)
    f.write_bytes(b"original bytes and more")
    fp_after = fingerprint(f)

    assert fp_before != fp_after


def test_fingerprint_stable_for_unchanged_file(tmp_path: Path) -> None:
    f = tmp_path / "input.mseed"
    f.write_bytes(b"same bytes")
    assert fingerprint(f) == fingerprint(f)


def test_fingerprint_missing_file_does_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mseed"
    fp = fingerprint(missing)
    assert isinstance(fp, str) and fp


def test_fingerprint_mixed_objects() -> None:
    cfg = QcConfig()
    fp1 = fingerprint(cfg, "seed_id", 42)
    fp2 = fingerprint(cfg, "seed_id", 42)
    fp3 = fingerprint(cfg, "seed_id", 43)
    assert fp1 == fp2
    assert fp1 != fp3


# ---------------------------------------------------------------------------
# StepManifest
# ---------------------------------------------------------------------------


def test_is_done_false_for_unknown_key(tmp_path: Path) -> None:
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest = StepManifest(paths, "qc")
    assert manifest.is_done("NET.STA..HHZ", "abc123") is False


def test_mark_done_then_is_done(tmp_path: Path) -> None:
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest = StepManifest(paths, "qc")

    manifest.mark_done("NET.STA..HHZ", "fp-1")

    assert manifest.is_done("NET.STA..HHZ", "fp-1") is True
    assert manifest.is_done("NET.STA..HHZ", "fp-2") is False
    assert manifest.is_done("NET.OTHER..HHZ", "fp-1") is False


def test_mark_done_persists_across_fresh_instance(tmp_path: Path) -> None:
    """Simulates a resumed run: a fresh StepManifest reads the same file."""
    paths = RunPaths(_make_config(tmp_path / "run"))

    first = StepManifest(paths, "qc")
    first.mark_done("key-a", "fp-a")
    first.mark_done("key-b", "fp-b")

    second = StepManifest(paths, "qc")
    assert second.is_done("key-a", "fp-a") is True
    assert second.is_done("key-b", "fp-b") is True
    assert second.is_done("key-c", "fp-c") is False


def test_manifest_json_written_at_documented_path(tmp_path: Path) -> None:
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest = StepManifest(paths, "qc")
    manifest.mark_done("key-a", "fp-a")

    manifest_path = paths.manifest_json("qc")
    assert manifest_path == paths.root / "1_qc" / "manifest.json"
    assert manifest_path.is_file()

    data = json.loads(manifest_path.read_text())
    assert data == {"step": "qc", "entries": {"key-a": "fp-a"}}


def test_manifest_updates_existing_key(tmp_path: Path) -> None:
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest = StepManifest(paths, "qc")
    manifest.mark_done("key-a", "fp-1")
    manifest.mark_done("key-a", "fp-2")

    assert manifest.is_done("key-a", "fp-1") is False
    assert manifest.is_done("key-a", "fp-2") is True

    second = StepManifest(paths, "qc")
    assert second.is_done("key-a", "fp-2") is True


def test_manifest_tolerates_corrupt_file(tmp_path: Path) -> None:
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest_path = paths.manifest_json("qc")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{not valid json")

    manifest = StepManifest(paths, "qc")
    assert manifest.is_done("anything", "fp") is False
    manifest.mark_done("anything", "fp")
    assert manifest.is_done("anything", "fp") is True


def test_skip_unchanged_workflow(tmp_path: Path) -> None:
    """End-to-end: unchanged input -> is_done True (skip); changed input -> False (redo)."""
    paths = RunPaths(_make_config(tmp_path / "run"))
    manifest = StepManifest(paths, "qc")

    input_file = tmp_path / "day1.mseed"
    input_file.write_bytes(b"day one data")
    qc_cfg = QcConfig()

    fp = fingerprint(qc_cfg, input_file)
    assert manifest.is_done("day1", fp) is False
    manifest.mark_done("day1", fp)
    assert manifest.is_done("day1", fp) is True

    # Input file changes -> fingerprint changes -> no longer considered done.
    time.sleep(0.01)
    input_file.write_bytes(b"day one data, revised")
    fp_after_change = fingerprint(qc_cfg, input_file)
    assert fp_after_change != fp
    assert manifest.is_done("day1", fp_after_change) is False

    # Config changes -> fingerprint changes too.
    fp_same_file_diff_cfg = fingerprint(QcConfig(min_availability_pct=50.0), input_file)
    assert fp_same_file_diff_cfg != fp_after_change
