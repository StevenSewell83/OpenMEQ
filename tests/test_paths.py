"""Tests for openmeq.paths.RunPaths (M0.4).

See docs/ARCHITECTURE.md §3 for the run-directory layout this module is a
literal, cross-checked contract for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openmeq.config import Config
from openmeq.paths import RunPaths


def _make_config(output_dir: Path) -> Config:
    """A minimal, schema-valid Config pointed at ``output_dir``.

    Constructed directly (not via ``load_config``/YAML) since only
    ``cfg.project.output_dir`` matters to ``RunPaths``.
    """
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


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return RunPaths(_make_config(tmp_path / "run"))


# ARCHITECTURE.md §3, literal file/dir properties (name -> relative path).
_EXPECTED_PROPERTIES = {
    "index_json": "inventory/index.json",
    "stations_xml": "inventory/stations.xml",
    "stations_csv": "inventory/stations.csv",
    "metadata_report_json": "inventory/metadata_report.json",
    "availability_csv": "qc/availability.csv",
    "daily_stats_csv": "qc/daily_stats.csv",
    "monthly_stats_csv": "qc/monthly_stats.csv",
    "channel_flags_json": "qc/channel_flags.json",
    "qc_diagnostics_dir": "qc/diagnostics",
    "detections_csv": "detections/detections.csv",
    "detect_diagnostics_dir": "detections/diagnostics",
    "chunk_dir": "chunks",
    "chunk_completeness_csv": "chunks/completeness.csv",
    "chunks_pn_dir": "chunks_pn",
    "data_list_csv": "chunks_pn/data_list.csv",
    "phasenet_raw_dir": "pick/phasenet_raw",
    "catalog_picks_xml": "pick/catalog_picks.xml",
    "passA_dir": "locate/passA",
    "passB_dir": "locate/passB",
    "catalog_auto_xml": "locate/catalog_auto.xml",
    "locate_summary_csv": "locate/summary.csv",
    "removal_log_csv": "locate/removal_log.csv",
    "review_state_json": "review/state.json",
    "review_events_dir": "review/events",
    "catalog_final_xml": "review/catalog_final.xml",
    "catalog_mag_xml": "magnitude/catalog_mag.xml",
    "station_magnitudes_csv": "magnitude/station_magnitudes.csv",
    "relocate_work_dir": "relocate/work",
    "relocate_cache_dir": "relocate/cache",
    "dt_ct": "relocate/dt.ct",
    "dt_cc": "relocate/dt.cc",
    "catalog_dd_xml": "relocate/catalog_dd.xml",
}


def test_root_is_output_dir(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "run")
    p = RunPaths(cfg)
    assert p.root == tmp_path / "run"


@pytest.mark.parametrize("name,relative", sorted(_EXPECTED_PROPERTIES.items()))
def test_every_documented_property_resolves(paths: RunPaths, name: str, relative: str) -> None:
    value = getattr(paths, name)
    assert isinstance(value, Path)
    assert value == paths.root / Path(relative)


def test_no_filesystem_side_effects_before_ensure(paths: RunPaths) -> None:
    """Constructing RunPaths and reading properties must not touch disk."""
    for name in _EXPECTED_PROPERTIES:
        getattr(paths, name)
    assert not paths.root.exists()


def test_chunk_file(paths: RunPaths) -> None:
    assert paths.chunk_file("ev20250101T000000_000") == (
        paths.chunk_dir / "ev20250101T000000_000" / "ev20250101T000000_000.mseed"
    )


def test_chunk_pn_file(paths: RunPaths) -> None:
    assert paths.chunk_pn_file("ev20250101T000000_000") == (
        paths.chunks_pn_dir / "ev20250101T000000_000" / "ev20250101T000000_000.mseed"
    )


def test_review_event_file(paths: RunPaths) -> None:
    assert paths.review_event_file("ev1") == paths.review_events_dir / "ev1.xml"


def test_relocate_cache_file(paths: RunPaths) -> None:
    assert paths.relocate_cache_file("ev1") == paths.relocate_cache_dir / "ev1.npz"


@pytest.mark.parametrize(
    "step",
    [
        "qc",
        "detections",
        "chunks",
        "chunks_pn",
        "pick",
        "locate",
        "review",
        "magnitude",
        "relocate",
    ],
)
def test_diagnostics_dir_and_manifest_json(paths: RunPaths, step: str) -> None:
    assert paths.diagnostics_dir(step) == paths.root / step / "diagnostics"
    assert paths.manifest_json(step) == paths.root / step / "manifest.json"


@pytest.mark.parametrize(
    "step",
    ["qc", "detect", "chunk", "pick", "locate", "review", "magnitude", "relocate", "run"],
)
def test_log_file(paths: RunPaths, step: str) -> None:
    assert paths.log_file(step) == paths.root / "logs" / f"{step}.log"


def test_ensure_creates_every_directory(paths: RunPaths) -> None:
    paths.ensure()

    expected_dirs = [
        paths.root,
        paths.root / "inventory",
        paths.qc_diagnostics_dir,
        paths.detect_diagnostics_dir,
        paths.chunk_dir,
        paths.diagnostics_dir("chunks"),
        paths.chunks_pn_dir,
        paths.diagnostics_dir("chunks_pn"),
        paths.phasenet_raw_dir,
        paths.diagnostics_dir("pick"),
        paths.passA_dir,
        paths.passB_dir,
        paths.diagnostics_dir("locate"),
        paths.review_events_dir,
        paths.diagnostics_dir("review"),
        paths.catalog_mag_xml.parent,
        paths.diagnostics_dir("magnitude"),
        paths.relocate_work_dir,
        paths.relocate_cache_dir,
        paths.diagnostics_dir("relocate"),
        paths.root / "logs",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"expected directory {d} to exist after ensure()"

    # ensure() only makes directories, never files.
    for name, relative in _EXPECTED_PROPERTIES.items():
        if not relative.endswith((".json", ".csv", ".xml", ".ct", ".cc")):
            continue
        assert not (paths.root / relative).exists(), f"{name} should not exist as a file"


def test_ensure_is_idempotent(paths: RunPaths) -> None:
    paths.ensure()
    paths.ensure()  # must not raise
    assert paths.root.is_dir()
