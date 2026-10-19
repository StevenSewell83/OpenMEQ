"""Tests for openmeq.paths.RunPaths (M0.4).

See docs/ARCHITECTURE.md §3 for the run-directory layout this module is a
literal, cross-checked contract for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openmeq.config import Config
from openmeq.paths import STEP_DIRS, RunPaths


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
    "setup_dir": "0_setup",
    "index_json": "0_setup/index.json",
    "stations_xml": "0_setup/stations.xml",
    "stations_csv": "0_setup/stations.csv",
    "metadata_report_json": "0_setup/metadata_report.json",
    "config_snapshot_yaml": "config_snapshot.yaml",
    "availability_csv": "1_qc/availability.csv",
    "daily_stats_csv": "1_qc/daily_stats.csv",
    "monthly_stats_csv": "1_qc/monthly_stats.csv",
    "channel_flags_json": "1_qc/channel_flags.json",
    "qc_diagnostics_dir": "1_qc/diagnostics",
    "noise_daily_csv": "1_qc/noise_daily.csv",
    "qc_noise_diagnostics_dir": "1_qc/diagnostics/noise",
    "qc_ppsd_dir": "1_qc/ppsd",
    "qc_ppsd_diagnostics_dir": "1_qc/diagnostics/ppsd",
    "detections_csv": "2_detection/detections.csv",
    "detect_diagnostics_dir": "2_detection/diagnostics",
    "chunk_dir": "4_data_chunks/chunks",
    "chunk_completeness_csv": "4_data_chunks/chunks/completeness.csv",
    "chunks_pn_dir": "4_data_chunks/chunks_pn",
    "data_list_csv": "4_data_chunks/chunks_pn/data_list.csv",
    "phasenet_raw_dir": "5_ml_picks/phasenet_raw",
    "catalog_picks_xml": "5_ml_picks/catalog_picks.xml",
    "nlloc_dir": "7_location/nlloc",
    "passA_dir": "7_location/passA",
    "passB_dir": "7_location/passB",
    "catalog_auto_xml": "7_location/catalog_auto.xml",
    "locate_summary_csv": "7_location/summary.csv",
    "removal_log_csv": "7_location/removal_log.csv",
    "review_state_json": "7b_review/state.json",
    "review_events_dir": "7b_review/events",
    "catalog_final_xml": "7b_review/catalog_final.xml",
    "catalog_mag_xml": "8_relocation/magnitude/catalog_mag.xml",
    "station_magnitudes_csv": "8_relocation/magnitude/station_magnitudes.csv",
    "relocate_work_dir": "8_relocation/work",
    "relocate_cache_dir": "8_relocation/cache",
    "dt_ct": "8_relocation/work/dt.ct",
    "dt_cc": "8_relocation/work/dt.cc",
    "catalog_dd_xml": "8_relocation/catalog_dd.xml",
    "run_diagnostics_dir": "9_plots",
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


def test_locate_pass_dir(paths: RunPaths) -> None:
    assert paths.locate_pass_dir("passA", "ev1") == paths.passA_dir / "ev1"
    assert paths.locate_pass_dir("passB", "ev1") == paths.passB_dir / "ev1"


def test_locate_pass_dir_rejects_unknown_pass_name(paths: RunPaths) -> None:
    with pytest.raises(ValueError, match="passA.*passB"):
        paths.locate_pass_dir("passC", "ev1")


@pytest.mark.parametrize(
    "step,folder",
    [
        ("qc", "1_qc"),
        ("detections", "2_detection"),
        ("chunks", "4_data_chunks/chunks"),
        ("chunks_pn", "4_data_chunks/chunks_pn"),
        ("pick", "5_ml_picks"),
        ("locate", "7_location"),
        ("nlloc", "7_location/nlloc"),
        ("review", "7b_review"),
        ("magnitude", "8_relocation/magnitude"),
        ("relocate", "8_relocation"),
    ],
)
def test_diagnostics_dir_and_manifest_json(paths: RunPaths, step: str, folder: str) -> None:
    assert paths.diagnostics_dir(step) == paths.root / folder / "diagnostics"
    assert paths.manifest_json(step) == paths.root / folder / "manifest.json"


def test_no_two_steps_share_a_manifest(paths: RunPaths) -> None:
    manifests = [paths.manifest_json(step) for step in STEP_DIRS]
    assert len(set(manifests)) == len(manifests)


def test_unknown_step_key_falls_back_to_its_own_name(paths: RunPaths) -> None:
    assert paths.diagnostics_dir("adhoc") == paths.root / "adhoc" / "diagnostics"


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
        paths.setup_dir,
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
