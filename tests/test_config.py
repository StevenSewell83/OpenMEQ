"""Tests for openmeq.config (M0.3): schema, loader, and config/example.yaml.

See docs/ARCHITECTURE.md §2 for the load_config(path) -> Config contract and
config/example.yaml as the documented reference schema.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openmeq.config import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"


def _load_example_raw() -> dict:
    with open(EXAMPLE_CONFIG) as f:
        return yaml.safe_load(f)


def test_load_example_config():
    cfg = load_config(EXAMPLE_CONFIG)

    # One attribute per top-level YAML section, per ARCHITECTURE.md §2.
    for section in (
        "project",
        "parallel",
        "data",
        "stations",
        "qc",
        "detection",
        "chunks",
        "phasenet",
        "nlloc",
        "review",
        "magnitude",
        "hypodd",
    ):
        assert hasattr(cfg, section)

    # Spot-check nested values, matching PLAN.md §3's literal sketch values.
    assert cfg.nlloc.filter.p_only.max_iter == 5
    assert cfg.nlloc.filter.p_only.max_residual_s == pytest.approx(0.10)
    assert cfg.hypodd.xcorr.min_cc == pytest.approx(0.7)
    assert cfg.hypodd.ph2dt.maxngh == 10
    assert cfg.detection.coincidence_sum == 4
    assert cfg.detection.trigger_on == pytest.approx(3.5)
    assert cfg.magnitude.ml.hutton_boore.a == pytest.approx(1.110)
    assert cfg.magnitude.statistic == "median"
    assert cfg.phasenet.target_rate_hz == pytest.approx(100)

    # Keys added ahead of later milestones (see report / commit message).
    assert cfg.qc.flat_run_min_samples == 50
    assert cfg.qc.clip_run_min_samples == 3
    # M1.9 / OpenMEQ-5nl.9 retuned max_spike_fraction (and added
    # spike_median_window/max_flat_run_s) for the isolated-impulse
    # spike_count definition -- see docs/performance.md for the measured
    # synthetic_project numbers these defaults are chosen against.
    assert cfg.qc.spike_threshold_mad == pytest.approx(10.0)
    assert cfg.qc.spike_median_window == 7
    assert cfg.qc.max_spike_fraction == pytest.approx(0.000012)
    assert cfg.qc.max_flat_fraction == pytest.approx(0.5)
    assert cfg.qc.max_flat_run_s == pytest.approx(60.0)
    assert cfg.qc.max_clip_count == 0
    assert cfg.qc.monthly_outlier_ratio == pytest.approx(3.0)
    assert cfg.detection.day_overlap_s == pytest.approx(60.0)
    assert cfg.detection.dedup_window_s == pytest.approx(2.0)
    assert cfg.phasenet.max_gap_fill_s == pytest.approx(2.0)
    assert cfg.phasenet.pick_uncertainty_s == pytest.approx(0.1)
    assert cfg.nlloc.obspyck_pluginpath.is_absolute()
    assert cfg.nlloc.locsearch.init_num_cells_x == 10
    assert cfg.nlloc.locgau.sigma_time_s == pytest.approx(0.1)
    assert cfg.nlloc.locgau2.sigma_time_min_s == pytest.approx(0.05)
    assert cfg.nlloc.locgau2.sigma_time_max_s == pytest.approx(2.0)
    assert cfg.nlloc.locqual2err_thresholds_s == [0.05, 0.1, 0.2, 0.5, 99999.9]
    assert cfg.magnitude.ml.wa_sensitivity == pytest.approx(2080)
    assert cfg.magnitude.ml.window_before_s == pytest.approx(1.0)
    assert cfg.magnitude.ml.window_after_s == pytest.approx(4.0)
    assert cfg.magnitude.ml.pre_filt == (0.5, 1.0, 45.0, 49.0)
    assert cfg.hypodd.ph2dt.minwght == pytest.approx(0.0)
    assert cfg.hypodd.ph2dt.maxdist_km == pytest.approx(200.0)
    assert cfg.hypodd.ph2dt.minobs == 8
    assert cfg.hypodd.ph2dt.maxobs == 20
    assert len(cfg.hypodd.weighting_rows) == 5
    assert cfg.hypodd.istart == 2
    assert cfg.hypodd.isolv == 2
    assert cfg.hypodd.iaq == 1
    assert cfg.hypodd.parallel_clusters is False
    assert cfg.hypodd.velocity_model.origin_3d == pytest.approx((37.878, -122.244))
    assert cfg.hypodd.velocity_model.simul2000.nitpb == 50
    assert cfg.hypodd.xcorr.filter.freqmin == pytest.approx(2.0)
    assert cfg.hypodd.parallel_clusters is False


def test_example_config_documents_every_plan_sketch_key():
    """Every top-level and nested key from PLAN.md §3's original YAML sketch
    must appear (and be validated) in config/example.yaml."""
    raw = _load_example_raw()
    assert set(raw.keys()) == set(Config.model_fields.keys())

    # A handful of representative nested keys from the original PLAN.md §3
    # sketch (not the keys added ahead of later milestones).
    assert "channel_map" in raw["stations"]
    assert "spike_threshold_mad" in raw["qc"]
    assert "coincidence_sum" in raw["detection"]
    assert "min_p_prob" in raw["phasenet"]
    assert "grid" in raw["nlloc"]
    assert "events" in raw["nlloc"]["filter"]
    assert "hutton_boore" in raw["magnitude"]["ml"]
    assert "xcorr" in raw["hypodd"]
    assert "ph2dt" in raw["hypodd"]


def test_relative_paths_resolve_against_config_dir(tmp_path):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    (config_dir / "stations.csv").write_text("net,sta,loc,lat,lon,elev_m,depth_m\n")
    (config_dir / "vel1d.txt").write_text("0.0 3.5 2.0 2.3\n")

    config_text = """
project:
  name: test
  output_dir: ./runs/out
data:
  waveform_dirs: [./mseed]
  start: 2025-01-01
  end: 2025-01-02
stations:
  csv: ./stations.csv
nlloc:
  velocity_model:
    type: 1d
    file: ./vel1d.txt
  grid:
    origin: [0.0, 0.0]
    size_km: [10, 10, 10]
    spacing_km: 0.1
hypodd:
  velocity_model:
    type: 1d
    file: ./vel1d.txt
"""
    config_path = config_dir / "config.yaml"
    config_path.write_text(config_text)

    cfg = load_config(config_path)

    assert cfg.stations.csv == (config_dir / "stations.csv").resolve()
    assert cfg.stations.csv.is_absolute()
    assert str(cfg.stations.csv).startswith(str(config_dir))
    assert cfg.project.output_dir == (config_dir / "runs" / "out").resolve()
    assert cfg.data.waveform_dirs[0] == (config_dir / "mseed").resolve()
    assert cfg.nlloc.velocity_model.file == (config_dir / "vel1d.txt").resolve()
    assert cfg.hypodd.velocity_model.file == (config_dir / "vel1d.txt").resolve()


def test_absolute_paths_left_unchanged(tmp_path):
    config_dir = tmp_path / "project2"
    config_dir.mkdir()
    abs_stations = tmp_path / "elsewhere" / "stations.csv"
    abs_stations.parent.mkdir()
    abs_stations.write_text("net,sta,loc,lat,lon,elev_m,depth_m\n")

    config_text = f"""
project:
  name: test
  output_dir: ./runs/out
data:
  waveform_dirs: [./mseed]
  start: 2025-01-01
  end: 2025-01-02
stations:
  csv: {abs_stations}
nlloc:
  velocity_model:
    type: 1d
    file: ./vel1d.txt
  grid:
    origin: [0.0, 0.0]
    size_km: [10, 10, 10]
    spacing_km: 0.1
hypodd:
  velocity_model:
    type: 1d
    file: ./vel1d.txt
"""
    config_path = config_dir / "config.yaml"
    config_path.write_text(config_text)

    cfg = load_config(config_path)
    assert cfg.stations.csv == abs_stations


def test_unknown_top_level_key_rejected(tmp_path):
    raw = _load_example_raw()
    raw["not_a_real_section"] = {}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_nested_key_rejected(tmp_path):
    raw = _load_example_raw()
    raw["qc"]["not_a_real_key"] = 123
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_required_key_rejected(tmp_path):
    raw = _load_example_raw()
    del raw["project"]["output_dir"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_required_section_rejected(tmp_path):
    raw = _load_example_raw()
    del raw["nlloc"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError):
        load_config(path)


def test_literal_fields_reject_bad_values(tmp_path):
    raw = _load_example_raw()
    raw["nlloc"]["velocity_model"]["type"] = "2d"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path)

    raw = _load_example_raw()
    raw["magnitude"]["ml"]["relation"] = "bogus"
    path2 = tmp_path / "config2.yaml"
    path2.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path2)

    raw = _load_example_raw()
    raw["hypodd"]["velocity_model"]["type"] = "2d"
    path3 = tmp_path / "config3.yaml"
    path3.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path3)

    raw = _load_example_raw()
    raw["parallel"]["phasenet_device"] = "tpu"
    path4 = tmp_path / "config4.yaml"
    path4.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path4)


def test_malformed_yaml_raises_config_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("project: [this is not: valid yaml structure")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_local_ml_coefficients_default_to_none():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.magnitude.ml.local.coefficients.a is None
    assert cfg.magnitude.ml.local.coefficients.b is None
    assert cfg.magnitude.ml.local.coefficients.c is None


def test_n_workers_auto_stored_as_literal_string():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.parallel.n_workers == "auto"


def test_n_workers_accepts_integer(tmp_path):
    raw = _load_example_raw()
    raw["parallel"]["n_workers"] = 8
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    assert cfg.parallel.n_workers == 8


def test_config_importable_without_obspy():
    """openmeq.config must not import obspy at module import time (module
    docstring's stated design decision) -- verified by checking obspy is not
    a required attribute/import of the module itself."""
    import sys

    assert "openmeq.config" in sys.modules
    # obspy may or may not be imported elsewhere in the test session by this
    # point; the real guarantee is structural: config.py's own source has no
    # "import obspy" statement.
    import inspect

    import openmeq.config as config_module

    source = inspect.getsource(config_module)
    assert "import obspy" not in source


def test_constructing_config_directly_without_context_leaves_paths_relative():
    """Constructing a Config directly (not via load_config) with no context
    leaves relative paths untouched, per the module docstring."""
    raw = _load_example_raw()
    cfg = Config.model_validate(raw)
    assert not cfg.project.output_dir.is_absolute()
    assert str(cfg.project.output_dir) == "runs/my_survey"


def test_pydantic_validation_error_is_the_cause(tmp_path):
    raw = _load_example_raw()
    raw["qc"]["spike_threshold_mad"] = "not-a-number"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert isinstance(excinfo.value.__cause__, ValidationError)
