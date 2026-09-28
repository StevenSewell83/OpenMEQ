"""Tests for openmeq.magnitude.relations (M8.2): Hutton & Boore, local
coefficients, local distance table, station corrections.

See docs/ARCHITECTURE.md §6b for the local calibration file format contract.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from openmeq.config import Config, ConfigError, load_config
from openmeq.magnitude.relations import (
    clear_caches,
    get_clamp_counter,
    interp_neg_log_a0,
    load_neg_log_a0_table,
    load_station_corrections,
    reset_clamp_counter,
    station_ml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example.yaml"


def _load_example_raw() -> dict:
    with open(EXAMPLE_CONFIG) as f:
        return yaml.safe_load(f)


def _base_config(tmp_path: Path, **magnitude_overrides) -> Config:
    """Build a minimal valid Config (from config/example.yaml) with
    magnitude.ml overridden for a given test, resolved against tmp_path so
    any relative file references work."""
    raw = _load_example_raw()
    raw["magnitude"]["ml"].update(magnitude_overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """The table/corrections caches and clamp counter are module-level
    (per-process); reset them around every test so tests don't leak state
    into each other."""
    clear_caches()
    reset_clamp_counter()
    yield
    clear_caches()
    reset_clamp_counter()


# ---------------------------------------------------------------------------
# Hutton & Boore reference values
# ---------------------------------------------------------------------------


def test_hutton_boore_reference_values(tmp_path):
    """Hand-computable reference values (design section), asserted to 4
    decimal places. A_mm=1.0, R=100 is Richter's original ML=3 anchor point.
    """
    cfg = _base_config(tmp_path, relation="hutton_boore")

    # A_mm=1.0, R=100: log10(1) + 1.110*log10(1) + 0.00189*0 + 3.0 = 3.000
    ml = station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(3.0000, abs=1e-4)

    # A_mm=10.0, R=100: log10(10) + 0 + 0 + 3.0 = 4.000
    ml = station_ml(amplitude_m=10.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(4.0000, abs=1e-4)

    # A_mm=1.0, R=200: 1.110*log10(2) + 0.00189*100 + 3.0 = 3.523143
    ml = station_ml(amplitude_m=1.0e-3, distance_km=200.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(3.523143, abs=1e-4)


def test_hutton_boore_matches_original_richter_anchored_form(tmp_path):
    """Verify the "/100" form used here:

        ML = log10(A_mm) + a*log10(R/100) + b*(R-100) + c

    numerically against the algebraically-equivalent expanded form:

        ML = log10(A_mm) + a*log10(R) + b*R + d

    Expanding ``a*log10(R/100)`` as ``a*log10(R) - a*log10(100)`` and
    ``b*(R-100)`` as ``b*R - 100*b`` shows the two forms are identical for
    every R exactly when ``d = c - a*log10(100) - 100*b = c - 2*a - 100*b``
    (since ``log10(100) = 2``). With Hutton & Boore's own coefficients
    (a=1.110, b=0.00189, c=3.0) that is ``d = 3.0 - 2.22 - 0.189 = 0.591``.

    NOTE (deviation from the issue's suggested value): the issue text asks
    to check this against a constant of "-2.09" rather than "+0.591". That
    value does *not* reproduce this module's c=3.0 anchor (it disagrees by
    2.681 magnitude units, e.g. it gives ML=0.319 rather than 3.000 at
    A=1mm/R=100km) and is presumably a mixup with a different, unrelated
    magnitude formula. This test instead uses the constant that is actually
    algebraically forced by c=3.0 (0.591), so the two forms provably agree;
    see this task's final report for the same note.
    """
    cfg = _base_config(tmp_path, relation="hutton_boore")
    a, b, c = 1.110, 0.00189, 3.0
    d = c - 2 * a - 100 * b
    assert d == pytest.approx(0.591, abs=1e-9)

    for a_mm, r_km in [(0.5, 15.0), (2.3, 87.0), (0.02, 250.0), (5.0, 100.0)]:
        got = station_ml(amplitude_m=a_mm / 1000.0, distance_km=r_km, cfg=cfg, net="XX", sta="STA1")
        expanded = math.log10(a_mm) + a * math.log10(r_km) + b * r_km + d
        assert got == pytest.approx(expanded, abs=1e-9)


# ---------------------------------------------------------------------------
# local coefficients (same 3-term formula)
# ---------------------------------------------------------------------------


def test_local_coefficients_same_formula(tmp_path):
    """A local relation with coefficients set equal to Hutton & Boore's
    defaults reproduces the same 3 reference values."""
    cfg = _base_config(
        tmp_path,
        relation="local",
        local={"coefficients": {"a": 1.110, "b": 0.00189, "c": 3.0}, "file": None},
    )

    ml = station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(3.0000, abs=1e-4)

    ml = station_ml(amplitude_m=10.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(4.0000, abs=1e-4)

    ml = station_ml(amplitude_m=1.0e-3, distance_km=200.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(3.523143, abs=1e-4)


# ---------------------------------------------------------------------------
# local distance table: loading, interpolation, clamping
# ---------------------------------------------------------------------------


def _write_table(path: Path, rows: list[tuple]) -> None:
    lines = ["distance_km,neg_log_a0"]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    path.write_text("\n".join(lines) + "\n")


def test_local_table_interpolation(tmp_path):
    table_path = tmp_path / "local_table.csv"
    _write_table(
        table_path,
        [(0, 1.4), (10, 1.6), (50, 2.2), (100, 3.0), (200, 3.4)],
    )
    table = load_neg_log_a0_table(table_path)

    # exact hand-computed midpoint: distance 30 is halfway between 10 (1.6)
    # and 50 (2.2) -> neg_log_a0 = 1.6 + 0.5*(2.2-1.6) = 1.9
    val = interp_neg_log_a0(30.0, table)
    assert val == pytest.approx(1.9, abs=1e-9)
    assert get_clamp_counter() == 0

    # exact table row values are reproduced
    assert interp_neg_log_a0(0.0, table) == pytest.approx(1.4, abs=1e-9)
    assert interp_neg_log_a0(200.0, table) == pytest.approx(3.4, abs=1e-9)

    # below range: clamped to first row's value, and the clamp counter increments
    val = interp_neg_log_a0(-50.0, table)
    assert val == pytest.approx(1.4, abs=1e-9)
    assert get_clamp_counter() == 1

    # above range: clamped to last row's value
    val = interp_neg_log_a0(500.0, table)
    assert val == pytest.approx(3.4, abs=1e-9)
    assert get_clamp_counter() == 2


def test_local_table_end_to_end_via_station_ml(tmp_path):
    table_path = tmp_path / "local_table.csv"
    _write_table(table_path, [(0, 1.4), (10, 1.6), (50, 2.2), (100, 3.0), (200, 3.4)])
    cfg = _base_config(
        tmp_path,
        relation="local",
        local={"coefficients": {"a": None, "b": None, "c": None}, "file": str(table_path)},
    )

    # A_mm = 1.0 -> log10(1) = 0; at R=30, neg_log_a0 = 1.9 -> ML = 1.9
    ml = station_ml(amplitude_m=1.0e-3, distance_km=30.0, cfg=cfg, net="XX", sta="STA1")
    assert ml == pytest.approx(1.9, abs=1e-6)


def test_local_table_validation_errors(tmp_path):
    # too few rows
    p1 = tmp_path / "too_few.csv"
    _write_table(p1, [(0, 1.4)])
    with pytest.raises(ConfigError, match=str(p1)):
        load_neg_log_a0_table(p1)

    # non-increasing distances
    p2 = tmp_path / "non_increasing.csv"
    _write_table(p2, [(0, 1.4), (10, 1.6), (10, 2.0), (100, 3.0)])
    with pytest.raises(ConfigError) as excinfo:
        load_neg_log_a0_table(p2)
    assert str(p2) in str(excinfo.value)
    assert "row" in str(excinfo.value).lower()

    # NaN value
    p3 = tmp_path / "has_nan.csv"
    p3.write_text("distance_km,neg_log_a0\n0,1.4\n10,\n100,3.0\n")
    with pytest.raises(ConfigError) as excinfo:
        load_neg_log_a0_table(p3)
    assert str(p3) in str(excinfo.value)
    assert "row" in str(excinfo.value).lower()

    # missing file
    with pytest.raises(ConfigError):
        load_neg_log_a0_table(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# station corrections: inline mapping and CSV, both produce the same lookup
# ---------------------------------------------------------------------------


def test_station_corrections_dict_and_csv(tmp_path):
    cfg_dict = _base_config(
        tmp_path / "dict_cfg", station_corrections={"XX.STA1": 0.25, "XX.STA2": -0.1}
    )

    csv_path = tmp_path / "corrections.csv"
    csv_path.write_text("net,sta,correction\nXX,STA1,0.25\nXX,STA2,-0.1\n")
    cfg_csv = _base_config(tmp_path / "csv_cfg", station_corrections=str(csv_path))

    corrections_dict = load_station_corrections(cfg_dict)
    corrections_csv = load_station_corrections(cfg_csv)

    assert corrections_dict == {"XX.STA1": pytest.approx(0.25), "XX.STA2": pytest.approx(-0.1)}
    assert corrections_csv == {"XX.STA1": pytest.approx(0.25), "XX.STA2": pytest.approx(-0.1)}

    # missing station defaults to 0.0
    assert corrections_dict.get("YY.NOPE", 0.0) == 0.0

    # end-to-end through station_ml: correction is added
    ml_with_correction = station_ml(
        amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg_dict, net="XX", sta="STA1"
    )
    ml_without_correction = station_ml(
        amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg_dict, net="ZZ", sta="NOSTA"
    )
    assert ml_with_correction == pytest.approx(ml_without_correction + 0.25, abs=1e-9)


def test_station_correction_missing_station_defaults_to_zero(tmp_path):
    cfg = _base_config(tmp_path, station_corrections={"XX.STA1": 0.5})
    ml_known = station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")
    ml_unknown = station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA2")
    assert ml_known - ml_unknown == pytest.approx(0.5, abs=1e-9)
    assert ml_unknown == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# unknown / misconfigured relation
# ---------------------------------------------------------------------------


def test_unknown_relation_raises(tmp_path):
    raw = _load_example_raw()
    raw["magnitude"]["ml"]["relation"] = "hutton_boore"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)

    # Force an invalid relation value past pydantic's Literal validation
    # (a defensive check inside station_ml itself, in addition to the config
    # schema's Literal["hutton_boore", "local"]).
    cfg.magnitude.ml.relation = "bogus"

    with pytest.raises(ConfigError, match="bogus"):
        station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")


def test_config_schema_rejects_unknown_relation(tmp_path):
    """The config schema itself (Literal["hutton_boore", "local"]) already
    rejects an unknown relation at load time."""
    raw = _load_example_raw()
    raw["magnitude"]["ml"]["relation"] = "bogus"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path)


def test_local_relation_without_coefficients_or_file_raises(tmp_path):
    cfg = _base_config(
        tmp_path,
        relation="local",
        local={"coefficients": {"a": None, "b": None, "c": None}, "file": None},
    )
    with pytest.raises(ConfigError, match="local"):
        station_ml(amplitude_m=1.0e-3, distance_km=100.0, cfg=cfg, net="XX", sta="STA1")


# ---------------------------------------------------------------------------
# per-process caching
# ---------------------------------------------------------------------------


def test_local_table_is_cached_by_resolved_path(tmp_path):
    table_path = tmp_path / "local_table.csv"
    _write_table(table_path, [(0, 1.4), (10, 1.6), (50, 2.2), (100, 3.0), (200, 3.4)])
    cfg = _base_config(
        tmp_path,
        relation="local",
        local={"coefficients": {"a": None, "b": None, "c": None}, "file": str(table_path)},
    )

    station_ml(amplitude_m=1.0e-3, distance_km=30.0, cfg=cfg, net="XX", sta="STA1")

    # mutate the file on disk; the cached table should NOT reload within
    # the same process, proving the module-level cache is in effect.
    _write_table(table_path, [(0, 9.9), (10, 9.9), (50, 9.9), (100, 9.9), (200, 9.9)])
    ml_after = station_ml(amplitude_m=1.0e-3, distance_km=30.0, cfg=cfg, net="XX", sta="STA1")
    assert ml_after == pytest.approx(1.9, abs=1e-6)
