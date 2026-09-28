"""Tests for openmeq.velocity.read_layers (ARCHITECTURE.md §6a format)."""

from __future__ import annotations

import pandas as pd
import pytest

from openmeq.velocity import read_layers

VALID_LAYERS = """\
# depth_top_km  vp_km_s  vs_km_s  [rho_g_cc]
0.0   3.50  2.00  2.3
1.0   4.50  2.60  2.5
3.0   5.80  3.35  2.7
"""


def test_read_layers_basic(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text(VALID_LAYERS)

    df = read_layers(path)

    assert list(df.columns) == ["depth_km", "vp", "vs", "rho"]
    assert len(df) == 3
    assert df["depth_km"].tolist() == [0.0, 1.0, 3.0]
    assert df["vp"].tolist() == [3.50, 4.50, 5.80]
    assert df["vs"].tolist() == [2.00, 2.60, 3.35]
    assert df["rho"].tolist() == [2.3, 2.5, 2.7]


def test_read_layers_default_rho(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("0.0 3.5 2.0\n1.0 4.5 2.6 2.5\n")

    df = read_layers(path)

    assert df["rho"].tolist() == [2.7, 2.5]


def test_read_layers_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("# a full-line comment\n\n0.0 3.5 2.0 2.3  # trailing comment\n\n2.0 5.0 2.9\n")

    df = read_layers(path)

    assert len(df) == 2
    assert df["depth_km"].tolist() == [0.0, 2.0]


def test_read_layers_sorts_out_of_order_input(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("2.0 5.0 2.9\n0.0 3.5 2.0\n")

    df = read_layers(path)

    assert df["depth_km"].tolist() == [0.0, 2.0]


def test_read_layers_negative_depth_above_sea_level(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("-0.5 3.0 1.7\n0.0 3.5 2.0\n")

    df = read_layers(path)

    assert df["depth_km"].tolist() == [-0.5, 0.0]


def test_read_layers_empty_file_raises(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("# only comments\n\n")

    with pytest.raises(ValueError, match="no layers"):
        read_layers(path)


def test_read_layers_too_few_columns_raises(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("0.0 3.5\n")

    with pytest.raises(ValueError, match="at least 3 columns"):
        read_layers(path)


def test_read_layers_non_numeric_raises(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("0.0 abc 2.0\n")

    with pytest.raises(ValueError, match="non-numeric"):
        read_layers(path)


def test_read_layers_duplicate_depth_raises(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text("0.0 3.5 2.0\n0.0 4.5 2.6\n")

    with pytest.raises(ValueError, match="strictly increase"):
        read_layers(path)


def test_read_layers_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_layers(tmp_path / "does_not_exist.txt")


def test_read_layers_returns_dataframe(tmp_path):
    path = tmp_path / "vel1d.txt"
    path.write_text(VALID_LAYERS)

    df = read_layers(path)

    assert isinstance(df, pd.DataFrame)
