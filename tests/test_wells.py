"""Tests for the well heads/tracks overlay (OpenMEQ-7zt): openmeq.wells
loading, the diagnostics.wells config section, and the map/section draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from openmeq.config import DiagnosticsConfig, WellsConfig
from openmeq.plotting import Wells, draw_wells_map, draw_wells_section, plot_map_sections
from openmeq.wells import DEFAULT_COLORS, load_wells, read_wells


@pytest.fixture
def well_dir(tmp_path):
    """Two groups in the RK-NM layout: heads and tracks independent."""
    d = tmp_path / "wells"
    d.mkdir()
    (d / "Production_well_heads.csv").write_text("P1,176.19,-38.60\nP2,176.20,-38.61\n")
    (d / "Production_well_tracks.csv").write_text(
        "P1,176.19,-38.60,350.0\nP1,176.191,-38.601,-500.0\nP1,176.192,-38.602,-2100.5\n"
        "P9,176.18,-38.59,300.0\nP9,176.18,-38.59,-1000.0\n"
    )
    # A header row is tolerated (dropped as non-numeric).
    (d / "Injection_well_tracks.csv").write_text(
        "name,lon,lat,elev\nI1,176.21,-38.62,340.0\nI1,176.212,-38.621,-1800.0\n"
    )
    return d


def test_read_wells_parses_heads_and_tracks(well_dir):
    w = read_wells(well_dir)
    assert set(w.groups()) == {"Production", "Injection"}
    assert list(w.heads["well"]) == ["P1", "P2"]
    assert w.tracks["well"].nunique() == 3
    p1 = w.tracks[w.tracks["well"] == "P1"]
    # elevation (m, up) -> depth below sea level (km, down)
    assert p1["depth_km"].tolist() == pytest.approx([-0.35, 0.5, 2.1005])
    assert w.colors["Production"] == DEFAULT_COLORS["production"]
    assert w.colors["Injection"] == DEFAULT_COLORS["injection"]


def test_read_wells_group_filter_and_colour_override(well_dir):
    w = read_wells(well_dir, groups=["Injection", "Nope"], colors={"Injection": "k"})
    assert w.groups() == ["Injection"]
    assert w.heads.empty
    assert w.colors == {"Injection": "k"}


def test_read_wells_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wells(tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        read_wells(tmp_path)  # exists but holds no well files


def test_load_wells_disabled_or_broken_returns_none(well_dir, tmp_path):
    assert load_wells(None) is None
    assert load_wells(WellsConfig()) is None
    assert load_wells(WellsConfig(enabled=True, dir=tmp_path / "absent")) is None
    w = load_wells(WellsConfig(enabled=True, dir=well_dir))
    assert w is not None and not w.labels  # names off by default


def test_wells_config_requires_dir_when_enabled():
    with pytest.raises(ValidationError, match="wells.dir"):
        WellsConfig(enabled=True)
    assert DiagnosticsConfig().wells.enabled is False


def test_draw_wells_noop_without_wells():
    fig, ax = plt.subplots()
    assert draw_wells_map(ax, None) is False
    assert draw_wells_section(ax, Wells(), "lon") is False
    assert not ax.lines
    plt.close(fig)


def test_draw_wells_map_and_section(well_dir):
    w = read_wells(well_dir, labels=True)
    fig, (ax_map, ax_sec) = plt.subplots(1, 2)
    assert draw_wells_map(ax_map, w)
    assert draw_wells_section(ax_sec, w, "lat")
    # 3 tracks + one legend proxy per group on the map; 3 tracks on the section
    assert len(ax_map.lines) == 3 + 2
    assert len(ax_sec.lines) == 3
    assert {t.get_text() for t in ax_map.texts} == {"P1", "P2", "P9", "I1"}
    plt.close(fig)
    fig, ax = plt.subplots()
    draw_wells_map(ax, read_wells(well_dir))
    assert not ax.texts
    plt.close(fig)


def test_plot_map_sections_with_wells(well_dir, tmp_path):
    events = pd.DataFrame(
        {"lon": [176.19, 176.2], "lat": [-38.6, -38.61], "depth_km": [2.0, 3.0], "rms": [0.1, 0.2]}
    )
    stations = pd.DataFrame({"latitude": [-38.605], "longitude": [176.195]})
    out = tmp_path / "map.png"
    figs_before = set(plt.get_fignums())
    plot_map_sections(events, stations, out, color_by="rms", wells=read_wells(well_dir))
    assert out.stat().st_size > 0
    assert set(plt.get_fignums()) <= figs_before  # figure closed after saving


def test_section_stations_at_effective_elevation():
    from openmeq.plotting._maps import _station_depth_km

    sta = pd.DataFrame({"elevation_m": [350.0, 400.0], "depth_m": [0.0, 1000.0]})
    assert _station_depth_km(sta).tolist() == pytest.approx([-0.35, 0.6])
    # no elevation columns -> sea level, as before
    assert _station_depth_km(pd.DataFrame({"latitude": [1.0]})).tolist() == [0.0]
