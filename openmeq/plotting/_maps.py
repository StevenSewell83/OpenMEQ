"""Map view + depth-section diagnostic figure.

Used for per-pass location diagnostics (M6.5), hypoDD before/after maps
(M10.5), and magnitude station maps (M8.3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._save import save_fig
from ._style import SEQUENTIAL_CMAP, setup_style
from ._util import auto_scale
from ._wells import Wells, draw_wells_map, draw_wells_section


def _station_xy(stations: pd.DataFrame | None, lon_col: str, lat_col: str) -> pd.DataFrame | None:
    """Return a usable station frame, or ``None`` if it can't be plotted.

    Handles the "mismatched/empty station data" edge case: missing frame,
    empty frame, or a frame that lacks the expected columns all degrade to
    "skip the station overlay" rather than raising.
    """
    if stations is None:
        return None
    if not isinstance(stations, pd.DataFrame) or stations.empty:
        return None
    if lon_col not in stations.columns or lat_col not in stations.columns:
        return None
    sta = stations[[lon_col, lat_col]].dropna()
    if sta.empty:
        return None
    return sta


def _station_depth_km(stations: pd.DataFrame) -> np.ndarray:
    """Station depth on the event-depth axis (km below sea level): the
    ARCHITECTURE.md §6 effective elevation ``elevation_m - depth_m``, negated,
    as NLLoc uses it. Missing columns/values fall back to 0 (sea level)."""
    elev = stations.get("elevation_m", pd.Series(0.0, index=stations.index))
    burial = stations.get("depth_m", pd.Series(0.0, index=stations.index))
    eff = pd.to_numeric(elev, errors="coerce").fillna(0.0) - pd.to_numeric(
        burial, errors="coerce"
    ).fillna(0.0)
    return -eff.to_numpy(dtype=float) / 1000.0


def _plot_map_panel(
    ax,
    df: pd.DataFrame,
    stations: pd.DataFrame | None,
    *,
    lon_col: str,
    lat_col: str,
    station_lon_col: str,
    station_lat_col: str,
    color_values: np.ndarray | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    marker_size: float,
    station_marker_size: float,
    wells: Wells | None = None,
):
    mappable = None
    if not df.empty:
        c = color_values if color_values is not None else "#0072B2"
        mappable = ax.scatter(
            df[lon_col],
            df[lat_col],
            c=c,
            cmap=cmap if color_values is not None else None,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            edgecolors="k",
            linewidths=0.3,
            zorder=3,
        )
    sta = _station_xy(stations, station_lon_col, station_lat_col)
    if sta is not None:
        ax.scatter(
            sta[station_lon_col],
            sta[station_lat_col],
            marker="v",
            c="black",
            s=station_marker_size,
            zorder=2,
            label="station",
        )
    if draw_wells_map(ax, wells):
        ax.legend(fontsize=7, loc="best")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Map view")
    ax.set_aspect("auto")
    return mappable


def _plot_section_panel(
    ax,
    df: pd.DataFrame,
    x_col: str,
    depth_col: str,
    stations: pd.DataFrame | None,
    station_x_col: str,
    *,
    xlabel: str,
    title: str,
    color_values: np.ndarray | None,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    marker_size: float,
    wells: Wells | None = None,
    wells_x_col: str | None = None,
    station_marker_size: float = 37.5,
):
    if not df.empty:
        c = color_values if color_values is not None else "#0072B2"
        ax.scatter(
            df[x_col],
            df[depth_col],
            c=c,
            cmap=cmap if color_values is not None else None,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            edgecolors="k",
            linewidths=0.3,
            zorder=3,
        )
    if stations is not None and isinstance(stations, pd.DataFrame) and not stations.empty:
        if station_x_col in stations.columns:
            sta = stations[stations[station_x_col].notna()]
            if not sta.empty:
                ax.scatter(
                    sta[station_x_col],
                    _station_depth_km(sta),
                    marker="v",
                    c="black",
                    s=station_marker_size,
                    zorder=2,
                )
    if wells_x_col is not None:
        draw_wells_section(ax, wells, wells_x_col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (km)")
    ax.set_title(title)
    ax.invert_yaxis()


def plot_map_sections(
    df: pd.DataFrame,
    stations: pd.DataFrame | None,
    path: str | Path,
    *,
    color_by: str | None = None,
    cmap: str = SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    marker_size: float = 10.0,
    station_marker_size: float = 60.0,
    title: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    depth_col: str = "depth_km",
    station_lon_col: str = "longitude",
    station_lat_col: str = "latitude",
    wells: Wells | None = None,
) -> None:
    """Draw a 3-panel map + E-W/N-S depth-section diagnostic figure.

    Panel 1 is a map view (longitude vs. latitude). Panel 2 is the E-W
    depth section (longitude vs. depth). Panel 3 is the N-S depth section
    (latitude vs. depth). Depth increases downward on both section panels.
    Stations are overlaid on the map panel (as black triangles) and, at
    their elevation (``elevation_m - depth_m`` when the station frame has
    those columns, else sea level), on the two section panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Event locations. Must have at least ``lon_col``, ``lat_col`` and
        ``depth_col`` columns (default names match ``locate/summary.csv``:
        ``lon``, ``lat``, ``depth_km``). May be empty (an empty figure with
        axes and no points is produced, not an error).
    stations : pandas.DataFrame or None
        Station locations, e.g. from ``openmeq.stations.station_table``
        (columns include ``latitude``, ``longitude``). May be ``None``,
        empty, or missing the expected columns -- the station overlay is
        then simply skipped rather than raising.
    path : str or pathlib.Path
        Output PNG path.
    color_by : str, optional
        Name of a column in ``df`` to color the event markers by (e.g.
        ``"rms"`` or a magnitude column). A colorbar is added when given.
        If omitted, all events are drawn in a single colour.
    cmap : str, optional
        Colormap for ``color_by`` (default: the package's sequential
        colormap).
    vmin, vmax : float, optional
        Explicit colour-scale bounds for ``color_by``. If not given, an
        auto range is used (2nd/98th percentile of the finite values), so
        the scale adapts to datasets of very different magnitude.
    cbar_label : str, optional
        Label for the colorbar (defaults to ``color_by``).
    marker_size, station_marker_size : float, optional
        Scatter marker sizes for events and stations.
    title : str, optional
        Overall figure title.
    lon_col, lat_col, depth_col : str, optional
        Column names in ``df`` (defaults match ``locate/summary.csv``).
    station_lon_col, station_lat_col : str, optional
        Column names in ``stations`` (defaults match ``station_table``).
    wells : openmeq.plotting.Wells, optional
        Well heads/tracks to overlay (tracks on all three panels, heads on
        the map); ``None`` (default) draws none.
    """
    setup_style()
    import matplotlib.pyplot as plt

    df = df if df is not None else pd.DataFrame(columns=[lon_col, lat_col, depth_col])

    color_values = None
    if color_by is not None and color_by in df.columns and not df.empty:
        color_values = df[color_by].to_numpy(dtype=float)
        vmin, vmax = auto_scale(color_values, vmin, vmax)

    fig, (ax_map, ax_ew, ax_ns) = plt.subplots(1, 3, figsize=(15, 4.8))

    mappable = _plot_map_panel(
        ax_map,
        df,
        stations,
        lon_col=lon_col,
        lat_col=lat_col,
        station_lon_col=station_lon_col,
        station_lat_col=station_lat_col,
        color_values=color_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker_size=marker_size,
        station_marker_size=station_marker_size,
        wells=wells,
    )
    _plot_section_panel(
        ax_ew,
        df,
        lon_col,
        depth_col,
        stations,
        station_lon_col,
        xlabel="Longitude (deg)",
        title="E-W section",
        color_values=color_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker_size=marker_size,
        wells=wells,
        wells_x_col="lon",
    )
    _plot_section_panel(
        ax_ns,
        df,
        lat_col,
        depth_col,
        stations,
        station_lat_col,
        xlabel="Latitude (deg)",
        title="N-S section",
        color_values=color_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker_size=marker_size,
        wells=wells,
        wells_x_col="lat",
    )

    if mappable is not None and color_values is not None:
        fig.colorbar(mappable, ax=[ax_map, ax_ew, ax_ns], label=cbar_label or color_by, shrink=0.85)

    if title:
        fig.suptitle(title)

    save_fig(fig, path)
