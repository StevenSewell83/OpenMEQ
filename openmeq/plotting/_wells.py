"""Well heads/tracks overlay for map and depth-section panels (OpenMEQ-7zt).

:class:`Wells` is plain data (built from config by :func:`openmeq.wells.load_wells`);
the draw helpers take it as an optional argument and do nothing when it is
``None`` or empty, so every map/section figure can accept it unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from matplotlib import patheffects

TRACK_COLUMNS = ["group", "well", "lon", "lat", "depth_km"]
HEAD_COLUMNS = ["group", "well", "lon", "lat"]


@dataclass(frozen=True)
class Wells:
    """Well geometry to overlay.

    ``tracks`` has one row per survey point (:data:`TRACK_COLUMNS`, in
    survey order within each well; ``depth_km`` is km below sea level, the
    same datum as event depths). ``heads`` has one row per wellhead
    (:data:`HEAD_COLUMNS`). ``colors`` maps group -> matplotlib colour.
    """

    tracks: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=TRACK_COLUMNS))
    heads: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=HEAD_COLUMNS))
    colors: dict[str, str] = field(default_factory=dict)
    labels: bool = False

    @property
    def empty(self) -> bool:
        return self.tracks.empty and self.heads.empty

    def groups(self) -> list[str]:
        return list(dict.fromkeys([*self.heads["group"], *self.tracks["group"]]))


# Semi-transparent so events behind a well stay visible.
WELL_ALPHA = 0.55


def _color(wells: Wells, group: str) -> str:
    return wells.colors.get(group, "0.3")


def draw_wells_map(ax, wells: Wells | None) -> bool:
    """Draw well tracks (lines), heads (open circles) and optional name
    labels on a lon/lat map panel. Returns whether anything was drawn."""
    if wells is None or wells.empty:
        return False
    for (group, _), trk in wells.tracks.groupby(["group", "well"], sort=False):
        ax.plot(
            trk["lon"], trk["lat"], color=_color(wells, group), lw=0.6, alpha=WELL_ALPHA, zorder=4
        )
    for group, heads in wells.heads.groupby("group", sort=False):
        ax.scatter(
            heads["lon"],
            heads["lat"],
            marker="o",
            s=14,
            facecolor="none",
            edgecolor=_color(wells, group),
            linewidths=0.8,
            alpha=WELL_ALPHA,
            zorder=4,
        )
    for group in wells.groups():
        ax.plot(
            [],
            [],
            color=_color(wells, group),
            lw=1.0,
            marker="o",
            mfc="none",
            ms=4,
            alpha=WELL_ALPHA,
            label=f"{group} wells",
        )
    if wells.labels:
        # One label per well: at its head, else at the top of its track.
        tops = wells.tracks.groupby("well", sort=False)[["lon", "lat"]].first()
        anchors = {
            **{w: (r.lon, r.lat) for w, r in tops.iterrows()},
            **{r.well: (r.lon, r.lat) for r in wells.heads.itertuples(index=False)},
        }
        for name, (x, y) in anchors.items():
            ax.annotate(
                name,
                (x, y),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5,
                color="0.1",
                zorder=5,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
            )
    return True


def draw_wells_section(ax, wells: Wells | None, x_col: str) -> bool:
    """Draw well tracks on a depth-section panel whose x axis is ``x_col``
    (``"lon"`` or ``"lat"``). Wellheads without a track have no elevation and
    are not drawn. Returns whether anything was drawn."""
    if wells is None or wells.tracks.empty:
        return False
    for (group, _), trk in wells.tracks.groupby(["group", "well"], sort=False):
        ax.plot(
            trk[x_col],
            trk["depth_km"],
            color=_color(wells, group),
            lw=0.6,
            alpha=WELL_ALPHA,
            zorder=4,
        )
    return True
