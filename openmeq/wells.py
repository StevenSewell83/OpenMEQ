"""Well heads/tracks for the map and depth-section diagnostics (OpenMEQ-7zt).

Reads a directory laid out like ``projects/RK-NM/wells``: per well group
(``Injection``, ``Production``, ``Other``, ...) a ``<Group>_well_heads.csv``
(``name,lon,lat``) and/or ``<Group>_well_tracks.csv`` (``name,lon,lat,
elevation_m``, one row per survey point, top down; elevation in metres
positive up relative to sea level). Files are headerless; a header row, if
present, is dropped because its coordinates are not numeric. Heads and tracks
are independent: a well may have either or both.

Enabled by ``diagnostics.wells`` in the config. Loading never raises -- a
missing directory or unreadable file is logged and the overlay is skipped,
since a figure decoration must not abort a pipeline step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import Wells
from openmeq.plotting._style import COLORBLIND_PALETTE
from openmeq.plotting._wells import HEAD_COLUMNS, TRACK_COLUMNS

if TYPE_CHECKING:
    from openmeq.config import Config, WellsConfig

logger = get_logger("wells")

HEADS_SUFFIX = "_well_heads.csv"
TRACKS_SUFFIX = "_well_tracks.csv"

#: Group colours used unless ``diagnostics.wells.colors`` sets one
#: (case-insensitive group name). Other groups cycle the remaining palette.
DEFAULT_COLORS = {"production": "#D55E00", "injection": "#56B4E9", "other": "0.35"}


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, usecols=range(len(columns)), names=columns, skipinitialspace=True
    )
    df["well"] = df["well"].astype(str).str.strip()
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=columns[1:])


def read_wells(
    directory: str | Path,
    groups: list[str] | None = None,
    colors: dict[str, str] | None = None,
    labels: bool = False,
) -> Wells:
    """Read every ``<Group>_well_{heads,tracks}.csv`` in ``directory``.

    Parameters
    ----------
    directory : str or pathlib.Path
    groups : list of str, optional
        Only these group prefixes (default: every group found).
    colors : dict, optional
        ``{group: colour}`` overriding :data:`DEFAULT_COLORS`.
    labels : bool
        Whether figures annotate well names.

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist or holds no well files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"well directory {directory} does not exist")
    found = sorted(
        {p.name[: -len(s)] for s in (HEADS_SUFFIX, TRACKS_SUFFIX) for p in directory.glob(f"*{s}")}
    )
    if groups:
        missing = [g for g in groups if g not in found]
        if missing:
            logger.warning("well groups %s not found in %s (found: %s)", missing, directory, found)
        found = [g for g in groups if g in found]
    if not found:
        raise FileNotFoundError(f"no *{HEADS_SUFFIX} / *{TRACKS_SUFFIX} files in {directory}")

    heads, tracks = [], []
    for group in found:
        h = directory / f"{group}{HEADS_SUFFIX}"
        t = directory / f"{group}{TRACKS_SUFFIX}"
        if h.is_file():
            heads.append(_read_csv(h, HEAD_COLUMNS[1:]).assign(group=group))
        if t.is_file():
            trk = _read_csv(t, ["well", "lon", "lat", "elevation_m"]).assign(group=group)
            trk["depth_km"] = -trk.pop("elevation_m") / 1000.0
            tracks.append(trk)

    spare = iter(c for c in COLORBLIND_PALETTE if c not in DEFAULT_COLORS.values())
    resolved = {}
    for group in found:
        resolved[group] = (
            (colors or {}).get(group) or DEFAULT_COLORS.get(group.lower()) or next(spare, "0.3")
        )

    wells = Wells(
        tracks=pd.concat(tracks, ignore_index=True)[TRACK_COLUMNS]
        if tracks
        else pd.DataFrame(columns=TRACK_COLUMNS),
        heads=pd.concat(heads, ignore_index=True)[HEAD_COLUMNS]
        if heads
        else pd.DataFrame(columns=HEAD_COLUMNS),
        colors=resolved,
        labels=labels,
    )
    logger.info(
        "loaded %d wellheads and %d well tracks (%s) from %s",
        len(wells.heads),
        wells.tracks["well"].nunique(),
        ", ".join(found),
        directory,
    )
    return wells


def load_wells(cfg: Config | WellsConfig | None) -> Wells | None:
    """The configured well overlay, or ``None`` when disabled or unreadable
    (logged as a warning, never raised)."""
    if cfg is None:
        return None
    wcfg = cfg.diagnostics.wells if hasattr(cfg, "diagnostics") else cfg
    if not wcfg.enabled or wcfg.dir is None:
        return None
    try:
        return read_wells(wcfg.dir, wcfg.groups or None, wcfg.colors, wcfg.labels)
    except Exception as exc:  # noqa: BLE001 - a plot overlay must never fail the step
        logger.warning("well overlay skipped: %s", exc)
        return None
