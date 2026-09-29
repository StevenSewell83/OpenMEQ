"""``locate/diagnostics/extended/`` -- opt-in extended locate QAQC (OpenMEQ-02o.5).

Written only when ``cfg.diagnostics.extended_enabled("locate")`` is true
(:func:`openmeq.locate.diagnostics.make_locate_diagnostics` does the
gating), on top of the always-on baseline set. For **each pass (A and B)**:

- ``passX_res_vs_distance.png`` -- P and S time residual vs epicentral
  distance (one panel per phase), with a binned median line. Distances come
  from :attr:`openmeq.locate.locate_step.PassDiagnostics.distances_km`
  (NLLoc's own ``Arrival.distance``, km).
- ``passX_station_res_map.png`` -- station map, one panel per phase, each
  station coloured by its mean residual (diverging colormap centred on 0)
  and sized by its residual count; this pass's epicentres drawn faintly
  behind. Stations with no residual for that phase are hollow grey.
- ``passX_station_residuals.csv`` -- ``station, phase, n, mean, median,
  std`` (seconds; ``std`` is the sample std, blank for ``n == 1``) -- the
  same numbers as the map, usable as candidate station corrections.
- ``passX_depth_hist_grid.png`` -- event depth histogram (depth downward)
  with the NLLoc ``LOCGRID`` top/bottom
  (:func:`openmeq.nlloc_setup.location_grid_bounds`, the same bounds the
  control file is built from) and, for a 1D model, the velocity-model layer
  interfaces (:func:`openmeq.velocity.read_layers`) marked, so events piling
  up on a grid edge or an interface stand out. A 3D model gets grid limits
  only.

Extended QAQC must never fail the step: every figure/CSV is built inside
its own ``try``/``except`` that logs a warning and moves on, and every
builder draws a "No data" placeholder rather than raising on an empty pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import save_fig, setup_style

if TYPE_CHECKING:
    from collections.abc import Callable

    from openmeq.config import Config
    from openmeq.locate.locate_step import LocateRunResult, PassDiagnostics
    from openmeq.paths import RunPaths

logger = get_logger("locate.diagnostics.extended")

_PHASES = ("P", "S")

_ARRIVAL_COLUMNS = ["event_id", "seed_id", "net_sta", "phase", "residual_s", "distance_km"]

STATION_RESIDUAL_COLUMNS = ["station", "phase", "n", "mean", "median", "std"]

#: Diverging, colour-blind-safe colormap for residuals centred on 0.
_DIVERGING_CMAP = "RdBu_r"

#: Minimum residuals per distance bin before a binned-median point is drawn.
_MIN_PER_BIN = 3

_N_DISTANCE_BINS = 12


def extended_dir(paths: RunPaths) -> Path:
    """``locate/diagnostics/extended/``."""
    return paths.diagnostics_dir("locate") / "extended"


# ---------------------------------------------------------------------------
# Data flattening
# ---------------------------------------------------------------------------


def _arrival_dataframe(diag_map: dict[str, PassDiagnostics]) -> pd.DataFrame:
    """One row per (event, seed id) residual, with its distance (NaN if unknown)."""
    rows = []
    for d in diag_map.values():
        distances = getattr(d, "distances_km", None) or {}
        for seed_id, residual in d.residuals.items():
            rows.append(
                {
                    "event_id": d.event_id,
                    "seed_id": seed_id,
                    "net_sta": ".".join(seed_id.split(".")[:2]),
                    "phase": d.phase_by_seed_id.get(seed_id),
                    "residual_s": residual,
                    "distance_km": distances.get(seed_id, np.nan),
                }
            )
    df = pd.DataFrame(rows, columns=_ARRIVAL_COLUMNS)
    return df.astype({"residual_s": float, "distance_km": float})


def _epicentres(diag_map: dict[str, PassDiagnostics]) -> pd.DataFrame:
    rows = [
        {"lat": d.lat, "lon": d.lon, "depth_km": d.depth_km}
        for d in diag_map.values()
        if d.ran and d.lat is not None and d.lon is not None
    ]
    return pd.DataFrame(rows, columns=["lat", "lon", "depth_km"])


def station_residual_table(arrivals: pd.DataFrame) -> pd.DataFrame:
    """Per-station, per-phase residual statistics (:data:`STATION_RESIDUAL_COLUMNS`).

    Parameters
    ----------
    arrivals : pandas.DataFrame
        :func:`_arrival_dataframe`-shaped (``net_sta``, ``phase``,
        ``residual_s``). May be empty.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``phase`` then ``station``; empty (with the columns) if
        there are no finite residuals.
    """
    sub = arrivals.dropna(subset=["phase", "residual_s"]) if not arrivals.empty else arrivals
    if sub.empty:
        return pd.DataFrame(columns=STATION_RESIDUAL_COLUMNS)
    grouped = sub.groupby(["phase", "net_sta"])["residual_s"]
    table = grouped.agg(n="count", mean="mean", median="median", std="std").reset_index()
    table = table.rename(columns={"net_sta": "station"})
    table["n"] = table["n"].astype(int)
    return table[STATION_RESIDUAL_COLUMNS].sort_values(["phase", "station"], ignore_index=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _no_data(ax) -> None:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def _binned_median(distance: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres and medians for bins holding at least :data:`_MIN_PER_BIN` points."""
    if distance.size < _MIN_PER_BIN:
        return np.array([]), np.array([])
    lo, hi = float(distance.min()), float(distance.max())
    if hi <= lo:
        return np.array([lo]), np.array([float(np.median(residual))])
    edges = np.linspace(lo, hi, _N_DISTANCE_BINS + 1)
    idx = np.clip(np.digitize(distance, edges) - 1, 0, _N_DISTANCE_BINS - 1)
    centres, medians = [], []
    for b in range(_N_DISTANCE_BINS):
        in_bin = residual[idx == b]
        if in_bin.size >= _MIN_PER_BIN:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(float(np.median(in_bin)))
    return np.asarray(centres), np.asarray(medians)


def plot_res_vs_distance(arrivals: pd.DataFrame, pass_label: str, path: Path) -> None:
    """P/S residual vs epicentral distance, one panel per phase, with binned median."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(_PHASES), figsize=(11.0, 4.2), sharey=True, squeeze=False)
    for ax, phase in zip(axes[0], _PHASES, strict=True):
        sub = arrivals[arrivals["phase"] == phase].dropna(subset=["distance_km", "residual_s"])
        if sub.empty:
            _no_data(ax)
        else:
            dist = sub["distance_km"].to_numpy(dtype=float)
            res = sub["residual_s"].to_numpy(dtype=float)
            ax.scatter(dist, res, s=12, alpha=0.5, color="#0072B2", edgecolor="none")
            centres, medians = _binned_median(dist, res)
            if centres.size:
                ax.plot(
                    centres, medians, "-o", color="#D55E00", lw=1.8, ms=4, label="binned median"
                )
                ax.legend(loc="best")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("Epicentral distance (km)")
        ax.set_title(f"{phase} (n={len(sub)})")
    axes[0][0].set_ylabel("Time residual (s)")
    fig.suptitle(f"Pass {pass_label}: residual vs distance")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_fig(fig, path)


def plot_station_res_map(
    table: pd.DataFrame,
    stations: pd.DataFrame | None,
    epicentres: pd.DataFrame,
    pass_label: str,
    path: Path,
) -> None:
    """Station map coloured by mean residual (per phase), sized by count."""
    setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    have_stations = (
        stations is not None
        and not stations.empty
        and {"network", "station", "latitude", "longitude"} <= set(stations.columns)
    )
    if have_stations:
        sta = stations.assign(net_sta=stations["network"] + "." + stations["station"])
        sta = sta.drop_duplicates("net_sta")[["net_sta", "latitude", "longitude"]]

    finite_means = table["mean"].to_numpy(dtype=float) if not table.empty else np.array([])
    finite_means = finite_means[np.isfinite(finite_means)]
    vmax = float(np.percentile(np.abs(finite_means), 98)) if finite_means.size else 0.0
    vmax = vmax if vmax > 0 else 0.1
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    n_max = int(table["n"].max()) if not table.empty else 1

    fig, axes = plt.subplots(1, len(_PHASES), figsize=(11.0, 5.0), squeeze=False)
    mappable = None
    for ax, phase in zip(axes[0], _PHASES, strict=True):
        if not epicentres.empty:
            ax.scatter(epicentres["lon"], epicentres["lat"], s=6, color="0.6", alpha=0.35, zorder=1)
        if not have_stations:
            _no_data(ax)
        else:
            merged = sta.merge(
                table[table["phase"] == phase], left_on="net_sta", right_on="station", how="left"
            )
            missing = merged[merged["mean"].isna()]
            present = merged[merged["mean"].notna()]
            ax.scatter(
                missing["longitude"],
                missing["latitude"],
                marker="^",
                s=40,
                facecolor="none",
                edgecolor="0.5",
                zorder=2,
            )
            if not present.empty:
                sizes = 30.0 + 170.0 * np.sqrt(present["n"].to_numpy(dtype=float) / n_max)
                mappable = ax.scatter(
                    present["longitude"],
                    present["latitude"],
                    c=present["mean"].to_numpy(dtype=float),
                    s=sizes,
                    marker="^",
                    cmap=_DIVERGING_CMAP,
                    norm=norm,
                    edgecolor="k",
                    linewidth=0.5,
                    zorder=3,
                )
            else:
                ax.text(0.5, 0.95, "No residuals", ha="center", va="top", transform=ax.transAxes)
            for row in merged.itertuples(index=False):
                ax.annotate(
                    row.net_sta.split(".")[-1],
                    (row.longitude, row.latitude),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{phase}: mean residual per station")
    if mappable is not None:
        fig.colorbar(mappable, ax=list(axes[0]), label="Mean residual (s)", shrink=0.85)
    fig.suptitle(f"Pass {pass_label}: station residuals (marker size = count)")
    save_fig(fig, path)


def _grid_and_interfaces(
    cfg: Config, stations: pd.DataFrame | None
) -> tuple[float | None, float | None, list[float]]:
    """``(grid_top_km, grid_bottom_km, interface_depths_km)`` from the same
    sources the NLLoc control files are built from. Each piece degrades to
    ``None``/``[]`` (with a warning) rather than raising."""
    from openmeq.nlloc_setup import location_grid_bounds

    z_top = z_bottom = None
    try:
        table = stations
        if table is None or not {"elevation_m", "depth_m"} <= set(table.columns):
            table = pd.DataFrame(columns=["elevation_m", "depth_m"])
        bounds = location_grid_bounds(cfg, table)
        z_top, z_bottom = float(bounds.z_min_km), float(bounds.z_max_km)
    except Exception as exc:  # noqa: BLE001 - extended QAQC must never abort the run
        logger.warning("could not compute NLLoc grid depth limits: %s", exc)

    interfaces: list[float] = []
    if cfg.nlloc.velocity_model.type == "1d":
        try:
            from openmeq.velocity import read_layers

            layers = read_layers(cfg.nlloc.velocity_model.file)
            interfaces = [float(z) for z in layers["depth_km"].to_numpy()[1:]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read velocity-model layers: %s", exc)
    return z_top, z_bottom, interfaces


def plot_depth_hist_grid(
    depths_km: np.ndarray,
    z_top: float | None,
    z_bottom: float | None,
    interfaces: list[float],
    pass_label: str,
    path: Path,
) -> None:
    """Horizontal depth histogram (depth downward) with grid limits and layer
    interfaces drawn as horizontal lines."""
    setup_style()
    import matplotlib.pyplot as plt

    depths_km = np.asarray(depths_km, dtype=float)
    depths_km = depths_km[np.isfinite(depths_km)]

    fig, ax = plt.subplots(figsize=(6.0, 6.5))
    limits = [z for z in (z_top, z_bottom) if z is not None]
    span = [*limits, *depths_km.tolist()]
    if depths_km.size:
        lo, hi = min(span), max(span)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        ax.hist(
            depths_km,
            bins=np.linspace(lo, hi, 41),
            orientation="horizontal",
            color="#0072B2",
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        _no_data(ax)

    for i, z in enumerate(limits):
        ax.axhline(
            z, color="#D55E00", ls="--", lw=1.5, label="NLLoc grid limit" if i == 0 else None
        )
    for i, z in enumerate(interfaces):
        ax.axhline(z, color="0.3", ls=":", lw=1.2, label="velocity interface" if i == 0 else None)

    if span or interfaces:
        all_z = span + interfaces
        pad = 0.03 * max(max(all_z) - min(all_z), 1.0)
        ax.set_ylim(max(all_z) + pad, min(all_z) - pad)  # depth increases downward
    else:
        ax.invert_yaxis()
    if limits or interfaces:
        ax.legend(loc="best")
    ax.set_xlabel("Event count")
    ax.set_ylabel("Depth (km)")
    note = "" if interfaces or not limits else " (grid limits only)"
    ax.set_title(f"Pass {pass_label}: event depths vs grid/model{note} (n={depths_km.size})")
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _safe(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - extended QAQC must never fail the step
        logger.warning("extended locate diagnostic %s failed: %s", name, exc)


def make_locate_extended_diagnostics(
    paths: RunPaths,
    cfg: Config,
    result: LocateRunResult,
    stations: pd.DataFrame | None = None,
) -> Path:
    """Write ``locate/diagnostics/extended/`` for both passes (module docstring).

    Does no gating itself -- the caller checks
    ``cfg.diagnostics.extended_enabled("locate")``.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config
    result : openmeq.locate.locate_step.LocateRunResult
    stations : pandas.DataFrame, optional
        ``openmeq.stations.station_table``-shaped; ``None`` skips the station
        map's stations (a "No data" panel) and uses the default grid top.

    Returns
    -------
    pathlib.Path
        ``locate/diagnostics/extended/``.
    """
    out_dir = extended_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_top, z_bottom, interfaces = _grid_and_interfaces(cfg, stations)

    for pass_label, diag_map in (("A", result.pass_a), ("B", result.pass_b)):
        prefix = f"pass{pass_label}"
        arrivals = _arrival_dataframe(diag_map)
        epicentres = _epicentres(diag_map)
        table = station_residual_table(arrivals)

        _safe(
            f"{prefix}_res_vs_distance",
            lambda a=arrivals, p=pass_label, pre=prefix: plot_res_vs_distance(
                a, p, out_dir / f"{pre}_res_vs_distance.png"
            ),
        )
        _safe(
            f"{prefix}_station_residuals.csv",
            lambda t=table, pre=prefix: t.to_csv(
                out_dir / f"{pre}_station_residuals.csv", index=False, float_format="%.6f"
            ),
        )
        _safe(
            f"{prefix}_station_res_map",
            lambda t=table, e=epicentres, p=pass_label, pre=prefix: plot_station_res_map(
                t, stations, e, p, out_dir / f"{pre}_station_res_map.png"
            ),
        )
        _safe(
            f"{prefix}_depth_hist_grid",
            lambda e=epicentres, p=pass_label, pre=prefix: plot_depth_hist_grid(
                e["depth_km"].to_numpy(dtype=float),
                z_top,
                z_bottom,
                interfaces,
                p,
                out_dir / f"{pre}_depth_hist_grid.png",
            ),
        )

    return out_dir


__all__ = [
    "STATION_RESIDUAL_COLUMNS",
    "extended_dir",
    "make_locate_extended_diagnostics",
    "station_residual_table",
]
