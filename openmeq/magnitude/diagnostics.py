"""Magnitude diagnostics (M8.3 / OpenMEQ-625.3, ``docs/PLAN.md`` §5's
magnitude row: "station ML against distance, station residuals, GR plot,
ML over time").

:func:`make_magnitude_diagnostics` works purely from
:func:`~openmeq.magnitude.pipeline.run_magnitude`'s own in-memory
``station_magnitudes.csv``-shaped ``pandas.DataFrame`` and the output
`Catalog`, mirroring :mod:`openmeq.detect.diagnostics`'s own "diagnostics
module never recomputes the step" convention. Every figure goes through
:mod:`openmeq.plotting`; nothing here imports ``matplotlib`` directly.

Outputs (``magnitude/diagnostics/``, ARCHITECTURE.md §3)
------------------------------------------------------------
- ``ml_vs_distance.png``: every *included* station ML vs hypocentral
  distance, coloured by station (:func:`~openmeq.plotting.plot_scatter`).
- ``station_residuals.png``: station ML minus its event's ML, grouped by
  station, aggregated across every event with a Magnitude
  (:func:`~openmeq.plotting.plot_boxplot`) -- feeds a future station-
  correction calibration.
- ``magnitude_frequency.png``: cumulative Gutenberg-Richter plot over every
  event ML in the run (:func:`~openmeq.plotting.plot_gutenberg_richter`).
- ``ml_through_time.png``: event ML vs origin time, one point per event
  with a Magnitude (:func:`~openmeq.plotting.plot_scatter`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.catalog import event_id as get_event_id
from openmeq.plotting import plot_boxplot, plot_gutenberg_richter, plot_scatter

if TYPE_CHECKING:
    from obspy import Catalog

    from openmeq.paths import RunPaths


def _event_ml_table(catalog: Catalog) -> pd.DataFrame:
    """``event_id, ml, origin_time`` -- one row per event with a preferred Magnitude."""
    rows = []
    for ev in catalog:
        magnitude = ev.preferred_magnitude()
        if magnitude is None and ev.magnitudes:
            magnitude = ev.magnitudes[-1]
        if magnitude is None:
            continue
        origin = ev.preferred_origin()
        if origin is None and ev.origins:
            origin = ev.origins[0]
        rows.append(
            {
                "event_id": get_event_id(ev),
                "ml": magnitude.mag,
                "origin_time": pd.Timestamp(str(origin.time)) if origin is not None else pd.NaT,
            }
        )
    return pd.DataFrame(rows, columns=["event_id", "ml", "origin_time"])


def make_magnitude_diagnostics(paths: RunPaths, station_df: pd.DataFrame, catalog: Catalog) -> None:
    """Build every ``magnitude/diagnostics/`` output.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
        Only ``paths.diagnostics_dir("magnitude")`` is used.
    station_df : pandas.DataFrame
        :data:`openmeq.magnitude.pipeline.STATION_MAGNITUDES_COLUMNS`-shaped
        table (one row per station-event pair actually measured). May be
        empty.
    catalog : obspy.Catalog
        The full output catalogue (``magnitude/catalog_mag.xml``'s
        in-memory form), used for the per-event ML values (``ml_through_time``,
        ``magnitude_frequency``) and to compute station residuals against
        each event's own ML.
    """
    out_dir = Path(paths.diagnostics_dir("magnitude"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = station_df.copy() if station_df is not None else pd.DataFrame()
    if not df.empty:
        df["net_sta"] = df["net"].astype(str) + "." + df["sta"].astype(str)
        df["station_ml"] = pd.to_numeric(df["station_ml"], errors="coerce")
        included = df[df["included"].astype(bool)]
    else:
        included = df

    event_table = _event_ml_table(catalog)

    # -- ml_vs_distance.png --------------------------------------------------
    if not included.empty:
        ml_vs_distance_df = included[["distance_km", "station_ml", "net_sta"]].dropna(
            subset=["distance_km", "station_ml"]
        )
    else:
        ml_vs_distance_df = pd.DataFrame(columns=["distance_km", "station_ml", "net_sta"])
    plot_scatter(
        ml_vs_distance_df,
        out_dir / "ml_vs_distance.png",
        x="distance_km",
        y="station_ml",
        group_by="net_sta",
        xlabel="Hypocentral distance (km)",
        ylabel="Station ML",
        title="Station ML vs distance",
    )

    # -- station_residuals.png -----------------------------------------------
    if not included.empty and not event_table.empty:
        merged = included.merge(event_table[["event_id", "ml"]], on="event_id", how="inner")
        merged = merged.dropna(subset=["station_ml", "ml"])
        merged["residual"] = merged["station_ml"] - merged["ml"]
    else:
        merged = pd.DataFrame(columns=["net_sta", "residual"])
    plot_boxplot(
        merged,
        out_dir / "station_residuals.png",
        by="net_sta",
        column="residual",
        xlabel="Station (NET.STA)",
        ylabel="Station ML - event ML",
        title="Station ML residuals",
    )

    # -- magnitude_frequency.png (Gutenberg-Richter) --------------------------
    plot_gutenberg_richter(
        event_table["ml"].to_numpy(dtype=float) if not event_table.empty else np.array([]),
        out_dir / "magnitude_frequency.png",
    )

    # -- ml_through_time.png ---------------------------------------------------
    plot_scatter(
        event_table,
        out_dir / "ml_through_time.png",
        x="origin_time",
        y="ml",
        xlabel="Origin time (UTC)",
        ylabel="Event ML",
        title="ML over time",
    )
