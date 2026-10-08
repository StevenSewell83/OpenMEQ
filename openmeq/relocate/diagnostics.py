"""``relocate/diagnostics/`` -- before/after maps, dt.cc CC distribution,
per-iteration residual/connectivity trends and the relative-shift
histogram (M10.5 / OpenMEQ-16d.5).

Plan key: M10.5 (PLAN.md §5's hypoDD diagnostics row: "Before/after map and
sections; dt.cc CC distribution; residuals by iteration; relative location
shift histogram"). OpenMEQ-16d.6 adds one more: hypoDD.log's per-iteration
``EV``/``AQ``/``CND`` columns are the main tuning signal
(docs/hypodd_tuning.md), so they are surfaced here too.

Written by :func:`write_relocate_diagnostics`, called once at the end of
:func:`openmeq.relocate.run_relocate` (M10.5's own orchestration, this
package's ``__init__.py``). Reads ``magnitude/catalog_mag.xml`` (the
pre-hypoDD catalogue) and ``relocate/catalog_dd.xml`` (M10.4's merged
output) directly from disk -- this module does not take a `Config`, only a
`~openmeq.paths.RunPaths`.

``hypoDD.log`` iteration-table parsing (:func:`parse_hypodd_log_iterations`)
----------------------------------------------------------------------------
``hypoDD.f`` (confirmed directly against the installed
``external/HYPODD/src/hypoDD/hypoDD.f:997-1170``, not just a doc page)
writes one summary header line (``"  IT   EV  CT  CC    RMSCT ..."``,
column set depends on ``ISOLV``/``IDATA``) **once**, right before
iteration 1's own data row, and then one bare data row per iteration
thereafter (no repeated header) -- both the real-run example bundled at
``external/HYPODD/examples/example1/hypoDD.log`` and the format strings
themselves confirm this. The data row's own field count does not match the
header's token count 1:1 either: two header words (``RMSCT``/``RMSCC``)
each cover *two* data columns (the value and its %-change from the
previous iteration), and the row's second field (labelled ``str3`` in the
Fortran, an internal solver-iteration count) prints as three blank spaces
-- vanishing when the line is split on whitespace -- whenever that
iteration removed >=1 air-quake (``hypoDD.f:989-994``). This parser
therefore:

1. Finds the *first* line whose first two whitespace-separated tokens are
   exactly ``"IT"``/``"EV"`` and reads off which optional columns
   (``CT``/``CC``/``CND``) are present, fixing this file's own column
   schema (`ISOLV`/`IDATA` are constant for the whole run).
2. Scans every remaining line in the file (data rows are not necessarily
   contiguous with the header -- lots of other stdout-style text
   intervenes between iterations) for one whose first token parses as an
   integer and whose total token count matches the schema's expected count
   either with or without the blank ``str3`` field.
3. Silently skips (this is a stdout-style human-readable log, not a
   versioned file format, per this issue's own design note -- not a bug to
   raise on) every other line, including the header/units lines
   themselves, any of hypoDD's many other numeric-looking status lines
   (residual summaries, weighting-parameter dumps, cluster event-id
   listings, ...), by relying on the token-count check.

If no header is ever found (or the file doesn't exist / can't be read),
`parse_hypodd_log_iterations` returns an empty list rather than raising --
:func:`write_relocate_diagnostics` treats that as "parsing failed", logs
one ``WARNING`` and skips only the two plots built from it (never fails
the whole diagnostics step, let alone ``openmeq relocate`` itself).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import event_id as _event_id
from openmeq.catalog import get_extra, read_catalog
from openmeq.log import get_logger
from openmeq.plotting import plot_hist, plot_map_sections, plot_timeseries
from openmeq.wells import load_wells

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate.diagnostics")

#: Header tokens hypoDD.f always writes first (`hypoDD.f:998` et al.).
_HEADER_START = ("IT", "EV")


@dataclass(frozen=True)
class HypoddIterationRow:
    """One parsed row of hypoDD.log's per-iteration summary table.

    Fields absent from this run's own column schema (``ct_pct``/``cc_pct``
    if only one of catalog/cross-correlation data was used; ``cnd`` for
    ``ISOLV=1``/SVD, which never reports it) are ``None``, never a
    fabricated ``0``/``nan``.
    """

    iteration: int
    ev_pct: float | None
    ct_pct: float | None
    cc_pct: float | None
    rmsct_ms: float | None
    rmscc_ms: float | None
    rmsst_ms: float | None
    dx_m: float | None
    dy_m: float | None
    dz_m: float | None
    dt_ms: float | None
    os_m: float | None
    aq: float | None
    cnd: float | None
    #: The row's second column (``str3``, ``hypoDD.f:989-994``): the
    #: updated-iteration count ``iter - jiter``, or ``None`` when hypoDD
    #: printed it blank (an air-quake iteration, no location update). Used
    #: by :mod:`openmeq.relocate.extended_diagnostics` to map rows to
    #: weighting sets.
    updated_iteration: int | None = None


def _value_schema(header_tokens: list[str]) -> list[str]:
    """Ordered value-column names (excluding ``IT``/``str3``) for one
    header token list, per `hypoDD.f`'s own format-string variants."""
    has_ct = "CT" in header_tokens
    has_cc = "CC" in header_tokens
    has_cnd = "CND" in header_tokens

    names: list[str] = ["ev"]
    if has_ct:
        names.append("ct")
    if has_cc:
        names.append("cc")
    if has_ct:
        names += ["rmsct", "rmsct_pct"]
    if has_cc:
        names += ["rmscc", "rmscc_pct"]
    names += ["rmsst", "dx", "dy", "dz", "dt", "os", "aq"]
    if has_cnd:
        names.append("cnd")
    return names


def parse_hypodd_log_iterations(path: Path | str) -> list[HypoddIterationRow]:
    """Best-effort parse of ``hypoDD.log``'s per-iteration summary table.

    See the module docstring for the full parsing strategy. Never raises on
    a malformed/missing file -- returns ``[]``.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to ``hypoDD.log`` (``relocate/work/hypoDD.log``).

    Returns
    -------
    list of HypoddIterationRow, sorted by ``iteration``.
    """
    path = Path(path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        logger.debug("parse_hypodd_log_iterations: could not read %s: %s", path, exc)
        return []

    lines = text.splitlines()

    value_names: list[str] | None = None
    for raw in lines:
        tokens = raw.split()
        if tokens[:2] == list(_HEADER_START):
            value_names = _value_schema(tokens)
            break

    if value_names is None:
        return []

    expected_with_str3 = len(value_names) + 2  # IT + str3 + values
    expected_without_str3 = len(value_names) + 1  # IT + values (str3 blank)

    rows: dict[int, HypoddIterationRow] = {}
    for raw in lines:
        tokens = raw.split()
        if not tokens:
            continue
        try:
            iteration = int(tokens[0])
        except ValueError:
            continue

        updated_iteration: int | None = None
        if len(tokens) == expected_with_str3:
            value_tokens = tokens[2:]
            try:
                updated_iteration = int(tokens[1])
            except ValueError:
                updated_iteration = None
        elif len(tokens) == expected_without_str3:
            value_tokens = tokens[1:]
        else:
            continue

        try:
            values = [float(v) for v in value_tokens]
        except ValueError:
            continue

        parsed = dict(zip(value_names, values, strict=True))
        rows[iteration] = HypoddIterationRow(
            iteration=iteration,
            ev_pct=parsed.get("ev"),
            ct_pct=parsed.get("ct"),
            cc_pct=parsed.get("cc"),
            rmsct_ms=parsed.get("rmsct"),
            rmscc_ms=parsed.get("rmscc"),
            rmsst_ms=parsed.get("rmsst"),
            dx_m=parsed.get("dx"),
            dy_m=parsed.get("dy"),
            dz_m=parsed.get("dz"),
            dt_ms=parsed.get("dt"),
            os_m=parsed.get("os"),
            aq=parsed.get("aq"),
            cnd=parsed.get("cnd"),
            updated_iteration=updated_iteration,
        )

    return [rows[k] for k in sorted(rows)]


def _read_dt_cc_cc_values(path: Path | str) -> list[float]:
    """``STA DT WEIGHT PHASE`` lines' ``WEIGHT`` (the CC value,
    ``openmeq.relocate.xcorr.write_dt_cc``'s own free-format writer) from
    ``dt.cc``, ignoring ``#``-prefixed pair headers. ``[]`` if the file
    doesn't exist (never raises -- xcorr may legitimately have produced no
    output, e.g. no chunk waveform data at all).
    """
    path = Path(path)
    if not path.is_file():
        return []
    values: list[float] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            values.append(float(parts[2]))
        except ValueError:
            continue
    return values


def _dist_3d_km(
    lat1: float, lon1: float, depth_km1: float, lat2: float, lon2: float, depth_km2: float
) -> float:
    horiz_m, _az, _baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    dz_km = depth_km1 - depth_km2
    return math.sqrt((horiz_m / 1000.0) ** 2 + dz_km**2)


@dataclass
class RelocateDiagnosticsResult:
    """What one `write_relocate_diagnostics` call did."""

    diagnostics_dir: Path
    n_relocated: int
    n_not_linked: int
    n_cc_observations: int
    iteration_rows: list[HypoddIterationRow] = field(default_factory=list)
    iteration_plots_written: bool = False
    #: File names written under ``diagnostics/extended/`` (OpenMEQ-02o.7);
    #: empty when extended QAQC is disabled.
    extended_written: list[str] = field(default_factory=list)


_SUMMARY_COLUMNS = [
    "event_id",
    "hypodd_status",
    "lat_before",
    "lon_before",
    "depth_km_before",
    "lat_after",
    "lon_after",
    "depth_km_after",
    "relative_shift_km",
    "hypodd_mag",
    "hypodd_nccp",
    "hypodd_nccs",
    "hypodd_nctp",
    "hypodd_ncts",
    "hypodd_rcc",
    "hypodd_rct",
    "hypodd_cluster_id",
]


def write_relocate_diagnostics(
    paths: RunPaths, cfg: Config | None = None
) -> RelocateDiagnosticsResult:
    """Write ``relocate/diagnostics/`` (PLAN.md §5's hypoDD row, plus
    OpenMEQ-16d.6's EV/AQ/CND note).

    Reads ``magnitude/catalog_mag.xml`` (pre-hypoDD) and
    ``relocate/catalog_dd.xml`` (M10.4's merged output, post-hypoDD)
    directly -- both must already exist (raises `FileNotFoundError` with a
    clear message otherwise; `openmeq.relocate.run_relocate` always calls
    this after a successful `run_hypodd`, so both are guaranteed present in
    the normal pipeline).

    Writes:

    - ``locations_before.png`` / ``locations_after.png``: map + E-W/N-S
      section (`openmeq.plotting.plot_map_sections`, reused verbatim, not
      reimplemented) of every relocated event's pre-/post-hypoDD location;
      the "after" map is coloured by `relative_shift_km`.
    - ``dt_cc_cc_histogram.png``: histogram of every surviving ``dt.cc``
      observation's CC value.
    - ``relative_shift_hist.png``: histogram of each relocated event's
      pre--post 3D distance (km).
    - ``iteration_rms.png``: RMSCT/RMSCC/RMSST vs. iteration, parsed from
      ``relocate/work/hypoDD.log`` (`parse_hypodd_log_iterations`) --
      skipped, with a logged ``WARNING``, if parsing fails.
    - ``iteration_connectivity.png`` / ``iteration_aq_cnd.png``
      (OpenMEQ-16d.6): EV/CT/CC connectivity percentages, and AQ
      (air-quakes removed)/CND (condition number) vs. iteration --
      same skip-on-parse-failure behaviour as ``iteration_rms.png``.
    - ``summary.csv``: one row per event in the pre-hypoDD catalogue
      (ARCHITECTURE.md's "never dropped" convention), before/after
      location, relative shift, and the per-event hypoDD QC extras
      (§6h: ``hypodd_mag``/``hypodd_nccp``/.../``hypodd_cluster_id``).

    If ``cfg`` is given and ``cfg.diagnostics.relocate.xcorr_plots.enabled``,
    also writes ``diagnostics/xcorr_pairs/`` (obspy ``xcorr_pick_correction``
    plots for a sample of pairs, `openmeq.relocate.xcorr_plots`).

    If ``cfg`` is given and ``cfg.diagnostics.extended_enabled("relocate")``
    is true, also writes ``diagnostics/extended/``
    (`openmeq.relocate.extended_diagnostics.write_relocate_extended_diagnostics`,
    OpenMEQ-02o.7) -- never raises from there; the baseline set above is
    identical either way.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
    cfg : openmeq.config.Config, optional
        Needed for the extended set and the ``diagnostics.wells`` overlay;
        ``None`` writes the baseline only, without wells.

    Returns
    -------
    RelocateDiagnosticsResult
    """
    pre_path = Path(paths.catalog_mag_xml)
    post_path = Path(paths.catalog_dd_xml)
    if not pre_path.is_file():
        raise FileNotFoundError(
            f"{pre_path} does not exist -- write_relocate_diagnostics needs the pre-hypoDD "
            "catalogue (magnitude/catalog_mag.xml). Run `openmeq magnitude` first."
        )
    if not post_path.is_file():
        raise FileNotFoundError(
            f"{post_path} does not exist -- write_relocate_diagnostics needs M10.4's merged "
            "output (relocate/catalog_dd.xml). Run openmeq.relocate.hypodd_runner.run_hypodd "
            "first."
        )

    pre_catalog = read_catalog(pre_path)
    post_catalog = read_catalog(post_path)

    pre_by_event = {_event_id(ev): ev for ev in pre_catalog}

    diagnostics_dir = paths.diagnostics_dir("relocate")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    before_rows: list[dict] = []
    after_rows: list[dict] = []
    shift_values: list[float] = []
    n_relocated = 0
    n_not_linked = 0

    for post_event in post_catalog:
        eid = _event_id(post_event)
        status = get_extra(post_event, "hypodd_status", default=None)
        pre_event = pre_by_event.get(eid)
        pre_origin = pre_event.preferred_origin() if pre_event is not None else None

        row: dict = {
            "event_id": eid,
            "hypodd_status": status,
            "lat_before": None,
            "lon_before": None,
            "depth_km_before": None,
            "lat_after": None,
            "lon_after": None,
            "depth_km_after": None,
            "relative_shift_km": None,
            "hypodd_mag": None,
            "hypodd_nccp": None,
            "hypodd_nccs": None,
            "hypodd_nctp": None,
            "hypodd_ncts": None,
            "hypodd_rcc": None,
            "hypodd_rct": None,
            "hypodd_cluster_id": None,
        }

        if pre_origin is not None and pre_origin.latitude is not None:
            row["lat_before"] = pre_origin.latitude
            row["lon_before"] = pre_origin.longitude
            row["depth_km_before"] = (
                pre_origin.depth / 1000.0 if pre_origin.depth is not None else None
            )

        if status == "relocated":
            n_relocated += 1
            post_origin = post_event.preferred_origin()
            row["lat_after"] = post_origin.latitude
            row["lon_after"] = post_origin.longitude
            row["depth_km_after"] = (
                post_origin.depth / 1000.0 if post_origin.depth is not None else None
            )
            row["hypodd_mag"] = get_extra(post_origin, "hypodd_mag")
            row["hypodd_nccp"] = get_extra(post_origin, "hypodd_nccp")
            row["hypodd_nccs"] = get_extra(post_origin, "hypodd_nccs")
            row["hypodd_nctp"] = get_extra(post_origin, "hypodd_nctp")
            row["hypodd_ncts"] = get_extra(post_origin, "hypodd_ncts")
            row["hypodd_rcc"] = get_extra(post_origin, "hypodd_rcc")
            row["hypodd_rct"] = get_extra(post_origin, "hypodd_rct")
            row["hypodd_cluster_id"] = get_extra(post_origin, "hypodd_cluster_id")

            if row["lat_before"] is not None:
                shift_km = _dist_3d_km(
                    row["lat_before"],
                    row["lon_before"],
                    row["depth_km_before"],
                    row["lat_after"],
                    row["lon_after"],
                    row["depth_km_after"],
                )
                row["relative_shift_km"] = shift_km
                shift_values.append(shift_km)

                before_rows.append(
                    {
                        "event_id": eid,
                        "lat": row["lat_before"],
                        "lon": row["lon_before"],
                        "depth_km": row["depth_km_before"],
                    }
                )
                after_rows.append(
                    {
                        "event_id": eid,
                        "lat": row["lat_after"],
                        "lon": row["lon_after"],
                        "depth_km": row["depth_km_after"],
                        "relative_shift_km": shift_km,
                    }
                )
        else:
            n_not_linked += 1

        summary_rows.append(row)

    stations_df = None
    if paths.stations_csv.exists():
        stations_df = pd.read_csv(paths.stations_csv)

    wells = load_wells(cfg)

    before_df = pd.DataFrame(before_rows, columns=["event_id", "lat", "lon", "depth_km"])
    after_df = pd.DataFrame(
        after_rows, columns=["event_id", "lat", "lon", "depth_km", "relative_shift_km"]
    )

    plot_map_sections(
        before_df,
        stations_df,
        diagnostics_dir / "locations_before.png",
        title="Pre-hypoDD event locations",
        wells=wells,
    )
    plot_map_sections(
        after_df,
        stations_df,
        diagnostics_dir / "locations_after.png",
        color_by="relative_shift_km",
        cbar_label="Relative shift (km)",
        title="Post-hypoDD (relocated) event locations (colour: relative shift)",
        wells=wells,
    )

    cc_values = _read_dt_cc_cc_values(paths.dt_cc)
    plot_hist(
        cc_values,
        diagnostics_dir / "dt_cc_cc_histogram.png",
        xlabel="dt.cc cross-correlation coefficient",
        title="dt.cc CC-value distribution",
        bins=30,
    )

    plot_hist(
        shift_values,
        diagnostics_dir / "relative_shift_hist.png",
        xlabel="Relative location shift, pre-hypoDD -> hypoDD (km)",
        title="Relative location shift",
        bins=30,
    )

    iteration_rows: list[HypoddIterationRow] = []
    iteration_plots_written = False
    hypodd_log_path = Path(paths.relocate_work_dir) / "hypoDD.log"
    try:
        iteration_rows = parse_hypodd_log_iterations(hypodd_log_path)
    except Exception as exc:  # noqa: BLE001 -- diagnostics-only, must never fail the step
        logger.warning(
            "Could not parse %s for the per-iteration diagnostics: %s: %s -- skipping "
            "iteration_rms.png/iteration_connectivity.png/iteration_aq_cnd.png.",
            hypodd_log_path,
            type(exc).__name__,
            exc,
        )
        iteration_rows = []

    if not iteration_rows:
        logger.warning(
            "No per-iteration summary table found in %s -- skipping "
            "iteration_rms.png/iteration_connectivity.png/iteration_aq_cnd.png (this is "
            "expected if hypoDD was skipped as unchanged, e.g. --force was not needed, or if "
            "hypoDD.log's stdout-style table format changed -- see this module's docstring).",
            hypodd_log_path,
        )
    else:
        rms_records = []
        connectivity_records = []
        aq_cnd_records = []
        for r in iteration_rows:
            if r.rmsct_ms is not None:
                rms_records.append(
                    {"iteration": r.iteration, "series": "RMSCT (ms)", "value": r.rmsct_ms}
                )
            if r.rmscc_ms is not None:
                rms_records.append(
                    {"iteration": r.iteration, "series": "RMSCC (ms)", "value": r.rmscc_ms}
                )
            if r.rmsst_ms is not None:
                rms_records.append(
                    {"iteration": r.iteration, "series": "RMSST (ms)", "value": r.rmsst_ms}
                )
            if r.ev_pct is not None:
                connectivity_records.append(
                    {"iteration": r.iteration, "series": "EV (%)", "value": r.ev_pct}
                )
            if r.ct_pct is not None:
                connectivity_records.append(
                    {"iteration": r.iteration, "series": "CT (%)", "value": r.ct_pct}
                )
            if r.cc_pct is not None:
                connectivity_records.append(
                    {"iteration": r.iteration, "series": "CC (%)", "value": r.cc_pct}
                )
            if r.aq is not None:
                aq_cnd_records.append(
                    {"iteration": r.iteration, "series": "AQ (count)", "value": r.aq}
                )
            if r.cnd is not None:
                aq_cnd_records.append({"iteration": r.iteration, "series": "CND", "value": r.cnd})

        plot_timeseries(
            pd.DataFrame(rms_records, columns=["iteration", "series", "value"]),
            diagnostics_dir / "iteration_rms.png",
            x="iteration",
            y="value",
            group_by="series",
            xlabel="Iteration",
            ylabel="RMS (ms)",
            title="hypoDD residual RMS by iteration",
        )
        plot_timeseries(
            pd.DataFrame(connectivity_records, columns=["iteration", "series", "value"]),
            diagnostics_dir / "iteration_connectivity.png",
            x="iteration",
            y="value",
            group_by="series",
            xlabel="Iteration",
            ylabel="Still connected (%)",
            title="hypoDD event/link connectivity by iteration",
        )
        plot_timeseries(
            pd.DataFrame(aq_cnd_records, columns=["iteration", "series", "value"]),
            diagnostics_dir / "iteration_aq_cnd.png",
            x="iteration",
            y="value",
            group_by="series",
            xlabel="Iteration",
            ylabel="AQ (count) / CND",
            title="Air-quakes removed (AQ) and LSQR condition number (CND) by iteration "
            "(OpenMEQ-16d.6 tuning signal, docs/hypodd_tuning.md)",
        )
        iteration_plots_written = True

    df = pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)
    df.to_csv(diagnostics_dir / "summary.csv", index=False, columns=_SUMMARY_COLUMNS)

    extended_written: list[str] = []
    if cfg is not None and cfg.diagnostics.extended_enabled("relocate"):
        from openmeq.relocate.extended_diagnostics import write_relocate_extended_diagnostics

        try:
            extended_written = write_relocate_extended_diagnostics(
                cfg, paths, iteration_rows=iteration_rows
            )
        except Exception as exc:  # noqa: BLE001 -- extended QAQC must never fail the step
            logger.warning("relocate extended diagnostics failed: %s: %s", type(exc).__name__, exc)

    if cfg is not None and cfg.diagnostics.relocate.xcorr_plots.enabled:
        from openmeq.relocate.xcorr_plots import write_xcorr_plots

        try:
            write_xcorr_plots(cfg, paths)
        except Exception as exc:  # noqa: BLE001 -- optional QC plots must never fail the step
            logger.warning("relocate xcorr plots failed: %s: %s", type(exc).__name__, exc)

    return RelocateDiagnosticsResult(
        diagnostics_dir=diagnostics_dir,
        n_relocated=n_relocated,
        n_not_linked=n_not_linked,
        n_cc_observations=len(cc_values),
        iteration_rows=iteration_rows,
        iteration_plots_written=iteration_plots_written,
        extended_written=extended_written,
    )
