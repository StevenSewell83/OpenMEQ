"""``relocate/diagnostics/extended/`` -- opt-in extended relocate QAQC
(OpenMEQ-02o.7): dt.cc-vs-dt.ct differential times, final hypoDD residuals
by data type, and per-iteration hypocentre shifts by weighting set.

Written by :func:`write_relocate_extended_diagnostics`, called from
:func:`openmeq.relocate.diagnostics.write_relocate_diagnostics` only when
``cfg.diagnostics.extended_enabled("relocate")`` is true -- the baseline
``relocate/diagnostics/`` figures are unchanged either way. Every figure
here is individually wrapped (:func:`_safe`): an exception logs one
``WARNING`` and moves on to the next figure, so extended QAQC can never fail
``openmeq relocate``.

Figures (all under ``relocate/diagnostics/extended/``):

- ``dtcc_vs_dtct.png`` / ``dtcc_vs_dtct.csv``: ``dt_cc - dt_ct`` for every
  ``(pair, station, phase)`` present in both files (cycle-skip detector).
- ``res_final_by_type.png``: ``hypoDD.res`` final residuals split CT-P /
  CT-S / CC-P / CC-S.
- ``iteration_shifts.png``: hypoDD.log's DX/DY/DZ/DT/OS vs. iteration with
  each ``cfg.hypodd.weighting_rows`` set shaded.

dt.ct vs dt.cc: comparing like with like
----------------------------------------
The two files store the differential time differently, so neither file's
raw number is compared directly:

- ``dt.ct`` (``ph2dt.f:698-701``: ``'(a1,1x,i9,1x,i9)'`` header,
  ``'(a7,1x,f7.3,1x,f7.3,1x,f6.4,1x,a1)'`` body) carries the two
  *absolute travel times* ``STA TT1 TT2 WGHT PHA``; hypoDD forms
  ``dt = TT1 - TT2`` itself (``getdata.f:485``). ``f7.3`` quantises each
  travel time to 1 ms, so ``dt_ct`` carries up to +-1 ms of rounding.
- ``dt.cc`` (``# ID1 ID2 OTC`` header, ``STA DT WGHT PHA`` body,
  free-format, ``getdata.f:369-402``) carries the differential time
  directly, and hypoDD subtracts the pair header's origin-time correction:
  ``dt = DT - OTC`` (``getdata.f:399``). A pair with ``OTC = -999`` is
  skipped by hypoDD (``getdata.f:374-378``) and is skipped here too.

So the compared quantity is ``(DT - OTC) - (TT1 - TT2)``, exactly the two
``dt_dt`` values hypoDD itself would use. This repo's own writer
(:func:`openmeq.relocate.xcorr.write_dt_cc`) always writes ``OTC = 0`` and
``DT = TT_i - (TT_j + pick2_corr)`` against the same catalogue origins
``phase.dat`` (and so ``dt.ct``) was built from, so the difference is just
``-pick2_corr``, bounded by ``cfg.hypodd.xcorr.maxlag_s`` (plus one sample).
Values piling up at those bounds, or a multi-modal distribution spaced by a
dominant period, flag cycle skips. Both files normally share the pair order
(``dt.cc`` reuses ``dt.ct``'s header order); a pair found only in swapped
order is still matched, with ``dt_ct`` negated.

``hypoDD.res`` (:func:`parse_hypodd_res`)
-----------------------------------------
Written once per cluster at the end of hypoDD's cluster loop
(``hypoDD.f:1265-1276``), fixed format
``'(a7,1x,f12.7,1x,i9,1x,i9,1x,i1,1x,f9.4,1x,f12.6,1x,f11.6,1x,f8.1)'``::

    STA  DT(s)  C1  C2  IDX  QUAL  RES(ms)  WT  OFFS(m)

``IDX`` is the data-type code set in ``getdata.f``: ``1`` = CC-P
(``:413``), ``2`` = CC-S (``:417``), ``3`` = CT-P (``:488``), ``4`` = CT-S
(``:492``). ``RES`` is ``dt_res * 1000`` with ``dt_res = dt_dt - dt_cal``
(``dtres.f:63``), i.e. already milliseconds; ``WT`` is the final a
posteriori weight (``weighting.f``; ``0`` = down-weighted out by the last
set's ``WRCC``/``WRCT``/``WDCC``/``WDCT`` cuts); ``OFFS`` is the pair's
inter-event distance in metres (``hypoDD.f:880``). The file is parsed by
fixed column, so a Fortran ``****`` overflow in a float field becomes
``nan`` rather than shifting every following field.

Because hypoDD re-``open``\\ s ``hypoDD.res`` inside its cluster loop
(``hypoDD.f:499``/``:1313``), a single ``CID=0`` run over several clusters
leaves only the *last* cluster's residuals on disk -- a hypoDD limitation.
The ``cfg.hypodd.parallel_clusters`` path does not have this problem:
:func:`openmeq.relocate.hypodd_runner._run_hypodd_clustered` concatenates
every ``_cluster_NNNN/hypoDD.res`` into ``relocate/work/hypoDD.res`` before
removing the per-cluster directories. :func:`_hypodd_res_paths` also falls
back to any surviving ``_cluster_*/hypoDD.res`` if the merged file is
missing. The figure title reports how many of ``hypoDD.reloc``'s clusters
the residuals actually cover, so a single-run partial file is visible.

Weighting-set boundaries for ``iteration_shifts.png``
-----------------------------------------------------
hypoDD makes the ``NITER`` column cumulative (``getinp.f:152-155``) and
selects set ``i`` for the first ``iter <= aiter(i)`` (``hypoDD.f:625-627``).
An iteration that removes air-quakes does not update locations and pushes
every later boundary back by one (``hypoDD.f:815-821``). hypoDD.log's
table row shows this: column 1 is the raw ``iter`` and column 2 (``str3``)
is the updated-iteration count ``iter - jiter``, printed blank for an
air-quake iteration (``hypoDD.f:989-994``). The set is therefore picked from
the updated count (``HypoddIterationRow.updated_iteration``). A blank row
counts as the previous updated count + 1, which is the value hypoDD used to
choose its weights before incrementing ``jiter``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openmeq.log import get_logger
from openmeq.plotting import save_fig, setup_style

if TYPE_CHECKING:
    from openmeq.config import Config, WeightingRow
    from openmeq.paths import RunPaths
    from openmeq.relocate.diagnostics import HypoddIterationRow

logger = get_logger("relocate.diagnostics")

#: ``hypoDD.res`` IDX code -> data-type label (``getdata.f:413/417/488/492``).
DATA_TYPE_BY_IDX = {1: "CC-P", 2: "CC-S", 3: "CT-P", 4: "CT-S"}

#: Plot order for ``res_final_by_type.png``.
_DATA_TYPES = ("CT-P", "CT-S", "CC-P", "CC-S")

#: ``hypoDD.res`` fixed-column slices, from ``hypoDD.f:1271-1272``'s
#: ``(a7,1x,f12.7,1x,i9,1x,i9,1x,i1,1x,f9.4,1x,f12.6,1x,f11.6,1x,f8.1)``.
_RES_COLUMNS = {
    "sta": slice(0, 7),
    "dt": slice(8, 20),
    "c1": slice(21, 30),
    "c2": slice(31, 40),
    "idx": slice(41, 42),
    "qual": slice(43, 52),
    "res": slice(53, 65),
    "wt": slice(66, 77),
    "offs": slice(78, 86),
}

_DTCC_CSV_COLUMNS = [
    "cuspid1",
    "cuspid2",
    "station",
    "phase",
    "dt_ct_s",
    "dt_cc_s",
    "otc_s",
    "diff_ms",
    "cc",
    "ct_weight",
    "pair_swapped",
]


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypoddResidual:
    """One ``hypoDD.res`` line (module docstring, "hypoDD.res").

    ``data_type`` is ``"CC-P"``/``"CC-S"``/``"CT-P"``/``"CT-S"`` from
    ``idx``. Float fields are ``nan`` where hypoDD printed an overflowed
    (``****``) value.
    """

    sta: str
    dt_s: float
    cuspid1: int
    cuspid2: int
    idx: int
    data_type: str
    qual: float
    res_ms: float
    weight: float
    offset_m: float


def _fortran_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_hypodd_res(path: Path | str) -> list[HypoddResidual]:
    """Parse a ``hypoDD.res`` final-residual file (``hypoDD.f:1265-1276``).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to ``hypoDD.res`` (``relocate/work/hypoDD.res``).

    Returns
    -------
    list of HypoddResidual
        One per data line, in file order. Blank lines are skipped. So is a
        line whose ``C1``/``C2``/``IDX`` field is not an integer or whose
        ``IDX`` is not ``1``-``4``, with a ``DEBUG`` log -- the file carries
        no header (the ``STA DT C1 ...`` header write is commented out in
        ``hypoDD.f:1269-1270``), so every non-blank line should be data.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    """
    path = Path(path)
    rows: list[HypoddResidual] = []
    with open(path, errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            cols = {name: line[sl] for name, sl in _RES_COLUMNS.items()}
            try:
                c1 = int(cols["c1"])
                c2 = int(cols["c2"])
                idx = int(cols["idx"])
            except ValueError:
                logger.debug("parse_hypodd_res: %s:%d: skipping unparseable line", path, lineno)
                continue
            data_type = DATA_TYPE_BY_IDX.get(idx)
            if data_type is None:
                logger.debug("parse_hypodd_res: %s:%d: unknown IDX %d", path, lineno, idx)
                continue
            rows.append(
                HypoddResidual(
                    sta=cols["sta"].strip(),
                    dt_s=_fortran_float(cols["dt"]),
                    cuspid1=c1,
                    cuspid2=c2,
                    idx=idx,
                    data_type=data_type,
                    qual=_fortran_float(cols["qual"]),
                    res_ms=_fortran_float(cols["res"]),
                    weight=_fortran_float(cols["wt"]),
                    offset_m=_fortran_float(cols["offs"]),
                )
            )
    return rows


def read_dt_ct_differentials(
    path: Path | str,
) -> dict[tuple[int, int, str, str], tuple[float, float]]:
    """``dt.ct`` -> ``{(cuspid1, cuspid2, station, phase): (TT1 - TT2, WGHT)}``.

    ``TT1 - TT2`` is hypoDD's own ``dt_dt`` for catalogue data
    (``getdata.f:485``). Malformed lines are skipped (this is a diagnostics
    reader -- :func:`openmeq.relocate.xcorr.parse_dt_ct_detail` is the strict
    one on the production path).
    """
    out: dict[tuple[int, int, str, str], tuple[float, float]] = {}
    current: tuple[int, int] | None = None
    for raw in Path(path).read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        if raw.startswith("#"):
            try:
                current = (int(parts[1]), int(parts[2]))
            except (IndexError, ValueError):
                current = None
            continue
        if current is None or len(parts) < 5:
            continue
        try:
            tt1, tt2, wght = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        out[(current[0], current[1], parts[0], parts[4])] = (tt1 - tt2, wght)
    return out


def read_dt_cc_differentials(
    path: Path | str,
) -> dict[tuple[int, int, str, str], tuple[float, float, float]]:
    """``dt.cc`` -> ``{(cuspid1, cuspid2, station, phase): (DT, OTC, CC)}``.

    The value hypoDD uses is ``DT - OTC`` (``getdata.f:399``); both are
    returned so the CSV can show them. Pairs whose header ``OTC`` is
    ``-999`` (hypoDD's "no origin-time correction, skip this pair",
    ``getdata.f:374-378``) are skipped, as are malformed lines.
    """
    out: dict[tuple[int, int, str, str], tuple[float, float, float]] = {}
    current: tuple[int, int, float] | None = None
    for raw in Path(path).read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        if raw.startswith("#"):
            try:
                c1, c2 = int(parts[1]), int(parts[2])
                otc = float(parts[3]) if len(parts) > 3 else 0.0
            except (IndexError, ValueError):
                current = None
                continue
            current = None if abs(otc + 999.0) < 0.001 else (c1, c2, otc)
            continue
        if current is None or len(parts) < 4:
            continue
        try:
            dt, cc = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        out[(current[0], current[1], parts[0], parts[3])] = (dt, current[2], cc)
    return out


def match_dtcc_dtct(dt_ct_path: Path | str, dt_cc_path: Path | str) -> pd.DataFrame:
    """Join ``dt.cc`` and ``dt.ct`` on ``(pair, station, phase)``.

    Returns a DataFrame with :data:`_DTCC_CSV_COLUMNS`, one row per
    ``dt.cc`` observation that also appears in ``dt.ct``, and
    ``diff_ms = ((DT - OTC) - (TT1 - TT2)) * 1000`` (module docstring, "dt.ct
    vs dt.cc"). A ``dt.cc`` pair found in ``dt.ct`` only with swapped event
    order is matched with ``dt_ct`` negated (``pair_swapped=True``).
    """
    ct = read_dt_ct_differentials(dt_ct_path)
    cc = read_dt_cc_differentials(dt_cc_path)
    rows = []
    for (c1, c2, sta, pha), (dt, otc, coeff) in cc.items():
        swapped = False
        ct_val = ct.get((c1, c2, sta, pha))
        if ct_val is None:
            rev = ct.get((c2, c1, sta, pha))
            if rev is None:
                continue
            ct_val = (-rev[0], rev[1])
            swapped = True
        dt_ct, ct_w = ct_val
        dt_cc = dt - otc
        rows.append(
            {
                "cuspid1": c1,
                "cuspid2": c2,
                "station": sta,
                "phase": pha,
                "dt_ct_s": dt_ct,
                "dt_cc_s": dt_cc,
                "otc_s": otc,
                "diff_ms": (dt_cc - dt_ct) * 1000.0,
                "cc": coeff,
                "ct_weight": ct_w,
                "pair_swapped": swapped,
            }
        )
    return pd.DataFrame(rows, columns=_DTCC_CSV_COLUMNS)


def _hypodd_res_paths(work_dir: Path) -> list[Path]:
    """``work_dir/hypoDD.res`` if present, else any surviving
    ``work_dir/_cluster_*/hypoDD.res`` (``parallel_clusters`` layout,
    :func:`openmeq.relocate.hypodd_runner._run_hypodd_clustered`)."""
    merged = work_dir / "hypoDD.res"
    if merged.is_file():
        return [merged]
    return sorted(p for p in work_dir.glob("_cluster_*/hypoDD.res") if p.is_file())


def _reloc_cluster_by_cuspid(path: Path) -> dict[int, int]:
    """``hypoDD.reloc`` CUSPID -> cluster id (field 24, ``hypoDD.f:1283-1298``);
    ``{}`` if missing/unreadable."""
    out: dict[int, int] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 24:
            continue
        try:
            out[int(parts[0])] = int(parts[23])
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# weighting sets
# ---------------------------------------------------------------------------


def iteration_weighting_sets(
    rows: list[HypoddIterationRow], weighting_rows: list[WeightingRow]
) -> list[int]:
    """0-based weighting-set index for each row of ``rows`` (in order).

    See the module docstring's "Weighting-set boundaries". A row without an
    ``updated_iteration`` (air-quake iteration) counts as the previous
    updated count + 1. Rows past the last cumulative boundary are put in
    the last set.
    """
    bounds = np.cumsum([max(int(w.niter), 0) for w in weighting_rows])
    out: list[int] = []
    prev_updated = 0
    for r in rows:
        updated = getattr(r, "updated_iteration", None)
        n = updated if updated is not None else prev_updated + 1
        if updated is not None:
            prev_updated = updated
        k = int(np.searchsorted(bounds, n, side="left")) if len(bounds) else 0
        out.append(min(k, max(len(bounds) - 1, 0)))
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    return med, float(np.median(np.abs(values - med)))


def _no_data(ax, text: str = "No data") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def plot_dtcc_vs_dtct(
    df: pd.DataFrame, path: Path, *, maxlag_s: float | None, s_maxlag_s: float | None = None
) -> None:
    """``dtcc_vs_dtct.png``: P and S histograms of ``diff_ms``, with
    median/MAD/n annotated and +-``maxlag_s`` bounds drawn (``s_maxlag_s``,
    when given, for the S panel).

    With ``maxlag_s`` set, the x range is fixed at +-3 x maxlag and values
    beyond it are clipped into the edge bins (so gross outliers stay
    visible as edge spikes without squashing the core of the
    distribution); median/MAD always use the unclipped values."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), layout="constrained")
    p_maxlag = maxlag_s
    for ax, phase in zip(axes, ("P", "S"), strict=True):
        maxlag_s = s_maxlag_s if phase == "S" and s_maxlag_s is not None else p_maxlag
        vals = df.loc[df["phase"] == phase, "diff_ms"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            _no_data(ax)
        else:
            if maxlag_s is not None and maxlag_s > 0:
                span = 3.0 * maxlag_s * 1000.0
                bins = np.linspace(-span, span, 61)
                shown = np.clip(vals, -span, span)
            else:
                bins, shown = 40, vals
            ax.hist(shown, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.5)
            med, mad = _median_mad(vals)
            text = f"n = {vals.size}\nmedian = {med:.2f} ms\nMAD = {mad:.2f} ms"
            if maxlag_s is not None:
                lim = maxlag_s * 1000.0
                n_out = int(np.sum(np.abs(vals) > lim))
                text += f"\n|diff| > maxlag: {n_out}"
            ax.text(
                0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )  # fmt: skip
        if maxlag_s is not None:
            lim = maxlag_s * 1000.0
            for x in (-lim, lim):
                ax.axvline(x, color="#D55E00", linestyle="--", linewidth=1.2)
        ax.set_xlabel("dt.cc - dt.ct (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{phase} phase")
    fig.suptitle(
        "Cross-correlation vs catalogue differential times, (DT - OTC) - (TT1 - TT2)"
        + ("; dashed = +-maxlag" if p_maxlag is not None else "")
    )
    save_fig(fig, path)


def plot_res_final_by_type(
    residuals: list[HypoddResidual], path: Path, *, coverage_note: str | None = None
) -> None:
    """``res_final_by_type.png``: 2x2 residual histograms (ms) for CT-P,
    CT-S, CC-P, CC-S. Outline = every observation; filled = those with a
    final weight > 0 (still used by the last iteration). n/median/RMS are
    for the weight > 0 set, plus a weighted RMS."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
    for ax, dtype in zip(axes.ravel(), _DATA_TYPES, strict=True):
        sub = [r for r in residuals if r.data_type == dtype and math.isfinite(r.res_ms)]
        if not sub:
            _no_data(ax)
            ax.set_title(dtype)
            ax.set_xlabel("Final residual (ms)")
            continue
        res_all = np.array([r.res_ms for r in sub])
        wt_all = np.array([r.weight for r in sub])
        used = np.isfinite(wt_all) & (wt_all > 0)
        bins = np.histogram_bin_edges(res_all, bins=40)
        ax.hist(res_all, bins=bins, histtype="step", color="#555555", label="all")
        if used.any():
            ax.hist(res_all[used], bins=bins, color="#0072B2", alpha=0.8, label="final weight > 0")
            r_used, w_used = res_all[used], wt_all[used]
            rms = float(np.sqrt(np.mean(r_used**2)))
            wrms = float(np.sqrt(np.sum(w_used * r_used**2) / np.sum(w_used)))
            text = (
                f"n = {used.sum()} / {len(sub)}\nmedian = {np.median(r_used):.2f} ms\n"
                f"RMS = {rms:.2f} ms\nweighted RMS = {wrms:.2f} ms"
            )
        else:
            text = f"n = 0 / {len(sub)} with weight > 0"
        ax.text(
            0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )  # fmt: skip
        ax.legend(loc="upper right")
        ax.set_title(dtype)
        ax.set_xlabel("Final residual (ms)")
        ax.set_ylabel("Count")
    title = "hypoDD final residuals by data type (hypoDD.res)"
    if coverage_note:
        title += f"\n{coverage_note}"
    fig.suptitle(title)
    save_fig(fig, path)


_SET_COLORS = ("#E8E8E8", "#FFFFFF")


def plot_iteration_shifts(
    rows: list[HypoddIterationRow], weighting_rows: list[WeightingRow], path: Path
) -> None:
    """``iteration_shifts.png``: three stacked panels -- DX/DY/DZ (m), DT
    (ms), OS (m) -- vs. iteration, with alternating shaded bands for each
    weighting set (:func:`iteration_weighting_sets`)."""
    setup_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True, layout="constrained")
    its = np.array([r.iteration for r in rows], dtype=float)

    def _series(attr: str) -> np.ndarray:
        return np.array(
            [getattr(r, attr) if getattr(r, attr) is not None else np.nan for r in rows],
            dtype=float,
        )

    panels = [
        (axes[0], [("DX", "dx_m"), ("DY", "dy_m"), ("DZ", "dz_m")], "Mean abs. shift (m)"),
        (axes[1], [("DT", "dt_ms")], "Origin-time shift (ms)"),
        (axes[2], [("OS", "os_m")], "Cluster centroid shift (m)"),
    ]
    for ax, series, ylabel in panels:
        for label, attr in series:
            ax.plot(its, _series(attr), marker="o", linewidth=1.2, label=label)
        ax.set_ylabel(ylabel)
        if len(series) > 1:
            ax.legend(loc="upper right")

    sets = iteration_weighting_sets(rows, weighting_rows) if weighting_rows else []
    if sets:
        start = 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or sets[i] != sets[start]:
                k = sets[start]
                x0, x1 = its[start] - 0.5, its[i - 1] + 0.5
                for ax in axes:
                    ax.axvspan(x0, x1, color=_SET_COLORS[k % 2], zorder=0, linewidth=0)
                    if k % 2 == 1:
                        ax.axvline(x0, color="#999999", linewidth=0.6, zorder=0)
                w = weighting_rows[k]
                axes[0].text(
                    (x0 + x1) / 2, 1.02, f"set {k + 1}\nCT {w.wtctp:g}/{w.wtcts:g}\n"
                    f"CC {w.wtccp:g}/{w.wtccs:g}", transform=axes[0].get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7,
                )  # fmt: skip
                start = i
        for ax in axes:
            ax.set_facecolor(_SET_COLORS[1])
    axes[-1].set_xlabel("Iteration (hypoDD.log IT)")
    fig.suptitle(
        "hypoDD per-iteration hypocentre adjustments by weighting set "
        "(band labels: P/S a priori weights)"
    )
    save_fig(fig, path)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _safe(name: str, fn: Callable[[], bool]) -> bool:
    """Run one figure writer; any exception -> WARNING, ``False``."""
    try:
        return bool(fn())
    except Exception as exc:  # noqa: BLE001 -- extended QAQC must never fail the step
        logger.warning(
            "relocate extended diagnostics: %s failed (%s: %s) -- skipping it.",
            name,
            type(exc).__name__,
            exc,
        )
        return False


def write_relocate_extended_diagnostics(
    cfg: Config,
    paths: RunPaths,
    *,
    iteration_rows: list[HypoddIterationRow] | None = None,
) -> list[str]:
    """Write ``relocate/diagnostics/extended/`` (module docstring).

    Does no gating itself -- callers check
    ``cfg.diagnostics.extended_enabled("relocate")`` first (as
    :func:`openmeq.relocate.diagnostics.write_relocate_diagnostics` does).
    A missing input file (``dt.ct``/``dt.cc``/``hypoDD.res``, or no
    iteration rows) skips just that figure with a ``WARNING``; any other
    error in one figure is logged and the rest are still written.

    Parameters
    ----------
    cfg : openmeq.config.Config
        Reads ``cfg.hypodd.xcorr.maxlag_s`` and ``cfg.hypodd.weighting_rows``.
    paths : openmeq.paths.RunPaths
    iteration_rows : list of HypoddIterationRow, optional
        Already-parsed hypoDD.log rows (the baseline diagnostics parse them
        once); parsed from ``relocate/work/hypoDD.log`` if ``None``.

    Returns
    -------
    list of str
        File names actually written, e.g. ``["dtcc_vs_dtct.png", ...]``.
    """
    out_dir = paths.diagnostics_dir("relocate") / "extended"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(paths.relocate_work_dir)
    written: list[str] = []

    def _dtcc() -> bool:
        dt_ct, dt_cc = Path(paths.dt_ct), Path(paths.dt_cc)
        missing = [p for p in (dt_ct, dt_cc) if not p.is_file()]
        if missing:
            logger.warning(
                "relocate extended diagnostics: %s missing -- skipping dtcc_vs_dtct.png/.csv.",
                ", ".join(str(p) for p in missing),
            )
            return False
        df = match_dtcc_dtct(dt_ct, dt_cc)
        df.to_csv(out_dir / "dtcc_vs_dtct.csv", index=False)
        written.append("dtcc_vs_dtct.csv")
        plot_dtcc_vs_dtct(
            df,
            out_dir / "dtcc_vs_dtct.png",
            maxlag_s=cfg.hypodd.xcorr.maxlag_s,
            s_maxlag_s=cfg.hypodd.xcorr.maxlag_for("S"),
        )
        written.append("dtcc_vs_dtct.png")
        return True

    def _res() -> bool:
        res_paths = _hypodd_res_paths(work_dir)
        if not res_paths:
            logger.warning(
                "relocate extended diagnostics: no hypoDD.res in %s -- skipping "
                "res_final_by_type.png.",
                work_dir,
            )
            return False
        residuals = [r for p in res_paths for r in parse_hypodd_res(p)]
        clusters = _reloc_cluster_by_cuspid(work_dir / "hypoDD.reloc")
        note = None
        if clusters:
            n_all = len(set(clusters.values()))
            covered = {
                clusters[c] for r in residuals for c in (r.cuspid1, r.cuspid2) if c in clusters
            }
            note = f"covers {len(covered)} of {n_all} hypoDD.reloc cluster(s)"
            if len(covered) < n_all:
                note += " (single CID=0 run: hypoDD keeps only the last cluster's hypoDD.res)"
        plot_res_final_by_type(residuals, out_dir / "res_final_by_type.png", coverage_note=note)
        written.append("res_final_by_type.png")
        return True

    def _shifts() -> bool:
        rows = iteration_rows
        if rows is None:
            from openmeq.relocate.diagnostics import parse_hypodd_log_iterations

            rows = parse_hypodd_log_iterations(work_dir / "hypoDD.log")
        if not rows:
            logger.warning(
                "relocate extended diagnostics: no hypoDD.log iteration rows -- skipping "
                "iteration_shifts.png."
            )
            return False
        plot_iteration_shifts(
            rows, list(cfg.hypodd.weighting_rows), out_dir / "iteration_shifts.png"
        )
        written.append("iteration_shifts.png")
        return True

    _safe("dtcc_vs_dtct", _dtcc)
    _safe("res_final_by_type", _res)
    _safe("iteration_shifts", _shifts)
    return written
