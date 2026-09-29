"""Run-level ``diagnostics/`` -- the cross-step ``event_funnel`` (OpenMEQ-02o.8).

Plan key: OpenMEQ-02o.8 (the ``run`` entry of ``cfg.diagnostics``'s opt-in
extended QAQC set, :data:`openmeq.config.DIAGNOSTIC_STEPS`). Written by
:func:`make_event_funnel`, called at the end of ``openmeq run``
(:func:`openmeq.cli.run_all`, including after a failed partial run) when
``cfg.diagnostics.extended_enabled("run")`` is true, and on demand by
``openmeq diagnostics funnel -c config.yaml`` (never gated).

Outputs (under :attr:`openmeq.paths.RunPaths.run_diagnostics_dir`):

- ``event_funnel.csv``: one row per stage (:data:`FUNNEL_COLUMNS`);
- ``event_funnel.png``: a horizontal bar funnel of the same rows, each bar
  annotated with its count and its percentage of ``detections``.

Design decisions
-----------------
**1. On-disk outputs only.** Every count is re-derived from the files each
step already writes (``detections/detections.csv``, ``chunks/<id>/<id>.mseed``,
``pick/catalog_picks.xml``, ``locate/catalog_auto.xml``/``summary.csv``,
``review/``, ``magnitude/catalog_mag.xml``, ``relocate/catalog_dd.xml``) --
never from in-memory step results -- so the funnel is identical whether
produced at the end of ``openmeq run`` or standalone later, and reflects a
partial run (``--from``/``--to``, or a failed step) exactly as it sits on
disk.

**2. A missing stage is "not run", never zero.** A stage whose defining
output file does not exist gets ``status="not run"`` and an empty
``n_events`` (drawn as a hatched, unlabelled bar), so "the step never ran"
cannot be mistaken for "the step ran and kept nothing". ``RunPaths.ensure``
creates every step's *directory* up front, so only files count as evidence.

**3. Stage definitions.**

- ``chunked``: events with a raw chunk file (``RunPaths.chunk_file``).
- ``picked``: events in ``catalog_picks.xml`` with >=1 non-rejected pick;
  total P/S counts go in ``detail``.
- ``located_pass_a``: events that got past pass A, i.e. whose preferred
  ``catalog_auto.xml`` origin was produced by pass B
  (``method_id`` ``smi:openmeq/nlloc/passB`` -- a pass-A rejection keeps its
  pass-A origin, :func:`openmeq.locate.finalize.finalize_event`). Without
  ``catalog_auto.xml`` this falls back to counting ``locate/passB/<id>/``
  directories.
- ``located_pass_b``: ``summary.csv`` rows with ``status == "accepted"`` (the
  final locate decision); the rejected count goes in ``detail``.
- ``reviewed``: accepted events in ``review/catalog_final.xml``; the number
  of ``review/events/*.xml`` files goes in ``detail``.
- ``magnitude``: events in ``catalog_mag.xml`` with >=1 ``ML`` magnitude.
- ``relocated``: events with ``hypodd_status == "relocated"`` in
  ``catalog_dd.xml`` (:func:`openmeq.relocate.hypodd_runner.merge_relocations`);
  ``detail`` gives the accepted events carried over un-relocated
  (``"not_linked"`` and not rejected).

**4. Never fails the caller.** :func:`make_event_funnel` catches everything,
logs one ``WARNING`` and returns ``None`` -- a diagnostic must never turn a
successful (or already-failed) ``openmeq run`` into a different outcome.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openmeq.catalog import get_extra, read_catalog
from openmeq.log import get_logger

if TYPE_CHECKING:
    from obspy import Catalog
    from obspy.core.event import Event

    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("run")

#: ``diagnostics/event_funnel.csv``'s exact column order.
FUNNEL_COLUMNS = ["stage", "label", "status", "n_events", "pct_of_detections", "detail"]

STATUS_OK = "ok"
STATUS_NOT_RUN = "not run"

_PASS_B_METHOD_ID = "smi:openmeq/nlloc/passB"
_EVENT_RESOURCE_PREFIX = "smi:openmeq/event/"


@dataclasses.dataclass
class FunnelStage:
    """One ``event_funnel.csv`` row (``n_events`` is ``None`` when not run)."""

    stage: str
    label: str
    n_events: int | None
    detail: str = ""

    @property
    def status(self) -> str:
        return STATUS_NOT_RUN if self.n_events is None else STATUS_OK


# ---------------------------------------------------------------------------
# per-stage counters (each reads one step's on-disk output)
# ---------------------------------------------------------------------------


def _eid(event: Event) -> str:
    rid = str(event.resource_id)
    return rid[len(_EVENT_RESOURCE_PREFIX) :] if rid.startswith(_EVENT_RESOURCE_PREFIX) else rid


def _is_rejected(event: Event) -> bool:
    origin = event.preferred_origin()
    return origin is not None and origin.evaluation_status == "rejected"


def _read(path: Path) -> Catalog | None:
    return read_catalog(path) if Path(path).is_file() else None


def _detections(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("detections", "Detections", None)
    if paths.detections_csv.is_file():
        stage.n_events = len(pd.read_csv(paths.detections_csv, usecols=["event_id"]))
    return stage


def _chunked(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("chunked", "Chunked", None)
    chunk_dir = paths.chunk_dir
    n = 0
    if chunk_dir.is_dir():
        n = sum(1 for d in chunk_dir.iterdir() if d.is_dir() and paths.chunk_file(d.name).is_file())
    if n or paths.chunk_completeness_csv.is_file():
        stage.n_events = n
    return stage


def _picked(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("picked", "Picked (>=1 pick)", None)
    cat = _read(paths.catalog_picks_xml)
    if cat is None:
        return stage
    n_events = n_p = n_s = 0
    for ev in cat:
        picks = [p for p in ev.picks if p.evaluation_status != "rejected"]
        if picks:
            n_events += 1
        for p in picks:
            hint = (p.phase_hint or "").upper()
            n_p += hint.startswith("P")
            n_s += hint.startswith("S")
    stage.n_events = n_events
    stage.detail = f"P={n_p}; S={n_s}"
    return stage


def _located(paths: RunPaths) -> tuple[FunnelStage, FunnelStage]:
    pass_a = FunnelStage("located_pass_a", "Located, pass A", None)
    pass_b = FunnelStage("located_pass_b", "Located, pass B (accepted)", None)

    auto = _read(paths.catalog_auto_xml)
    if auto is not None:
        pass_a.n_events = sum(
            1
            for ev in auto
            if ev.preferred_origin() is not None
            and str(ev.preferred_origin().method_id) == _PASS_B_METHOD_ID
        )
    elif paths.locate_summary_csv.is_file() and paths.passB_dir.is_dir():
        pass_a.n_events = sum(1 for d in paths.passB_dir.iterdir() if d.is_dir())
        pass_a.detail = "from locate/passB/ directories (no catalog_auto.xml)"

    if paths.locate_summary_csv.is_file():
        status = pd.read_csv(paths.locate_summary_csv, usecols=["status"])["status"]
        pass_b.n_events = int((status == "accepted").sum())
        pass_b.detail = f"rejected={int((status == 'rejected').sum())}"
    elif auto is not None:
        n_rej = sum(_is_rejected(ev) for ev in auto)
        pass_b.n_events = len(auto) - n_rej
        pass_b.detail = f"rejected={n_rej}"
    return pass_a, pass_b


def _reviewed(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("reviewed", "Reviewed (final catalogue)", None)
    n_reviewed = (
        sum(1 for _ in paths.review_events_dir.glob("*.xml"))
        if paths.review_events_dir.is_dir()
        else 0
    )
    final = _read(paths.catalog_final_xml)
    if final is not None:
        stage.n_events = sum(not _is_rejected(ev) for ev in final)
        stage.detail = f"reviewed={n_reviewed}"
    elif n_reviewed:
        stage.detail = f"reviewed={n_reviewed}; not compiled (no catalog_final.xml)"
    return stage


def _magnitude(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("magnitude", "With ML", None)
    cat = _read(paths.catalog_mag_xml)
    if cat is not None:
        stage.n_events = sum(
            any((m.magnitude_type or "").upper() == "ML" for m in ev.magnitudes) for ev in cat
        )
    return stage


def _relocated(paths: RunPaths) -> FunnelStage:
    stage = FunnelStage("relocated", "Relocated (hypoDD)", None)
    cat = _read(paths.catalog_dd_xml)
    if cat is None:
        return stage
    n_reloc = n_carried = 0
    for ev in cat:
        status = get_extra(ev, "hypodd_status", default=None)
        if status == "relocated":
            n_reloc += 1
        elif not _is_rejected(ev):
            n_carried += 1
    stage.n_events = n_reloc
    stage.detail = f"carried_over={n_carried}"
    return stage


def collect_funnel(paths: RunPaths) -> list[FunnelStage]:
    """Count events at every pipeline stage from ``paths``' on-disk outputs.

    Returns
    -------
    list[FunnelStage]
        In pipeline order; ``n_events`` is ``None`` for a stage not run.
    """
    pass_a, pass_b = _located(paths)
    return [
        _detections(paths),
        _chunked(paths),
        _picked(paths),
        pass_a,
        pass_b,
        _reviewed(paths),
        _magnitude(paths),
        _relocated(paths),
    ]


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def _funnel_frame(stages: list[FunnelStage]) -> pd.DataFrame:
    n_det = stages[0].n_events
    rows = []
    for s in stages:
        pct = (
            round(100.0 * s.n_events / n_det, 2)
            if s.n_events is not None and n_det  # neither None nor 0
            else None
        )
        rows.append(
            {
                "stage": s.stage,
                "label": s.label,
                "status": s.status,
                "n_events": s.n_events,
                "pct_of_detections": pct,
                "detail": s.detail,
            }
        )
    df = pd.DataFrame(rows, columns=FUNNEL_COLUMNS)
    df["n_events"] = df["n_events"].astype("Int64")
    return df


def _plot_funnel(df: pd.DataFrame, path: Path) -> None:
    from openmeq.plotting import save_fig, setup_style

    setup_style()
    import matplotlib.pyplot as plt

    n = len(df)
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * n + 1.5))
    counts = [0 if pd.isna(v) else int(v) for v in df["n_events"]]
    x_max = max(max(counts, default=0), 1)
    y = list(range(n))[::-1]  # first stage at the top

    for yi, (_, row), count in zip(y, df.iterrows(), counts, strict=True):
        if row["status"] == STATUS_NOT_RUN:
            ax.barh(yi, x_max, color="none", edgecolor="#BBBBBB", hatch="//", linewidth=0.5)
            ax.text(
                0.02 * x_max,
                yi,
                "not run",
                va="center",
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
            continue
        ax.barh(yi, count, color="#0072B2", edgecolor="white", linewidth=0.5)
        label = f"{count}"
        if not pd.isna(row["pct_of_detections"]):
            label += f"  ({row['pct_of_detections']:.1f}%)"
        if row["detail"]:
            label += f"  [{row['detail']}]"
        ax.text(count + 0.01 * x_max, yi, label, va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, x_max * 1.6)  # room for the right-hand annotations
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Events (% of detections)")
    ax.set_title("Event funnel")
    save_fig(fig, path)


def make_event_funnel(cfg: Config, paths: RunPaths) -> Path | None:  # noqa: ARG001
    """Write ``diagnostics/event_funnel.{png,csv}`` from the run's on-disk outputs.

    Never raises (module docstring, decision #4); does no gating itself --
    callers check ``cfg.diagnostics.extended_enabled("run")``.

    Parameters
    ----------
    cfg : openmeq.config.Config
        Unused today (every input is a :class:`~openmeq.paths.RunPaths`
        file); kept for the ``(cfg, paths)`` diagnostics signature.
    paths : openmeq.paths.RunPaths

    Returns
    -------
    pathlib.Path or None
        The PNG's path, or ``None`` if anything failed (logged as a warning).
    """
    try:
        out_dir = paths.run_diagnostics_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = _funnel_frame(collect_funnel(paths))
        df.to_csv(out_dir / "event_funnel.csv", index=False)
        png = out_dir / "event_funnel.png"
        _plot_funnel(df, png)
        logger.info("event funnel written to %s", png)
        return png
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the run
        logger.warning("event funnel diagnostic failed: %s", exc, exc_info=True)
        return None
