"""Cross-correlation QC plots for a sample of event pairs (OpenMEQ relocate).

Opt-in via ``diagnostics.relocate.xcorr_plots`` (or ``openmeq diagnostics
xcorr``): every ``pair_interval``-th ph2dt-linked pair, in ``dt.ct`` order, up
to ``max_pairs`` pairs, gets one figure per shared station/phase/channel
drawn by obspy's own
:func:`obspy.signal.cross_correlation.xcorr_pick_correction` with
``plot=True`` -- the two windows, pick 2 shifted by the correction, and the
correlation function with its parabolic fit.

Inputs are exactly what :mod:`openmeq.relocate.xcorr` correlated: the
pre-filtered, resampled windows in ``relocate/cache/<event_id>.npz`` (so no
extra filter is applied here), picks/origins from
``magnitude/catalog_mag.xml``, and ``cfg.hypodd.xcorr``'s windows and
``maxlag_s``. The pipeline itself uses a faster equivalent of
``xcorr_pick_correction`` (``xcorr._lag_and_cc``, positive peak only), so each
figure's title also shows whether -- and with which cc -- that station/phase
made it into ``dt.cc`` (best channel, ``cc >= min_cc``). Each pair's channels
are also listed in ``xcorr_pairs/summary.csv``.

Outputs::

    relocate/diagnostics/xcorr_pairs/
        summary.csv
        <event_i>__<event_j>_<STA>_<PHASE>_<CHA>.png

Read-only with respect to the relocation: nothing here changes ``dt.cc``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from openmeq.catalog import event_id, read_catalog
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.relocate.xcorr import (
    _event_pick_data,
    _find_seed_id,
    obspy_window,
    parse_dt_ct_detail,
)

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("relocate.xcorr_plots")

OUT_SUBDIR = "xcorr_pairs"
SUMMARY_COLUMNS = [
    "pair_index",
    "cuspid_i",
    "cuspid_j",
    "event_i",
    "event_j",
    "station",
    "phase",
    "seed_id",
    "obspy_cc",
    "obspy_pick2_corr_s",
    "dtcc_cc",
    "dtcc_dt_s",
    "in_dtcc",
    "png",
    "error",
]


def select_pairs(
    pairs: list[tuple[int, int]], pair_interval: int, max_pairs: int
) -> list[tuple[int, tuple[int, int]]]:
    """``(index_in_dt_ct, pair)`` for every ``pair_interval``-th pair, first
    ``max_pairs`` of them."""
    return list(enumerate(pairs))[::pair_interval][:max_pairs]


def _read_map(path: Path, key: str, value: str) -> dict:
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df[key], df[value], strict=True))


def _read_xcorr_results(path: Path) -> dict[str, dict[tuple[str, str], tuple[float, float]]]:
    """``{"ci_cj": {(station_label, phase): (cc, dt_s)}}`` from
    ``xcorr_results.jsonl`` (the per-pair lines that went into ``dt.cc``)."""
    out: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    if not path.is_file():
        return out
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            out[rec["key"]] = {
                (ln["sta"], ln["phase"]): (ln["cc"], ln["dt_s"])
                for ln in rec.get("result", {}).get("lines", [])
            }
    return out


def _load_traces(npz_path: Path) -> dict[str, Any] | None:
    from obspy import Trace, UTCDateTime

    if not npz_path.is_file():
        return None
    with np.load(npz_path) as npz:
        sr = float(npz["sampling_rate"])
        traces = {}
        for seed_id in (str(s) for s in npz["seed_ids"]):
            net, sta, loc, cha = seed_id.split(".")
            traces[seed_id] = Trace(
                data=np.asarray(npz[f"{seed_id}__data"], dtype=np.float64),
                header={
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": cha,
                    "sampling_rate": sr,
                    "starttime": UTCDateTime(float(npz[f"{seed_id}__start"])),
                },
            )
    return traces


def _plot_one(
    tr_i, tr_j, pick_i, pick_j, t_before: float, t_after: float, maxlag_s: float, png: Path, title
) -> tuple[float | None, float | None, str]:
    """Run obspy's xcorr_pick_correction with plot=True into ``png``, adding
    ``title``. Returns ``(pick2_corr, cc, error_or_warnings)``."""
    import matplotlib.pyplot as plt
    from obspy.signal.cross_correlation import xcorr_pick_correction

    before = set(plt.get_fignums())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            corr, cc = xcorr_pick_correction(
                pick_i,
                tr_i,
                pick_j,
                tr_j,
                t_before,
                t_after,
                maxlag_s,
                plot=True,
                # No filename: obspy would switch backends to save it, which
                # closes the figure before the title can be added. With
                # filename=None it draws on the current (Agg) backend and calls
                # plt.show(), a no-op here; the figure is saved below instead.
            )
        notes = "; ".join(
            dict.fromkeys(str(w.message) for w in caught if "non-interactive" not in str(w.message))
        )
        new = [n for n in plt.get_fignums() if n not in before]
        if new:
            fig = plt.figure(new[-1])
            fig.set_size_inches(8.0, 6.5)
            fig.suptitle(title, fontsize=8)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            fig.savefig(png, dpi=110)
        else:
            notes = "; ".join(filter(None, [notes, "obspy drew no figure"]))
        return float(corr), float(cc), notes
    except Exception as exc:  # noqa: BLE001 - one bad window must not stop the rest
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        for n in set(plt.get_fignums()) - before:
            plt.close(n)


def _plot_pair_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Worker: every station/phase/channel figure for one pair."""
    from obspy import UTCDateTime

    from openmeq.plotting import setup_style

    setup_style()
    traces_i = _load_traces(Path(item["npz_i"]))
    traces_j = _load_traces(Path(item["npz_j"]))
    out_dir = Path(item["out_dir"])
    base = {
        "pair_index": item["pair_index"],
        "cuspid_i": item["cuspid_i"],
        "cuspid_j": item["cuspid_j"],
        "event_i": item["event_i"],
        "event_j": item["event_j"],
    }
    if traces_i is None or traces_j is None:
        return [{**base, "error": "no waveform cache for one of the events"}]

    rows: list[dict[str, Any]] = []
    for label, net_sta, phase in item["station_phases"]:
        pick_i = item["picks_i"].get(net_sta, {}).get(phase)
        pick_j = item["picks_j"].get(net_sta, {}).get(phase)
        kept = item["kept"].get(f"{label}|{phase}")
        row0 = {
            **base,
            "station": net_sta,
            "phase": phase,
            "dtcc_cc": kept[0] if kept else None,
            "dtcc_dt_s": kept[1] if kept else None,
            "in_dtcc": kept is not None,
        }
        if pick_i is None or pick_j is None:
            rows.append({**row0, "error": "pick missing at one event"})
            continue
        t_before, t_after = item["window_p"] if phase == "P" else item["window_s"]
        maxlag = item["s_maxlag_s"] if phase == "S" else item["maxlag_s"]
        seeds = []
        for comp in item["components"].get(phase, []):
            seed_i = _find_seed_id(list(traces_i), net_sta, comp)
            seed_j = _find_seed_id(list(traces_j), net_sta, comp)
            if seed_i and seed_j:
                seeds.append((seed_i, seed_j))
        if not seeds:
            rows.append({**row0, "error": "no cached channel at both events"})
            continue
        for seed_i, seed_j in seeds:
            cha = seed_i.split(".")[-1]
            png = out_dir / (
                f"{item['event_i']}__{item['event_j']}_{net_sta.split('.')[-1]}_{phase}_{cha}.png"
            )
            in_dtcc = (
                f"in dt.cc (best channel cc={kept[0]:.2f}, dt={kept[1]:+.4f} s)"
                if kept
                else "NOT in dt.cc"
            )
            # Same (t_before, t_after) xcorr.cross_correlate_pair passes on, so
            # the plotted window is exactly the one the pipeline correlated:
            # [pick - t_before - maxlag/2, pick + t_after + maxlag/2].
            w0 = -t_before - maxlag / 2.0
            w1 = t_after + maxlag / 2.0
            title = (
                f"pair {item['pair_index']}: {item['event_i']} (1, black) vs "
                f"{item['event_j']} (2, red)\n{seed_i} {phase}   correlated window: pick "
                f"{w0:+.3f} to {w1:+.3f} s (time 0 = window start), maxlag "
                f"{maxlag} s   [{in_dtcc}]"
            )
            corr, cc, note = _plot_one(
                traces_i[seed_i],
                traces_j[seed_j],
                UTCDateTime(pick_i),
                UTCDateTime(pick_j),
                t_before,
                t_after,
                maxlag,
                png,
                title,
            )
            rows.append(
                {
                    **row0,
                    "seed_id": seed_i,
                    "obspy_cc": cc,
                    "obspy_pick2_corr_s": corr,
                    "png": png.name if png.exists() else None,
                    "error": note or None,
                }
            )
    return rows


def write_xcorr_plots(
    cfg: Config,
    paths: RunPaths,
    *,
    pair_interval: int | None = None,
    max_pairs: int | None = None,
) -> Path | None:
    """Write ``relocate/diagnostics/xcorr_pairs/`` (module docstring).

    ``pair_interval``/``max_pairs`` override
    ``cfg.diagnostics.relocate.xcorr_plots``; does no gating itself (the
    caller checks ``.enabled``). Returns the summary CSV path, or ``None`` if
    the relocation inputs (``dt.ct``, maps, catalogue) are missing.
    """
    pcfg = cfg.diagnostics.relocate.xcorr_plots
    pair_interval = pair_interval or pcfg.pair_interval
    max_pairs = max_pairs or pcfg.max_pairs
    work = paths.relocate_work_dir
    needed = [
        paths.dt_ct,
        work / "event_id_map.csv",
        work / "station_label_map.csv",
        paths.catalog_mag_xml,
    ]
    missing = [str(p) for p in needed if not Path(p).is_file()]
    if missing:
        logger.warning("xcorr plots skipped; missing %s", ", ".join(missing))
        return None

    detail = parse_dt_ct_detail(paths.dt_ct)
    chosen = select_pairs(list(detail), pair_interval, max_pairs)
    cuspid_to_event = {
        int(c): e for e, c in _read_map(work / "event_id_map.csv", "event_id", "cuspid").items()
    }
    label_to_net_sta = {
        lab: ns for ns, lab in _read_map(work / "station_label_map.csv", "net_sta", "label").items()
    }
    results = _read_xcorr_results(paths.xcorr_results_jsonl)

    wanted = {cuspid_to_event.get(c) for _, pair in chosen for c in pair}
    picks = {}
    for event in read_catalog(paths.catalog_mag_xml):
        eid = event_id(event)
        if eid in wanted:
            data = _event_pick_data(event)
            if data is not None:
                picks[eid] = data["picks"]

    out_root = paths.diagnostics_dir("relocate") / OUT_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    xc = cfg.hypodd.xcorr
    items = []
    for idx, (ci, cj) in chosen:
        ev_i, ev_j = cuspid_to_event.get(ci), cuspid_to_event.get(cj)
        if ev_i is None or ev_j is None:
            continue
        items.append(
            {
                "pair_index": idx,
                "cuspid_i": ci,
                "cuspid_j": cj,
                "event_i": ev_i,
                "event_j": ev_j,
                "npz_i": str(paths.relocate_cache_file(ev_i)),
                "npz_j": str(paths.relocate_cache_file(ev_j)),
                "out_dir": str(out_root),
                "picks_i": picks.get(ev_i, {}),
                "picks_j": picks.get(ev_j, {}),
                "station_phases": [
                    (lab, label_to_net_sta[lab], ph)
                    for lab, ph in detail[(ci, cj)]
                    if lab in label_to_net_sta
                ],
                "kept": {f"{s}|{p}": v for (s, p), v in results.get(f"{ci}_{cj}", {}).items()},
                # (t_before, t_after) exactly as xcorr.cross_correlate_pair uses them
                "window_p": obspy_window(xc.window_p),
                "window_s": obspy_window(xc.window_s),
                "maxlag_s": xc.maxlag_s,
                "s_maxlag_s": xc.maxlag_for("S"),
                "components": xc.components,
            }
        )

    rows = [
        r for res in pmap(_plot_pair_item, items, cfg.parallel, desc="xcorr plots") for r in res
    ]
    summary = out_root / "summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(summary, index=False, float_format="%.6f")
    n_png = sum(1 for r in rows if r.get("png"))
    logger.info(
        "xcorr plots: %d of %d linked pair(s) (every %d, max %d) -> %d figure(s) in %s",
        len(items),
        len(detail),
        pair_interval,
        max_pairs,
        n_png,
        out_root,
    )
    return summary
