"""OpenMEQ-qlg.10: throwaway measurement script -- PhaseNet pick recall/
precision on the M0.8 synthetic dataset (``tests/data/make_synthetic.py``).

**Not production code.** This is a one-off measurement tool for the
OpenMEQ-qlg.10 spike/check: it generates the synthetic dataset, prepares
PhaseNet-ready 100 Hz windows itself (following ``docs/phasenet_interface.md``
and ``config/example.yaml``'s ``phasenet:`` block, but hand-rolled here
rather than importing from the not-yet-built M4.2 chunk-prep module), runs
``predict.py`` once (``--batch_size=1``, ``CUDA_VISIBLE_DEVICES=""``), and
scores the resulting picks against ``truth["events"][i]["arrivals"]``.

Usage::

    OPENMEQ_EXTERNAL_DIR=/path/to/external \\
    .venv/bin/python scripts/phasenet_recall_check.py [--seed SEED] [--keep DIR]

Prints a recall/precision table to stdout and (with ``--json``) writes the
full per-arrival match table as JSON for further analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime, read

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "data"))

import make_synthetic  # noqa: E402

from openmeq.external import ExternalToolError, phasenet_dir, phasenet_model_dir, run  # noqa: E402

# --- PhaseNet-ready preprocessing parameters (config/example.yaml's
# `phasenet:` block defaults; ARCHITECTURE.md doesn't yet have a phasenet
# preprocessing module (that's M4.2), so this mirrors the config values by
# hand). ---
TARGET_RATE_HZ = 100.0
TAPER_PCT = 0.05
HIGHPASS_HZ = 1.0
MIN_P_PROB = 0.3
MIN_S_PROB = 0.3

# Window: 60 s around the true origin time, shared by every station in an
# event (per the OpenMEQ-qlg.10 work item: "cut a 60 s window around the
# true origin for all stations").
PRE_ORIGIN_S = 5.0
WINDOW_LEN_S = 60.0

# A pick counts as a match if within this many seconds of the true arrival,
# same phase (per the bd issue).
MATCH_TOL_S = 0.1

DECIMATE_FACTOR = int(round(make_synthetic.SAMPLE_RATE_HZ / TARGET_RATE_HZ))


def _read_window(outdir: Path, seed_id: str, start: UTCDateTime, end: UTCDateTime):
    """Read raw 200 Hz data for [start, end] (with filter padding), which may
    span the day-1/day-2 boundary."""
    pad = 5.0
    read_start, read_end = start - pad, end + pad
    traces = []
    for day_start in (
        UTCDateTime(read_start.date),
        UTCDateTime(read_end.date),
    ):
        date_str = day_start.strftime("%Y-%m-%d")
        path = outdir / "waveforms" / date_str / f"{seed_id}.mseed"
        if path.is_file():
            traces.extend(read(str(path), starttime=read_start, endtime=read_end))
    if not traces:
        return None
    st = Stream(traces)
    st.merge(fill_value=0)
    tr = st[0]
    tr.data = tr.data.astype(np.float64)  # avoid float fill_value on an int32 buffer
    tr.trim(read_start, read_end, pad=True, fill_value=0.0)
    return tr


def _prepare_trace(tr, start: UTCDateTime, end: UTCDateTime):
    """PhaseNet-ready preprocessing: demean/detrend, taper, highpass (before
    decimation), anti-alias decimate 200->100 Hz, trim to the exact window."""
    tr = tr.copy()
    tr.data = tr.data.astype(np.float64)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(TAPER_PCT, type="hann")
    if HIGHPASS_HZ > 0:
        tr.filter("highpass", freq=HIGHPASS_HZ, corners=4, zerophase=True)
    tr.decimate(DECIMATE_FACTOR, no_filter=False)  # anti-alias lowpass + downsample
    tr.trim(start, end, pad=True, fill_value=0.0)
    return tr


def build_chunks(outdir: Path, truth: dict, chunks_dir: Path) -> Path:
    """Build one PhaseNet-ready mseed file per event under `chunks_dir`, and
    return the data_list.csv path."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ev in truth["events"]:
        event_id = ev["event_id_hint"]
        origin_time = ev["origin"]["time"]
        start = origin_time - PRE_ORIGIN_S
        end = start + WINDOW_LEN_S

        traces = []
        for sta_code, _kind, _x, _y, _elev, _depth, channels in make_synthetic.STATIONS:
            for cha in channels:
                seed_id = f"{make_synthetic.NETWORK}.{sta_code}.{make_synthetic.LOCATION}.{cha}"
                raw = _read_window(outdir, seed_id, start, end)
                if raw is None:
                    continue
                traces.append(_prepare_trace(raw, start, end))

        event_dir = chunks_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        out_path = event_dir / f"{event_id}.mseed"
        Stream(traces).write(str(out_path), format="MSEED")
        rows.append(f"{event_id}/{event_id}.mseed")

    data_list = chunks_dir / "data_list.csv"
    with data_list.open("w") as fh:
        fh.write("fname\n")
        for r in rows:
            fh.write(r + "\n")
    return data_list


def run_phasenet(data_dir: Path, data_list: Path, result_dir: Path) -> Path:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    predict_py = phasenet_dir() / "phasenet" / "predict.py"
    cmd = [
        sys.executable,
        str(predict_py),
        f"--model_dir={phasenet_model_dir()}",
        f"--data_dir={data_dir}",
        f"--data_list={data_list}",
        "--format=mseed",
        f"--result_dir={result_dir}",
        "--batch_size=1",
        f"--sampling_rate={int(TARGET_RATE_HZ)}",
        f"--min_p_prob={MIN_P_PROB}",
        f"--min_s_prob={MIN_S_PROB}",
    ]
    try:
        run(cmd, cwd=phasenet_dir() / "phasenet", timeout=600)
    except ExternalToolError as exc:
        print("predict.py FAILED")
        print("stdout:\n", exc.stdout)
        print("stderr:\n", exc.stderr)
        raise
    return result_dir / "picks.csv"


def _read_picks(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _measure_snr(outdir: Path, ev: dict, sta_code: str, channels: tuple, phase: str) -> float:
    """Empirical SNR proxy: peak abs amplitude in a short window around the
    true arrival on the phase's dominant component (Z for P, max horizontal
    for S), in RAW 200 Hz counts, divided by 1.4826*MAD of a quiet segment
    of the same channel earlier in the window (before the P arrival)."""
    station_key = f"{make_synthetic.NETWORK}.{sta_code}"
    arr = ev["arrivals"][station_key]
    origin_time = ev["origin"]["time"]
    win_start = origin_time - PRE_ORIGIN_S
    t0 = arr["P"] if phase == "P" else arr["S"]

    comps = [channels[0]] if phase == "P" else list(channels[1:])
    best = 0.0
    for cha in comps:
        seed_id = f"{make_synthetic.NETWORK}.{sta_code}.{make_synthetic.LOCATION}.{cha}"
        tr = _read_window(outdir, seed_id, win_start, win_start + WINDOW_LEN_S)
        if tr is None:
            continue
        data = tr.data.astype(float)
        sr = tr.stats.sampling_rate
        onset = int(round((t0 - tr.stats.starttime) * sr))
        lo, hi = max(0, onset - int(0.3 * sr)), min(len(data), onset + int(0.3 * sr))
        if hi <= lo:
            continue
        sig_amp = float(np.max(np.abs(data[lo:hi])))
        quiet = data[: max(1, int(2.0 * sr))]
        mad = np.median(np.abs(quiet - np.median(quiet))) or 1.0
        noise = 1.4826 * mad
        best = max(best, sig_amp / noise)
    return best


def score(outdir: Path, truth: dict, picks: list[dict]) -> dict:
    # picks by (event_id, station_key, phase) -> sorted list of (time, row)
    by_key = defaultdict(list)
    for p in picks:
        # file_name is "<event_id>/<event_id>.mseed" per data_list.csv row
        event_id = Path(p["file_name"]).parent.name or Path(p["file_name"]).stem
        parts = p["station_id"].split(".")
        station_key = f"{parts[0]}.{parts[1]}"
        by_key[(event_id, station_key, p["phase_type"])].append((UTCDateTime(p["phase_time"]), p))

    station_channels = {s[0]: s[6] for s in make_synthetic.STATIONS}

    records = []
    used_picks: set[tuple] = set()
    for ev in truth["events"]:
        event_id = ev["event_id_hint"]
        for station_key, arr in ev["arrivals"].items():
            sta_code = station_key.split(".")[1]
            for phase in ("P", "S"):
                truth_t = arr[phase]
                candidates = by_key.get((event_id, station_key, phase), [])
                match = None
                for i, (t, row) in enumerate(candidates):
                    if abs(t - truth_t) <= MATCH_TOL_S:
                        key = (event_id, station_key, phase, i)
                        if key not in used_picks:
                            match = row
                            used_picks.add(key)
                            break
                snr = _measure_snr(outdir, ev, sta_code, station_channels[sta_code], phase)
                records.append(
                    {
                        "event_id": event_id,
                        "ml": ev["ml"],
                        "station": station_key,
                        "phase": phase,
                        "snr": snr,
                        "matched": match is not None,
                        "score": float(match["phase_score"]) if match else None,
                    }
                )

    # False positives: picks never used as a match for any truth arrival.
    n_fp = {"P": 0, "S": 0}
    for key, candidates in by_key.items():
        _event_id, _station_key, phase = key
        for i, _ in enumerate(candidates):
            full_key = (*key, i)
            if full_key not in used_picks:
                n_fp[phase] += 1

    return {"records": records, "n_fp": n_fp}


def _summarize(records: list[dict], n_fp: dict, ml_threshold: float) -> str:
    lines = []
    for phase in ("P", "S"):
        recs = [r for r in records if r["phase"] == phase]
        recs_hi = [r for r in recs if r["ml"] >= ml_threshold]
        n_tp = sum(1 for r in recs if r["matched"])
        n_tp_hi = sum(1 for r in recs_hi if r["matched"])
        recall_all = n_tp / len(recs) if recs else float("nan")
        recall_hi = n_tp_hi / len(recs_hi) if recs_hi else float("nan")
        precision = n_tp / (n_tp + n_fp[phase]) if (n_tp + n_fp[phase]) else float("nan")
        lines.append(
            f"{phase}: recall(all)={recall_all:.1%} ({n_tp}/{len(recs)})  "
            f"recall(ML>={ml_threshold})={recall_hi:.1%} ({n_tp_hi}/{len(recs_hi)})  "
            f"precision={precision:.1%} ({n_tp}/{n_tp + n_fp[phase]})  fp={n_fp[phase]}"
        )

    lines.append("")
    lines.append(f"Per-station recall (ML>={ml_threshold:.1f}):")
    stations = sorted({r["station"] for r in records})
    for sta in stations:
        row = [sta]
        for phase in ("P", "S"):
            recs = [
                r
                for r in records
                if r["station"] == sta and r["phase"] == phase and r["ml"] >= ml_threshold
            ]
            n_tp = sum(1 for r in recs if r["matched"])
            pct = n_tp / len(recs) if recs else float("nan")
            row.append(f"{phase}={pct:.0%}({n_tp}/{len(recs)})")
        lines.append("  " + "  ".join(row))

    lines.append("")
    lines.append("Recall by magnitude bin:")
    bins = [(-1.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0)]
    for lo, hi in bins:
        for phase in ("P", "S"):
            recs = [r for r in records if r["phase"] == phase and lo <= r["ml"] < hi]
            n_tp = sum(1 for r in recs if r["matched"])
            pct = n_tp / len(recs) if recs else float("nan")
            lines.append(f"  ML[{lo:+.1f},{hi:+.1f}) {phase}: {pct:.0%} ({n_tp}/{len(recs)})")

    lines.append("")
    lines.append("Recall by measured SNR bin:")
    snr_bins = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]
    for lo, hi in snr_bins:
        for phase in ("P", "S"):
            recs = [r for r in records if r["phase"] == phase and lo <= r["snr"] < hi]
            n_tp = sum(1 for r in recs if r["matched"])
            pct = n_tp / len(recs) if recs else float("nan")
            lines.append(f"  SNR[{lo},{hi}) {phase}: {pct:.0%} ({n_tp}/{len(recs)})")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=make_synthetic.DEFAULT_SEED)
    ap.add_argument("--keep", type=str, default=None, help="dir to keep generated data in")
    ap.add_argument("--json", type=str, default=None, help="path to dump full records as JSON")
    ap.add_argument("--ml-threshold", type=float, default=0.5)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(args.keep) if args.keep else Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        outdir = base / "synthetic"
        chunks_dir = base / "chunks_pn"
        result_dir = base / "pick_raw"

        import time

        t0 = time.time()
        truth = make_synthetic.generate(outdir, seed=args.seed)
        print(f"generate(): {time.time() - t0:.2f}s")

        t0 = time.time()
        data_list = build_chunks(outdir, truth, chunks_dir)
        print(f"build_chunks(): {time.time() - t0:.2f}s")

        t0 = time.time()
        picks_csv = run_phasenet(chunks_dir, data_list, result_dir)
        print(f"predict.py: {time.time() - t0:.2f}s")

        picks = _read_picks(picks_csv)
        print(f"total picks: {len(picks)}")

        result = score(outdir, truth, picks)
        print()
        print(_summarize(result["records"], result["n_fp"], args.ml_threshold))

        if args.json:
            Path(args.json).write_text(json.dumps(result["records"], indent=2))


if __name__ == "__main__":
    main()
