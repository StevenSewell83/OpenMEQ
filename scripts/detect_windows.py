#!/usr/bin/env python
"""Network coincidence detection restricted to multi-station recording windows.

For triggered archives (e.g. MacBan, converted from SUDS event files), where
data exist only in short event windows. Instead of running ``openmeq
detect`` over whole days, this:

1. builds, from the waveform index (``0_setup/index.json``, written by
   ``openmeq qc``), the time windows in which at least ``--min-stations``
   stations record at once (a station counts while *all* of its indexed
   channels have data), keeping those at least ``--min-duration`` s long;
2. runs the same network coincidence STA/LTA as ``openmeq detect``
   (``cfg.detection``: filter, sta/lta, trigger thresholds,
   ``min_trace_coincidence``, ``min_stations``, ``station_weights``; the
   same preprocessing, :func:`openmeq.detect.worker.preprocess_stream`)
   inside each window, dropping triggers in its first ``--edge-s`` seconds
   (STA/LTA and filter start-up at the window edge);
3. merges and de-duplicates the triggers exactly like ``openmeq detect``
   (:func:`openmeq.detect.merge.merge_windows`, ``detection.dedup_window_s``)
   and writes ``2_detection/detections.csv`` plus the usual detect
   diagnostics, so ``openmeq chunk`` / ``pick`` continue from it.

Also writes ``2_detection/coincidence_windows.csv`` (the windows searched,
with their station lists and trigger counts), and with ``--examples`` (or
``diagnostics.extended`` / ``diagnostics.detect.extended`` set) the
``openmeq detect`` extended ``cf_examples_NN.png`` figures in
``2_detection/diagnostics/extended/``: per station, the detection-filtered
trace and its STA/LTA for ``diagnostics.n_examples`` detections sampled
evenly in time, plus two quiet windows drawn from inside the recording
windows. ``--examples-only`` draws them from the existing detections.csv
without rerunning detection.

Note: a later ``openmeq detect`` (or ``openmeq run`` from detect) would
replace ``detections.csv`` with a whole-day run; start ``openmeq run`` with
``--from chunk`` to keep this script's detections.

Usage
-----
::

    python scripts/detect_windows.py -c config/macban.yaml [--min-stations 8]
        [--min-duration 30] [--edge-s 2] [--windows-only]
        [--examples | --examples-only] [--n-examples N]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.signal.trigger import coincidence_trigger

from openmeq.config import load_config
from openmeq.detect.diagnostics import make_detect_diagnostics
from openmeq.detect.extended import (
    _plot_cf_example,
    _safely,
    _warmup_s,
    extended_dir,
    select_example_windows,
)
from openmeq.detect.merge import merge_windows, write_detections_csv
from openmeq.detect.worker import (
    _build_trace_ids,
    _cached_index,
    _normalize_trigger,
    _read_window,
    preprocess_stream,
)
from openmeq.index import DataIndex
from openmeq.log import configure_logging, get_logger
from openmeq.parallel import pmap
from openmeq.paths import RunPaths

logger = get_logger("detect_windows")

#: Gaps up to this long (s) inside one channel's data are bridged when
#: building recording intervals (sample-rounding between segments).
_MERGE_TOL_S = 1.0


# ---------------------------------------------------------------------------
# Recording windows
# ---------------------------------------------------------------------------


def _channel_intervals(segments, t0: float, t1: float) -> list[list[float]]:
    """Merged ``[start, end]`` (epoch s) data intervals of one channel, clipped to [t0, t1]."""
    spans = []
    for _path, start, end, sr in segments:
        a = UTCDateTime(start).timestamp
        b = UTCDateTime(end).timestamp + 1.0 / sr
        if b > t0 and a < t1:
            spans.append((max(a, t0), min(b, t1)))
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + _MERGE_TOL_S:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _intersect(x: list[list[float]], y: list[list[float]]) -> list[list[float]]:
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            out.append([a, b])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def station_intervals(
    index: DataIndex, t0: float, t1: float, exclude: set[str]
) -> dict[str, list[list[float]]]:
    """``{"NET.STA": intervals}`` when every indexed channel of the station has data."""
    by_station: dict[str, list[str]] = defaultdict(list)
    for seed_id in index.seed_ids():
        if seed_id not in exclude:
            by_station[".".join(seed_id.split(".")[:2])].append(seed_id)
    result = {}
    for station, seed_ids in sorted(by_station.items()):
        iv = None
        for seed_id in seed_ids:
            ch = _channel_intervals(index.channels[seed_id], t0, t1)
            iv = ch if iv is None else _intersect(iv, ch)
        if iv:
            result[station] = iv
    return result


def recording_windows(
    stations: dict[str, list[list[float]]], min_stations: int, min_duration: float
) -> pd.DataFrame:
    """Windows where at least ``min_stations`` stations record, >= ``min_duration`` s.

    A sweep over every station's interval edges; one row per window with
    ``start``, ``end`` (ISO), ``duration_s``, ``n_stations`` (most recording
    at once) and ``stations`` (every station recording at some point in it).
    """
    edges = sorted(
        [(a, 1, s) for s, iv in stations.items() for a, _ in iv]
        + [(b, -1, s) for s, iv in stations.items() for _, b in iv],
        key=lambda e: (e[0], e[1]),  # close before open at equal times
    )
    rows = []
    active: set[str] = set()
    start = None
    members: set[str] = set()
    peak = 0
    for t, delta, station in edges:
        if delta > 0:
            active.add(station)
        else:
            active.discard(station)
        if len(active) >= min_stations:
            if start is None:
                start, members, peak = t, set(active), len(active)
            members |= active
            peak = max(peak, len(active))
        elif start is not None:
            if t - start >= min_duration:
                rows.append((start, t, peak, sorted(members)))
            start = None
    df = pd.DataFrame(rows, columns=["t0", "t1", "n_stations", "stations"])
    df["duration_s"] = df["t1"] - df["t0"]
    df["start"] = [str(UTCDateTime(t)) for t in df["t0"]]
    df["end"] = [str(UTCDateTime(t)) for t in df["t1"]]
    return df


# ---------------------------------------------------------------------------
# Detection inside one window (pmap worker)
# ---------------------------------------------------------------------------


def detect_in_window(item: tuple) -> list[dict]:
    """Coincidence triggers in one window; drops on-times in the first ``edge_s``."""
    t0_iso, t1_iso, index_path, det_cfg, exclude, edge_s = item
    t1, t2 = UTCDateTime(t0_iso), UTCDateTime(t1_iso)
    index = _cached_index(index_path)
    seed_ids = sorted(s for s in index.seed_ids() if s not in exclude)
    stream = index.read(seed_ids, t1, t2)
    if not len(stream):
        return []
    stream = preprocess_stream(stream, det_cfg)
    if not len(stream):
        return []
    raw = coincidence_trigger(
        "recstalta",
        det_cfg.trigger_on,
        det_cfg.trigger_off,
        stream,
        det_cfg.min_trace_coincidence,
        trace_ids=_build_trace_ids(stream, det_cfg.station_weights),
        sta=det_cfg.sta,
        lta=det_cfg.lta,
        details=True,
    )
    triggers = [_normalize_trigger(t) for t in raw]
    return [
        t
        for t in triggers
        if t["n_stations"] >= det_cfg.min_stations and UTCDateTime(t["time"]) >= t1 + edge_s
    ]


def _detect_item(item: tuple) -> list[dict]:
    return detect_in_window(item)


# ---------------------------------------------------------------------------
# Example figures (openmeq detect's extended cf_examples)
# ---------------------------------------------------------------------------


def make_examples(
    cfg, paths: RunPaths, detections: pd.DataFrame, windows: pd.DataFrame, n_examples: int
) -> int:
    """Draw ``cf_examples_NN.png`` (module docstring); returns the number written.

    Quiet windows are drawn only from inside the recording ``windows``, so
    they show real data rather than gaps. Old ``cf_examples_*.png`` files are
    removed first so a smaller set never leaves stale figures behind.
    """
    det = cfg.detection
    index = DataIndex.load(paths.index_json)
    warmup = _warmup_s(det)
    coverage = [
        (UTCDateTime(a), UTCDateTime(b)) for a, b in zip(windows.t0, windows.t1, strict=True)
    ]
    span = (UTCDateTime(cfg.data.start), UTCDateTime(cfg.data.end) + 86400.0)
    examples = select_example_windows(
        detections,
        n_examples,
        span,
        coverage=coverage,
        warmup_s=warmup,
        margin_s=det.dedup_window_s,
    )
    out_dir = extended_dir(paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("cf_examples_*.png"):
        old.unlink()
    for i, ex in enumerate(examples, start=1):
        w0 = ex["t_ref"] - 10.0 - warmup
        w1 = ex["t_ref"] + ex["duration_s"] + 20.0

        def _one(ex=ex, w0=w0, w1=w1, i=i) -> None:
            stream = preprocess_stream(
                _read_window(index, cfg.qc.exclude_channels, w0, w1, pad=0.0), det
            )
            _plot_cf_example(ex, stream, det, out_dir / f"cf_examples_{i:02d}.png")

        _safely(f"cf_examples_{i:02d} ({ex['label']})", _one)
    n = len(list(out_dir.glob("cf_examples_*.png")))
    logger.info("wrote %d cf_examples figure(s) to %s", n, out_dir)
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", required=True, type=Path, help="OpenMEQ YAML config")
    ap.add_argument("--min-stations", type=int, default=8, help="(default: %(default)s)")
    ap.add_argument("--min-duration", type=float, default=30.0, help="s (default: %(default)s)")
    ap.add_argument(
        "--edge-s",
        type=float,
        default=None,
        help="ignore triggers this long after each window start, s (default: 2 x detection.lta)",
    )
    ap.add_argument(
        "--windows-only",
        action="store_true",
        help="write coincidence_windows.csv and stop (no detection)",
    )
    ap.add_argument(
        "--examples", action="store_true", help="also draw the cf_examples waveform figures"
    )
    ap.add_argument(
        "--examples-only",
        action="store_true",
        help="draw cf_examples from the existing detections.csv; no detection run",
    )
    ap.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="detections to draw (default: diagnostics.n_examples)",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = RunPaths(cfg)
    paths.ensure()
    configure_logging(paths, "detect")
    if not paths.index_json.is_file():
        logger.error("%s not found -- run `openmeq qc -c %s` first", paths.index_json, args.config)
        return 1
    shutil.copyfile(args.config, paths.config_snapshot_yaml)
    det = cfg.detection
    edge_s = 2.0 * det.lta if args.edge_s is None else args.edge_s
    exclude = set(cfg.qc.exclude_channels)

    t_start = time.perf_counter()
    index = DataIndex.load(paths.index_json)
    t0 = UTCDateTime(cfg.data.start).timestamp
    t1 = (UTCDateTime(cfg.data.end) + 86400).timestamp
    stations = station_intervals(index, t0, t1, exclude)
    windows = recording_windows(stations, args.min_stations, args.min_duration)
    logger.info(
        "%d station(s) with data in %s..%s; %d window(s) with >= %d stations for >= %g s "
        "(%.1f h, median %.0f s)",
        len(stations),
        cfg.data.start,
        cfg.data.end,
        len(windows),
        args.min_stations,
        args.min_duration,
        windows["duration_s"].sum() / 3600 if len(windows) else 0.0,
        windows["duration_s"].median() if len(windows) else 0.0,
    )
    windows_csv = paths.detections_csv.with_name("coincidence_windows.csv")
    windows_csv.parent.mkdir(parents=True, exist_ok=True)
    out = windows[["start", "end", "duration_s", "n_stations"]].copy()
    out["stations"] = [";".join(s) for s in windows["stations"]]

    if args.windows_only:
        out.to_csv(windows_csv, index=False)
        logger.info("wrote %s (--windows-only: no detection run)", windows_csv)
        return 0
    n_examples = args.n_examples or cfg.diagnostics.n_examples
    if args.examples_only:
        if not paths.detections_csv.is_file():
            logger.error("%s not found -- run without --examples-only first", paths.detections_csv)
            return 1
        make_examples(cfg, paths, pd.read_csv(paths.detections_csv), windows, n_examples)
        return 0

    items = [
        (s, e, str(paths.index_json), det, exclude, edge_s)
        for s, e in zip(windows["start"], windows["end"], strict=True)
    ]
    per_window = pmap(_detect_item, items, cfg.parallel, desc="detect-windows")
    out["n_triggers"] = [len(w) for w in per_window]
    out.to_csv(windows_csv, index=False)

    df = merge_windows(per_window, cfg)
    write_detections_csv(df, paths.detections_csv)
    make_detect_diagnostics(paths, df)
    if args.examples or cfg.diagnostics.extended_enabled("detect"):
        make_examples(cfg, paths, df, windows, n_examples)
    logger.info(
        "%d raw trigger(s) in %d window(s) (%d with none) -> %d detection(s) after de-dup "
        "-> %s (%.1fs)",
        sum(len(w) for w in per_window),
        len(windows),
        int((out["n_triggers"] == 0).sum()),
        len(df),
        paths.detections_csv,
        time.perf_counter() - t_start,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
