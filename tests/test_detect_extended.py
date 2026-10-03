"""Tests for the opt-in extended detect QAQC (``OpenMEQ-02o.2``,
:mod:`openmeq.detect.extended`).

These build their own tiny project in ``tmp_path`` (4 single-channel
stations, 20 minutes at 100 Hz, three impulsive "events" plus one
single-station glitch) rather than using the shared ``synthetic_project``
fixture, so a full ``run_detect`` pass takes well under a second and every
enabled/disabled combination can be exercised end to end without the
``slow`` marker.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from obspy import Stream, Trace, UTCDateTime

from openmeq.config import load_config
from openmeq.detect import extended as extended_mod
from openmeq.detect.extended import (
    STATION_TRIGGERS_COLUMNS,
    select_example_windows,
    station_trigger_rate_matrix,
    station_triggers_csv,
)
from openmeq.detect.pipeline import run_detect
from openmeq.detect.worker import detect_window, detect_window_with_station_triggers
from openmeq.index import DataIndex, build_index
from openmeq.paths import RunPaths

DAY = UTCDateTime("2024-01-01")
T0 = DAY + 3600.0  # data runs 01:00-01:20
SR = 100.0
DURATION_S = 1200.0
STATIONS = ["S01", "S02", "S03", "S04"]
EVENT_OFFSETS_S = [200.0, 600.0, 1000.0]
GLITCH_OFFSET_S = 400.0  # S01 only -> a station trigger but no detection


def _pulse(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    return 60.0 * np.exp(-t / 0.4) * np.sin(2 * np.pi * 8.0 * t) + rng.normal(0, 0.5, n)


def _make_project(tmp_path: Path, *, extended: bool) -> tuple[Path, RunPaths]:
    rng = np.random.default_rng(42)
    wf_dir = tmp_path / "waveforms"
    wf_dir.mkdir(parents=True)
    n = int(DURATION_S * SR)
    seed_ids = []
    for k, sta in enumerate(STATIONS):
        data = rng.normal(0.0, 1.0, n)
        offsets = list(EVENT_OFFSETS_S) + ([GLITCH_OFFSET_S] if sta == "S01" else [])
        for off in offsets:
            i0 = int((off + 0.3 * k) * SR)
            data[i0 : i0 + 300] += _pulse(300, rng)
        tr = Trace(data=data.astype(np.float32))
        tr.stats.update({"network": "XX", "station": sta, "location": "", "channel": "HHZ"})
        tr.stats.sampling_rate = SR
        tr.stats.starttime = T0
        Stream([tr]).write(str(wf_dir / f"XX.{sta}.HHZ.mseed"), format="MSEED")
        seed_ids.append(tr.id)

    cfg_dict = {
        "project": {"name": "tiny", "output_dir": "./run"},
        "parallel": {"n_workers": 1, "threads_per_worker": 1},
        "data": {
            "waveform_dirs": ["./waveforms"],
            "file_pattern": "*.mseed",
            "start": DAY.date,
            "end": DAY.date,
        },
        "detection": {
            "filter": {"type": "bandpass", "freqmin": 2.0, "freqmax": 20.0},
            "coincidence_sum": 3,
        },
        "nlloc": {
            "velocity_model": {"type": "1d", "file": "./vel.txt"},
            "grid": {"origin": [0.0, 0.0], "size_km": [10.0, 10.0, 10.0], "spacing_km": 1.0},
        },
        "hypodd": {"velocity_model": {"type": "1d", "file": "./vel.txt"}},
        "diagnostics": {"extended": extended, "n_examples": 2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    build_index(cfg)
    return config_path, paths


def _run(config_path: Path, **kwargs) -> RunPaths:
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    run_detect(cfg, paths, config_path=str(config_path), **kwargs)
    return paths


def _set_extended(config_path: Path, value: bool) -> None:
    d = yaml.safe_load(config_path.read_text())
    d["diagnostics"]["extended"] = value
    config_path.write_text(yaml.safe_dump(d))


def test_disabled_writes_no_extended_outputs(tmp_path):
    config_path, _ = _make_project(tmp_path, extended=False)
    paths = _run(config_path)

    assert paths.detections_csv.is_file()
    assert not station_triggers_csv(paths).exists()
    assert not (paths.detect_diagnostics_dir / "extended").exists()
    # Baseline diagnostics are unaffected.
    assert (paths.detect_diagnostics_dir / "detections_per_hour.png").is_file()


def test_enabled_writes_station_triggers_and_figures(tmp_path):
    config_path, _ = _make_project(tmp_path, extended=True)
    paths = _run(config_path)

    detections = pd.read_csv(paths.detections_csv)
    assert len(detections) == len(EVENT_OFFSETS_S)

    st = pd.read_csv(station_triggers_csv(paths))
    assert list(st.columns) == STATION_TRIGGERS_COLUMNS
    assert set(st["station"]) == {f"XX.{s}" for s in STATIONS}
    # The S01-only glitch is a station trigger without a detection.
    glitch = T0 + GLITCH_OFFSET_S
    s01_on = [UTCDateTime(t) for t in st.loc[st["station"] == "XX.S01", "on_time"]]
    assert any(abs(t - glitch) < 2.0 for t in s01_on)
    assert (st["peak_ratio"] >= load_config(config_path).detection.trigger_on).all()
    assert list(st["on_time"]) == sorted(st["on_time"])

    ext = paths.detect_diagnostics_dir / "extended"
    assert (ext / "station_trigger_rate.png").stat().st_size > 0
    examples = sorted(p.name for p in ext.glob("cf_examples_*.png"))
    # n_examples=2 detections + N_QUIET_WINDOWS quiet windows (drawn inside
    # the 20-minute data span, which has plenty of detection-free room).
    n_expected = 2 + extended_mod.N_QUIET_WINDOWS
    assert examples == [f"cf_examples_{i:02d}.png" for i in range(1, n_expected + 1)]


def test_enabled_does_not_change_detections(tmp_path):
    off_path, _ = _make_project(tmp_path / "off", extended=False)
    on_path, _ = _make_project(tmp_path / "on", extended=True)
    off = pd.read_csv(_run(off_path).detections_csv)
    on = pd.read_csv(_run(on_path).detections_csv)
    pd.testing.assert_frame_equal(off, on)


def test_toggle_reruns_and_cleans_up(tmp_path):
    config_path, _ = _make_project(tmp_path, extended=False)
    paths = _run(config_path)
    assert not station_triggers_csv(paths).exists()

    # Switching extended on for an already-detected run must rerun (only way
    # to get per-station triggers), not skip.
    _set_extended(config_path, True)
    paths = _run(config_path)
    assert station_triggers_csv(paths).is_file()
    assert (paths.detect_diagnostics_dir / "extended" / "station_trigger_rate.png").is_file()

    # A skipped rerun with the extended dir removed rebuilds it from disk.
    for p in (paths.detect_diagnostics_dir / "extended").iterdir():
        p.unlink()
    (paths.detect_diagnostics_dir / "extended").rmdir()
    paths = _run(config_path)
    assert (paths.detect_diagnostics_dir / "extended" / "station_trigger_rate.png").is_file()

    # Switching it off again + a real rerun drops the stale CSV.
    _set_extended(config_path, False)
    paths = _run(config_path, force=True)
    assert not station_triggers_csv(paths).exists()


def test_figure_failure_never_fails_step(tmp_path, monkeypatch, caplog):
    config_path, _ = _make_project(tmp_path, extended=True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr(extended_mod, "_plot_cf_example", _boom)
    with caplog.at_level(logging.WARNING):
        paths = _run(config_path)

    ext = paths.detect_diagnostics_dir / "extended"
    assert (ext / "station_trigger_rate.png").is_file()
    assert not list(ext.glob("cf_examples_*.png"))
    assert "synthetic plotting failure" in caplog.text
    assert station_triggers_csv(paths).is_file()


def test_worker_variants_agree(tmp_path):
    config_path, paths = _make_project(tmp_path, extended=True)
    args = (
        DAY.isoformat(),
        (DAY + 86400.0).isoformat(),
        str(paths.index_json),
        str(config_path),
    )
    plain = detect_window(*args)
    triggers, station = detect_window_with_station_triggers(*args)
    assert triggers == plain
    assert station
    # Only on-times inside the unpadded window are kept.
    assert all(DAY <= UTCDateTime(r["on_time"]) < DAY + 86400.0 for r in station)


def test_station_trigger_rate_matrix_absolute_and_folded():
    df = pd.DataFrame(
        [
            {"station": "XX.A", "on_time": "2024-01-01T01:10:00.000Z"},
            {"station": "XX.A", "on_time": "2024-01-01T01:50:00.000Z"},
            {"station": "XX.B", "on_time": "2024-01-02T05:00:00.000Z"},
        ]
    )
    span = (UTCDateTime("2024-01-01"), UTCDateTime("2024-01-03"))
    matrix, _ = station_trigger_rate_matrix(df, ["XX.A", "XX.B", "XX.C"], span)
    assert matrix.shape == (3, 48)
    assert matrix.loc["XX.A", "01-01 01"] == 2
    assert matrix.loc["XX.B", "01-02 05"] == 1
    assert matrix.loc["XX.C"].sum() == 0

    long_span = (UTCDateTime("2024-01-01"), UTCDateTime("2024-01-11"))
    folded, xlabel = station_trigger_rate_matrix(df, ["XX.A"], long_span)
    assert list(folded.columns) == list(range(24))
    assert folded.loc["XX.A", 1] == pytest.approx(2 / 10)
    assert "Hour of day" in xlabel


def test_select_example_windows_deterministic_and_quiet():
    times = [UTCDateTime("2024-01-01T00:00:00") + 3600.0 * i for i in range(10)]
    det = pd.DataFrame(
        {
            "event_id": [f"ev{i}" for i in range(10)],
            "time": [str(t) for t in times],
            "duration_s": [2.0] * 10,
            "stations": ["XX.A;XX.B"] * 10,
            "trace_ids": ["XX.A..HHZ;XX.B..HHZ"] * 10,
        }
    )
    span = (UTCDateTime("2024-01-01"), UTCDateTime("2024-01-02"))
    w1 = select_example_windows(det, 4, span, margin_s=10.0)
    w2 = select_example_windows(det, 4, span, margin_s=10.0)
    assert [w["t_ref"] for w in w1] == [w["t_ref"] for w in w2]

    detections = [w for w in w1 if w["kind"] == "detection"]
    quiet = [w for w in w1 if w["kind"] == "quiet"]
    assert [w["label"] for w in detections] == ["ev0", "ev3", "ev6", "ev9"]
    assert detections[0]["stations"] == ["XX.A", "XX.B"]
    assert len(quiet) == extended_mod.N_QUIET_WINDOWS
    for w in quiet:
        assert min(abs(w["t_ref"] - t) for t in times) > 10.0 + extended_mod.PRE_S

    # Restricted to data coverage (incl. warm-up lead-in).
    cov = [(UTCDateTime("2024-01-01T12:30:00"), UTCDateTime("2024-01-01T12:40:00"))]
    w3 = select_example_windows(det, 4, span, coverage=cov, warmup_s=30.0, margin_s=10.0)
    quiet = [w for w in w3 if w["kind"] == "quiet"]
    assert len(quiet) == extended_mod.N_QUIET_WINDOWS
    for w in quiet:
        assert w["t_ref"] - extended_mod.PRE_S - 30.0 >= cov[0][0]
        assert w["t_ref"] + w["duration_s"] + extended_mod.POST_S <= cov[0][1]


def test_data_coverage_from_index(tmp_path):
    _config_path, paths = _make_project(tmp_path, extended=True)
    index = DataIndex.load(paths.index_json)
    span = (DAY, DAY + 86400.0)
    cov = extended_mod.data_coverage(index, span)
    assert len(cov) == 1
    assert abs(cov[0][0] - T0) < 1.0 and abs(cov[0][1] - (T0 + DURATION_S)) < 1.0
