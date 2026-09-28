"""Tests for openmeq.relocate.waveform_cache (M10.1 / OpenMEQ-16d.1).

Per this issue's own orchestrator note, the upstream steps (detection,
chunk writing's own detections, PhaseNet picking, magnitude) are not all
merged yet, so every test here builds its own minimal, hand-made inputs:

- a config built directly from the pydantic models (not a full YAML file),
  with just enough of ``nlloc``/``hypodd`` filled in to validate (this
  module never reads those sections).
- a raw chunk written directly as miniSEED (:func:`_write_chunk`) --
  standing in for ``openmeq.chunk.write_chunks``'s own output format
  (``chunks/<event_id>/<event_id>.mseed``).
- a small hand-built QuakeML catalogue (:func:`_write_catalog`), standing
  in for ``magnitude/catalog_mag.xml``, using only what
  ``openmeq.relocate.waveform_cache`` actually reads (``Pick.phase_hint``/
  ``waveform_id``/``time`` -- no origins/arrivals needed).

The one exception is ``test_synthetic_project_cache_builds``, which uses
the real ``synthetic_project`` fixture and the real, merged
``openmeq.chunk.write_chunks`` (per this issue's own notes: "the chunk
writer has merged ... so you can use it on hand-made detections to
produce real chunks").

Disk-space note (worker instructions): every test here writes only into
its own function-scoped ``tmp_path``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Pick, WaveformStreamID

from openmeq.catalog import event_id_from_time, make_event, write_catalog
from openmeq.chunk import write_chunks
from openmeq.config import (
    Config,
    DataConfig,
    HypoddConfig,
    HypoddVelocityModelConfig,
    NllocConfig,
    NllocGridConfig,
    NllocVelocityModelConfig,
    ParallelConfig,
    ProjectConfig,
    StationsConfig,
    XcorrConfig,
)
from openmeq.index import build_index
from openmeq.paths import RunPaths
from openmeq.relocate.waveform_cache import build_cache

pytestmark = []


# ---------------------------------------------------------------------------
# Minimal config / chunk / catalog builders
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path, **xcorr_kwargs) -> Config:
    """A minimal, valid :class:`Config` -- only ``hypodd.xcorr`` varies per test.

    ``nlloc``/``hypodd.velocity_model`` point at files that don't need to
    exist (this module never touches them); ``parallel.n_workers=1`` runs
    every ``pmap`` call in-process (no subprocess spawn overhead) for fast,
    deterministic unit tests.
    """
    xcorr = XcorrConfig(**xcorr_kwargs)
    return Config(
        project=ProjectConfig(name="test", output_dir=tmp_path / "run"),
        parallel=ParallelConfig(n_workers=1),
        data=DataConfig(waveform_dirs=[tmp_path], start=date(2025, 1, 1), end=date(2025, 1, 2)),
        stations=StationsConfig(),
        nlloc=NllocConfig(
            velocity_model=NllocVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
            grid=NllocGridConfig(origin=(0.0, 0.0), size_km=(10.0, 10.0, 10.0), spacing_km=0.5),
        ),
        hypodd=HypoddConfig(
            velocity_model=HypoddVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
            xcorr=xcorr,
        ),
    )


def _write_chunk(
    paths: RunPaths,
    event_id: str,
    traces: list[tuple[str, str, str, str, float, UTCDateTime, int]],
) -> None:
    """Write a hand-built raw chunk (``chunks/<event_id>/<event_id>.mseed``).

    ``traces`` is a list of ``(net, sta, loc, chan, sampling_rate, start, npts)``;
    each trace's data is deterministic pseudo-random noise (content doesn't
    matter for this module -- only timing/rate).
    """
    st = Stream()
    for i, (net, sta, loc, chan, rate, start, npts) in enumerate(traces):
        rng = np.random.default_rng(1000 + i)
        data = rng.uniform(-1.0, 1.0, size=npts).astype(np.float64)
        tr = Trace(data=data)
        tr.stats.network = net
        tr.stats.station = sta
        tr.stats.location = loc
        tr.stats.channel = chan
        tr.stats.sampling_rate = rate
        tr.stats.starttime = start
        st.append(tr)
    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write(str(out_path), format="MSEED")


def _write_catalog(paths: RunPaths, events: dict[str, list[tuple[str, str, UTCDateTime]]]) -> None:
    """Write a minimal QuakeML catalogue with only picks (no origins needed --
    ``waveform_cache`` reads ``event.picks`` directly).

    ``events`` is ``{event_id: [(net_sta, phase_hint, time), ...]}``.
    """
    cat = Catalog()
    for event_id, picks in events.items():
        ev = make_event(event_id)
        for net_sta, phase, t in picks:
            net, sta = net_sta.split(".", 1)
            pick = Pick(
                time=t,
                phase_hint=phase,
                waveform_id=WaveformStreamID(
                    network_code=net, station_code=sta, channel_code=f"HH{phase}"
                ),
            )
            ev.picks.append(pick)
        cat.append(ev)
    write_catalog(cat, paths.catalog_mag_xml)


def _load_npz(paths: RunPaths, event_id: str):
    return np.load(paths.relocate_cache_file(event_id))


# ---------------------------------------------------------------------------
# test_window_covers_configured_xcorr_range
# ---------------------------------------------------------------------------


def test_window_covers_configured_xcorr_range(tmp_path):
    cfg = _make_cfg(tmp_path, window_p=(-0.05, 0.2), window_s=(-0.05, 0.3), maxlag_s=0.05)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 15.0
    s_time = t0 + 17.0
    event_id = event_id_from_time(t0)

    _write_chunk(
        paths,
        event_id,
        [
            ("XX", "STA1", "00", "HHZ", 100.0, t0, 3000),
            ("XX", "STA1", "00", "HHN", 100.0, t0, 3000),
            ("XX", "STA1", "00", "HHE", 100.0, t0, 3000),
        ],
    )
    _write_catalog(paths, {event_id: [("XX.STA1", "P", p_time), ("XX.STA1", "S", s_time)]})

    build_cache(cfg, paths, [event_id])

    data = _load_npz(paths, event_id)
    seed_ids = list(data["seed_ids"])
    assert "XX.STA1.00.HHZ" in seed_ids
    assert "XX.STA1.00.HHN" in seed_ids or "XX.STA1.00.HHE" in seed_ids

    z_start = float(data["XX.STA1.00.HHZ__start"])
    z_data = data["XX.STA1.00.HHZ__data"]
    rate = float(data["sampling_rate"])
    z_end = z_start + (len(z_data) - 1) / rate
    dt = 1.0 / rate

    p_before, p_after = cfg.hypodd.xcorr.window_p
    range_start = (p_time + p_before).timestamp
    range_end = (p_time + p_after).timestamp

    # Strictly contains the configured analysis range (small dt tolerance
    # for slice-to-nearest-sample rounding)...
    assert z_start <= range_start + dt
    assert z_end >= range_end - dt
    # ...with genuine margin left over (maxlag_s room for M10.3's lag
    # search), not merely the bare configured range.
    assert range_start - z_start >= cfg.hypodd.xcorr.maxlag_s - dt
    assert z_end - range_end >= cfg.hypodd.xcorr.maxlag_s - dt


# ---------------------------------------------------------------------------
# test_no_hardcoded_window_truncation
# ---------------------------------------------------------------------------


def test_no_hardcoded_window_truncation(tmp_path):
    # Wider than hypoddpy's audited hardcoded 1s-before/5s-after window.
    cfg = _make_cfg(tmp_path, window_s=(-2.0, 8.0), maxlag_s=0.05)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    s_time = t0 + 15.0
    event_id = event_id_from_time(t0)

    _write_chunk(paths, event_id, [("XX", "STA1", "00", "HHN", 100.0, t0, 3000)])
    _write_catalog(paths, {event_id: [("XX.STA1", "S", s_time)]})

    build_cache(cfg, paths, [event_id])

    data = _load_npz(paths, event_id)
    seed_id = "XX.STA1.00.HHN"
    assert seed_id in list(data["seed_ids"])

    rate = float(data["sampling_rate"])
    start = float(data[f"{seed_id}__start"])
    npts = len(data[f"{seed_id}__data"])
    end = start + (npts - 1) / rate
    dt = 1.0 / rate

    before, after = cfg.hypodd.xcorr.window_s
    # hypoddpy's audited defect would have truncated this to [-1, +5] s
    # regardless of config; assert the full configured range is covered.
    assert start <= (s_time + before).timestamp + dt
    assert end >= (s_time + after).timestamp - dt


# ---------------------------------------------------------------------------
# test_resampling_to_common_rate
# ---------------------------------------------------------------------------


def test_resampling_to_common_rate(tmp_path):
    cfg = _make_cfg(tmp_path, resample_rate_hz=50.0)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    p_time = t0 + 15.0
    event_id = event_id_from_time(t0)

    _write_chunk(
        paths,
        event_id,
        [
            ("XX", "AAA", "00", "HHZ", 100.0, t0, 3000),
            ("XX", "BBB", "00", "HHZ", 80.0, t0, 2400),
        ],
    )
    _write_catalog(
        paths,
        {event_id: [("XX.AAA", "P", p_time), ("XX.BBB", "P", p_time)]},
    )

    build_cache(cfg, paths, [event_id])

    data = _load_npz(paths, event_id)
    assert float(data["sampling_rate"]) == pytest.approx(50.0)
    assert "XX.AAA.00.HHZ" in list(data["seed_ids"])
    assert "XX.BBB.00.HHZ" in list(data["seed_ids"])


# ---------------------------------------------------------------------------
# test_skip_missing_chunk_logs_and_continues
# ---------------------------------------------------------------------------


def test_skip_missing_chunk_logs_and_continues(tmp_path, caplog):
    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    missing_event_id = event_id_from_time(t0)

    with caplog.at_level("WARNING", logger="openmeq.relocate.waveform_cache"):
        build_cache(cfg, paths, [missing_event_id])  # must not raise

    assert not paths.relocate_cache_file(missing_event_id).exists()
    assert any(
        missing_event_id in record.message and "missing_chunk" in record.message
        for record in caplog.records
    )


def test_skip_missing_chunk_continues_with_other_events(tmp_path):
    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    missing_event_id = event_id_from_time(t0)
    ok_event_id = event_id_from_time(t0 + 60.0)

    p_time = (t0 + 60.0) + 15.0
    _write_chunk(paths, ok_event_id, [("XX", "STA1", "00", "HHZ", 100.0, t0 + 60.0, 3000)])
    _write_catalog(paths, {ok_event_id: [("XX.STA1", "P", p_time)]})

    build_cache(cfg, paths, [missing_event_id, ok_event_id])

    assert not paths.relocate_cache_file(missing_event_id).exists()
    assert paths.relocate_cache_file(ok_event_id).exists()


# ---------------------------------------------------------------------------
# test_skip_unchanged
# ---------------------------------------------------------------------------


def test_skip_unchanged(tmp_path):
    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    event_id = event_id_from_time(t0)
    p_time = t0 + 15.0

    _write_chunk(paths, event_id, [("XX", "STA1", "00", "HHZ", 100.0, t0, 3000)])
    _write_catalog(paths, {event_id: [("XX.STA1", "P", p_time)]})

    build_cache(cfg, paths, [event_id])
    cache_path = paths.relocate_cache_file(event_id)
    assert cache_path.exists()
    mtime1 = cache_path.stat().st_mtime_ns

    build_cache(cfg, paths, [event_id])  # nothing changed
    mtime2 = cache_path.stat().st_mtime_ns

    assert mtime1 == mtime2


def test_rebuilds_when_config_changes(tmp_path):
    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    event_id = event_id_from_time(t0)
    p_time = t0 + 15.0

    _write_chunk(paths, event_id, [("XX", "STA1", "00", "HHZ", 100.0, t0, 3000)])
    _write_catalog(paths, {event_id: [("XX.STA1", "P", p_time)]})

    build_cache(cfg, paths, [event_id])
    cache_path = paths.relocate_cache_file(event_id)
    mtime1 = cache_path.stat().st_mtime_ns

    cfg.hypodd.xcorr.maxlag_s = 0.5  # config change -> fingerprint changes
    build_cache(cfg, paths, [event_id])
    mtime2 = cache_path.stat().st_mtime_ns

    assert mtime2 != mtime1


# ---------------------------------------------------------------------------
# test_synthetic_project_cache_builds
# ---------------------------------------------------------------------------


def test_synthetic_project_cache_builds(synthetic_project, tmp_path):
    from openmeq.config import load_config

    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    cfg.parallel.n_workers = 1
    index = build_index(cfg)
    paths = RunPaths(cfg)
    paths.ensure()

    rows = []
    picks_by_event: dict[str, list[tuple[str, str, UTCDateTime]]] = {}
    for ev in truth["events"]:
        event_id = ev["event_id_hint"]
        t = UTCDateTime(ev["origin"]["time"])
        stations = sorted(ev["arrivals"].keys())
        rows.append(
            {
                "event_id": event_id,
                "time": str(t),
                "duration_s": 5.0,
                "coincidence_sum": len(stations),
                "n_stations": len(stations),
                "stations": ";".join(stations),
                "trace_ids": "",
            }
        )
        picks = []
        for net_sta, phases in ev["arrivals"].items():
            if phases.get("P") is not None:
                picks.append((net_sta, "P", UTCDateTime(phases["P"])))
            if phases.get("S") is not None:
                picks.append((net_sta, "S", UTCDateTime(phases["S"])))
        picks_by_event[event_id] = picks

    detections_df = pd.DataFrame(rows)
    write_chunks(detections_df, index, cfg, paths)
    _write_catalog(paths, picks_by_event)

    event_ids = [ev["event_id_hint"] for ev in truth["events"]]
    build_cache(cfg, paths, event_ids)

    for event_id in event_ids:
        cache_path = paths.relocate_cache_file(event_id)
        assert cache_path.exists(), f"no cache written for {event_id}"
        data = np.load(cache_path)
        assert "sampling_rate" in data
        assert float(data["sampling_rate"]) == pytest.approx(cfg.hypodd.xcorr.resample_rate_hz)
