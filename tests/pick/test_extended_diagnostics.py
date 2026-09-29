"""Tests for ``openmeq.pick.extended_diagnostics`` (``OpenMEQ-02o.4``).

Hand-built ``Catalog``/``chunks_pn/``/``result.h5``/``detections.csv``
fixtures (mirrors ``tests/pick/test_diagnostics.py``) -- no PhaseNet
needed. The synthetic ``result.h5`` puts a Gaussian bump of height
``probability`` at every pick's own sample, in the sorted-station-key
column order the module verifies against.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Pick, WaveformStreamID

from openmeq.catalog import get_extra, make_event, set_extra
from openmeq.config import load_config
from openmeq.paths import RunPaths
from openmeq.pick import extended_diagnostics as ext
from openmeq.pick.diagnostics import make_pick_diagnostics

_START = UTCDateTime(2025, 6, 1, 0, 0, 0)
_NPTS = 6000
_SR = 100.0
_EVENT_ID = "ev20250601T000030_000"
_DET_TIME = _START + 30.0

#: ``(station, P offset s, S offset s or None)`` from ``_START``; Vp/Vs = 1.75.
_ARRIVALS = [("STA3", 22.0, None), ("STA1", 20.0, 20.0 + 0.75 * 4.0), ("STA2", 21.0, 21.0 + 3.75)]


def _station_key(sta: str) -> str:
    return f"XX.{sta}.00.HH"


def _make_trace(sta: str, channel: str, arrivals: list[float]) -> Trace:
    rng = np.random.default_rng(abs(hash((sta, channel))) % (2**32))
    t = np.arange(_NPTS) / _SR
    data = rng.normal(scale=0.1, size=_NPTS)
    for ta in arrivals:
        env = np.where(t >= ta, np.exp(-(t - ta) / 0.4), 0.0)
        data += env * np.sin(2 * np.pi * 8.0 * (t - ta))
    return Trace(
        data=data.astype(np.float64),
        header={
            "network": "XX",
            "station": sta,
            "location": "00",
            "channel": channel,
            "starttime": _START,
            "sampling_rate": _SR,
        },
    )


def _make_pick(sta: str, cha: str, phase: str, t: UTCDateTime, probability: float) -> Pick:
    wid = WaveformStreamID(
        network_code="XX", station_code=sta, location_code="00", channel_code=cha
    )
    pick = Pick(time=t, waveform_id=wid, phase_hint=phase, evaluation_mode="automatic")
    set_extra(pick, "probability", probability)
    return pick


def _write_inputs(paths: RunPaths, *, with_h5: bool = True, extra_bump=None) -> Catalog:
    """chunks_pn (stations in unsorted order), detections.csv, optional result.h5."""
    traces = []
    picks = []
    for sta, tp, ts in _ARRIVALS:
        traces.append(_make_trace(sta, "HHZ", [tp] + ([ts] if ts else [])))
        horiz = [ts] if ts else []
        traces.append(_make_trace(sta, "HHN", horiz))
        traces.append(_make_trace(sta, "HHE", horiz))
        picks.append(_make_pick(sta, "HHZ", "P", _START + tp, 0.9))
        if ts:
            picks.append(_make_pick(sta, "HHN", "S", _START + ts, 0.7))
    out = paths.chunk_pn_file(_EVENT_ID)
    out.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces).write(str(out), format="MSEED")

    paths.detections_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"event_id": [_EVENT_ID], "time": [str(_DET_TIME)]}).to_csv(
        paths.detections_csv, index=False
    )

    if with_h5:
        keys = sorted(_station_key(sta) for sta, _, _ in _ARRIVALS)
        prob = np.zeros((_NPTS, len(keys), 3), dtype=np.float32)
        idx = np.arange(_NPTS)
        bumps = [
            (
                _station_key(p.waveform_id.station_code),
                p.phase_hint,
                p.time - _START,
                float(get_extra(p, "probability")),
            )
            for p in picks
        ]
        if extra_bump is not None:
            bumps.append(extra_bump)
        for key, phase, t, height in bumps:
            ch = 1 if phase == "P" else 2
            prob[:, keys.index(key), ch] += height * np.exp(-0.5 * ((idx - t * _SR) / 10.0) ** 2)
        paths.phasenet_raw_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(paths.phasenet_raw_dir / "result.h5", "w") as f:
            f.create_dataset(f"{_EVENT_ID}/{_EVENT_ID}.mseed", data=prob)

    event = make_event(_EVENT_ID)
    event.picks.extend(picks)
    return Catalog(events=[event])


@pytest.fixture
def ext_project(tmp_path, synthetic_project):
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    cfg.diagnostics.pick.extended = True
    cfg.diagnostics.pick.pick_zoom.half_window_s = 1.5
    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, paths


def _extended_dir(paths: RunPaths):
    return paths.diagnostics_dir("pick") / ext.EXTENDED_SUBDIR


def _texts(fig) -> str:
    return " | ".join(t.get_text() for ax in fig.axes for t in ax.texts)


def _read_stream(paths):
    from obspy import read

    return read(str(paths.chunk_pn_file(_EVENT_ID)))


def _read_prob(paths):
    with h5py.File(paths.phasenet_raw_dir / "result.h5", "r") as f:
        return np.asarray(f[f"{_EVENT_ID}/{_EVENT_ID}.mseed"])


# ---------------------------------------------------------------------------
# enabled / disabled
# ---------------------------------------------------------------------------


def test_enabled_writes_every_figure(ext_project):
    cfg, paths = ext_project
    catalog = _write_inputs(paths)

    make_pick_diagnostics(paths, catalog, cfg)

    out = _extended_dir(paths)
    for name in [f"pick_zoom_{_EVENT_ID}.png", "wadati_prelocate.png", "pick_offset_hist.png"]:
        assert (out / name).is_file(), name
        assert (out / name).stat().st_size > 0
    # Baseline set unchanged and still written.
    assert (paths.diagnostics_dir("pick") / f"example_{_EVENT_ID}.png").is_file()


def test_zoom_window_width_and_prob_overlay(ext_project):
    cfg, paths = ext_project
    catalog = _write_inputs(paths)

    fig = ext._build_pick_zoom_figure(
        catalog[0], _EVENT_ID, _read_stream(paths), _read_prob(paths), cfg
    )
    try:
        visible = [ax for ax in fig.axes if ax.axison]
        # 3 stations x (P, S) minus STA3's missing S = 5 panels, each with a prob twin.
        assert len(visible) == 10
        for ax in visible:
            lo, hi = ax.get_xlim()
            assert hi - lo == pytest.approx(2 * cfg.diagnostics.pick.pick_zoom.half_window_s)
            assert lo == pytest.approx(-1.5)
        texts = _texts(fig)
        assert "P p=0.90" in texts and "S p=0.70" in texts
        assert "no S pick" in texts  # STA3 blank panel
        assert "(prob)" in fig._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_disabled_writes_no_extended_dir(ext_project):
    cfg, paths = ext_project
    cfg.diagnostics.pick.extended = None  # inherit global default (False)
    catalog = _write_inputs(paths)

    make_pick_diagnostics(paths, catalog, cfg)
    assert not _extended_dir(paths).exists()
    # No cfg at all (legacy call) -> baseline only, too.
    make_pick_diagnostics(paths, catalog)
    assert not _extended_dir(paths).exists()
    assert (paths.diagnostics_dir("pick") / "summary.csv").is_file()


# ---------------------------------------------------------------------------
# result.h5 absent / untrustworthy
# ---------------------------------------------------------------------------


def test_missing_result_h5(ext_project):
    cfg, paths = ext_project
    catalog = _write_inputs(paths, with_h5=False)

    make_pick_diagnostics(paths, catalog, cfg)
    assert (_extended_dir(paths) / f"pick_zoom_{_EVENT_ID}.png").is_file()

    fig = ext._build_pick_zoom_figure(catalog[0], _EVENT_ID, _read_stream(paths), None, cfg)
    try:
        assert len([ax for ax in fig.axes if ax.axison]) == 5  # no twin axes
        assert "(no prob)" in fig._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_mismatched_result_h5_disables_overlay(ext_project):
    cfg, paths = ext_project
    catalog = _write_inputs(paths)
    prob = _read_prob(paths)[:, ::-1, :]  # wrong column order -> probabilities don't reproduce

    stream = _read_stream(paths)
    begin = min(tr.stats.starttime for tr in stream)
    assert ext._prob_columns(stream, prob, catalog[0], begin, _SR) == {}
    assert ext._prob_columns(stream, prob[:, :2, :], catalog[0], begin, _SR) == {}
    assert len(ext._prob_columns(stream, _read_prob(paths), catalog[0], begin, _SR)) == 3


# ---------------------------------------------------------------------------
# show_rejected
# ---------------------------------------------------------------------------


def test_show_rejected(ext_project):
    cfg, paths = ext_project
    # Sub-threshold S bump on STA1, 0.8 s after its S pick (inside the window).
    s_time = 20.0 + 3.0
    catalog = _write_inputs(paths, extra_bump=(_station_key("STA1"), "S", s_time + 0.8, 0.2))
    rejected = [
        {
            "event_id": _EVENT_ID,
            "station_id": _station_key("STA1"),
            "phase_type": "P",
            "time": _START + 20.4,
            "probability": 0.5,
            "reason": "duplicate",
        },
        {
            "event_id": _EVENT_ID,
            "station_id": _station_key("STA3"),
            "phase_type": "S",
            "time": _START + 26.0,
            "probability": 0.4,
            "reason": "unresolved channel",
        },
        {  # another event's record -- must be ignored
            "event_id": "ev_other",
            "station_id": _station_key("STA2"),
            "phase_type": "P",
            "time": _START + 21.2,
            "probability": 0.6,
            "reason": "duplicate",
        },
    ]
    stream, prob = _read_stream(paths), _read_prob(paths)
    import matplotlib.pyplot as plt

    fig = ext._build_pick_zoom_figure(catalog[0], _EVENT_ID, stream, prob, cfg, rejected)
    try:
        assert "duplicate" not in _texts(fig)  # show_rejected is False by default
        assert "no S pick" in _texts(fig)
    finally:
        plt.close(fig)

    cfg.diagnostics.pick.pick_zoom.show_rejected = True
    fig = ext._build_pick_zoom_figure(catalog[0], _EVENT_ID, stream, prob, cfg, rejected)
    try:
        texts = _texts(fig)
        assert "0.50 duplicate" in texts
        assert "0.40 unresolved channel" in texts
        assert "< min_s_prob" in texts
        assert "0.60" not in texts  # other event's record
        assert "no S pick" not in texts  # STA3's S panel now shows the rejected candidate
    finally:
        plt.close(fig)

    make_pick_diagnostics(paths, catalog, cfg, rejected=rejected)
    assert (_extended_dir(paths) / f"pick_zoom_{_EVENT_ID}.png").is_file()


# ---------------------------------------------------------------------------
# wadati / offsets / failure isolation
# ---------------------------------------------------------------------------


def test_wadati_points_and_offsets(ext_project):
    cfg, paths = ext_project
    catalog = _write_inputs(paths)

    df = ext._wadati_points(catalog)
    assert sorted(df["station"]) == ["XX.STA1", "XX.STA2"]  # STA3 has no S
    slope, _ = np.polyfit(df["p_rel_s"], df["s_minus_p_s"], 1)
    assert 1.0 + slope == pytest.approx(1.75)

    assert ext._detection_times(paths) == {_EVENT_ID: _DET_TIME}


def test_one_failing_figure_does_not_fail_the_step(ext_project, monkeypatch):
    cfg, paths = ext_project
    catalog = _write_inputs(paths)

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ext, "_plot_wadati", _boom)
    monkeypatch.setattr(ext, "_build_pick_zoom_figure", _boom)
    make_pick_diagnostics(paths, catalog, cfg)

    out = _extended_dir(paths)
    assert (out / "pick_offset_hist.png").is_file()
    assert not (out / "wadati_prelocate.png").exists()
    assert not (out / f"pick_zoom_{_EVENT_ID}.png").exists()
