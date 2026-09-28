"""Tests for ``openmeq.pick.diagnostics`` (M4.4, ``OpenMEQ-1m7.4``).

Works purely from hand-built ``Catalog``/``chunks_pn/`` fixtures (mirrors
``tests/pick/test_to_quakeml.py``'s ``hand_built_project`` pattern) -- no
PhaseNet needed, since :func:`~openmeq.pick.diagnostics.make_pick_diagnostics`
never recomputes picking (module docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Pick, WaveformStreamID

from openmeq.catalog import make_event, set_extra
from openmeq.config import load_config
from openmeq.paths import RunPaths
from openmeq.pick.diagnostics import make_pick_diagnostics

_START = UTCDateTime(2025, 6, 1, 0, 0, 0)
_NPTS = 6000
_SR = 100.0


def _make_trace(network, station, location, channel) -> Trace:
    rng = np.random.default_rng(abs(hash((network, station, location, channel))) % (2**32))
    data = rng.normal(scale=1.0, size=_NPTS).astype(np.float64)
    return Trace(
        data=data,
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "starttime": _START,
            "sampling_rate": _SR,
        },
    )


def _make_pick(net, sta, loc, cha, phase_hint, t, probability) -> Pick:
    wid = WaveformStreamID(network_code=net, station_code=sta, location_code=loc, channel_code=cha)
    pick = Pick(
        time=t,
        waveform_id=wid,
        phase_hint=phase_hint,
        method_id="smi:openmeq/phasenet",
        evaluation_mode="automatic",
    )
    pick.time_errors.uncertainty = 0.1
    set_extra(pick, "probability", probability)
    return pick


@pytest.fixture
def diag_project(tmp_path, synthetic_project):
    config_path, _truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, paths


def _write_chunk_pn(paths: RunPaths, event_id: str, stations: list[tuple[str, str, str]]) -> None:
    """``stations``: list of ``(network, station, location)``; each gets a full Z/N/E trio."""
    traces = []
    for net, sta, loc in stations:
        traces += [
            _make_trace(net, sta, loc, "HHZ"),
            _make_trace(net, sta, loc, "HHN"),
            _make_trace(net, sta, loc, "HHE"),
        ]
    stream = Stream(traces)
    out_path = paths.chunk_pn_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(out_path), format="MSEED")


def _build_catalog(event_specs: dict[str, list[Pick]]) -> Catalog:
    """``event_specs``: ``{event_id: [Pick, ...]}``."""
    catalog = Catalog()
    for event_id, picks in event_specs.items():
        event = make_event(event_id)
        event.picks.extend(picks)
        catalog.events.append(event)
    return catalog


# ---------------------------------------------------------------------------
# test_diagnostics_outputs_and_summary
# ---------------------------------------------------------------------------


def test_diagnostics_outputs_and_summary(diag_project):
    cfg, paths = diag_project
    del cfg

    event_id_1 = "ev20250601T000000_000"
    event_id_2 = "ev20250601T010000_000"
    _write_chunk_pn(paths, event_id_1, [("XX", "STA1", "00"), ("XX", "STA2", "00")])
    _write_chunk_pn(paths, event_id_2, [("XX", "STA1", "00")])

    catalog = _build_catalog(
        {
            event_id_1: [
                _make_pick("XX", "STA1", "00", "HHZ", "P", _START + 20.0, 0.9),
                _make_pick("XX", "STA1", "00", "HHN", "S", _START + 32.0, 0.8),
                _make_pick("XX", "STA2", "00", "HHZ", "P", _START + 21.0, 0.7),
            ],
            event_id_2: [
                _make_pick("XX", "STA1", "00", "HHZ", "P", _START + 20.5, 0.95),
            ],
        }
    )

    diagnostics_dir = make_pick_diagnostics(paths, catalog)
    assert diagnostics_dir == paths.diagnostics_dir("pick")

    expected_pngs = [
        "p_pick_counts_per_station.png",
        "s_pick_counts_per_station.png",
        "p_probability_hist.png",
        "s_probability_hist.png",
        "picks_per_event_hist.png",
        f"example_{event_id_1}.png",
        f"example_{event_id_2}.png",
    ]
    for name in expected_pngs:
        p = diagnostics_dir / name
        assert p.is_file(), f"{p} was not created"
        assert p.stat().st_size > 0

    summary_path = diagnostics_dir / "summary.csv"
    assert summary_path.is_file()
    summary_df = pd.read_csv(summary_path)
    assert list(summary_df.columns) == ["metric", "value"]
    metrics = dict(zip(summary_df["metric"], summary_df["value"], strict=True))
    assert int(metrics["n_events"]) == 2
    assert int(metrics["n_picks_total"]) == 4
    assert int(metrics["n_p_picks"]) == 3
    assert int(metrics["n_s_picks"]) == 1


# ---------------------------------------------------------------------------
# test_no_exception_on_zero_pick_event / empty catalog
# ---------------------------------------------------------------------------


def test_no_exception_on_zero_pick_event(diag_project):
    """An ``Event`` with zero picks, and an entirely empty ``Catalog``, must
    not raise."""
    cfg, paths = diag_project
    del cfg

    event_id = "ev20250601T020000_000"
    _write_chunk_pn(paths, event_id, [("XX", "STA1", "00")])

    zero_pick_event = make_event(event_id)  # no picks appended
    catalog = Catalog(events=[zero_pick_event])

    diagnostics_dir = make_pick_diagnostics(paths, catalog)
    assert (diagnostics_dir / "summary.csv").is_file()
    summary_df = pd.read_csv(diagnostics_dir / "summary.csv")
    metrics = dict(zip(summary_df["metric"], summary_df["value"], strict=True))
    assert int(metrics["n_events"]) == 1
    assert int(metrics["n_picks_total"]) == 0

    # Fully empty catalog must also not raise.
    empty_diag_dir = make_pick_diagnostics(paths, Catalog())
    assert (empty_diag_dir / "summary.csv").is_file()


# ---------------------------------------------------------------------------
# test_missing_chunk_pn_file_skips_example_plot
# ---------------------------------------------------------------------------


def test_missing_chunk_pn_file_skips_example_plot(diag_project):
    """An event whose ``chunks_pn`` file is missing must not raise -- its
    example plot is simply skipped."""
    cfg, paths = diag_project
    del cfg

    event_id = "ev_missing_chunk_pn"
    catalog = _build_catalog(
        {event_id: [_make_pick("XX", "STA1", "00", "HHZ", "P", _START + 20.0, 0.9)]}
    )

    diagnostics_dir = make_pick_diagnostics(paths, catalog)
    assert not (diagnostics_dir / f"example_{event_id}.png").exists()
    assert (diagnostics_dir / "summary.csv").is_file()
