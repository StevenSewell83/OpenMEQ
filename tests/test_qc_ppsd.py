"""Tests for ``openmeq.qc.ppsd`` (OpenMEQ-itl).

Uses obspy's bundled example inventory (``read_inventory()``), whose
``BW.RJOB..EHZ`` channel has a full response at 200 Hz, with synthetic noise
traces under that id -- no archive or network access needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime, read_inventory

from openmeq.config import QcPpsdConfig
from openmeq.qc import ppsd as qc_ppsd

SEED_ID = "BW.RJOB..EHZ"
DAY = "2009-08-24"  # inside RJOB's metadata epoch
_SR = 200.0


def _params(**overrides) -> dict:
    cfg = QcPpsdConfig(**overrides).model_dump(mode="json")
    return {k: cfg[k] for k in qc_ppsd.COMPUTE_KEYS}


class _FakeIndex:
    """Stands in for DataIndex: ``read`` returns ``hours`` of noise from the day start."""

    def __init__(self, hours: float, seed_id: str = SEED_ID):
        self.hours = hours
        self.seed_id = seed_id

    def read(self, seed_ids, t1, t2, pad=0.0):
        if self.hours <= 0:
            return Stream()
        net, sta, loc, cha = self.seed_id.split(".")
        n = int(self.hours * 3600 * _SR)
        data = np.random.default_rng(0).normal(scale=100.0, size=n).astype(np.int32)
        header = {
            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,
            "sampling_rate": _SR,
            "starttime": t1,
        }
        return Stream([Trace(data=data, header=header)])


@pytest.fixture(scope="module")
def inv():
    return read_inventory()


def test_channel_day_statuses(tmp_path, inv):
    params = _params()
    status = qc_ppsd.ppsd_channel_day(SEED_ID, DAY, _FakeIndex(3.0), inv, params, tmp_path)
    assert status == "ok"
    npz = tmp_path / SEED_ID / f"{DAY}.npz"
    assert npz.is_file()
    # Already stored -> skipped.
    assert qc_ppsd.ppsd_channel_day(SEED_ID, DAY, _FakeIndex(3.0), inv, params, tmp_path) == "done"

    # No data -> empty marker, so the day isn't retried.
    status = qc_ppsd.ppsd_channel_day(SEED_ID, "2009-08-25", _FakeIndex(0), inv, params, tmp_path)
    assert status == "no_data"
    assert (tmp_path / SEED_ID / "2009-08-25.empty").is_file()

    # Shorter than one ppsd_length_s segment.
    status = qc_ppsd.ppsd_channel_day(SEED_ID, "2009-08-26", _FakeIndex(0.2), inv, params, tmp_path)
    assert status == "no_segments"

    # No response for this channel: skipped without a marker (retried later).
    other = "XX.NONE..EHZ"
    status = qc_ppsd.ppsd_channel_day(other, DAY, _FakeIndex(3.0, other), inv, params, tmp_path)
    assert status == "no_response"
    assert not (tmp_path / other).exists()


def test_select_seed_ids():
    cfg = SimpleNamespace(qc=SimpleNamespace(ppsd=QcPpsdConfig(), exclude_channels=["XX.B..HHZ"]))
    ids = ["XX.A..HHZ", "XX.A..HHN", "XX.B..HHZ", "XX.C..EHZ"]
    assert qc_ppsd.select_seed_ids(ids, cfg) == ["XX.A..HHZ", "XX.C..EHZ"]
    cfg.qc.ppsd = QcPpsdConfig(components=[])
    assert qc_ppsd.select_seed_ids(ids, cfg) == ["XX.A..HHN", "XX.A..HHZ", "XX.C..EHZ"]


def test_store_cleared_when_settings_change(tmp_path):
    stamp = {"version": 1, "ppsd": _params(), "stations_xml_sha256": "a"}
    qc_ppsd._prepare_store(tmp_path, stamp, force=False)
    day = tmp_path / SEED_ID / f"{DAY}.npz"
    day.parent.mkdir()
    day.touch()

    qc_ppsd._prepare_store(tmp_path, stamp, force=False)  # unchanged: kept
    assert day.exists()
    changed = {**stamp, "stations_xml_sha256": "b"}  # new responses: cleared
    qc_ppsd._prepare_store(tmp_path, changed, force=False)
    assert not day.exists()
    assert json.loads((tmp_path / "settings.json").read_text()) == changed

    day.parent.mkdir()
    day.touch()
    qc_ppsd._prepare_store(tmp_path, changed, force=True)  # --force: cleared
    assert not day.exists()


def test_plot_channel_and_medians(tmp_path, inv):
    store = tmp_path / "store"
    params = _params()
    for day in ("2009-08-24", "2009-08-25"):
        assert qc_ppsd.ppsd_channel_day(SEED_ID, day, _FakeIndex(3.0), inv, params, store) == "ok"
    files = sorted(str(f) for f in (store / SEED_ID).glob("*.npz"))
    out = tmp_path / "figs"
    out.mkdir()
    plot = {"xaxis_frequency": True, "temporal_periods_s": [0.05, 0.2, 1000.0]}
    result = qc_ppsd._plot_channel((SEED_ID, files, str(out), plot))
    names = sorted(p.split("/")[-1] for p in result["written"])
    assert names == [f"{SEED_ID}_ppsd.png", f"{SEED_ID}_spectrogram.png", f"{SEED_ID}_temporal.png"]
    assert result["n_psd"] > 0
    assert len(result["periods"]) == len(result["median_db"])

    cfg = SimpleNamespace(
        project=SimpleNamespace(name="test"), qc=SimpleNamespace(ppsd=QcPpsdConfig())
    )
    medians = out / "ppsd_medians.png"
    qc_ppsd._plot_medians([result], {SEED_ID: "GS-11D"}, medians, cfg)
    assert medians.stat().st_size > 0
    assert qc_ppsd._plot_channel((SEED_ID, [], str(out), plot)) is None


def test_ppsd_config_validation():
    with pytest.raises(ValueError):
        QcPpsdConfig(db_bins=(-50.0, -200.0, 1.0))
    with pytest.raises(ValueError):
        QcPpsdConfig(period_limits=(10.0, 0.1))
    assert QcPpsdConfig().enabled is False


def test_ppsd_not_in_qc_fingerprint():
    """Enabling/tuning PPSD must not invalidate existing qc manifests."""
    from openmeq.config import QcConfig

    assert (
        QcConfig().model_dump_json()
        == QcConfig(ppsd=QcPpsdConfig(enabled=True, components=[])).model_dump_json()
    )


def test_compute_ppsd_requires_qc_outputs(tmp_path):
    paths = SimpleNamespace(
        index_json=tmp_path / "index.json", stations_xml=tmp_path / "stations.xml"
    )
    with pytest.raises(FileNotFoundError, match="openmeq qc"):
        qc_ppsd.compute_ppsd(SimpleNamespace(), paths)


def test_day_start_is_utc_midnight():
    assert UTCDateTime(DAY) == UTCDateTime(2009, 8, 24)


def test_qc_step_ppsd_gate(monkeypatch):
    """qc runs PPSD only when enabled, and a PPSD failure never fails the step."""
    import openmeq.qc as qc

    calls = []
    monkeypatch.setattr(qc, "run_ppsd", lambda cfg, paths, **kw: calls.append(kw))
    cfg = SimpleNamespace(qc=SimpleNamespace(ppsd=QcPpsdConfig()))
    qc._ppsd(cfg, None, force=False)
    assert calls == []
    cfg.qc.ppsd = QcPpsdConfig(enabled=True)
    qc._ppsd(cfg, None, force=True)
    assert calls == [{"force": True, "instruments": None}]

    def boom(cfg, paths, **kw):
        raise RuntimeError("bad")

    monkeypatch.setattr(qc, "run_ppsd", boom)
    qc._ppsd(cfg, None, force=False)  # logged, not raised
