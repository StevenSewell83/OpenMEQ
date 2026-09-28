"""Tests for the synthetic dataset generator itself (``tests/data/make_synthetic.py``,
M0.8), in isolation from the ``synthetic_project`` fixture.

ARCHITECTURE.md §8 is the authoritative spec for what this dataset must
contain; ``tests/data/make_synthetic.py``'s module docstring documents the
exact travel-time method, ML calibration, anomaly locations and truth
structure these tests check against.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from obspy import UTCDateTime, read

sys.path.insert(0, str(Path(__file__).parent / "data"))
import make_synthetic  # noqa: E402

from openmeq.config import QcConfig, load_config

QC_DEFAULTS = QcConfig()


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, dict]:
    """One shared generation for every test in this module except
    ``test_deterministic`` (which needs two independent generations)."""
    outdir = tmp_path_factory.mktemp("make_synthetic")
    truth = make_synthetic.generate(outdir)
    return outdir, truth


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: Representative subset of generated files checksummed by test_deterministic
#: (M0.9 / OpenMEQ-qlg.9 trim: a full byte-for-byte comparison needs both
#: ~500 MB generations on disk at once and reads every generated file twice
#: -- unnecessary given generation itself, not hashing, dominates this
#: test's cost; see that test for the disk-saving trick this enables). The
#: three "known issue" channels are picked deliberately -- they're where a
#: stray/reordered RNG draw would first show up -- plus one plain ("quiet")
#: channel and every non-waveform output file, so this still exercises
#: every *kind* of thing generate() writes, not just a handful of arbitrary
#: waveform files.
def _determinism_check_paths(outdir: Path) -> list[Path]:
    checks = [
        Path("config.yaml"),
        Path("truth.json"),
        Path("inventory/stations.xml"),
        Path("inventory/stations.csv"),
        Path("models/vel1d.txt"),
        Path("waveforms")
        / make_synthetic.GAP_START.strftime("%Y-%m-%d")
        / f"{make_synthetic.GAP_SEED_ID}.mseed",
        Path("waveforms")
        / make_synthetic.SPIKE_START.strftime("%Y-%m-%d")
        / f"{make_synthetic.SPIKE_SEED_ID}.mseed",
        Path("waveforms")
        / make_synthetic.FLAT_START.strftime("%Y-%m-%d")
        / f"{make_synthetic.FLAT_SEED_ID}.mseed",
        Path("waveforms")
        / make_synthetic.DAY0.strftime("%Y-%m-%d")
        / f"{make_synthetic.NETWORK}.STA01.00.HHZ.mseed",
    ]
    for rel in checks:
        assert (outdir / rel).is_file(), f"expected generated file missing: {rel}"
    return checks


def test_deterministic(tmp_path_factory):
    out1 = tmp_path_factory.mktemp("det1")
    truth1 = make_synthetic.generate(out1, seed=make_synthetic.DEFAULT_SEED)

    # The full file *listing* (names/paths, not content) is cheap to check
    # and still confirms the same seed produces the same set of files, so
    # it isn't part of the trim -- only per-file content hashing is limited
    # to `check_paths` below.
    files1 = sorted(p.relative_to(out1) for p in out1.rglob("*") if p.is_file())
    check_paths = _determinism_check_paths(out1)
    hashes1 = {rel: _file_sha256(out1 / rel) for rel in check_paths}

    # Free the first ~500 MB generation before writing the second, rather
    # than holding both on disk simultaneously (the actual "cut ... disk"
    # part of this trim -- hashing alone was never the expensive part).
    shutil.rmtree(out1)

    out2 = tmp_path_factory.mktemp("det2")
    truth2 = make_synthetic.generate(out2, seed=make_synthetic.DEFAULT_SEED)

    assert truth1 == truth2

    files2 = sorted(p.relative_to(out2) for p in out2.rglob("*") if p.is_file())
    assert files1 == files2, "same seed should produce the same set of files"

    for rel in check_paths:
        assert _file_sha256(out2 / rel) == hashes1[rel], f"{rel} differs"


def test_station_counts_and_channels(generated):
    _outdir, truth = generated

    stations = truth["stations"]
    assert len(stations) == 8

    surface = [s for s in stations if s["depth_m"] == 0.0]
    borehole = [s for s in stations if s["depth_m"] > 0.0]
    assert len(surface) == 6
    assert len(borehole) == 2

    sta_codes = {(s["net"], s["sta"]) for s in stations}
    hh_channels = []
    dp_channels = []
    for sta_code, _kind, _x_km, _y_km, _elev_m, _depth_m, channels in make_synthetic.STATIONS:
        assert (make_synthetic.NETWORK, sta_code) in sta_codes
        for cha in channels:
            if cha.startswith("HH"):
                hh_channels.append((sta_code, cha))
            elif cha.startswith("DP"):
                dp_channels.append((sta_code, cha))

    hh_stations = {sta for sta, _ in hh_channels}
    dp_stations = {sta for sta, _ in dp_channels}
    assert len(hh_stations) == 6
    assert len(dp_stations) == 2
    assert {cha[-1] for _sta, cha in hh_channels} == {"Z", "N", "E"}
    assert {cha[-1] for _sta, cha in dp_channels} == {"Z", "1", "2"}


def test_example_config_validates(generated):
    outdir, _truth = generated
    cfg = load_config(outdir / "config.yaml")

    assert cfg.project.name == "synthetic"
    assert cfg.data.start == UTCDateTime(2025, 1, 1).date
    assert cfg.data.end == UTCDateTime(2025, 1, 2).date
    assert cfg.stations.stationxml.exists()
    assert cfg.stations.csv.exists()
    assert cfg.nlloc.velocity_model.file.exists()
    assert cfg.hypodd.velocity_model.file.exists()


def _read_full_day(outdir: Path, seed_id: str, date_str: str):
    path = outdir / "waveforms" / date_str / f"{seed_id}.mseed"
    st = read(str(path))
    st.merge(fill_value=0)
    return st[0]


def test_anomalies_present(generated):
    outdir, truth = generated
    issues = {issue["kind"]: issue for issue in truth["known_issues"]}
    assert set(issues) == {"gap", "spikes", "flat"}

    # --- gap: a real time gap, not zero-filled data ---
    gap = issues["gap"]
    st = read(str(outdir / "waveforms" / "2025-01-01" / f"{gap['seed_id']}.mseed"))
    assert len(st) >= 2, "gap channel's day file should have >=2 trace segments"
    gaps = st.get_gaps()
    assert any(
        g[4] <= UTCDateTime(gap["start"]) + 1 and g[5] >= UTCDateTime(gap["end"]) - 1 for g in gaps
    ), f"no detected gap covering the documented window: {gaps}"

    # --- spikes: MAD-based outlier count exceeds qc.spike_threshold_mad ---
    spikes = issues["spikes"]
    date_str = UTCDateTime(spikes["start"]).strftime("%Y-%m-%d")
    tr = _read_full_day(outdir, spikes["seed_id"], date_str)
    data = tr.data.astype(float)
    median = np.median(data)
    mad = np.median(np.abs(data - median)) or 1.0
    outlier_count = int(np.sum(np.abs(data - median) > QC_DEFAULTS.spike_threshold_mad * mad))
    assert outlier_count > 100, (
        f"expected many MAD outliers on the spike channel, got {outlier_count}"
    )

    # a channel with no deliberate anomaly should have far fewer outliers
    quiet_tr = _read_full_day(outdir, f"{make_synthetic.NETWORK}.STA01.00.HHZ", date_str)
    quiet_data = quiet_tr.data.astype(float)
    q_median = np.median(quiet_data)
    q_mad = np.median(np.abs(quiet_data - q_median)) or 1.0
    quiet_outliers = int(
        np.sum(np.abs(quiet_data - q_median) > QC_DEFAULTS.spike_threshold_mad * q_mad)
    )
    assert quiet_outliers < outlier_count

    # --- flat: near-zero variance and a long constant run within the window ---
    flat = issues["flat"]
    date_str = UTCDateTime(flat["start"]).strftime("%Y-%m-%d")
    tr = _read_full_day(outdir, flat["seed_id"], date_str)
    window = tr.slice(UTCDateTime(flat["start"]), UTCDateTime(flat["end"]) - 0.01)
    assert window.stats.npts > QC_DEFAULTS.flat_run_min_samples
    assert np.var(window.data.astype(float)) < 1.0

    vals, counts = np.unique(window.data, return_counts=True)
    assert counts.max() >= QC_DEFAULTS.flat_run_min_samples


def test_event_count_and_arrivals(generated):
    _outdir, truth = generated
    events = truth["events"]

    assert 15 <= len(events) <= 25

    mls = [ev["ml"] for ev in events]
    assert min(mls) < 0.5, "expected at least one weak (low-SNR) event"
    assert max(mls) > 1.5

    seen_ids = set()
    for ev in events:
        assert ev["event_id_hint"].startswith("ev")
        assert ev["event_id_hint"] not in seen_ids
        seen_ids.add(ev["event_id_hint"])

        origin = ev["origin"]
        assert isinstance(origin["time"], UTCDateTime)
        assert -90 <= origin["lat"] <= 90
        assert -180 <= origin["lon"] <= 180
        assert origin["depth_km"] > 0

        arrivals = ev["arrivals"]
        assert 4 <= len(arrivals) <= 8, "expected a plausible number of stations with arrivals"
        for sta, times in arrivals.items():
            assert sta.startswith(make_synthetic.NETWORK + ".")
            assert isinstance(times["P"], UTCDateTime)
            assert isinstance(times["S"], UTCDateTime)
            assert times["S"] > times["P"] > origin["time"]

    # the deliberate day-boundary event (ARCHITECTURE.md §8 pitfall)
    boundary_time = make_synthetic.DAY0 + 86400.0 - make_synthetic.BOUNDARY_EVENT_OFFSET_S
    assert any(abs(ev["origin"]["time"] - boundary_time) < 1e-6 for ev in events)


def test_fixture_smoke(synthetic_project):
    config_path, truth = synthetic_project
    assert config_path.exists()
    assert truth["events"]
    assert truth["known_issues"]
