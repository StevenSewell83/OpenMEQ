"""M4.1 spike test: verify the installed PhaseNet ``predict.py`` (external/
PhaseNet at the commit pinned by ``install.sh``) actually runs end-to-end as
a subprocess against two tiny hand-made 100 Hz miniSEED fixtures, and that
its ``picks.csv`` output recovers the injected P and S arrivals.

Per this spike's spec, the fixtures are built inline here (not via the
``synthetic_project`` fixture, which doesn't exist yet at this point in the
build order -- M4.2 hasn't landed). See ``docs/phasenet_interface.md`` for
the full write-up of everything this test (and manual exploration alongside
it) established about PhaseNet's CLI/data contract; this test only checks
the load-bearing claims: exit code 0, expected ``picks.csv`` columns, and
recovery of both phases within 0.1 s of the injected sample.

Fixture design notes (see docs/phasenet_interface.md for the full story):
a plain sinusoidal (Ricker-style) wavelet did NOT reliably get labelled
correctly by the pretrained model; a band-limited-noise burst with a
causal envelope does, but only for specific RNG seeds (~10% of seeds
tried in a grid search landed both phases within 0.1 s at the model's
default min_p_prob/min_s_prob=0.3). The parameters below are one verified
seed (base 660); don't change the amplitude/frequency/seed constants
without re-running predict.py to check the fixtures still work.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from scipy.signal import butter, sosfiltfilt

from openmeq.external import ExternalToolError, phasenet_dir, phasenet_model_dir, run

pytestmark = [pytest.mark.requires_phasenet, pytest.mark.slow]

SAMPLING_RATE = 100.0
DURATION_S = 60.0
NPTS = int(DURATION_S * SAMPLING_RATE)
START = UTCDateTime("2025-06-01T00:00:00.000000Z")

# Empirically-tuned P/S wavelet shapes and RNG seed (see module docstring
# and docs/phasenet_interface.md "Synthetic fixture design" section).
P_SHAPE = dict(n=300, f_lo=8.0, f_hi=25.0, rise_s=0.02, decay_s=0.5)
P_VERT_AMP = 6e-5
P_HORIZ_AMP = 1.2e-5
S_SHAPE = dict(n=1400, f_lo=1.0, f_hi=4.0, rise_s=0.15, decay_s=4.0)
S_VERT_AMP = 1e-5
S_HORIZ_AMP = 1.3e-4
NOISE_AMP = 3e-7
BURST_SEED_BASE = 660

# Both phases must land within this many seconds of the injected sample.
PICK_TOLERANCE_S = 0.1


def _phasenet_available() -> bool:
    try:
        model_dir = phasenet_model_dir()
        predict_py = phasenet_dir() / "phasenet" / "predict.py"
    except Exception:
        return False
    return predict_py.is_file() and (model_dir / "checkpoint").is_file()


skip_reason = (
    "external/PhaseNet clone/model not found (set OPENMEQ_EXTERNAL_DIR, or run install.sh)"
)


def _band_burst(n, dt, f_lo, f_hi, amp, rise_s, decay_s, seed):
    """A causal, band-limited-noise burst: filtered white noise shaped by a
    fast-rise/exponential-decay envelope, deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    pad = 100
    raw = rng.normal(0, 1, n + pad)
    sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=1 / dt, output="sos")
    filt = sosfiltfilt(sos, raw)[pad : pad + n]
    t = np.arange(n) * dt
    envelope = (1 - np.exp(-t / rise_s)) * np.exp(-t / decay_s)
    sig = filt * envelope
    return amp * sig / (np.max(np.abs(sig)) + 1e-12)


def _noise(npts, amp, seed):
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, npts)
    kernel = np.array([0.25, 0.5, 0.25])
    coloured = np.convolve(white, kernel, mode="same")
    return (amp * coloured / (np.std(coloured) + 1e-12)).astype(np.float64)


def _inject(data, onset_idx, wavelet, scale=1.0):
    w = wavelet * scale
    end = min(onset_idx + len(w), len(data))
    data[onset_idx:end] += w[: end - onset_idx]


def _make_station_traces(net, sta, band_code_inst, comp_codes, p_time, s_time):
    """Build a 3-component, 60 s @ 100 Hz station with an injected P and S
    arrival. `comp_codes` is (vertical, horizontal1, horizontal2), e.g.
    ("Z", "N", "E") for an oriented station or ("Z", "1", "2") for an
    unoriented borehole pair.
    """
    dt = 1.0 / SAMPLING_RATE
    vert, h1, h2 = comp_codes

    p_idx = int(round((p_time - START) * SAMPLING_RATE))
    s_idx = int(round((s_time - START) * SAMPLING_RATE))

    wv_pv = _band_burst(dt=dt, amp=P_VERT_AMP, seed=BURST_SEED_BASE + 1, **P_SHAPE)
    wv_ph = _band_burst(dt=dt, amp=P_HORIZ_AMP, seed=BURST_SEED_BASE + 2, **P_SHAPE)
    wv_sv = _band_burst(dt=dt, amp=S_VERT_AMP, seed=BURST_SEED_BASE + 3, **S_SHAPE)
    wv_sh = _band_burst(dt=dt, amp=S_HORIZ_AMP, seed=BURST_SEED_BASE + 4, **S_SHAPE)

    data_v = _noise(NPTS, NOISE_AMP, BURST_SEED_BASE + 5)
    data_h1 = _noise(NPTS, NOISE_AMP, BURST_SEED_BASE + 6)
    data_h2 = _noise(NPTS, NOISE_AMP, BURST_SEED_BASE + 7)

    _inject(data_v, p_idx, wv_pv)
    _inject(data_h1, p_idx, wv_ph)
    _inject(data_h2, p_idx, wv_ph, 0.85)
    _inject(data_v, s_idx, wv_sv)
    _inject(data_h1, s_idx, wv_sh)
    _inject(data_h2, s_idx, wv_sh, 0.9)

    header = {
        "network": net,
        "station": sta,
        "location": "",
        "starttime": START,
        "sampling_rate": SAMPLING_RATE,
    }
    return [
        Trace(data=data_v.astype(np.float32), header={**header, "channel": band_code_inst + vert}),
        Trace(data=data_h1.astype(np.float32), header={**header, "channel": band_code_inst + h1}),
        Trace(data=data_h2.astype(np.float32), header={**header, "channel": band_code_inst + h2}),
    ]


def _read_picks(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _find_pick(picks, file_name, station_id, phase_type, truth_time, tol_s=PICK_TOLERANCE_S):
    """Return the first pick of `phase_type` for `file_name`/`station_id`
    whose phase_time is within `tol_s` seconds of `truth_time`, or None."""
    for p in picks:
        if p["file_name"] != file_name or p["phase_type"] != phase_type:
            continue
        if p["station_id"] != station_id:
            continue
        pick_time = UTCDateTime(p["phase_time"])
        if abs(pick_time - truth_time) <= tol_s:
            return p
    return None


@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_predict_py_recovers_p_and_s_picks(tmp_path, monkeypatch):
    # Force CPU: openmeq.external.run() has no `env` parameter, so a caller
    # forces CPU-only execution the same way this test does -- by setting
    # the env var in the parent process before calling run(), since
    # subprocess.run (with env=None) inherits the parent's os.environ.
    # This is a real gap worth fixing in M4.3: run() should probably grow an
    # `env` passthrough instead of requiring callers to mutate process-wide
    # state. See docs/phasenet_interface.md.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    # Fixture 1: oriented 3-comp station, HH{Z,N,E}.
    p1, s1 = START + 20.0, START + 32.0
    traces1 = _make_station_traces("XX", "STA1", "HH", ("Z", "N", "E"), p1, s1)
    Stream(traces1).write(str(data_dir / "single_zne.mseed"), format="MSEED")

    # Fixture 2: unoriented borehole-style 3-comp station, DP{Z,1,2}.
    p2, s2 = START + 25.0, START + 39.0
    traces2 = _make_station_traces("XX", "STA3", "DP", ("Z", "1", "2"), p2, s2)
    Stream(traces2).write(str(data_dir / "single_z12.mseed"), format="MSEED")

    data_list = tmp_path / "data_list.csv"
    data_list.write_text("fname\nsingle_zne.mseed\nsingle_z12.mseed\n")

    predict_py = phasenet_dir() / "phasenet" / "predict.py"
    cmd = [
        sys.executable,
        str(predict_py),
        f"--model_dir={phasenet_model_dir()}",
        f"--data_dir={data_dir}",
        f"--data_list={data_list}",
        "--format=mseed",
        f"--result_dir={result_dir}",
        "--batch_size=1",  # see docs/phasenet_interface.md: batch_size>1
        # requires every chunk in the same predict.py call to have the same
        # number of stations (not just the same time length); real per-event
        # chunks generally don't, so batch_size=1 is the safe default.
        "--sampling_rate=100",
        "--min_p_prob=0.3",
        "--min_s_prob=0.3",
    ]

    try:
        proc = run(cmd, cwd=phasenet_dir() / "phasenet", timeout=120)
    except ExternalToolError as exc:
        pytest.fail(
            f"predict.py failed (exit {exc.returncode}):\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        )

    assert proc.returncode == 0

    picks_csv = result_dir / "picks.csv"
    assert picks_csv.is_file(), f"no picks.csv written to {result_dir}"

    picks = _read_picks(picks_csv)
    assert len(picks) > 0, "predict.py produced no picks at all"

    expected_columns = {
        "station_id",
        "begin_time",
        "phase_index",
        "phase_time",
        "phase_score",
        "phase_type",
        "file_name",
    }
    assert expected_columns.issubset(set(picks[0].keys()))

    # station_id is NET.STA.LOC.BandInst -- no final component character.
    for p in picks:
        assert not p["station_id"].endswith((".Z", ".N", ".E", ".1", ".2"))

    # Fixture 1 (oriented ZNE): both P and S recovered within tolerance.
    p_pick_1 = _find_pick(picks, "single_zne.mseed", "XX.STA1..HH", "P", p1)
    s_pick_1 = _find_pick(picks, "single_zne.mseed", "XX.STA1..HH", "S", s1)
    assert p_pick_1 is not None, (
        f"no P pick within {PICK_TOLERANCE_S}s of {p1} for single_zne.mseed"
    )
    assert s_pick_1 is not None, (
        f"no S pick within {PICK_TOLERANCE_S}s of {s1} for single_zne.mseed"
    )

    # Fixture 2 (unoriented Z12): both P and S recovered within tolerance.
    p_pick_2 = _find_pick(picks, "single_z12.mseed", "XX.STA3..DP", "P", p2)
    s_pick_2 = _find_pick(picks, "single_z12.mseed", "XX.STA3..DP", "S", s2)
    assert p_pick_2 is not None, (
        f"no P pick within {PICK_TOLERANCE_S}s of {p2} for single_z12.mseed"
    )
    assert s_pick_2 is not None, (
        f"no S pick within {PICK_TOLERANCE_S}s of {s2} for single_z12.mseed"
    )


@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_predict_py_batch_size_gt_1_requires_equal_station_count(tmp_path, monkeypatch):
    """Documents a real gotcha for M4.3: predict.py's mseed DataReader pads
    each file's data to shape (nt, nx, 3) where nx is THAT file's station
    count. dataset.batch(batch_size, drop_remainder=False) then requires
    every item in a batch to have an identical shape, so batch_size > 1
    raises a TensorFlow InvalidArgumentError as soon as two chunks in the
    same batch have a different number of stations -- not just a different
    time length, as PhaseNet's own README implies. Real per-event chunks
    generally have varying station counts (network availability differs
    per event), so this is a real production constraint, not just a
    contrived unit-test scenario.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    # One file with 1 station.
    traces1 = _make_station_traces("XX", "STA1", "HH", ("Z", "N", "E"), START + 20.0, START + 32.0)
    Stream(traces1).write(str(data_dir / "one_station.mseed"), format="MSEED")

    # One file with 2 stations (different nx).
    traces_a = _make_station_traces("XX", "STA1", "HH", ("Z", "N", "E"), START + 15.0, START + 27.0)
    traces_b = _make_station_traces("XX", "STA2", "HH", ("Z", "N", "E"), START + 18.0, START + 30.0)
    Stream(traces_a + traces_b).write(str(data_dir / "two_stations.mseed"), format="MSEED")

    data_list = tmp_path / "data_list.csv"
    data_list.write_text("fname\none_station.mseed\ntwo_stations.mseed\n")

    predict_py = phasenet_dir() / "phasenet" / "predict.py"
    cmd = [
        sys.executable,
        str(predict_py),
        f"--model_dir={phasenet_model_dir()}",
        f"--data_dir={data_dir}",
        f"--data_list={data_list}",
        "--format=mseed",
        f"--result_dir={result_dir}",
        "--batch_size=2",  # deliberately > 1, with mismatched nx
        "--sampling_rate=100",
    ]

    with pytest.raises(ExternalToolError) as excinfo:
        run(cmd, cwd=phasenet_dir() / "phasenet", timeout=120)
    assert "batch" in excinfo.value.stderr.lower() or "shape" in excinfo.value.stderr.lower()
