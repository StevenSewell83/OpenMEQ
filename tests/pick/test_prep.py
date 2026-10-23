"""Tests for ``openmeq.pick.prep``/``openmeq.pick.data_list`` (M4.2, ``OpenMEQ-1m7.2``).

Per this task's orchestrator note: M3.1 (the raw ``chunks/<event_id>/
<event_id>.mseed`` chunk writer) is being built concurrently and isn't
available yet, so raw input chunks are built by hand here -- either cut from
the ``synthetic_project`` fixture (``tests/conftest.py``) with
``openmeq.index.DataIndex.read`` around the true origin times (mirroring
``scripts/phasenet_recall_check.py``'s own approach, which independently
verified 100% P/S recall on this exact synthetic dataset with this exact
preprocessing recipe -- see ``docs/phasenet_interface.md`` §10), or built
directly from synthetic ``obspy.Trace`` objects for scenarios the fixture
dataset doesn't otherwise exercise (an isolated known-frequency sinusoid for
the decimation check; a two-location-code station for the collation check).

``synthetic_project`` is session-scoped and shared across the whole test
session (~500 MB on disk); per its own usage note, any test that writes
into a run's ``cfg.project.output_dir`` must use its own private output
directory rather than the shared fixture's. Every test here builds a
private :class:`~openmeq.config.Config` (``cfg.model_copy(deep=True)`` with
``project.output_dir`` repointed into that test's own ``tmp_path``) instead
of copying the whole ~500 MB project directory -- cheaper, and the disk-
space discipline this build's worker instructions call for. ``build_index``
itself only *reads* the shared waveform files and (idempotently) writes
``inventory/index.json`` under the shared session directory, exactly like
``tests/test_qc_availability.py``'s own ``indexed_project`` fixture -- safe
to call directly against ``synthetic_project``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime, read

from openmeq.config import load_config
from openmeq.external import ExternalToolError, phasenet_dir, phasenet_model_dir, run
from openmeq.index import DataIndex, build_index
from openmeq.paths import RunPaths
from openmeq.pick.prep import PREP_REPORT_COLUMNS, prepare_all, prepare_chunk
from openmeq.stations import station_table as compute_station_table

PRE_ORIGIN_S = 5.0
WINDOW_LEN_S = 60.0  # matches scripts/phasenet_recall_check.py's own window


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prep_world(synthetic_project, tmp_path_factory):
    """``(cfg, index, truth, station_table)`` for the shared synthetic project.

    ``build_index`` only reads waveform files and idempotently (re)writes
    ``inventory/index.json`` under the *shared* session directory (see the
    module docstring) -- safe to call directly against ``synthetic_project``,
    matching ``tests/test_qc_availability.py``'s established pattern.
    """
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    index = build_index(cfg)
    station_table = compute_station_table(cfg.stations)
    return cfg, index, truth, station_table


def _private_cfg(cfg, tmp_path: Path, name: str = "run"):
    """A deep copy of ``cfg`` with ``project.output_dir`` repointed into
    this test's own ``tmp_path`` -- never write into the shared session
    fixture's output directory (see module docstring)."""
    cfg2 = cfg.model_copy(deep=True)
    cfg2.project.output_dir = tmp_path / name
    return cfg2


def _write_raw_chunk_from_index(
    index: DataIndex, paths: RunPaths, event_id: str, origin_time: UTCDateTime
) -> tuple[Path, UTCDateTime, UTCDateTime]:
    """Cut a 60s window (matching ``scripts/phasenet_recall_check.py``) for
    every ``XX.*.00.*`` channel and write it as the raw
    ``chunks/<event_id>/<event_id>.mseed`` chunk.

    ``DataIndex.read`` merges with ``method=0`` (masked arrays across any
    gap); ``.split()`` turns that back into plain, contiguous, *unmasked*
    per-segment traces before writing -- multi-segment traces with a real
    gap between them (exactly what M3.1's own raw chunk writer would
    produce for the one deliberate gap in this dataset) rather than a
    masked array MiniSEED can't represent.
    """
    start = origin_time - PRE_ORIGIN_S
    end = start + WINDOW_LEN_S
    stream = index.read("XX.*.00.*", start, end)
    stream = stream.split()

    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(out_path), format="MSEED")
    return out_path, start, end


# ---------------------------------------------------------------------------
# 1. full-dataset PhaseNet-ready acceptance checks
# ---------------------------------------------------------------------------


def test_prepare_all_100hz_three_components_and_data_list(prep_world, tmp_path):
    cfg, index, truth, station_table = prep_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_all")
    paths2 = RunPaths(cfg2)

    event_ids = [ev["event_id_hint"] for ev in truth["events"]]
    for ev in truth["events"]:
        _write_raw_chunk_from_index(index, paths2, ev["event_id_hint"], ev["origin"]["time"])

    data_list_path = prepare_all(cfg2, paths2, event_ids, station_table=station_table)

    assert data_list_path == paths2.data_list_csv
    assert data_list_path.is_file()

    borehole_stations = {"BH01", "BH02"}
    surface_stations = {"STA01", "STA02", "STA03", "STA04", "STA05", "STA06"}

    for event_id in event_ids:
        out_path = paths2.chunk_pn_file(event_id)
        assert out_path.is_file(), f"missing chunks_pn output for {event_id}"
        stream = read(str(out_path))
        assert len(stream) > 0

        for tr in stream:
            assert tr.stats.sampling_rate == pytest.approx(100.0), (
                f"{event_id}/{tr.id}: expected 100 Hz, got {tr.stats.sampling_rate}"
            )
            assert tr.stats.channel[-1] in {"Z", "1", "2", "N", "E"}

        by_station: dict[str, list] = {}
        for tr in stream:
            by_station.setdefault(tr.stats.station, []).append(tr)

        for sta_code, traces in by_station.items():
            assert len(traces) == 3, (
                f"{event_id}/{sta_code}: expected exactly 3 components, got {len(traces)}"
            )
            last_chars = {tr.stats.channel[-1] for tr in traces}
            if sta_code in borehole_stations:
                assert last_chars == {"Z", "1", "2"}, (
                    f"{event_id}/{sta_code}: borehole station should map to unoriented Z/1/2, "
                    f"got {last_chars}"
                )
            elif sta_code in surface_stations:
                assert last_chars == {"Z", "N", "E"}, (
                    f"{event_id}/{sta_code}: surface station should pass through as Z/N/E, "
                    f"got {last_chars}"
                )

    # data_list.csv: one row per event, correct relative fname.
    df = pd.read_csv(data_list_path)
    assert list(df.columns) == ["fname"]
    assert len(df) == len(event_ids)
    expected = {f"{event_id}/{event_id}.mseed" for event_id in event_ids}
    assert set(df["fname"]) == expected

    # prep_report.csv is always (re)written, even with zero rows.
    assert paths2.prep_report_csv.is_file()
    report_df = pd.read_csv(paths2.prep_report_csv)
    assert list(report_df.columns) == PREP_REPORT_COLUMNS


def test_prepare_all_skip_unchanged_is_idempotent(prep_world, tmp_path):
    """A second ``prepare_all`` call with nothing changed should not need to
    reprocess any event (StepManifest skip-unchanged, ``cfg.phasenet`` +
    ``cfg.stations`` + chunk file mtime/size)."""
    cfg, index, truth, station_table = prep_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_skip")
    paths2 = RunPaths(cfg2)

    events = truth["events"][:3]
    event_ids = [ev["event_id_hint"] for ev in events]
    for ev in events:
        _write_raw_chunk_from_index(index, paths2, ev["event_id_hint"], ev["origin"]["time"])

    prepare_all(cfg2, paths2, event_ids, station_table=station_table)
    mtimes_before = {
        event_id: paths2.chunk_pn_file(event_id).stat().st_mtime_ns for event_id in event_ids
    }

    # Re-run with nothing changed: outputs must not be rewritten.
    prepare_all(cfg2, paths2, event_ids, station_table=station_table)
    mtimes_after = {
        event_id: paths2.chunk_pn_file(event_id).stat().st_mtime_ns for event_id in event_ids
    }
    assert mtimes_before == mtimes_after


# ---------------------------------------------------------------------------
# 2. decimation correctness
# ---------------------------------------------------------------------------


def _make_sinusoid_chunk(paths: RunPaths, event_id: str, freq_hz: float) -> Path:
    """A single 3-component (already-oriented ``HHZ/HHN/HHE``) 200 Hz station,
    60s duration, with a pure sinusoid of ``freq_hz`` injected on the
    vertical channel (plus small noise on all three)."""
    sr = 200.0
    npts = int(round(WINDOW_LEN_S * sr))
    start = UTCDateTime(2025, 6, 1)
    t = np.arange(npts) / sr
    rng = np.random.default_rng(1234)

    z_data = 1000.0 * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, 1.0, npts)
    n_data = rng.normal(0, 1.0, npts)
    e_data = rng.normal(0, 1.0, npts)

    header = {
        "network": "XX",
        "station": "STAX",
        "location": "00",
        "starttime": start,
        "sampling_rate": sr,
    }
    traces = [
        Trace(data=z_data.astype(np.float64), header={**header, "channel": "HHZ"}),
        Trace(data=n_data.astype(np.float64), header={**header, "channel": "HHN"}),
        Trace(data=e_data.astype(np.float64), header={**header, "channel": "HHE"}),
    ]
    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces).write(str(out_path), format="MSEED")
    return out_path


def test_decimation_preserves_injected_frequency(prep_world, tmp_path):
    cfg, _index, _truth, _station_table = prep_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_decimate")
    paths2 = RunPaths(cfg2)

    event_id = "ev20250601T000000_000"
    freq_hz = 15.0  # an exact FFT bin for 60s @ 100 Hz (bin spacing 100/6000 Hz)
    chunk_path = _make_sinusoid_chunk(paths2, event_id, freq_hz)

    empty_station_table = pd.DataFrame(
        columns=[
            "network",
            "station",
            "location",
            "latitude",
            "longitude",
            "elevation_m",
            "depth_m",
        ]
    )
    out_path = prepare_chunk(event_id, chunk_path, empty_station_table, cfg2)

    stream = read(str(out_path))
    (z_trace,) = [tr for tr in stream if tr.stats.channel.endswith("Z")]
    assert z_trace.stats.sampling_rate == pytest.approx(100.0)

    data = z_trace.data.astype(np.float64)
    n = data.size
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, d=1.0 / z_trace.stats.sampling_rate)
    # Ignore DC/very-low-frequency bins (detrend/taper/highpass leave some
    # residual low-frequency energy) when finding the dominant peak.
    valid = freqs > 2.0
    peak_freq = freqs[valid][np.argmax(spectrum[valid])]

    freq_resolution = z_trace.stats.sampling_rate / n
    assert abs(peak_freq - freq_hz) <= freq_resolution + 1e-9, (
        f"expected dominant frequency near {freq_hz} Hz, got {peak_freq} Hz "
        f"(resolution {freq_resolution} Hz)"
    )


@pytest.mark.parametrize("freq_hz", [5.0, 12.0, 25.0])
def test_decimation_does_not_delay_onsets(freq_hz):
    """OpenMEQ-tr9: ``Trace.decimate``'s own (causal) anti-alias filter
    delayed a 200 -> 100 Hz onset by 20-30 ms, and every PhaseNet pick with
    it. The decimated wavelet must line up with the original to within a
    fraction of a 100 Hz sample."""
    from openmeq.pick.prep import _decimate_to_target

    sr = 200.0
    t = np.arange(int(60 * sr)) / sr
    tau = np.clip(t - 30.0, 0.0, None)
    wavelet = np.where(t >= 30.0, np.sin(2 * np.pi * freq_hz * tau) * np.exp(-3.0 * tau), 0.0)
    tr = Trace(data=wavelet.copy(), header={"sampling_rate": sr, "channel": "HHZ"})
    stream = Stream([tr])
    _decimate_to_target(stream, "ev_test", 100.0)

    out = stream[0]
    assert out.stats.sampling_rate == pytest.approx(100.0)
    back = np.interp(t, out.times(), out.data)
    lags = np.arange(-40, 41)  # +-0.2 s at 200 Hz
    cc = [np.dot(np.roll(wavelet, k), back) for k in lags]
    lag_s = lags[int(np.argmax(cc))] / sr
    assert abs(lag_s) <= 0.0025, f"{freq_hz} Hz onset shifted by {lag_s * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# 3. two-location-code collation
# ---------------------------------------------------------------------------


def _make_two_location_code_chunk(paths: RunPaths, event_id: str) -> Path:
    """One station, two location codes: ``"00"`` has a full 3-component
    set, ``"01"`` has only the vertical -- ``"00"`` should be kept (more
    complete) and ``"01"`` dropped entirely."""
    sr = 200.0
    npts = int(round(WINDOW_LEN_S * sr))
    start = UTCDateTime(2025, 6, 1)
    rng = np.random.default_rng(99)

    def _trace(loc, cha):
        data = rng.normal(0, 1.0, npts).astype(np.float64)
        return Trace(
            data=data,
            header={
                "network": "XX",
                "station": "STAY",
                "location": loc,
                "channel": cha,
                "starttime": start,
                "sampling_rate": sr,
            },
        )

    traces = [
        _trace("00", "HHZ"),
        _trace("00", "HHN"),
        _trace("00", "HHE"),
        _trace("01", "HHZ"),
    ]
    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces).write(str(out_path), format="MSEED")
    return out_path


def test_two_location_codes_collapse_to_single_group(prep_world, tmp_path):
    cfg, _index, _truth, _station_table = prep_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_locs")
    paths2 = RunPaths(cfg2)

    event_id = "ev20250601T000100_000"
    chunk_path = _make_two_location_code_chunk(paths2, event_id)

    empty_station_table = pd.DataFrame(
        columns=[
            "network",
            "station",
            "location",
            "latitude",
            "longitude",
            "elevation_m",
            "depth_m",
        ]
    )
    out_path = prepare_chunk(event_id, chunk_path, empty_station_table, cfg2)

    stream = read(str(out_path))
    station_traces = [tr for tr in stream if tr.stats.station == "STAY"]
    assert len(station_traces) == 3, (
        "expected a single 3-component group, not two half-complete ones"
    )

    locations = {tr.stats.location for tr in station_traces}
    assert locations == {"00"}, (
        f"expected the more-complete location code '00' to be kept, got {locations}"
    )
    assert {tr.stats.channel[-1] for tr in station_traces} == {"Z", "N", "E"}


# ---------------------------------------------------------------------------
# 4. requires_phasenet: prepared chunks are actually PhaseNet-ready
# ---------------------------------------------------------------------------


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


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_prepared_chunks_are_phasenet_ready(prep_world, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    cfg, index, truth, station_table = prep_world
    cfg2 = _private_cfg(cfg, tmp_path, "run_phasenet")
    paths2 = RunPaths(cfg2)

    events = truth["events"][:3]
    event_ids = [ev["event_id_hint"] for ev in events]
    for ev in events:
        _write_raw_chunk_from_index(index, paths2, ev["event_id_hint"], ev["origin"]["time"])

    data_list_path = prepare_all(cfg2, paths2, event_ids, station_table=station_table)

    result_dir = tmp_path / "pick_raw"
    predict_py = phasenet_dir() / "phasenet" / "predict.py"
    cmd = [
        sys.executable,
        str(predict_py),
        f"--model_dir={phasenet_model_dir()}",
        f"--data_dir={paths2.chunks_pn_dir}",
        f"--data_list={data_list_path}",
        "--format=mseed",
        f"--result_dir={result_dir}",
        "--batch_size=1",
        "--sampling_rate=100",
        "--min_p_prob=0.3",
        "--min_s_prob=0.3",
    ]
    try:
        run(cmd, cwd=phasenet_dir() / "phasenet", timeout=120)
    except ExternalToolError as exc:
        pytest.fail(
            f"predict.py failed (exit {exc.returncode}):\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        )

    picks_csv = result_dir / "picks.csv"
    assert picks_csv.is_file()
    picks = pd.read_csv(picks_csv)
    assert len(picks) > 0, "predict.py produced no picks at all from prepared chunks"
    assert set(picks["phase_type"]) & {"P", "S"}

    # Light correctness check (not a full recall study -- see
    # docs/phasenet_interface.md §10 for that): at least one P and one S
    # pick should land within 0.1s of a true arrival somewhere in this
    # small subset.
    matched = {"P": False, "S": False}
    for ev in events:
        event_id = ev["event_id_hint"]
        event_picks = picks[picks["file_name"] == f"{event_id}/{event_id}.mseed"]
        for station_key, arrival in ev["arrivals"].items():
            station_id_prefix = f"{station_key}."  # e.g. "XX.STA01."
            candidates = event_picks[event_picks["station_id"].str.startswith(station_id_prefix)]
            for phase in ("P", "S"):
                truth_t = arrival[phase]
                phase_rows = candidates[candidates["phase_type"] == phase]
                for _, row in phase_rows.iterrows():
                    if abs(UTCDateTime(row["phase_time"]) - truth_t) <= 0.1:
                        matched[phase] = True

    assert matched["P"], "no P pick recovered within 0.1s of any truth arrival in this subset"
    assert matched["S"], "no S pick recovered within 0.1s of any truth arrival in this subset"
