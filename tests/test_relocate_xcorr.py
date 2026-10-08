"""Tests for openmeq.relocate.xcorr (M10.3 / OpenMEQ-16d.3).

Per this issue's own scope, the true M10.3 inputs come from ``run_ph2dt``
(M10.2) and ``build_cache`` (M10.1); most tests here build minimal,
hand-made ``.npz`` caches and pick/origin dicts directly against
:func:`openmeq.relocate.xcorr.cross_correlate_pair` (a pure, picklable,
per-pair worker function -- no pipeline needed to exercise it), mirroring
``tests/test_relocate_waveform_cache.py``'s own "build the minimal inputs
this module actually reads" pattern. ``test_resume_skips_completed_pairs``
and the hypoDD smoke check exercise the full :func:`openmeq.relocate.xcorr
.run_xcorr` orchestration, including reading a real catalogue/dt.ct off
disk.

Disk-space note (worker instructions): every test here writes only into
its own function-scoped ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Catalog, Trace, UTCDateTime
from obspy.core.event import Event, Origin, Pick, WaveformStreamID
from obspy.signal.cross_correlation import xcorr_pick_correction

from openmeq.catalog import event_id_from_time, write_catalog
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
from openmeq.external import ExternalToolMissing, bin_path, run
from openmeq.paths import RunPaths
from openmeq.relocate import xcorr as xcorr_module
from openmeq.relocate.hypodd_files import (
    build_event_id_map,
    build_station_label_map,
    write_hypodd_inp_1d,
)
from openmeq.relocate.ph2dt_step import run_ph2dt
from openmeq.relocate.waveform_cache import build_cache
from openmeq.relocate.xcorr import (
    DtccLine,
    PairXcorrParams,
    cross_correlate_pair,
    obspy_window,
    parse_dt_ct_detail,
    run_xcorr,
    write_dt_cc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
HYPODD_MISSING = not (
    (EXTERNAL_DIR / "bin" / "ph2dt").exists() and (EXTERNAL_DIR / "bin" / "hypoDD").exists()
)

SR = 200.0
DT = 1.0 / SR


# ---------------------------------------------------------------------------
# Synthetic waveform helper: a clean, analytic (noiseless) wavelet, so a
# deliberately-introduced sub-sample shift can be recovered to well within
# one sample without interpolation artefacts confounding the result.
# ---------------------------------------------------------------------------


def _template(t_rel: np.ndarray, freq: float = 8.0, half_width_s: float = 0.15) -> np.ndarray:
    """A symmetric, Gaussian-enveloped sinusoid centred at ``t_rel == 0``."""
    sigma = half_width_s / 3.0
    return np.sin(2 * np.pi * freq * t_rel) * np.exp(-(t_rel**2) / (2 * sigma**2))


def _make_window(
    start: UTCDateTime,
    n: int,
    peak_time: UTCDateTime,
    *,
    freq: float = 8.0,
    amplitude: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A buffer of ``n`` samples at :data:`SR` starting at ``start``, holding
    one :func:`_template` pulse peaking at ``peak_time``, optionally with
    added Gaussian noise (for the "different-quality channel" tests)."""
    rel_t = np.arange(n) * DT
    peak_offset = peak_time - start
    data = amplitude * _template(rel_t - peak_offset, freq=freq)
    if noise_std > 0:
        rng = rng or np.random.default_rng(0)
        data = data + rng.normal(0.0, noise_std, size=n)
    return data.astype(np.float64)


def _write_npz_cache(
    cache_dir: Path, event_id: str, channels: dict[str, tuple[np.ndarray, UTCDateTime]]
) -> None:
    """Write a hand-built ``relocate/cache/<event_id>.npz``
    (``docs/ARCHITECTURE.md`` §6f's layout) directly -- ``channels`` is
    ``{seed_id: (data, start_time)}``."""
    arrays: dict[str, np.ndarray] = {}
    seed_ids = sorted(channels.keys())
    for seed_id in seed_ids:
        data, start = channels[seed_id]
        arrays[f"{seed_id}__data"] = data.astype(np.float32)
        arrays[f"{seed_id}__start"] = np.array(start.timestamp, dtype=np.float64)
    arrays["seed_ids"] = np.array(seed_ids, dtype=str)
    arrays["sampling_rate"] = np.array(SR, dtype=np.float64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{event_id}.npz", **arrays)


def _pick_data(origin_time: UTCDateTime, picks: dict[str, dict[str, UTCDateTime]]) -> dict:
    """``{"origin_time": iso, "picks": {net_sta: {"P"|"S": iso}}}`` -- the
    ``picks_by_event`` entry shape :func:`cross_correlate_pair` reads."""
    return {
        "origin_time": str(origin_time),
        "picks": {
            net_sta: {phase: str(t) for phase, t in phases.items()}
            for net_sta, phases in picks.items()
        },
    }


_DEFAULT_COMPONENTS = {"P": ["Z"], "S": ["1", "2", "N", "E"]}


def _params(
    *,
    min_cc: float = -1.0,
    maxlag_s: float = 0.05,
    station_phase_list: list[tuple[str, str]],
    cuspid_to_event_id: dict[int, str],
    components: dict[str, list[str]] | None = None,
    window_p: tuple[float, float] = (-0.05, 0.2),
    window_s: tuple[float, float] = (-0.05, 0.3),
) -> PairXcorrParams:
    return PairXcorrParams(
        window_p=window_p,
        window_s=window_s,
        min_cc=min_cc,
        maxlag_s=maxlag_s,
        components=components or _DEFAULT_COMPONENTS,
        label_to_net_sta={"STA1": "XX.STA1"},
        station_phase_list=station_phase_list,
        cuspid_to_event_id=cuspid_to_event_id,
    )


# ---------------------------------------------------------------------------
# test_sign_convention_recovers_known_shift
# ---------------------------------------------------------------------------


def test_sign_convention_recovers_known_shift(tmp_path):
    """A synthetic pair with a known, deliberately-introduced 0.037 s lag:
    the written dt.cc value must equal ``TT_i - TT_j_corrected`` (obspy's
    own documented ``xcorr_pick_correction`` convention), independently
    re-derived here via a direct, ground-truth ``xcorr_pick_correction``
    call -- a flipped-sign implementation fails this test.
    """
    known_shift_s = 0.037
    n = int(4.0 * SR)

    start_i = UTCDateTime(2020, 1, 1, 0, 0, 0)
    start_j = UTCDateTime(2020, 1, 1, 0, 5, 0)  # different reference, deliberately
    pick_i = start_i + 2.0
    pick_j = start_j + 2.0  # same nominal offset into its own buffer as pick_i

    # event j's *true* waveform feature sits known_shift_s later than pick_j
    # naively assumes -- xcorr_pick_correction must recover that shift.
    data_i = _make_window(start_i, n, peak_time=pick_i)
    data_j = _make_window(start_j, n, peak_time=pick_j + known_shift_s)

    # Origins chosen so the *naive* (uncorrected) travel times are equal --
    # isolates the cross-correlation-recovered shift as the entire signal
    # (TT_i - TT_j_naive == 0), rather than burying it under an unrelated
    # travel-time difference.
    origin_i = pick_i - 5.0
    origin_j = pick_j - 5.0

    # Ground truth: obspy's own xcorr_pick_correction, called directly on
    # obspy Trace objects built from the same arrays -- independent of this
    # module's own (faster) _lag_and_cc implementation.
    tr_i = Trace(data=data_i)
    tr_i.stats.sampling_rate = SR
    tr_i.stats.starttime = start_i
    tr_j = Trace(data=data_j)
    tr_j.stats.sampling_rate = SR
    tr_j.stats.starttime = start_j

    pick2_corr_ref, coeff_ref = xcorr_pick_correction(
        pick_i, tr_i, pick_j, tr_j, t_before=0.05, t_after=0.2, cc_maxlag=0.08, filter=None
    )
    assert coeff_ref > 0.9  # sanity: clean, noiseless synthetic pulses correlate well
    expected_dt_s = (pick_i - origin_i) - (pick_j + pick2_corr_ref - origin_j)

    # This construction makes the shift the whole signal (dt.ct-naive value
    # would be exactly 0), so the recovered value must be clearly nonzero
    # and clearly distinguishable from its own negation.
    assert abs(expected_dt_s) > 5 * DT

    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_id: (data_j, start_j)})

    picks_by_event = {
        "evt_i": _pick_data(origin_i, {"XX.STA1": {"P": pick_i}}),
        "evt_j": _pick_data(origin_j, {"XX.STA1": {"P": pick_j}}),
    }
    params = _params(
        maxlag_s=0.08,
        station_phase_list=[("STA1", "P")],
        cuspid_to_event_id={1: "evt_i", 2: "evt_j"},
    )

    lines = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)

    assert len(lines) == 1
    line = lines[0]
    assert line.sta == "STA1"
    assert line.phase == "P"
    assert line.dt_s == pytest.approx(expected_dt_s, abs=DT)

    # A flipped-sign implementation would write -expected_dt_s -- clearly
    # distinguishable given the shift dominates (isolated by construction).
    assert abs(line.dt_s - (-expected_dt_s)) > 5 * DT


# ---------------------------------------------------------------------------
# test_matches_dt_ct_convention_on_shared_pair
# ---------------------------------------------------------------------------


def test_matches_dt_ct_convention_on_shared_pair(tmp_path):
    """Zero-noise synthetic arrivals with the true waveform feature sitting
    exactly at each event's own pick time (no deliberate cross-correlation
    lag): the catalog-derived dt.ct-style differential time
    (``TT_i - TT_j`` from picks/origins alone) and this module's dt.cc value
    must agree closely, in the same sign -- a sign flip would instead land
    roughly ``2x`` away (the negation), a dramatic disagreement.
    """
    n = int(4.0 * SR)
    start_i = UTCDateTime(2021, 6, 1, 0, 0, 0)
    start_j = UTCDateTime(2021, 6, 1, 0, 10, 0)
    pick_i = start_i + 2.0
    pick_j = start_j + 2.0

    # No deliberate shift: each event's waveform genuinely peaks at its own
    # pick time (a noise-free, perfectly-picked synthetic arrival).
    data_i = _make_window(start_i, n, peak_time=pick_i)
    data_j = _make_window(start_j, n, peak_time=pick_j)

    # Distinct, non-trivial travel times so dt_ct itself is meaningfully
    # nonzero (not a degenerate 0.0 that a sign flip couldn't be told apart
    # from).
    origin_i = pick_i - 5.000
    origin_j = pick_j - 5.240
    dt_ct_value = (pick_i - origin_i) - (pick_j - origin_j)
    assert abs(dt_ct_value) > 0.1  # comfortably larger than the 1-sample tolerance below

    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_id: (data_j, start_j)})

    picks_by_event = {
        "evt_i": _pick_data(origin_i, {"XX.STA1": {"P": pick_i}}),
        "evt_j": _pick_data(origin_j, {"XX.STA1": {"P": pick_j}}),
    }
    params = _params(
        station_phase_list=[("STA1", "P")], cuspid_to_event_id={1: "evt_i", 2: "evt_j"}
    )

    lines = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)
    assert len(lines) == 1
    dt_cc_value = lines[0].dt_s

    # Same sign, close magnitude (zero-noise arrivals -> near-zero
    # cross-correlation correction).
    assert dt_cc_value == pytest.approx(dt_ct_value, abs=3 * DT)
    # A flipped-sign bug would land near -dt_ct_value, roughly 2x away.
    assert abs(dt_cc_value - (-dt_ct_value)) > 0.15


# ---------------------------------------------------------------------------
# test_best_channel_selected_for_s
# ---------------------------------------------------------------------------


def test_best_channel_selected_for_s(tmp_path):
    """Two horizontal channels with deliberately different cross-correlation
    quality: the higher-cc channel's (dt, cc) must be written outright, not
    blended with the lower-quality channel's independent measurement."""
    n = int(4.0 * SR)
    start_i = UTCDateTime(2022, 3, 1, 0, 0, 0)
    start_j = UTCDateTime(2022, 3, 1, 0, 20, 0)
    pick_i = start_i + 2.0
    pick_j = start_j + 2.0
    origin_i = pick_i - 5.0
    origin_j = pick_j - 5.0

    # "N": clean, noiseless, a small known shift -> high cc.
    shift_n = 0.02
    data_i_n = _make_window(start_i, n, peak_time=pick_i)
    data_j_n = _make_window(start_j, n, peak_time=pick_j + shift_n)

    # "E": same nominal shift is irrelevant -- heavy independent noise on
    # both sides gives a measurably lower (but not necessarily
    # min_cc-filtered) correlation coefficient, and its own recovered lag
    # is essentially uncorrelated with N's.
    rng_i = np.random.default_rng(1)
    rng_j = np.random.default_rng(2)
    data_i_e = _make_window(start_i, n, peak_time=pick_i, amplitude=0.3, noise_std=0.6, rng=rng_i)
    data_j_e = _make_window(
        start_j, n, peak_time=pick_j + 0.09, amplitude=0.3, noise_std=0.6, rng=rng_j
    )

    seed_n = "XX.STA1.00.HH1"
    seed_e = "XX.STA1.00.HH2"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_n: (data_i_n, start_i), seed_e: (data_i_e, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_n: (data_j_n, start_j), seed_e: (data_j_e, start_j)})

    picks_by_event = {
        "evt_i": _pick_data(origin_i, {"XX.STA1": {"S": pick_i}}),
        "evt_j": _pick_data(origin_j, {"XX.STA1": {"S": pick_j}}),
    }

    def _run(components_s: list[str]):
        params = _params(
            station_phase_list=[("STA1", "S")],
            cuspid_to_event_id={1: "evt_i", 2: "evt_j"},
            components={"P": ["Z"], "S": components_s},
        )
        return cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)

    lines_n_only = _run(["1"])
    lines_e_only = _run(["2"])
    lines_both = _run(["1", "2"])

    assert len(lines_n_only) == 1 and len(lines_e_only) == 1 and len(lines_both) == 1
    cc_n, cc_e = lines_n_only[0].cc, lines_e_only[0].cc
    assert cc_n > cc_e, "the clean channel must correlate better than the noisy one"

    # The combined-channel result must match the single best (N-only)
    # result exactly -- not some blend of N and E.
    assert lines_both[0].cc == pytest.approx(cc_n)
    assert lines_both[0].dt_s == pytest.approx(lines_n_only[0].dt_s)
    # ... and must clearly differ from E's own (lower-quality) measurement,
    # proving E was not averaged in.
    assert lines_both[0].dt_s != pytest.approx(lines_e_only[0].dt_s, abs=5 * DT)


# ---------------------------------------------------------------------------
# test_missing_phase_at_one_event_skipped
# ---------------------------------------------------------------------------


def test_missing_phase_at_one_event_skipped(tmp_path):
    n = int(4.0 * SR)
    start_i = UTCDateTime(2023, 1, 1, 0, 0, 0)
    start_j = UTCDateTime(2023, 1, 1, 0, 0, 30)
    pick_i = start_i + 2.0

    data_i = _make_window(start_i, n, peak_time=pick_i)
    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    # event j has no cache/pick for this station+phase at all.
    _write_npz_cache(cache_dir, "evt_j", {})

    picks_by_event = {
        "evt_i": _pick_data(pick_i - 5.0, {"XX.STA1": {"S": pick_i}}),
        "evt_j": _pick_data(start_j + 1.0, {}),  # no S pick at XX.STA1
    }
    params = _params(
        station_phase_list=[("STA1", "S")], cuspid_to_event_id={1: "evt_i", 2: "evt_j"}
    )

    lines = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)
    assert lines == []  # skipped, not a crash


# ---------------------------------------------------------------------------
# test_min_cc_filtering
# ---------------------------------------------------------------------------


def test_min_cc_filtering(tmp_path):
    n = int(4.0 * SR)
    start_i = UTCDateTime(2024, 1, 1, 0, 0, 0)
    start_j = UTCDateTime(2024, 1, 1, 0, 1, 0)
    pick_i = start_i + 2.0
    pick_j = start_j + 2.0

    rng_i = np.random.default_rng(10)
    rng_j = np.random.default_rng(11)
    # Deliberately low-quality: mostly independent noise on both sides.
    data_i = _make_window(start_i, n, peak_time=pick_i, amplitude=0.1, noise_std=1.0, rng=rng_i)
    data_j = _make_window(
        start_j, n, peak_time=pick_j + 0.5, amplitude=0.1, noise_std=1.0, rng=rng_j
    )

    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_id: (data_j, start_j)})

    picks_by_event = {
        "evt_i": _pick_data(pick_i - 5.0, {"XX.STA1": {"P": pick_i}}),
        "evt_j": _pick_data(pick_j - 5.0, {"XX.STA1": {"P": pick_j}}),
    }

    lenient = _params(
        min_cc=-1.0, station_phase_list=[("STA1", "P")], cuspid_to_event_id={1: "evt_i", 2: "evt_j"}
    )
    strict = _params(
        min_cc=0.99, station_phase_list=[("STA1", "P")], cuspid_to_event_id={1: "evt_i", 2: "evt_j"}
    )

    lines_lenient = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, lenient)
    lines_strict = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, strict)

    assert len(lines_lenient) == 1
    assert lines_lenient[0].cc < 0.99  # confirms this pair genuinely is low-cc
    assert lines_strict == []  # filtered out by min_cc


# ---------------------------------------------------------------------------
# test_dt_cc_format_matches_hypodd_expectations
# ---------------------------------------------------------------------------


def _parse_dt_cc(path: Path) -> dict[tuple[int, int], list[tuple[str, float, float, str]]]:
    """Minimal, independent parser for the confirmed dt.cc grammar
    (getdata.f:369-402, ``# ID1 ID2 OTC`` header / ``STA DT WEIGHT PHA``
    data lines) -- mirrors ``tests/test_relocate_hypodd_files.py``'s own
    ``_parse_dt_ct`` pattern.
    """
    pairs: dict[tuple[int, int], list] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                fields = line.split()
                assert len(fields) == 4, f"expected '# ID1 ID2 OTC', got {line!r}"
                current = (int(fields[1]), int(fields[2]))
                float(fields[3])  # OTC must parse as a float
                pairs[current] = []
            elif line.strip():
                parts = line.split()
                assert len(parts) == 4, f"expected 'STA DT WEIGHT PHA', got {line!r}"
                sta, dt_s, wt, pha = parts
                assert len(sta) <= 7
                assert pha in ("P", "S")
                pairs[current].append((sta, float(dt_s), float(wt), pha))
    return pairs


def test_dt_cc_format_matches_hypodd_expectations(tmp_path):
    lines_by_pair = [
        [DtccLine(sta="STA1", dt_s=0.0123, cc=0.85, phase="P")],
        [],  # a pair with zero surviving lines must be omitted entirely
        [
            DtccLine(sta="STA1", dt_s=-0.045, cc=0.91, phase="S"),
            DtccLine(sta="STA2", dt_s=0.002, cc=0.72, phase="P"),
        ],
    ]
    pairs = [(1, 2), (3, 4), (5, 6)]

    out_path = tmp_path / "dt.cc"
    n_written = write_dt_cc(out_path, pairs, lines_by_pair, otc=0.0)
    assert n_written == 2

    parsed = _parse_dt_cc(out_path)
    assert set(parsed) == {(1, 2), (5, 6)}  # the empty-lines pair is omitted
    assert parsed[(1, 2)] == [("STA1", pytest.approx(0.0123), pytest.approx(0.85), "P")]
    assert len(parsed[(5, 6)]) == 2


# ---------------------------------------------------------------------------
# parse_dt_ct_detail
# ---------------------------------------------------------------------------


def test_parse_dt_ct_detail(tmp_path):
    dt_ct = tmp_path / "dt.ct"
    dt_ct.write_text(
        "#     38542     38520\n"
        "NCCBW     3.430   3.430 1.0000 P\n"
        "NCCSP     2.850   2.840 0.5000 S\n"
        "#     38542     45165\n"
        "NCCBW     3.430   3.500 1.0000 P\n"
    )
    detail = parse_dt_ct_detail(dt_ct)
    assert detail[(38542, 38520)] == [("NCCBW", "P"), ("NCCSP", "S")]
    assert detail[(38542, 45165)] == [("NCCBW", "P")]


# ---------------------------------------------------------------------------
# test_resume_skips_completed_pairs (full run_xcorr orchestration)
# ---------------------------------------------------------------------------


def _minimal_cfg(tmp_path: Path, stations_csv: Path) -> Config:
    return Config(
        project=ProjectConfig(name="test", output_dir=tmp_path / "run"),
        parallel=ParallelConfig(n_workers=1),
        data=DataConfig(waveform_dirs=[tmp_path], start=date(2025, 1, 1), end=date(2025, 1, 2)),
        stations=StationsConfig(csv=stations_csv),
        nlloc=NllocConfig(
            velocity_model=NllocVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
            grid=NllocGridConfig(origin=(0.0, 0.0), size_km=(10.0, 10.0, 10.0), spacing_km=0.5),
        ),
        hypodd=HypoddConfig(
            velocity_model=HypoddVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
            xcorr=XcorrConfig(min_cc=-1.0),
        ),
    )


def _write_stations_csv(path: Path) -> Path:
    row = {
        "net": "XX",
        "sta": "STA1",
        "loc": "00",
        "lat": 0.0,
        "lon": 0.0,
        "elev_m": 0.0,
        "depth_m": 0.0,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def _make_event(event_id: str, origin_time: UTCDateTime, pick_time: UTCDateTime) -> Event:
    from openmeq.catalog import make_event

    ev = make_event(event_id)
    origin = Origin(time=origin_time, latitude=0.0, longitude=0.0, depth=1000.0)
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    pick = Pick(
        time=pick_time,
        phase_hint="P",
        waveform_id=WaveformStreamID(network_code="XX", station_code="STA1", channel_code="HHZ"),
    )
    ev.picks.append(pick)
    return ev


def test_resume_skips_completed_pairs(tmp_path, monkeypatch):
    stations_csv = _write_stations_csv(tmp_path / "stations.csv")
    cfg = _minimal_cfg(tmp_path, stations_csv)
    paths = RunPaths(cfg)
    paths.ensure()

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    n = int(4.0 * SR)

    events_meta = []  # (event_id, origin_time, pick_time, cache_start)
    for i in range(3):
        origin_t = t0 + i * 60.0
        pick_t = origin_t + 5.0
        cache_start = pick_t - 2.0
        eid = event_id_from_time(origin_t)
        events_meta.append((eid, origin_t, pick_t, cache_start))

    cat = Catalog(events=[_make_event(eid, ot, pt) for eid, ot, pt, _ in events_meta])
    write_catalog(cat, paths.catalog_mag_xml)

    event_id_map = build_event_id_map(cat, paths.relocate_work_dir / "event_id_map.csv")
    label_map = build_station_label_map(
        pd.DataFrame(
            [
                {
                    "network": "XX",
                    "station": "STA1",
                    "location": "00",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "elevation_m": 0.0,
                    "depth_m": 0.0,
                }
            ]
        ),
        paths.relocate_work_dir / "station_label_map.csv",
    )
    assert label_map["XX.STA1"] == "STA1"

    cuspids = {eid: event_id_map[eid] for eid, *_ in events_meta}
    seed_id = "XX.STA1.00.HHZ"
    for eid, _origin_t, pick_t, cache_start in events_meta:
        data = _make_window(cache_start, n, peak_time=pick_t)
        _write_npz_cache(paths.relocate_cache_dir, eid, {seed_id: (data, cache_start)})

    eid0, eid1, eid2 = (m[0] for m in events_meta)
    pair_01 = (cuspids[eid0], cuspids[eid1])
    pair_02 = (cuspids[eid0], cuspids[eid2])
    pairs = [pair_01, pair_02]

    dt_ct_lines = [
        f"# {pair_01[0]:9d} {pair_01[1]:9d}\n",
        "STA1      5.000   5.000 1.0000 P\n",
        f"# {pair_02[0]:9d} {pair_02[1]:9d}\n",
        "STA1      5.000   5.000 1.0000 P\n",
    ]
    paths.dt_ct.write_text("".join(dt_ct_lines))

    # Pre-populate the incremental results file with pair_01 already done.
    pre_done_line = {"sta": "STA1", "dt_s": 0.0, "cc": 0.99, "phase": "P"}
    pre_done = {"key": f"{pair_01[0]}_{pair_01[1]}", "result": {"lines": [pre_done_line]}}
    paths.xcorr_results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Stamp it as written with these same settings (as a previous run would).
    xcorr_module._reset_stale_results(paths.xcorr_results_jsonl, cfg.hypodd.xcorr)
    paths.xcorr_results_jsonl.write_text(json.dumps(pre_done) + "\n")

    call_count = 0
    real_fn = xcorr_module.cross_correlate_pair

    def _counting_fn(pair, cache_dir, picks_by_event, cfg_params):
        nonlocal call_count
        call_count += 1
        return real_fn(pair, cache_dir, picks_by_event, cfg_params)

    monkeypatch.setattr(xcorr_module, "cross_correlate_pair", _counting_fn)

    run_xcorr(cfg, paths, pairs)

    assert call_count == 1  # only pair_02 was actually (re)computed

    parsed = _parse_dt_cc(paths.dt_cc)
    assert set(parsed) == {pair_01, pair_02}  # both pairs present in the final dt.cc


# ---------------------------------------------------------------------------
# hypoDD smoke check on the synthetic catalogue
# ---------------------------------------------------------------------------


_FAKE_HYPODD_INC = """\
c hypoDD.inc: FAKE, tiny, for tests only.
\tinteger*4 MAXEVE, MAXLAY, MAXDATA, MAXSTA, MAXEVE0, MAXDATA0
\tinteger*4 MAXCL
\tparameter(MAXEVE=  1000,
     &          MAXDATA=  100000,
     &          MAXEVE0=  1000,
     &          MAXDATA0= 100000,
     &          MAXLAY=   30,
     &          MAXSTA=   500,
     &          MAXCL=    10)
"""

_FAKE_PH2DT_INC = """\
c ph2dt.inc: FAKE, tiny, for tests only.
\tinteger\tMEV, MSTA, MOBS
\tparameter(MEV=    1000,
     &\t\tMSTA=   500,
     &\t\tMOBS=   200)
"""

_FAKE_VEL3D_INC = """\
      integer\tmaxnx,maxny,maxnz,maxnz2
      parameter(maxnx= 25)
      parameter(maxny= 25)
      parameter(maxnz= 20)
      parameter(maxnz2=40)
      integer\tixkms,iykms,izkms
      parameter(ixkms= 10000)
      parameter(iykms= 10000)
      parameter(izkms= 10000)
"""


def _write_fake_inc_tree(root: Path) -> Path:
    include_dir = root / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "hypoDD.inc").write_text(_FAKE_HYPODD_INC)
    (include_dir / "ph2dt.inc").write_text(_FAKE_PH2DT_INC)
    (include_dir / "vel3d.inc").write_text(_FAKE_VEL3D_INC)
    return root


def _catalog_from_truth(truth: dict) -> Catalog:
    from obspy.core.event import Arrival, Magnitude

    from openmeq.catalog import make_event

    cat = Catalog()
    for ev_truth in truth["events"]:
        origin_truth = ev_truth["origin"]
        eid = ev_truth["event_id_hint"]
        event = make_event(eid)
        origin = Origin(
            time=origin_truth["time"],
            latitude=origin_truth["lat"],
            longitude=origin_truth["lon"],
            depth=origin_truth["depth_km"] * 1000.0,
        )
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        mag_obj = Magnitude(mag=ev_truth["ml"], magnitude_type="ML")
        event.magnitudes = [mag_obj]
        event.preferred_magnitude_id = mag_obj.resource_id

        for net_sta, phases in ev_truth["arrivals"].items():
            net, sta = net_sta.split(".", 1)
            for phase, t in phases.items():
                pick = Pick(
                    time=t,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(network_code=net, station_code=sta),
                )
                event.picks.append(pick)
                arrival = Arrival(pick_id=pick.resource_id, phase=phase, time_weight=1.0)
                origin.arrivals.append(arrival)
        cat.append(event)
    return cat


@pytest.mark.requires_hypodd
@pytest.mark.slow
@pytest.mark.skipif(HYPODD_MISSING, reason="external/HYPODD (ph2dt/hypoDD) not available")
def test_synthetic_project_xcorr_and_hypodd_smoke(synthetic_project, tmp_path, monkeypatch, capsys):
    """End-to-end smoke check (M10.4/M10.5 are out of scope -- this is not a
    numeric-accuracy test): waveform cache -> ph2dt -> run_xcorr -> a
    minimal hypoDD.inp -> the real hypoDD binary runs to completion on the
    resulting dt.cc for the synthetic catalogue.
    """
    monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", str(EXTERNAL_DIR))
    from openmeq.chunk import write_chunks
    from openmeq.config import load_config
    from openmeq.index import build_index

    config_path, truth = synthetic_project
    project_copy = tmp_path / "project"
    shutil.copytree(config_path.parent, project_copy)
    cfg = load_config(project_copy / "config.yaml")
    cfg.parallel.n_workers = 1
    cfg.hypodd.ph2dt.maxdist_km = 200
    cfg.hypodd.ph2dt.maxsep_km = 50
    cfg.hypodd.ph2dt.minobs = 1
    cfg.hypodd.ph2dt.maxobs = 100
    paths = RunPaths(cfg)
    paths.ensure()

    cat = _catalog_from_truth(truth)
    write_catalog(cat, paths.catalog_mag_xml)

    # Build raw chunks (for the waveform cache) from the synthetic archive's
    # own index, exactly as tests/test_relocate_waveform_cache.py does.
    index = build_index(cfg)
    rows = []
    for ev_truth in truth["events"]:
        eid = ev_truth["event_id_hint"]
        stations = sorted(ev_truth["arrivals"].keys())
        rows.append(
            {
                "event_id": eid,
                "time": str(UTCDateTime(ev_truth["origin"]["time"])),
                "duration_s": 5.0,
                "coincidence_sum": len(stations),
                "n_stations": len(stations),
                "stations": ";".join(stations),
                "trace_ids": "",
            }
        )
    detections_df = pd.DataFrame(rows)
    write_chunks(detections_df, index, cfg, paths)

    event_ids = [ev["event_id_hint"] for ev in truth["events"]]
    build_cache(cfg, paths, event_ids)

    fake_inc_root = _write_fake_inc_tree(tmp_path / "fake_hypodd")

    try:
        bin_path("ph2dt")
        bin_path("hypoDD")
    except ExternalToolMissing:
        pytest.skip("ph2dt/hypoDD binaries not available")

    pairs = run_ph2dt(
        cfg, paths, external_hypodd_dir=fake_inc_root, external_bin_dir=tmp_path / "fake_bin"
    )
    assert pairs, "expected at least one linked event pair from the synthetic catalogue"

    import time as _time

    t0 = _time.monotonic()
    run_xcorr(cfg, paths, pairs)
    elapsed = max(_time.monotonic() - t0, 1e-9)
    pairs_per_s = len(pairs) / elapsed
    with capsys.disabled():
        print(
            f"\nM10.3 xcorr throughput: {len(pairs)} pairs in {elapsed:.3f}s "
            f"-> {pairs_per_s:.1f} pairs/s"
        )

    assert paths.dt_cc.is_file()
    assert paths.dt_cc.stat().st_size > 0, "expected >=1 surviving cross-correlation observation"

    model = pd.DataFrame(
        {"depth_km": [0.0, 1.0, 3.0], "vp": [3.50, 4.50, 5.80], "vs": [2.00, 2.60, 3.35]}
    )
    write_hypodd_inp_1d(
        paths.relocate_work_dir / "hypoDD.inp", cfg, cfg.hypodd.weighting_rows, model
    )

    hypodd = bin_path("hypoDD")
    run([str(hypodd), "hypoDD.inp"], cwd=paths.relocate_work_dir, timeout=600)

    reloc_path = paths.relocate_work_dir / "hypoDD.reloc"
    assert reloc_path.is_file(), "hypoDD smoke check: expected hypoDD.reloc to be produced"
    assert reloc_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# OpenMEQ-6y1: window_p/window_s are signed [start, end] offsets from the pick
# ---------------------------------------------------------------------------


def test_obspy_window_negates_signed_start():
    assert obspy_window((-0.05, 0.2)) == (0.05, 0.2)
    assert obspy_window((0.01, 0.2)) == (-0.01, 0.2)


def test_window_starts_before_pick(tmp_path):
    """A short pulse entirely *before* the pick must be inside the correlated
    window for window_p=(-0.1, 0.15). The pre-fix code started the window
    after the pick (pick+0.08 s here), saw only zeros and wrote no line."""
    n = int(4.0 * SR)
    start_i = UTCDateTime(2020, 1, 1)
    start_j = UTCDateTime(2020, 1, 1, 0, 5)
    pick_i, pick_j = start_i + 2.0, start_j + 2.0
    rel = np.arange(n) * DT
    data_i = _template(rel - (pick_i - 0.04 - start_i), freq=20.0, half_width_s=0.04)
    data_j = _template(rel - (pick_j - 0.04 - start_j), freq=20.0, half_width_s=0.04)
    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_id: (data_j, start_j)})
    picks_by_event = {
        "evt_i": _pick_data(pick_i - 1.0, {"XX.STA1": {"P": pick_i}}),
        "evt_j": _pick_data(pick_j - 1.0, {"XX.STA1": {"P": pick_j}}),
    }
    params = _params(
        min_cc=0.9,
        maxlag_s=0.04,
        window_p=(-0.1, 0.15),
        station_phase_list=[("STA1", "P")],
        cuspid_to_event_id={1: "evt_i", 2: "evt_j"},
    )
    lines = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)
    assert len(lines) == 1
    assert lines[0].cc > 0.99
    assert lines[0].dt_s == pytest.approx(0.0, abs=DT)


def test_stale_xcorr_results_discarded_on_settings_change(tmp_path):
    from openmeq.config import XcorrConfig
    from openmeq.relocate.xcorr import _reset_stale_results

    out = tmp_path / "xcorr_results.jsonl"
    out.write_text('{"key": "1_2", "result": {"lines": []}}\n')
    _reset_stale_results(out, XcorrConfig())  # no stamp yet -> pre-fix results, discarded
    assert not out.exists()
    out.write_text('{"key": "1_2", "result": {"lines": []}}\n')
    _reset_stale_results(out, XcorrConfig())  # same settings -> kept for resume
    assert out.exists()
    _reset_stale_results(out, XcorrConfig(min_cc=0.8))  # changed -> discarded
    assert not out.exists()


def test_xcorr_config_validation():
    from pydantic import ValidationError

    from openmeq.config import XcorrConfig

    with pytest.raises(ValidationError, match="start < end"):
        XcorrConfig(window_p=(0.2, -0.05))
    with pytest.raises(ValidationError, match="freqmin < freqmax"):
        XcorrConfig(filter={"freqmin": 20.0, "freqmax": 2.0})
    cfg = XcorrConfig(filter={"corners": 2, "zerophase": False}, taper_pct=0.1)
    assert (cfg.filter.corners, cfg.filter.zerophase, cfg.taper_pct) == (2, False, 0.1)


@pytest.mark.parametrize("reject", [True, False])
def test_edge_lags_rejected(tmp_path, reject):
    """OpenMEQ-9fh: a true shift (0.06 s) beyond maxlag_s (0.03 s) pins the
    best lag to the search edge -- dropped when reject_edge_lags is on."""
    n = int(4.0 * SR)
    start_i = UTCDateTime(2020, 1, 1)
    start_j = UTCDateTime(2020, 1, 1, 0, 5)
    pick_i, pick_j = start_i + 2.0, start_j + 2.0
    data_i = _make_window(start_i, n, peak_time=pick_i, freq=4.0)
    data_j = _make_window(start_j, n, peak_time=pick_j + 0.06, freq=4.0)
    seed_id = "XX.STA1.00.HHZ"
    cache_dir = tmp_path / "cache"
    _write_npz_cache(cache_dir, "evt_i", {seed_id: (data_i, start_i)})
    _write_npz_cache(cache_dir, "evt_j", {seed_id: (data_j, start_j)})
    picks_by_event = {
        "evt_i": _pick_data(pick_i - 1.0, {"XX.STA1": {"P": pick_i}}),
        "evt_j": _pick_data(pick_j - 1.0, {"XX.STA1": {"P": pick_j}}),
    }
    params = dataclasses.replace(
        _params(
            maxlag_s=0.03,
            station_phase_list=[("STA1", "P")],
            cuspid_to_event_id={1: "evt_i", 2: "evt_j"},
        ),
        reject_edge_lags=reject,
    )
    lines = cross_correlate_pair((1, 2), str(cache_dir), picks_by_event, params)
    if reject:
        assert lines == []
    else:
        assert len(lines) == 1
        # Pinned at the edge: |correction| == maxlag, not the true 0.06 s.
        assert abs(lines[0].dt_s - (1.0 - 1.0)) == pytest.approx(0.03, abs=DT)
