"""Tests for openmeq.magnitude.pipeline / the ``openmeq magnitude`` command
(M8.3 / OpenMEQ-625.3).

Every unit test here builds its own minimal, hand-made inputs (config,
inventory, catalogue, raw chunk), the same pattern
``tests/test_relocate_waveform_cache.py`` and ``tests/test_magnitude_amplitude.py``
use, since this is an orchestration layer over already-tested M8.1
(``openmeq.magnitude.amplitude``) / M8.2 (``openmeq.magnitude.relations``)
pieces. ``test_synthetic_project_end_to_end`` is the exception: per this
task's own notes, it builds the located catalogue by hand from
``synthetic_project``'s ``truth`` (location, M6, is being built concurrently)
and writes real chunks via ``openmeq.chunk.write_chunks``, then runs the
real ``openmeq magnitude`` CLI command end to end.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Origin, Pick, WaveformStreamID
from obspy.core.inventory import Channel, Inventory, Network, Response, Station
from obspy.core.inventory.response import InstrumentSensitivity, PolesZerosResponseStage

from openmeq.catalog import (
    event_id as get_event_id,
)
from openmeq.catalog import (
    iso_z,
    make_event,
    read_catalog,
    reject_event,
    write_catalog,
)
from openmeq.chunk import write_chunks
from openmeq.cli import main
from openmeq.config import (
    Config,
    DataConfig,
    DiagnosticsConfig,
    HypoddConfig,
    HypoddVelocityModelConfig,
    LocalMlCoefficientsConfig,
    LocalMlConfig,
    MagnitudeConfig,
    MlConfig,
    NllocConfig,
    NllocGridConfig,
    NllocVelocityModelConfig,
    ParallelConfig,
    ProjectConfig,
    StationsConfig,
    load_config,
)
from openmeq.index import build_index
from openmeq.magnitude import pipeline as magnitude_pipeline
from openmeq.magnitude.amplitude import StationAmplitude
from openmeq.magnitude.pipeline import run_magnitude, select_catalog_path
from openmeq.manifest import StepManifest
from openmeq.paths import RunPaths
from openmeq.stations import load_inventory

NETWORK = "XX"
LOCATION = "00"

pytestmark = []


# ---------------------------------------------------------------------------
# Minimal config / inventory / event / chunk builders
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path, *, magnitude: MagnitudeConfig | None = None) -> Config:
    """A minimal, valid :class:`Config`. ``nlloc``/``hypodd`` point at files
    that don't need to exist -- this module never touches them."""
    kwargs = {}
    if magnitude is not None:
        kwargs["magnitude"] = magnitude
    return Config(
        project=ProjectConfig(name="test", output_dir=tmp_path / "run"),
        parallel=ParallelConfig(n_workers=1),
        data=DataConfig(waveform_dirs=[tmp_path], start=date(2025, 1, 1), end=date(2025, 1, 2)),
        stations=StationsConfig(stationxml=tmp_path / "inventory" / "stations.xml"),
        nlloc=NllocConfig(
            velocity_model=NllocVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
            grid=NllocGridConfig(origin=(0.0, 0.0), size_km=(10.0, 10.0, 10.0), spacing_km=0.5),
        ),
        hypodd=HypoddConfig(
            velocity_model=HypoddVelocityModelConfig(type="1d", file=tmp_path / "vel.txt"),
        ),
        **kwargs,
    )


def _flat_velocity_response(sensitivity: float) -> Response:
    """A flat, velocity-sensitive (M/S -> COUNTS) response -- matches
    tests/test_magnitude_amplitude.py's own convention."""
    stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=sensitivity,
        stage_gain_frequency=1.0,
        input_units="M/S",
        output_units="COUNTS",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[],
        poles=[],
        normalization_factor=1.0,
    )
    sensitivity_obj = InstrumentSensitivity(
        value=sensitivity, frequency=1.0, input_units="M/S", output_units="COUNTS"
    )
    return Response(instrument_sensitivity=sensitivity_obj, response_stages=[stage])


def _make_station(
    sta: str, lat: float, lon: float, elevation_m: float, sensitivity: float
) -> Station:
    station = Station(code=sta, latitude=lat, longitude=lon, elevation=elevation_m, channels=[])
    for cha in ("HHZ", "HHN", "HHE"):
        station.channels.append(
            Channel(
                code=cha,
                location_code=LOCATION,
                latitude=lat,
                longitude=lon,
                elevation=elevation_m,
                depth=0.0,
                azimuth=0.0,
                dip=-90.0 if cha.endswith("Z") else 0.0,
                sample_rate=100.0,
                response=_flat_velocity_response(sensitivity),
            )
        )
    return station


def _write_inventory(cfg: Config, stations: list[Station]) -> None:
    inv = Inventory(networks=[Network(code=NETWORK, stations=stations)], source="test")
    cfg.stations.stationxml.parent.mkdir(parents=True, exist_ok=True)
    inv.write(str(cfg.stations.stationxml), format="STATIONXML")


def _pick(sta: str, cha: str, phase: str, time: UTCDateTime, net: str = NETWORK) -> Pick:
    return Pick(
        time=time,
        phase_hint=phase,
        evaluation_mode="automatic",
        waveform_id=WaveformStreamID(
            network_code=net, station_code=sta, location_code=LOCATION, channel_code=cha
        ),
    )


def _make_event_with_origin(
    event_id: str, origin_time: UTCDateTime, lat: float, lon: float, depth_m: float = 3000.0
):
    event = make_event(event_id)
    origin = Origin(time=origin_time, latitude=lat, longitude=lon, depth=depth_m)
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    return event, origin


def _sine_burst(n: int, sampling_rate: float, freq_hz: float, amplitude: float) -> np.ndarray:
    t = np.arange(n) / sampling_rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _write_chunk_with_pulse(
    paths: RunPaths, event_id: str, stations: list[str], t0: UTCDateTime
) -> None:
    """Write ``chunks/<event_id>/<event_id>.mseed``: every station gets a
    3-component sine burst covering the whole chunk (matches
    tests/test_magnitude_amplitude.py's own convention -- a full-trace
    burst avoids having to line up a short pulse with the pick window
    exactly)."""
    n = int(20.0 * 100.0)
    st = Stream()
    for sta in stations:
        for cha in ("HHZ", "HHN", "HHE"):
            tr = Trace(data=_sine_burst(n, 100.0, 5.0, 2000.0))
            tr.stats.network = NETWORK
            tr.stats.station = sta
            tr.stats.location = LOCATION
            tr.stats.channel = cha
            tr.stats.sampling_rate = 100.0
            tr.stats.starttime = t0
            st.append(tr)
    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write(str(out_path), format="MSEED")


def _scenario(
    tmp_path: Path,
    *,
    n_stations: int = 3,
    magnitude: MagnitudeConfig | None = None,
) -> tuple[Config, RunPaths, str]:
    """A full, hand-built, minimal-but-real single-event scenario: config,
    inventory, a located catalogue (``locate/catalog_auto.xml``) and a raw
    chunk with a measurable amplitude at every station. Returns
    ``(cfg, paths, event_id)``.
    """
    cfg = _make_cfg(tmp_path, magnitude=magnitude)
    paths = RunPaths(cfg)
    paths.ensure()

    station_codes = [f"STA{i + 1}" for i in range(n_stations)]
    stations = [
        _make_station(sta, 0.01 * (i + 1), 0.0, 0.0, sensitivity=2.0e8)
        for i, sta in enumerate(station_codes)
    ]
    _write_inventory(cfg, stations)

    t0 = UTCDateTime(2025, 1, 1, 0, 0, 0)
    origin_time = t0 + 5.0
    event_id = "ev20250101T000005_000"
    event, _origin = _make_event_with_origin(event_id, origin_time, 0.0, 0.0)
    for sta in station_codes:
        event.picks.append(_pick(sta, "HHZ", "P", origin_time + 1.0))
        event.picks.append(_pick(sta, "HHN", "S", origin_time + 2.0))

    write_catalog(Catalog(events=[event]), paths.catalog_auto_xml)
    _write_chunk_with_pulse(paths, event_id, station_codes, t0)

    return cfg, paths, event_id


# ---------------------------------------------------------------------------
# test_catalog_selection_precedence
# ---------------------------------------------------------------------------


def test_catalog_selection_precedence(tmp_path):
    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    _write_inventory(cfg, [_make_station("STA1", 0.0, 0.0, 0.0, 2.0e8)])

    write_catalog(Catalog(events=[], description="auto"), paths.catalog_auto_xml)

    # No override, no review/catalog_final.xml -> locate/catalog_auto.xml.
    assert select_catalog_path(paths) == paths.catalog_auto_xml

    run_magnitude(cfg, paths)
    written = read_catalog(paths.catalog_mag_xml)
    assert written.description == "auto"

    # review/catalog_final.xml appears -> a fresh run switches to it,
    # without --catalog.
    write_catalog(Catalog(events=[], description="final"), paths.catalog_final_xml)
    assert select_catalog_path(paths) == paths.catalog_final_xml

    run_magnitude(cfg, paths)
    written = read_catalog(paths.catalog_mag_xml)
    assert written.description == "final"

    # Explicit --catalog overrides both.
    override_path = tmp_path / "external.xml"
    write_catalog(Catalog(events=[], description="override"), override_path)
    assert select_catalog_path(paths, override_path) == override_path

    run_magnitude(cfg, paths, override_path)
    written = read_catalog(paths.catalog_mag_xml)
    assert written.description == "override"


def test_catalog_override_resolves_relative_to_config_dir(tmp_path):
    config_dir = tmp_path / "cfgdir"
    config_dir.mkdir()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()

    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    catalog_path = config_dir / "sub" / "cat.xml"
    catalog_path.parent.mkdir(parents=True)
    write_catalog(Catalog(events=[]), catalog_path)

    resolved = select_catalog_path(
        paths,
        Path("sub/cat.xml"),
        config_path=config_dir / "config.yaml",
    )
    assert resolved == catalog_path.resolve()


def test_missing_catalog_raises_clear_error(tmp_path):
    import click

    cfg = _make_cfg(tmp_path)
    paths = RunPaths(cfg)
    paths.ensure()

    with pytest.raises(click.ClickException):
        run_magnitude(cfg, paths)


# ---------------------------------------------------------------------------
# test_amplitude_unit_is_metres_in_quakeml
# ---------------------------------------------------------------------------


def test_amplitude_unit_is_metres_in_quakeml(tmp_path):
    cfg, paths, event_id = _scenario(
        tmp_path, n_stations=1, magnitude=MagnitudeConfig(min_stations=1)
    )

    run_magnitude(cfg, paths)

    df = pd.read_csv(paths.station_magnitudes_csv)
    assert len(df) == 1
    amplitude_m = float(df.iloc[0]["amplitude_m"])
    assert amplitude_m > 0

    raw_xml = paths.catalog_mag_xml.read_text()
    match = re.search(r"<genericAmplitude>\s*<value>([^<]+)</value>", raw_xml)
    assert match is not None, raw_xml
    xml_value = float(match.group(1))

    # The raw QuakeML value must be the metre-scale amplitude, not the
    # mm-converted value openmeq.magnitude.relations.station_ml uses
    # internally for its own log10(A) formula (a 1000x error would be easy
    # to get backwards here).
    assert xml_value == pytest.approx(amplitude_m, rel=1e-6)
    assert xml_value != pytest.approx(amplitude_m * 1000.0, rel=1e-3)

    out_cat = read_catalog(paths.catalog_mag_xml)
    ev = out_cat[0]
    assert len(ev.amplitudes) == 1
    amp = ev.amplitudes[0]
    assert amp.type == "AML"
    assert amp.unit == "m"
    assert amp.generic_amplitude == pytest.approx(amplitude_m, rel=1e-6)
    assert amp.waveform_id.network_code == NETWORK
    assert amp.waveform_id.station_code == "STA1"
    assert amp.pick_id is not None

    assert len(ev.station_magnitudes) == 1
    assert ev.station_magnitudes[0].station_magnitude_type == "ML"
    assert ev.station_magnitudes[0].amplitude_id == amp.resource_id

    assert len(ev.magnitudes) == 1
    assert ev.magnitudes[0].magnitude_type == "ML"
    assert ev.magnitudes[0].method_id.id == "smi:openmeq/ml/hutton_boore"
    assert ev.preferred_magnitude_id == ev.magnitudes[0].resource_id


# ---------------------------------------------------------------------------
# test_min_stations_enforced
# ---------------------------------------------------------------------------


def test_min_stations_enforced(tmp_path):
    magnitude_cfg = MagnitudeConfig(min_stations=5)
    cfg, paths, event_id = _scenario(tmp_path, n_stations=2, magnitude=magnitude_cfg)

    run_magnitude(cfg, paths)

    out_cat = read_catalog(paths.catalog_mag_xml)
    assert len(out_cat) == 1
    ev = out_cat[0]

    # Fewer than min_stations valid station MLs -> no event Magnitude...
    assert len(ev.magnitudes) == 0
    assert ev.preferred_magnitude_id is None
    # ...but the event is still present, and still not rejected.
    assert ev.preferred_origin().evaluation_status != "rejected"
    # ...and its measured stations still got Amplitude/StationMagnitude
    # objects (the measurement, as opposed to the event-level aggregate).
    assert len(ev.amplitudes) == 2
    assert len(ev.station_magnitudes) == 2


def test_min_stations_satisfied_gets_magnitude(tmp_path):
    magnitude_cfg = MagnitudeConfig(min_stations=2)
    cfg, paths, event_id = _scenario(tmp_path, n_stations=3, magnitude=magnitude_cfg)

    run_magnitude(cfg, paths)

    out_cat = read_catalog(paths.catalog_mag_xml)
    ev = out_cat[0]
    assert len(ev.magnitudes) == 1
    assert ev.magnitudes[0].station_count == 3


# ---------------------------------------------------------------------------
# test_event_ml_is_median_of_station_mls
# ---------------------------------------------------------------------------


def _write_minimal_chunk(paths: RunPaths, event_id: str) -> None:
    tr = Trace(data=np.zeros(10))
    tr.stats.network = NETWORK
    tr.stats.station = "DUMMY"
    tr.stats.location = LOCATION
    tr.stats.channel = "HHZ"
    tr.stats.sampling_rate = 100.0
    tr.stats.starttime = UTCDateTime(2025, 1, 1)
    out_path = paths.chunk_file(event_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Stream([tr]).write(str(out_path), format="MSEED")


def test_event_ml_is_median_of_station_mls(tmp_path, monkeypatch):
    """Aggregation logic in isolation: ``openmeq.magnitude.amplitude
    .measure_event_amplitudes`` is monkeypatched to return known
    per-station amplitudes, and ``cfg.magnitude.ml.relation = "local"``
    with all-zero coefficients makes ``station_ml`` exactly
    ``log10(amplitude_mm)`` (no distance/station-correction term) -- so
    each station's ML is engineered to an exact target value, letting this
    test assert the event ``Magnitude.mag`` (the QuakeML ``<mag><value>``
    element ARCHITECTURE.md §5 documents) against a known median without
    depending on M8.1's own Wood-Anderson physics (already covered by
    tests/test_magnitude_amplitude.py).
    """
    magnitude_cfg = MagnitudeConfig(
        min_stations=2,
        statistic="median",
        ml=MlConfig(
            relation="local",
            local=LocalMlConfig(coefficients=LocalMlCoefficientsConfig(a=0.0, b=0.0, c=0.0)),
        ),
    )
    cfg = _make_cfg(tmp_path, magnitude=magnitude_cfg)
    paths = RunPaths(cfg)
    paths.ensure()
    _write_inventory(cfg, [_make_station("DUMMY", 0.0, 0.0, 0.0, 2.0e8)])

    event_id = "ev20250101T000005_000"
    origin_time = UTCDateTime(2025, 1, 1, 0, 0, 5)
    event, _origin = _make_event_with_origin(event_id, origin_time, 0.0, 0.0)
    write_catalog(Catalog(events=[event]), paths.catalog_auto_xml)
    _write_minimal_chunk(paths, event_id)

    target_mls = [1.0, 1.2, 1.4, 2.5, 3.0]
    expected_median = float(np.median(target_mls))
    expected_mean = float(np.mean(target_mls))

    fake_amps = [
        StationAmplitude(
            net=NETWORK,
            sta=f"STA{i + 1}",
            amplitude_m=(10.0**ml) / 1000.0,  # log10(amplitude_mm) == ml exactly
            distance_km=10.0,
            component_used="HHN",
            window_start=origin_time,
            window_end=origin_time + 1.0,
        )
        for i, ml in enumerate(target_mls)
    ]

    monkeypatch.setattr(magnitude_pipeline, "measure_event_amplitudes", lambda *a, **k: fake_amps)

    run_magnitude(cfg, paths)

    out_cat = read_catalog(paths.catalog_mag_xml)
    ev = out_cat[0]
    assert len(ev.magnitudes) == 1
    assert ev.magnitudes[0].mag == pytest.approx(expected_median, abs=1e-6)
    assert ev.magnitudes[0].station_count == len(target_mls)
    assert expected_median != pytest.approx(expected_mean, abs=1e-3)  # sanity: not a tie


# ---------------------------------------------------------------------------
# test_diagnostics_written
# ---------------------------------------------------------------------------


def test_diagnostics_written(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=3)

    run_magnitude(cfg, paths)

    diag_dir = paths.diagnostics_dir("magnitude")
    expected = [
        "ml_vs_distance.png",
        "station_residuals.png",
        "magnitude_frequency.png",
        "ml_through_time.png",
    ]
    for name in expected:
        p = diag_dir / name
        assert p.is_file(), f"{p} missing"
        assert p.stat().st_size > 0, f"{p} is empty"


def test_extended_diagnostics_disabled_by_default(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=3)

    run_magnitude(cfg, paths)

    assert not (paths.diagnostics_dir("magnitude") / "extended").exists()


def test_extended_diagnostics_written_when_enabled(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=3)
    cfg.diagnostics = DiagnosticsConfig(extended=True)

    run_magnitude(cfg, paths)

    ext = paths.diagnostics_dir("magnitude") / "extended"
    for name in (
        "res_vs_distance.png",
        "gr_mc_bvalue.png",
        "gr_mc_bvalue.csv",
        f"amp_window_{event_id}.png",
    ):
        assert (ext / name).is_file(), f"{name} missing"
        assert (ext / name).stat().st_size > 0
    # One-event catalogue: the tiny-catalogue branch, not a fit.
    gr = pd.read_csv(ext / "gr_mc_bvalue.csv")
    assert gr.loc[0, "status"] == "insufficient events"
    # Baseline set still written, unchanged.
    assert (paths.diagnostics_dir("magnitude") / "magnitude_frequency.png").is_file()


def test_extended_diagnostics_failure_never_fails_step(tmp_path, monkeypatch, caplog):
    from openmeq.magnitude import diagnostics_extended

    cfg, paths, event_id = _scenario(tmp_path, n_stations=2)
    cfg.diagnostics = DiagnosticsConfig(magnitude={"extended": True})

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics_extended, "plot_residual_vs_distance", boom)
    with caplog.at_level("WARNING", logger="openmeq.magnitude"):
        run_magnitude(cfg, paths)

    ext = paths.diagnostics_dir("magnitude") / "extended"
    assert not (ext / "res_vs_distance.png").exists()
    assert (ext / "gr_mc_bvalue.png").is_file()
    assert any("res_vs_distance.png failed" in r.getMessage() for r in caplog.records)
    assert paths.catalog_mag_xml.is_file()


# ---------------------------------------------------------------------------
# test_skip_unchanged
# ---------------------------------------------------------------------------


def test_skip_unchanged(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=2)

    run_magnitude(cfg, paths)
    mtime_1 = paths.catalog_mag_xml.stat().st_mtime_ns

    run_magnitude(cfg, paths)
    mtime_2 = paths.catalog_mag_xml.stat().st_mtime_ns

    assert mtime_1 == mtime_2

    manifest = StepManifest(paths, magnitude_pipeline.STEP_NAME)
    from openmeq.manifest import fingerprint

    fp = fingerprint(cfg.magnitude, paths.catalog_auto_xml)
    assert manifest.is_done(magnitude_pipeline.MANIFEST_KEY, fp)

    # --force busts the cache even though nothing changed.
    run_magnitude(cfg, paths, force=True)
    mtime_3 = paths.catalog_mag_xml.stat().st_mtime_ns
    assert mtime_3 != mtime_2


def test_skip_unchanged_busted_by_catalog_change(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=2)

    run_magnitude(cfg, paths)
    mtime_1 = paths.catalog_mag_xml.stat().st_mtime_ns

    # A newly-appeared review/catalog_final.xml changes catalogue
    # selection -> must not be skipped even without --force.
    cat = read_catalog(paths.catalog_auto_xml)
    write_catalog(cat, paths.catalog_final_xml)

    run_magnitude(cfg, paths)
    mtime_2 = paths.catalog_mag_xml.stat().st_mtime_ns
    assert mtime_2 != mtime_1


# ---------------------------------------------------------------------------
# test_rejected_events_pass_through_unchanged
# ---------------------------------------------------------------------------


def test_rejected_events_pass_through_unchanged(tmp_path):
    cfg, paths, event_id = _scenario(tmp_path, n_stations=2)

    cat = read_catalog(paths.catalog_auto_xml)
    rejected_event, _origin = _make_event_with_origin(
        "ev20250101T999999_000", UTCDateTime(2025, 1, 1, 1, 0, 0), 0.0, 0.0
    )
    reject_event(rejected_event, "test", "test")
    cat.events.append(rejected_event)
    write_catalog(cat, paths.catalog_auto_xml)

    run_magnitude(cfg, paths)

    out_cat = read_catalog(paths.catalog_mag_xml)
    ids = {get_event_id(ev) for ev in out_cat}
    assert ids == {event_id, "ev20250101T999999_000"}

    rejected_out = next(ev for ev in out_cat if get_event_id(ev) == "ev20250101T999999_000")
    assert rejected_out.preferred_origin().evaluation_status == "rejected"
    assert len(rejected_out.amplitudes) == 0
    assert len(rejected_out.magnitudes) == 0

    # And the CSV has no rows for the rejected event.
    df = pd.read_csv(paths.station_magnitudes_csv)
    assert "ev20250101T999999_000" not in set(df["event_id"])


# ---------------------------------------------------------------------------
# test_synthetic_project_end_to_end
# ---------------------------------------------------------------------------


def _absolute_synthetic_config(config_path: Path, output_dir: Path) -> Path:
    """A copy of ``config_path`` with every path field written out
    **absolute** and ``project.output_dir`` redirected to ``output_dir``.

    Written into ``output_dir``'s own (function-scoped) temp directory,
    never into the shared session-scoped ``synthetic_project`` directory
    (``tests/conftest.py``'s own usage note) -- a plain text patch of just
    ``output_dir`` (as ``tests/test_cli.py``'s ``_patched_example_config``
    does against the illustrative, non-existent paths in
    ``config/example.yaml``) would leave every *other* relative path (e.g.
    ``./inventory/stations.xml``) resolving against *this* copy's own
    directory instead of the original ``config_path.parent`` where the
    real synthetic files live, per ``openmeq.config``'s "resolve relative
    to the config file's own directory" convention.
    """
    import yaml

    from openmeq.config import load_config as _load_config

    resolved = _load_config(config_path)  # every path field, correctly resolved

    raw = yaml.safe_load(config_path.read_text())
    raw["project"]["output_dir"] = str(output_dir)
    raw["data"]["waveform_dirs"] = [str(p) for p in resolved.data.waveform_dirs]
    if raw.get("stations", {}).get("stationxml") is not None:
        raw["stations"]["stationxml"] = str(resolved.stations.stationxml)
    if raw.get("stations", {}).get("csv") is not None:
        raw["stations"]["csv"] = str(resolved.stations.csv)
    raw["nlloc"]["velocity_model"]["file"] = str(resolved.nlloc.velocity_model.file)
    raw["hypodd"]["velocity_model"]["file"] = str(resolved.hypodd.velocity_model.file)

    new_path = output_dir.parent / "config.yaml"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return new_path


def _z_and_horizontal_channels(inv: Inventory, net: str, sta: str) -> tuple[str, str]:
    for network in inv.networks:
        if network.code != net:
            continue
        for station in network.stations:
            if station.code != sta:
                continue
            codes = [c.code for c in station.channels]
            z = next(c for c in codes if c.endswith("Z"))
            horiz = next(c for c in codes for last in ("N", "E", "1", "2") if c.endswith(last))
            return z, horiz
    raise AssertionError(f"station {net}.{sta} not found in inventory")


def test_synthetic_project_end_to_end(synthetic_project, tmp_path):
    config_path, truth = synthetic_project

    output_dir = tmp_path / "run"
    patched_config_path = _absolute_synthetic_config(config_path, output_dir)

    cfg = load_config(patched_config_path)
    inv = load_inventory(cfg.stations)
    index = build_index(cfg)
    paths = RunPaths(cfg)
    paths.ensure()

    n_events = 8
    events_subset = truth["events"][:n_events]

    cat_events = []
    detection_rows = []
    for ev in events_subset:
        event_id = ev["event_id_hint"]
        origin_time = UTCDateTime(ev["origin"]["time"])
        depth_m = ev["origin"]["depth_km"] * 1000.0
        event, _origin = _make_event_with_origin(
            event_id, origin_time, ev["origin"]["lat"], ev["origin"]["lon"], depth_m
        )

        for station_key, arrivals in ev["arrivals"].items():
            net, sta = station_key.split(".", 1)
            p_time = UTCDateTime(arrivals["P"])
            s_time = UTCDateTime(arrivals["S"])
            z_code, horiz_code = _z_and_horizontal_channels(inv, net, sta)
            event.picks.append(_pick(sta, z_code, "P", p_time, net=net))
            event.picks.append(_pick(sta, horiz_code, "S", s_time, net=net))

        cat_events.append(event)
        detection_rows.append({"event_id": event_id, "time": iso_z(origin_time)})

    write_catalog(Catalog(events=cat_events), paths.catalog_auto_xml)

    detections_df = pd.DataFrame(detection_rows)
    write_chunks(detections_df, index, cfg, paths)

    runner = CliRunner()
    result = runner.invoke(main, ["magnitude", "-c", str(patched_config_path)])
    assert result.exit_code == 0, result.output

    assert paths.catalog_mag_xml.is_file()
    assert paths.station_magnitudes_csv.is_file()

    out_cat = read_catalog(paths.catalog_mag_xml)
    assert len(out_cat) == n_events

    truth_by_id = {ev["event_id_hint"]: ev for ev in events_subset}
    abs_diffs = []
    for out_ev in out_cat:
        event_id = get_event_id(out_ev)
        truth_ev = truth_by_id[event_id]
        if len(out_ev.magnitudes) == 0:
            # Below min_stations -- still present, per the acceptance
            # criterion; just not counted in the recovery check below.
            continue
        recovered_ml = out_ev.magnitudes[0].mag
        abs_diffs.append(abs(recovered_ml - truth_ev["ml"]))

    assert len(abs_diffs) >= n_events // 2, "too few events recovered a magnitude at all"
    median_abs_diff = float(np.median(abs_diffs))
    assert median_abs_diff < 0.15, f"median |ΔML| = {median_abs_diff:.3f} (truth tolerance ~0.15)"
