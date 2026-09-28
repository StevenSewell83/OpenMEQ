"""Tests for OpenMEQ-c3l.3 (M7.3): ``openmeq.review.batch``.

Location (M6.x) is being built concurrently, so -- per the orchestrator's
own note on this issue -- test inputs are built by hand rather than reused
from a real locate step: a small 2-station project (following
``tests/test_review_obspyck_config.py``'s own minimal-fixture convention),
a hand-written ``locate/catalog_auto.xml`` (``openmeq.catalog`` + a plain
``obspy`` event tree), a hand-built merged StationXML, and raw chunk
miniSEED cut directly with ``obspy``/``numpy`` (standing in for what
``openmeq.chunk``/``DataIndex.read`` would hand back).

Every test here calls (transitively, via ``run_review`` ->
``generate_obspyck_config`` -> ``openmeq.nlloc_setup.link_obspyck_model``)
``openmeq.external.bin_path("NLLoc")``, matching
``tests/test_review_obspyck_config.py``'s own ``requires_nlloc`` module-wide
marker -- the non-GUI tests below never actually *launch* ObsPyck (they
mock ``openmeq.review.batch._launch_obspyck_subprocess``, the one dedicated
seam around ``subprocess.Popen`` -- see that function's own docstring for
why mocking the *global* ``subprocess.Popen`` instead would break unrelated
code in the same call stack), but the config/pluginpath generation step
that precedes launching it always runs for real.
``test_end_to_end_gui`` additionally requires a working, importable
``obspyck``/PyQt5 and is marked ``requires_gui``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from obspy import Catalog, Stream, Trace, UTCDateTime
from obspy.core.event import Arrival, Event, Origin, Pick, WaveformStreamID
from obspy.core.event.base import ResourceIdentifier

from openmeq.catalog import (
    event_id_from_time,
    make_event,
    read_catalog,
    reject_event,
    write_catalog,
)
from openmeq.config import load_config
from openmeq.external import ExternalToolMissing, bin_path
from openmeq.paths import RunPaths
from openmeq.review.batch import ReviewLaunchError, run_review

pytestmark = pytest.mark.requires_nlloc

NET = "OQ"
SR = 100.0
STATIONS = [
    ("ST1", -38.020000, 176.010000, 100.0),
    ("ST2", -37.985000, 175.975000, 50.0),
]


def _require_nlloc_tools() -> None:
    try:
        bin_path("NLLoc")
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


def _write_project_config(tmp_path: Path) -> Path:
    """A minimal, valid OpenMEQ project config (2 stations, no waveform
    archive) -- following ``tests/test_review_obspyck_config.py``'s own
    ``_write_minimal_project`` convention."""
    project = tmp_path / "project"
    project.mkdir()

    (project / "stations.csv").write_text(
        "net,sta,loc,lat,lon,elev_m,depth_m\n"
        f"{NET},ST1,,-38.02,176.01,100.0,0.0\n"
        f"{NET},ST2,,-37.985,175.975,50.0,0.0\n"
    )
    (project / "vel1d.txt").write_text(
        "# depth_top_km  vp_km_s  vs_km_s  rho_g_cc\n"
        "0.0   3.50  2.00  2.3\n"
        "1.0   4.50  2.60  2.5\n"
        "3.0   5.80  3.35  2.7\n"
    )

    config_yaml = project / "config.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            """\
            project:
              name: review_batch_test
              output_dir: ./run

            data:
              waveform_dirs: [./data]
              start: 2026-01-01
              end: 2026-01-02

            parallel:
              n_workers: 1

            stations:
              csv: ./stations.csv

            chunks:
              pre_s: 10
              post_s: 20

            nlloc:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
              grid:
                origin: [-38.00, 176.00]
                size_km: [20, 20, 10]
                spacing_km: 0.5
              obspyck_pluginpath: ./obspyck_plugin

            review:
              enabled: true
              obspyck_config: ./config/obspyck.cfg

            hypodd:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
            """
        )
    )
    return config_yaml


def _write_stations_xml(paths: RunPaths) -> None:
    from obspy.core.inventory import Channel, Inventory, Network, Site, Station
    from obspy.core.inventory.response import InstrumentSensitivity, Response

    net = Network(code=NET, stations=[])
    for name, lat, lon, elev in STATIONS:
        channels = []
        for cha, azimuth, dip in (("HHZ", 0, -90), ("HHN", 0, 0), ("HHE", 90, 0)):
            resp = Response(
                instrument_sensitivity=InstrumentSensitivity(
                    value=1.0, frequency=1.0, input_units="M/S", output_units="COUNTS"
                )
            )
            channels.append(
                Channel(
                    code=cha,
                    location_code="",
                    latitude=lat,
                    longitude=lon,
                    elevation=elev,
                    depth=0.0,
                    azimuth=azimuth,
                    dip=dip,
                    sample_rate=SR,
                    response=resp,
                )
            )
        net.stations.append(
            Station(
                code=name,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                site=Site(name="review batch test"),
                channels=channels,
                # Well before any event's chunk window (chunk_start is
                # cfg.chunks.pre_s seconds before the event -- ObsPyck's own
                # inv.get_coordinates/get_response lookups are exact-time,
                # and reject a query outside the station's validity window).
                start_date=UTCDateTime("2000-01-01"),
            )
        )
    inv = Inventory(networks=[net], source="OpenMEQ-c3l.3 test fixture")
    paths.stations_xml.parent.mkdir(parents=True, exist_ok=True)
    inv.write(str(paths.stations_xml), format="STATIONXML")


def _make_event(t0: UTCDateTime, event_id: str, lon: float, lat: float) -> Event:
    picks = []
    arrivals = []
    for name, *_rest in STATIONS:
        for phase, off in (("P", 10.0), ("S", 11.5)):
            wid = WaveformStreamID(
                network_code=NET,
                station_code=name,
                location_code="",
                channel_code="HHZ" if phase == "P" else "HHN",
            )
            pick = Pick(
                time=t0 + off,
                waveform_id=wid,
                phase_hint=phase,
                method_id=ResourceIdentifier("smi:openmeq/phasenet"),
                evaluation_mode="automatic",
            )
            picks.append(pick)
            arrivals.append(Arrival(pick_id=pick.resource_id, phase=phase))
    origin = Origin(
        time=t0 + 10.0,
        longitude=lon,
        latitude=lat,
        depth=4000.0,
        method_id=ResourceIdentifier("smi:openmeq/nlloc/passB"),
        arrivals=arrivals,
    )
    ev = make_event(event_id)
    ev.picks = picks
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    return ev


def _write_chunk(
    paths: RunPaths, event_id: str, start: UTCDateTime, duration: float, seed: int
) -> None:
    st = Stream()
    npts = int(duration * SR)
    rng = np.random.default_rng(seed)
    for name, *_rest in STATIONS:
        for cha in ("HHZ", "HHN", "HHE"):
            data = rng.normal(0, 5, npts).astype(np.float32)
            tr = Trace(data=data)
            tr.stats.update(
                dict(
                    network=NET,
                    station=name,
                    location="",
                    channel=cha,
                    sampling_rate=SR,
                    starttime=start,
                )
            )
            st.append(tr)
    chunk_path = paths.chunk_file(event_id)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    st.write(str(chunk_path), format="MSEED")


def _build_project(tmp_path: Path, n_events: int = 1):
    """Build a full hand-made OpenMEQ project: config, merged StationXML,
    ``locate/catalog_auto.xml`` (``n_events`` accepted events, an hour
    apart) and each event's raw ``chunks/<event_id>/<event_id>.mseed``.

    Returns
    -------
    (cfg, paths, config_path, event_ids)
        ``event_ids`` in ascending origin-time order (the order
        ``run_review`` itself must process them in).
    """
    config_path = _write_project_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    _write_stations_xml(paths)

    t0_base = UTCDateTime("2026-01-01T00:00:00")
    event_ids = []
    events = []
    for i in range(n_events):
        t0 = t0_base + i * 3600
        eid = event_id_from_time(t0)
        event_ids.append(eid)
        events.append(_make_event(t0, eid, lon=176.005 + 0.001 * i, lat=-38.003))
        chunk_start = t0 - cfg.chunks.pre_s
        duration = cfg.chunks.pre_s + cfg.chunks.post_s
        _write_chunk(paths, eid, chunk_start, duration, seed=1000 + i)

    write_catalog(Catalog(events=events), paths.catalog_auto_xml)
    return cfg, paths, config_path, event_ids


class _FakeCompletedProcess:
    """Stand-in for the ``subprocess.Popen`` object ``run_review`` uses.

    Only ``.wait()`` is ever called on it (``run_review`` never touches
    ``.returncode``/``.communicate``/... directly, per the module's own
    "never trust the exit code alone" design), so that's all this needs.
    """

    def __init__(self, returncode: int = 0) -> None:
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def _make_fake_popen(launched: list, *, save: bool = False, returncode: int = 0):
    """Build a ``subprocess.Popen``-compatible fake.

    Parameters
    ----------
    launched : list
        Every ``(cmd, cwd)`` call is appended here, for assertions.
    save : bool, optional
        If true, drop a fresh ``obspyck_<timestamp>.xml`` file into ``cwd``
        before returning -- simulating a reviewer who saved -- exactly the
        save-detection mechanism ``test_save_detection_via_file_polling``
        exercises (mock ``subprocess.Popen`` with a fake that writes the
        file into ``cwd`` before returning, per this issue's own acceptance
        criteria).
    returncode : int, optional
        What ``.wait()`` returns.
    """

    def _fake_popen(cmd, cwd=None, env=None):
        launched.append((cmd, Path(cwd) if cwd else None))
        if save:
            saved_cat = Catalog(events=[make_event("whatever-obspyck-calls-it")])
            ts = UTCDateTime().strftime("%Y%m%d%H%M%S")
            write_catalog(saved_cat, Path(cwd) / f"obspyck_{ts}.xml")
        return _FakeCompletedProcess(returncode)

    return _fake_popen


def test_resume_skips_done_events(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, event_ids = _build_project(tmp_path, n_events=2)
    done_id, other_id = event_ids

    paths.review_state_json.parent.mkdir(parents=True, exist_ok=True)
    paths.review_state_json.write_text(json.dumps({done_id: "done"}))

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(cfg, paths, config_path=str(config_path), non_interactive=True)

    assert result.processed == 1
    launched_cwds = [cwd for _cmd, cwd in launched]
    assert paths.review_work_dir(done_id) not in launched_cwds
    assert paths.review_work_dir(other_id) in launched_cwds

    state = json.loads(paths.review_state_json.read_text())
    assert state[done_id] == "done"
    assert state[other_id] == "skipped"


def test_save_detection_via_file_polling(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=True)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(cfg, paths, config_path=str(config_path))

    assert result.processed == 1
    assert result.done == 1
    assert len(launched) == 1

    reviewed_path = paths.review_event_file(event_id)
    assert reviewed_path.exists()
    saved_cat = read_catalog(reviewed_path)
    assert len(saved_cat) == 1
    # Filed under OpenMEQ's own event_id, not whatever ObsPyck's
    # setXMLEventID() put in the saved file's resource_id.
    assert str(saved_cat[0].resource_id) == f"smi:openmeq/event/{event_id}"

    state = json.loads(paths.review_state_json.read_text())
    assert state[event_id] == "done"

    # The raw ObsPyck save artifact is left in place in review/work/, not
    # deleted (kept for debugging, per the design's own pitfall note).
    work_dir = paths.review_work_dir(event_id)
    assert list(work_dir.glob("obspyck_*.xml"))


def test_no_save_prompts_and_records_choice(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    def _input_should_not_be_called(prompt):
        raise AssertionError("non_interactive=True must not prompt")

    result = run_review(
        cfg,
        paths,
        config_path=str(config_path),
        non_interactive=True,
        input_func=_input_should_not_be_called,
    )

    assert result.processed == 1
    assert result.skipped == 1
    assert result.done == 0
    assert not paths.review_event_file(event_id).exists()

    state = json.loads(paths.review_state_json.read_text())
    assert state[event_id] == "skipped"


def test_interactive_prompt_is_driveable_by_injected_input(tmp_path, monkeypatch):
    """The interactive prompt (non-``non_interactive`` mode) must be
    driveable from a test by injecting an ``input``-compatible callable --
    this issue's own explicit requirement."""
    _require_nlloc_tools()
    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    answers = iter(["r"])  # reject
    result = run_review(
        cfg,
        paths,
        config_path=str(config_path),
        input_func=lambda _prompt: next(answers),
    )

    assert result.rejected == 1
    state = json.loads(paths.review_state_json.read_text())
    assert state[event_id] == "rejected"


def test_quit_stops_the_loop_without_marking_the_current_event(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, event_ids = _build_project(tmp_path, n_events=2)
    first_id, second_id = event_ids

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(
        cfg,
        paths,
        config_path=str(config_path),
        input_func=lambda _prompt: "q",
    )

    assert result.quit is True
    assert result.processed == 1
    assert len(launched) == 1

    state = json.loads(paths.review_state_json.read_text())
    # The event being reviewed when "quit" was chosen gets no entry at all,
    # so it is offered again next run; the second event was never launched.
    assert first_id not in state
    assert second_id not in state


def test_reset_flag_clears_state(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    paths.review_state_json.parent.mkdir(parents=True, exist_ok=True)
    paths.review_state_json.write_text(json.dumps({event_id: "skipped"}))

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=True)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(
        cfg, paths, config_path=str(config_path), reset=[event_id], non_interactive=True
    )

    # Cleared, then immediately re-offered in the same run (and this time
    # saved), rather than staying "skipped" forever.
    assert result.processed == 1
    assert result.done == 1
    state = json.loads(paths.review_state_json.read_text())
    assert state[event_id] == "done"


def test_reset_all_flag_clears_every_entry(tmp_path, monkeypatch):
    _require_nlloc_tools()
    cfg, paths, config_path, event_ids = _build_project(tmp_path, n_events=2)

    paths.review_state_json.parent.mkdir(parents=True, exist_ok=True)
    paths.review_state_json.write_text(json.dumps({eid: "skipped" for eid in event_ids}))

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(
        cfg, paths, config_path=str(config_path), reset_all=True, non_interactive=True
    )

    assert result.processed == 2
    assert len(launched) == 2


def test_state_persisted_incrementally(tmp_path, monkeypatch):
    """Simulate an interruption after event N of M: state.json must reflect
    events 1..N, never lose them, even though the run as a whole raises."""
    _require_nlloc_tools()
    cfg, paths, config_path, event_ids = _build_project(tmp_path, n_events=3)

    launched: list = []
    calls = {"n": 0}

    def _flaky_popen(cmd, cwd=None, env=None):
        calls["n"] += 1
        launched.append(Path(cwd))
        if calls["n"] == 2:
            # Simulate the process being killed / crashing hard for the
            # second event -- a nonzero exit code is M7.1's confirmed
            # signal for a startup/config error, never a review decision.
            return _FakeCompletedProcess(returncode=1)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr("openmeq.review.batch._launch_obspyck_subprocess", _flaky_popen)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    with pytest.raises(ReviewLaunchError):
        run_review(cfg, paths, config_path=str(config_path), non_interactive=True)

    assert calls["n"] == 2
    state = json.loads(paths.review_state_json.read_text())
    # Event 1 completed (recorded as skipped: no save + non_interactive)
    # before event 2's hard failure aborted the run.
    assert event_ids[0] in state
    assert state[event_ids[0]] == "skipped"
    assert event_ids[1] not in state
    assert event_ids[2] not in state


def test_hard_launch_failure_is_not_recorded_as_skipped(tmp_path, monkeypatch):
    """A nonzero exit code must never be silently treated the same as a
    reviewer choosing to skip (docs/obspyck_notes.md summary item 7)."""
    _require_nlloc_tools()
    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess",
        lambda cmd, cwd=None, env=None: _FakeCompletedProcess(returncode=1),
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    with pytest.raises(ReviewLaunchError):
        run_review(cfg, paths, config_path=str(config_path), non_interactive=True)

    assert not paths.review_state_json.exists() or event_id not in json.loads(
        paths.review_state_json.read_text() or "{}"
    )


def test_rejected_events_are_excluded_from_the_offered_set(tmp_path, monkeypatch):
    """An already-rejected (by an earlier compile/reviewer pass over the
    automatic catalogue) event is never offered at all -- ``accepted()``
    filters it out before the resume check even runs."""
    _require_nlloc_tools()
    cfg, paths, config_path, event_ids = _build_project(tmp_path, n_events=2)

    cat = read_catalog(paths.catalog_auto_xml)
    reject_event(cat[0], reason="test", step="test")
    write_catalog(cat, paths.catalog_auto_xml)

    launched: list = []
    monkeypatch.setattr(
        "openmeq.review.batch._launch_obspyck_subprocess", _make_fake_popen(launched, save=False)
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    result = run_review(cfg, paths, config_path=str(config_path), non_interactive=True)

    assert result.processed == 1
    assert len(launched) == 1


@pytest.mark.requires_gui
def test_end_to_end_gui(tmp_path, monkeypatch):
    """Really launches ObsPyck offscreen (M7.1's confirmed in-process
    launch recipe) against one synthetic event, programmatically triggers
    a save (:mod:`openmeq.review._launch_obspyck`'s test-only
    ``OPENMEQ_REVIEW_TEST_AUTOSAVE`` hook -- a ``QTimer.singleShot``
    callback that nudges a pick and calls the exact same
    ``setXMLEventID``/``save_event_locally`` methods the real "save Event
    locally" button calls, then quits, the same way the OpenMEQ-c3l.1
    spike's own exit-code driver worked), and confirms the save is captured
    into ``review/events/<event_id>.xml`` through the real subprocess
    boundary (:mod:`openmeq.review.batch`'s actual ``subprocess.Popen`` /
    file-polling code path, not a mock) -- full GUI mouse/keyboard
    automation of a separate child process is impractical, so the "human
    clicks save" step is simulated via this in-process timer hook instead,
    while everything else (real ObsPyck launch, real compat patches, real
    NLLoc-backed pluginpath, real save-file detection) is exercised for
    real.
    """
    _require_nlloc_tools()
    try:
        bin_path("Vel2Grid")
        bin_path("Grid2Time")
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc toolchain not available: {exc}")
    pytest.importorskip("obspyck")
    pytest.importorskip("PyQt5")

    from openmeq.nlloc_setup import build_1d_grid, run_grid2time
    from openmeq.stations import station_table as load_station_table

    cfg, paths, config_path, (event_id,) = _build_project(tmp_path, n_events=1)

    table = load_station_table(cfg.stations)
    build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("OPENMEQ_REVIEW_TEST_AUTOSAVE", "1")

    result = run_review(cfg, paths, config_path=str(config_path), non_interactive=True)

    assert result.processed == 1
    assert result.done == 1, "expected the autosave hook's save to be detected"

    reviewed_path = paths.review_event_file(event_id)
    assert reviewed_path.exists()
    saved_cat = read_catalog(reviewed_path)
    assert len(saved_cat) == 1
    assert str(saved_cat[0].resource_id) == f"smi:openmeq/event/{event_id}"
    assert len(saved_cat[0].picks) == 4  # 2 stations x P/S

    state = json.loads(paths.review_state_json.read_text())
    assert state[event_id] == "done"
