"""Tests for OpenMEQ-c3l.4 (M7.4): ``openmeq.review.compile``.

Location (M6) and the batch review loop (M7.3) are exercised by their own
test modules against hand-built fixtures; this compile step only ever reads
already-produced QuakeML (``locate/catalog_auto.xml``,
``review/events/<event_id>.xml``) and ``review/state.json``, so -- per the
issue's own "Test with hand-built auto catalogues and hand-built 'reviewed'
QuakeML files" instruction -- every fixture here is built directly with
``openmeq.catalog`` + plain ``obspy`` objects. No NLLoc/ObsPyck/GUI is
needed anywhere in this module (the compile step never launches either),
and no large synthetic dataset is used (module doesn't touch waveform data
at all -- see the worker instructions' "Disk space" note).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from obspy import Catalog, UTCDateTime
from obspy.core.event import Arrival, Origin, Pick, WaveformStreamID
from obspy.core.event.base import ResourceIdentifier

from openmeq.catalog import (
    accepted,
    get_extra,
    make_event,
    read_catalog,
    write_catalog,
)
from openmeq.config import load_config
from openmeq.paths import RunPaths
from openmeq.review.compile import compile_review

NET = "OQ"
_OBSPYCK_NAMESPACE = "http://erdbeben-in-bayern.de/xmlns/0.1"


def _write_project_config(tmp_path: Path) -> Path:
    """A minimal, valid OpenMEQ project config. ``nlloc``/``hypodd`` are
    required sections of the schema but are never read by the compile step
    (it does no location/relocation work, out of scope per the issue), so
    their file references need not exist on disk."""
    project = tmp_path / "project"
    project.mkdir()

    config_yaml = project / "config.yaml"
    config_yaml.write_text(
        textwrap.dedent(
            """\
            project:
              name: review_compile_test
              output_dir: ./run

            data:
              waveform_dirs: [./data]
              start: 2026-01-01
              end: 2026-01-02

            parallel:
              n_workers: 1

            nlloc:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
              grid:
                origin: [-38.00, 176.00]
                size_km: [20, 20, 10]
                spacing_km: 0.5

            hypodd:
              velocity_model:
                type: 1d
                file: ./vel1d.txt
            """
        )
    )
    return config_yaml


def _make_picks_and_arrivals(t0: UTCDateTime, offsets: dict[str, dict[str, float]]):
    """``offsets``: ``{station: {"P": p_offset_s, "S": s_offset_s}}``."""
    picks = []
    arrivals = []
    for station, phases in offsets.items():
        for phase, off in phases.items():
            wid = WaveformStreamID(
                network_code=NET,
                station_code=station,
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
    return picks, arrivals


def _make_auto_event(
    event_id: str,
    t0: UTCDateTime,
    lon: float,
    lat: float,
    depth_m: float,
    offsets: dict[str, dict[str, float]],
):
    picks, arrivals = _make_picks_and_arrivals(t0, offsets)
    origin = Origin(
        time=t0,
        longitude=lon,
        latitude=lat,
        depth=depth_m,
        method_id=ResourceIdentifier("smi:openmeq/nlloc/passB"),
        arrivals=arrivals,
    )
    ev = make_event(event_id)
    ev.picks = picks
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    return ev


def _make_reviewed_event(
    event_id: str,
    t0: UTCDateTime,
    lon: float,
    lat: float,
    depth_m: float,
    offsets: dict[str, dict[str, float]],
    *,
    resource_id: str | None = None,
    with_obspyck_extra: bool = False,
):
    """A hand-built "reviewed" event, standing in for what ObsPyck's own
    save would produce: no ``evaluation_mode``/``method_id`` set on the
    picks/origin (ObsPyck's own output never sets those standard QuakeML
    fields, module docstring), optionally with the real erdbeben-in-bayern.de
    extra marker ObsPyck actually sets on every save
    (``obspyck.obspyck.ObsPyck.get_QUAKEML_string``)."""
    picks, arrivals = _make_picks_and_arrivals(t0, offsets)
    for pick in picks:
        pick.evaluation_mode = None
        pick.method_id = None
    origin = Origin(
        time=t0,
        longitude=lon,
        latitude=lat,
        depth=depth_m,
        arrivals=arrivals,
    )
    ev = make_event(event_id)
    if resource_id is not None:
        # Simulate ObsPyck's own setXMLEventID() overwriting resource_id --
        # M7.3's batch loop always restamps this before filing the file, but
        # a hand-built fixture (or a differently-produced reviewed file)
        # need not already carry the "correct" one, per the issue's own
        # note ("ObsPyck's saved events may have a different resource_id").
        ev.resource_id = ResourceIdentifier(resource_id)
    ev.picks = picks
    ev.origins = [origin]
    ev.preferred_origin_id = origin.resource_id
    if with_obspyck_extra:
        ev.extra = {
            "evaluationMode": {"value": "manual", "namespace": _OBSPYCK_NAMESPACE},
            "public": {"value": "false", "namespace": _OBSPYCK_NAMESPACE},
        }
    return ev


def _build_project(tmp_path: Path):
    config_path = _write_project_config(tmp_path)
    cfg = load_config(config_path)
    paths = RunPaths(cfg)
    paths.ensure()
    return cfg, paths


def _write_state(paths: RunPaths, state: dict[str, str]) -> None:
    import json

    paths.review_state_json.parent.mkdir(parents=True, exist_ok=True)
    paths.review_state_json.write_text(json.dumps(state))


def test_reviewed_event_replaces_automatic(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_offsets = {"ST1": {"P": 10.0, "S": 11.5}, "ST2": {"P": 10.2, "S": 11.9}}
    auto_event = _make_auto_event(eid, t0, 176.010, -38.020, 4000.0, auto_offsets)
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)

    # Deliberately shifted picks (+0.05 s) and a shifted location.
    reviewed_offsets = {
        "ST1": {"P": 10.05, "S": 11.55},
        "ST2": {"P": 10.25, "S": 11.95},
    }
    reviewed_event = _make_reviewed_event(eid, t0, 176.015, -38.017, 4200.0, reviewed_offsets)
    write_catalog(Catalog(events=[reviewed_event]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    final_catalog = compile_review(cfg, paths)

    assert len(final_catalog) == 1
    final_event = final_catalog[0]
    origin = final_event.preferred_origin()

    # The reviewed location/picks win, not the automatic ones.
    assert origin.longitude == pytest.approx(176.015)
    assert origin.latitude == pytest.approx(-38.017)
    assert origin.depth == pytest.approx(4200.0)

    picks_by_station_phase = {
        (p.waveform_id.station_code, p.phase_hint): p for p in final_event.picks
    }
    assert picks_by_station_phase[("ST1", "P")].time == t0 + 10.05
    assert picks_by_station_phase[("ST1", "S")].time == t0 + 11.55

    # OpenMEQ's own manual-review conventions were applied (ObsPyck's own
    # output never sets these).
    assert origin.evaluation_mode == "manual"
    assert str(origin.method_id) == "smi:openmeq/manual/obspyck"
    for pick in final_event.picks:
        assert pick.evaluation_mode == "manual"
        assert str(pick.method_id) == "smi:openmeq/manual/obspyck"

    # The automatic origin is kept, but only as a non-preferred origin for
    # comparison.
    assert len(final_event.origins) == 2
    other_origins = [o for o in final_event.origins if o is not origin]
    assert len(other_origins) == 1
    auto_origin_kept = other_origins[0]
    assert auto_origin_kept.longitude == pytest.approx(176.010)
    assert auto_origin_kept.latitude == pytest.approx(-38.020)
    assert str(auto_origin_kept.method_id) == "smi:openmeq/nlloc/passB"

    # Event id is stable (the filename stem), even though this fixture
    # never had a mismatched resource_id to restamp.
    assert str(final_event.resource_id) == f"smi:openmeq/event/{eid}"

    # Round-trips through the written catalog_final.xml too.
    assert paths.catalog_final_xml.exists()
    reread = read_catalog(paths.catalog_final_xml)
    assert len(reread) == 1
    assert reread[0].preferred_origin().longitude == pytest.approx(176.015)


def test_reviewed_event_restamps_mismatched_resource_id(tmp_path):
    """ObsPyck's own saved events may carry a different resource_id
    (``setXMLEventID``'s own timestamp-based id) -- the filename stem is
    authoritative regardless."""
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)

    reviewed_event = _make_reviewed_event(
        eid,
        t0,
        176.001,
        -38.001,
        4000.0,
        {"ST1": {"P": 10.02}},
        resource_id="smi:de.erdbeben-in-bayern/event/20260101000512.5",
    )
    write_catalog(Catalog(events=[reviewed_event]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    final_catalog = compile_review(cfg, paths)
    assert str(final_catalog[0].resource_id) == f"smi:openmeq/event/{eid}"


def test_rejected_event_flagged_not_deleted(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)
    _write_state(paths, {eid: "rejected"})

    final_catalog = compile_review(cfg, paths)

    assert len(final_catalog) == 1
    final_event = final_catalog[0]
    origin = final_event.preferred_origin()
    assert origin.evaluation_status == "rejected"
    assert any("rejected:review:manual review" in str(c.text) for c in final_event.comments)
    # Never deleted: accepted() excludes it, but it's still present.
    assert len(accepted(final_catalog)) == 0

    # Location is untouched (the automatic one -- there was no reviewed file).
    assert origin.longitude == pytest.approx(176.0)


def test_skipped_and_unattempted_pass_through_unchanged(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid_skipped = "ev20260101T000000_000"
    eid_unattempted = "ev20260101T010000_000"

    auto_events = [
        _make_auto_event(eid_skipped, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}}),
        _make_auto_event(eid_unattempted, t0 + 3600, 176.1, -38.1, 3000.0, {"ST1": {"P": 10.0}}),
    ]
    write_catalog(Catalog(events=auto_events), paths.catalog_auto_xml)
    _write_state(paths, {eid_skipped: "skipped"})  # eid_unattempted: no entry at all

    final_catalog = compile_review(cfg, paths)

    assert len(final_catalog) == 2
    by_id = {str(ev.resource_id).rsplit("/", 1)[-1]: ev for ev in final_catalog}

    for eid, expected_lon in ((eid_skipped, 176.0), (eid_unattempted, 176.1)):
        ev = by_id[eid]
        origin = ev.preferred_origin()
        assert origin.evaluation_status != "rejected"
        assert origin.longitude == pytest.approx(expected_lon)
        assert len(ev.origins) == 1  # no comparison origin appended
        for pick in ev.picks:
            assert pick.evaluation_mode == "automatic"  # untouched


def test_extra_namespace_translated(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)

    reviewed_event = _make_reviewed_event(
        eid, t0, 176.001, -38.001, 4000.0, {"ST1": {"P": 10.0}}, with_obspyck_extra=True
    )
    write_catalog(Catalog(events=[reviewed_event]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    final_catalog = compile_review(cfg, paths)
    final_event = final_catalog[0]

    # Translated into OpenMEQ's own extra namespace.
    assert get_extra(final_event, "obspyck_evaluationMode") == "manual"
    assert get_extra(final_event, "obspyck_public") is False

    # ObsPyck's own foreign-namespace keys don't leak through unexplained.
    extra = final_event.extra or {}
    assert "evaluationMode" not in extra
    assert "public" not in extra

    # Survives the write_catalog/read_catalog round trip too.
    reread = read_catalog(paths.catalog_final_xml)[0]
    assert get_extra(reread, "obspyck_evaluationMode") == "manual"
    reread_extra = reread.extra or {}
    assert "evaluationMode" not in reread_extra


def test_extra_namespace_absent_is_a_noop(tmp_path):
    """A reviewed file with no ObsPyck extra marker at all (e.g. a
    hand-built fixture, or a future non-ObsPyck review tool) compiles fine
    -- OpenMEQ's own evaluation_mode="manual" convention is still applied
    directly, independent of whether the foreign marker was present."""
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)
    reviewed_event = _make_reviewed_event(
        eid, t0, 176.001, -38.001, 4000.0, {"ST1": {"P": 10.0}}, with_obspyck_extra=False
    )
    write_catalog(Catalog(events=[reviewed_event]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    final_catalog = compile_review(cfg, paths)
    final_event = final_catalog[0]
    assert final_event.preferred_origin().evaluation_mode == "manual"
    assert get_extra(final_event, "obspyck_evaluationMode") is None


def test_diagnostics_written(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")

    eid_reviewed_1 = "ev20260101T000000_000"
    eid_reviewed_2 = "ev20260101T010000_000"
    eid_skipped = "ev20260101T020000_000"
    eid_rejected = "ev20260101T030000_000"

    auto_events = [
        _make_auto_event(eid_reviewed_1, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0, "S": 11.5}}),
        _make_auto_event(
            eid_reviewed_2,
            t0 + 3600,
            176.05,
            -38.05,
            3500.0,
            {"ST1": {"P": 10.0}, "ST2": {"P": 10.3, "S": 11.8}},
        ),
        _make_auto_event(eid_skipped, t0 + 7200, 176.1, -38.1, 3000.0, {"ST1": {"P": 10.0}}),
        _make_auto_event(eid_rejected, t0 + 10800, 176.2, -38.2, 2500.0, {"ST1": {"P": 10.0}}),
    ]
    write_catalog(Catalog(events=auto_events), paths.catalog_auto_xml)

    write_catalog(
        Catalog(
            events=[
                _make_reviewed_event(
                    eid_reviewed_1,
                    t0,
                    176.002,
                    -38.001,
                    4100.0,
                    {"ST1": {"P": 10.06, "S": 11.6}},
                )
            ]
        ),
        paths.review_event_file(eid_reviewed_1),
    )
    write_catalog(
        Catalog(
            events=[
                _make_reviewed_event(
                    eid_reviewed_2,
                    t0 + 3600,
                    176.052,
                    -38.048,
                    3400.0,
                    {"ST1": {"P": 10.04}, "ST2": {"P": 10.33, "S": 11.75}},
                )
            ]
        ),
        paths.review_event_file(eid_reviewed_2),
    )
    _write_state(
        paths,
        {
            eid_reviewed_1: "done",
            eid_reviewed_2: "done",
            eid_skipped: "skipped",
            eid_rejected: "rejected",
        },
    )

    compile_review(cfg, paths)

    diagnostics_dir = paths.diagnostics_dir("review")
    expected_files = [
        "pick_time_diff_P.png",
        "pick_time_diff_S.png",
        "location_shift_horizontal_km.png",
        "location_shift_depth_km.png",
        "location_shift_map.png",
        "summary.csv",
    ]
    for name in expected_files:
        f = diagnostics_dir / name
        assert f.exists(), f"missing diagnostic file: {f}"
        assert f.stat().st_size > 0

    import pandas as pd

    df = pd.read_csv(diagnostics_dir / "summary.csv")
    assert len(df) == 2  # one row per *reviewed* event
    assert set(df["event_id"]) == {eid_reviewed_1, eid_reviewed_2}
    assert (df["horizontal_shift_km"] > 0).all()


def test_skip_unchanged(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)
    _write_state(paths, {eid: "skipped"})

    compile_review(cfg, paths)
    assert paths.catalog_final_xml.exists()
    mtime_1 = paths.catalog_final_xml.stat().st_mtime_ns
    summary_csv = paths.diagnostics_dir("review") / "summary.csv"
    summary_mtime_1 = summary_csv.stat().st_mtime_ns

    # Re-run with nothing changed at all.
    compile_review(cfg, paths)
    mtime_2 = paths.catalog_final_xml.stat().st_mtime_ns
    summary_mtime_2 = summary_csv.stat().st_mtime_ns
    assert mtime_2 == mtime_1
    assert summary_mtime_2 == summary_mtime_1

    # A real change (a new reviewed file) does trigger a rewrite.
    reviewed_event = _make_reviewed_event(eid, t0, 176.01, -38.01, 4000.0, {"ST1": {"P": 10.1}})
    write_catalog(Catalog(events=[reviewed_event]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    final_catalog = compile_review(cfg, paths)
    assert final_catalog[0].preferred_origin().longitude == pytest.approx(176.01)


def test_zero_events_and_zero_reviewed(tmp_path):
    """An empty automatic catalogue (or one with no reviewed events at all)
    still compiles cleanly, and diagnostics are written (empty) rather than
    raising."""
    cfg, paths = _build_project(tmp_path)
    write_catalog(Catalog(events=[]), paths.catalog_auto_xml)

    final_catalog = compile_review(cfg, paths)
    assert len(final_catalog) == 0
    assert paths.catalog_final_xml.exists()
    assert (paths.diagnostics_dir("review") / "summary.csv").exists()


def test_reviewed_file_with_multiple_events_raises(tmp_path):
    cfg, paths = _build_project(tmp_path)
    t0 = UTCDateTime("2026-01-01T00:00:00")
    eid = "ev20260101T000000_000"

    auto_event = _make_auto_event(eid, t0, 176.0, -38.0, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[auto_event]), paths.catalog_auto_xml)

    ev1 = _make_reviewed_event(eid, t0, 176.01, -38.01, 4000.0, {"ST1": {"P": 10.0}})
    ev2 = _make_reviewed_event(eid, t0, 176.02, -38.02, 4000.0, {"ST1": {"P": 10.0}})
    write_catalog(Catalog(events=[ev1, ev2]), paths.review_event_file(eid))
    _write_state(paths, {eid: "done"})

    with pytest.raises(ValueError, match="exactly one Event"):
        compile_review(cfg, paths)
