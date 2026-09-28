"""Tests for openmeq.catalog."""

from obspy import UTCDateTime
from obspy.core.event import (
    Arrival,
    Catalog,
    Comment,
    Origin,
    Pick,
    WaveformStreamID,
)

from openmeq.catalog import (
    accepted,
    event_id,
    event_id_from_time,
    get_extra,
    make_event,
    read_catalog,
    reject_event,
    reject_pick,
    set_extra,
    write_catalog,
)


def _pick(phase="P", channel="HHZ"):
    return Pick(
        time=UTCDateTime("2025-01-01T00:00:00"),
        phase_hint=phase,
        waveform_id=WaveformStreamID(network_code="NN", station_code="STA", channel_code=channel),
    )


def test_event_id_from_time_format():
    t = UTCDateTime("2025-01-01T12:34:56.789123Z")
    assert event_id_from_time(t) == "ev20250101T123456_789"


def test_event_id_from_time_zero_millis():
    t = UTCDateTime("2025-01-01T00:00:00Z")
    assert event_id_from_time(t) == "ev20250101T000000_000"


def test_event_id_roundtrip():
    assert event_id(make_event("ev20250101T000000_000")) == "ev20250101T000000_000"


def test_make_event_resource_id():
    ev = make_event("ev20250101T000000_000")
    assert str(ev.resource_id) == "smi:openmeq/event/ev20250101T000000_000"


def test_extra_roundtrip_through_quakeml_file(tmp_path):
    pick = _pick()
    set_extra(pick, "probability", 0.987)

    ev = make_event("ev20250101T000000_000")
    ev.picks.append(pick)
    cat = Catalog(events=[ev])

    path = tmp_path / "cat.xml"
    write_catalog(cat, path)
    assert path.exists()

    cat2 = read_catalog(path)
    pick2 = cat2[0].picks[0]
    value = get_extra(pick2, "probability")
    assert value == 0.987
    assert isinstance(value, float)


def test_extra_missing_key_returns_default():
    pick = _pick()
    assert get_extra(pick, "nope") is None
    assert get_extra(pick, "nope", default="fallback") == "fallback"


def test_write_catalog_creates_parent_dir(tmp_path):
    ev = make_event("ev20250101T000000_000")
    cat = Catalog(events=[ev])
    path = tmp_path / "nested" / "dir" / "cat.xml"
    write_catalog(cat, path)
    assert path.exists()


def test_reject_pick_sets_status_weight_and_comment():
    pick = _pick()
    arrival = Arrival(phase="P", time_weight=1.0)

    reject_pick(pick, arrival, reason="residual>3sigma", step="locate_passA")

    assert pick.evaluation_status == "rejected"
    assert arrival.time_weight == 0
    texts = [c.text for c in pick.comments]
    assert "rejected:locate_passA:residual>3sigma" in texts


def test_reject_event_and_accepted_filters_it_out():
    ev_rejected = make_event("ev20250101T000000_000")
    origin_r = Origin(time=UTCDateTime("2025-01-01T00:00:00"), latitude=0.0, longitude=0.0)
    ev_rejected.origins.append(origin_r)
    ev_rejected.preferred_origin_id = origin_r.resource_id

    ev_kept = make_event("ev20250101T000001_000")
    origin_k = Origin(time=UTCDateTime("2025-01-01T00:00:01"), latitude=0.0, longitude=0.0)
    ev_kept.origins.append(origin_k)
    ev_kept.preferred_origin_id = origin_k.resource_id

    cat = Catalog(events=[ev_rejected, ev_kept])

    reject_event(ev_rejected, reason="bad_gap", step="review")

    assert origin_r.evaluation_status == "rejected"
    texts = [c.text for c in ev_rejected.comments]
    assert "rejected:review:bad_gap" in texts

    result = accepted(cat)
    result_ids = [event_id(ev) for ev in result]
    assert event_id(ev_kept) in result_ids
    assert event_id(ev_rejected) not in result_ids
    assert len(result) == 1


def test_reject_event_survives_quakeml_roundtrip(tmp_path):
    ev = make_event("ev20250101T000000_000")
    origin = Origin(time=UTCDateTime("2025-01-01T00:00:00"), latitude=0.0, longitude=0.0)
    ev.origins.append(origin)
    ev.preferred_origin_id = origin.resource_id
    ev.comments.append(Comment(text="unrelated note"))

    reject_event(ev, reason="bad_gap", step="review")

    cat = Catalog(events=[ev])
    path = tmp_path / "cat.xml"
    write_catalog(cat, path)
    cat2 = read_catalog(path)

    assert len(accepted(cat2)) == 0
    ev2 = cat2[0]
    assert ev2.preferred_origin().evaluation_status == "rejected"
    texts = [c.text for c in ev2.comments]
    assert "rejected:review:bad_gap" in texts


def test_accepted_keeps_event_with_no_preferred_origin():
    ev = make_event("ev20250101T000000_000")
    cat = Catalog(events=[ev])
    result = accepted(cat)
    assert len(result) == 1


def test_iso_z_always_has_microseconds():
    from obspy import UTCDateTime

    from openmeq.catalog import iso_z

    assert iso_z(UTCDateTime(2025, 1, 2, 3, 8, 15)) == "2025-01-02T03:08:15.000000Z"
    assert iso_z(UTCDateTime(2025, 1, 2, 3, 8, 15, 250000)) == "2025-01-02T03:08:15.250000Z"
