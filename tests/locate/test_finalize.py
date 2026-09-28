"""Acceptance tests for M6.4 / OpenMEQ-xdf.4: ``openmeq.locate.finalize``.

Table-driven, one case per event-level rejection reason
(``openmeq.locate.finalize``'s own fixed-order criteria list), plus the
accepted case and the two "no origin at all" edge cases (a rejected pass A
with no successful relocation, and a total pass-B NLLoc failure).

No NLLoc grids/binary are needed here: ``finalize_event`` only reads
``cfg.nlloc.filter.events`` and the already-computed
``PassAResult``/``PassBResult`` objects, which this module hand-builds
directly (following ``tests/locate/test_pass_a.py``/``test_pass_b.py``'s own
convention of hand-building inputs, but one level up the pipeline -- no
``requires_nlloc`` marker, no real relocation). ``cfg`` comes from
``synthetic_project`` purely for a valid, fully-populated
``openmeq.config.Config`` to read ``cfg.nlloc.filter.events`` off of.
"""

from __future__ import annotations

import pytest
from obspy.core.event import Origin, OriginQuality, OriginUncertainty, Pick, WaveformStreamID

from openmeq.catalog import accepted, make_event
from openmeq.catalog import event_id as get_event_id
from openmeq.config import load_config
from openmeq.locate.finalize import finalize_event
from openmeq.locate.pass_a import PassAResult
from openmeq.locate.pass_b import PassBResult

_EVENT_ID = "ev20250101T000000_000"


@pytest.fixture
def cfg(synthetic_project):
    config_path, _truth = synthetic_project
    return load_config(config_path)


def _pick(phase: str, station: str = "STA01") -> Pick:
    pick = Pick()
    pick.waveform_id = WaveformStreamID(
        network_code="XX", station_code=station, location_code="00", channel_code="HHZ"
    )
    pick.phase_hint = phase
    pick.time_errors.uncertainty = 0.02
    return pick


def _picks(n: int, phase: str) -> list[Pick]:
    return [_pick(phase, station=f"STA{i:02d}") for i in range(n)]


def _origin(
    *,
    rms: float = 0.02,
    gap: float = 100.0,
    err_h_km: float | None = 0.1,
    err_z_km: float | None = 0.1,
) -> Origin:
    origin = Origin(time="2025-01-01T00:00:00.000000Z", latitude=1.0, longitude=2.0, depth=3000.0)
    origin.quality = OriginQuality(standard_error=rms, azimuthal_gap=gap)
    origin.origin_uncertainty = OriginUncertainty(
        horizontal_uncertainty=None if err_h_km is None else err_h_km * 1000.0
    )
    origin.depth_errors.uncertainty = None if err_z_km is None else err_z_km * 1000.0
    return origin


def _pass_a_ok(n_p: int = 8) -> PassAResult:
    return PassAResult(
        origin=_origin(),
        accepted_picks=_picks(n_p, "P"),
        removed=[],
        n_iterations=1,
        status="ok",
        reason=None,
    )


def _pass_b(
    *,
    n_p: int = 8,
    n_s: int = 4,
    origin: Origin | None = None,
    status: str = "ok",
    reason: str | None = None,
) -> PassBResult:
    if origin is None and status == "ok":
        origin = _origin()
    return PassBResult(
        origin=origin,
        accepted_p=_picks(n_p, "P"),
        accepted_s=_picks(n_s, "S"),
        removed=[],
        n_iterations=1,
        status=status,
        reason=reason,
    )


def _make_event() -> object:
    return make_event(_EVENT_ID)


class TestEventCriteria:
    """One case per ``cfg.nlloc.filter.events`` criterion, table-driven."""

    def test_accepted(self, cfg):
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b()

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "accepted"
        assert result.summary.reason == ""
        assert event.preferred_origin().evaluation_status != "rejected"
        assert result.summary.err_h_km == pytest.approx(0.1)
        assert result.summary.err_z_km == pytest.approx(0.1)
        assert result.summary.n_p == 8
        assert result.summary.n_s == 4
        assert event.preferred_origin().method_id.id == "smi:openmeq/nlloc/passB"

    def test_max_rms_s(self, cfg):
        cfg.nlloc.filter.events.max_rms_s = 0.05
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(rms=0.5))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "max_rms_s"
        assert event.preferred_origin().evaluation_status == "rejected"

    def test_min_phases(self, cfg):
        cfg.nlloc.filter.events.min_phases = 20
        cfg.nlloc.filter.events.min_p_phases = 1
        cfg.nlloc.filter.events.min_s_phases = 0
        event = _make_event()
        pass_a = _pass_a_ok(n_p=4)
        pass_b = _pass_b(n_p=4, n_s=2)

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "min_phases"
        assert result.summary.n_p == 4
        assert result.summary.n_s == 2

    def test_min_p_phases(self, cfg):
        cfg.nlloc.filter.events.min_phases = 1
        cfg.nlloc.filter.events.min_p_phases = 5
        cfg.nlloc.filter.events.min_s_phases = 0
        event = _make_event()
        pass_a = _pass_a_ok(n_p=4)
        # Pass B's own residual filtering can drop *more* P picks than pass
        # A already accepted -- re-checked here (module docstring decision).
        pass_b = _pass_b(n_p=3, n_s=2)

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "min_p_phases"

    def test_min_s_phases(self, cfg):
        cfg.nlloc.filter.events.min_phases = 1
        cfg.nlloc.filter.events.min_p_phases = 1
        cfg.nlloc.filter.events.min_s_phases = 3
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(n_p=8, n_s=1)

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "min_s_phases"

    def test_max_gap_deg(self, cfg):
        cfg.nlloc.filter.events.max_gap_deg = 180.0
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(gap=270.0))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "max_gap_deg"

    def test_max_err_h_km(self, cfg):
        cfg.nlloc.filter.events.max_err_h_km = 0.5
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(err_h_km=1.0))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "max_err_h_km"
        assert result.summary.err_h_km == pytest.approx(1.0)

    def test_max_err_z_km(self, cfg):
        cfg.nlloc.filter.events.max_err_z_km = 0.5
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(err_z_km=2.0))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "max_err_z_km"
        assert result.summary.err_z_km == pytest.approx(2.0)

    def test_first_failing_reason_wins(self, cfg):
        """Both rms and gap fail -- rms (earlier in the fixed order) wins."""
        cfg.nlloc.filter.events.max_rms_s = 0.05
        cfg.nlloc.filter.events.max_gap_deg = 50.0
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(rms=0.5, gap=270.0))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.reason == "max_rms_s"

    def test_missing_uncertainty_fields_are_not_treated_as_failures(self, cfg):
        """A hand-built Origin missing err_h_km/err_z_km never fails those
        two checks (module docstring decision #2)."""
        cfg.nlloc.filter.events.max_err_h_km = 0.001
        cfg.nlloc.filter.events.max_err_z_km = 0.001
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(origin=_origin(err_h_km=None, err_z_km=None))

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "accepted"
        assert result.summary.err_h_km is None
        assert result.summary.err_z_km is None


class TestNoOriginEdgeCases:
    def test_pass_a_rejected_with_no_origin_at_all(self, cfg):
        """A total pass-A NLLoc failure (min_p_phases shortfall via
        NLLocFailed, no successful relocation): still gets a preferred
        origin (a placeholder) so the event is properly flagged rejected
        and appears in the accepted()/rejected() split correctly."""
        from obspy import Catalog

        event = _make_event()
        pass_a = PassAResult(
            origin=None,
            accepted_picks=[],
            removed=[],
            n_iterations=1,
            status="rejected",
            reason="nlloc_failed:synthetic failure",
        )

        result = finalize_event(event, pass_a, None, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "nlloc_failed:synthetic failure"
        for field in ("time", "lat", "lon", "depth_km", "rms", "gap", "err_h_km", "err_z_km"):
            assert getattr(result.summary, field) is None
        assert event.preferred_origin() is not None
        assert event.preferred_origin().evaluation_status == "rejected"
        cat = Catalog(events=[event])
        assert len(accepted(cat)) == 0

    def test_pass_a_rejected_min_p_phases_uses_last_origin(self, cfg):
        """pass_a.status == 'rejected' with an origin present (min_p_phases
        shortfall, not an NLLoc crash): summary row uses pass A's last
        origin."""
        event = _make_event()
        pass_a = PassAResult(
            origin=_origin(rms=0.03),
            accepted_picks=_picks(3, "P"),
            removed=[],
            n_iterations=2,
            status="rejected",
            reason="min_p_phases",
        )

        result = finalize_event(event, pass_a, None, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "min_p_phases"
        assert result.summary.n_p == 3
        assert result.summary.n_s == 0
        assert result.summary.rms == pytest.approx(0.03)
        assert event.preferred_origin().method_id.id == "smi:openmeq/nlloc/passA"

    def test_total_pass_b_failure_with_no_origin(self, cfg):
        event = _make_event()
        pass_a = _pass_a_ok()
        pass_b = _pass_b(
            n_p=0, n_s=0, origin=None, status="rejected", reason="nlloc_failed:synthetic failure"
        )

        result = finalize_event(event, pass_a, pass_b, cfg)

        assert result.summary.status == "rejected"
        assert result.summary.reason == "nlloc_failed:synthetic failure"
        assert result.summary.n_p == 0
        assert result.summary.n_s == 0
        for field in ("time", "lat", "lon", "depth_km", "rms", "gap", "err_h_km", "err_z_km"):
            assert getattr(result.summary, field) is None
        assert event.preferred_origin().evaluation_status == "rejected"


def test_accepted_excludes_from_accepted_catalog(cfg):
    """``openmeq.catalog.accepted`` excludes a finalized-rejected event but
    keeps an accepted one (ARCHITECTURE.md §5)."""
    from obspy import Catalog

    accepted_event = make_event("ev20250101T000000_001")
    rejected_event = make_event("ev20250101T000000_002")

    finalize_event(accepted_event, _pass_a_ok(), _pass_b(), cfg)

    rej_cfg = cfg.model_copy(deep=True)
    rej_cfg.nlloc.filter.events.max_rms_s = 0.001
    finalize_event(rejected_event, _pass_a_ok(), _pass_b(origin=_origin(rms=1.0)), rej_cfg)

    cat = Catalog(events=[accepted_event, rejected_event])
    kept = accepted(cat)

    assert [get_event_id(ev) for ev in kept] == [get_event_id(accepted_event)]


def test_horizontal_error_falls_back_to_max_horizontal_uncertainty():
    """NLLoc OCT writes horUnc -1 (ObsPy: None) but still sets the ellipse axes."""
    from obspy.core.event import OriginUncertainty

    from openmeq.locate.finalize import horizontal_error_km

    assert horizontal_error_km(None) is None
    assert horizontal_error_km(OriginUncertainty()) is None
    assert horizontal_error_km(OriginUncertainty(max_horizontal_uncertainty=850.0)) == 0.85
    both = OriginUncertainty(horizontal_uncertainty=400.0, max_horizontal_uncertainty=850.0)
    assert horizontal_error_km(both) == 0.4
