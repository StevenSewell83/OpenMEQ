"""End-to-end recovery test against the synthetic dataset (M11.2 / OpenMEQ-m1f.2).

The project's single most important correctness test: run ``openmeq run``
(``openmeq.cli.run_all``, M11.1) once, end to end, against the
``synthetic_project`` fixture (M0.8/``OpenMEQ-qlg.8``, ``docs/ARCHITECTURE.md``
§8), and check recovery against truth at every stage -- detection, picking,
location, magnitude and relocation -- plus the pipeline's other headline
contracts (QC flags, every §3 output file).

Root cause of "18/20 events accepted at locate" (M6.5's open question,
resolved here)
------------------------------------------------------------------------------
M6.5 (``OpenMEQ-xdf.5``) measured the full ``qc -> detect -> chunk -> pick ->
locate`` chain accepting only 18 of the 20 synthetic truth events, with a
note to find out why and, if a default criterion was too strict for this
network, fix the default rather than loosen test thresholds. Investigating
here (a real, deterministic ``openmeq run`` against this dataset, seed
unchanged):

- **Detection**: 20/20 truth events matched one to one, 0 unmatched
  detections (``detections/detections.csv`` has exactly 20 rows for 20 truth
  events). Not the cause.
- **Picking**: 160/160 true P arrivals and 160/160 true S arrivals (20
  events x 8 stations) recovered within 0.05 s / 0.08 s. ``locate/
  removal_log.csv`` (pass A/B's own residual-filter removals) was
  **completely empty** -- not one pick was dropped by pass A or pass B's
  iterative residual filtering for any event. Not the cause.
- **Location filtering**: ``locate/summary.csv`` showed both rejected
  events (``ev20250101T182411_955``, ``ev20250102T231740_455``) failing
  ``reason == "max_gap_deg"`` with azimuth gaps of 270.6 deg and 270.3 deg
  against the (then-)default ``max_gap_deg = 270.0`` -- while using *all* 8
  stations' P and S picks (``n_p == n_s == 8``, matching every accepted
  event) and with ``err_z_km`` well inside ``max_err_z_km`` (the ``err_h_km``
  field is unset by NLLoc's OCT search for every event in this dataset, so
  that check never fires either way, consistent with
  ``openmeq/locate/finalize.py``'s documented "a missing value is never
  counted as a failure" rule). The other 18 accepted events span a
  continuous 75-269 deg gap distribution; the two rejects sit a fraction of
  a degree past the cutoff, not in some separate, badly-constrained
  population.

This is a network-geometry artifact, not a real location failure: on
``tests/data/make_synthetic.py``'s 8-station, ~13 km-aperture, roughly
ring-shaped array (event offsets drawn up to 9 km from the array centre in
each of x/y), an event sitting near/outside the station hull inherently
gets a large azimuthal gap even when every station's P and S arrivals
constrain it tightly. **Fix**: ``openmeq.config.EventsFilterConfig.max_gap_deg``
default raised 270.0 -> 285.0 (``openmeq/config.py``'s own docstring on that
field, and ``config/example.yaml``, have the full justification/numbers) --
a ~15 deg margin above the largest gap measured in this run, still enough
to reject a genuinely one-sided geometry. With the fix, a full ``openmeq
run`` against this dataset locates and accepts all 20/20 truth events (this
module's own measured numbers, see the tolerance table below).

Tolerances (chosen with headroom below the numbers measured while writing
this test, real chain, CPU, ``n_workers=1`` -- see ``docs/validation.md``
for the up-to-date measured table)
------------------------------------------------------------------------------
- **Detection**: >=90% of the 20 truth events matched to a detection within
  5 s, one to one (measured: 20/20, 100%) -- ``_MIN_DETECTION_MATCH_FRACTION``/
  ``_EVENT_MATCH_TOLERANCE_S``, mirrors ``tests/integration/
  test_detect_integration.py``'s own threshold and matching convention.
- **Picking**: for stations of a matched truth event, >=90% P recall within
  0.05 s and >=90% S recall within 0.08 s (measured: 160/160 and 160/160,
  100% each; P median 19 ms/max 44 ms, S median 26 ms/max 47 ms) --
  ``_MIN_PICK_RECALL``/``_P_PICK_TOL_S``/``_S_PICK_TOL_S``. The tolerances
  are set above PhaseNet's own measured real-pick timing jitter on this
  dataset (``OpenMEQ-qlg.10``'s ~19-23 ms mean residual), not at the
  network's exact-travel-time precision.
- **Location** (accepted events only): median horizontal/depth error
  <= 0.15 km, max <= 0.5 km (measured: horiz median 33 m/max 103 m; depth
  median 39 m/max 172 m) -- ``_MEDIAN_HORIZ_KM``/``_MEDIAN_DEPTH_KM``/
  ``_MAX_HORIZ_KM``/``_MAX_DEPTH_KM``, the same "tight §8 scale" thresholds
  ``tests/locate/test_locate_recovery.py`` already established for this
  dataset's exact 1D layered travel times (``OpenMEQ-m1f.5``), reused here
  rather than re-deriving a second convention. At least
  ``_MIN_LOCATE_MATCH_FRACTION`` (95%, i.e. at most one of the 20) of
  detection-matched events must reach an *accepted* location -- looser than
  this run's own measured 20/20 to absorb a legitimate machine/BLAS-level
  NLLoc difference, while still catching a real regression back towards
  M6.5's original 18/20 (which would fail this bound outright).
- **Magnitude**: median \\|delta ML\\| < 0.15 (measured: median 0.096, max
  0.102) -- ``_MAX_MEDIAN_ABS_DML``, the orchestrator note's own bound;
  comfortably above ``docs/ARCHITECTURE.md`` §8's own "synthetic ML is
  accurate to about +/-0.15" statement about the truth generator itself, so
  this mostly checks the pipeline doesn't add much error on top of that
  floor.
- **Relocation**: the dd catalogue's median horizontal/depth error must be
  no worse than the NLLoc catalogue's own, times ``_DD_SLACK`` (1.5x
  headroom so a legitimate small regression in hypoDD's own iteration
  doesn't flake this test); both medians are printed either way (measured:
  dd better on both axes -- horiz 31 m vs 33 m, depth 24 m vs 39 m).
- **QC flags**: exactly the two known issues that ``qc/channel_flags.json``
  is expected to flag (the spike and flat channel-days) -- the dataset's
  third known issue, the short data gap, is *deliberately* not flagged by
  ``cfg.qc.min_availability_pct``'s default (ARCHITECTURE.md §4.2's own gap
  policy), so "exactly two" (not three) is the correct invariant, derived
  here from ``truth["known_issues"]`` rather than hard-coded station names.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from openmeq.catalog import event_id, get_extra, read_catalog
from openmeq.cli import STEP_ORDER, run_all
from openmeq.config import load_config
from openmeq.paths import RunPaths

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
_EXTERNAL_MISSING = not (
    (EXTERNAL_DIR / "bin" / "NLLoc").exists()
    and (EXTERNAL_DIR / "bin" / "hypoDD").exists()
    and (EXTERNAL_DIR / "bin" / "ph2dt").exists()
    and (EXTERNAL_DIR / "PhaseNet").is_dir()
)
_skip_no_external = pytest.mark.skipif(
    _EXTERNAL_MISSING, reason="external/{bin/NLLoc,bin/hypoDD,bin/ph2dt,PhaseNet} not available"
)

#: Every test in this module drives one real, expensive chain (real
#: PhaseNet/NLLoc/hypoDD -- ~1-2 min on this machine) via the shared
#: ``e2e_result`` fixture, so the whole module carries the same markers
#: (ARCHITECTURE.md §1) -- auto-skipped when a required external tool is
#: missing, and excluded from the fast default/CI run via ``slow``.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_nlloc,
    pytest.mark.requires_hypodd,
    pytest.mark.requires_phasenet,
    _skip_no_external,
]

# ---------------------------------------------------------------------------
# Tolerances -- see the module docstring for the measured numbers/rationale
# behind each one.
# ---------------------------------------------------------------------------
_EVENT_MATCH_TOLERANCE_S = 5.0
_MIN_DETECTION_MATCH_FRACTION = 0.90
_MIN_PICK_RECALL = 0.90
_P_PICK_TOL_S = 0.05
_S_PICK_TOL_S = 0.08
_MIN_LOCATE_MATCH_FRACTION = 0.95
_MEDIAN_HORIZ_KM = 0.15
_MEDIAN_DEPTH_KM = 0.15
_MAX_HORIZ_KM = 0.5
_MAX_DEPTH_KM = 0.5
_MAX_MEDIAN_ABS_DML = 0.15
_DD_SLACK = 1.5


def _match_truth_to_times(
    true_events: list[dict], candidates: pd.DataFrame, time_col: str, tolerance_s: float
) -> list[tuple[dict, int]]:
    """One-to-one nearest-time matching of truth events to rows of
    ``candidates`` (never claiming the same row for two truth events) --
    mirrors ``tests/integration/test_detect_integration.py``'s own
    ``_match_truth_events`` / ``tests/locate/test_locate_recovery.py``'s own
    ``_match_truth_to_accepted``. Returns ``(truth_event, candidate_index)``
    pairs.
    """
    unclaimed = list(candidates.index)
    pairs: list[tuple[dict, int]] = []
    for event in sorted(true_events, key=lambda e: e["origin"]["time"]):
        if not unclaimed:
            break
        origin_time = event["origin"]["time"]
        nearest_idx = min(
            unclaimed, key=lambda i: abs(UTCDateTime(candidates.at[i, time_col]) - origin_time)
        )
        nearest_dt = abs(UTCDateTime(candidates.at[nearest_idx, time_col]) - origin_time)
        if nearest_dt <= tolerance_s:
            pairs.append((event, nearest_idx))
            unclaimed.remove(nearest_idx)
    return pairs


@pytest.fixture(scope="module", autouse=True)
def _set_external_dir():
    """Module-scoped equivalent of ``monkeypatch.setenv`` (``monkeypatch`` is
    function-scoped only) -- mirrors ``tests/test_run_all.py``'s identical
    fixture."""
    old = os.environ.get("OPENMEQ_EXTERNAL_DIR")
    os.environ["OPENMEQ_EXTERNAL_DIR"] = str(EXTERNAL_DIR)
    yield
    if old is None:
        os.environ.pop("OPENMEQ_EXTERNAL_DIR", None)
    else:
        os.environ["OPENMEQ_EXTERNAL_DIR"] = old


@pytest.fixture(scope="module")
def e2e_result(synthetic_project, tmp_path_factory) -> SimpleNamespace:
    """Runs ``openmeq.cli.run_all`` **once** against a private copy of the
    shared ``synthetic_project`` fixture, with output under ``tmp_path``
    (``tests/conftest.py``'s own "mutating steps must use a private copy"
    usage note) -- every test below only reads the result of this one real
    chain (real qc/detect/chunk/PhaseNet/NLLoc/hypoDD)."""
    config_path, truth = synthetic_project
    project_dir = tmp_path_factory.mktemp("e2e") / "project"
    shutil.copytree(config_path.parent, project_dir)
    project_config_path = project_dir / "config.yaml"

    cfg = load_config(project_config_path)
    paths = RunPaths(cfg)
    paths.ensure()

    result = run_all(cfg, paths, config_path=str(project_config_path))
    assert result.ok, (
        f"openmeq run failed at step {result.failed_step!r}: "
        f"{[(o.step, o.status, o.detail) for o in result.outcomes]}"
    )
    assert [o.step for o in result.outcomes] == list(STEP_ORDER)

    true_events = truth["events"]
    detections_df = pd.read_csv(paths.detections_csv)
    det_pairs = _match_truth_to_times(true_events, detections_df, "time", _EVENT_MATCH_TOLERANCE_S)
    # event_id_hint (detection's own trigger time) is what threads the same
    # truth event through every later stage (ARCHITECTURE.md §4.4: the event
    # id is fixed at detection and never changes afterwards) -- avoids
    # re-matching by nearest time at every stage.
    matched = [(true_ev, detections_df.at[idx, "event_id"]) for true_ev, idx in det_pairs]

    return SimpleNamespace(
        cfg=cfg,
        paths=paths,
        truth=truth,
        true_events=true_events,
        detections_df=detections_df,
        matched=matched,  # [(true_event, event_id), ...] -- one-to-one, detection-matched
        result=result,
    )


# ---------------------------------------------------------------------------
# Detection recovery
# ---------------------------------------------------------------------------


def test_detection_recovery(e2e_result: SimpleNamespace) -> None:
    r = e2e_result
    n_true = len(r.true_events)
    n_matched = len(r.matched)
    n_detections = len(r.detections_df)
    match_fraction = n_matched / n_true if n_true else 0.0

    print(
        f"\n[e2e] detection: {n_matched}/{n_true} truth events matched ({match_fraction:.1%}), "
        f"{n_detections} total detections"
    )

    assert match_fraction >= _MIN_DETECTION_MATCH_FRACTION, (
        f"only {n_matched}/{n_true} ({match_fraction:.1%}) truth events matched a detection "
        f"within {_EVENT_MATCH_TOLERANCE_S}s, below the {_MIN_DETECTION_MATCH_FRACTION:.0%} "
        f"threshold; {n_detections} detections total"
    )


# ---------------------------------------------------------------------------
# Picking recovery
# ---------------------------------------------------------------------------


def test_picking_recovery(e2e_result: SimpleNamespace) -> None:
    r = e2e_result
    picks_cat = read_catalog(r.paths.catalog_picks_xml)
    picks_by_eid = {event_id(ev): ev for ev in picks_cat}

    n_p_total = n_p_ok = n_s_total = n_s_ok = 0
    p_errors: list[float] = []
    s_errors: list[float] = []
    missing_events: list[str] = []

    for true_ev, eid in r.matched:
        pick_event = picks_by_eid.get(eid)
        if pick_event is None:
            missing_events.append(eid)
            continue
        picks_by_station: dict[str, dict[str, UTCDateTime]] = {}
        for pick in pick_event.picks:
            sta = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
            picks_by_station.setdefault(sta, {})[pick.phase_hint] = pick.time

        for sta, arrivals in true_ev["arrivals"].items():
            got = picks_by_station.get(sta, {})
            n_p_total += 1
            n_s_total += 1
            if "P" in got:
                err = abs(got["P"] - arrivals["P"])
                p_errors.append(err)
                if err <= _P_PICK_TOL_S:
                    n_p_ok += 1
            if "S" in got:
                err = abs(got["S"] - arrivals["S"])
                s_errors.append(err)
                if err <= _S_PICK_TOL_S:
                    n_s_ok += 1

    p_recall = n_p_ok / n_p_total if n_p_total else 0.0
    s_recall = n_s_ok / n_s_total if n_s_total else 0.0
    print(
        f"\n[e2e] picking: P {n_p_ok}/{n_p_total} ({p_recall:.1%}) within {_P_PICK_TOL_S}s, "
        f"S {n_s_ok}/{n_s_total} ({s_recall:.1%}) within {_S_PICK_TOL_S}s"
    )
    if p_errors:
        print(
            f"[e2e] picking: P error median={np.median(p_errors) * 1000:.1f}ms "
            f"max={np.max(p_errors) * 1000:.1f}ms"
        )
    if s_errors:
        print(
            f"[e2e] picking: S error median={np.median(s_errors) * 1000:.1f}ms "
            f"max={np.max(s_errors) * 1000:.1f}ms"
        )
    if missing_events:
        print(
            f"[e2e] picking: {len(missing_events)} matched event(s) with no picks: {missing_events}"
        )

    assert p_recall >= _MIN_PICK_RECALL, (
        f"P recall {n_p_ok}/{n_p_total} ({p_recall:.1%}) below {_MIN_PICK_RECALL:.0%} "
        f"within {_P_PICK_TOL_S}s"
    )
    assert s_recall >= _MIN_PICK_RECALL, (
        f"S recall {n_s_ok}/{n_s_total} ({s_recall:.1%}) below {_MIN_PICK_RECALL:.0%} "
        f"within {_S_PICK_TOL_S}s"
    )


# ---------------------------------------------------------------------------
# Location recovery
# ---------------------------------------------------------------------------


def test_location_recovery(e2e_result: SimpleNamespace) -> None:
    r = e2e_result
    summary = pd.read_csv(r.paths.locate_summary_csv)
    by_eid = summary.set_index("event_id")

    horiz_errors_km: list[float] = []
    depth_errors_km: list[float] = []
    not_accepted: list[tuple[str, str]] = []

    for true_ev, eid in r.matched:
        if eid not in by_eid.index:
            not_accepted.append((eid, "missing from locate/summary.csv"))
            continue
        row = by_eid.loc[eid]
        if row["status"] != "accepted":
            not_accepted.append((eid, str(row["reason"])))
            continue
        horiz_km = (
            gps2dist_azimuth(
                true_ev["origin"]["lat"], true_ev["origin"]["lon"], row["lat"], row["lon"]
            )[0]
            / 1000.0
        )
        depth_err_km = abs(row["depth_km"] - true_ev["origin"]["depth_km"])
        horiz_errors_km.append(horiz_km)
        depth_errors_km.append(depth_err_km)

    n_accepted_matched = len(horiz_errors_km)
    n_matched = len(r.matched)
    locate_fraction = n_accepted_matched / n_matched if n_matched else 0.0

    horiz_series = pd.Series(horiz_errors_km)
    depth_series = pd.Series(depth_errors_km)
    median_horiz = horiz_series.median() if not horiz_series.empty else float("nan")
    median_depth = depth_series.median() if not depth_series.empty else float("nan")
    max_horiz = horiz_series.max() if not horiz_series.empty else float("nan")
    max_depth = depth_series.max() if not depth_series.empty else float("nan")

    print(
        f"\n[e2e] location: {n_accepted_matched}/{n_matched} matched events accepted "
        f"({locate_fraction:.1%})"
    )
    if not_accepted:
        print(f"[e2e] location: not accepted -- {not_accepted}")
    print(
        f"[e2e] location: horizontal error median={median_horiz * 1000:.1f}m "
        f"max={max_horiz * 1000:.1f}m"
    )
    print(
        f"[e2e] location: depth error median={median_depth * 1000:.1f}m max={max_depth * 1000:.1f}m"
    )

    assert locate_fraction >= _MIN_LOCATE_MATCH_FRACTION, (
        f"only {n_accepted_matched}/{n_matched} ({locate_fraction:.1%}) detection-matched events "
        f"reached an accepted location, below {_MIN_LOCATE_MATCH_FRACTION:.0%}; "
        f"not-accepted: {not_accepted} -- see openmeq/config.py's EventsFilterConfig.max_gap_deg "
        f"docstring for this project's known network-geometry gap-filter tuning"
    )
    assert median_horiz <= _MEDIAN_HORIZ_KM, (
        f"median horizontal error {median_horiz:.3f} km exceeds {_MEDIAN_HORIZ_KM} km"
    )
    assert median_depth <= _MEDIAN_DEPTH_KM, (
        f"median depth error {median_depth:.3f} km exceeds {_MEDIAN_DEPTH_KM} km"
    )
    assert max_horiz <= _MAX_HORIZ_KM, (
        f"max horizontal error {max_horiz:.3f} km exceeds {_MAX_HORIZ_KM} km"
    )
    assert max_depth <= _MAX_DEPTH_KM, (
        f"max depth error {max_depth:.3f} km exceeds {_MAX_DEPTH_KM} km"
    )


# ---------------------------------------------------------------------------
# Magnitude recovery
# ---------------------------------------------------------------------------


def test_magnitude_recovery(e2e_result: SimpleNamespace) -> None:
    r = e2e_result
    mag_cat = read_catalog(r.paths.catalog_mag_xml)
    mag_by_eid = {event_id(ev): ev for ev in mag_cat}

    dml: list[float] = []
    missing: list[str] = []
    for true_ev, eid in r.matched:
        ev = mag_by_eid.get(eid)
        preferred = ev.preferred_magnitude() if ev is not None else None
        if preferred is None or preferred.mag is None:
            missing.append(eid)
            continue
        assert np.isfinite(preferred.mag), f"{eid}: non-finite ML {preferred.mag!r}"
        dml.append(abs(preferred.mag - true_ev["ml"]))

    median_dml = float(np.median(dml)) if dml else float("nan")
    max_dml = float(np.max(dml)) if dml else float("nan")
    print(
        f"\n[e2e] magnitude: {len(dml)}/{len(r.matched)} matched events with a finite ML; "
        f"median |dML|={median_dml:.3f} max={max_dml:.3f}"
    )
    if missing:
        print(f"[e2e] magnitude: {len(missing)} matched event(s) with no magnitude: {missing}")

    assert dml, "no matched event had a finite preferred magnitude to compare against truth"
    assert median_dml < _MAX_MEDIAN_ABS_DML, (
        f"median |delta ML| {median_dml:.3f} exceeds {_MAX_MEDIAN_ABS_DML}"
    )


# ---------------------------------------------------------------------------
# Relocation: dd catalogue no worse than NLLoc, ideally better -- both reported.
# ---------------------------------------------------------------------------


def test_relocation_no_worse_than_nlloc(e2e_result: SimpleNamespace) -> None:
    r = e2e_result
    nlloc_cat = read_catalog(r.paths.catalog_auto_xml)
    dd_cat = read_catalog(r.paths.catalog_dd_xml)
    nlloc_by_eid = {event_id(ev): ev for ev in nlloc_cat}
    dd_by_eid = {event_id(ev): ev for ev in dd_cat}

    nlloc_horiz: list[float] = []
    nlloc_depth: list[float] = []
    dd_horiz: list[float] = []
    dd_depth: list[float] = []
    n_relocated = n_not_linked = n_missing = 0

    for true_ev, eid in r.matched:
        true_lat, true_lon = true_ev["origin"]["lat"], true_ev["origin"]["lon"]
        true_depth_km = true_ev["origin"]["depth_km"]

        nlloc_ev = nlloc_by_eid.get(eid)
        if nlloc_ev is not None:
            origin = nlloc_ev.preferred_origin()
            if origin is not None and origin.latitude is not None:
                h = (
                    gps2dist_azimuth(true_lat, true_lon, origin.latitude, origin.longitude)[0]
                    / 1000.0
                )
                d = abs((origin.depth or 0.0) / 1000.0 - true_depth_km)
                nlloc_horiz.append(h)
                nlloc_depth.append(d)

        dd_ev = dd_by_eid.get(eid)
        if dd_ev is None:
            n_missing += 1
            continue
        status = get_extra(dd_ev, "hypodd_status", None)
        if status == "relocated":
            n_relocated += 1
        else:
            n_not_linked += 1
        origin = dd_ev.preferred_origin()
        if origin is None or origin.latitude is None:
            continue
        h = gps2dist_azimuth(true_lat, true_lon, origin.latitude, origin.longitude)[0] / 1000.0
        d = abs((origin.depth or 0.0) / 1000.0 - true_depth_km)
        dd_horiz.append(h)
        dd_depth.append(d)

    median_nlloc_horiz = float(np.median(nlloc_horiz)) if nlloc_horiz else float("nan")
    median_nlloc_depth = float(np.median(nlloc_depth)) if nlloc_depth else float("nan")
    median_dd_horiz = float(np.median(dd_horiz)) if dd_horiz else float("nan")
    median_dd_depth = float(np.median(dd_depth)) if dd_depth else float("nan")

    print(
        f"\n[e2e] relocation: {n_relocated} relocated, {n_not_linked} not linked, "
        f"{n_missing} missing"
    )
    print(
        f"[e2e] relocation: horizontal median -- NLLoc={median_nlloc_horiz * 1000:.1f}m "
        f"dd={median_dd_horiz * 1000:.1f}m"
    )
    print(
        f"[e2e] relocation: depth median -- NLLoc={median_nlloc_depth * 1000:.1f}m "
        f"dd={median_dd_depth * 1000:.1f}m"
    )
    better_horiz = median_dd_horiz <= median_nlloc_horiz
    better_depth = median_dd_depth <= median_nlloc_depth
    print(f"[e2e] relocation: dd better on horiz={better_horiz}, depth={better_depth}")

    assert dd_horiz, "no relocated/not-linked event had a usable dd origin to compare"
    assert median_dd_horiz <= _DD_SLACK * median_nlloc_horiz, (
        f"dd median horizontal error {median_dd_horiz:.3f} km is more than {_DD_SLACK}x "
        f"the NLLoc median {median_nlloc_horiz:.3f} km"
    )
    assert median_dd_depth <= _DD_SLACK * median_nlloc_depth, (
        f"dd median depth error {median_dd_depth:.3f} km is more than {_DD_SLACK}x "
        f"the NLLoc median {median_nlloc_depth:.3f} km"
    )


# ---------------------------------------------------------------------------
# QC flags: exactly the two known issues that are expected to be flagged
# ---------------------------------------------------------------------------


def test_qc_flags_exactly_the_two_known_issues(e2e_result: SimpleNamespace) -> None:
    """``truth["known_issues"]`` has three entries (gap, spikes, flat), but
    the gap is deliberately small enough not to trip ``cfg.qc.
    min_availability_pct``'s default (ARCHITECTURE.md §4.2's gap policy) --
    only the spike and flat channel-days should end up in ``qc/
    channel_flags.json`` with a non-empty ``reasons`` list. Derived from
    ``truth["known_issues"]`` rather than hard-coded station names, so this
    stays correct if the synthetic generator's station/anomaly placement
    ever changes.
    """
    import json

    r = e2e_result
    flags = json.loads(r.paths.channel_flags_json.read_text())

    flagged = [
        (seed_id, day)
        for seed_id, days in flags["flags"].items()
        for day, info in days.items()
        if info["reasons"]
    ]
    print(f"\n[e2e] qc flags: {flagged}")

    expected_kinds = {"spikes", "flat"}  # not "gap" -- see docstring
    expected_issues = [
        issue for issue in r.truth["known_issues"] if issue["kind"] in expected_kinds
    ]
    assert len(expected_issues) == 2, (
        f"expected exactly 2 non-gap known_issues in truth, got {expected_issues}"
    )

    assert len(flagged) == 2, (
        f"expected exactly 2 flagged channel-days (the spike and flat known_issues, not the "
        f"gap -- ARCHITECTURE.md §4.2's gap policy), got {len(flagged)}: {flagged}"
    )
    flagged_seed_ids = {seed_id for seed_id, _day in flagged}
    expected_seed_ids = {issue["seed_id"] for issue in expected_issues}
    assert flagged_seed_ids == expected_seed_ids, (
        f"flagged channels {flagged_seed_ids} don't match the expected spike/flat "
        f"known_issues channels {expected_seed_ids}"
    )


# ---------------------------------------------------------------------------
# Every ARCHITECTURE.md §3 output exists
# ---------------------------------------------------------------------------


def test_every_architecture_output_exists(e2e_result: SimpleNamespace) -> None:
    paths = e2e_result.paths
    expected_files = [
        paths.index_json,
        paths.stations_xml,
        paths.stations_csv,
        paths.metadata_report_json,
        paths.availability_csv,
        paths.daily_stats_csv,
        paths.monthly_stats_csv,
        paths.channel_flags_json,
        paths.detections_csv,
        paths.chunk_completeness_csv,
        paths.data_list_csv,
        paths.prep_report_csv,
        paths.catalog_picks_xml,
        paths.catalog_auto_xml,
        paths.locate_summary_csv,
        paths.removal_log_csv,
        paths.catalog_mag_xml,
        paths.station_magnitudes_csv,
        paths.catalog_dd_xml,
        paths.dt_ct,
        paths.dt_cc,
    ]
    missing = [p for p in expected_files if not p.is_file()]
    assert not missing, f"missing ARCHITECTURE.md §3 output file(s): {missing}"

    expected_dirs = [
        paths.qc_diagnostics_dir,
        paths.detect_diagnostics_dir,
        paths.diagnostics_dir("chunks"),
        paths.diagnostics_dir("chunks_pn"),
        paths.diagnostics_dir("pick"),
        paths.diagnostics_dir("locate"),
        paths.diagnostics_dir("magnitude"),
        paths.diagnostics_dir("relocate"),
        paths.passA_dir,
        paths.passB_dir,
        paths.relocate_work_dir,
        paths.relocate_cache_dir,
    ]
    missing_dirs = [p for p in expected_dirs if not p.is_dir()]
    assert not missing_dirs, f"missing ARCHITECTURE.md §3 output directory/-ies: {missing_dirs}"

    for step in STEP_ORDER:
        log_path = paths.log_file(step)
        assert log_path.is_file(), f"missing logs/{step}.log"

    for step in (
        "qc",
        "detections",
        "chunks",
        "chunks_pn",
        "pick",
        "locate",
        "magnitude",
        "relocate",
    ):
        manifest_path = paths.manifest_json(step)
        assert manifest_path.is_file(), f"missing {step}/manifest.json"
