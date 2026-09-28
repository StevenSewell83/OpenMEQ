"""Location recovery test against ``synthetic_project``'s truth hypocentres
(M6.5 / OpenMEQ-xdf.5, the M6 epic's headline recovery test).

Runs the real pipeline end to end via ``CliRunner``: ``openmeq pick`` (real
PhaseNet) then ``openmeq locate`` (real NLLoc), starting from the shared,
session-scoped ``chunked_project`` fixture (``tests/locate/conftest.py``,
which already carries real ``openmeq detect``/``openmeq chunk`` output --
duplicating ``tests/pick/conftest.py``'s ``pick_ready_project`` pattern
since pytest fixtures aren't visible across sibling test directories).
Matches ``locate/summary.csv``'s accepted events to
``tests/data/make_synthetic.py``'s truth hypocentres by nearest origin time
(one-to-one, mirrors ``tests/integration/test_detect_integration.py``'s own
``_match_truth_events``), and asserts the location-recovery error at the
tight scale ARCHITECTURE.md §8 now documents for this dataset's exact 1D
layered synthetic travel times.

Design note -- deviates from the M6.5 issue's own "DESIGN" section
------------------------------------------------------------------------
That section recommended driving this test from a fixture-provided, fixed
picks catalogue built directly from truth with injected timing noise
(*not* a live PhaseNet run), so it would exercise only M5+M6 and carry
``requires_nlloc`` alone. This module instead runs the full real chain
(``qc``-tables-shortcut -> real ``detect``/``chunk``/``pick``/``locate``),
per this build's own explicit orchestrator instruction ("Integration and
recovery test via CliRunner: qc, detect, chunk, pick, locate on the
synthetic fixture into a tmp_path output dir"), which overrides the
original issue text per this project's own worker-instructions precedence
rule. Accordingly this test is marked both ``requires_nlloc`` *and*
``requires_phasenet`` (skipped if either is unavailable), and picks up
whatever real, end-to-end pick-time noise PhaseNet actually produces
(``OpenMEQ-qlg.10`` measured ~19-23 ms mean P/S timing residual on this
exact dataset) rather than a synthetic noise model -- a strictly harder,
more representative test than the DESIGN section's own proposal.

No truth events are excluded/corrupted for filter-testing purposes here
(unlike the DESIGN section's own proposal, which would have injected bad
picks on a subset of events to exercise pass A/B's residual filtering) --
every one of the dataset's ~20 truth events is a candidate for matching,
and the >=85% match-fraction threshold (below) is what absorbs the small
number of events real detection/picking/filtering may legitimately still
drop or mis-locate, rather than an explicit per-event exclusion list.

Thresholds -- final chosen values (measured while writing this test)
----------------------------------------------------------------------
ARCHITECTURE.md §8 (``OpenMEQ-m1f.5``): real single-event NLLoc recovery
against this dataset's own **clean, hand-built** picks measures a median of
~23 m / worst case ~102 m across all 20 events. This test instead locates
from **real, automatically picked** (PhaseNet) picks through the full
detect -> chunk -> pick -> two-pass-locate pipeline, so some extra error
from real pick-time jitter and detection/filtering is expected. A real run
against this exact dataset (CPU, n_workers=1, printed by this test's own
``[locate recovery]`` lines) measured:

- **20/20 (100%) truth events matched** to an accepted location within
  ``_EVENT_MATCH_TOLERANCE_S`` (the >=85% ``_MIN_MATCH_FRACTION`` threshold
  has headroom below this). This module originally measured 18/20 here
  (2 truth events rejected by the pass-A/B/``finalize`` chain), root-caused
  by M11.2 (``OpenMEQ-m1f.2``) to both events' azimuthal gap sitting a
  fraction of a degree past the then-default ``max_gap_deg = 270.0`` despite
  full 8-station P+S phase counts and small errors on both -- a
  network-geometry artifact of this dataset's small, ~ring-shaped array, not
  a real location failure. Fixed by raising the default to 285.0
  (``openmeq/config.py``'s ``EventsFilterConfig.max_gap_deg`` docstring has
  the full numbers/rationale), which is why this module now measures 20/20.
- **horizontal error: median 32.9 m, max 103.2 m.**
- **depth error: median 38.6 m, max 172.4 m** (depth's own worst case is
  looser than horizontal's, consistent with the shallow-event depth/
  origin-time trade-off M5.2 found and this issue's own notes restate).

Chosen thresholds, with headroom below the measured numbers above (so this
test only fails on a real regression, never on ordinary machine/BLAS-
library noise): ``_MEDIAN_HORIZ_KM``/``_MEDIAN_DEPTH_KM`` = **0.15 km**
(~4.5x the measured medians); ``_MAX_HORIZ_KM``/``_MAX_DEPTH_KM`` =
**0.5 km** (~5x the measured horizontal max, ~3x the measured depth max,
loose enough to absorb the shallow-event trade-off on a re-run without
being so loose it stops meaning anything); ``_MIN_MATCH_FRACTION`` =
**0.85** (below the measured 100%, with headroom for a legitimate
machine/BLAS-level NLLoc difference dropping at most a couple of events).
"""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from openmeq.cli import main

_EVENT_MATCH_TOLERANCE_S = 5.0  # matches test_detect_integration.py's own _MATCH_TOLERANCE_S
_MEDIAN_HORIZ_KM = 0.15
_MEDIAN_DEPTH_KM = 0.15
_MAX_HORIZ_KM = 0.5
_MAX_DEPTH_KM = 0.5
_MIN_MATCH_FRACTION = 0.85  # >= 85% of (non-excluded) truth events must match an accepted origin


def _phasenet_available() -> bool:
    from openmeq.external import phasenet_dir, phasenet_model_dir

    try:
        model_dir = phasenet_model_dir()
        predict_py = phasenet_dir() / "phasenet" / "predict.py"
    except Exception:
        return False
    return predict_py.is_file() and (model_dir / "checkpoint").is_file()


skip_reason = (
    "external/PhaseNet clone/model not found (set OPENMEQ_EXTERNAL_DIR, or run install.sh)"
)


def _match_truth_to_accepted(
    true_events: list[dict], accepted: pd.DataFrame, tolerance_s: float
) -> list[tuple[dict, pd.Series]]:
    """One-to-one nearest-origin-time matching, never claiming the same
    accepted row for two truth events (mirrors ``tests/integration/
    test_detect_integration.py``'s own ``_match_truth_events``)."""
    unclaimed = list(accepted.index)
    pairs: list[tuple[dict, pd.Series]] = []
    for event in sorted(true_events, key=lambda e: e["origin"]["time"]):
        if not unclaimed:
            break
        origin_time = event["origin"]["time"]
        nearest_idx = min(
            unclaimed, key=lambda i: abs(UTCDateTime(accepted.at[i, "time"]) - origin_time)
        )
        nearest_dt = abs(UTCDateTime(accepted.at[nearest_idx, "time"]) - origin_time)
        if nearest_dt <= tolerance_s:
            pairs.append((event, accepted.loc[nearest_idx]))
            unclaimed.remove(nearest_idx)
    return pairs


@pytest.mark.requires_nlloc
@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_locate_recovery_against_synthetic_truth(chunked_project):
    cfg, paths, config_path, truth = chunked_project

    runner = CliRunner()
    pick_result = runner.invoke(main, ["pick", "-c", str(config_path)])
    assert pick_result.exit_code == 0, pick_result.output

    locate_result = runner.invoke(main, ["locate", "-c", str(config_path)])
    assert locate_result.exit_code == 0, locate_result.output

    assert paths.catalog_auto_xml.is_file()
    summary = pd.read_csv(paths.locate_summary_csv)
    accepted = summary[summary["status"] == "accepted"].reset_index(drop=True)

    true_events = truth["events"]
    pairs = _match_truth_to_accepted(true_events, accepted, _EVENT_MATCH_TOLERANCE_S)

    n_matched = len(pairs)
    match_fraction = n_matched / len(true_events) if true_events else 0.0

    horiz_errors_km: list[float] = []
    depth_errors_km: list[float] = []
    for truth_event, row in pairs:
        true_lat = truth_event["origin"]["lat"]
        true_lon = truth_event["origin"]["lon"]
        true_depth_km = truth_event["origin"]["depth_km"]
        horiz_km = gps2dist_azimuth(true_lat, true_lon, row["lat"], row["lon"])[0] / 1000.0
        depth_err_km = abs(row["depth_km"] - true_depth_km)
        horiz_errors_km.append(horiz_km)
        depth_errors_km.append(depth_err_km)

    horiz_series = pd.Series(horiz_errors_km)
    depth_series = pd.Series(depth_errors_km)
    median_horiz = horiz_series.median() if not horiz_series.empty else float("nan")
    median_depth = depth_series.median() if not depth_series.empty else float("nan")
    max_horiz = horiz_series.max() if not horiz_series.empty else float("nan")
    max_depth = depth_series.max() if not depth_series.empty else float("nan")

    print(
        f"\n[locate recovery] {n_matched}/{len(true_events)} truth events matched "
        f"({match_fraction:.1%}) to an accepted location within {_EVENT_MATCH_TOLERANCE_S}s"
    )
    print(
        f"[locate recovery] horizontal error: median={median_horiz * 1000:.1f} m, "
        f"max={max_horiz * 1000:.1f} m"
    )
    print(
        f"[locate recovery] depth error: median={median_depth * 1000:.1f} m, "
        f"max={max_depth * 1000:.1f} m"
    )

    assert match_fraction >= _MIN_MATCH_FRACTION, (
        f"only {n_matched}/{len(true_events)} ({match_fraction:.1%}) truth events matched an "
        f"accepted location, below the {_MIN_MATCH_FRACTION:.0%} threshold"
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
