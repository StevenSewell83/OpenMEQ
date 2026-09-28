"""PhaseNet pick recovery test against the synthetic dataset's truth arrivals
(M4.4, ``OpenMEQ-1m7.4``).

Runs the real pipeline end to end via ``CliRunner``
(``openmeq detect`` -> ``openmeq chunk``, from the shared, session-scoped
``pick_ready_project`` fixture, then ``openmeq pick`` here) and matches
``pick/catalog_picks.xml`` against ``synthetic_project``'s truth arrivals.

Event matching (detection-trigger jitter)
-------------------------------------------
A real ``openmeq detect`` run's own event ids/trigger times do **not**
exactly equal the truth's own ``event_id_hint``/origin time (STA/LTA
triggers a little after the true onset, and ``openmeq.catalog.
event_id_from_time`` is derived from that trigger time, not the origin) --
confirmed empirically while building this test: with hand-built
detections at the exact true origin time (``tests/pick/test_to_quakeml.py``'s
approach) ids match by construction, but with a *real* detect run they
generally do not. So each truth event is matched to its nearest real
detection by time (mirrors ``tests/integration/test_detect_integration.py``'s
own ``_match_truth_events`` "one-to-one nearest neighbour" pattern,
tolerance ``_EVENT_MATCH_TOLERANCE_S``), and that detection's own
``event_id`` is used to look up the corresponding ``Event`` in
``catalog_picks.xml``. Recall is computed only over truth arrivals whose
event was actually matched to a detection at all (an unmatched truth event
means detection itself missed it -- ``tests/integration/
test_detect_integration.py`` is the test that gates detection recall, not
this one).

Thresholds (qlg.10 / ``docs/phasenet_interface.md`` §10)
--------------------------------------------------------------
``OpenMEQ-qlg.10`` measured 100% P and 100% S recall (``ML>=0.5``) for this
exact synthetic dataset/preprocessing recipe, with ~19/~23 ms mean timing
residual. Per that task's own note ("M4.4 can set its own pick-recovery
test thresholds from these numbers... with headroom below the 100%/100%
measured"), this test requires **>=90% of true P arrivals matched within
0.05s** and **>=80% of true S arrivals matched within 0.08s** -- comfortably
below the measured 100%/100% and ~20ms residuals, so it only fails on a
real regression, never on ordinary machine-to-machine noise.
"""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner
from obspy import UTCDateTime

from openmeq.catalog import event_id as get_event_id
from openmeq.catalog import read_catalog
from openmeq.cli import main

_EVENT_MATCH_TOLERANCE_S = 5.0  # matches test_detect_integration.py's own _MATCH_TOLERANCE_S
_P_TOLERANCE_S = 0.05
_S_TOLERANCE_S = 0.08
_P_RECALL_THRESHOLD = 0.90
_S_RECALL_THRESHOLD = 0.80


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


def _nearest_detection_event_id(
    true_time: UTCDateTime, detections: list[tuple[str, UTCDateTime]]
) -> str | None:
    if not detections:
        return None
    event_id, t = min(detections, key=lambda row: abs(row[1] - true_time))
    if abs(t - true_time) <= _EVENT_MATCH_TOLERANCE_S:
        return event_id
    return None


@pytest.mark.requires_phasenet
@pytest.mark.slow
@pytest.mark.skipif(not _phasenet_available(), reason=skip_reason)
def test_pick_recovery_against_synthetic_truth(pick_ready_project):
    cfg, paths, config_path, truth = pick_ready_project

    runner = CliRunner()
    result = runner.invoke(main, ["pick", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    assert paths.catalog_picks_xml.is_file()
    catalog = read_catalog(paths.catalog_picks_xml)
    by_event_id = {get_event_id(event): event for event in catalog}

    detections_df = pd.read_csv(paths.detections_csv)
    detections = [
        (row["event_id"], UTCDateTime(row["time"])) for _, row in detections_df.iterrows()
    ]

    n_p_true = n_p_matched = 0
    n_s_true = n_s_matched = 0
    p_residuals: list[float] = []
    s_residuals: list[float] = []
    n_events_unmatched = 0

    for ev in truth["events"]:
        true_origin = ev["origin"]["time"]
        detected_event_id = _nearest_detection_event_id(true_origin, detections)
        if detected_event_id is None:
            n_events_unmatched += 1
            continue
        event = by_event_id.get(detected_event_id)
        picks = event.picks if event is not None else []

        for station_key, arrival in ev["arrivals"].items():
            for phase, tol in (("P", _P_TOLERANCE_S), ("S", _S_TOLERANCE_S)):
                truth_t = arrival[phase]
                if phase == "P":
                    n_p_true += 1
                else:
                    n_s_true += 1

                best_residual = None
                for pick in picks:
                    if pick.phase_hint != phase:
                        continue
                    wid = pick.waveform_id
                    if f"{wid.network_code}.{wid.station_code}" != station_key:
                        continue
                    residual = abs(pick.time - truth_t)
                    if best_residual is None or residual < best_residual:
                        best_residual = residual

                if best_residual is not None and best_residual <= tol:
                    if phase == "P":
                        n_p_matched += 1
                        p_residuals.append(best_residual)
                    else:
                        n_s_matched += 1
                        s_residuals.append(best_residual)

    p_recall = n_p_matched / n_p_true if n_p_true else 0.0
    s_recall = n_s_matched / n_s_true if n_s_true else 0.0
    p_mean_ms = (sum(p_residuals) / len(p_residuals) * 1000) if p_residuals else float("nan")
    s_mean_ms = (sum(s_residuals) / len(s_residuals) * 1000) if s_residuals else float("nan")

    print(
        f"\n[pick recovery] P: {n_p_matched}/{n_p_true} ({p_recall:.1%}) within "
        f"{_P_TOLERANCE_S}s, mean residual {p_mean_ms:.1f} ms"
    )
    print(
        f"[pick recovery] S: {n_s_matched}/{n_s_true} ({s_recall:.1%}) within "
        f"{_S_TOLERANCE_S}s, mean residual {s_mean_ms:.1f} ms"
    )
    print(
        f"[pick recovery] {n_events_unmatched}/{len(truth['events'])} truth event(s) "
        "unmatched to any detection"
    )

    assert p_recall >= _P_RECALL_THRESHOLD, (
        f"P recall {p_recall:.1%} ({n_p_matched}/{n_p_true}) below {_P_RECALL_THRESHOLD:.0%} "
        f"threshold (within {_P_TOLERANCE_S}s)"
    )
    assert s_recall >= _S_RECALL_THRESHOLD, (
        f"S recall {s_recall:.1%} ({n_s_matched}/{n_s_true}) below {_S_RECALL_THRESHOLD:.0%} "
        f"threshold (within {_S_TOLERANCE_S}s)"
    )
