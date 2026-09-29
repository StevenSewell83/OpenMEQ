"""Tests for the run-level ``event_funnel`` diagnostic (OpenMEQ-02o.8).

``openmeq.run_diagnostics.make_event_funnel`` against a synthetic set of
on-disk step outputs under ``tmp_path`` (some stages deliberately missing),
its gating inside :func:`openmeq.cli.run_all` (step modules monkeypatched out
exactly as in ``tests/test_run_all.py``), and the standalone
``openmeq diagnostics funnel`` subcommand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from obspy import Catalog, UTCDateTime
from obspy.core.event import Magnitude, Origin, Pick, WaveformStreamID

from openmeq.catalog import make_event, reject_event, set_extra, write_catalog
from openmeq.cli import StepFailed, main, run_all
from openmeq.config import load_config
from openmeq.paths import RunPaths
from openmeq.run_diagnostics import FUNNEL_COLUMNS, make_event_funnel

sys.path.insert(0, str(Path(__file__).parent))
from test_cli import _patched_example_config  # noqa: E402
from test_run_all import _install_fake_steps  # noqa: E402

EIDS = [f"ev20260101T00000{i}_000" for i in range(5)]


def _cfg_paths(tmp_path: Path, *, run_extended: bool | None = None):
    config_path, _ = _patched_example_config(tmp_path)
    cfg = load_config(config_path)
    cfg.diagnostics.run.extended = run_extended
    paths = RunPaths(cfg)
    paths.ensure()
    return config_path, cfg, paths


def _event(eid: str, *, method: str | None = None, rejected: bool = False, picks=()):
    ev = make_event(eid)
    t0 = UTCDateTime("2026-01-01")
    for phase in picks:
        ev.picks.append(
            Pick(time=t0, phase_hint=phase, waveform_id=WaveformStreamID("NZ", "ABC", "", "HHZ"))
        )
    if method is not None or rejected:
        origin = Origin(time=t0, latitude=-38.0, longitude=176.0, depth=5000.0)
        if method is not None:
            origin.method_id = method
        ev.origins.append(origin)
        ev.preferred_origin_id = origin.resource_id
        if rejected:
            reject_event(ev, reason="max_rms_s", step="locate")
    return ev


def _write_synthetic_run(paths: RunPaths) -> None:
    """5 detections -> 4 chunked -> 3 picked -> locate (2 past pass A, 1
    accepted) -> magnitude; review and relocate deliberately not run."""
    paths.detections_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"event_id": EIDS, "time": ["2026-01-01T00:00:00Z"] * 5}).to_csv(
        paths.detections_csv, index=False
    )
    for eid in EIDS[:4]:
        f = paths.chunk_file(eid)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")

    write_catalog(
        Catalog(
            events=[
                _event(EIDS[0], picks=("P", "S", "P")),
                _event(EIDS[1], picks=("P",)),
                _event(EIDS[2], picks=("S",)),
                _event(EIDS[3]),  # chunked, no picks
            ]
        ),
        paths.catalog_picks_xml,
    )

    passA, passB = "smi:openmeq/nlloc/passA", "smi:openmeq/nlloc/passB"
    auto = Catalog(
        events=[
            _event(EIDS[0], method=passB),
            _event(EIDS[1], method=passB, rejected=True),
            _event(EIDS[2], method=passA, rejected=True),
        ]
    )
    write_catalog(auto, paths.catalog_auto_xml)
    pd.DataFrame({"event_id": EIDS[:3], "status": ["accepted", "rejected", "rejected"]}).to_csv(
        paths.locate_summary_csv, index=False
    )

    mag = Catalog(events=[_event(EIDS[0], method=passB), _event(EIDS[1], method=passB)])
    mag[0].magnitudes.append(Magnitude(mag=1.2, magnitude_type="ML"))
    write_catalog(mag, paths.catalog_mag_xml)


def test_event_funnel_counts_and_missing_stages(tmp_path: Path) -> None:
    _, cfg, paths = _cfg_paths(tmp_path)
    _write_synthetic_run(paths)

    png = make_event_funnel(cfg, paths)

    assert png == paths.run_diagnostics_dir / "event_funnel.png"
    assert png.is_file() and png.stat().st_size > 0
    df = pd.read_csv(paths.run_diagnostics_dir / "event_funnel.csv", keep_default_na=False)
    assert list(df.columns) == FUNNEL_COLUMNS
    by = df.set_index("stage")
    expected = {
        "detections": "5",
        "chunked": "4",
        "picked": "3",
        "located_pass_a": "2",
        "located_pass_b": "1",
        "reviewed": "",
        "magnitude": "1",
        "relocated": "",
    }
    assert list(by.index) == list(expected)
    for stage, n in expected.items():
        assert str(by.loc[stage, "n_events"]) == n, stage
    assert by.loc["reviewed", "status"] == "not run"
    assert by.loc["relocated", "status"] == "not run"
    assert by.loc["picked", "detail"] == "P=3; S=2"
    assert by.loc["located_pass_b", "detail"] == "rejected=2"
    assert float(by.loc["chunked", "pct_of_detections"]) == 80.0


def test_event_funnel_relocated_vs_carried_over(tmp_path: Path) -> None:
    _, cfg, paths = _cfg_paths(tmp_path)
    passB = "smi:openmeq/nlloc/passB"
    events = [
        _event(EIDS[0], method=passB),
        _event(EIDS[1], method=passB),
        _event(EIDS[2], method=passB, rejected=True),
    ]
    for ev, status in zip(events, ["relocated", "not_linked", "not_linked"], strict=True):
        set_extra(ev, "hypodd_status", status)
    write_catalog(Catalog(events=events), paths.catalog_dd_xml)

    make_event_funnel(cfg, paths)

    df = pd.read_csv(paths.run_diagnostics_dir / "event_funnel.csv", keep_default_na=False)
    by = df.set_index("stage")
    assert str(by.loc["relocated", "n_events"]) == "1"
    assert by.loc["relocated", "detail"] == "carried_over=1"
    # No detections.csv -> "not run", and so no percentages anywhere.
    assert by.loc["detections", "status"] == "not run"
    assert (df["pct_of_detections"] == "").all()


def test_event_funnel_empty_run_all_not_run(tmp_path: Path) -> None:
    _, cfg, paths = _cfg_paths(tmp_path)
    assert make_event_funnel(cfg, paths) is not None
    df = pd.read_csv(paths.run_diagnostics_dir / "event_funnel.csv")
    assert (df["status"] == "not run").all()


def test_event_funnel_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, cfg, paths = _cfg_paths(tmp_path)
    import openmeq.run_diagnostics as rd

    def _boom(_paths):
        raise RuntimeError("boom")

    monkeypatch.setattr(rd, "collect_funnel", _boom)
    assert make_event_funnel(cfg, paths) is None


@pytest.mark.parametrize("enabled", [True, False])
def test_run_all_gates_event_funnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    config_path, cfg, paths = _cfg_paths(tmp_path, run_extended=enabled)
    _install_fake_steps(monkeypatch, [])

    run_all(cfg, paths, config_path=str(config_path))

    assert (paths.run_diagnostics_dir / "event_funnel.png").is_file() is enabled
    assert (paths.run_diagnostics_dir / "event_funnel.csv").is_file() is enabled


def test_run_all_writes_event_funnel_after_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, cfg, paths = _cfg_paths(tmp_path, run_extended=True)
    _install_fake_steps(monkeypatch, [], fail_at="pick")

    with pytest.raises(StepFailed):
        run_all(cfg, paths, config_path=str(config_path))

    assert (paths.run_diagnostics_dir / "event_funnel.png").is_file()


def test_run_all_survives_event_funnel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, cfg, paths = _cfg_paths(tmp_path, run_extended=True)
    _install_fake_steps(monkeypatch, [])
    import openmeq.run_diagnostics as rd

    def _boom(_cfg, _paths):
        raise RuntimeError("boom")

    monkeypatch.setattr(rd, "make_event_funnel", _boom)

    assert run_all(cfg, paths, config_path=str(config_path)).ok


def test_cli_diagnostics_funnel(tmp_path: Path) -> None:
    config_path, _, paths = _cfg_paths(tmp_path)  # gating off: standalone ignores it
    _write_synthetic_run(paths)

    result = CliRunner().invoke(main, ["diagnostics", "funnel", "-c", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (paths.run_diagnostics_dir / "event_funnel.png").is_file()
    assert (paths.run_diagnostics_dir / "event_funnel.csv").is_file()
    assert "event_funnel.png" in result.output
