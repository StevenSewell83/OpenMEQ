"""Tests for the ``openmeq relocate`` CLI command, `openmeq.relocate.run_relocate`
and `openmeq.relocate.diagnostics.write_relocate_diagnostics` (M10.5 /
OpenMEQ-16d.5).

Covers PLAN.md §5's hypoDD diagnostics row/OpenMEQ-16d.6's EV/AQ/CND note,
ARCHITECTURE.md §2's ``run(cfg, paths, **opts)`` step contract, and this
issue's own acceptance criteria: a synthetic end-to-end ``openmeq relocate``
run improving on a known perturbation, diagnostics files present, and
skip-unchanged behaviour across the whole chain.

M6.5 (``openmeq locate``) is concurrent with this task (not depended on,
per this issue's own notes) -- ``magnitude/catalog_mag.xml`` is built by
hand here, reusing ``test_relocate_hypodd_runner.py``'s own
``_catalog_from_truth_perturbed``/``_dist_3d_km`` helpers (the same
0.5/0.3 km perturbation spec M10.4's own synthetic end-to-end test already
uses) rather than duplicating that logic -- the difference here is driving
the whole chain through the real ``openmeq relocate`` CLI command (M10.5's
own deliverable) instead of calling each M10.1-M10.4 function directly.

The expensive part (waveform cache + real ``ph2dt``/cross-correlation/
``hypoDD`` binaries) runs once per test module, via the module-scoped
``relocated_run`` fixture; the three acceptance tests below each make their
own independent assertions against that one run's output.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from test_relocate_hypodd_runner import _catalog_from_truth_perturbed, _dist_3d_km  # noqa: E402

from openmeq.catalog import event_id, get_extra, read_catalog, write_catalog
from openmeq.cli import main
from openmeq.paths import RunPaths
from openmeq.relocate.diagnostics import write_relocate_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = Path(os.environ.get("OPENMEQ_EXTERNAL_DIR", str(REPO_ROOT / "external")))
HYPODD_BIN_MISSING = not (
    (EXTERNAL_DIR / "bin" / "hypoDD").exists() and (EXTERNAL_DIR / "bin" / "ph2dt").exists()
)

pytestmark = pytest.mark.skipif(
    HYPODD_BIN_MISSING, reason="external/HYPODD (ph2dt/hypoDD) not available"
)


@pytest.fixture(scope="module", autouse=True)
def _set_external_dir():
    """Module-scoped equivalent of ``monkeypatch.setenv("OPENMEQ_EXTERNAL_DIR", ...)``
    (``monkeypatch`` itself is function-scoped only) -- every real-binary
    test in this module needs it, and `relocated_run` (below) is itself
    module-scoped.
    """
    old = os.environ.get("OPENMEQ_EXTERNAL_DIR")
    os.environ["OPENMEQ_EXTERNAL_DIR"] = str(EXTERNAL_DIR)
    yield
    if old is None:
        os.environ.pop("OPENMEQ_EXTERNAL_DIR", None)
    else:
        os.environ["OPENMEQ_EXTERNAL_DIR"] = old


@pytest.fixture(scope="module")
def relocate_cli_project(synthetic_project, tmp_path_factory):
    """``(cfg, paths, config_path, truth, perturbed_by_event)``: a private
    copy of the shared synthetic project with a hand-built, perturbed
    ``magnitude/catalog_mag.xml`` and real waveform chunks written --
    everything ``openmeq relocate`` itself needs as input, built without
    depending on ``openmeq locate``/``openmeq magnitude`` (M6.5 is
    concurrent with this task).
    """
    from openmeq.chunk import write_chunks
    from openmeq.config import load_config
    from openmeq.index import build_index

    config_path, truth = synthetic_project
    project_dir = tmp_path_factory.mktemp("relocate_cli_project") / "project"
    import shutil

    shutil.copytree(config_path.parent, project_dir)
    cfg = load_config(project_dir / "config.yaml")
    cfg.parallel.n_workers = 1
    paths = RunPaths(cfg)
    paths.ensure()

    rng = np.random.default_rng(20260928)
    cat, perturbed_by_event = _catalog_from_truth_perturbed(truth, rng)
    write_catalog(cat, paths.catalog_mag_xml)

    index = build_index(cfg)
    rows = []
    for ev_truth in truth["events"]:
        eid = ev_truth["event_id_hint"]
        stations = sorted(ev_truth["arrivals"].keys())
        rows.append(
            {
                "event_id": eid,
                "time": str(ev_truth["origin"]["time"]),
                "duration_s": 5.0,
                "coincidence_sum": len(stations),
                "n_stations": len(stations),
                "stations": ";".join(stations),
                "trace_ids": "",
            }
        )
    detections_df = pd.DataFrame(rows)
    write_chunks(detections_df, index, cfg, paths)

    return cfg, paths, project_dir / "config.yaml", truth, perturbed_by_event


@pytest.fixture(scope="module")
def relocated_run(relocate_cli_project):
    """Runs ``openmeq relocate -c <config>`` once (the module's one
    expensive real-binary invocation) and hands back everything the
    acceptance tests need.
    """
    cfg, paths, config_path, truth, perturbed_by_event = relocate_cli_project

    runner = CliRunner()
    result = runner.invoke(main, ["relocate", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    return cfg, paths, config_path, truth, perturbed_by_event, result


# ---------------------------------------------------------------------------
# test_relocate_command_end_to_end
# ---------------------------------------------------------------------------


def test_relocate_command_end_to_end(relocated_run):
    """``openmeq relocate`` runs the full chain and produces
    ``relocate/catalog_dd.xml`` with relocated origins closer to truth than
    the pre-hypoDD (perturbed) locations for the well-linked subset."""
    _cfg, paths, _config_path, truth, perturbed_by_event, _result = relocated_run

    assert paths.catalog_dd_xml.is_file()
    merged = read_catalog(paths.catalog_dd_xml)
    truth_by_event = {ev["event_id_hint"]: ev["origin"] for ev in truth["events"]}

    perturbed_errors = []
    relocated_errors = []
    for event in merged:
        eid = event_id(event)
        if get_extra(event, "hypodd_status") != "relocated":
            continue
        truth_origin = truth_by_event[eid]
        p_lat, p_lon, p_depth_km = perturbed_by_event[eid]
        perturbed_err = _dist_3d_km(
            p_lat,
            p_lon,
            p_depth_km,
            truth_origin["lat"],
            truth_origin["lon"],
            truth_origin["depth_km"],
        )
        new_origin = event.preferred_origin()
        relocated_err = _dist_3d_km(
            new_origin.latitude,
            new_origin.longitude,
            new_origin.depth / 1000.0,
            truth_origin["lat"],
            truth_origin["lon"],
            truth_origin["depth_km"],
        )
        perturbed_errors.append(perturbed_err)
        relocated_errors.append(relocated_err)

    assert perturbed_errors, "expected at least one relocated (well-linked) event"
    mean_perturbed = sum(perturbed_errors) / len(perturbed_errors)
    mean_relocated = sum(relocated_errors) / len(relocated_errors)
    n_improved = sum(1 for p, r in zip(perturbed_errors, relocated_errors, strict=True) if r < p)

    # "well below the perturbation" (the issue's own phrasing): the relocated
    # mean error should be a small fraction of the ~0.5-0.6 km perturbation
    # magnitude, matching M10.4's own default-config measurement (~0.04-0.06
    # km, docs/hypodd_tuning.md).
    assert mean_relocated < 0.15, f"mean relocated error {mean_relocated:.3f} km"
    assert mean_relocated < 0.5 * mean_perturbed
    assert n_improved >= len(perturbed_errors) - 1, (
        f"only {n_improved}/{len(perturbed_errors)} events improved"
    )


# ---------------------------------------------------------------------------
# test_diagnostics_written
# ---------------------------------------------------------------------------


def test_diagnostics_written(relocated_run):
    """Before/after map+sections, CC-distribution histogram, relative-shift
    histogram all present and non-empty; iteration-RMS plot present (this
    run's own ``hypoDD.log`` does parse successfully, per the real-binary
    output asserted directly below) -- the "logged skip on parse failure"
    branch is covered separately by `test_diagnostics_skip_on_unparseable_log`.
    """
    _cfg, paths, _config_path, _truth, _perturbed_by_event, _result = relocated_run

    diagnostics_dir = paths.diagnostics_dir("relocate")
    for name in (
        "locations_before.png",
        "locations_after.png",
        "dt_cc_cc_histogram.png",
        "relative_shift_hist.png",
        "summary.csv",
    ):
        f = diagnostics_dir / name
        assert f.is_file(), f"missing {f}"
        assert f.stat().st_size > 0, f"{f} is empty"

    # This run's real hypoDD.log does have a per-iteration table (the real
    # binary always writes one, docs module docstring) -- the RMS/
    # connectivity/AQ-CND plots must all have been written, not skipped.
    for name in ("iteration_rms.png", "iteration_connectivity.png", "iteration_aq_cnd.png"):
        f = diagnostics_dir / name
        assert f.is_file(), f"missing {f} -- expected hypoDD.log parsing to succeed"
        assert f.stat().st_size > 0

    summary = pd.read_csv(diagnostics_dir / "summary.csv")
    assert len(summary) > 0
    assert (summary["hypodd_status"] == "relocated").any()


def test_diagnostics_skip_on_unparseable_log(tmp_path, caplog):
    """`write_relocate_diagnostics` logs a WARNING and skips only the
    iteration plots (never raises) when ``hypoDD.log`` has no per-iteration
    table -- this issue's own pitfall note ("keep this diagnostic
    best-effort... rather than making the whole openmeq relocate command
    fail on a diagnostics-only parsing hiccup"). No real hypoDD binary
    needed -- pure unit test against hand-built catalogues.
    """
    from openmeq.catalog import make_event
    from openmeq.config import load_config

    config_path = REPO_ROOT / "config" / "example.yaml"
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path / "run"
    paths = RunPaths(cfg)
    paths.ensure()

    from obspy import Catalog
    from obspy.core.event import Origin

    cat = Catalog()
    event = make_event("ev20250101T000000_000")
    origin = Origin(time="2025-01-01T00:00:00Z", latitude=37.0, longitude=-122.0, depth=1000.0)
    event.origins = [origin]
    event.preferred_origin_id = origin.resource_id
    cat.append(event)

    write_catalog(cat, paths.catalog_mag_xml)
    write_catalog(cat, paths.catalog_dd_xml)

    paths.relocate_work_dir.mkdir(parents=True, exist_ok=True)
    (paths.relocate_work_dir / "hypoDD.log").write_text(
        "this is not a hypoDD.log iteration table at all\njust some other text\n"
    )

    with caplog.at_level("WARNING", logger="openmeq"):
        result = write_relocate_diagnostics(paths)

    assert result.iteration_plots_written is False
    assert result.iteration_rows == []
    assert any("iteration_rms.png" in rec.message for rec in caplog.records)

    diagnostics_dir = paths.diagnostics_dir("relocate")
    assert not (diagnostics_dir / "iteration_rms.png").exists()
    assert not (diagnostics_dir / "iteration_connectivity.png").exists()
    assert not (diagnostics_dir / "iteration_aq_cnd.png").exists()
    # The rest of diagnostics must still be written -- a parsing hiccup
    # never fails the whole diagnostics step.
    assert (diagnostics_dir / "summary.csv").is_file()
    assert (diagnostics_dir / "locations_before.png").is_file()


# ---------------------------------------------------------------------------
# test_skip_unchanged_end_to_end
# ---------------------------------------------------------------------------


def test_skip_unchanged_end_to_end(relocated_run):
    """Re-running ``openmeq relocate`` with nothing changed does not
    re-run ph2dt/hypoDD/xcorr (each already skip-unchanged via its own
    `openmeq.manifest.StepManifest` key, ARCHITECTURE.md §6c-2/§6g/§6h) --
    the real binaries' own output files' mtimes must not move.
    """
    _cfg, paths, config_path, _truth, _perturbed_by_event, _first_result = relocated_run

    dt_ct = paths.dt_ct
    dt_cc = paths.dt_cc
    hypodd_reloc = paths.relocate_work_dir / "hypoDD.reloc"
    for f in (dt_ct, dt_cc, hypodd_reloc):
        assert f.is_file()
    mtimes_before = {f: f.stat().st_mtime_ns for f in (dt_ct, dt_cc, hypodd_reloc)}

    runner = CliRunner()
    result = runner.invoke(main, ["relocate", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    for f, mtime_before in mtimes_before.items():
        assert f.stat().st_mtime_ns == mtime_before, f"{f} was rewritten on an unchanged rerun"

    # catalog_dd.xml is always re-derived (ARCHITECTURE.md §6h) even when
    # hypoDD itself is skipped -- still relocated as before.
    merged = read_catalog(paths.catalog_dd_xml)
    assert any(get_extra(ev, "hypodd_status") == "relocated" for ev in merged)
