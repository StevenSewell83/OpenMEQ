"""Validation for M11 / OpenMEQ-m1f.5: the exact 1D layered first-arrival
travel times ``tests/data/make_synthetic.py`` now computes (direct ray by
ray-parameter shooting + head/refracted waves, see that module's own
"Travel-time method" docstring) against NLLoc's own ``Grid2Time`` output,
built via ``openmeq.nlloc_setup`` (never re-derived here).

Method
------
For each of several grid spacings, this test builds a fresh 1D velocity/
travel-time grid (``build_1d_grid`` + ``run_grid2time``) and then runs the
NonLinLoc ``Time2EQ`` utility (``EQMODE SRCE_TO_STA``) for every synthetic
event's *true* hypocentre against every station, which reads travel times
directly out of the just-built ``Grid2Time`` grids with no additional
timing error of its own (the same technique ``tests/test_nlloc_spike.py``
uses to build noise-free synthetic picks). Comparing those grid-read times
against ``tests/data/make_synthetic.py``'s own exact travel times (recovered
from ``truth["events"][i]["arrivals"]`` as ``arrival_time - origin_time``)
isolates *only* Grid2Time's own Podvin & Lecomte finite-difference eikonal
discretisation error -- there is no other source of disagreement once both
sides solve the same layered-model physics.

Findings (recorded here since this is the test the bd issue asks to
"report the difference as a function of spacing"; see also the module
docstring)
-----------------------------------------------------------------------
Measured against this synthetic network/model (8 stations, 20 events, the
``config/example.yaml``-derived default grid spacing of 0.1 km):

    spacing_km   max P diff (ms)   max S diff (ms)
    0.10         ~5.6              ~9.8
    0.05         ~3.4              ~5.9
    0.02         ~1.9              ~3.3

The error roughly halves each time the spacing halves -- a first-order
finite-difference discretisation signature -- confirming the residual is
Grid2Time's own grid error, not a modelling mismatch with this generator's
exact ray equations. The default 0.1 km spacing (``config/example.yaml``,
this dataset's own ``config.yaml``) is *not quite* inside this issue's
<5 ms P / <10 ms S target at every station-event pair, so this test builds
its own finer (0.05 km) validation grid -- comfortably inside the target
with a healthy margin -- rather than loosening the target or changing the
dataset's own (coarser, faster-to-build) default NLLoc grid spacing.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from openmeq.config import load_config
from openmeq.external import ExternalToolMissing, bin_path, run
from openmeq.nlloc_setup import build_1d_grid, run_grid2time, write_control_common
from openmeq.paths import RunPaths
from openmeq.stations import station_table as load_station_table

pytestmark = [pytest.mark.requires_nlloc, pytest.mark.slow]

#: This issue's own acceptance criterion.
_MAX_P_DIFF_S = 0.005
_MAX_S_DIFF_S = 0.010

#: The finest spacing used for the pass/fail assertion (see module
#: docstring: the dataset's own default 0.1 km grid measures ~5.6/~9.8 ms,
#: not quite inside target at every pair; 0.05 km comfortably is).
_ASSERTED_SPACING_KM = 0.05

#: Every spacing measured for the "difference as a function of spacing"
#: report (module docstring), coarsest first.
_REPORT_SPACINGS_KM = (0.1, _ASSERTED_SPACING_KM, 0.02)


def _require_nlloc_tools() -> None:
    try:
        for name in ("Vel2Grid", "Grid2Time", "Time2EQ"):
            bin_path(name)
    except ExternalToolMissing as exc:
        pytest.skip(f"NonLinLoc not available: {exc}")


def _parse_time2eq_obs(path: Path) -> dict[tuple[str, str], float]:
    """Parse a ``Time2EQ``-generated ``NLLOC_OBS`` file into
    ``{(station, phase): travel_time_s}`` (same format
    ``tests/test_nlloc_spike.py`` already parses)."""
    times: dict[tuple[str, str], float] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("PUBLIC_ID"):
            continue
        fields = line.split()
        station, phase = fields[0], fields[4]
        times[(station, phase)] = float(fields[8])
    return times


def _grid_travel_times(
    cfg, paths, truth: dict, work_dir: Path
) -> dict[tuple[str, str, str], float]:
    """``{(event_id_hint, station, phase): grid_travel_time_s}`` for every
    truth event, read directly out of this run's own ``Grid2Time`` grids via
    ``Time2EQ`` (``EQMODE SRCE_TO_STA``)."""
    grid_dir = paths.nlloc_grid_root.parent
    root_name = paths.nlloc_grid_root.name
    control_common = write_control_common(cfg, paths)

    stations = sorted(
        {net_sta.split(".")[1] for ev in truth["events"] for net_sta in ev["arrivals"]}
    )
    eqsta_lines = "\n".join(
        f"EQSTA  {sta}  {phase}  GAU 0.00   GAU 0.05" for sta in stations for phase in ("P", "S")
    )

    out: dict[tuple[str, str, str], float] = {}
    for ev in truth["events"]:
        origin = ev["origin"]
        ctrl = (
            control_common + "\n"
            f"EQFILES  {grid_dir / root_name}  {work_dir / 'synth'}\n"
            "EQMODE SRCE_TO_STA\n"
            f"EQSRCE  Ev  LATLON  {origin['lat']}  {origin['lon']}  {origin['depth_km']}  0.0\n"
            "EQVPVS  -1.0\n"
            "\n" + eqsta_lines + "\n"
            "\nEQQUAL2ERR 0.05 0.1 0.5 1.0 99999.9\n"
        )
        ctrl_path = work_dir / "time2eq.in"
        ctrl_path.write_text(ctrl)
        run([str(bin_path("Time2EQ")), ctrl_path.name], cwd=work_dir)

        grid_times = _parse_time2eq_obs(work_dir / "synth")
        for net_sta in ev["arrivals"]:
            _net, sta = net_sta.split(".")
            for phase in ("P", "S"):
                tt = grid_times.get((sta, phase))
                if tt is not None:
                    out[(ev["event_id_hint"], sta, phase)] = tt
    return out


def _max_diffs_at_spacing(
    synthetic_project, tmp_path_factory, spacing_km: float
) -> tuple[float, float, int]:
    """Build a fresh grid at ``spacing_km`` and return
    ``(max_p_diff_s, max_s_diff_s, n_pairs)`` against the generator's own
    exact truth arrivals."""
    config_path, truth = synthetic_project
    cfg = load_config(config_path)
    cfg.project.output_dir = tmp_path_factory.mktemp(f"traveltime_validation_{spacing_km}")
    cfg.nlloc.grid.spacing_km = spacing_km

    paths = RunPaths(cfg)
    paths.ensure()
    table = load_station_table(cfg.stations)

    build_1d_grid(cfg, paths, table)
    run_grid2time(cfg, paths, table)

    work_dir = cfg.project.output_dir / "time2eq"
    work_dir.mkdir(parents=True, exist_ok=True)
    grid_times = _grid_travel_times(cfg, paths, truth, work_dir)

    max_p_diff = 0.0
    max_s_diff = 0.0
    n = 0
    for ev in truth["events"]:
        origin_time = ev["origin"]["time"]
        for net_sta, phases in ev["arrivals"].items():
            _net, sta = net_sta.split(".")
            for phase in ("P", "S"):
                exact_t = phases[phase] - origin_time
                grid_t = grid_times.get((ev["event_id_hint"], sta, phase))
                if grid_t is None:
                    continue
                diff = abs(exact_t - grid_t)
                n += 1
                if phase == "P":
                    max_p_diff = max(max_p_diff, diff)
                else:
                    max_s_diff = max(max_s_diff, diff)

    # Clean up the (few-MB) grid/work directory immediately -- several
    # spacings are built in one test session (worker instructions: don't
    # leave scratch data larger than a few MB lying around).
    shutil.rmtree(cfg.project.output_dir, ignore_errors=True)

    return max_p_diff, max_s_diff, n


def test_exact_traveltimes_agree_with_grid2time_and_report_spacing_trend(
    synthetic_project, tmp_path_factory, capsys
):
    """Compare the generator's exact travel times against real ``Grid2Time``
    output at several grid spacings; assert the target (<5 ms P, <10 ms S)
    at :data:`_ASSERTED_SPACING_KM`, and print every spacing measured (the
    "report the difference as a function of spacing" requirement) -- see
    the module docstring for the recorded numbers and why the dataset's own
    default (coarser) NLLoc grid spacing is left unchanged.
    """
    _require_nlloc_tools()

    results = []
    for spacing_km in _REPORT_SPACINGS_KM:
        max_p_diff, max_s_diff, n = _max_diffs_at_spacing(
            synthetic_project, tmp_path_factory, spacing_km
        )
        results.append((spacing_km, max_p_diff, max_s_diff, n))

    with capsys.disabled():
        print("\nOpenMEQ-m1f.5 synthetic-vs-Grid2Time travel-time agreement:")
        for spacing_km, max_p_diff, max_s_diff, n in results:
            print(
                f"  spacing={spacing_km:.3f} km  n_pairs={n}  "
                f"max_P_diff={max_p_diff * 1000:.3f} ms  max_S_diff={max_s_diff * 1000:.3f} ms"
            )

    assert results, "expected at least one spacing measured"
    assert all(n > 0 for *_, n in results), "expected at least one station-event pair per spacing"

    # The finite-difference discretisation error should shrink as the grid
    # is refined (monotonic non-increase, coarsest to finest spacing) --
    # confirms the residual is genuinely Grid2Time's own grid error, not a
    # fixed modelling mismatch that finer spacing wouldn't touch.
    p_diffs = [r[1] for r in results]
    s_diffs = [r[2] for r in results]
    by_spacing_p = list(zip(_REPORT_SPACINGS_KM, p_diffs, strict=True))
    by_spacing_s = list(zip(_REPORT_SPACINGS_KM, s_diffs, strict=True))
    assert p_diffs == sorted(p_diffs, reverse=True), (
        f"expected P diff to shrink as spacing is refined: {by_spacing_p}"
    )
    assert s_diffs == sorted(s_diffs, reverse=True), (
        f"expected S diff to shrink as spacing is refined: {by_spacing_s}"
    )

    asserted = next(r for r in results if math.isclose(r[0], _ASSERTED_SPACING_KM))
    _spacing_km, max_p_diff, max_s_diff, _n = asserted
    assert max_p_diff < _MAX_P_DIFF_S, (
        f"max P diff {max_p_diff * 1000:.3f} ms exceeds target {_MAX_P_DIFF_S * 1000:.0f} ms "
        f"at spacing={_ASSERTED_SPACING_KM} km"
    )
    assert max_s_diff < _MAX_S_DIFF_S, (
        f"max S diff {max_s_diff * 1000:.3f} ms exceeds target {_MAX_S_DIFF_S * 1000:.0f} ms "
        f"at spacing={_ASSERTED_SPACING_KM} km"
    )
