"""Tests for the chunk step's opt-in extended QAQC (``OpenMEQ-02o.3``):
``chunks/diagnostics/extended/component_coverage.{csv,png}`` and
``raw_vs_pn_<event_id>.png``.

Uses small hand-built miniSEED chunks in ``tmp_path`` (no shared synthetic
dataset) and a minimal :class:`~openmeq.config.Config`; ``chunks_pn/`` is
produced by the real :func:`openmeq.pick.prep.prepare_chunk`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime

import openmeq.chunk_diagnostics as cd
from openmeq.chunk_diagnostics import (
    COMPONENT_COVERAGE_COLUMNS,
    compute_component_coverage,
    make_chunk_extended_diagnostics,
    make_chunk_pn_extended_diagnostics,
)
from openmeq.config import Config
from openmeq.paths import RunPaths
from openmeq.pick.prep import prepare_chunk

EVENT_IDS = ["ev_a", "ev_b", "ev_c"]
CHANNEL_MAP = {"EHZ": "Z", "EHN": "1", "EHE": "2"}


def _make_config(output_dir: Path, extended: bool | None, chunk_override: bool | None = None):
    diagnostics: dict = {"n_examples": 2}
    if extended is not None:
        diagnostics["extended"] = extended
    if chunk_override is not None:
        diagnostics["chunk"] = {"extended": chunk_override}
    return Config.model_validate(
        {
            "project": {"name": "test", "output_dir": str(output_dir)},
            "data": {
                "waveform_dirs": [str(output_dir)],
                "start": "2025-01-01",
                "end": "2025-01-02",
            },
            "stations": {"channel_map": CHANNEL_MAP},
            "nlloc": {
                "velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")},
                "grid": {"origin": [0.0, 0.0], "size_km": [1.0, 1.0, 1.0], "spacing_km": 0.1},
            },
            "hypodd": {"velocity_model": {"type": "1d", "file": str(output_dir / "vel1d.txt")}},
            "diagnostics": diagnostics,
        }
    )


def _trace(sta: str, cha: str, t0: UTCDateTime, seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal(3001) * 100 + 500).astype(np.int32)
    return Trace(
        data=data,
        header={
            "network": "XX",
            "station": sta,
            "location": "00",
            "channel": cha,
            "sampling_rate": 100.0,
            "starttime": t0,
        },
    )


def _write_raw_chunks(paths: RunPaths) -> None:
    """STA01: full EH[ZNE] every event; STA02: Z only, and absent from ev_c."""
    t0 = UTCDateTime(2025, 1, 1, 1, 0, 0)
    for i, event_id in enumerate(EVENT_IDS):
        traces = [_trace("STA01", cha, t0, 10 * i + k) for k, cha in enumerate(CHANNEL_MAP)]
        if event_id != "ev_c":
            traces.append(_trace("STA02", "EHZ", t0, 10 * i + 5))
        out = paths.chunk_file(event_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        Stream(traces).write(str(out), format="MSEED")


def _setup(tmp_path: Path, extended: bool | None, chunk_override: bool | None = None):
    cfg = _make_config(tmp_path / "run", extended, chunk_override)
    paths = RunPaths(cfg)
    paths.ensure()
    _write_raw_chunks(paths)
    return cfg, paths


def _extended_dir(paths: RunPaths) -> Path:
    return Path(paths.diagnostics_dir("chunks")) / "extended"


def _prep_all(cfg: Config, paths: RunPaths) -> None:
    for event_id in EVENT_IDS:
        prepare_chunk(event_id, paths.chunk_file(event_id), pd.DataFrame(), cfg)


def test_component_coverage_raw_only(tmp_path):
    cfg, paths = _setup(tmp_path, extended=True)
    df = compute_component_coverage(cfg, paths, EVENT_IDS, include_pn=False)

    assert list(df.columns) == COMPONENT_COVERAGE_COLUMNS
    by_key = df.set_index(["station", "component"])
    # channel_map applied: EHN/EHE -> 1/2
    assert set(by_key.loc["XX.STA01"].index) == {"Z", "1", "2"}
    assert by_key.loc[("XX.STA01", "1"), "frac_raw"] == pytest.approx(1.0)
    assert by_key.loc[("XX.STA02", "Z"), "frac_raw"] == pytest.approx(2 / 3)
    assert df["n_pn"].isna().all()


def test_component_coverage_pn_excludes_zero_filled(tmp_path):
    cfg, paths = _setup(tmp_path, extended=True)
    _prep_all(cfg, paths)
    df = compute_component_coverage(cfg, paths, EVENT_IDS, include_pn=True)
    by_key = df.set_index(["station", "component"])

    assert by_key.loc[("XX.STA02", "Z"), "frac_pn"] == pytest.approx(2 / 3)
    # STA02's horizontals are zero-filled by prep -> never "present".
    for comp in ("1", "2"):
        assert by_key.loc[("XX.STA02", comp), "n_pn"] == 0
        assert by_key.loc[("XX.STA02", comp), "n_raw"] == 0
    assert by_key.loc[("XX.STA01", "2"), "frac_pn"] == pytest.approx(1.0)


def test_extended_enabled_writes_files(tmp_path):
    cfg, paths = _setup(tmp_path, extended=True)
    make_chunk_extended_diagnostics(cfg, paths, EVENT_IDS)
    out = _extended_dir(paths)
    assert (out / "component_coverage.csv").is_file()
    assert (out / "component_coverage.png").is_file()
    assert not list(out.glob("raw_vs_pn_*.png"))  # deferred to the pick-prep hook

    _prep_all(cfg, paths)
    make_chunk_pn_extended_diagnostics(cfg, paths, EVENT_IDS)
    examples = sorted(p.name for p in out.glob("raw_vs_pn_*.png"))
    # n_examples=2, evenly sampled from 3 -> first and last
    assert examples == ["raw_vs_pn_ev_a.png", "raw_vs_pn_ev_c.png"]
    coverage = pd.read_csv(out / "component_coverage.csv")
    assert coverage["frac_pn"].notna().all()


def test_step_override_enables_extended(tmp_path):
    cfg, paths = _setup(tmp_path, extended=False, chunk_override=True)
    make_chunk_extended_diagnostics(cfg, paths, EVENT_IDS)
    assert (_extended_dir(paths) / "component_coverage.png").is_file()


@pytest.mark.parametrize(("extended", "chunk_override"), [(None, None), (True, False)])
def test_extended_disabled_writes_nothing(tmp_path, extended, chunk_override):
    cfg, paths = _setup(tmp_path, extended=extended, chunk_override=chunk_override)
    _prep_all(cfg, paths)
    make_chunk_extended_diagnostics(cfg, paths, EVENT_IDS)
    make_chunk_pn_extended_diagnostics(cfg, paths, EVENT_IDS)
    assert not _extended_dir(paths).exists()


def test_pn_hook_without_chunks_pn_skips_examples(tmp_path):
    cfg, paths = _setup(tmp_path, extended=True)
    make_chunk_pn_extended_diagnostics(cfg, paths, EVENT_IDS)
    out = _extended_dir(paths)
    assert (out / "component_coverage.png").is_file()
    assert not list(out.glob("raw_vs_pn_*.png"))


def test_figure_failure_never_raises(tmp_path, monkeypatch):
    cfg, paths = _setup(tmp_path, extended=True)
    _prep_all(cfg, paths)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cd, "_plot_raw_vs_pn", _boom)
    monkeypatch.setattr(cd, "_write_component_coverage", _boom)
    make_chunk_extended_diagnostics(cfg, paths, EVENT_IDS)
    make_chunk_pn_extended_diagnostics(cfg, paths, EVENT_IDS)
    assert not list(_extended_dir(paths).glob("*.png"))
