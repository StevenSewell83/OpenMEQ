"""Tests for ``openmeq.cd11``, ``scripts/cd11tomseed.py``, ``data.convert.format: cd11``
and ``openmeq convert``.

Writes tiny hand-built little-endian CD-1.1 files laid out like the MacBan
Feb 2021 recorder files (one frame header, then one channel subframe per
channel per minute), so no real data is needed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from obspy import UTCDateTime, read

from openmeq import cd11
from openmeq.cli import main as cli_main
from openmeq.config import ConfigError, ParallelConfig, load_config
from openmeq.ingest import convert_raw_data
from openmeq.paths import RunPaths

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import cd11tomseed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SERIAL = ParallelConfig(n_workers=1)
RATE = 200
CHANS = ("chz", "chn", "che")


def _cd11_time(t: UTCDateTime) -> bytes:
    return f"{t.year}{t.julday:03d} {t.strftime('%H:%M:%S')}.{t.microsecond // 1000:03d}".encode()


def _subframe(sta: str, cha: str, t: UTCDateTime, data: np.ndarray) -> bytes:
    data = data.astype("<i4")
    site = f"{sta:<5}{cha}01".encode()
    desc = b"\0\0\0\0" + site + b"s4" + struct.pack("<ff", 0.0, 1.0)
    body = (
        struct.pack("<i", 0)  # authentication offset
        + desc
        + _cd11_time(t)
        + struct.pack("<iii", round(len(data) / RATE * 1000), len(data), 48)
        + bytes(48)  # channel status
        + struct.pack("<i", data.nbytes)
        + data.tobytes()
        + struct.pack("<iii", 0, 0, 0)  # subframe count, auth key id, auth size
    )
    return struct.pack("<i", len(body)) + body


def _write_cd11(path: Path, sta: str, t0: UTCDateTime, minutes: int, value0: int = 0) -> None:
    """``minutes`` minutes of 3-channel data; sample i of the file = value0 + i."""
    chan_str = b"".join(f"{sta:<5}{c}01".encode() for c in CHANS)
    head = (
        struct.pack("<ii", 5, 0)
        + sta.encode().ljust(8, b"\0")
        + b"FILE_CON"
        + struct.pack("<qi", 0, 0)
        + struct.pack("<ii", len(CHANS), 60000)
        + _cd11_time(t0)
        + struct.pack("<i", len(chan_str))
        + chan_str.ljust((len(chan_str) + 3) // 4 * 4, b"\0")
    )
    subs = b""
    for m in range(minutes):
        data = value0 + np.arange(m * 60 * RATE, (m + 1) * 60 * RATE)
        for c in CHANS:
            subs += _subframe(sta, c, t0 + 60 * m, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + subs)


def _name(kind: str, sta: str, t: UTCDateTime) -> str:
    return f"{t.strftime('%Y%m%d_%H%M%S')}_{kind}_{sta}.cd11"


@pytest.fixture
def cd11_tree(tmp_path: Path) -> Path:
    """MB-01: 23:58-00:02 continuous (crosses midnight) + an identical event copy
    of 23:59-00:00; MB-7A: one minute."""
    root = tmp_path / "raw"
    t0 = UTCDateTime(2021, 2, 12, 23, 58)
    _write_cd11(root / "MB-01/cont" / _name("C", "MB-01", t0), "MB-01", t0, 4)
    t1 = t0 + 60
    _write_cd11(root / "MB-01/event" / _name("E", "MB-01", t1), "MB-01", t1, 1, value0=60 * RATE)
    t2 = UTCDateTime(2021, 2, 12, 1)
    _write_cd11(root / "MB-07/cont" / _name("C", "MB-7A", t2), "MB-7A", t2, 1)
    return root


def test_read_cd11_decodes_every_subframe(cd11_tree: Path) -> None:
    src = next((cd11_tree / "MB-01/cont").glob("*.cd11"))
    sfs = cd11.read_cd11(src)
    assert len(sfs) == 4 * 3
    assert [s["cha"] for s in sfs[:3]] == list(CHANS)
    assert sfs[3]["start"] == UTCDateTime(2021, 2, 12, 23, 59)
    assert sfs[3]["rate"] == RATE
    np.testing.assert_array_equal(sfs[3]["data"][:3], [60 * RATE, 60 * RATE + 1, 60 * RATE + 2])


def test_convert_merges_event_files_and_splits_at_midnight(cd11_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "mseed"
    results = cd11.convert(cd11_tree, out, SERIAL)
    assert not any(r["warnings"] for r in results)
    day1 = read(str(out / "2021/MB01/EHZ.D/MB.MB01..EHZ.D.2021.043.mseed"))
    day2 = read(str(out / "2021/MB01/EHN.D/MB.MB01..EHN.D.2021.044.mseed"))
    assert len(day1) == len(day2) == 1  # the identical event overlap collapsed
    assert day1[0].stats.starttime == UTCDateTime(2021, 2, 12, 23, 58)
    assert day1[0].stats.npts == 2 * 60 * RATE
    assert day2[0].stats.starttime == UTCDateTime(2021, 2, 13)
    assert day2[0].data[0] == 2 * 60 * RATE
    assert (out / "2021/MB7A/EHE.D/MB.MB7A..EHE.D.2021.043.mseed").is_file()


def test_convert_reports_conflicting_overlap(cd11_tree: Path, tmp_path: Path) -> None:
    t = UTCDateTime(2021, 2, 12, 23, 59)
    _write_cd11(cd11_tree / "MB-01/event" / _name("E", "MB-01", t), "MB-01", t, 1, value0=-5)
    results = cd11.convert(cd11_tree, tmp_path / "mseed", SERIAL)
    warnings = [w for r in results for w in r["warnings"]]
    assert len(warnings) == 3 and "overlap" in warnings[0]


def test_convert_never_overwrites_without_flag_and_dry_run_writes_nothing(
    cd11_tree: Path, tmp_path: Path
) -> None:
    dry = tmp_path / "dry"
    results = cd11.convert(cd11_tree, dry, SERIAL, dry_run=True)
    assert sum(r["written"] for r in results) == 9 and not dry.exists()
    out = tmp_path / "mseed"
    cd11.convert(cd11_tree, out, SERIAL)
    again = cd11.convert(cd11_tree, out, SERIAL)
    assert sum(r["written"] for r in again) == 0 and sum(r["skipped"] for r in again) == 9
    forced = cd11.convert(cd11_tree, out, SERIAL, overwrite=True)
    assert sum(r["written"] for r in forced) == 9


def test_script_pattern_selects_continuous_only(cd11_tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "mseed"
    cd11tomseed.main([str(cd11_tree), str(out), "--pattern", "*_C_*.cd11", "--n-workers", "1"])
    assert len(list(out.rglob("*.mseed"))) == 9


# --------------------------------------------------------------------------- #
# data.convert (format cd11) + `openmeq convert`
# --------------------------------------------------------------------------- #


def _write_config(tmp_path: Path, convert: dict | None) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config" / "example.yaml").read_text())
    raw["project"]["output_dir"] = str(tmp_path / "run")
    raw["parallel"]["n_workers"] = 1
    raw["data"]["waveform_dirs"] = [str(tmp_path / "mseed")]
    raw["data"]["convert"] = convert
    raw["stations"] = {"stationxml": None, "stationxml_dir": None, "csv": None}
    raw["nlloc"]["velocity_model"]["file"] = str(tmp_path / "vel.txt")
    raw["hypodd"]["velocity_model"]["file"] = str(tmp_path / "vel.txt")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    return cfg_path


def _block(in_dir: Path) -> dict:
    return {"format": "cd11", "input_dir": str(in_dir), "output_dir": "./mseed", "network": "MB"}


def test_config_cd11_defaults_pattern_and_no_stationxml(cd11_tree: Path, tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, _block(cd11_tree)))
    assert cfg.data.convert.pattern == "*.cd11"
    assert cfg.data.convert.stationxml is False
    assert cfg.stations.stationxml is None
    suds_block = _block(cd11_tree) | {"format": "suds"}
    assert load_config(_write_config(tmp_path, suds_block)).data.convert.pattern == "*.suds"
    with pytest.raises(ConfigError, match="stationxml"):
        load_config(_write_config(tmp_path, _block(cd11_tree) | {"stationxml": True}))


def test_convert_raw_data_cd11_runs_once_then_skips(cd11_tree: Path, tmp_path: Path) -> None:
    cfg = load_config(_write_config(tmp_path, _block(cd11_tree)))
    paths = RunPaths(cfg)
    paths.ensure()
    assert convert_raw_data(cfg, paths) is True
    assert (tmp_path / "mseed/2021/MB01/EHZ.D/MB.MB01..EHZ.D.2021.044.mseed").is_file()
    assert convert_raw_data(cfg, paths) is False


def test_cli_convert(cd11_tree: Path, tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, _block(cd11_tree))
    runner = CliRunner()
    res = runner.invoke(cli_main, ["convert", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "converted" in res.output
    assert len(list((tmp_path / "mseed").rglob("*.mseed"))) == 9
    res = runner.invoke(cli_main, ["convert", "-c", str(cfg_path)])
    assert res.exit_code == 0 and "up to date" in res.output

    no_block = _write_config(tmp_path, None)
    res = runner.invoke(cli_main, ["convert", "-c", str(no_block)])
    assert res.exit_code != 0 and "no data.convert" in res.output
