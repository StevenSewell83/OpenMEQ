"""Tests for ``openmeq.suds``, ``scripts/suds2mseed.py`` and the ``data.convert``
run-start hook (``openmeq.ingest``) (OpenMEQ-ekd / OpenMEQ-18a / OpenMEQ-6zu).

Writes tiny hand-built extended PC-SUDS files (the MacBan variant: standard
STATIONCOMP/DESCRIPTRACE plus 32 bytes of network/station/channel names) so
no real data is needed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from obspy import UTCDateTime, read, read_inventory

from openmeq import suds
from openmeq.config import ConfigError, ParallelConfig, load_config
from openmeq.ingest import convert_raw_data
from openmeq.paths import RunPaths

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import suds2mseed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SERIAL = ParallelConfig(n_workers=1)

RATE = 100.0


def _tag(sid: int, struct_bytes: bytes, data: bytes = b"") -> bytes:
    return (
        struct.pack("<ccHii", b"S", b"6", sid, len(struct_bytes), len(data)) + struct_bytes + data
    )


def _ext(sta: str, cha: str) -> bytes:
    return b"NET".ljust(8, b"\0") + sta.encode().ljust(16, b"\0") + cha.encode().ljust(8, b"\0")


def _stationcomp(sta: str, cha: str, lat: float, lon: float, elev: float) -> bytes:
    s = bytearray(76)
    s[0:4] = b"NET\0"
    struct.pack_into("<ddf", s, 16, lat, lon, elev)
    return _tag(5, bytes(s) + _ext(sta, cha))


def _descriptrace(sta: str, cha: str, begin: UTCDateTime, data: np.ndarray) -> bytes:
    s = bytearray(64)
    s[0:4] = b"NET\0"
    struct.pack_into("<dhcchhif", s, 12, begin.timestamp, -32767, b"l", b"_", 0, 0, len(data), RATE)
    return _tag(7, bytes(s) + _ext(sta, cha), data.astype("<i4").tobytes())


def _write_event(
    path: Path, begin: UTCDateTime, signal: np.ndarray, start_idx: int, n: int, **kw
) -> None:
    """Event file with MB-01 chz/che holding ``signal[start_idx:start_idx+n]``."""
    lat, lon = kw.get("coords", (14.1, 121.2))
    body = _tag(32, bytes(46), bytes(60))  # a CHANSET struct, to be skipped
    for cha, scale in (("chz", 1), ("che", -1)):
        body += _stationcomp("MB-01", cha, lat, lon, 350.0)
        trace = _descriptrace("MB-01", cha, begin, scale * signal[start_idx : start_idx + n])
        # As in the real corrupt files: a garbage block in place of one whole trace.
        body += b"\xff\xfe" * 500 if kw.get("garbage") and cha == "chz" else trace
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _name(t: UTCDateTime) -> str:
    return t.strftime("%Y%m%d_%H%M%S") + "_NET.suds"


@pytest.fixture
def suds_tree(tmp_path: Path) -> tuple[Path, np.ndarray, UTCDateTime]:
    """Two overlapping events on 2018-01-05 and one crossing into 2018-01-06."""
    rng = np.random.default_rng(0)
    signal = rng.integers(-1000, 1000, 200_000)
    t_a = UTCDateTime(2018, 1, 5, 10, 0, 0)
    src = tmp_path / "suds" / "local, 2018-01"
    # Event A: 90 s; event B starts 30 s later and overlaps 60 s with identical samples.
    _write_event(src / _name(t_a), t_a, signal, 0, 9000)
    _write_event(src / _name(t_a + 30), t_a + 30, signal, 3000, 9000, coords=(0.0, 0.0))
    # Event C: starts 30 s before midnight, lasts 90 s.
    t_c = UTCDateTime(2018, 1, 5, 23, 59, 30)
    _write_event(src / _name(t_c), t_c, signal, 0, 9000, coords=(14.1001, 121.2001))
    return tmp_path / "suds", signal, t_a


def test_read_suds_parses_traces_coords_and_skips_other_structs(tmp_path: Path) -> None:
    sig = np.arange(500)
    t = UTCDateTime(2019, 4, 22, 9, 11)
    _write_event(tmp_path / _name(t), t, sig, 0, 500)
    traces, coords, warnings = suds.read_suds(tmp_path / _name(t))
    assert warnings == []
    assert [(tr["sta"], tr["cha"]) for tr in traces] == [("MB-01", "chz"), ("MB-01", "che")]
    assert UTCDateTime(traces[0]["begin"]) == t
    assert traces[0]["rate"] == RATE
    np.testing.assert_array_equal(traces[0]["data"], sig)
    np.testing.assert_array_equal(traces[1]["data"], -sig)
    assert coords["MB-01"] == (14.1, 121.2, 350.0)


def test_read_suds_resyncs_past_a_corrupt_block(tmp_path: Path) -> None:
    sig = np.arange(500)
    t = UTCDateTime(2018, 3, 19, 7, 20, 30)
    _write_event(tmp_path / _name(t), t, sig, 0, 500, garbage=True)
    traces, _, warnings = suds.read_suds(tmp_path / _name(t))
    assert len(warnings) == 1 and "bad struct tag" in warnings[0]
    assert [tr["cha"] for tr in traces] == ["che"]
    np.testing.assert_array_equal(traces[0]["data"], -sig)


def test_codes() -> None:
    codes = suds.Codes()
    assert suds.station_code("MB-02") == "MB02"
    assert suds.channel_code("chz", codes) == "EHZ"
    assert suds.channel_code("MB1v", codes) == "EHZ"
    assert suds.channel_code("che", suds.Codes(band="H")) == "HHE"


def test_convert_writes_day_files_merged_and_split_at_midnight(suds_tree, tmp_path: Path) -> None:
    in_dir, signal, t_a = suds_tree
    out = tmp_path / "mseed"
    suds.convert(in_dir, out, SERIAL)

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.mseed"))
    assert files == [
        "2018/MB01/EHE.D/MB.MB01..EHE.D.2018.005.mseed",
        "2018/MB01/EHE.D/MB.MB01..EHE.D.2018.006.mseed",
        "2018/MB01/EHZ.D/MB.MB01..EHZ.D.2018.005.mseed",
        "2018/MB01/EHZ.D/MB.MB01..EHZ.D.2018.006.mseed",
    ]
    day5 = read(str(out / files[2])).sort()
    # A and B merge into one 120 s segment (identical overlap); C's pre-midnight 30 s is separate.
    assert len(day5) == 2
    assert day5[0].stats.starttime == t_a
    np.testing.assert_array_equal(day5[0].data, signal[:12000])
    assert day5[1].stats.npts == 3000
    assert day5[1].stats.endtime < UTCDateTime(2018, 1, 6)
    day6 = read(str(out / files[3]))
    assert len(day6) == 1 and day6[0].stats.starttime == UTCDateTime(2018, 1, 6)
    np.testing.assert_array_equal(day6[0].data, signal[3000:9000])
    np.testing.assert_array_equal(read(str(out / files[0])).sort()[0].data, -signal[:12000])

    inv = read_inventory(str(out / "stations.xml"))
    sta = inv[0][0]
    assert (inv[0].code, sta.code) == ("MB", "MB01")
    assert [c.code for c in sta] == ["EHE", "EHZ"]
    # Median of the two valid fixes; the (0, 0) fix is dropped.
    assert sta.latitude == pytest.approx(14.10005) and sta.longitude == pytest.approx(121.20005)
    assert sta.start_date == t_a and sta.end_date >= UTCDateTime(2018, 1, 6, 0, 0, 59)


def test_convert_never_overwrites_without_flag_and_dry_run_writes_nothing(
    suds_tree, tmp_path: Path
) -> None:
    in_dir, _, _ = suds_tree
    dry = tmp_path / "dry"
    results = suds.convert(in_dir, dry, SERIAL, dry_run=True)
    assert sum(r["written"] for r in results) == 4
    assert not dry.exists()

    out = tmp_path / "mseed"
    existing = out / "2018/MB01/EHZ.D/MB.MB01..EHZ.D.2018.005.mseed"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep me")
    results = suds.convert(in_dir, out, SERIAL)
    assert existing.read_bytes() == b"keep me"
    assert sum(r["skipped"] for r in results) == 1
    suds.convert(in_dir, out, SERIAL, overwrite=True)
    assert len(read(str(existing))) == 2


def test_main_cli_with_custom_codes(suds_tree, tmp_path: Path) -> None:
    in_dir, _, _ = suds_tree
    out = tmp_path / "mseed"
    rc = suds2mseed.main(
        [str(in_dir), str(out), "--network", "XX", "--location", "00", "--band", "H",
         "--n-workers", "2", "--no-stationxml"]
    )  # fmt: skip
    assert rc == 0
    assert (out / "2018/MB01/HHZ.D/XX.MB01.00.HHZ.D.2018.005.mseed").exists()
    assert not (out / "stations.xml").exists()
    with pytest.raises(SystemExit):
        suds2mseed.main([str(in_dir), str(out), "--network", "TOO-LONG"])


# --------------------------------------------------------------------------- #
# data.convert config + run-start hook
# --------------------------------------------------------------------------- #


def _write_config(tmp_path: Path, convert: dict | None, **data: object) -> Path:
    """example.yaml with ``data`` pointed at tmp_path and ``data.convert`` set."""
    raw = yaml.safe_load((REPO_ROOT / "config" / "example.yaml").read_text())
    raw["project"]["output_dir"] = str(tmp_path / "run")
    raw["parallel"]["n_workers"] = 1
    raw["data"].update(waveform_dirs=[str(tmp_path / "mseed")], **data)
    raw["data"]["convert"] = convert
    raw["stations"] = {"stationxml": None, "stationxml_dir": None, "csv": None}
    raw["nlloc"]["velocity_model"]["file"] = str(tmp_path / "vel.txt")
    raw["hypodd"]["velocity_model"]["file"] = str(tmp_path / "vel.txt")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    return cfg_path


def _convert_block(in_dir: Path, out: str = "./mseed/macban", **kw: object) -> dict:
    return {"format": "suds", "input_dir": str(in_dir), "output_dir": out, "network": "MB", **kw}


def test_config_convert_resolves_paths_and_defaults_stationxml(suds_tree, tmp_path: Path) -> None:
    in_dir, _, _ = suds_tree
    cfg = load_config(_write_config(tmp_path, _convert_block(in_dir)))
    conv = cfg.data.convert
    assert conv.output_dir == tmp_path / "mseed" / "macban"
    assert (conv.band, conv.instrument, conv.location, conv.overwrite) == ("E", "H", "", False)
    assert cfg.stations.stationxml == conv.output_dir / "stations.xml"
    assert load_config(_write_config(tmp_path, None)).data.convert is None


@pytest.mark.parametrize(
    "block, data, message",
    [
        ({"output_dir": "./elsewhere"}, {}, "waveform_dirs"),
        ({}, {"file_pattern": "*.D.*[!d]"}, "file_pattern"),
        ({"network": "TOO"}, {}, "network"),
        ({"format": "seisan"}, {}, "format"),
        ({"band": "e"}, {}, "band"),
    ],
)
def test_config_convert_rejects_bad_settings(
    suds_tree, tmp_path: Path, block: dict, data: dict, message: str
) -> None:
    in_dir, _, _ = suds_tree
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, _convert_block(in_dir) | block, **data))


def test_convert_raw_data_runs_once_then_skips_until_inputs_change(
    suds_tree, tmp_path: Path
) -> None:
    in_dir, _, _ = suds_tree
    cfg = load_config(_write_config(tmp_path, _convert_block(in_dir)))
    paths = RunPaths(cfg)
    paths.ensure()
    out = cfg.data.convert.output_dir

    assert convert_raw_data(cfg, paths) is True
    assert (out / "2018/MB01/EHZ.D/MB.MB01..EHZ.D.2018.005.mseed").is_file()
    assert (out / "stations.xml").is_file()
    assert convert_raw_data(cfg, paths) is False  # unchanged -> skipped
    assert convert_raw_data(cfg, paths, force=True) is True

    (out / "stations.xml").unlink()  # output gone -> redone
    assert convert_raw_data(cfg, paths) is True

    t = UTCDateTime(2018, 2, 1)
    _write_event(in_dir / _name(t), t, np.arange(1000), 0, 1000)  # new input -> redone
    assert convert_raw_data(cfg, paths) is True
    assert (out / "2018/MB01/EHZ.D/MB.MB01..EHZ.D.2018.032.mseed").is_file()
    assert convert_raw_data(cfg, paths) is False


def test_convert_raw_data_noop_without_config_and_errors_on_missing_input(
    suds_tree, tmp_path: Path
) -> None:
    cfg = load_config(_write_config(tmp_path, None))
    assert convert_raw_data(cfg, RunPaths(cfg)) is False
    cfg = load_config(_write_config(tmp_path, _convert_block(tmp_path / "nope")))
    with pytest.raises(FileNotFoundError, match="input_dir"):
        convert_raw_data(cfg, RunPaths(cfg))


def test_qc_step_converts_before_fingerprinting(
    suds_tree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qc.run calls the conversion first, so its fingerprint sees the new files."""
    import openmeq.qc as qc

    in_dir, _, _ = suds_tree
    cfg = load_config(_write_config(tmp_path, _convert_block(in_dir)))
    paths = RunPaths(cfg)
    paths.ensure()
    seen = {}

    def stop(cfg_):
        seen["files"] = qc._waveform_files(cfg_)
        raise RuntimeError("stop after fingerprint")

    monkeypatch.setattr(qc, "_input_fingerprint", stop)
    with pytest.raises(RuntimeError, match="stop after fingerprint"):
        qc.run(cfg, paths)
    assert len(seen["files"]) == 4
