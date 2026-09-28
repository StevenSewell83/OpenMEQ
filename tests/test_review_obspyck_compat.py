"""Tests for OpenMEQ-c3l.2 (M7.2): ``openmeq.review._obspyck_compat``.

The two ObsPyck 0.8.2 bugfix monkeypatches (``docs/obspyck_notes.md`` §6);
the reference implementation is ``scripts/obspyck_spike_smoke.py``'s
``apply_bugfix_patches``. Full end-to-end verification against a real,
launched ObsPyck window (as that spike script does) is M7.3's job (the
launcher module that actually constructs ``ObsPyck(...)``, per this
module's own docstring) -- these tests cover the monkeypatch logic itself
in isolation: the coordinate math against an independently-written
reimplementation of NonLinLoc's own ``TRANS SIMPLE`` forward transform, and
the ``PROGRAMS['nlloc']['Call']`` wrapping/idempotency against a fake
``Call``, without needing a display or a real NLLoc run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from openmeq.review._obspyck_compat import (
    _strip_stray_gt_tokens,
    make_gk2lonlat,
    patch_gk2lonlat,
    patch_nlloc_call,
)

#: Independently-written copy of NonLinLoc's own ``TRANS SIMPLE`` forward
#: transform (``external/NonLinLoc/src/GridLib.c``'s ``latlon2rect``,
#: ``rotAngle=0``: ``map_cosang=1``, ``map_sinang=0``), used to check
#: :func:`make_gk2lonlat`'s inverse against a *separate* implementation
#: rather than just self-consistency with the module under test.
_AVG_ERAD_KM = 6371.0087714  # NonLinLoc geo.h AVG_ERAD ("GMT Sphere")
_DEG2KM = math.pi * _AVG_ERAD_KM / 180.0  # geo.h DEG2KM == GridLib.c's c111


def _nlloc_forward(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    dlon = lon - lon0
    if dlon > 180.0:
        dlon -= 360.0
    elif dlon < -180.0:
        dlon += 360.0
    x = dlon * _DEG2KM * math.cos(math.radians(lat))
    y = (lat - lat0) * _DEG2KM
    return x, y


@pytest.mark.parametrize(
    "lat0,lon0,lat,lon",
    [
        # This project's own reference point and a real NLLoc output
        # coordinate from the OpenMEQ-c3l.1 spike (docs/obspyck_notes.md §6a).
        (-38.0, 176.0, -38.062109, 175.956096),
        (-38.0, 176.0, -38.0, 176.0),
        (52.5, 13.4, 52.41, 13.55),
        (0.0, 0.0, 0.02, -0.03),
        (-89.0, 179.5, -88.99, -179.8),  # near a longitude-wrap boundary
    ],
)
def test_gk2lonlat_inverts_nlloc_trans_simple(lat0, lon0, lat, lon):
    x, y = _nlloc_forward(lat0, lon0, lat, lon)
    gk2lonlat = make_gk2lonlat(lat0, lon0)
    got_lon, got_lat = gk2lonlat(x, y)
    assert got_lat == pytest.approx(lat, abs=1e-9)
    assert got_lon == pytest.approx(lon, abs=1e-9)


def test_gk2lonlat_rejects_bavaria_default():
    """The bug being fixed: the old hard-coded EPSG:31468 (Bavaria) inverse
    would not even be a lon/lat pair in the right numeric range for a
    project far from Germany -- confirm the replacement, parameterized on
    this project's own origin, gives a physically sane result instead."""
    gk2lonlat = make_gk2lonlat(-38.0, 176.0)
    lon, lat = gk2lonlat(-3.875, -6.875)
    assert -180.0 <= lon <= 180.0
    assert -90.0 <= lat <= 90.0
    assert lon == pytest.approx(176.0, abs=1.0)
    assert lat == pytest.approx(-38.0, abs=1.0)


def test_gk2lonlat_accepts_numpy_arrays():
    gk2lonlat = make_gk2lonlat(-38.0, 176.0)
    lon, lat = gk2lonlat(np.array([0.0, -3.875]), np.array([0.0, -6.875]))
    assert isinstance(lon, np.ndarray)
    assert isinstance(lat, np.ndarray)
    assert lat[0] == pytest.approx(-38.0)
    assert lon[0] == pytest.approx(176.0)


def test_gk2lonlat_m_to_km_flag():
    gk2lonlat = make_gk2lonlat(-38.0, 176.0)
    lon_km, lat_km = gk2lonlat(10.0, 20.0, m_to_km=True)
    lon_m, lat_m = gk2lonlat(10_000.0, 20_000.0, m_to_km=False)
    assert lon_m == pytest.approx(lon_km)
    assert lat_m == pytest.approx(lat_km)


def test_patch_gk2lonlat_installs_on_both_modules():
    pytest.importorskip("obspyck")
    import obspyck.obspyck as op
    import obspyck.util as ou

    saved_util, saved_op = ou.gk2lonlat, op.gk2lonlat
    try:
        replacement = patch_gk2lonlat(-38.0, 176.0)
        assert ou.gk2lonlat is replacement
        assert op.gk2lonlat is replacement
        lon, lat = op.gk2lonlat(0.0, 0.0)
        assert (lon, lat) == pytest.approx((176.0, -38.0))
    finally:
        ou.gk2lonlat, op.gk2lonlat = saved_util, saved_op


def test_strip_stray_gt_tokens_only_touches_phase_block():
    text = (
        "NLLOC ...\n"
        "PHASE ID  Ins Cmp On Pha  FM Date  HrMn  Sec  Err ErrMag  "
        "Coda Amp Per PriorWt  >   TTpred    Res       Weight\n"
        "ST1 ?    ?   ?  P    ?  20260101 0000 10.10 GAU  0.05 -1 -1 -1 1.0000 "
        ">    1.8940   -0.0068    1.0440\n"
        "END_PHASE\n"
        "GEOGRAPHIC  OT ... Lat -38.062109 > Long 175.956096 Depth 6.320312\n"
    )
    fixed = _strip_stray_gt_tokens(text)
    lines = fixed.splitlines()

    phase_data_line = lines[2]
    assert ">" not in phase_data_line
    # every field survives, just the stray token removed.
    assert "1.0000" in phase_data_line
    assert "1.8940" in phase_data_line

    # Lines outside the PHASE ... END_PHASE block are untouched verbatim,
    # even one that happens to contain ">".
    assert lines[4] == "GEOGRAPHIC  OT ... Lat -38.062109 > Long 175.956096 Depth 6.320312"
    assert "PHASE ID" in lines[1]


def test_patch_nlloc_call_wraps_strips_and_is_idempotent(tmp_path):
    pytest.importorskip("obspyck")
    import obspyck.util as ou

    summary = tmp_path / "nlloc.hyp"
    summary.write_text(
        "PHASE ID  ... PriorWt  >   TTpred    Res\n"
        "ST1 ... 1.0000 >    1.8940   -0.0068\n"
        "END_PHASE\n"
    )

    calls = []

    def fake_original_call(prog_dict, controlfilename):
        calls.append(controlfilename)
        return ("msg", "err", 0)

    prog_dict = ou.PROGRAMS["nlloc"]
    saved_call = prog_dict.get("Call")
    saved_files = prog_dict.get("files")
    try:
        prog_dict["Call"] = fake_original_call
        prog_dict["files"] = {"summary": str(summary)}

        patch_nlloc_call()
        wrapped = prog_dict["Call"]
        assert wrapped is not fake_original_call

        result = wrapped(prog_dict, "control.in")
        assert result == ("msg", "err", 0)
        assert calls == ["control.in"]
        # Only the PHASE *data* line is stripped -- the header line (which
        # legitimately has its own literal ">" column heading in real NLLoc
        # 7.1.05 output, see docs/obspyck_notes.md §6b) is left alone.
        data_line = summary.read_text().splitlines()[1]
        assert ">" not in data_line
        assert "1.8940" in data_line

        # Re-applying does not double-wrap (checked via identity).
        patch_nlloc_call()
        assert prog_dict["Call"] is wrapped
    finally:
        if saved_call is not None:
            prog_dict["Call"] = saved_call
        else:
            prog_dict.pop("Call", None)
        if saved_files is not None:
            prog_dict["files"] = saved_files
        else:
            prog_dict.pop("files", None)
