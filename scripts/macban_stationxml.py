#!/usr/bin/env python
"""MacBan StationXML with instrument response: GS-ONE LF 4.5 Hz geophone + Geotech SMART-24R.

``openmeq.suds`` writes a coordinates-only ``stations.xml``. This adds a
two-stage response to every channel of it and writes a new file.

Stage 1, sensor (M/S -> V): Geospace GS-ONE LF 4.5 Hz geophone (brochure
592-15720-01 Rev. B; vertical and horizontal specs are identical). Two zeros
at 0 and two poles ``-h w0 +- i w0 sqrt(1 - h^2)``, ``w0 = 2 pi 4.5``.
Damping and sensitivity depend on the load ``R`` across the coil
(``--load-ohm``), from the brochure's open-circuit (h0 = 0.42,
S0 = 100.4 V/(m/s)) and 20 kOhm (h = 0.735, 89.4 V/(m/s)) values with coil
resistance Rc = 2450 Ohm: ``h = h0 + K / (Rc + R)``, ``S = S0 R / (Rc + R)``.
The default 2.0 MOhm is the SMART-24 40 Vpp ADC Ver. 06 input (standard from
Nov 2011; Ver. 01 is 1.7 MOhm, Ver. 03 17.2 kOhm; SMART-24 manual 2.1.3),
i.e. h = 0.424, 100.3 V/(m/s). An external shunt resistor, if fitted, is in
parallel with it.

Stage 2, digitizer (V -> COUNTS): a gain-only stage, ``1 / con_mvolts``.
``con_mvolts`` is the PC-SUDS STATIONCOMP "conversion to mvolts" (float at
byte 56; data units ``'d'`` = digital counts, USGS OFR 89-188). The MacBan
files carry a calibrated per-channel value (3.22-3.31 uV/count, the
SMART-24 40 Vpp gain-1 nominal LSB being 3.271096 uV/count) that changes
when a digitizer is moved to another site, so each channel gets one epoch
per run of constant value. The SMART-24 FIR filters (linear phase, unity
passband gain) are not included.

Usage::

    python scripts/macban_stationxml.py SUDS_DIR STATIONS_XML OUT_XML [--load-ohm R] [--plot PNG]
"""

from __future__ import annotations

import argparse
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read_inventory
from obspy.core.inventory import (
    CoefficientsTypeResponseStage,
    Equipment,
    InstrumentSensitivity,
    PolesZerosResponseStage,
    Response,
)

from openmeq.suds import (
    _RESYNC,
    _STATIONCOMP,
    _STATIONCOMP_LEN,
    _TAG,
    Codes,
    _names,
    channel_code,
    filename_time,
    station_code,
)

F0 = 4.5  # Hz
H0, S0, RC = 0.42, 100.4, 2450.0  # open-circuit damping, V/(m/s), coil Ohm
K_DAMP = (0.735 - H0) * (RC + 20e3)  # electrical damping constant from the 20 kOhm spec
NORM_FREQ = 15.0  # Hz, in the flat velocity passband
SENSOR = Equipment(
    type="Geophone",
    description="Geospace GS-ONE LF 4.5 Hz",
    manufacturer="Geospace Technologies",
    model="GS-ONE LF 4.5 Hz",
)
DATALOGGER = Equipment(
    type="Datalogger",
    description="Geotech SMART-24R, 40 Vpp, gain 1",
    manufacturer="Geotech Instruments",
    model="SMART-24R",
)


def read_gains(path: Path) -> dict[tuple[str, str], float]:
    """{(station, channel): con_mvolts} from the STATIONCOMP structs of one SUDS file."""
    b = path.read_bytes()
    out, p = {}, 0
    while p + _TAG.size <= len(b):
        sync, mach, sid, slen, dlen = _TAG.unpack_from(b, p)
        if (
            sync != b"S"
            or mach != b"6"
            or slen < 0
            or dlen < 0
            or p + _TAG.size + slen + dlen > len(b)
        ):
            m = _RESYNC.search(b, p + 1)
            if not m:
                break
            p = m.start()
            continue
        if sid == _STATIONCOMP and slen >= 60:
            s = b[p + _TAG.size : p + _TAG.size + slen]
            _, sta, cha = _names(s, _STATIONCOMP_LEN)
            (mv,) = struct.unpack_from("<f", s, 56)
            if np.isfinite(mv) and mv > 0:
                out[(station_code(sta), channel_code(cha, Codes()))] = float(mv)
        p += _TAG.size + slen + dlen
    return out


def gain_epochs(suds_dir: Path) -> dict[tuple[str, str], list[tuple[UTCDateTime, float]]]:
    """Per channel: [(start time, con_mvolts)] for each run of constant value."""
    obs = defaultdict(list)
    for f in sorted(suds_dir.rglob("*.suds")):
        t = filename_time(f)
        for key, mv in read_gains(f).items():
            obs[key].append((t, mv))
    epochs = {}
    for key, rows in obs.items():
        rows.sort()
        runs = [rows[0]]
        for t, mv in rows[1:]:
            if not np.isclose(mv, runs[-1][1], rtol=1e-6):
                runs.append((t, mv))
        epochs[key] = runs
    return epochs


def loaded_sensor(load_ohm: float) -> tuple[float, float]:
    """(damping, sensitivity V/(m/s)) with ``load_ohm`` across the coil."""
    return H0 + K_DAMP / (RC + load_ohm), S0 * load_ohm / (RC + load_ohm)


def sensor_stage(damping: float, sensitivity: float) -> PolesZerosResponseStage:
    w0 = 2 * np.pi * F0
    wd = w0 * np.sqrt(1 - damping**2)
    poles = [complex(-damping * w0, wd), complex(-damping * w0, -wd)]
    s = 2j * np.pi * NORM_FREQ
    h = s**2 / ((s - poles[0]) * (s - poles[1]))  # -> 1 well above F0
    return PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=sensitivity * abs(h),
        stage_gain_frequency=NORM_FREQ,
        input_units="M/S",
        output_units="V",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=NORM_FREQ,
        normalization_factor=1 / abs(h),
        zeros=[0j, 0j],
        poles=poles,
        description="Geospace GS-ONE LF 4.5 Hz",
    )


def response(damping: float, sensitivity: float, mv_per_count: float, rate: float) -> Response:
    adc = CoefficientsTypeResponseStage(
        stage_sequence_number=2,
        stage_gain=1e3 / mv_per_count,
        stage_gain_frequency=NORM_FREQ,
        input_units="V",
        output_units="COUNTS",
        cf_transfer_function_type="DIGITAL",
        numerator=[],
        denominator=[],
        decimation_input_sample_rate=rate,
        decimation_factor=1,
        decimation_offset=0,
        decimation_delay=0.0,
        decimation_correction=0.0,
        description=f"digitizer, SUDS con_mvolts {mv_per_count:.7g} mV/count",
    )
    r = Response(
        instrument_sensitivity=InstrumentSensitivity(
            value=1.0, frequency=NORM_FREQ, input_units="M/S", output_units="COUNTS"
        ),
        response_stages=[sensor_stage(damping, sensitivity), adc],
    )
    r.recalculate_overall_sensitivity(NORM_FREQ)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("suds_dir", type=Path)
    ap.add_argument("stationxml", type=Path, help="coordinates-only stations.xml from openmeq.suds")
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--load-ohm",
        type=float,
        default=2.0e6,
        help="load across the coil, Ohm (default: SMART-24 ADC Ver. 06)",
    )
    ap.add_argument("--plot", type=Path, help="write a Bode plot of one channel's response")
    a = ap.parse_args()

    damping, sensitivity = loaded_sensor(a.load_ohm)
    epochs = gain_epochs(a.suds_dir)
    inv = read_inventory(str(a.stationxml))
    for net in inv:
        for sta in net:
            new = []
            for cha in sta:
                runs = epochs.get((sta.code, cha.code))
                if not runs:
                    print(f"{sta.code}.{cha.code}: no SUDS gain, left without response")
                    new.append(cha)
                    continue
                for i, (t, mv) in enumerate(runs):
                    c = cha.copy()
                    c.start_date = cha.start_date if i == 0 else t
                    c.end_date = runs[i + 1][0] if i + 1 < len(runs) else cha.end_date
                    c.sensor = SENSOR
                    c.data_logger = DATALOGGER
                    c.response = response(damping, sensitivity, mv, cha.sample_rate)
                    new.append(c)
                    print(
                        f"{sta.code}.{cha.code} {c.start_date.date} .. {c.end_date.date}: "
                        f"{mv * 1e3:.4f} uV/count -> "
                        f"{c.response.instrument_sensitivity.value:.4g} counts/(m/s)"
                    )
            sta.channels = new
    inv.source = "scripts/macban_stationxml.py"
    inv.write(str(a.out), format="STATIONXML", validate=True)
    print(f"wrote {a.out} (load {a.load_ohm:g} Ohm: h = {damping:.3f}, {sensitivity:.1f} V/(m/s))")
    if a.plot:
        cha = inv[0][0][0]
        cha.response.plot(
            0.2, output="VEL", label=f"{inv[0][0].code}.{cha.code}", outfile=str(a.plot)
        )
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
