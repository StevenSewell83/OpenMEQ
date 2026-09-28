"""Deterministic synthetic test dataset generator (M0.8, ARCHITECTURE.md §8).

``generate(outdir, seed=DEFAULT_SEED) -> dict`` builds, under ``outdir``:

- ``inventory/stations.xml`` / ``inventory/stations.csv`` -- 8 stations (6
  surface ``HH?``, 2 borehole ``DP?``), each with a *flat* instrument
  response (see "Flat response" below).
- ``waveforms/<YYYY-MM-DD>/<NET>.<STA>.<LOC>.<CHA>.mseed`` -- 2 days of
  continuous 200 Hz data, one miniSEED file per channel per day
  (STEIM2-encoded int32 samples), with one data gap, one spiky channel and
  one flat-lined channel (see "Anomalies" below).
- ``models/vel1d.txt`` -- the 1D layer file in the ARCHITECTURE.md §6a
  format, parsed by :func:`openmeq.velocity.read_layers`.
- ``config.yaml`` -- an ``openmeq.config.load_config``-valid example config
  pointing at all of the above.
- ``truth.json`` -- a JSON dump of the returned ``truth`` structure (times
  as ISO-8601 ``...Z`` strings), for the standalone CLI / manual use.

It returns ``truth`` (see "Truth structure" below); the caller (normally the
``synthetic_project`` pytest fixture, ``tests/conftest.py``) pairs it with
``outdir / "config.yaml"``.

Determinism
------------
Every random draw goes through a single ``numpy.random.default_rng(seed)``
(default ``seed=DEFAULT_SEED``), consumed in a fixed order, so the same seed
always reproduces the same station geometry perturbations, event catalogue,
noise and anomaly placement, and therefore bit-identical output files (see
``tests/test_synthetic.py::test_deterministic``). Station geometry and
channel/network naming are plain module-level constants, not randomly
drawn, so they never vary between runs regardless of seed.

Coordinate system
------------------
Stations and events are placed on a local flat-Earth tangent plane centred
on ``(LAT0, LON0)`` (chosen to match ``config/example.yaml``'s NLLoc grid
origin comment): ``lat = LAT0 + y_km / KM_PER_DEG_LAT``,
``lon = LON0 + x_km / (KM_PER_DEG_LAT * cos(radians(LAT0)))``. This is a
fine approximation at the ~20 km scale of this network and is not meant to
be geodesy-grade.

Travel-time method (straight-ray secant through the 1D layered model)
----------------------------------------------------------------------
For a source at depth ``z_src`` (km, positive down from sea level per
ARCHITECTURE.md §6a) and a receiver at *effective* depth ``z_rcv`` (see
below) separated by horizontal distance ``x`` (km), the straight-line
(chord) path length is ``D = sqrt(x**2 + (z_rcv - z_src)**2)`` and its
constant angle from vertical satisfies ``cos(theta) = |z_rcv - z_src| / D``.
Because the ray is a straight line (not bent at layer interfaces -- no
Snell's law refraction), ``cos(theta)`` is the same in every layer it
crosses, so the path length within layer ``i`` is ``dz_i / cos(theta)``,
where ``dz_i`` is the vertical extent of that layer between ``z_src`` and
``z_rcv``. Travel time is the sum, over every layer the chord crosses, of
(layer path length) / (layer velocity for the requested phase):

    t = (1 / cos(theta)) * sum_i(dz_i / v_i)

A receiver above the shallowest tabulated layer top (e.g. a mountaintop
surface station) or a source below the deepest tabulated layer top both use
the nearest layer's velocity, extending it indefinitely, per ARCHITECTURE.md
§6a ("the first layer applies from the top of the model to the next
depth" -- read here as also extending upward past the first entry, and the
last layer extending downward past its entry). A source and receiver at the
same depth (``cos(theta)`` undefined) fall back to ``D / v`` at that shared
depth.

**Accuracy**: a true ray-tracer (e.g. NLLoc's ``Grid2Time``, which bends the
ray at each interface to find the *fastest* first-arrival path per Fermat's
principle) always finds a travel time less than or equal to this straight
chord's. For this network's geometry (offsets and depths both a few km, a
three-layer model with modest velocity contrasts, 3.5/4.5/5.8 km/s P and
2.0/2.6/3.35 km/s S), the straight-ray overestimate is typically a few
percent of the total travel time -- on the order of a few tens of
milliseconds at these distances -- growing for long, shallow-incidence
raypaths that cross a layer boundary far from vertical. Downstream location
recovery tests (e.g. comparing NLLoc-derived origins against
``truth["events"][i]["origin"]``) should budget residual/error tolerances
of this order (documented starting point: 20-50 ms per P/S residual, a few
hundred m to ~1 km horizontal/depth location error) rather than expecting
bit-for-bit agreement with this generator's own travel times.

**Station "effective depth"**: per ARCHITECTURE.md §6, a station's
elevation for location purposes is ``elevation_m - depth_m`` (boreholes have
``depth_m > 0``); converted to the model's depth-below-sea-level convention,
``z_rcv_km = -(elevation_m - depth_m) / 1000``.

Flat response
-------------
Every channel's StationXML response is a single
``PolesZerosResponseStage`` with **no poles or zeros** (so its transfer
function is the real constant ``stage_gain`` at every frequency -- a true
flat velocity response, not merely flat *within a passband*), input units
``M/S`` (ground velocity), output units ``COUNTS``, and
``stage_gain == instrument_sensitivity.value == SENSITIVITY_CPS``
(:data:`SENSITIVITY_CPS`, counts per m/s). Removing this response is
therefore exactly ``ground_velocity_m_s = raw_counts / SENSITIVITY_CPS`` at
every frequency, with no filtering, phase, or water-level considerations
-- by design, so a future magnitude/response-removal implementation's
correctness on *this* fixture reduces to "did it divide by the sensitivity",
not "did it correctly evaluate a non-trivial pole-zero response".

Wood-Anderson ML calibration (exact, closed-form)
--------------------------------------------------
Per ARCHITECTURE.md §8 / config/example.yaml's ``magnitude.ml`` block, a
station magnitude is the Hutton & Boore (1987) relation

    ML = log10(A) + a*log10(R/100) + b*(R - 100) + c      (A in mm on a
                                                             simulated Wood-
                                                             Anderson, R the
                                                             hypocentral
                                                             distance in km)

with ``(a, b, c)`` read from :class:`openmeq.config.HuttonBooreConfig`'s
defaults (``1.110, 0.00189, 3.0``) so this module can never drift out of
sync with the config schema those later tasks will actually use. This
generator works the relation backwards: given a target ``truth.ml`` and a
station's hypocentral distance ``R``, it solves for the target
zero-to-peak Wood-Anderson amplitude ``A_mm``, then solves for the raw
*counts* amplitude that produces exactly that ``A_mm`` reading, using the
same closed-form relationship ``obspy.signal.invsim`` itself provides for
turning a "peak amplitude + timespan" pick into a Wood-Anderson amplitude
(:func:`obspy.signal.invsim.estimate_wood_anderson_amplitude`):

    A_mm = (counts_peak_to_peak / 2) / (|H_inst(f)| * S_inst)
           * (|H_WA(f)| * S_WA) * 1000

``H_inst`` is our flat instrument response (``|H_inst(f)| == 1`` at every
frequency, by construction above, so this term is just ``S_inst ==
SENSITIVITY_CPS``); ``H_WA``/``S_WA`` are ``obspy.signal.invsim.WOODANDERSON``
(poles/zeros/gain) and :class:`openmeq.config.MlConfig`'s default
``wa_sensitivity`` (2080 -- the IASPEI/Uhrhammer & Collins value, which
happens to be exactly obspy's own ``WOODANDERSON['sensitivity']`` constant);
``f`` is fixed at :data:`S_DOM_FREQ_HZ` (the injected S-pulse's dominant
frequency, see below), matching the single-frequency approximation this
formula makes for a "peak-to-peak + timespan" style pick.

The injected S-arrival pulse (see "Pulse shape" below) is then scaled so its
actual, numerically-measured zero-to-peak counts amplitude equals the value
this inversion demands, on whichever horizontal channel is nominated
"primary" for that event/station (see below) -- so *that* channel's
amplitude reproduces ``truth.ml`` through this formula essentially exactly.

**Expected fidelity / tolerance for downstream tests**: a future magnitude
step will most plausibly measure amplitude by fully simulating the
Wood-Anderson response in the time domain over a measurement window (per
``magnitude.ml.wood_anderson: true``), not via the single-frequency
pick-based formula above. Because ``H_WA`` is *not* flat (it has a real
resonance near 1.25 Hz), a full time-domain simulation of the injected
pulse -- which has most, but not all, of its energy at ``S_DOM_FREQ_HZ`` --
will recover an amplitude close to, but not bit-identical to, the
pick-based value used to set it up. In addition, only one of a station's
two horizontal channels is the exact "primary" one per event (see below);
the other, and the two borehole stations' unknown-orientation channels,
carry a documented, deliberately-reduced fraction. The per-station ML
spread this produces is expected to average out via
``magnitude.statistic: median`` across the network. A reasonable starting
tolerance for a recovered **event** ML against ``truth["events"][i]["ml"]``
is therefore roughly **+/- 0.1-0.15 magnitude units**, not exact equality.

Pulse shape
-----------
Both P and S arrivals are injected as a causal, exponentially-decaying
tapered sinusoid ("ringing" pulse, a simple stand-in for a
Ricker/Brune-like source pulse -- sharp onset, decaying coda), each
normalized so its actual numerical zero-to-peak amplitude equals the target
exactly (see :func:`_pulse`):

    w(t) = sin(2*pi*f*t) * exp(-t / tau),   0 <= t < duration
    tau = (PULSE_CYCLES / 2) / f,           duration = PULSE_CYCLES / f

The pulse's first sample (``t=0``) is placed exactly at the computed P/S
arrival time (the onset convention used throughout ``truth["events"]``).
P uses :data:`P_DOM_FREQ_HZ`, S uses :data:`S_DOM_FREQ_HZ` (S deliberately
lower, both to be more Brune-like -- larger, lower-frequency S relative to P
-- and because that is the frequency the Wood-Anderson calibration above is
built around). This is a deliberate simplification: real source spectra
have a corner frequency that scales with (inverse) seismic moment, but this
fixture uses one fixed pulse shape per phase for every event regardless of
``ml``, since only the *amplitude* needs to track magnitude for this
fixture's purpose (functional recovery testing, not a source-physics
benchmark).

S is larger than P, and larger on the horizontals than the vertical, via
the fixed ratios :data:`P_TO_S_Z_RATIO`, :data:`P_HORIZ_RATIO` and
:data:`S_Z_RATIO` applied to the calibrated horizontal S amplitude.

Anomalies (``truth["known_issues"]``)
--------------------------------------
Exactly one gap, one spiky channel and one flat-lined channel, at fixed,
documented (seed_id, time span) locations (see :data:`GAP_SEED_ID` /
:data:`SPIKE_SEED_ID` / :data:`FLAT_SEED_ID` and the ``_ANOMALY_*`` window
constants) -- not randomly placed, so QC tests can assert exact recovery of
*those* anomalies:

- **Gap**: the channel's data is genuinely missing for the window (the
  miniSEED file for that day has two Trace segments with a real time gap
  between them, not zeros).
- **Spikes**: within the window, several hundred single-sample impulsive
  outliers are added, each tens to a hundred times the background noise
  amplitude -- comfortably past ``qc.spike_threshold_mad`` (default 10).
- **Flat**: within the window, the channel's samples are overwritten with a
  single constant value (zero variance), for far longer than
  ``qc.flat_run_min_samples`` (default 50 samples = 0.25 s at 200 Hz; the
  window here is a full hour).

Truth structure
----------------
``generate()`` returns a dict:

- ``truth["seed"]``: the seed used.
- ``truth["events"]``: a list of per-event dicts (ordered by origin time):
  ``event_id_hint`` (``openmeq.catalog.event_id_from_time`` of the true
  origin time -- a *hint*: a real detection step assigns its own event id
  from its own trigger time, which need not exactly equal the true origin
  time, so downstream tests should match by nearest time, not assume an
  exact id match), ``origin`` (``{"time": UTCDateTime, "lat", "lon",
  "depth_km"}``), ``ml``, and ``arrivals``.
- ``arrivals`` is keyed by **station** (``"NET.STA"``, e.g. ``"XX.STA01"``),
  not by full channel-level SEED id: ``{"NET.STA": {"P": UTCDateTime, "S":
  UTCDateTime}}``. ARCHITECTURE.md §8 writes this generically as
  ``{seed_id: {"P": t, "S": t}}``; the bd issue for this task is explicit
  that these are "per-station true P/S arrival times", and picking
  convention (ARCHITECTURE.md §5: P on the Z channel, S on the first
  available horizontal) means a single arrival time is a property of the
  *station* (instrument), not of one specific channel -- so this module
  uses ``NET.STA`` as the key. This is documented here precisely because
  ``seed_id`` means the full ``NET.STA.LOC.CHA`` form everywhere else in
  ARCHITECTURE.md (including in ``truth["known_issues"]``, below).
- ``truth["known_issues"]``: a list of ``{"seed_id": "NET.STA.LOC.CHA",
  "kind": "gap"|"spikes"|"flat", "start": UTCDateTime, "end": UTCDateTime}``
  -- here ``seed_id`` *is* the full channel-level SEED id, per
  ARCHITECTURE.md's usual meaning.
- ``truth["stations"]``: convenience list of
  ``{"net", "sta", "loc", "lat", "lon", "elev_m", "depth_m"}`` (the same
  content as ``inventory/stations.csv``, for tests that would rather not
  re-parse the CSV).

All times are ``obspy.UTCDateTime`` (ARCHITECTURE.md §1: UTC
``UTCDateTime`` in code). ``truth.json`` on disk instead uses ISO-8601
``...Z`` strings (ARCHITECTURE.md §1: in files), for the standalone CLI
path.

Usage pattern / isolation
--------------------------
This is also used by the session-scoped ``synthetic_project`` pytest
fixture (``tests/conftest.py``), which generates this dataset **once** per
test session. Any test that runs a pipeline step against it (writing into
``cfg.project.output_dir`` or otherwise mutating files under the project
directory) must first copy the whole project directory to its own
per-test ``tmp_path`` and load a config pointed at the copy -- never write
into the shared session fixture's directory, or other tests in the same
session will see corrupted/partial state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import (
    Channel,
    InstrumentSensitivity,
    Inventory,
    Network,
    PolesZerosResponseStage,
    Response,
    Site,
    Station,
)
from obspy.signal.invsim import WOODANDERSON, paz_2_amplitude_value_of_freq_resp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow running as a bare script

from openmeq.catalog import event_id_from_time  # noqa: E402
from openmeq.config import HuttonBooreConfig, MlConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed constants (deterministic by construction; not drawn from the RNG)
# ---------------------------------------------------------------------------

#: Default seed. Every random draw in :func:`generate` uses
#: ``numpy.random.default_rng(seed)`` in a fixed order.
DEFAULT_SEED = 20250101

NETWORK = "XX"
LOCATION = "00"
SAMPLE_RATE_HZ = 200.0
N_DAYS = 2
DAY0 = UTCDateTime(2025, 1, 1)

#: Local tangent-plane origin (matches config/example.yaml's NLLoc grid
#: origin comment, "[lat, lon]").
LAT0 = 37.80
LON0 = -122.20
KM_PER_DEG_LAT = 111.19


def _km_per_deg_lon(lat0: float = LAT0) -> float:
    return KM_PER_DEG_LAT * math.cos(math.radians(lat0))


#: Station geometry: (code, kind, x_km east, y_km north, elev_m, depth_m,
#: channel codes). Aperture (max station-station distance) is ~14.9 km.
#: Fixed constants -- not randomly drawn -- so the network layout never
#: varies with ``seed``.
STATIONS = [
    ("STA01", "surface", -6.5, -3.0, 615.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("STA02", "surface", 6.0, -3.8, 580.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("STA03", "surface", -5.8, 5.2, 705.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("STA04", "surface", 5.5, 5.8, 655.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("STA05", "surface", 0.2, -7.2, 600.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("STA06", "surface", -0.3, 7.0, 590.0, 0.0, ("HHZ", "HHN", "HHE")),
    ("BH01", "borehole", -1.8, -1.2, 555.0, 300.0, ("DPZ", "DP1", "DP2")),
    ("BH02", "borehole", 2.0, 0.9, 535.0, 240.0, ("DPZ", "DP1", "DP2")),
]

#: channel_map / orientations convention for the two borehole stations'
#: unoriented horizontals (mirrors config/example.yaml's ``stations.channel_map``).
CHANNEL_MAP = {"DPZ": "Z", "DP1": "1", "DP2": "2"}

#: 1D layer model (ARCHITECTURE.md §6a's own worked example, reused
#: verbatim so the doc and the fixture never disagree).
VEL1D_TEXT = """\
# depth_top_km  vp_km_s  vs_km_s  [rho_g_cc]
0.0   3.50  2.00  2.3
1.0   4.50  2.60  2.5
3.0   5.80  3.35  2.7
"""

N_EVENTS = 20
ML_MIN, ML_MAX = -0.5, 2.2
EVENT_OFFSET_KM_RANGE = (-9.0, 9.0)
EVENT_DEPTH_KM_RANGE = (0.3, 8.0)
#: Seconds before the day-1/day-2 boundary for the deliberately-placed
#: cross-day event (ARCHITECTURE.md §8 pitfall: exercises M2.2's
#: cross-day-boundary detection merge).
BOUNDARY_EVENT_OFFSET_S = 3.0

#: Background noise: continuous uniform noise in [-NOISE_HALF_RANGE,
#: NOISE_HALF_RANGE) counts on every channel (a deliberate simplification --
#: not Gaussian -- purely so bulk generation is fast; irrelevant to the
#: MAD-based / flat-run / availability QC statistics this fixture exists to
#: exercise).
NOISE_HALF_RANGE = 5.0

#: Flat instrument response sensitivity, counts per (m/s). See "Flat
#: response" in the module docstring.
SENSITIVITY_CPS = 2.0e8

#: Wood-Anderson simulation constants, read from the shared config schema's
#: own defaults so this module can't silently drift out of sync with it.
_HB = HuttonBooreConfig()
_ML_DEFAULTS = MlConfig()
HB_A, HB_B, HB_C = _HB.a, _HB.b, _HB.c
WA_SENSITIVITY = _ML_DEFAULTS.wa_sensitivity

#: Pulse dominant frequencies (Hz) and shape (see "Pulse shape" above).
P_DOM_FREQ_HZ = 10.0
S_DOM_FREQ_HZ = 6.0
PULSE_CYCLES = 6

#: Fixed amplitude ratios (relative to the calibrated "primary" horizontal
#: S zero-to-peak amplitude at that station) giving "S larger on
#: horizontals" and a realistic-but-simple P/S balance.
S_Z_RATIO = 0.5
P_TO_S_Z_RATIO = 0.35
P_HORIZ_RATIO = 0.5

#: Minimum hypocentral distance (km) used in the Hutton & Boore relation, to
#: keep log10(R/100) finite for a station essentially at the epicentre.
MIN_R_KM = 0.5

# --- Anomalies: fixed (seed_id, window) locations -------------------------

GAP_SEED_ID = f"{NETWORK}.STA05.{LOCATION}.HHZ"
GAP_START = UTCDateTime(2025, 1, 1, 6, 0, 0)
GAP_END = UTCDateTime(2025, 1, 1, 6, 10, 0)

SPIKE_SEED_ID = f"{NETWORK}.STA06.{LOCATION}.HHE"
SPIKE_START = UTCDateTime(2025, 1, 2, 0, 0, 0)
SPIKE_END = UTCDateTime(2025, 1, 3, 0, 0, 0)
N_SPIKES = 300
SPIKE_MIN_MULT, SPIKE_MAX_MULT = 80.0, 150.0

FLAT_SEED_ID = f"{NETWORK}.STA03.{LOCATION}.HHN"
FLAT_START = UTCDateTime(2025, 1, 1, 12, 0, 0)
FLAT_END = UTCDateTime(2025, 1, 1, 13, 0, 0)
FLAT_VALUE = 0.0

DAY_SAMPLES = int(round(86400 * SAMPLE_RATE_HZ))
TOTAL_SAMPLES = DAY_SAMPLES * N_DAYS

_WA_RESP_AT_S_FREQ = paz_2_amplitude_value_of_freq_resp(WOODANDERSON, S_DOM_FREQ_HZ)


# ---------------------------------------------------------------------------
# Geometry / travel times
# ---------------------------------------------------------------------------


def _to_latlon(x_km: float, y_km: float) -> tuple[float, float]:
    lat = LAT0 + y_km / KM_PER_DEG_LAT
    lon = LON0 + x_km / _km_per_deg_lon()
    return lat, lon


def _velocity_at_depth(layers: pd.DataFrame, depth_km: float, phase: str) -> float:
    col = "vp" if phase == "P" else "vs"
    idx = 0
    for i, top in enumerate(layers["depth_km"]):
        if top <= depth_km:
            idx = i
        else:
            break
    return float(layers[col].iloc[idx])


def travel_time(
    depth_src_km: float, depth_rcv_km: float, horiz_km: float, layers: pd.DataFrame, phase: str
) -> float:
    """Straight-ray travel time through the 1D layered model (see module docstring)."""
    z1, z2 = sorted([depth_src_km, depth_rcv_km])
    dz_total = z2 - z1
    dist = math.hypot(horiz_km, dz_total)
    if dist < 1e-9:
        return 0.0
    if dz_total < 1e-6:
        v = _velocity_at_depth(layers, z1, phase)
        return dist / v

    cos_theta = dz_total / dist
    col = "vp" if phase == "P" else "vs"
    tops = layers["depth_km"].tolist()
    vels = layers[col].tolist()
    # Extend the first layer upward (-inf) and the last downward (+inf),
    # per the module docstring's "extends indefinitely" reading of §6a.
    bounds = [-math.inf, *tops[1:], math.inf]

    total_time = 0.0
    for i in range(len(tops)):
        lo = max(bounds[i], z1)
        hi = min(bounds[i + 1], z2)
        if hi > lo:
            total_time += (hi - lo) / cos_theta / vels[i]
    return total_time


# ---------------------------------------------------------------------------
# Wood-Anderson amplitude calibration
# ---------------------------------------------------------------------------


def target_zero_to_peak_counts(ml: float, r_km: float) -> float:
    """Invert the Hutton & Boore relation + the flat response + the
    obspy Wood-Anderson pick formula to get the raw counts zero-to-peak
    amplitude that reproduces ``ml`` at hypocentral distance ``r_km``
    (see the module docstring's "Wood-Anderson ML calibration" section).
    """
    r_km = max(r_km, MIN_R_KM)
    wa_mm = 10.0 ** (ml - HB_A * math.log10(r_km / 100.0) - HB_B * (r_km - 100.0) - HB_C)
    counts_pp = wa_mm * 2.0 * SENSITIVITY_CPS / (_WA_RESP_AT_S_FREQ * WA_SENSITIVITY * 1000.0)
    return counts_pp / 2.0


# ---------------------------------------------------------------------------
# Pulse shape
# ---------------------------------------------------------------------------


def _pulse(freq_hz: float, cycles: float, dt: float) -> np.ndarray:
    """A causal, decaying tapered sinusoid, normalized to unit zero-to-peak
    amplitude (see "Pulse shape" in the module docstring)."""
    duration = cycles / freq_hz
    n = max(int(round(duration / dt)), 4)
    t = np.arange(n) * dt
    tau = (cycles / 2.0) / freq_hz
    wave = np.sin(2 * np.pi * freq_hz * t) * np.exp(-t / tau)
    peak = np.max(np.abs(wave))
    return wave / peak if peak > 0 else wave


_P_UNIT_PULSE = _pulse(P_DOM_FREQ_HZ, PULSE_CYCLES, 1.0 / SAMPLE_RATE_HZ)
_S_UNIT_PULSE = _pulse(S_DOM_FREQ_HZ, PULSE_CYCLES, 1.0 / SAMPLE_RATE_HZ)


def _add_pulse(
    buffer: np.ndarray, onset_sample: int, unit_pulse: np.ndarray, amplitude: float
) -> None:
    n = len(unit_pulse)
    start = onset_sample
    end = onset_sample + n
    if end <= 0 or start >= len(buffer):
        return
    src_lo = max(0, -start)
    src_hi = n - max(0, end - len(buffer))
    dst_lo = max(0, start)
    dst_hi = dst_lo + (src_hi - src_lo)
    if src_hi > src_lo:
        buffer[dst_lo:dst_hi] += amplitude * unit_pulse[src_lo:src_hi]


# ---------------------------------------------------------------------------
# Event catalogue
# ---------------------------------------------------------------------------


def _build_events(rng: np.random.Generator) -> list[dict]:
    offsets = rng.uniform(*EVENT_OFFSET_KM_RANGE, size=(N_EVENTS, 2))
    depths = rng.uniform(*EVENT_DEPTH_KM_RANGE, size=N_EVENTS)
    mls = rng.uniform(ML_MIN, ML_MAX, size=N_EVENTS)
    time_offsets = rng.uniform(0.0, N_DAYS * 86400.0, size=N_EVENTS)
    polarities = rng.choice([-1.0, 1.0], size=N_EVENTS)

    # Deliberately place event 0 a few seconds before the day-1/day-2
    # boundary (ARCHITECTURE.md §8 pitfall).
    time_offsets[0] = 86400.0 - BOUNDARY_EVENT_OFFSET_S

    order = np.argsort(time_offsets)
    events = []
    seen_ids: set[str] = set()
    for i in order:
        origin_time = DAY0 + float(time_offsets[i])
        event_id_hint = event_id_from_time(origin_time)
        while event_id_hint in seen_ids:  # guard against a (near-impossible) ms collision
            origin_time += 0.001
            event_id_hint = event_id_from_time(origin_time)
        seen_ids.add(event_id_hint)
        events.append(
            {
                # Stable original (pre-sort) index, used only to seed a
                # deterministic per-(event, station) RNG below -- *not*
                # Python's built-in hash() (randomized per-process by
                # PYTHONHASHSEED, which would break determinism).
                "idx": int(i),
                "event_id_hint": event_id_hint,
                "origin": {
                    "time": origin_time,
                    "lat": _to_latlon(offsets[i, 0], offsets[i, 1])[0],
                    "lon": _to_latlon(offsets[i, 0], offsets[i, 1])[1],
                    "depth_km": float(depths[i]),
                },
                "ml": float(mls[i]),
                "x_km": float(offsets[i, 0]),
                "y_km": float(offsets[i, 1]),
                "polarity": float(polarities[i]),
                "arrivals": {},
            }
        )
    return events


# ---------------------------------------------------------------------------
# Waveform generation
# ---------------------------------------------------------------------------


def _station_effective_depth_km(elev_m: float, depth_m: float) -> float:
    return -(elev_m - depth_m) / 1000.0


#: STATIONS order -> integer index, used only to seed the deterministic
#: per-(event, station) RNG below.
_STATION_INDEX = {s[0]: i for i, s in enumerate(STATIONS)}


def _generate_channel_buffer(
    seed_id: str,
    station: tuple,
    channels: list[str],
    channel: str,
    events: list[dict],
    layers: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    """Build the full 2-day float buffer (noise + injected pulses) for one channel."""
    buffer = rng.uniform(-NOISE_HALF_RANGE, NOISE_HALF_RANGE, size=TOTAL_SAMPLES)

    sta_code, kind, x_km, y_km, elev_m, depth_m, _ = station
    is_vertical = channel.endswith("Z")
    horiz_channels = [c for c in channels if not c.endswith("Z")]
    rcv_depth_km = _station_effective_depth_km(elev_m, depth_m)

    for ev in events:
        horiz_km = math.hypot(ev["x_km"] - x_km, ev["y_km"] - y_km)
        r_km = math.hypot(horiz_km, ev["origin"]["depth_km"] - rcv_depth_km)
        tp = travel_time(ev["origin"]["depth_km"], rcv_depth_km, horiz_km, layers, "P")
        ts = travel_time(ev["origin"]["depth_km"], rcv_depth_km, horiz_km, layers, "S")
        p_time = ev["origin"]["time"] + tp
        s_time = ev["origin"]["time"] + ts

        station_key = f"{NETWORK}.{sta_code}"
        ev["arrivals"].setdefault(station_key, {"P": p_time, "S": s_time})

        s_h_zp = target_zero_to_peak_counts(ev["ml"], r_km)
        # Deterministic per-(event, station) primary-horizontal / split
        # fraction draw, seeded from (seed, event index, station index) --
        # stable regardless of which channel of the station is being built,
        # and independent of Python's randomized-per-process hash().
        local_rng = np.random.default_rng((seed, ev["idx"], _STATION_INDEX[sta_code]))
        primary_idx = int(local_rng.integers(0, 2))
        secondary_frac = float(local_rng.uniform(0.5, 0.9))

        polarity = ev["polarity"]
        onset_p = int(round((p_time - DAY0) * SAMPLE_RATE_HZ))
        onset_s = int(round((s_time - DAY0) * SAMPLE_RATE_HZ))

        if is_vertical:
            _add_pulse(buffer, onset_p, _P_UNIT_PULSE, polarity * P_TO_S_Z_RATIO * s_h_zp)
            _add_pulse(buffer, onset_s, _S_UNIT_PULSE, polarity * S_Z_RATIO * s_h_zp)
        else:
            h_idx = horiz_channels.index(channel)
            is_primary = h_idx == primary_idx
            s_amp = s_h_zp if is_primary else s_h_zp * secondary_frac
            p_amp = P_TO_S_Z_RATIO * s_h_zp * P_HORIZ_RATIO
            p_amp = p_amp if is_primary else p_amp * secondary_frac
            _add_pulse(buffer, onset_p, _P_UNIT_PULSE, polarity * p_amp)
            _add_pulse(buffer, onset_s, _S_UNIT_PULSE, polarity * s_amp)

    if seed_id == SPIKE_SEED_ID:
        _apply_spikes(buffer, rng)
    if seed_id == FLAT_SEED_ID:
        _apply_flat(buffer)

    return buffer


def _apply_spikes(buffer: np.ndarray, rng: np.random.Generator) -> None:
    start_idx = int(round((SPIKE_START - DAY0) * SAMPLE_RATE_HZ))
    end_idx = int(round((SPIKE_END - DAY0) * SAMPLE_RATE_HZ))
    idx = rng.integers(start_idx, end_idx, size=N_SPIKES)
    signs = rng.choice([-1.0, 1.0], size=N_SPIKES)
    mults = rng.uniform(SPIKE_MIN_MULT, SPIKE_MAX_MULT, size=N_SPIKES)
    buffer[idx] = signs * mults * NOISE_HALF_RANGE


def _apply_flat(buffer: np.ndarray) -> None:
    start_idx = int(round((FLAT_START - DAY0) * SAMPLE_RATE_HZ))
    end_idx = int(round((FLAT_END - DAY0) * SAMPLE_RATE_HZ))
    buffer[start_idx:end_idx] = FLAT_VALUE


def _write_channel_days(
    outdir: Path, sta_code: str, channel: str, buffer: np.ndarray, seed_id: str
) -> None:
    counts = np.round(buffer).astype(np.int32)
    for day in range(N_DAYS):
        day_start = DAY0 + day * 86400.0
        lo, hi = day * DAY_SAMPLES, (day + 1) * DAY_SAMPLES
        day_data = counts[lo:hi]
        date_str = day_start.strftime("%Y-%m-%d")
        day_dir = outdir / "waveforms" / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{seed_id}.mseed"

        traces = []
        if seed_id == GAP_SEED_ID and GAP_START >= day_start and GAP_START < day_start + 86400.0:
            gap_lo = int(round((GAP_START - day_start) * SAMPLE_RATE_HZ))
            gap_hi = int(round((GAP_END - day_start) * SAMPLE_RATE_HZ))
            gap_hi = min(gap_hi, len(day_data))
            if gap_lo > 0:
                traces.append(_make_trace(day_data[:gap_lo], day_start, sta_code, channel))
            if gap_hi < len(day_data):
                seg_start = day_start + gap_hi / SAMPLE_RATE_HZ
                traces.append(_make_trace(day_data[gap_hi:], seg_start, sta_code, channel))
        else:
            traces.append(_make_trace(day_data, day_start, sta_code, channel))

        Stream(traces).write(str(path), format="MSEED", encoding="STEIM2", reclen=4096)


def _make_trace(data: np.ndarray, starttime: UTCDateTime, sta_code: str, channel: str) -> Trace:
    tr = Trace(data=data)
    tr.stats.network = NETWORK
    tr.stats.station = sta_code
    tr.stats.location = LOCATION
    tr.stats.channel = channel
    tr.stats.sampling_rate = SAMPLE_RATE_HZ
    tr.stats.starttime = starttime
    return tr


# ---------------------------------------------------------------------------
# StationXML / CSV
# ---------------------------------------------------------------------------


def _flat_response() -> Response:
    stage = PolesZerosResponseStage(
        stage_sequence_number=1,
        stage_gain=SENSITIVITY_CPS,
        stage_gain_frequency=1.0,
        input_units="M/S",
        output_units="COUNTS",
        pz_transfer_function_type="LAPLACE (RADIANS/SECOND)",
        normalization_frequency=1.0,
        zeros=[],
        poles=[],
        normalization_factor=1.0,
        input_units_description="Velocity in meters per second",
        output_units_description="Digitizer counts",
    )
    sensitivity = InstrumentSensitivity(
        value=SENSITIVITY_CPS, frequency=1.0, input_units="M/S", output_units="COUNTS"
    )
    return Response(instrument_sensitivity=sensitivity, response_stages=[stage])


_CHANNEL_ORIENTATION = {
    "Z": (0.0, -90.0),
    "N": (0.0, 0.0),
    "E": (90.0, 0.0),
    # DP1/DP2 are unoriented in reality (channel_map treats them as such
    # regardless of these values); nominal placeholders only, required by
    # the StationXML schema.
    "1": (0.0, 0.0),
    "2": (90.0, 0.0),
}


def _build_inventory() -> Inventory:
    network = Network(code=NETWORK, stations=[], description="OpenMEQ synthetic test network")
    for sta_code, kind, x_km, y_km, elev_m, depth_m, channels in STATIONS:
        lat, lon = _to_latlon(x_km, y_km)
        station = Station(
            code=sta_code,
            latitude=lat,
            longitude=lon,
            elevation=elev_m,
            site=Site(name=f"Synthetic {kind} site {sta_code}"),
            channels=[],
            creation_date=DAY0,
        )
        for cha in channels:
            azimuth, dip = _CHANNEL_ORIENTATION[cha[-1]]
            station.channels.append(
                Channel(
                    code=cha,
                    location_code=LOCATION,
                    latitude=lat,
                    longitude=lon,
                    elevation=elev_m,
                    depth=depth_m,
                    azimuth=azimuth,
                    dip=dip,
                    sample_rate=SAMPLE_RATE_HZ,
                    response=_flat_response(),
                )
            )
        network.stations.append(station)
    return Inventory(
        networks=[network], source="openmeq synthetic test data generator", created=DAY0
    )


def _station_rows() -> list[dict]:
    rows = []
    for sta_code, _kind, x_km, y_km, elev_m, depth_m, _channels in STATIONS:
        lat, lon = _to_latlon(x_km, y_km)
        rows.append(
            {
                "net": NETWORK,
                "sta": sta_code,
                "loc": LOCATION,
                "lat": lat,
                "lon": lon,
                "elev_m": elev_m,
                "depth_m": depth_m,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _build_config(outdir: Path) -> dict:
    return {
        "project": {"name": "synthetic", "output_dir": "./runs/synthetic"},
        "parallel": {"n_workers": 1, "threads_per_worker": 1},
        "data": {
            "waveform_dirs": ["./waveforms"],
            "file_pattern": "*.mseed",
            "start": DAY0.date,
            "end": (DAY0 + (N_DAYS - 1) * 86400.0).date,
        },
        "stations": {
            "stationxml": "./inventory/stations.xml",
            "csv": "./inventory/stations.csv",
            "channel_map": dict(CHANNEL_MAP),
            "orientations": {},
        },
        "nlloc": {
            "velocity_model": {"type": "1d", "file": "./models/vel1d.txt"},
            "grid": {
                "origin": [LAT0, LON0],
                "size_km": [30.0, 30.0, 12.0],
                "spacing_km": 0.1,
            },
        },
        "hypodd": {
            "velocity_model": {"type": "1d", "file": "./models/vel1d.txt"},
        },
    }


# ---------------------------------------------------------------------------
# Truth <-> JSON
# ---------------------------------------------------------------------------


def _truth_to_jsonable(truth: dict) -> dict:
    def iso(t: UTCDateTime) -> str:
        return str(t)

    return {
        "seed": truth["seed"],
        "events": [
            {
                "event_id_hint": ev["event_id_hint"],
                "origin": {
                    "time": iso(ev["origin"]["time"]),
                    "lat": ev["origin"]["lat"],
                    "lon": ev["origin"]["lon"],
                    "depth_km": ev["origin"]["depth_km"],
                },
                "ml": ev["ml"],
                "arrivals": {
                    sta: {"P": iso(times["P"]), "S": iso(times["S"])}
                    for sta, times in ev["arrivals"].items()
                },
            }
            for ev in truth["events"]
        ],
        "known_issues": [
            {
                "seed_id": issue["seed_id"],
                "kind": issue["kind"],
                "start": iso(issue["start"]),
                "end": iso(issue["end"]),
            }
            for issue in truth["known_issues"]
        ],
        "stations": truth["stations"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(outdir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    """Generate the full synthetic dataset under ``outdir``. Returns ``truth``.

    See the module docstring for the exact directory layout, truth
    structure, travel-time method and ML calibration.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    from openmeq.velocity import read_layers

    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vel1d_path = models_dir / "vel1d.txt"
    vel1d_path.write_text(VEL1D_TEXT)
    layers = read_layers(vel1d_path)

    events = _build_events(rng)

    for sta_code, _kind, x_km, y_km, elev_m, depth_m, channels in STATIONS:
        for cha in channels:
            seed_id = f"{NETWORK}.{sta_code}.{LOCATION}.{cha}"
            buffer = _generate_channel_buffer(
                seed_id,
                (sta_code, _kind, x_km, y_km, elev_m, depth_m, channels),
                list(channels),
                cha,
                events,
                layers,
                rng,
                seed,
            )
            _write_channel_days(outdir, sta_code, cha, buffer, seed_id)

    inventory_dir = outdir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inv = _build_inventory()
    inv.write(str(inventory_dir / "stations.xml"), format="STATIONXML")

    station_rows = _station_rows()
    pd.DataFrame(station_rows).to_csv(inventory_dir / "stations.csv", index=False)

    config = _build_config(outdir)
    (outdir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    known_issues = [
        {"seed_id": GAP_SEED_ID, "kind": "gap", "start": GAP_START, "end": GAP_END},
        {"seed_id": SPIKE_SEED_ID, "kind": "spikes", "start": SPIKE_START, "end": SPIKE_END},
        {"seed_id": FLAT_SEED_ID, "kind": "flat", "start": FLAT_START, "end": FLAT_END},
    ]

    truth = {
        "seed": seed,
        "events": [
            {
                "event_id_hint": ev["event_id_hint"],
                "origin": ev["origin"],
                "ml": ev["ml"],
                "arrivals": ev["arrivals"],
            }
            for ev in events
        ],
        "known_issues": known_issues,
        "stations": station_rows,
    }

    (outdir / "truth.json").write_text(json.dumps(_truth_to_jsonable(truth), indent=2))

    return truth


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", help="Output directory for the synthetic dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.outdir, seed=args.seed)
    print(f"Wrote synthetic dataset to {args.outdir}")


if __name__ == "__main__":
    _cli()
