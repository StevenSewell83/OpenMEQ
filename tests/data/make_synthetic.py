"""Deterministic synthetic test dataset generator (M0.8, ARCHITECTURE.md §8).

``generate(outdir, seed=DEFAULT_SEED, n_stations=None, n_days=None) -> dict``
builds, under ``outdir``:

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

``n_stations``/``n_days`` (M11.3, ``OpenMEQ-m1f.3``): optional overrides,
each ``>=`` the fixed default (8 stations, 2 days) -- this dataset only
ever grows, it does not generalize to an arbitrary smaller network. Left at
their default ``None``, every call is byte-identical to every pre-M11.3
call (no extra RNG draws happen at all); see :func:`generate`'s own
docstring for the full contract. Built for ``scripts/benchmark.py``'s
opt-in larger benchmark dataset (``--n-stations``/``--n-days``), not for
the correctness-test fixture other tests rely on.

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

Travel-time method (exact 1D layered first arrival: direct ray + head waves)
-----------------------------------------------------------------------------
For a source at depth ``z_src`` (km, positive down from sea level per
ARCHITECTURE.md §6a) and a receiver at *effective* depth ``z_rcv`` (see
below) separated by epicentral (horizontal) distance ``x`` (km), the first
arrival is the minimum of two families of ray, computed exactly (constant-
velocity layers, proper Snell's-law bending at each interface -- **not** a
straight chord) rather than approximated:

- **Direct ray** (:func:`_direct_wave_time`): the ray parameter (horizontal
  slowness) ``p`` is constant along the whole path (Snell's law for a stack
  of horizontal constant-velocity layers). For every layer ``i`` between
  ``z_src`` and ``z_rcv`` (thickness ``h_i``, velocity ``v_i`` for the
  requested phase), the ray's horizontal offset within that layer is
  ``h_i * p * v_i / sqrt(1 - p**2 * v_i**2)``; ``p`` (``0 <= p < 1/v_max``,
  ``v_max`` the fastest layer traversed) is found by bisection so the sum of
  these offsets equals ``x`` exactly (:func:`_bisect_ray_parameter`). Travel
  time is then ``sum_i(h_i / (v_i * sqrt(1 - p**2 * v_i**2)))``. A source and
  receiver at the same depth (no vertical separation to bisect over) fall
  back to the exact ``x / v`` at that shared depth.
- **Head (refracted) waves** (:func:`_head_wave_time`): for every interface
  at or below both ``z_src`` and ``z_rcv`` where velocity increases with
  depth, the ray travels down from the source (and up to the receiver) at
  the critical ray parameter ``p = 1 / v_refractor``, then along the
  interface at ``v_refractor``. This is only a real arrival once the
  epicentral distance exceeds the critical distance (the down-going and
  up-going legs' own horizontal offsets, computed the same way as the direct
  ray's, must not exceed ``x``) -- closer offsets have no head wave on that
  interface and are excluded from the minimum.

The first arrival is ``min(direct, min-over-interfaces(head))``. A receiver
above the shallowest tabulated layer top (e.g. a mountaintop surface
station) or a source below the deepest tabulated layer top both use the
nearest layer's velocity, extending it indefinitely, per ARCHITECTURE.md §6a
("the first layer applies from the top of the model to the next depth" --
read here as also extending upward past the first entry, and the last layer
extending downward past its entry). A borehole receiver (effective elevation
below sea level, i.e. positive ``z_rcv``, per "Station effective depth"
below) is handled exactly like a surface one: the ray path (direct or head)
only runs between ``z_src`` and ``z_rcv`` -- it is never assumed to start or
end at the surface.

**Accuracy**: this reproduces the same physics NLLoc's ``Grid2Time``
(Podvin & Lecomte finite-difference eikonal solver) and hypoDD's 1D ray
tracer solve for, so validated agreement is at the millisecond level, not
the tens-of-milliseconds-to-percent level a straight-ray chord would give
(see ``tests/locate/test_synthetic_traveltime_validation.py``, a
``requires_nlloc`` test that compares these arrivals directly against
``Grid2Time`` output built via ``openmeq.nlloc_setup``, and reports the
remaining difference -- almost entirely Grid2Time's own finite-difference
grid-spacing error -- as a function of grid spacing). Downstream location/
relocation tests can therefore budget much tighter residual and recovery
tolerances than the old straight-ray generator allowed; see
ARCHITECTURE.md §8 and each test module's own tolerance comments for the
current numbers.

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


def _layer_bounds_vels(layers: pd.DataFrame, phase: str) -> tuple[list[float], list[float]]:
    """``(bounds, vels)`` for the requested phase: ``bounds`` has
    ``len(vels) + 1`` entries, with ``[bounds[i], bounds[i+1])`` layer ``i``'s
    vertical extent -- the first layer's top extended to ``-inf`` and the
    last layer's bottom extended to ``+inf``, per the module docstring's
    "extends indefinitely" reading of ARCHITECTURE.md §6a.
    """
    col = "vp" if phase == "P" else "vs"
    tops = layers["depth_km"].tolist()
    vels = layers[col].tolist()
    bounds = [-math.inf, *tops[1:], math.inf]
    return bounds, vels


def _segments(
    z1: float, z2: float, bounds: list[float], vels: list[float]
) -> list[tuple[float, float]]:
    """Non-empty ``(thickness_km, velocity_km_s)`` pairs for every layer
    overlapping ``[z1, z2]`` (``z1 <= z2``), in depth order."""
    segs = []
    for i, v in enumerate(vels):
        lo = max(bounds[i], z1)
        hi = min(bounds[i + 1], z2)
        if hi > lo:
            segs.append((hi - lo, v))
    return segs


def _bisect_ray_parameter(x_km: float, segs: list[tuple[float, float]]) -> float:
    """Ray parameter ``p`` (s/km), ``0 <= p < 1/v_max``, such that
    ``sum(h_i * p * v_i / sqrt(1 - p**2 * v_i**2) for h_i, v_i in segs) ==
    x_km`` (the direct-wave shooting equation -- see "Travel-time method" in
    the module docstring). The offset is monotonically increasing in ``p``
    (0 at ``p=0``, diverging as ``p -> 1/v_max``), so bisection always
    converges to the unique root for any finite ``x_km >= 0``.
    """
    v_max = max(v for _h, v in segs)
    lo, hi = 0.0, (1.0 / v_max) * (1.0 - 1e-12)

    def offset(p: float) -> float:
        return sum(h * p * v / math.sqrt(1.0 - (p * v) ** 2) for h, v in segs)

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if offset(mid) < x_km:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _direct_wave_time(
    z_src: float, z_rcv: float, x_km: float, bounds: list[float], vels: list[float]
) -> float:
    """Direct-ray travel time between depths ``z_src``/``z_rcv`` separated by
    epicentral distance ``x_km``: ray-parameter shooting (bisection on
    ``p``) through every layer between the two depths. Returns ``inf`` for
    the degenerate same-depth case (see :func:`travel_time`'s caller, which
    handles it separately)."""
    z1, z2 = sorted([z_src, z_rcv])
    if z2 - z1 < 1e-6:
        return math.inf
    segs = _segments(z1, z2, bounds, vels)
    if not segs:
        return math.inf
    if x_km < 1e-9:
        return sum(h / v for h, v in segs)
    p = _bisect_ray_parameter(x_km, segs)
    return sum(h / (v * math.sqrt(1.0 - (p * v) ** 2)) for h, v in segs)


def _head_wave_time(
    z_src: float, z_rcv: float, x_km: float, bounds: list[float], vels: list[float]
) -> float:
    """Minimum refracted (head-wave) travel time over every interface at or
    below both ``z_src`` and ``z_rcv`` where velocity increases (see
    "Travel-time method" in the module docstring). Returns ``inf`` if no
    interface yields a valid (post-critical-distance) head wave.
    """
    best = math.inf
    z_deep = max(z_src, z_rcv)
    for j in range(1, len(vels)):
        d = bounds[j]  # interior interface depth (finite: j in [1, n-1])
        if d < z_deep - 1e-9:
            continue  # interface is above the source or receiver
        v_ref = vels[j]
        segs = _segments(z_src, d, bounds, vels) + _segments(z_rcv, d, bounds, vels)
        if any(v_i >= v_ref - 1e-12 for _h, v_i in segs):
            continue  # refractor not faster than every overlying layer traversed
        if not segs:
            # Source and receiver both sit exactly at the interface: the
            # head wave degenerates to travelling along it the whole way.
            best = min(best, x_km / v_ref)
            continue
        p = 1.0 / v_ref
        # Delay-time method (e.g. Sheriff & Geldart): T = x/v_ref + delay,
        # delay = sum(h_i * sqrt(1/v_i**2 - 1/v_ref**2)) over the layers
        # traversed on *both* legs (down from the source, up to the
        # receiver) -- algebraically ``h_i * sqrt(1 - (p*v_i)**2) / v_i``
        # here, since ``p = 1/v_ref``. This is **not** the down/up legs'
        # own travel time (that would double-count each leg's own offset,
        # already accounted for by dividing the *full* x_km by v_ref) --
        # only the *excess* time each leg needs beyond travelling that same
        # leg's own offset at v_ref.
        delay = sum(h * math.sqrt(1.0 - (p * v) ** 2) / v for h, v in segs)
        offset = sum(h * p * v / math.sqrt(1.0 - (p * v) ** 2) for h, v in segs)
        remaining = x_km - offset
        if remaining < 0:
            continue  # closer than the critical distance: no head wave here
        best = min(best, x_km / v_ref + delay)
    return best


def travel_time(
    depth_src_km: float, depth_rcv_km: float, horiz_km: float, layers: pd.DataFrame, phase: str
) -> float:
    """Exact 1D layered first-arrival travel time (see "Travel-time method"
    in the module docstring): the minimum of the direct ray
    (:func:`_direct_wave_time`) and every candidate head/refracted wave
    (:func:`_head_wave_time`).
    """
    z1, z2 = sorted([depth_src_km, depth_rcv_km])
    dist = math.hypot(horiz_km, z2 - z1)
    if dist < 1e-9:
        return 0.0

    bounds, vels = _layer_bounds_vels(layers, phase)

    if z2 - z1 < 1e-6:
        v = _velocity_at_depth(layers, z1, phase)
        direct = horiz_km / v
    else:
        direct = _direct_wave_time(depth_src_km, depth_rcv_km, horiz_km, bounds, vels)

    head = _head_wave_time(depth_src_km, depth_rcv_km, horiz_km, bounds, vels)
    return min(direct, head)


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


def _build_events(
    rng: np.random.Generator, n_days: int = N_DAYS, n_events: int = N_EVENTS
) -> list[dict]:
    offsets = rng.uniform(*EVENT_OFFSET_KM_RANGE, size=(n_events, 2))
    depths = rng.uniform(*EVENT_DEPTH_KM_RANGE, size=n_events)
    mls = rng.uniform(ML_MIN, ML_MAX, size=n_events)
    time_offsets = rng.uniform(0.0, n_days * 86400.0, size=n_events)
    polarities = rng.choice([-1.0, 1.0], size=n_events)

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
    total_samples: int = TOTAL_SAMPLES,
    station_index: dict[str, int] = _STATION_INDEX,
) -> np.ndarray:
    """Build the full multi-day float buffer (noise + injected pulses) for one channel.

    ``total_samples`` defaults to the fixed 2-day ``TOTAL_SAMPLES`` constant;
    :func:`generate`'s ``n_days`` override (M11.3, ``OpenMEQ-m1f.3``) passes
    a larger value to extend the buffer (and therefore the number of
    per-day files :func:`_write_channel_days` later splits it into) without
    changing anything about the default 2-day path's own RNG draw sequence.
    """
    buffer = rng.uniform(-NOISE_HALF_RANGE, NOISE_HALF_RANGE, size=total_samples)

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
        local_rng = np.random.default_rng((seed, ev["idx"], station_index[sta_code]))
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
    outdir: Path,
    sta_code: str,
    channel: str,
    buffer: np.ndarray,
    seed_id: str,
    n_days: int = N_DAYS,
    day_samples: int = DAY_SAMPLES,
) -> None:
    counts = np.round(buffer).astype(np.int32)
    for day in range(n_days):
        day_start = DAY0 + day * 86400.0
        lo, hi = day * day_samples, (day + 1) * day_samples
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


def _build_inventory(stations: list[tuple] = STATIONS) -> Inventory:
    network = Network(code=NETWORK, stations=[], description="OpenMEQ synthetic test network")
    for sta_code, kind, x_km, y_km, elev_m, depth_m, channels in stations:
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


def _station_rows(stations: list[tuple] = STATIONS) -> list[dict]:
    rows = []
    for sta_code, _kind, x_km, y_km, elev_m, depth_m, _channels in stations:
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


def _build_config(outdir: Path, n_days: int = N_DAYS) -> dict:
    return {
        "project": {"name": "synthetic", "output_dir": "./runs/synthetic"},
        "parallel": {"n_workers": 1, "threads_per_worker": 1},
        "data": {
            "waveform_dirs": ["./waveforms"],
            "file_pattern": "*.mseed",
            "start": DAY0.date,
            "end": (DAY0 + (n_days - 1) * 86400.0).date,
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


#: Aperture (km) each *extra* station (beyond the fixed 8, ``STATIONS``) is
#: drawn from, centred on the network origin -- wide enough to keep station
#: density roughly comparable to the original 8-station, ~14.9 km-aperture
#: network as ``n_stations`` grows (M11.3, ``OpenMEQ-m1f.3``: "reuse M0.8's
#: generator, parametrized for a larger station count", see
#: :func:`generate`'s ``n_stations`` parameter).
_EXTRA_STATION_APERTURE_KM = 10.0


def _extra_stations(rng: np.random.Generator, n_extra: int) -> list[tuple]:
    """``n_extra`` additional, deterministically-placed *surface* ``HH?`` stations.

    Appended *after* the fixed :data:`STATIONS` list (never mutating it --
    several tests import ``make_synthetic.STATIONS`` directly and expect
    exactly the original 8), so the original 8 stations' own per-channel RNG
    draw sequence (and therefore their exact waveform content) is completely
    unaffected by ``n_extra`` -- only consumed *after* every original
    station/channel's own draws, when ``n_extra > 0``.  Named ``STA{9,10,...}``
    to avoid colliding with the fixed ``STA01``-``STA06``/``BH01``-``BH02``
    codes.  All-surface, all-``HHZ``/``HHN``/``HHE`` (no borehole variety)
    -- this task only needs *more channel-days of realistic-shaped data* to
    profile/benchmark against, not more hand-tuned network-geometry
    diversity (that's what the fixed 8 already exercise).
    """
    if n_extra <= 0:
        return []
    xs = rng.uniform(-_EXTRA_STATION_APERTURE_KM, _EXTRA_STATION_APERTURE_KM, size=n_extra)
    ys = rng.uniform(-_EXTRA_STATION_APERTURE_KM, _EXTRA_STATION_APERTURE_KM, size=n_extra)
    elevs = rng.uniform(500.0, 700.0, size=n_extra)
    stations = []
    for i in range(n_extra):
        code = f"STA{9 + i:02d}"
        stations.append(
            (
                code,
                "surface",
                float(xs[i]),
                float(ys[i]),
                float(elevs[i]),
                0.0,
                ("HHZ", "HHN", "HHE"),
            )
        )
    return stations


def generate(
    outdir: str | Path,
    seed: int = DEFAULT_SEED,
    n_stations: int | None = None,
    n_days: int | None = None,
) -> dict:
    """Generate the full synthetic dataset under ``outdir``. Returns ``truth``.

    See the module docstring for the exact directory layout, truth
    structure, travel-time method and ML calibration.

    Parameters
    ----------
    outdir : str or pathlib.Path
    seed : int, optional
        Default :data:`DEFAULT_SEED`.
    n_stations : int or None, optional
        Total station count. ``None`` (default) uses exactly the fixed
        :data:`STATIONS` list (8 stations) -- byte-identical to every
        pre-M11.3 call, since no extra RNG draws happen at all in that case
        (see :func:`_extra_stations`). A larger value appends
        deterministically-placed extra surface stations (M11.3,
        ``OpenMEQ-m1f.3``: "parametrized for a larger station count", for
        ``scripts/benchmark.py``'s larger benchmark dataset) *after* the
        fixed 8, which are otherwise completely unchanged. Must be
        ``>= len(STATIONS)`` (this parameter only ever makes the network
        bigger, matching the fixed 8's own deliberate anomaly/geometry
        design -- it is not a general-purpose "any station count" generator).
    n_days : int or None, optional
        Total day count. ``None`` (default) uses exactly :data:`N_DAYS`
        (2) -- byte-identical to every pre-M11.3 call. A larger value
        extends the continuous background-noise buffer (the known
        gap/spike/flat anomalies and the day-boundary event stay fixed on
        their original days 1/2) and scales the event count proportionally
        (:data:`N_EVENTS` per :data:`N_DAYS`, rounded) so a larger dataset
        still has a comparable event rate per day rather than the same 20
        events spread arbitrarily thin. Must be ``>= N_DAYS`` (the fixed
        spike anomaly's window is on calendar day 2, which must exist).

    Raises
    ------
    ValueError
        If ``n_stations < len(STATIONS)`` or ``n_days < N_DAYS`` -- this
        parametrization only ever grows the fixed dataset (see above), not
        the pre-existing tiny correctness-test fixture other tests depend on.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if n_stations is None:
        n_stations = len(STATIONS)
    elif n_stations < len(STATIONS):
        raise ValueError(
            f"n_stations must be >= {len(STATIONS)} (the fixed base network); got {n_stations}"
        )
    if n_days is None:
        n_days = N_DAYS
    elif n_days < N_DAYS:
        raise ValueError(
            f"n_days must be >= {N_DAYS} (the fixed anomalies span 2 days); got {n_days}"
        )

    day_samples = int(round(86400 * SAMPLE_RATE_HZ))
    total_samples = day_samples * n_days
    n_events = N_EVENTS if n_days <= N_DAYS else int(round(N_EVENTS * n_days / N_DAYS))

    rng = np.random.default_rng(seed)

    from openmeq.velocity import read_layers

    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vel1d_path = models_dir / "vel1d.txt"
    vel1d_path.write_text(VEL1D_TEXT)
    layers = read_layers(vel1d_path)

    events = _build_events(rng, n_days=n_days, n_events=n_events)

    stations = STATIONS + _extra_stations(rng, n_stations - len(STATIONS))
    station_index = {s[0]: i for i, s in enumerate(stations)}

    for sta_code, _kind, x_km, y_km, elev_m, depth_m, channels in stations:
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
                total_samples=total_samples,
                station_index=station_index,
            )
            _write_channel_days(
                outdir, sta_code, cha, buffer, seed_id, n_days=n_days, day_samples=day_samples
            )

    inventory_dir = outdir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inv = _build_inventory(stations)
    inv.write(str(inventory_dir / "stations.xml"), format="STATIONXML")

    station_rows = _station_rows(stations)
    pd.DataFrame(station_rows).to_csv(inventory_dir / "stations.csv", index=False)

    config = _build_config(outdir, n_days=n_days)
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
