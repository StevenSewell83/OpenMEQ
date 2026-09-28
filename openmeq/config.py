"""YAML config loading and pydantic validation.

``load_config(path) -> Config`` turns a single user YAML file (documented in
full in ``config/example.yaml``, the reference schema — see PLAN.md §3 for
the original sketch) into a validated, typed :class:`Config` object with one
attribute per top-level YAML section, per ARCHITECTURE.md §2:
``cfg.project``, ``cfg.parallel``, ``cfg.data``, ``cfg.stations``, ``cfg.qc``,
``cfg.detection``, ``cfg.chunks``, ``cfg.phasenet``, ``cfg.nlloc``,
``cfg.review``, ``cfg.magnitude``, ``cfg.hypodd``.

Design notes
------------
* Every section model uses ``extra="forbid"`` (see :class:`_StrictModel`), so
  a typo'd or renamed YAML key raises a validation error instead of being
  silently ignored.
* **Path resolution:** any relative path in the YAML is resolved against the
  *config file's own directory* (not the process CWD), per ARCHITECTURE.md
  §2. This is implemented in one place: :meth:`Config._resolve_relative_paths`,
  a top-level ``model_validator(mode="after")`` that receives the config
  file's directory via pydantic's validation ``context`` (see
  :func:`load_config`) and walks every known path-like field. Do not add
  path-resolution logic to individual section models; extend the list in
  ``_resolve_relative_paths`` instead, so the "resolve against the config
  file's directory" rule stays in exactly one place.
* **Time representation:** ``cfg.data.start`` / ``cfg.data.end`` are parsed as
  plain :class:`datetime.date` objects (PyYAML parses unquoted ISO-8601 dates
  in the YAML directly into ``date`` instances). This is a config-layer
  convention only; it does *not* contradict ARCHITECTURE.md §1's runtime
  convention that step code uses ``obspy.UTCDateTime`` throughout. Step code
  that needs a timestamp constructs ``obspy.UTCDateTime(cfg.data.start)``
  itself. Keeping obspy out of this module's imports means ``import
  openmeq.config`` never pays obspy's import-time cost and stays usable from
  contexts (e.g. quick config sanity checks) that shouldn't need it.
* ``parallel.n_workers: "auto"`` is stored as the literal string "auto";
  resolving it to a physical core count is ``openmeq.parallel``'s job
  (M0.5), not this module's.
* Several sections carry keys beyond PLAN.md §3's original sketch, added
  ahead of the later milestones that specified them so those tasks don't
  need to touch this module concurrently. Each is commented at its
  definition with the plan key/beads issue that introduced it, and mirrored
  with the same comment in ``config/example.yaml``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger("openmeq.config")


class ConfigError(Exception):
    """A user-facing configuration problem.

    Raised for malformed YAML, failed schema validation (wrapping pydantic's
    ``ValidationError`` so field paths are readable), or an invalid
    referenced file discovered while a later step loads a config-pointed
    resource (e.g. a local ML calibration CSV, a station-label collision).
    Later modules are encouraged to raise this same type for config-adjacent,
    user-fixable problems rather than a bare ``ValueError``.
    """


class _StrictModel(BaseModel):
    """Base class for every config section.

    ``extra="forbid"`` makes an unknown/misspelled key a hard validation
    error instead of a silently-ignored typo (PLAN.md §3's YAML sketch is
    long, and typos are likely).
    """

    model_config = ConfigDict(extra="forbid")


def _resolve(path: Path | None, base_dir: Path) -> Path | None:
    """Resolve ``path`` against ``base_dir`` if it is relative.

    An absolute path is returned unchanged. ``None`` passes through
    unchanged (many path fields are optional).
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


class ProjectConfig(_StrictModel):
    name: str
    output_dir: Path


# ---------------------------------------------------------------------------
# parallel
# ---------------------------------------------------------------------------


class ParallelConfig(_StrictModel):
    n_workers: int | Literal["auto"] = "auto"
    threads_per_worker: int = 1
    phasenet_device: Literal["cpu", "gpu"] = "cpu"
    chunk_days: int = 1


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class DataConfig(_StrictModel):
    waveform_dirs: list[Path]
    file_pattern: str = "*.mseed"
    start: date
    end: date


# ---------------------------------------------------------------------------
# stations
# ---------------------------------------------------------------------------


class StationsConfig(_StrictModel):
    stationxml: Path | None = None
    stationxml_dir: Path | None = None
    csv: Path | None = None
    channel_map: dict[str, str] = Field(default_factory=dict)
    orientations: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------


class QcConfig(_StrictModel):
    daily_stats: list[str] = Field(
        default_factory=lambda: ["rms", "mean_abs", "spike_count", "kurtosis"]
    )
    spike_threshold_mad: float = 10.0
    min_availability_pct: float = 80.0
    exclude_channels: list[str] = Field(default_factory=list)
    # M1.4 / OpenMEQ-5nl.4: daily noise-statistics run-length thresholds.
    flat_run_min_samples: int = 50
    clip_run_min_samples: int = 3
    # M1.5 / OpenMEQ-5nl.5: monthly cross-station comparison pass/fail thresholds.
    # M1.9 / OpenMEQ-5nl.9 retuned max_spike_fraction for the new isolated-impulse
    # spike_count definition -- see docs/performance.md for the measured
    # synthetic_project numbers this default is chosen against (the known
    # spiky channel/day measures ~2.25e-5, every ordinary channel/day <= ~7.3e-6).
    max_spike_fraction: float = 0.000012
    max_flat_fraction: float = 0.5
    max_clip_count: int = 0
    monthly_outlier_ratio: float = 3.0
    # M1.9 / OpenMEQ-5nl.9: isolated-impulse spike metric (ARCHITECTURE.md §3)
    # -- the odd-length running-median window (samples) the residual
    # r = x - running_median(x, spike_median_window) is computed against.
    spike_median_window: int = 7
    # M1.9 / OpenMEQ-5nl.9: flag "flat" when the single longest flat run on
    # a channel/day exceeds this many seconds (a real fault, e.g. a dead
    # sensor for an hour, regardless of what fraction of the whole day that
    # is), in addition to the existing max_flat_fraction criterion.
    max_flat_run_s: float = 60.0


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


class DetectionFilterConfig(_StrictModel):
    type: Literal["bandpass", "highpass", "lowpass", "bandstop"] = "bandpass"
    freqmin: float | None = 5.0
    freqmax: float | None = 80.0


class DetectionConfig(_StrictModel):
    sta: float = 0.05
    lta: float = 1.0
    trigger_on: float = 3.5
    trigger_off: float = 1.5
    filter: DetectionFilterConfig = Field(default_factory=DetectionFilterConfig)
    coincidence_sum: int = 4
    station_weights: dict[str, float] = Field(default_factory=dict)
    # M2.1 / OpenMEQ-ujx.1: padding read on each side of a per-day detection
    # window so a trigger isn't lost at a day boundary; must comfortably
    # exceed lta plus a trigger's typical duration.
    day_overlap_s: float = 60.0
    # M2.2 / OpenMEQ-ujx.2: two triggers within this many seconds of each
    # other (greedy chained clustering) are merged into one detection.
    dedup_window_s: float = 2.0


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------


class ChunksConfig(_StrictModel):
    pre_s: float = 30.0
    post_s: float = 30.0


# ---------------------------------------------------------------------------
# phasenet
# ---------------------------------------------------------------------------


class PhasenetConfig(_StrictModel):
    target_rate_hz: float = 100.0
    taper_pct: float = 0.05
    highpass_hz: float = 1.0
    min_p_prob: float = 0.3
    min_s_prob: float = 0.3
    # M4.3 / OpenMEQ-1m7.3: currently unused/aspirational -- the batch
    # runner (openmeq.pick.run_phasenet) always invokes predict.py with
    # --batch_size=1 regardless of this value. predict.py's own batching
    # requires every chunk in one call to share both time length AND
    # station count; this project's chunks_pn/ files generally have
    # different station counts event to event, so batch_size>1 would fail
    # on the first mismatched pair (docs/phasenet_interface.md §7), and
    # there's no measured CPU speedup from it either way (§6). Kept as a
    # contract note for whoever revisits GPU support/per-station-count
    # batching.
    batch_size: int = 20
    # M4.2 / OpenMEQ-1m7.2: gaps longer than this are zero-filled rather than
    # interpolated during chunk preprocessing.
    max_gap_fill_s: float = 2.0
    # M4.4 / OpenMEQ-1m7.4: PhaseNet emits no timing uncertainty (only a
    # probability); this becomes Pick.time_errors.uncertainty.
    pick_uncertainty_s: float = 0.1


# ---------------------------------------------------------------------------
# nlloc
# ---------------------------------------------------------------------------


class NllocVelocityModelConfig(_StrictModel):
    type: Literal["1d", "3d"] = "1d"
    file: Path


class NllocGridConfig(_StrictModel):
    origin: tuple[float, float]
    size_km: tuple[float, float, float]
    spacing_km: float


class POnlyFilterConfig(_StrictModel):
    """Pass A (P-picks-only) iterative residual filter."""

    max_iter: int = 5
    max_residual_s: float = 0.10
    residual_sigma: float = 3.0


class PhaseResidualFilterConfig(_StrictModel):
    max_residual_s: float
    residual_sigma: float


class PAndSFilterConfig(_StrictModel):
    """Pass B (P + S picks) iterative residual filter."""

    max_iter: int = 5
    s_predicted_window_s: float = 0.3
    p: PhaseResidualFilterConfig = Field(
        default_factory=lambda: PhaseResidualFilterConfig(max_residual_s=0.10, residual_sigma=3.0)
    )
    s: PhaseResidualFilterConfig = Field(
        default_factory=lambda: PhaseResidualFilterConfig(max_residual_s=0.20, residual_sigma=3.0)
    )


class EventsFilterConfig(_StrictModel):
    """Event-level accept/reject thresholds, applied after pass A/B."""

    min_p_phases: int = 4
    min_phases: int = 6
    min_s_phases: int = 1
    max_rms_s: float = 0.1
    max_gap_deg: float = 270.0
    max_err_h_km: float = 1.0
    max_err_z_km: float = 2.0


class NllocFilterConfig(_StrictModel):
    p_only: POnlyFilterConfig = Field(default_factory=POnlyFilterConfig)
    p_and_s: PAndSFilterConfig = Field(default_factory=PAndSFilterConfig)
    events: EventsFilterConfig = Field(default_factory=EventsFilterConfig)


class NllocLocSearchConfig(_StrictModel):
    """NLLoc ``LOCSEARCH OCT`` (octtree search) parameters.

    Verified against the installed NonLinLoc (docs/nlloc_interface.md §5):
    ``LOCSEARCH OCT initNumCells_x initNumCells_y initNumCells_z minNodeSize
    maxNumNodes numScatter useStationsDensity stopOnMinNodeSize`` -- field
    names, order and defaults below match this exactly (the field order was
    already correct when this schema was first written).
    """

    init_num_cells_x: int = 10
    init_num_cells_y: int = 10
    init_num_cells_z: int = 10
    min_node_size: float = 0.01
    max_num_nodes: int = 20000
    num_scatter: int = 5000
    use_stations_density: bool = False
    stop_on_min_node_size: bool = True


class NllocLocGauConfig(_StrictModel):
    """NLLoc ``LOCGAU`` (constant Gaussian model-error covariance).

    Verified against the installed NonLinLoc (docs/nlloc_interface.md §5):
    ``LOCGAU SigmaTime CorrLen`` -- matches exactly.
    """

    sigma_time_s: float = 0.1
    corr_len_km: float = 0.0


class NllocLocGau2Config(_StrictModel):
    """NLLoc ``LOCGAU2`` (travel-time-dependent covariance), used together
    with or instead of ``LOCGAU`` for the OCT search.

    Verified against the installed NonLinLoc (docs/nlloc_interface.md §5):
    ``LOCGAU2 SigmaTfraction SigmaTmin SigmaTmax`` -- **correction**: the
    original guess had a third field ``corr_len_km``, but LOCGAU2 has no
    correlation-length parameter at all (unlike LOCGAU); its third
    parameter is ``SigmaTmax``, the maximum travel-time error in seconds.
    Renamed accordingly.
    """

    sigma_time_fraction: float = 0.01
    sigma_time_min_s: float = 0.05
    sigma_time_max_s: float = 2.0


class NllocLocMethConfig(_StrictModel):
    """NLLoc ``LOCMETH`` (location method) parameters.

    ``LOCMETH method maxDistStaGrid minNumberPhases maxNumberPhases
    minNumberSphases VpVsRatio maxNum3DGridMemory minDistStaGrid
    iRejectDuplicateArrivals`` (docs/nlloc_interface.md §4) -- no
    ``openmeq.config`` model existed for this statement before M6.1
    (docs/nlloc_interface.md §8's flagged follow-up). Defaults match the
    values verified end-to-end in the M5.1 spike
    (``tests/test_nlloc_spike.py``) and already hardcoded in
    ``openmeq.nlloc_setup._obspyck_nlloc_control``'s own control file
    (``LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1.0 6 -1.0 1``):
    ``vp_vs_ratio=-1.0`` so NLLoc uses the real S travel-time grids
    Grid2Time built (this project always has real per-station S grids,
    ARCHITECTURE.md §6d) rather than deriving S times from a fixed Vp/Vs
    ratio applied to the P grid.
    """

    method: Literal["GAU_ANALYTIC", "EDT", "EDT_OT_WT", "EDT_OT_WT_ML"] = "EDT_OT_WT"
    max_dist_sta_grid: float = 9999.0
    min_number_phases: int = 4
    max_number_phases: int = -1
    min_number_s_phases: int = -1
    vp_vs_ratio: float = -1.0
    max_num_3d_grid_memory: int = 6
    min_dist_sta_grid: float = -1.0
    reject_duplicate_arrivals: bool = True


class NllocConfig(_StrictModel):
    velocity_model: NllocVelocityModelConfig
    grid: NllocGridConfig
    filter: NllocFilterConfig = Field(default_factory=NllocFilterConfig)
    # M5.2 / OpenMEQ-mev.2: root directory ObsPyck's setup_external_programs
    # expects (a VELOCITY_MODELS dir as a sibling of per-program plugin dirs).
    obspyck_pluginpath: Path = Path("./external/obspyck_plugin")
    # M6.1 / OpenMEQ-xdf.1: per-event NLLoc runner control-file parameters.
    locsearch: NllocLocSearchConfig = Field(default_factory=NllocLocSearchConfig)
    locgau: NllocLocGauConfig = Field(default_factory=NllocLocGauConfig)
    locgau2: NllocLocGau2Config = Field(default_factory=NllocLocGau2Config)
    locmeth: NllocLocMethConfig = Field(default_factory=NllocLocMethConfig)
    # Ascending time_errors.uncertainty (s) bucket edges mapping to NLLoc
    # LOCQUAL2ERR quality classes 0..N. **Correction (docs/nlloc_interface.md
    # §5, LOCQUAL2ERR)**: this statement is "required, non-repeatable, for
    # phase/observation file formats that do not include time uncertainties;
    # ignored, non-repeatable, otherwise". OpenMEQ's NLLoc runner (M6.1) will
    # always write LOCFILES obsFileType NLLOC_OBS via
    # ``Catalog.write(..., format="NLLOC_OBS")``, and that format writes a
    # literal numeric ``Err`` field per pick (obspy always emits ``GAU
    # <uncertainty>``, never an integer quality code) -- so for OpenMEQ's
    # pipeline NLLoc will always ignore this statement's values and use the
    # per-pick uncertainty directly. It must still be present and
    # syntactically valid (NLLoc rejects a missing required statement), so
    # M6.1 must still emit some ascending list here; the values themselves
    # have no effect on the location. The last bucket edge is conventionally
    # a very large sentinel (NonLinLoc's own samples use ``99999.9``) for the
    # "reject this phase" quality class, kept here for that convention only.
    locqual2err_thresholds_s: list[float] = Field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 99999.9]
    )


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


class ReviewConfig(_StrictModel):
    enabled: bool = False
    obspyck_config: Path = Path("./config/obspyck.cfg")


# ---------------------------------------------------------------------------
# magnitude
# ---------------------------------------------------------------------------


class HuttonBooreConfig(_StrictModel):
    """Hutton & Boore (1987) ML relation coefficients.

    ML = log10(A_mm) + a*log10(R/100) + b*(R-100) + c
    """

    a: float = 1.110
    b: float = 0.00189
    c: float = 3.0


class LocalMlCoefficientsConfig(_StrictModel):
    """Same 3-term form as :class:`HuttonBooreConfig`, for a locally
    calibrated relation. Legitimately all-``None`` when the local relation
    instead uses a distance/-logA0 table (``local.file``)."""

    a: float | None = None
    b: float | None = None
    c: float | None = None


class LocalMlConfig(_StrictModel):
    coefficients: LocalMlCoefficientsConfig = Field(default_factory=LocalMlCoefficientsConfig)
    # CSV, header "distance_km,neg_log_a0", distance ascending, >= 2 rows;
    # ML = log10(A_mm) + neg_log_a0(R) after linear interpolation (clamped
    # outside the table's range). See docs/ARCHITECTURE.md's local ML
    # calibration file format section (added by M8.2 / OpenMEQ-625.2).
    file: Path | None = None


class MlConfig(_StrictModel):
    wood_anderson: bool = True
    amplitude_window: Literal["s"] = "s"
    # "horizontal" is currently the only supported value (max of the two
    # horizontals in the measurement window; automatically falls back to Z
    # if no horizontal channels are present for that station).
    component: Literal["horizontal"] = "horizontal"
    relation: Literal["hutton_boore", "local"] = "hutton_boore"
    hutton_boore: HuttonBooreConfig = Field(default_factory=HuttonBooreConfig)
    local: LocalMlConfig = Field(default_factory=LocalMlConfig)
    # Either an inline {"NET.STA": correction} mapping, or a CSV path with
    # header "net,sta,correction". A station absent from either defaults to
    # a 0.0 correction.
    station_corrections: dict[str, float] | Path = Field(default_factory=dict)
    # M8.1 / OpenMEQ-625.1: Wood-Anderson simulation and amplitude-window
    # parameters (PLAN.md §3's sketch omits all five of these).
    # wa_sensitivity: the historical/obspy-default constant is 2800, but
    # Uhrhammer & Collins (1990) / IASPEI recommend 2080 (changes every ML by
    # log10(2800/2080) = +0.13) -- default to the IASPEI value.
    wa_sensitivity: float = 2080.0
    # Measurement window is [S_pick - window_before_s, S_pick + window_after_s];
    # falls back to [P_pick + 1.0, P_pick + 8.0] if no S pick exists.
    window_before_s: float = 1.0
    window_after_s: float = 4.0
    taper_pct: float = 0.05
    # 4-corner bandpass [f1, f2, f3, f4] Hz passed to trace.remove_response's
    # pre_filt (cosine taper corners: stop, pass, pass, stop). Conservative
    # broadband default; override per-project/per-network if a channel's
    # usable passband differs.
    # TODO(OpenMEQ-625.1): confirm these corners are appropriate once real
    # station response curves are available; PLAN.md's sketch does not
    # specify them.
    pre_filt: tuple[float, float, float, float] = (0.5, 1.0, 45.0, 49.0)


class MagnitudeConfig(_StrictModel):
    type: Literal["ml"] = "ml"
    ml: MlConfig = Field(default_factory=MlConfig)
    min_stations: int = 2
    statistic: Literal["median", "mean"] = "median"


# ---------------------------------------------------------------------------
# hypodd
# ---------------------------------------------------------------------------


class Simul2000RaytraceConfig(_StrictModel):
    """SIMUL2000 ray-tracing parameters for hypoDD.inp's IMOD=9 (3D) block:
    ``IPHA NDIP ISKIP SCALE1 SCALE2 XFAC TLIM NITPB``.

    M9.3 / OpenMEQ-6ld.3's second confirmation pass (called for by the M9
    epic design note's original ``TODO`` here) confirms this field order
    against ``src/hypoDD/get_vel3d.f``'s subroutine signature
    (``fn_vel3d,iphase,lat_3d,lon_3d,rot_3d,indip,iiskip,iscale1,iscale2,
    ixfac,itlim,initpb,log``), ``src/hypoDD/hypoDD.f:449-452``'s call site
    (same variables, same order), and ``src/hypoDD/getinp2.f:236-237``'s
    ``hypoDD.inp`` read (``ipha3d,ndip,iskip,scale1,scale2,xfac,tlim,
    nitpb``) -- all three agree, and this field order already matched
    before the confirmation pass; see
    `openmeq.relocate.hypodd_files.write_hypodd_inp_3d`'s docstring for the
    full citation trail. The defaults below (taken from
    ``examples/example1/hypoDD.inp``'s commented-out, non-authoritative but
    corroborating ``IMOD=9`` block, ``"2 9 4 1 1 1.35 0.0005 50"``) are
    confirmed final.
    """

    iphase: int = 2
    ndip: int = 9
    iskip: int = 4
    scale1: float = 1.0
    scale2: float = 1.0
    xfac: float = 1.35
    tlim: float = 0.0005
    nitpb: int = 50


class HypoddVelocityModelConfig(_StrictModel):
    type: Literal["1d", "3d"] = "1d"
    file: Path
    # M9.3 / OpenMEQ-6ld.3: only used when type == "3d" (hypoDD.inp's
    # "LAT_3D LON_3D ROT_3D" line: SIMUL2000 model origin and the rotation,
    # in degrees, of the Cartesian coordinate system, positive=anticlockwise).
    origin_3d: tuple[float, float] | None = None
    rotation_3d_deg: float | None = None
    simul2000: Simul2000RaytraceConfig | None = None


class XcorrFilterConfig(_StrictModel):
    """Bandpass applied once when building the cross-correlation waveform
    cache (M10.1 / OpenMEQ-16d.1); PLAN.md §3's sketch does not show an
    explicit xcorr filter band.

    Resolved by M10.3 / OpenMEQ-16d.3 (``openmeq.relocate.xcorr`` module
    docstring, "The xcorr filter band"): kept at 2-20 Hz rather than the
    5-40 Hz alternative suggested for microseismic events, since an
    empirical Monte-Carlo comparison against this project's own synthetic
    P/S pulse spectra (``tests/data/make_synthetic.py``, P dominant 10 Hz, S
    dominant 6 Hz) found no lag-recovery accuracy benefit from the narrower
    5-40 Hz band, while its low corner sits close enough to the synthetic S
    energy's 6 Hz peak to risk attenuating real S arrivals with a similarly
    low corner frequency. Still fully project-overridable for a deployment
    with a genuinely higher-frequency source spectrum.
    """

    freqmin: float = 2.0
    freqmax: float = 20.0


class XcorrConfig(_StrictModel):
    window_p: tuple[float, float] = (-0.05, 0.2)
    window_s: tuple[float, float] = (-0.05, 0.3)
    min_cc: float = 0.7
    maxlag_s: float = 0.05
    components: dict[str, list[str]] = Field(
        default_factory=lambda: {"P": ["Z"], "S": ["1", "2", "N", "E"]}
    )
    filter: XcorrFilterConfig = Field(default_factory=XcorrFilterConfig)
    # M10.1 / OpenMEQ-16d.1 (per the M0.3 note appended to that issue: no
    # config key existed yet for the xcorr waveform cache's common resample
    # rate). Every cached window (``relocate/cache/<event_id>.npz``) is
    # resampled to this single project-wide rate, regardless of a channel's
    # native rate -- deliberately a fixed, explicit target rather than a
    # per-event "highest native rate seen" choice, since M10.3 cross-
    # correlates pairs drawn from *different* events' caches, and those must
    # already share one common rate (an auto-detected per-event rate could
    # legitimately differ event to event, re-introducing the very
    # mismatched-sampling-rate problem this cache is meant to remove --
    # docs/hypoddpy_audit.md defect #8). Default 100 Hz mirrors
    # ``phasenet.target_rate_hz``'s existing convention and comfortably
    # exceeds twice ``filter.freqmax``'s default (20 Hz) for a project
    # recording at 200 Hz, per ``tests/data/make_synthetic.py``.
    resample_rate_hz: float = 200.0  # typical native rate; do not downsample xcorr data


class Ph2dtConfig(_StrictModel):
    maxsep_km: float = 1.0
    maxngh: int = 10
    minlnk: int = 8
    # M9.2 / OpenMEQ-6ld.2: extends PLAN.md §3's hypodd.ph2dt block. Defaults
    # are ph2dt.inp's own bracketed-comment defaults (confirmed against
    # HYPODD.zip), not hypoddpy template's ad-hoc choices.
    minwght: float = 0.0
    maxdist_km: float = 200.0
    minobs: int = 8
    maxobs: int = 20


class WeightingRow(_StrictModel):
    """One row of hypoDD.inp's iterative data-weighting/re-weighting table.

    ``niter`` is the last iteration to use this row's weights; ``wtccp``/
    ``wtccs`` weight cross-correlation P/S data, ``wtctp``/``wtcts`` weight
    catalog P/S data, ``wrcc``/``wrct`` are residual thresholds (s) and
    ``wdcc``/``wdct`` max linked-pair distances (km) for cross/catalog data
    respectively, and ``damp`` is the LSQR damping factor.
    """

    niter: int
    wtccp: float
    wtccs: float
    wrcc: float
    wdcc: float
    wtctp: float
    wtcts: float
    wrct: float
    wdct: float
    damp: float


def _default_weighting_rows() -> list[WeightingRow]:
    """The 5-row weighting scheme from examples/example1/hypoDD.inp (verbatim
    values), per M9.1's audit recommendation against copying hypoddpy's
    minimal 2-row default."""
    return [
        WeightingRow(
            niter=1,
            wtccp=0.01,
            wtccs=0.005,
            wrcc=-999,
            wdcc=-999,
            wtctp=1.00,
            wtcts=0.5,
            wrct=20,
            wdct=-999,
            damp=20,
        ),
        WeightingRow(
            niter=3,
            wtccp=0.01,
            wtccs=0.005,
            wrcc=-999,
            wdcc=-999,
            wtctp=1.00,
            wtcts=0.5,
            wrct=10,
            wdct=4,
            damp=20,
        ),
        WeightingRow(
            niter=3,
            wtccp=1,
            wtccs=0.5,
            wrcc=6,
            wdcc=3,
            wtctp=1.00,
            wtcts=0.5,
            wrct=10,
            wdct=3,
            damp=20,
        ),
        WeightingRow(
            niter=3,
            wtccp=1,
            wtccs=0.5,
            wrcc=5,
            wdcc=2,
            wtctp=0.1,
            wtcts=0.05,
            wrct=9,
            wdct=2,
            damp=20,
        ),
        WeightingRow(
            niter=3,
            wtccp=1,
            wtccs=0.5,
            wrcc=4,
            wdcc=1,
            wtctp=0.01,
            wtcts=0.005,
            wrct=8,
            wdct=2,
            damp=20,
        ),
    ]


class EventClusteringConfig(_StrictModel):
    """hypoDD.inp's event-clustering line: ``OBSCC OBSCT MINDIST MAXDIST
    MAXGAP`` (confirmed ``examples/example1/hypoDD.inp:41-42``,
    ``getinp2.f``'s ``minobs_cc,minobs_ct,minds,maxds,maxgap`` read).

    docs/hypoddpy_audit.md §6 defect #14: hypoddpy's template hardcodes
    ``OBSCC=OBSCT=0``, silently disabling hypoDD's event-clustering feature
    entirely, even though ``examples/example1`` itself ships with
    ``OBSCT=8`` and clustering active. M9.2 exposes this block as config
    instead of hardcoding it either way.
    """

    obscc: int = 0
    #: example1's own working value (docs/hypoddpy_audit.md defect #14),
    #: used as this project's default rather than hypoddpy's silent 0.
    obsct: int = 8
    mindist_km: float = -999
    maxdist_km: float = -999
    maxgap_deg: float = -999


class HypoddConfig(_StrictModel):
    velocity_model: HypoddVelocityModelConfig
    xcorr: XcorrConfig = Field(default_factory=XcorrConfig)
    ph2dt: Ph2dtConfig = Field(default_factory=Ph2dtConfig)
    # M9.2 / OpenMEQ-6ld.2: hypoDD.inp's weighting table and solution-control
    # block. Defaults: ISTART=2 (locate from network sources), ISOLV=2
    # (LSQR -- example1 itself uses ISOLV=1/SVD, but LSQR scales to much
    # larger datasets and is this project's documented choice), IAQ=1
    # (remove airquakes), matching the M9 epic design note's recommendation
    # over hypoddpy's unexplained defaults.
    #
    # M9.5 / OpenMEQ-6ld.5 (docs/hypoddpy_audit.md defect #4): the currently
    # documented IAQ values (hypoDD2_UserGuide.pdf §B.3.1) are 1 = remove
    # air-quakes / 2 = keep air-quakes (reset depths, continue) -- M9.2's
    # original `Literal[0, 1]` predates that audit finding and only allowed
    # the wrong pair. `0` is kept too: it remains a functional, if
    # undocumented, legacy value in the compiled source (`hypoDD.f:803`,
    # `iaq.eq.0` --> don't remove air-quakes at all) that a synthetic/test
    # dataset may legitimately want (a small network's brief mid-iteration
    # negative-depth excursions can otherwise shrink a cluster below
    # hypoDD's own "less than 2 events" cutoff).
    weighting_rows: list[WeightingRow] = Field(default_factory=_default_weighting_rows)
    istart: Literal[1, 2] = 2
    isolv: Literal[1, 2] = 2
    iaq: Literal[0, 1, 2] = 1
    # M9.2 / OpenMEQ-6ld.2: hypoDD.inp's "IDAT IPHA DIST" data-type-selection
    # line (confirmed examples/example1/hypoDD.inp:32-33, getinp2.f's
    # idata,iphase,maxdist read). Defaults match example1: both cross-corr
    # and catalog data, P&S phases, 200 km per ph2dt.inp's own MAXDIST
    # bracketed-comment default (example1 itself uses 120 for its smaller
    # aperture network).
    idat: Literal[1, 2, 3] = 3
    ipha: Literal[1, 2, 3] = 3
    max_station_dist_km: float = 200.0
    clustering: EventClusteringConfig = Field(default_factory=EventClusteringConfig)
    # M10.4 / OpenMEQ-16d.4: opt-in cluster-level parallelism (run each
    # ph2dt cluster as an independent hypoDD subprocess). Off by default;
    # only worth enabling once the single-run path is solid.
    parallel_clusters: bool = False


# ---------------------------------------------------------------------------
# top-level Config
# ---------------------------------------------------------------------------


class Config(_StrictModel):
    project: ProjectConfig
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    data: DataConfig
    stations: StationsConfig = Field(default_factory=StationsConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    chunks: ChunksConfig = Field(default_factory=ChunksConfig)
    phasenet: PhasenetConfig = Field(default_factory=PhasenetConfig)
    nlloc: NllocConfig
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    magnitude: MagnitudeConfig = Field(default_factory=MagnitudeConfig)
    hypodd: HypoddConfig

    @model_validator(mode="after")
    def _resolve_relative_paths(self, info) -> Config:
        """Resolve every relative path field against the config file's
        directory (ARCHITECTURE.md §2), supplied via ``model_validate``'s
        ``context={"base_dir": ...}`` (see :func:`load_config`).

        This is the single place that knows every path-like field in the
        schema, per this task's design decision (see module docstring).
        When no ``base_dir`` is in context (e.g. constructing a ``Config``
        directly in a test without going through ``load_config``), paths are
        left exactly as given.
        """
        base_dir = info.context.get("base_dir") if isinstance(info.context, dict) else None
        if base_dir is None:
            return self
        base_dir = Path(base_dir)

        self.project.output_dir = _resolve(self.project.output_dir, base_dir)
        self.data.waveform_dirs = [_resolve(p, base_dir) for p in self.data.waveform_dirs]
        self.stations.stationxml = _resolve(self.stations.stationxml, base_dir)
        self.stations.stationxml_dir = _resolve(self.stations.stationxml_dir, base_dir)
        self.stations.csv = _resolve(self.stations.csv, base_dir)
        self.nlloc.velocity_model.file = _resolve(self.nlloc.velocity_model.file, base_dir)
        self.nlloc.obspyck_pluginpath = _resolve(self.nlloc.obspyck_pluginpath, base_dir)
        self.review.obspyck_config = _resolve(self.review.obspyck_config, base_dir)
        self.magnitude.ml.local.file = _resolve(self.magnitude.ml.local.file, base_dir)
        if isinstance(self.magnitude.ml.station_corrections, Path):
            self.magnitude.ml.station_corrections = _resolve(
                self.magnitude.ml.station_corrections, base_dir
            )
        self.hypodd.velocity_model.file = _resolve(self.hypodd.velocity_model.file, base_dir)
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file, returning a typed :class:`Config`.

    Relative paths in the YAML are resolved against ``path``'s own directory
    (ARCHITECTURE.md §2), not the process's current working directory.

    Raises
    ------
    ConfigError
        If the file cannot be parsed as YAML, or fails schema validation.
        Every underlying pydantic error is logged (with its field path)
        before the aggregated ``ConfigError`` is raised, so a user sees every
        problem at once rather than fixing one key at a time.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: top-level YAML content must be a mapping, got {type(raw).__name__}"
        )

    base_dir = path.resolve().parent
    try:
        return Config.model_validate(raw, context={"base_dir": base_dir})
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"]) or "<root>"
            logger.error("%s: config validation error at %s: %s", path, loc, error["msg"])
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc
