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
  project base directory: ``project.base_dir`` if set (itself resolved
  against the config file's directory when relative), otherwise the
  *config file's own directory* (never the process CWD), per ARCHITECTURE.md
  §2. A leading ``~`` is expanded in every path. This is implemented in one place: :meth:`Config._resolve_relative_paths`,
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

    A leading ``~`` is expanded first. An absolute path is returned
    unchanged. ``None`` passes through unchanged (many path fields are
    optional).
    """
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


class ProjectConfig(_StrictModel):
    name: str
    output_dir: Path
    # OpenMEQ-e8x: root every other relative path in the config resolves
    # against. Relative base_dir resolves against the config file's
    # directory; None = the config file's directory itself.
    base_dir: Path | None = None


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


class QcNoiseConfig(_StrictModel):
    """Hourly-window band-limited noise statistics (OpenMEQ-h8j, ``openmeq.qc.noise``).

    Defaults match MicroQuake's ``plot_daily_noise_stats.py``.
    """

    # Cross-station comparable band, Hz: above the corner of every common
    # short-period sensor (GS-11D 4.5 Hz, L4C/LE-3Dlite 1 Hz), so a scalar
    # sensitivity is enough to compare sensors of one family.
    core_band: tuple[float, float] = (5.0, 25.0)
    # Low band, Hz: only feeds the low/core ratio, a per-station drift tracker.
    low_band: tuple[float, float] = (1.0, 5.0)
    window_s: float = 3600.0
    # A window counts if at least this fraction of its samples is present.
    min_window_frac: float = 0.5
    # A window whose core-band RMS (counts) is below this is dead/flatlined.
    dead_rms_counts: float = 0.5

    @model_validator(mode="after")
    def _check_bands(self) -> QcNoiseConfig:
        for name in ("core_band", "low_band"):
            lo, hi = getattr(self, name)
            if not 0 < lo < hi:
                raise ValueError(f"qc.noise.{name} must be [low, high] with 0 < low < high")
        if not 0 < self.window_s <= 86400 or 86400 % self.window_s:
            raise ValueError("qc.noise.window_s must divide a day (86400 s)")
        return self


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
    # OpenMEQ-h8j: hourly noise statistics and the per-instrument noise plots.
    noise: QcNoiseConfig = Field(default_factory=QcNoiseConfig)
    # Instrument group -> stations ("STA", "NET.STA" or a full seed id), for
    # the noise plots. Overrides the sensor model read from StationXML; any
    # station not listed keeps its StationXML group.
    instrument_groups: dict[str, list[str]] = Field(default_factory=dict)


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
    # OpenMEQ-8a0: obspy coincidence_trigger's thr_coincidence_sum -- counts
    # triggering *traces* (channels, weighted by station_weights), not stations:
    # a 3-C station contributes up to 3. Formerly named coincidence_sum.
    min_trace_coincidence: float = 4
    # OpenMEQ-8a0: drop a coincidence trigger whose distinct NET.STA count is
    # below this. 1 = no station filter.
    min_stations: int = 1
    station_weights: dict[str, float] = Field(default_factory=dict)
    # M2.1 / OpenMEQ-ujx.1: padding read on each side of a per-day detection
    # window so a trigger isn't lost at a day boundary; must comfortably
    # exceed lta plus a trigger's typical duration.
    day_overlap_s: float = 60.0
    # M2.2 / OpenMEQ-ujx.2: two triggers within this many seconds of each
    # other (greedy chained clustering) are merged into one detection.
    # Must exceed the max S-P time plus trigger length so P and S triggers of one
    # event merge into a single detection (STA/LTA often fires on both).
    dedup_window_s: float = 10.0

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_keys(cls, data):
        if isinstance(data, dict) and "coincidence_sum" in data:
            raise ValueError(
                "detection.coincidence_sum was renamed to detection.min_trace_coincidence "
                "(it counts triggering traces/channels, not stations; a 3-C station counts up "
                "to 3). Use detection.min_stations for a station-count threshold."
            )
        return data


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------


class ChunksConfig(_StrictModel):
    pre_s: float = 30.0
    post_s: float = 30.0
    # M3.2 / OpenMEQ-rq8.2: memory fix for M3.1's chunk_day_group, which
    # originally read the whole span from a day's first to its last
    # detection, for every channel, in one go -- on a real network (many
    # channels x a full day at a high sampling rate) that single read could
    # run out of memory. A day group is now split into sub-windows wherever
    # two consecutive detections' trigger times are more than
    # max_read_window_s apart (openmeq.chunk._split_detections_into_subwindows),
    # keeping the "one read per sub-window" property while bounding any
    # single read's time span. 600 s (10 min) comfortably covers a cluster
    # of nearby detections (e.g. an aftershock sequence) in one read while
    # still splitting a day's worth of sparse, spread-out detections into
    # many small reads.
    max_read_window_s: float = 600.0


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
    # OpenMEQ-9kn: how Pick.time_errors.uncertainty is set. "peak_width":
    # half-width of the pick's PhaseNet probability peak (result.h5) at
    # pick_uncertainty_peak_frac * peak, clamped to
    # [pick_uncertainty_min_s, pick_uncertainty_max_s]. "fixed": every pick
    # gets pick_uncertainty_s.
    pick_uncertainty_method: Literal["peak_width", "fixed"] = "peak_width"
    # M4.4 / OpenMEQ-1m7.4: fixed uncertainty; also the peak_width fallback
    # when result.h5 or its station column can't be used.
    pick_uncertainty_s: float = 0.1
    pick_uncertainty_peak_frac: float = Field(default=0.5, gt=0.0, lt=1.0)
    pick_uncertainty_min_s: float = Field(default=0.02, gt=0.0)
    pick_uncertainty_max_s: float = 0.5
    # OpenMEQ-nzf: keep only picks within [before, after] seconds of the
    # detection trigger time (before <= 0 <= after typical); null keeps the
    # whole chunk. PhaseNet fires on the processed chunk's start (taper +
    # high-pass), and the per-station dedup keeps the highest-probability
    # pick anywhere, so without a window those artefacts replace real picks.
    pick_window_s: tuple[float, float] | None = (-5.0, 15.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> PhasenetConfig:
        if self.pick_window_s is not None and self.pick_window_s[0] >= self.pick_window_s[1]:
            raise ValueError(
                f"phasenet.pick_window_s must be [before, after] with before < after, "
                f"got {list(self.pick_window_s)}"
            )
        if self.pick_uncertainty_min_s > self.pick_uncertainty_max_s:
            raise ValueError(
                "phasenet.pick_uncertainty_min_s must be <= pick_uncertainty_max_s, got "
                f"{self.pick_uncertainty_min_s} > {self.pick_uncertainty_max_s}"
            )
        return self


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
    """Event-level accept/reject thresholds, applied after pass A/B.

    ``max_gap_deg`` (M11.2 / OpenMEQ-m1f.2): raised from an earlier 270.0 to
    285.0. On ``tests/data/make_synthetic.py``'s own 8-station, ~13 km-
    aperture network (event offsets drawn up to 9 km from the array centre
    in each of x/y, so some fall near or outside the stations' convex
    hull), a full deterministic ``openmeq run`` over the 20-event synthetic
    truth catalogue located every single event with all 8 stations'
    P *and* S picks (``locate/summary.csv``'s ``n_p``/``n_s`` == 8
    throughout, and every ``err_z_km`` well inside ``max_err_z_km``;
    ``err_h_km`` itself is left empty by NLLoc's OCT search for every event
    in this dataset, so that check never fires either way -- see
    ``openmeq/locate/finalize.py``'s "a missing value is never counted as a
    failure" rule) -- i.e. every event was fully, tightly constrained -- yet
    two of them (azimuthal gap 270.3 deg and 270.6 deg, against a
    continuous 75-269 deg spread for the other 18) were rejected purely for
    sitting a fraction of a degree past the old 270.0 deg cutoff. That is a
    network-geometry artifact of a small, roughly ring-shaped station
    layout (any event outside the ring inherently has less azimuthal
    coverage), not evidence of a poorly constrained location -- the
    location-quality criteria (``max_err_h_km``/``max_err_z_km``) already
    gate on solution quality directly. 285.0 keeps a ~15 deg margin above
    the largest gap observed in that full run, so it still rejects a
    genuinely one-sided/poorly-surrounded geometry without discarding
    otherwise fully-constrained events at this network's edge.
    """

    min_p_phases: int = 4
    min_phases: int = 6
    min_s_phases: int = 1
    max_rms_s: float = 0.1
    max_gap_deg: float = 285.0
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
    # OpenMEQ-tch: stations ("STA" or "NET.STA") whose amplitudes are still
    # measured and written to station_magnitudes.csv (excluded=True) but are
    # never used in the event magnitude -- e.g. a station with a bad or
    # unknown response.
    exclude_stations: list[str] = Field(default_factory=list)


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
    # OpenMEQ-6y1: formerly hard-coded in openmeq.relocate.waveform_cache.
    corners: int = Field(default=4, ge=1)
    zerophase: bool = True

    @model_validator(mode="after")
    def _check_band(self) -> XcorrFilterConfig:
        if not 0 < self.freqmin < self.freqmax:
            raise ValueError("hypodd.xcorr.filter needs 0 < freqmin < freqmax")
        return self


class XcorrConfig(_StrictModel):
    # [start, end] of the correlation window, seconds relative to the pick
    # (signed: [-0.1, 0.1] = 100 ms before to 100 ms after). Defaults tuned on
    # RK-NM 2026 Q1 (OpenMEQ-9fh): most observations at the lowest hypoDD cc
    # residual among the windows/lags/thresholds compared.
    window_p: tuple[float, float] = (-0.1, 0.1)
    window_s: tuple[float, float] = (-0.1, 0.2)
    min_cc: float = Field(default=0.75, ge=-1.0, le=1.0)
    maxlag_s: float = Field(default=0.05, gt=0)
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
    # OpenMEQ-6y1: formerly hard-coded in openmeq.relocate.waveform_cache --
    # Hann taper fraction (each end) before filtering, and the extra data
    # (s) cached either side of window +/- maxlag to absorb filter edge effects.
    taper_pct: float = Field(default=0.05, ge=0.0, le=0.5)
    filter_edge_margin_s: float = Field(default=0.5, ge=0.0)
    # OpenMEQ-9fh: drop a channel's measurement when its best lag sits at the
    # edge of the +/- maxlag_s search (within half a sample): the true peak is
    # outside the search, so the lag is not a real correlation maximum.
    reject_edge_lags: bool = True

    @model_validator(mode="after")
    def _check_windows(self) -> XcorrConfig:
        for name in ("window_p", "window_s"):
            start, end = getattr(self, name)
            if not start < end:
                raise ValueError(f"hypodd.xcorr.{name} must be [start, end] with start < end")
        return self


class Ph2dtConfig(_StrictModel):
    # OpenMEQ-16d.6: scaled to a small-aperture (~10-15 km) microseismic
    # network rather than ph2dt.inp's own bracketed-comment defaults (a
    # regional-network convention -- MAXSEP [10 km], MAXDIST [200 km] --
    # tuned for networks with tens-to-hundreds-of-km station apertures; see
    # docs/hypodd_tuning.md). maxsep_km=10 keeps the user guide's own
    # unscaled "10 km is appropriate to start with in many regions" figure
    # rather than shrinking it further: this project's own 20-event/~15
    # km-aperture synthetic dataset has two events whose *only* neighbour
    # within 8 km of each other is each other (their next-nearest true
    # neighbour is ~8.6-8.9 km away) -- below 10 km, ph2dt gives that pair no
    # second link into the rest of the network and hypoDD's own OBSCT
    # clustering (below) splits them into an isolated 2-event cluster that
    # cannot be relocated (hypoDD's own "less than 2 events" cluster-size
    # floor). maxobs=16
    # is exactly 2 phases x 8 stations, i.e. "keep every possible pick" for a
    # small network (user guide: for a small, tightly-linked catalogue,
    # MAXOBS should be the number of stations, doubled here for P&S) rather
    # than the bracket default's 20 (silently tuned to fewer than 10
    # stations x 2 phases already, just not documented as such). minobs=1
    # (also the user guide's own small-network recommendation) keeps every
    # ph2dt-linked pair, even a weakly-linked one, since a small network
    # cannot afford to discard already-scarce inter-event links.
    maxsep_km: float = 10.0
    maxngh: int = 10
    minlnk: int = 8
    minwght: float = 0.0
    maxdist_km: float = 25.0
    minobs: int = 1
    maxobs: int = 16


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
    """A 5-row progressive weighting/re-weighting scheme, modelled on
    ``examples/example1/hypoDD.inp``'s own real scheme (loose-then-tight,
    per the user guide's own recommended structure) but with the ``WDCC``/
    ``WDCT`` distance cutoffs rescaled from example1's regional-network
    values (bottoming out at 1-2 km) to a small-aperture (~10-15 km)
    microseismic network (bottoming out at 6 km, about 40% of a ~15 km
    aperture -- comfortably above the ~5.2 km worst-case link this
    project's own 20-event synthetic dataset needs to keep every event
    connected through the final iteration; see docs/hypodd_tuning.md for the
    full derivation and how to rescale these for a different network).

    example1's own final-row cutoff of 1-2 km is a large fraction of *its*
    network's own inter-event spread, deliberately collapsing well-linked
    event pairs down to near-duplicate "repeating earthquake" resolution.
    Applied unscaled to a small-aperture network whose events are not
    confined to a sub-km patch, that same 1-2 km cutoff instead disconnects
    most pairs by the final iteration -- exactly the "most events end up as
    air-quakes" failure mode this issue was opened to fix. WRCC/WRCT (>=1,
    hypoDD's "multiply the current residual standard deviation" dynamic
    mode, not an absolute-seconds cutoff) are left at example1's own values:
    being relative to the data's own residual spread, they self-scale to
    whatever noise level the real data has and need no network-size
    rescaling.
    """
    return [
        WeightingRow(
            niter=5,
            wtccp=0.01,
            wtccs=0.005,
            wrcc=-999,
            wdcc=-999,
            wtctp=1.00,
            wtcts=0.5,
            wrct=-999,
            wdct=-999,
            damp=20,
        ),
        WeightingRow(
            niter=5,
            wtccp=0.01,
            wtccs=0.005,
            wrcc=-999,
            wdcc=-999,
            wtctp=1.00,
            wtcts=0.5,
            wrct=10,
            wdct=10,
            damp=20,
        ),
        WeightingRow(
            niter=5,
            wtccp=1,
            wtccs=0.5,
            wrcc=6,
            wdcc=8,
            wtctp=1.00,
            wtcts=0.5,
            wrct=8,
            wdct=8,
            damp=20,
        ),
        WeightingRow(
            niter=5,
            wtccp=1,
            wtccs=0.5,
            wrcc=5,
            wdcc=6,
            wtctp=0.1,
            wtcts=0.05,
            wrct=6,
            wdct=6,
            damp=20,
        ),
        WeightingRow(
            niter=5,
            wtccp=1,
            wtccs=0.5,
            wrcc=4,
            wdcc=6,
            wtctp=0.01,
            wtcts=0.005,
            wrct=6,
            wdct=6,
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
    # idata,iphase,maxdist read). Both cross-corr and catalog data, P&S
    # phases. OpenMEQ-16d.6: DIST (max cluster-centroid-to-station distance)
    # rescaled from 200 km (a regional-network value, example1 itself already
    # uses a smaller 120 km) to 25 km -- comfortably above this project's own
    # ~15 km-aperture synthetic network's worst-case ~17.5 km event-station
    # hypocentral distance, but tight enough that a station well outside the
    # monitored volume (e.g. a shared regional network station on a
    # multi-site deployment) is excluded rather than silently diluting the
    # local solution. See docs/hypodd_tuning.md.
    idat: Literal[1, 2, 3] = 3
    ipha: Literal[1, 2, 3] = 3
    max_station_dist_km: float = 25.0
    clustering: EventClusteringConfig = Field(default_factory=EventClusteringConfig)
    # M10.4 / OpenMEQ-16d.4: opt-in cluster-level parallelism (run each
    # ph2dt cluster as an independent hypoDD subprocess). Off by default;
    # only worth enabling once the single-run path is solid.
    parallel_clusters: bool = False


# ---------------------------------------------------------------------------
# diagnostics (OpenMEQ-02o: optional extended QAQC figures)
# ---------------------------------------------------------------------------


#: Steps with an opt-in extended-QAQC figure set (``qc`` is itself QAQC and
#: has none; ``run`` is the cross-step ``openmeq run`` summary).
DIAGNOSTIC_STEPS: tuple[str, ...] = (
    "detect",
    "chunk",
    "pick",
    "locate",
    "magnitude",
    "relocate",
    "run",
)


class StepDiagnosticsConfig(_StrictModel):
    # None = inherit ``diagnostics.extended``; true/false overrides it for this step.
    extended: bool | None = None


class PickZoomConfig(_StrictModel):
    half_window_s: float = Field(default=2.0, gt=0)
    n_events: int = Field(default=10, ge=1)
    show_rejected: bool = False


class PickDiagnosticsConfig(StepDiagnosticsConfig):
    pick_zoom: PickZoomConfig = Field(default_factory=PickZoomConfig)


class XcorrPlotsConfig(_StrictModel):
    # Per-station/phase obspy xcorr_pick_correction plots for every
    # ``pair_interval``-th ph2dt-linked pair (dt.ct order), at most
    # ``max_pairs`` pairs -> relocate/diagnostics/xcorr_pairs/.
    enabled: bool = False
    pair_interval: int = Field(default=100, ge=1)
    max_pairs: int = Field(default=20, ge=1)


class RelocateDiagnosticsConfig(StepDiagnosticsConfig):
    xcorr_plots: XcorrPlotsConfig = Field(default_factory=XcorrPlotsConfig)


class WellsConfig(_StrictModel):
    """OpenMEQ-7zt: well heads/tracks overlaid on every map and depth-section
    diagnostic figure. ``dir`` holds ``<Group>_well_heads.csv``
    (``name,lon,lat``) and ``<Group>_well_tracks.csv`` (``name,lon,lat,
    elevation_m``, elevation positive up relative to sea level), headerless
    -- see :mod:`openmeq.wells`."""

    enabled: bool = False
    dir: Path | None = None
    # [] = every group found in ``dir``; otherwise only these group prefixes.
    groups: list[str] = Field(default_factory=list)
    # {group: matplotlib colour}; unset groups get openmeq.wells' defaults.
    colors: dict[str, str] = Field(default_factory=dict)
    labels: bool = False

    @model_validator(mode="after")
    def _require_dir(self) -> WellsConfig:
        if self.enabled and self.dir is None:
            raise ValueError("diagnostics.wells.enabled is true but diagnostics.wells.dir is unset")
        return self


class DiagnosticsConfig(_StrictModel):
    """Opt-in extended QAQC figures, on top of every step's always-on
    baseline ``<step>/diagnostics/`` set. Extended figures go to
    ``<step>/diagnostics/extended/``."""

    extended: bool = False
    n_examples: int = Field(default=10, ge=1)
    wells: WellsConfig = Field(default_factory=WellsConfig)
    detect: StepDiagnosticsConfig = Field(default_factory=StepDiagnosticsConfig)
    chunk: StepDiagnosticsConfig = Field(default_factory=StepDiagnosticsConfig)
    pick: PickDiagnosticsConfig = Field(default_factory=PickDiagnosticsConfig)
    locate: StepDiagnosticsConfig = Field(default_factory=StepDiagnosticsConfig)
    magnitude: StepDiagnosticsConfig = Field(default_factory=StepDiagnosticsConfig)
    relocate: RelocateDiagnosticsConfig = Field(default_factory=RelocateDiagnosticsConfig)
    run: StepDiagnosticsConfig = Field(default_factory=StepDiagnosticsConfig)

    def extended_enabled(self, step: str) -> bool:
        """Whether ``step``'s extended QAQC figures should be written: the
        step's own ``extended`` if set, else the global ``extended``."""
        if step not in DIAGNOSTIC_STEPS:
            raise ValueError(
                f"unknown diagnostics step {step!r} (choices: {', '.join(DIAGNOSTIC_STEPS)})"
            )
        override = getattr(self, step).extended
        return self.extended if override is None else override


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
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    @model_validator(mode="after")
    def _resolve_relative_paths(self, info) -> Config:
        """Resolve every relative path field against the project base
        directory (ARCHITECTURE.md §2).

        The base is ``project.base_dir`` when set, else the config file's
        directory, which arrives via ``model_validate``'s
        ``context={"base_dir": ...}`` (see :func:`load_config`). A relative
        ``project.base_dir`` is itself resolved against the config file's
        directory.

        This is the single place that knows every path-like field in the
        schema, per this task's design decision (see module docstring).
        When there is no config directory in context (e.g. constructing a
        ``Config`` directly in a test without going through
        ``load_config``), only an absolute ``project.base_dir`` is applied;
        otherwise paths are left exactly as given.
        """
        config_dir = info.context.get("base_dir") if isinstance(info.context, dict) else None
        project_base = self.project.base_dir
        if project_base is not None:
            project_base = project_base.expanduser()
            if not project_base.is_absolute():
                if config_dir is None:
                    return self
                project_base = (Path(config_dir) / project_base).resolve()
            self.project.base_dir = project_base
            base_dir = project_base
        elif config_dir is None:
            return self
        else:
            base_dir = Path(config_dir)

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
        self.diagnostics.wells.dir = _resolve(self.diagnostics.wells.dir, base_dir)
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file, returning a typed :class:`Config`.

    Relative paths in the YAML are resolved against ``project.base_dir`` if
    set, else ``path``'s own directory (ARCHITECTURE.md §2), never the
    process's current working directory.

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
