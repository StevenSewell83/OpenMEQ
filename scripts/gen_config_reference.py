#!/usr/bin/env python
"""Generate ``docs/config_reference.md`` from ``openmeq.config.Config``.

M11.4 / OpenMEQ-m1f.4. Every YAML key ``config/example.yaml`` documents by
comment is also validated by a pydantic model in ``openmeq/config.py``
(ARCHITECTURE.md §2's ``openmeq.config`` contract), so the field *names*,
*types* and *defaults* below are read directly off the live schema
(``Config.model_fields``, recursed into every nested section model) rather
than hand-copied -- they cannot silently drift from what ``load_config``
actually accepts.

``openmeq/config.py`` does not use pydantic's ``Field(description=...)``
(confirmed: ``grep -c description= openmeq/config.py`` is 0), and M11.4's own
"out of scope" note explicitly says adding ~140 of them now would be a large,
unplanned rework of a shared file (``openmeq/config.py``) beyond this
documentation task's budget. Instead, each leaf field's **meaning and unit**
comes from :data:`MEANINGS` below, a hand-maintained ``{"dotted.path":
"prose, with (unit) if applicable"}`` mapping, cross-checked line-by-line
against ``config/example.yaml``'s own comments and ``openmeq/config.py``'s
docstrings/inline comments at the time this script was written.

This script (not the generated Markdown) is therefore the single source of
truth for the *prose*; the schema (``openmeq.config``) is the single source
of truth for the *structure* (path/type/default). Two invariants are
enforced at generation time, so the two can never silently drift apart
without this script failing loudly:

1. Every leaf field the live :class:`~openmeq.config.Config` schema has must
   have a :data:`MEANINGS` entry (a field added to ``openmeq/config.py``
   without updating this script raises here, not just producing a doc with a
   blank cell).
2. Every :data:`MEANINGS` entry must correspond to a real leaf field (a
   renamed/removed field's stale entry raises too, instead of silently
   documenting a key that no longer exists).

``tests/test_config_reference.py`` re-runs this script's ``build_reference()``
and asserts it matches the checked-in ``docs/config_reference.md`` byte for
byte, so the checked-in doc can never go stale relative to either the schema
or this script's own prose without failing ``pytest``.

Usage
-----
::

    python scripts/gen_config_reference.py                    # writes docs/config_reference.md
    python scripts/gen_config_reference.py --check             # exit 1 if the file would change
"""

from __future__ import annotations

import sys
import types
import typing
from datetime import date
from pathlib import Path as _Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from openmeq.config import Config

REPO_ROOT = _Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "config_reference.md"

# ---------------------------------------------------------------------------
# Hand-maintained prose: dotted schema path -> "meaning, with (unit) if any".
#
# Cross-checked against config/example.yaml's own comments and
# openmeq/config.py's docstrings/inline comments. Units follow
# ARCHITECTURE.md §1 (km for location/hypoDD distances, metres for station
# elevation/depth, seconds/Hz/degrees as stated per field).
# ---------------------------------------------------------------------------
MEANINGS: dict[str, str] = {
    # --- project ---
    "project.name": "Project/survey name, used only for display and logging.",
    "project.output_dir": "Directory every pipeline step writes under (ARCHITECTURE.md §3's run "
    "directory layout). Relative paths resolve against `project.base_dir`.",
    "project.base_dir": "Root directory every relative path in the config resolves against "
    "(`output_dir`, `waveform_dirs`, station files, velocity models, ...). `null` = the config "
    "file's own directory; a relative value is itself resolved against the config file's "
    "directory. A leading `~` is expanded.",
    # --- parallel ---
    "parallel.n_workers": '`"auto"` (all physical cores) or an integer worker-process count for '
    "`openmeq.parallel.pmap`'s `ProcessPoolExecutor`.",
    "parallel.threads_per_worker": "BLAS/OpenMP/TensorFlow thread count set inside each worker "
    "process, to avoid oversubscribing the machine when `n_workers` > 1.",
    "parallel.phasenet_device": "`cpu` or `gpu` -- device PhaseNet's `predict.py` runs on. This "
    "project's own dev machine runs PhaseNet on CPU (TensorFlow 2.11 has no matching CUDA build "
    "there); `gpu` is supported by the config schema but not exercised in this project's own "
    "testing.",
    "parallel.chunk_days": "Unit of parallel work (in days) for the QC and detection steps' "
    "per-task batching.",
    # --- data ---
    "data.waveform_dirs": "List of directories searched recursively for miniSEED files.",
    "data.file_pattern": "Glob pattern (relative to each `waveform_dirs` entry) selecting "
    "miniSEED files, e.g. `*.mseed`.",
    "data.start": "Inclusive UTC calendar date (day granularity) the run processes from.",
    "data.end": "Inclusive UTC calendar date (day granularity) the run processes through.",
    "data.convert.format": "Raw-data format converted by `openmeq convert` and at the start "
    "of the `qc` step (and so of `openmeq run`): `suds` (PC-SUDS event files) or `cd11` "
    "(little-endian CD-1.1 recorder files). Omit `data.convert` entirely when the archive "
    "is already miniSEED.",
    "data.convert.input_dir": "Directory searched recursively for raw files to convert.",
    "data.convert.output_dir": "Root of the converted miniSEED day-file archive "
    "(`YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed`); must be one of, or inside one "
    "of, `data.waveform_dirs`.",
    "data.convert.pattern": "Glob selecting the raw files under `input_dir`; "
    "unset = `*.<format>` (`*.suds` / `*.cd11`).",
    "data.convert.network": "SEED network code (1-2 alphanumerics) given to every "
    "converted channel.",
    "data.convert.location": "SEED location code (0-2 alphanumerics) given to every "
    "converted channel.",
    "data.convert.band": "SEED band code of the converted channels (`E` = short period "
    ">=80 Hz, `H` = broadband >=80 Hz).",
    "data.convert.instrument": "SEED instrument code of the converted channels "
    "(`H` = high-gain seismometer).",
    "data.convert.stationxml": "Write `output_dir/stations.xml` (coordinates only, no "
    "instrument response); `stations.stationxml` defaults to it when unset. Always "
    "false for `cd11`, whose files carry no coordinates.",
    "data.convert.overwrite": "Replace existing day files when converting; otherwise "
    "they are kept. The conversion itself is skipped when its input files and settings "
    "are unchanged.",
    # --- stations ---
    "stations.stationxml": "A single StationXML file merged into the run's inventory. StationXML "
    "with a real instrument response is required for the magnitude step.",
    "stations.stationxml_dir": "A folder of StationXML files, all merged in (in addition to "
    "`stationxml`).",
    "stations.csv": "A CSV of station metadata (no response), columns "
    "`net,sta,loc,lat,lon,elev_m,depth_m` (elevation/depth in metres). Merged with any StationXML "
    "given.",
    "stations.channel_map": "`{raw_channel_code: mapped_code}`, mapping non-standard/borehole "
    "component codes onto `Z`/`1`/`2` (unoriented horizontals) before picking. An orientation "
    "given in `orientations` below rotates `1`/`2` to true `N`/`E` instead.",
    "stations.orientations": '`{"NET.STA...CHA": azimuth_deg}` -- optional known component '
    "azimuths (degrees from north), for a borehole channel whose orientation is actually known. "
    "Omitted channels are treated as unoriented horizontals.",
    # --- qc ---
    "qc.daily_stats": "Which per-channel-day statistics to compute; informational list, does not "
    "gate which flags are possible (see the `max_*`/`min_*` thresholds below).",
    "qc.spike_threshold_mad": "A sample counts as an isolated-impulse spike if "
    "`|residual| > this * MAD(residual)` (unitless multiplier on the running-median residual's own "
    "MAD; ARCHITECTURE.md §3).",
    "qc.min_availability_pct": "Minimum data coverage (percent, 0-100) for a channel/day to count "
    "as `ok` on availability grounds alone.",
    "qc.exclude_channels": "Seed ids (e.g. `XX.STA01..HHZ`) left out of detection, and always "
    "flagged not-ok in the QC report. The only way to drop a channel: QC flags themselves are "
    "informational.",
    "qc.flat_run_min_samples": "A run of this many or more identical consecutive raw samples "
    "counts toward `flat_count` (sample count, not seconds).",
    "qc.clip_run_min_samples": "A run of this many or more consecutive saturation-value samples "
    "counts toward `clip_count` (sample count).",
    "qc.max_spike_fraction": 'Flag `"spikes"` if `spike_count / npts` exceeds this fraction '
    "(unitless, 0-1) for a channel/day. Tuned against this project's own synthetic dataset "
    "(`docs/performance.md`); **expect to retune against a real dataset's own spike statistics.**",
    "qc.max_flat_fraction": 'Flag `"flat"` if `flat_count / npts` exceeds this fraction '
    "(unitless, 0-1).",
    "qc.max_clip_count": 'Flag `"clipped"` if the raw `clip_count` (sample count) exceeds this.',
    "qc.monthly_outlier_ratio": 'Flag `"network_outlier"` if a station\'s monthly statistic is '
    "more than this multiple of the network median, or below its reciprocal (unitless ratio).",
    "qc.spike_median_window": "Running-median window width (samples, odd) the isolated-impulse "
    "spike residual is computed against.",
    "qc.max_flat_run_s": 'Flag `"flat"` if the single longest flat run on a channel/day exceeds '
    "this many seconds, independent of `max_flat_fraction` (either criterion alone is sufficient).",
    "qc.noise.core_band": "`[low, high]` Hz band for the hourly noise level, kurtosis and crest "
    "factor; keep it above every sensor's corner so a scalar sensitivity makes like sensors "
    "comparable.",
    "qc.noise.low_band": "`[low, high]` Hz band whose ratio to the core band tracks per-station "
    "response/coupling drift (not comparable between sensor models).",
    "qc.noise.window_s": "Noise-statistics window length (seconds); must divide a day.",
    "qc.noise.min_window_frac": "A noise window counts only if at least this fraction of its "
    "samples is present.",
    "qc.noise.dead_rms_counts": "A noise window whose core-band RMS (counts) is below this is "
    "dead/flatlined hardware.",
    "qc.instrument_groups": "Instrument group -> stations (`STA`, `NET.STA` or seed id) for the "
    "per-instrument noise plots; overrides the StationXML sensor model for the stations listed.",
    "qc.plot_exclude_stations": "Stations (`STA` or `NET.STA`) left out of the `1_qc/diagnostics/` "
    "plots and `summary.csv` only; the QC tables keep them. The plots' monthly network comparison "
    "and flags are recomputed without them. Not part of the qc step's skip-unchanged check: apply "
    "a change with `openmeq diagnostics qc` (redraws from the existing tables).",
    "qc.ppsd.enabled": "Compute an obspy PPSD per channel at the end of `openmeq qc` (also when "
    "the rest of QC is up to date). Not part of the qc skip-unchanged check. "
    "`openmeq diagnostics ppsd` runs it regardless of this flag.",
    "qc.ppsd.components": "Last channel-code characters to include, e.g. `[Z]`; `[]` = every "
    "channel.",
    "qc.ppsd.ppsd_length_s": "Length of each PSD segment (seconds).",
    "qc.ppsd.overlap": "Fractional overlap between PSD segments (0-1).",
    "qc.ppsd.db_bins": "`[min, max, step]` power histogram bins (dB).",
    "qc.ppsd.period_limits": "`[short, long]` period range of the PSD (seconds); `null` = obspy's "
    "default from the sampling rate and segment length. Keep the short end below the "
    "anti-alias corner and the long end within the sensor's band.",
    "qc.ppsd.period_smoothing_width_octaves": "Width of the period smoothing window (octaves).",
    "qc.ppsd.period_step_octaves": "Step between PSD period bins (octaves).",
    "qc.ppsd.skip_on_gaps": "Skip PSD segments containing gaps instead of zero-filling them.",
    "qc.ppsd.xaxis_frequency": "Plot frequency (Hz) rather than period on the PPSD x axis.",
    "qc.ppsd.temporal_periods_s": "Periods (seconds) for the PSD-through-time figure; `[]` skips "
    "it. Periods outside the PSD's range are dropped.",
    # --- detection ---
    "detection.sta": "STA/LTA short-term-average window (seconds).",
    "detection.lta": "STA/LTA long-term-average window (seconds).",
    "detection.trigger_on": 'STA/LTA ratio (unitless) that triggers "on".',
    "detection.trigger_off": 'STA/LTA ratio (unitless) that triggers "off".',
    "detection.filter.type": "Filter applied before STA/LTA: `bandpass`, `highpass`, `lowpass` or "
    "`bandstop`.",
    "detection.filter.freqmin": "Filter low corner frequency (Hz); unused/optional for `lowpass`.",
    "detection.filter.freqmax": "Filter high corner frequency (Hz); unused/optional for "
    "`highpass`.",
    "detection.min_trace_coincidence": "Minimum (weighted) number of *traces* (channels) that "
    'must trigger "on" together for a detection -- `obspy` `coincidence_trigger`\'s '
    "`thr_coincidence_sum`. A 3-component station contributes up to 3, so this is not a station "
    "count. Formerly `coincidence_sum` (the old key is rejected).",
    "detection.min_stations": "Drop a coincidence trigger whose distinct `NET.STA` count is below "
    "this (applied after `coincidence_trigger`). `1` disables the filter.",
    "detection.station_weights": '`{"NET.STA": weight}` -- optional per-station weights passed to '
    "`coincidence_trigger`'s `trace_ids` weighting; a station absent from this map weighs 1.0.",
    "detection.day_overlap_s": "Padding (seconds) read on each side of a per-day detection window "
    "so a trigger straddling midnight isn't lost; must comfortably exceed `lta` plus a trigger's "
    "typical duration.",
    "detection.dedup_window_s": "Triggers within this many seconds of each other are merged into "
    "one detection (STA/LTA often fires separately on P and S); should exceed the network's max "
    "S-P time plus trigger length.",
    # --- detection.sweep (OpenMEQ-cl6; `openmeq diagnostics detect-sweep` only) ---
    "detection.sweep.days": "Sample days (UTC dates) to sweep; empty = `n_days` days spread over "
    "the days with the most channels.",
    "detection.sweep.n_days": "Number of sample days chosen automatically when `days` is empty.",
    "detection.sweep.filters": "Bandpass `[freqmin, freqmax]` pairs (Hz) to sweep.",
    "detection.sweep.sta": "STA windows (seconds) to sweep.",
    "detection.sweep.lta_factor": "LTA windows to sweep, as multiples of `sta`.",
    "detection.sweep.trigger_on": "`trigger_on` values (the L-curve axis); `null` = 15 "
    "log-spaced values from 2.5 to 12.",
    "detection.sweep.trigger_off": "`trigger_off` values; pairs with `trigger_off >= trigger_on` "
    "are skipped.",
    "detection.sweep.min_stations": "`min_stations` values (one L-curve each); `null` = 2 to "
    "min(number of stations, 12).",
    "detection.sweep.trace_coincidence_per_station": "Each `min_stations` value is run with "
    "`min_trace_coincidence = ceil(min_stations * this)`.",
    "detection.sweep.max_station_triggers": "A grid point with more single-trace triggers than "
    "this in one sample day/window is recorded as saturated (noise) and not coincidence-tested.",
    "detection.sweep.reference": "Optional reference catalogue CSV; enables the recall vs "
    "extra-detections curves.",
    "detection.sweep.reference_time_column": "Event-time column of the reference catalogue "
    "(ISO-8601, UTC).",
    "detection.sweep.match_window_s": "`[lo, hi]` (seconds): a detection matches a reference "
    "event if `detection_time - event_time` is in this range (one-to-one).",
    "detection.sweep.windows_csv": "`coincidence_windows.csv` from `scripts/detect_windows.py`: "
    "sweep inside these recording windows (triggered archives) instead of whole days.",
    "detection.sweep.max_windows": "Use at most this many recording windows, spread evenly; "
    "`null` = all.",
    "detection.sweep.edge_s": "Window mode: ignore triggers this many seconds after a window "
    "start; `null` = 2 x the largest swept `lta`.",
    "detection.sweep.n_workers": "Worker processes (one sample day x filter band each; a day of "
    "many 200 Hz channels can use >10 GB).",
    # --- chunks ---
    "chunks.pre_s": "Seconds of context kept before a detection's trigger time in the raw chunk.",
    "chunks.post_s": "Seconds of context kept after.",
    "chunks.max_read_window_s": "A day's chunk-writing reads are split into sub-windows wherever "
    "two consecutive detections' trigger times are more than this many seconds apart (seconds; "
    "bounds peak memory use per read).",
    # --- phasenet ---
    "phasenet.target_rate_hz": "PhaseNet's own trained sampling rate (Hz); other rates are "
    "decimated to this during chunk preparation.",
    "phasenet.taper_pct": "Hann taper fraction (unitless, 0-1) applied before filtering/"
    "decimation.",
    "phasenet.highpass_hz": "High-pass corner (Hz) applied before decimation; `0` disables it.",
    "phasenet.min_p_prob": "Discard PhaseNet P picks below this probability (unitless, 0-1).",
    "phasenet.min_s_prob": "Discard PhaseNet S picks below this probability (unitless, 0-1).",
    "phasenet.batch_size": "Currently unused: the PhaseNet runner always invokes `predict.py` "
    "with `--batch_size=1` regardless of this value, since `predict.py`'s own batching needs equal "
    "station counts across events too (docs/phasenet_interface.md §7). Kept as a forward-looking "
    "config hook.",
    "phasenet.max_gap_fill_s": "Gaps up to this long (seconds) are zero-filled rather than "
    "interpolated during chunk preprocessing; longer gaps are left as real gaps.",
    "phasenet.pick_uncertainty_method": "How PhaseNet `Pick.time_errors.uncertainty` is set. "
    "`peak_width`: half-width (seconds) of the pick's P/S probability peak in "
    "`phasenet_raw/result.h5`, measured at `pick_uncertainty_peak_frac` x peak and clamped to "
    "`[pick_uncertainty_min_s, pick_uncertainty_max_s]`. `fixed`: every pick gets "
    "`pick_uncertainty_s`. The method used is stored per pick as extra `uncertainty_method`.",
    "phasenet.pick_uncertainty_s": "Fixed pick uncertainty (seconds) for `fixed`, and the "
    "`peak_width` fallback when `result.h5` or the pick's station column cannot be used.",
    "phasenet.pick_uncertainty_peak_frac": "Fraction of the peak probability at which the peak "
    "width is measured (0.5 = half the full width at half maximum).",
    "phasenet.pick_uncertainty_min_s": "Lower clamp (seconds) on `peak_width` uncertainties.",
    "phasenet.pick_uncertainty_max_s": "Upper clamp (seconds) on `peak_width` uncertainties.",
    "phasenet.pick_window_s": "`[before, after]` seconds relative to the detection trigger time: "
    "PhaseNet picks outside it are dropped before choosing one pick per station and phase "
    "(rejected reason `outside window`). PhaseNet fires on the start of every processed chunk "
    "(taper + high-pass), and those artefacts can out-score real picks. `null` keeps the whole "
    "chunk.",
    # --- nlloc ---
    "nlloc.velocity_model.type": "`1d` (layer file) or `3d` (user-supplied NLLoc grid).",
    "nlloc.velocity_model.file": "1D: path to the layer file (ARCHITECTURE.md §6a). 3D: root name "
    "of a pre-built NLLoc grid (`<file>.{P,S}.mod.{hdr,buf}` must already exist; 3D grids are "
    "always user-supplied, never built by this project).",
    "nlloc.grid.origin": "`[lat, lon]` (degrees) of the location-grid origin (also `TRANS "
    "SIMPLE`'s reference point).",
    "nlloc.grid.size_km": "`[x, y, z]` full grid extent (km).",
    "nlloc.grid.spacing_km": "Grid node spacing (km).",
    "nlloc.filter.p_only.max_iter": "Pass A (P-only) maximum relocate/filter iterations.",
    "nlloc.filter.p_only.max_residual_s": "Pass A: drop a P pick if `|residual|` exceeds this "
    "many seconds.",
    "nlloc.filter.p_only.residual_sigma": "Pass A: drop a P pick if its residual exceeds this many "
    "standard deviations from the rest (unitless multiplier).",
    "nlloc.filter.p_and_s.max_iter": "Pass B (P+S) maximum relocate/filter iterations.",
    "nlloc.filter.p_and_s.s_predicted_window_s": "Pass B: drop an S pick more than this many "
    "seconds from the S arrival time predicted by the pass-A location.",
    "nlloc.filter.p_and_s.p.max_residual_s": "Pass B: drop a P pick if `|residual|` exceeds this "
    "many seconds.",
    "nlloc.filter.p_and_s.p.residual_sigma": "Pass B: drop a P pick beyond this many standard "
    "deviations (unitless).",
    "nlloc.filter.p_and_s.s.max_residual_s": "Pass B: drop an S pick if `|residual|` exceeds this "
    "many seconds.",
    "nlloc.filter.p_and_s.s.residual_sigma": "Pass B: drop an S pick beyond this many standard "
    "deviations (unitless).",
    "nlloc.filter.events.min_p_phases": "Reject an event (checked after pass A) with fewer than "
    "this many accepted P picks.",
    "nlloc.filter.events.min_phases": "Reject an event (checked after pass B) with fewer than this "
    "many total accepted picks.",
    "nlloc.filter.events.min_s_phases": "Reject an event (checked after pass B) with fewer than "
    "this many accepted S picks.",
    "nlloc.filter.events.max_rms_s": "Reject an event whose final origin RMS residual exceeds this "
    "many seconds.",
    "nlloc.filter.events.max_gap_deg": "Reject an event whose azimuthal gap exceeds this many "
    "degrees.",
    "nlloc.filter.events.max_err_h_km": "Reject an event whose horizontal location error exceeds "
    "this many km.",
    "nlloc.filter.events.max_err_z_km": "Reject an event whose vertical (depth) location error "
    "exceeds this many km.",
    "nlloc.obspyck_pluginpath": "Directory ObsPyck's `setup_external_programs` expects as its "
    "`pluginpath` (ARCHITECTURE.md §2's `openmeq.review.obspyck_config` entry); refreshed by "
    "`scripts/setup_nlloc.py`/the `locate` step.",
    "nlloc.locsearch.init_num_cells_x": "NLLoc `LOCSEARCH OCT` initial octree cell count, x axis.",
    "nlloc.locsearch.init_num_cells_y": "NLLoc `LOCSEARCH OCT` initial octree cell count, y axis.",
    "nlloc.locsearch.init_num_cells_z": "NLLoc `LOCSEARCH OCT` initial octree cell count, z axis.",
    "nlloc.locsearch.min_node_size": "NLLoc `LOCSEARCH OCT` minimum octree node size (km) to stop "
    "subdividing.",
    "nlloc.locsearch.max_num_nodes": "NLLoc `LOCSEARCH OCT` maximum total octree nodes.",
    "nlloc.locsearch.num_scatter": "NLLoc `LOCSEARCH OCT` number of scatter-sample points saved "
    "for the location PDF.",
    "nlloc.locsearch.use_stations_density": "NLLoc `LOCSEARCH OCT`: weight octree subdivision by "
    "station density (boolean).",
    "nlloc.locsearch.stop_on_min_node_size": "NLLoc `LOCSEARCH OCT`: stop once `min_node_size` is "
    "reached rather than continuing to `max_num_nodes` (boolean).",
    "nlloc.locgau.sigma_time_s": "NLLoc `LOCGAU` constant Gaussian model-error standard deviation "
    "(seconds).",
    "nlloc.locgau.corr_len_km": "NLLoc `LOCGAU` model-error spatial correlation length (km).",
    "nlloc.locgau2.sigma_time_fraction": "NLLoc `LOCGAU2` travel-time-dependent error fraction "
    "(unitless multiplier of predicted travel time).",
    "nlloc.locgau2.sigma_time_min_s": "NLLoc `LOCGAU2` minimum travel-time error (seconds).",
    "nlloc.locgau2.sigma_time_max_s": "NLLoc `LOCGAU2` maximum travel-time error (seconds, "
    "`SigmaTmax`).",
    "nlloc.locmeth.method": "NLLoc `LOCMETH` location method, e.g. `EDT_OT_WT` (this project's "
    "default).",
    "nlloc.locmeth.max_dist_sta_grid": "NLLoc `LOCMETH` max station-to-grid distance (km) to use a "
    "phase.",
    "nlloc.locmeth.min_number_phases": "NLLoc `LOCMETH` minimum phases required to attempt a "
    "location.",
    "nlloc.locmeth.max_number_phases": "NLLoc `LOCMETH` maximum phases used; `-1` = no limit.",
    "nlloc.locmeth.min_number_s_phases": "NLLoc `LOCMETH` minimum S phases required; `-1` = no "
    "minimum.",
    "nlloc.locmeth.vp_vs_ratio": "NLLoc `LOCMETH` Vp/Vs ratio; negative (this project's default) "
    'means "use the real S travel-time grids" rather than deriving S times from a fixed ratio '
    "applied to the P grid.",
    "nlloc.locmeth.max_num_3d_grid_memory": "NLLoc `LOCMETH` max number of 3D travel-time grids "
    "kept in memory at once.",
    "nlloc.locmeth.min_dist_sta_grid": "NLLoc `LOCMETH` minimum station-to-grid distance (km); "
    "`-1` = no minimum.",
    "nlloc.locmeth.reject_duplicate_arrivals": "NLLoc `LOCMETH`: reject duplicate arrivals for the "
    "same station/phase (boolean).",
    "nlloc.locqual2err_thresholds_s": "Ascending `time_errors.uncertainty` (seconds) bucket edges "
    "for NLLoc's required `LOCQUAL2ERR` statement. NLLoc ignores these values for this project's "
    "own `NLLOC_OBS` input (which always carries a literal per-pick error) -- a syntactically "
    "valid ascending list is still required to satisfy NLLoc, but the numbers themselves have no "
    "effect on the location (docs/nlloc_interface.md §5).",
    "nlloc.station_delays_csv": "Optional CSV of per-station, per-phase time delays (columns "
    "`station`, `phase` P/S, `delay_s`; optional `n`, `std_s`), written to every NLLoc control "
    "file as `LOCDELAY` statements. NLLoc subtracts the delay from the observed time, so a "
    "station's mean residual is its delay. `null` = no delays.",
    # --- review ---
    "review.enabled": "`true` to pause the pipeline for manual ObsPyck review "
    "(`openmeq run`'s branching rule, ARCHITECTURE.md §2); `false` (default) to go straight from "
    "`locate/catalog_auto.xml` to magnitudes.",
    "review.obspyck_config": "Path to the generated ObsPyck `.cfg` file (`openmeq.review."
    "obspyck_config.generate_obspyck_config`'s output).",
    # --- magnitude ---
    "magnitude.type": "Magnitude type; currently only `ml` (local magnitude) is implemented.",
    "magnitude.ml.wood_anderson": "Simulate a Wood-Anderson response from the StationXML "
    "instrument response before measuring amplitude (boolean; effectively always `true` for a "
    "real ML measurement).",
    "magnitude.ml.amplitude_window": "Amplitude measurement window anchor; only `s` (S-pick-"
    "centred, falling back to a P-based coda window) is currently supported.",
    "magnitude.ml.component": "Which component(s) to measure; only `horizontal` (max of the two "
    "horizontals, falling back to `Z`) is currently supported.",
    "magnitude.ml.relation": "`hutton_boore` (Hutton & Boore 1987 formula) or `local` (a locally "
    "calibrated formula or distance/-logA0 table).",
    "magnitude.ml.hutton_boore.a": "Hutton & Boore (1987) coefficient `a` in "
    "`ML = log10(A_mm) + a*log10(R/100) + b*(R-100) + c` (unitless).",
    "magnitude.ml.hutton_boore.b": "Hutton & Boore (1987) coefficient `b` (units: 1/km, applied to "
    "hypocentral distance `R` in km).",
    "magnitude.ml.hutton_boore.c": "Hutton & Boore (1987) coefficient `c` (unitless constant).",
    "magnitude.ml.local.coefficients.a": "Locally calibrated coefficient `a`, same 3-term form as "
    "`hutton_boore.a`; leave all three of `a`/`b`/`c` unset (`null`) to use `local.file` (a "
    "distance table) instead.",
    "magnitude.ml.local.coefficients.b": "Locally calibrated coefficient `b` (units: 1/km).",
    "magnitude.ml.local.coefficients.c": "Locally calibrated coefficient `c` (unitless constant).",
    "magnitude.ml.local.file": "CSV path, header `distance_km,neg_log_a0`, distance ascending, "
    ">= 2 rows -- a distance/-logA0 calibration table, linearly interpolated (clamped outside its "
    "range) instead of the 3-coefficient formula (ARCHITECTURE.md §6b). Used only when "
    "`relation: local` and `local.coefficients` is entirely unset.",
    "magnitude.ml.station_corrections": 'Either an inline `{"NET.STA": correction_mag}` mapping or '
    "a CSV path (header `net,sta,correction`) of per-station ML corrections (magnitude units, "
    "added to the station ML); a station absent from either defaults to `0.0`.",
    "magnitude.ml.wa_sensitivity": "Wood-Anderson magnification constant used when simulating the "
    "WA response (unitless simulator gain). Default `2080` is the IASPEI/Uhrhammer & Collins "
    "(1990) value, not obspy's historical default of `2800` (ARCHITECTURE.md §6e).",
    "magnitude.ml.window_before_s": "Amplitude measurement window starts this many seconds before "
    "the S pick (falls back to `P_pick + 1.0` if no S pick).",
    "magnitude.ml.window_after_s": "Amplitude measurement window ends this many seconds after the "
    "S pick (falls back to `P_pick + 8.0` if no S pick).",
    "magnitude.ml.taper_pct": "Cosine taper fraction (unitless, 0-1) applied before instrument-"
    "response removal.",
    "magnitude.ml.pre_filt": "4-corner bandpass `[f1, f2, f3, f4]` (Hz) passed to ObsPy's "
    "`remove_response(pre_filt=...)`; conservative broadband default -- **retune once real "
    "station response passbands are known.**",
    "magnitude.min_stations": "Minimum number of station MLs required to compute an event ML.",
    "magnitude.statistic": "`median` or `mean` -- how station MLs are combined into the event ML.",
    "magnitude.exclude_stations": "Stations (`STA` or `NET.STA`) whose amplitudes are still "
    "measured and written to `station_magnitudes.csv` (`excluded=True`) but never used in the "
    "event magnitude. A station with no instrument response is skipped automatically.",
    # --- hypodd ---
    "hypodd.velocity_model.type": "`1d` (layer file) or `3d` (SIMUL2000 `vel3D.dat`).",
    "hypodd.velocity_model.file": "1D: path to the layer file (same format/file as "
    "`nlloc.velocity_model.file` may point to). 3D: path to a SIMUL2000 `vel3D.dat` (can be "
    "generated from an NLLoc 3D grid via `scripts/setup_hypodd.py --nlloc-to-simul`).",
    "hypodd.velocity_model.origin_3d": "`[lat, lon]` (degrees) SIMUL2000 model origin; only used "
    "when `velocity_model.type: 3d`.",
    "hypodd.velocity_model.rotation_3d_deg": "SIMUL2000 Cartesian-frame rotation (degrees, "
    "positive = anticlockwise); only used when `velocity_model.type: 3d`.",
    "hypodd.velocity_model.simul2000.iphase": "SIMUL2000 ray-tracing block (`IPHA NDIP ISKIP "
    "SCALE1 SCALE2 XFAC TLIM NITPB`, hypoDD.inp's `IMOD=9` block; only used when "
    "`velocity_model.type: 3d`). `IPHA` -- phase(s) to ray-trace (1=P, 2=P&S, per `get_vel3d.f`).",
    "hypodd.velocity_model.simul2000.ndip": "SIMUL2000 `NDIP` -- number of take-off-angle "
    "dip bins.",
    "hypodd.velocity_model.simul2000.iskip": "SIMUL2000 `ISKIP` -- ray-tracing point-skip factor.",
    "hypodd.velocity_model.simul2000.scale1": "SIMUL2000 `SCALE1` ray-tracing scale parameter.",
    "hypodd.velocity_model.simul2000.scale2": "SIMUL2000 `SCALE2` ray-tracing scale parameter.",
    "hypodd.velocity_model.simul2000.xfac": "SIMUL2000 `XFAC` ray-tracing expansion factor "
    "(unitless).",
    "hypodd.velocity_model.simul2000.tlim": "SIMUL2000 `TLIM` ray-tracing convergence limit "
    "(seconds).",
    "hypodd.velocity_model.simul2000.nitpb": "SIMUL2000 `NITPB` maximum ray-tracing iterations per "
    "ray.",
    "hypodd.xcorr.window_p": "`[start, end]` of the P cross-correlation window, signed seconds "
    "relative to the P pick (`[-0.1, 0.1]` = 100 ms before to 100 ms after); `start < end`. "
    "Start early enough to include the onset when a pick is late (about the typical pick "
    "error); end after the first 1-2 cycles of P. Longer windows add coda and lower cc; "
    "windows under ~2 cycles correlate by chance.",
    "hypodd.xcorr.window_s": "As `window_p`, for S, relative to the S pick. S is lower "
    "frequency with a less sharp onset, so allow more time after the pick than for P.",
    "hypodd.xcorr.min_cc": "Minimum correlation coefficient (0-1) for a lag measurement to be "
    "kept in `dt.cc`. Raise it if hypoDD's cc residual grows or the "
    "`diagnostics.relocate.xcorr_plots` figures show poor alignments being kept. Shorter "
    "windows reach high cc more easily and warrant a higher threshold; lower it (to ~0.7) if "
    "too few pairs link.",
    "hypodd.xcorr.maxlag_s": "Maximum lag (seconds) searched. Must cover real pick errors but "
    "stay below one cycle of the dominant frequency, or the correlation can lock onto the wrong "
    "cycle (cycle skipping) and still pass `min_cc`; about half to two-thirds of a period.",
    "hypodd.xcorr.s_maxlag_s": "Maximum lag (seconds) for S only; `null` = `maxlag_s`. S picks "
    "are less precise than P and S has a lower dominant frequency, so S can usually take a "
    "longer search than P before cycle skipping.",
    "hypodd.xcorr.components": '`{"P": [...], "S": [...]}` -- which mapped component code(s) to '
    "cross-correlate P and S on (e.g. `Z` for P; every configured horizontal tried independently "
    "for S, keeping only the best-correlating one).",
    "hypodd.xcorr.filter.freqmin": "Bandpass low corner (Hz) applied once when building the "
    "cross-correlation waveform cache. Choose the band where P and S stand above the noise from "
    "event spectra; raise it if long-period noise dominates. The 2-20 Hz default was checked "
    "against this project's **synthetic** P/S spectra (ARCHITECTURE.md §6g) -- **retune against "
    "real event spectra**.",
    "hypodd.xcorr.filter.freqmax": "Bandpass high corner (Hz), same note as `freqmin`; keep it "
    "below `resample_rate_hz / 2`.",
    "hypodd.xcorr.filter.corners": "Bandpass filter order (obspy `corners`) for the "
    "cross-correlation waveform cache; 4 is standard, higher orders ring more.",
    "hypodd.xcorr.filter.zerophase": "Zero-phase (forward-backward) bandpass for the cache "
    "(boolean). Keep true: a causal filter shifts waveforms in time.",
    "hypodd.xcorr.taper_pct": "Hann taper fraction applied to each end of the chunk segment "
    "before the cache's bandpass filter (0-0.5); only limits filter edge artefacts.",
    "hypodd.xcorr.filter_edge_margin_s": "Extra data (seconds) cached either side of each "
    "window +/- `maxlag_s`, to keep the bandpass filter's edge effects out of the window. "
    "At least one cycle at `freqmin` (0.5 s = one cycle at 2 Hz); raise it if `freqmin` is "
    "lowered.",
    "hypodd.xcorr.reject_edge_lags": "Drop a channel's lag measurement when its best lag is at "
    "the edge of the `+/- maxlag_s` search (within half a sample) -- the true correlation peak "
    "lies outside the search, so the lag is unreliable (boolean). Keep true.",
    "hypodd.xcorr.resample_rate_hz": "Common sampling rate (Hz) every cached cross-correlation "
    "window is resampled to, regardless of native rate, so pairs from different events' caches "
    "always share one rate (ARCHITECTURE.md §6f). Use the native rate of most stations; keep it "
    "at least ~2.5x `filter.freqmax`.",
    "hypodd.ph2dt.maxsep_km": "ph2dt `MAXSEP` -- max. hypocentral separation (km) between a "
    "candidate event pair. See docs/hypodd_tuning.md for how this and the other ph2dt/hypoDD "
    "distance fields below were scaled for this project's own ~15 km-aperture synthetic network, "
    "and how to rescale them for a different network.",
    "hypodd.ph2dt.maxngh": "ph2dt `MAXNGH` -- max. number of neighbours per event.",
    "hypodd.ph2dt.minlnk": "ph2dt `MINLNK` -- min. phase observations for a neighbour to count as "
    '"strong"; should stay `>= hypodd.clustering.obsct` below.',
    "hypodd.ph2dt.minwght": "ph2dt `MINWGHT` -- min. pick weight kept (0-1); hypoDD's own `-1` "
    '"no info" sentinel always bypasses this regardless of value.',
    "hypodd.ph2dt.maxdist_km": "ph2dt `MAXDIST` -- max. distance (km) between an event pair and a "
    "station.",
    "hypodd.ph2dt.minobs": "ph2dt `MINOBS` -- min. observations kept per linked pair (pairs below "
    "this are discarded entirely).",
    "hypodd.ph2dt.maxobs": "ph2dt `MAXOBS` -- max. observations kept per linked pair (nearest "
    "stations first).",
    "hypodd.weighting_rows": "hypoDD's iterative data-(re)weighting table: a list of rows "
    "`{niter, wtccp, wtccs, wrcc, wdcc, wtctp, wtcts, wrct, wdct, damp}` (`niter` = last iteration "
    "using that row; `wtcc*`/`wtct*` = a-priori weights on cross-correlation/catalog P and S data, "
    "unitless; `wrcc`/`wrct` = residual outlier cutoffs, seconds if `<1` or a dynamic "
    "residual-stddev multiplier if `>=1`; `wdcc`/`wdct` = max linked-pair distance, km; `damp` = "
    "LSQR damping factor, unitless). See docs/hypodd_tuning.md for the full derivation of this "
    "project's default 5-row schedule and how to rescale `wdcc`/`wdct` for a different network's "
    "aperture.",
    "hypodd.istart": "hypoDD `ISTART` -- `1` = locate from a single source, `2` = from network "
    "sources (this project's default).",
    "hypodd.isolv": "hypoDD `ISOLV` -- `1` = SVD, `2` = LSQR (this project's default; scales to "
    "much larger event counts than SVD).",
    "hypodd.iaq": "hypoDD `IAQ` -- air-quake handling: `1` = remove (default), `2` = keep and "
    'reset depth, `0` = legacy/undocumented "don\'t remove at all" (a synthetic/test-only '
    'escape hatch). See docs/hypodd_tuning.md\'s "Air-quakes" section for what this means on a '
    "shallow/borehole-referenced network.",
    "hypodd.idat": "hypoDD `IDAT` -- data types used: `1` = cross-correlation only, `2` = catalog "
    "only, `3` = both (this project's default).",
    "hypodd.ipha": "hypoDD `IPHA` -- phases used: `1` = P only, `2` = S only, `3` = P & S (this "
    "project's default).",
    "hypodd.max_station_dist_km": "hypoDD.inp `DIST` -- max. distance (km) between a cluster "
    "centroid and a station.",
    "hypodd.clustering.obscc": "hypoDD `OBSCC` -- min. cross-correlation observations/pair to "
    "cluster on; `0` disables clustering on cross-correlation data.",
    "hypodd.clustering.obsct": "hypoDD `OBSCT` -- min. catalog observations/pair to cluster on; "
    "should stay `<= hypodd.ph2dt.minlnk` above.",
    "hypodd.clustering.mindist_km": "hypoDD `MINDIST` -- min. pair-station distance (km) for "
    "clustering; `-999` = not active.",
    "hypodd.clustering.maxdist_km": "hypoDD `MAXDIST` -- max. pair-station distance (km) for "
    "clustering; `-999` = not active.",
    "hypodd.clustering.maxgap_deg": "hypoDD `MAXGAP` -- max. azimuthal gap (degrees) for "
    "clustering; `-999` = not active.",
    "hypodd.parallel_clusters": "Opt-in/experimental (boolean, default `false`): relocate each of "
    "hypoDD's own event clusters (`OBSCC`/`OBSCT` above) as an independent `hypoDD` subprocess in "
    "parallel, then merge `hypoDD.reloc` (ARCHITECTURE.md §6h).",
    # --- diagnostics ---
    "diagnostics.extended": "Default for every step's opt-in extended QAQC figures (written to "
    "`<step>/diagnostics/extended/`); the always-on baseline diagnostics are unaffected.",
    "diagnostics.n_examples": "Number of sample events for per-event waveform QAQC figures.",
    "diagnostics.wells.enabled": "Overlay well heads/tracks on every map and depth-section "
    "figure (locate pass maps and comparison PDF, extended station-residual map, relocate "
    "before/after, review shift map); requires `diagnostics.wells.dir`.",
    "diagnostics.wells.dir": "Directory of headerless `<Group>_well_heads.csv` (`name,lon,lat`) "
    "and `<Group>_well_tracks.csv` (`name,lon,lat,elevation_m`, m above sea level) files.",
    "diagnostics.wells.groups": "Well groups (file prefixes, e.g. `Production`) to draw; "
    "empty = every group found in `dir`.",
    "diagnostics.wells.colors": "`{group: matplotlib colour}` overrides; defaults are "
    "Production vermillion, Injection sky blue, Other grey.",
    "diagnostics.wells.labels": "Annotate well names on map panels (boolean).",
    "diagnostics.detect.extended": "Extended-QAQC override for `detect` "
    "(STA/LTA characteristic-function examples, per-station trigger-rate heatmap); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.chunk.extended": "Extended-QAQC override for `chunk` "
    "(component coverage matrix, raw vs PhaseNet-prepped chunk examples); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.pick.extended": "Extended-QAQC override for `pick` "
    "(zoomed pick panels, pre-location Wadati diagram, pick offset from detection time); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.locate.extended": "Extended-QAQC override for `locate` "
    "(residual vs distance, station mean-residual map, depth histogram vs grid/model); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.magnitude.extended": "Extended-QAQC override for `magnitude` "
    "(station ML residual vs distance, GR with Mc/b-value, Wood-Anderson amplitude windows); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.relocate.xcorr_plots.enabled": "Write obspy `xcorr_pick_correction` "
    "alignment/correlation plots for a sample of event pairs to "
    "`relocate/diagnostics/xcorr_pairs/` (also: `openmeq diagnostics xcorr`).",
    "diagnostics.relocate.xcorr_plots.pair_interval": "Plot every Nth ph2dt-linked event pair, "
    "in `dt.ct` order (1 = every pair).",
    "diagnostics.relocate.xcorr_plots.max_pairs": "Maximum number of event pairs to plot.",
    "diagnostics.relocate.extended": "Extended-QAQC override for `relocate` "
    "(dt.cc - dt.ct histogram, hypoDD.res residuals by data type, per-iteration mean shifts); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.run.extended": "Extended-QAQC override for `run` "
    "(event-count funnel across every stage of `openmeq run`); "
    "unset = inherit `diagnostics.extended`.",
    "diagnostics.pick.pick_zoom.half_window_s": "Half-width of each zoomed pick panel around "
    "the pick time (s).",
    "diagnostics.pick.pick_zoom.n_events": "Number of sample events to draw zoomed pick "
    "panels for.",
    "diagnostics.pick.pick_zoom.show_rejected": "Also draw (greyed) picks dropped by the pick "
    "step's probability/duplicate filtering (boolean).",
}


def _render_type(ann: Any) -> str:
    """Render a pydantic field annotation as a short, human-readable type string."""
    if ann is type(None):
        return "None"
    origin = get_origin(ann)
    if origin is Literal:
        return " | ".join(repr(a) for a in get_args(ann))
    if origin in (Union, types.UnionType):
        parts = [_render_type(a) for a in get_args(ann)]
        return " | ".join(parts)
    if origin is list:
        (item,) = get_args(ann) or (Any,)
        return f"list[{_render_type(item)}]"
    if origin is dict:
        k, v = get_args(ann) or (Any, Any)
        return f"dict[{_render_type(k)}, {_render_type(v)}]"
    if origin is tuple:
        args = get_args(ann)
        return f"tuple[{', '.join(_render_type(a) for a in args)}]"
    if ann is _Path:
        return "path"
    if ann is date:
        return "date"
    if isinstance(ann, type):
        return ann.__name__
    return str(ann)


def _render_default(value: Any) -> str:
    """Render a resolved default value as a Markdown-table-safe literal."""
    if isinstance(value, BaseModel):
        # Only reached for a list-of-models default (e.g. weighting_rows);
        # a bare nested-model field is recursed into, never rendered as a leaf.
        return "`" + str(value.model_dump()) + "`"
    if isinstance(value, list):
        return "`" + str([v.model_dump() if isinstance(v, BaseModel) else v for v in value]) + "`"
    if isinstance(value, _Path):
        return f"`{value}`"
    if isinstance(value, str):
        return f'`"{value}"`'
    return f"`{value}`"


def _unwrap_optional_model(ann: Any) -> type[BaseModel] | None:
    """If ``ann`` is ``SomeModel``, or ``SomeModel | None``, return ``SomeModel``; else ``None``."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):
        models = [a for a in get_args(ann) if isinstance(a, type) and issubclass(a, BaseModel)]
        if len(models) == 1:
            return models[0]
    return None


#: Path -> default rendering override, for a default value too long/unwieldy to dump inline
#: (e.g. a 5-row list of objects) -- the full value is still exactly what
#: ``openmeq.config`` uses; only the *rendering* in this doc is shortened.
DEFAULT_OVERRIDES: dict[str, str] = {
    "hypodd.weighting_rows": "*(5-row loose-then-tight schedule; full values in "
    "`config/example.yaml` and derivation in `docs/hypodd_tuning.md`)*",
}


class Leaf(typing.NamedTuple):
    path: str
    type_str: str
    default_str: str


def walk(model: type[BaseModel], prefix: str = "") -> list[Leaf]:
    """Recurse through ``model``'s fields, returning every leaf field's (path, type, default)."""
    leaves: list[Leaf] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        nested = _unwrap_optional_model(field.annotation)
        if nested is not None:
            leaves.extend(walk(nested, prefix=f"{path}."))
            continue
        type_str = _render_type(field.annotation)
        if path in DEFAULT_OVERRIDES:
            default_str = DEFAULT_OVERRIDES[path]
        elif field.default is PydanticUndefined and field.default_factory is None:
            default_str = "*(required)*"
        else:
            default_str = _render_default(field.get_default(call_default_factory=True))
        leaves.append(Leaf(path, type_str, default_str))
    return leaves


SECTION_TITLES: dict[str, str] = {
    "project": "`project` -- run identity and output location",
    "parallel": "`parallel` -- process-pool sizing (openmeq.parallel)",
    "data": "`data` -- input miniSEED selection",
    "stations": "`stations` -- station metadata sources",
    "qc": "`qc` -- QA/QC thresholds (`openmeq qc`)",
    "detection": "`detection` -- STA/LTA network detection (`openmeq detect`)",
    "chunks": "`chunks` -- per-event waveform chunk writing (`openmeq chunk`)",
    "phasenet": "`phasenet` -- PhaseNet preparation and picking (`openmeq pick`)",
    "nlloc": "`nlloc` -- NonLinLoc location and filtering "
    "(`openmeq locate`, `scripts/setup_nlloc.py`)",
    "review": "`review` -- manual ObsPyck review (`openmeq review`)",
    "magnitude": "`magnitude` -- local magnitude (`openmeq magnitude`)",
    "hypodd": "`hypodd` -- hypoDD relocation (`openmeq relocate`, `scripts/setup_hypodd.py`)",
    "diagnostics": "`diagnostics` -- optional extended QAQC figures (every step except `qc`)",
}


def build_reference() -> str:
    """Build the full ``docs/config_reference.md`` Markdown text."""
    leaves = walk(Config)

    by_section: dict[str, list[Leaf]] = {}
    for leaf in leaves:
        section = leaf.path.split(".", 1)[0]
        by_section.setdefault(section, []).append(leaf)

    missing = [leaf.path for leaf in leaves if leaf.path not in MEANINGS]
    if missing:
        raise SystemExit(
            "gen_config_reference.py: MEANINGS is missing an entry for these live "
            f"openmeq.config.Config field(s): {missing}. Add a description (with unit, if "
            "applicable) to scripts/gen_config_reference.py's MEANINGS dict."
        )
    extra = [path for path in MEANINGS if path not in {leaf.path for leaf in leaves}]
    if extra:
        raise SystemExit(
            "gen_config_reference.py: MEANINGS has stale entries with no matching "
            f"openmeq.config.Config field (renamed or removed?): {extra}. Remove them from "
            "scripts/gen_config_reference.py's MEANINGS dict."
        )

    lines: list[str] = [
        "# OpenMEQ configuration reference",
        "",
        "Generated by `scripts/gen_config_reference.py` from the live "
        "`openmeq.config.Config` pydantic schema -- **do not hand-edit**; re-run the "
        "generator instead (`python scripts/gen_config_reference.py`). "
        "`tests/test_config_reference.py` fails if this file is stale relative to the "
        "schema or the generator's own field-meaning text.",
        "",
        "Every key below is documented, in the same order, in `config/example.yaml` -- the "
        "reference example config `openmeq.config.load_config` validates against. See "
        "[usage.md](usage.md) for what each pipeline step reads/writes, and "
        "[ARCHITECTURE.md](ARCHITECTURE.md) for the underlying contracts (§1 for the units "
        "convention, §3 for the run directory layout).",
        "",
        'Every section is `extra="forbid"`: a misspelled or unknown key in your config '
        "raises a validation error naming the bad path, rather than being silently ignored. "
        "Any relative path value is resolved against `project.base_dir` if set, otherwise "
        "against *this config file's own directory* -- never the process's current working "
        "directory.",
        "",
        "A field's default is shown as `*(required)*` if the YAML must supply it explicitly "
        "(no schema default).",
        "",
    ]

    for section, title in SECTION_TITLES.items():
        section_leaves = by_section.get(section, [])
        if not section_leaves:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Key | Type | Default | Meaning |")
        lines.append("|---|---|---|---|")
        for leaf in section_leaves:
            meaning = MEANINGS[leaf.path].replace("|", "\\|")
            type_str = leaf.type_str.replace("|", "\\|")
            lines.append(f"| `{leaf.path}` | {type_str} | {leaf.default_str} | {meaning} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    text = build_reference()
    check = "--check" in sys.argv[1:]
    if check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.is_file() else ""
        if current != text:
            print(f"{OUTPUT_PATH} is stale; run `python scripts/gen_config_reference.py`.")
            return 1
        print(f"{OUTPUT_PATH} is up to date.")
        return 0
    OUTPUT_PATH.write_text(text)
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
