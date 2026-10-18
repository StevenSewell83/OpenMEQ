# OpenMEQ parameter guide

How to choose every setting in an OpenMEQ config file: what it does, how to
pick a value, and which diagnostic shows when it is wrong. Settings are in
pipeline order, the order to tune them in: a bad detection setting cannot be
fixed by relocation settings.

- Every key, its type and default: [`config_reference.md`](config_reference.md).
- A short comment per key: [`config/example.yaml`](../config/example.yaml).
- The extended QAQC figures named below: [`diagnostics_extended.md`](diagnostics_extended.md).
- hypoDD/ph2dt in depth: [`hypodd_tuning.md`](hypodd_tuning.md).

Worked values marked **RK-NM** come from the Rotokawa / Ngatamariki 2026 Q1
run (23 stations, ~9 x 18 km, mostly 200 Hz short-period geophones, local
geothermal events with P/S energy around 10-15 Hz). Treat them as a starting
point for a similar network, not as universal values.

## Before you start

Two numbers drive many choices below. Find them early (from a few events in
the data, or after a first run):

- **Dominant frequency of P and S** (one cycle = 1/f). Sets filter bands and the
  cross-correlation windows and lag.
- **Typical pick error.** Sets the pick and residual tolerances, and the
  cross-correlation lead-in and lag.

Turn on `diagnostics.extended: true` for a first run on a short date range
(a few days). Every step then writes the figures used for tuning, under
`<step>/diagnostics/extended/`.

---

## `project`

| Key | What it does | How to choose |
|---|---|---|
| `name` | Run name, logs only. | Anything recognisable. |
| `output_dir` | Root of the run folders `0_setup` ... `8_relocation`. Relative paths resolve against `base_dir`. | One directory per run. Put trial runs in separate directories so results can be compared. |
| `base_dir` | Root that every relative path in the config resolves against. `null` = the config file's directory. A relative value resolves against the config file's directory, and `~` is expanded. | Set it to the project folder (e.g. `~/projects/RK-NM`) so the config can live anywhere and the data, metadata and run paths stay short. |

## `parallel`

- **`n_workers`**: process-pool size for the parallel steps (`"auto"` = all
  physical cores). Memory, not CPU, is usually the limit. Rough per-worker
  peaks on 200 Hz data: QC ~0.9 GB, chunk ~0.45 GB, but **detection** can peak
  at ~17 GB per worker for a day window of ~70 channels. Run detection with a
  lower value (e.g. 3) if memory is short. RK-NM: 12 for most steps, 3 for
  detection (12 caused an out-of-memory kill).
- **`threads_per_worker`**: BLAS/OpenMP threads per worker. Keep 1 when
  `n_workers` > 1, so `n_workers x threads` does not exceed the core count.
- **`phasenet_device`**: `cpu` or `gpu` for PhaseNet. Use `gpu` if a CUDA build
  of TensorFlow is installed; picking is the only step that benefits.
- **`chunk_days`**: days per QC/detection work unit. 1 keeps per-worker memory
  bounded; larger values only reduce scheduling overhead.

## `data`

- **`waveform_dirs`**: folders searched recursively for miniSEED. List station
  folders individually when some must be left out (RK-NM: THQ2's EH1/EH2 are
  mislabelled copies of ARAZ EHZ, so only `THQ2/EHZ.D` is listed).
- **`file_pattern`**: glob for the files to index. Check that it matches every
  file: archives often mix `.mseed` and extension-less names (RK-NM: ARAZ,
  THQ2, WPRZ have no extension, and `*.mseed` silently dropped them). Narrow it
  by day number to keep channels outside the run period out of the index.
- **`start` / `end`**: inclusive UTC dates processed.

## `stations`

- **`stationxml` / `stationxml_dir` / `csv`**: station metadata, merged in that
  order. StationXML with a real instrument response is needed for magnitudes;
  a CSV gives positions only. Check that location codes match the miniSEED
  (RK-NM: NS12/NS13 had to be relabelled 00 -> 10) and that each channel's
  epochs cover the run period. Overlapping open epochs are closed
  automatically, with a warning.
- **`channel_map`**: maps non-standard component codes (e.g. borehole
  geophones) to Z/1/2 before picking.
- **`orientations`**: known horizontal azimuths, rotating 1/2 to N/E. Leave
  empty for unoriented sensors: PhaseNet works on unoriented horizontals.

## `qc`

QC flags are **informational**: they never remove data from detection. Only
`exclude_channels` does that. Check the QC report and the noise figures in
`1_qc/diagnostics/`.

- **`daily_stats`**: statistics computed per channel-day. The default set
  covers the flags below.
- **`spike_threshold_mad`** / **`spike_median_window`**: a sample is a spike if
  it departs from a running median (window in samples, odd) by more than this
  multiple of the MAD. Raise the threshold if real impulsive signals are being
  counted as spikes.
- **`max_spike_fraction`**: fraction of spike samples above which a
  channel-day is flagged. The default was tuned on synthetic data. Check the
  spike counts on a few known-good channels and set it just above them.
- **`min_availability_pct`**: coverage below which a channel-day is flagged.
- **`flat_run_min_samples`**, **`max_flat_fraction`**, **`max_flat_run_s`**:
  flatline detection: identical consecutive samples counted as flat, the
  fraction of flat samples allowed, and the longest flat run allowed (either
  test flags). Lower `max_flat_run_s` to catch short dropouts.
- **`clip_run_min_samples`**, **`max_clip_count`**: consecutive full-scale
  samples counted as clipping, and how many clipped runs are allowed (0 =
  report any).
- **`monthly_outlier_ratio`**: flags a station whose monthly statistic is more
  than this factor above (or below 1/this) the network median. On a network
  with mixed sensor types every station can look like an outlier; use the
  per-instrument noise plots instead of this flag in that case.
- **`exclude_channels`**: seed ids removed from detection (and flagged).
  Use for channels known to be bad, mislabelled or duplicated.
- **`noise.core_band`**: band (Hz) for hourly noise levels. Keep it above every
  sensor's corner frequency so like sensors are comparable with a scalar
  sensitivity.
- **`noise.low_band`**: band whose ratio to the core band tracks per-station
  drift (coupling, response changes); not comparable between sensor models.
- **`noise.window_s`**, **`noise.min_window_frac`**, **`noise.dead_rms_counts`**:
  noise window length (must divide a day), minimum data fraction per window,
  and the core-band RMS (counts) below which a window counts as dead hardware.
- **`instrument_groups`**: override the instrument group (taken from the
  StationXML sensor model) for the per-instrument noise plots. Needed only
  when the StationXML sensor model is wrong or missing.
- **`plot_exclude_stations`**: stations (`STA` or `NET.STA`) to leave out of
  the QC plots, e.g. stations with known-bad metadata that would otherwise
  dominate the colour scales and the network medians. The QC tables (and so
  detection's channel flags) still include them. Changing it does not rerun
  QC; run `openmeq diagnostics qc -c config.yaml` to redraw the plots.
  RK-NM 2026 Q1: `[NS12, NS13, NS14]`.

## `detection`

Network STA/LTA with coincidence triggering. Tune on one or two days that
contain known events (a reference catalogue helps), using
`station_trigger_rate.png` and `cf_examples_NN.png`.

- **`filter`** (`type`, `freqmin`, `freqmax`): applied before STA/LTA. Choose the
  band where local events stand above the noise. Keep `freqmax` below the
  Nyquist frequency of the **lowest-rate** station (RK-NM: 5-45 Hz, below the
  50 Hz Nyquist of the 100 Hz stations).
- **`sta`**: short window (s). About the duration of the first arrival's main
  energy: too short and every noise burst triggers. RK-NM: 0.05 s triggered
  every few seconds on every station (86,942 triggers/day); 0.2 s works.
- **`lta`**: long window (s), the background level. 10-50x `sta`; longer is
  more stable but adapts more slowly to changing noise. RK-NM: 10 s.
- **`trigger_on` / `trigger_off`**: STA/LTA ratio to start / end a trigger.
  Raise `trigger_on` until the station trigger rate in quiet periods is low but
  known events still trigger. RK-NM: 6 / 2.
- **`min_trace_coincidence`**: number of **channels** (not stations) that must
  trigger together; a 3-component station counts up to 3. About 3x the station
  count you want for an all-3-component network (RK-NM: 9).
- **`min_stations`**: distinct stations a detection must include, applied after
  coincidence. Stops one noisy 3-component station plus one other from making a
  detection. Choose from a reference catalogue: the lowest value that keeps the
  false-detection count acceptable. RK-NM (one test day, 63 reference events):
  `sta` 0.2 / `lta` 10 / on 6 / off 2 with 4 stations gave 52 detections, 43
  matched, 10 extra; a looser `lta` 5 / on 4 with 6 stations matched 44 but gave
  33 extra.
- **`station_weights`**: weight per station in the coincidence sum; lower it
  for persistently noisy stations instead of removing them.
- **`day_overlap_s`**: extra data read around each day so events at midnight
  are not lost. Must exceed `lta` plus a trigger's duration.
- **`dedup_window_s`**: triggers closer than this (s) are merged (STA/LTA often
  fires separately on P and S). Set above the network's largest S-P time plus
  trigger length (RK-NM: 10 s; S-P is under 5 s).

## `chunks`

- **`pre_s` / `post_s`**: seconds kept before / after each trigger in the event
  waveform chunks. Must contain the earliest P and latest S at the farthest
  station, plus margin for PhaseNet's edge effects. 30 s each is generous for a
  local network; check `pick_offset_hist.png` (picks should not crowd the
  chunk edges).
- **`max_read_window_s`**: detections further apart than this are read
  separately; only affects memory and speed.

## `phasenet`

- **`target_rate_hz`**: rate PhaseNet was trained at (100 Hz); fixed by the
  model.
- **`taper_pct`**, **`highpass_hz`**: preprocessing before decimation. Raise
  `highpass_hz` if long-period noise dominates; it must stay well below the
  event band.
- **`min_p_prob` / `min_s_prob`**: minimum pick probability. Low values (0.3)
  keep more picks; the location residual filters remove the bad ones later.
  Raise them if `wadati_prelocate.png` shows many outliers or many picks are
  removed during location.
- **`max_gap_fill_s`**: gaps up to this long are zero-filled before PhaseNet.
- **`pick_window_s`**: keep picks only within `[before, after]` s of the
  trigger. PhaseNet fires on the start of every chunk (taper + high-pass), and
  those artefacts can out-score real picks. Set `before` a few seconds before
  the earliest P, and `after` beyond the latest S. RK-NM: `[-5, 15]`; without
  it, 59% of P picks sat at the chunk start.
- **`pick_uncertainty_method`**: `peak_width` (default) sets each pick's
  timing uncertainty to the half-width of its PhaseNet probability peak at
  `pick_uncertainty_peak_frac` of the peak (0.5 = half the FWHM), clamped to
  `[pick_uncertainty_min_s, pick_uncertainty_max_s]`. Sharp onsets get
  smaller uncertainties and more weight in NLLoc. The width includes
  PhaseNet's own label width (Gaussian, sigma about 0.1 s), so values rarely
  drop below about 0.12 s. RK-NM 2026 Q1: P median 0.13 s (5-95%: 0.12-0.17),
  S median 0.15 s (0.12-0.24). `fixed` gives every pick `pick_uncertainty_s`.
- **`pick_uncertainty_s`**: fixed uncertainty for `fixed`, and the fallback
  when `result.h5` is missing or its station columns can't be verified (a
  warning gives the count).
- **`batch_size`**: unused.

## `nlloc`

### `velocity_model`, `grid`

- **`velocity_model.type` / `file`**: 1D layer file (or a pre-built 3D grid).
  The model controls absolute location accuracy more than any other setting;
  check `passX_res_vs_distance.png` for distance-dependent residual trends.
- **`grid.origin`**: grid centre `[lat, lon]`. Use the centre of the station
  spread.
- **`grid.size_km`**: `[x, y, z]` extent. Cover the stations with a margin of a
  few km, and make it deep enough for the deepest expected events. Events
  piling up at the grid top or bottom (see `passX_depth_hist_grid.png` and the
  depth sections) are grid-boundary locations, not real depths: NLLoc marks
  them "location on grid boundary". Extend the grid in that direction, or
  check the picks and velocity model for those events. RK-NM: 24 x 32 x 12 km;
  the Q1 pass maps show pile-ups at both the top and the 12 km floor.
- **`grid.spacing_km`**: node spacing. 0.1 km suits a local network; coarser
  grids are faster but limit location precision.

### `filter` (iterative pick and event filtering)

Location runs in two passes: **pass A** with P picks only, then **pass B** with
P and S. Each pass relocates, removes picks with large residuals, and
repeats.

- **`p_only.max_iter`**, **`p_and_s.max_iter`**: maximum relocate-and-filter
  iterations per pass. 5 is usually enough; check whether events still lose
  picks at the last iteration.
- **`max_residual_s`**: absolute residual limit for a pick. A few times the
  pick error plus model error (P: 0.1 s for a local network). S is picked less
  precisely: about twice the P value.
- **`residual_sigma`**: also remove picks more than this many standard
  deviations from the event's other residuals. 3 removes clear outliers only.
- **`p_and_s.s_predicted_window_s`**: before pass B, drop S picks more than
  this far from the S time predicted by the pass-A location. Removes S picks on
  the wrong arrival; widen it if the velocity model's Vp/Vs is uncertain.
- **`events.min_p_phases`**, **`min_phases`**, **`min_s_phases`**: minimum picks
  for an event to be kept: at least 4 P for a 4-parameter location, and at
  least one S, which constrains depth and origin time.
- **`events.max_rms_s`**: maximum RMS residual of an accepted event.
- **`events.max_gap_deg`**: maximum azimuthal gap. Loose values (285) keep
  events at the network edge, whose epicentres are less certain; tighten for a
  high-quality subset.
- **`events.max_err_h_km`**, **`max_err_z_km`**: maximum horizontal and depth
  uncertainty. Set by the precision you need; depth is always less certain.

The pass comparison PDF (`7_location/diagnostics/pass_comparison.pdf`) shows
pass A, pass B and accepted events side by side: use it to see which filters
remove which events.

### NLLoc search and error settings

- **`locsearch.*`**: octree search. The defaults suit local grids: start with
  10 x 10 x 10 cells, stop at 0.01 km nodes or 20,000 nodes. Increase
  `max_num_nodes` only if location PDFs look truncated.
- **`locgau.sigma_time_s`**, **`locgau2.*`**: modelling (velocity-model) error
  added to pick errors: a constant term and a term proportional to travel
  time (1%, between 0.05 and 2 s). Larger values give larger, more honest
  uncertainty ellipsoids when the model is poor.
- **`locmeth.method`**: `EDT_OT_WT` (equal differential time) is robust to
  outlier picks; keep it.
- **`locmeth.min_number_phases`**: minimum phases for NLLoc to attempt a
  location (4).
- **`locmeth.vp_vs_ratio`**: negative = use S travel-time grids built from the
  model's Vs. Set a positive value only to derive S from P with a fixed ratio.
- **`locmeth.max_dist_sta_grid`**, **`min_dist_sta_grid`**,
  **`max_number_phases`**, **`min_number_s_phases`**,
  **`max_num_3d_grid_memory`**, **`reject_duplicate_arrivals`**: rarely
  changed; see `config_reference.md`.
- **`locqual2err_thresholds_s`**: required by NLLoc's syntax but has no effect
  on OpenMEQ locations.
- **`obspyck_pluginpath`**: where ObsPyck finds NLLoc during review.

## `review`

- **`enabled`**: pause after location for manual review in ObsPyck. Use it for
  a reviewed catalogue or to check automatic picks on a sample.
- **`obspyck_config`**: path of the generated ObsPyck configuration.

## `magnitude`

Local magnitude from Wood-Anderson amplitudes. Check `res_vs_distance.png`
(station ML residual vs distance) and `amp_window_<event>.png`.

- **`ml.wood_anderson`**, **`amplitude_window`**, **`component`**: keep the
  defaults (simulated Wood-Anderson, S window, larger horizontal).
- **`ml.relation`**, **`hutton_boore.{a,b,c}`**, **`local.*`**: distance
  correction. Hutton & Boore (1987) is a reasonable start; if
  `res_vs_distance.png` shows a trend with distance, calibrate a local
  relation (`relation: local`, coefficients or a -log A0 table).
- **`ml.station_corrections`**: per-station ML corrections. Use the mean
  station residual once enough events exist; leave empty before that.
- **`ml.wa_sensitivity`**: Wood-Anderson gain; keep 2080 (IASPEI).
- **`ml.window_before_s` / `window_after_s`**: amplitude window around the S
  pick. Long enough to include the S peak at the farthest station, short
  enough to avoid the next event.
- **`ml.taper_pct`**: taper before response removal.
- **`ml.pre_filt`**: 4-corner pre-filter (Hz) for response removal. The low
  corners must sit above the band where the sensor response is too small to
  deconvolve stably: for short-period geophones, near or above their natural
  frequency (RK-NM: `[1, 2, 45, 49]` instead of `[0.5, 1, ...]`). The high
  corners must stay below the Nyquist frequency of the lowest-rate station.
- **`min_stations`**: station MLs needed for an event ML. 2 still gives small
  events a magnitude; raise it for a more robust catalogue.
- **`statistic`**: `median` is robust to one bad station.
- **`exclude_stations`**: measured but kept out of the event ML (e.g. a
  station with a doubtful response). Stations with no response are skipped
  automatically.

## `hypodd`

### `velocity_model`

Use the same model as NLLoc so relocations are consistent with the absolute
locations.

### `xcorr` (cross-correlation, `dt.cc`)

Differential times from waveform cross-correlation. The most precise data
hypoDD gets, and the easiest to get wrong. Check with the xcorr plots
(`diagnostics.relocate.xcorr_plots`), `dtcc_vs_dtct.png`, and hypoDD's cc
residual in `8_relocation/diagnostics/summary.csv` (`hypodd_rcc`). Any change
here makes the next `openmeq relocate` recompute every pair.

- **`window_p`**: `[start, end]` of the P window, seconds relative to the P
  pick, signed (`-0.1` = 100 ms before). Start early enough to include the
  onset when the pick is late (about the typical pick error); end after the
  first 1-2 cycles of P. Longer windows add coda, which differs between events
  and lowers cc; windows under ~2 cycles correlate by chance.
- **`window_s`**: the same for S. S is lower frequency with a less sharp onset
  (often inside the P coda), so give it more time after the pick than P.
- **`min_cc`**: minimum correlation coefficient kept. Raise it if the cc
  residual grows or the plots show poor alignments being kept; lower it toward
  0.7 if too few pairs link. Short windows reach high cc more easily and need a
  higher threshold.
- **`maxlag_s`**: largest shift searched. It must cover real pick errors but
  stay **below one cycle** of the dominant frequency: a longer search lets the
  correlation lock onto the neighbouring cycle (cycle skipping), which still
  passes `min_cc`. About half to two-thirds of a period.
- **`reject_edge_lags`**: drop measurements whose best lag is at the edge of
  the search (the real peak is outside it). Keep on.
- **`components`**: channels per phase, by the last letter of the mapped
  channel code. P on Z; S on every horizontal, keeping the best one.
- **`filter.freqmin` / `freqmax`**: band where P and S stand above the noise;
  look at event spectra. Keep `freqmax` below `resample_rate_hz / 2`.
- **`filter.corners`**, **`filter.zerophase`**: filter order (4) and
  zero-phase filtering. Keep zero-phase: a causal filter shifts waveforms.
- **`taper_pct`**, **`filter_edge_margin_s`**: taper and extra data around each
  window, both only to keep filter edge effects out of the window. The margin
  should be at least one cycle at `freqmin` (0.5 s = one cycle at 2 Hz).
- **`resample_rate_hz`**: all cached waveforms are resampled to this rate so
  any two events correlate sample for sample. Use the native rate of most
  stations; keep it at least 2.5x `freqmax`.

**RK-NM tuning** (now the OpenMEQ defaults). Five relocation runs compared:

| Run | P window | S window | Lag (s) | min_cc | dt.cc observations | Median cc | cc residual |
|---|---|---|---|---|---|---|---|
| Before the window fix | [-0.05, 0.2], applied as +0.025 to +0.225 s | [-0.05, 0.3], applied late likewise | 0.05 | 0.7 | 24,900 | 0.78 | 0.026 s |
| | [-0.05, 0.2] | [-0.05, 0.3] | 0.05 | 0.7 | 27,188 | 0.79 | 0.009 s |
| | [-0.1, 0.2] | [-0.1, 0.3] | 0.1 | 0.7 | 26,137 | 0.78 | 0.008 s |
| | [-0.1, 0.1] | [-0.1, 0.2] | 0.05 | 0.7 | 38,081 | 0.80 | 0.015 s |
| **Chosen** | **[-0.1, 0.1]** | **[-0.1, 0.2]** | **0.05** | **0.75** | 26,863 | 0.83 | 0.007 s |

The chosen run also rejected edge lags. A 0.1 s lag produced many shifts of
~70 ms, one full cycle of the ~14 Hz arrivals (cycle skipping); at 0.05 s,
about 11% of measurements sat at the lag limit until `reject_edge_lags` was
added.

### `ph2dt` (catalogue differential times and event pairing)

These scale with network size; [`hypodd_tuning.md`](hypodd_tuning.md) explains
the defaults (set for a ~10-15 km network) and how to rescale them.

- **`maxsep_km`**: maximum distance between two linked events. ~10 km is the
  usual start; reduce it for a tight cluster with a heterogeneous velocity
  structure.
- **`maxngh`**: maximum neighbours per event; more links make the system
  better connected but larger.
- **`minlnk`**: minimum shared picks for a strong link. Keep it at least
  `clustering.obsct`.
- **`minwght`**: minimum pick weight; 0 keeps every pick.
- **`maxdist_km`**: maximum event-station distance; about 1.5x the network
  aperture keeps every station.
- **`minobs` / `maxobs`**: minimum / maximum picks per linked pair. On a small
  network, `minobs: 1` keeps every link, and `maxobs` = 2 x station count keeps
  every pick (RK-NM: 46).

### `weighting_rows` (hypoDD iteration schedule)

Each row runs `niter` iterations with its own data weights (`wtccp`, `wtccs`,
`wtctp`, `wtcts`), residual cutoffs (`wrcc`, `wrct`: values >= 1 are
multiples of the residual standard deviation, -999 = off), maximum pair
separations (`wdcc`, `wdct`, km) and damping (`damp`). The default schedule
first relocates on catalogue data, then shifts weight to cross-correlation
data while tightening the cutoffs. Check `iteration_shifts.png` (shifts should
shrink row by row) and `res_final_by_type.png`. Rescale `wdcc`/`wdct` for a
network of a different size (see `hypodd_tuning.md`). If hypoDD reports a
condition number (CND) far above ~40-80, increase `damp`.

### Solution and data selection

- **`istart`**: 2 = start from the NLLoc locations. Keep.
- **`isolv`**: 2 = LSQR, which scales to large catalogues. Keep.
- **`iaq`**: 1 = remove events relocated above the surface (air-quakes). See
  `hypodd_tuning.md` on shallow datums.
- **`idat`**, **`ipha`**: data types (3 = cross-correlation and catalogue) and
  phases (3 = P and S). Restrict them only to test the effect of one data set.
- **`max_station_dist_km`**: maximum cluster-to-station distance; cover the
  whole network.
- **`clustering.*`**: hypoDD splits events into clusters linked by at least
  `obsct` catalogue picks; isolated events are not relocated ("not linked").
  Lower `obsct` (and `ph2dt.minlnk`) to link more events at the cost of weaker
  links.
- **`parallel_clusters`**: relocate clusters in parallel. Experimental; only
  useful for very large runs.

## `diagnostics`

- **`extended`**: write the extended QAQC figures for every step (per-step
  overrides: `diagnostics.<step>.extended`). On for tuning runs.
- **`n_examples`**: sample events for per-event figures.
- **`pick.pick_zoom.*`**: zoomed pick panels: half-width, number of events,
  and whether to show rejected picks.
- **`wells.*`**: overlay well heads and tracks on every map and depth section:
  `dir` holds `<Group>_well_heads.csv` (`name,lon,lat`) and
  `<Group>_well_tracks.csv` (`name,lon,lat,elevation_m`); `groups` selects
  groups, `colors` overrides colours, `labels` shows names (off when wells
  cluster).
- **`relocate.xcorr_plots.*`**: obspy cross-correlation plots for every
  `pair_interval`-th linked pair, up to `max_pairs` pairs. Choose the interval
  so the sample spreads across the whole run (about the number of linked pairs
  / `max_pairs`). Also available afterwards with `openmeq diagnostics xcorr`.
