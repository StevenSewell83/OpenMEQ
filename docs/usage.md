# OpenMEQ usage guide

This guide walks through the pipeline one step at a time: what each `openmeq
<step>` command reads and writes, when to run a step on its own versus the
whole chain (`openmeq run`), the manual review step's interactive behaviour,
where diagnostics land, and how to troubleshoot a broken run. For what to
install and a first end-to-end run, start with [README.md](../README.md). For
every config key's type/default/meaning, see
[config_reference.md](config_reference.md). For the underlying data contracts
(run directory layout, QuakeML conventions, file formats), see
[ARCHITECTURE.md](ARCHITECTURE.md); this guide summarises and links to it
rather than repeating it.

## The pipeline, end to end

Raw miniSEED goes in; a relocated, magnitude-tagged earthquake catalogue
comes out, through eight steps that each read the previous step's output and
write their own subfolder of `cfg.project.output_dir`:

```
qc -> detect -> chunk -> pick -> locate -> [review] -> magnitude -> relocate
```

The step folders are numbered to match the MicroQuake run layout (e.g.
`projects/RK-NM/run_2024_2026`), so the same stage has the same number in
runs from either tool:

| Folder | Written by |
|---|---|
| `0_setup/` | `qc`: file index, merged StationXML, station table, metadata report |
| `1_qc/` | `qc` |
| `2_detection/` | `detect` |
| `4_data_chunks/chunks/`, `4_data_chunks/chunks_pn/` | `chunk` (raw) and `pick` prep (PhaseNet-ready) |
| `5_ml_picks/` | `pick` |
| `7_location/` (NLLoc grids in `7_location/nlloc/`) | `locate` |
| `7b_review/` | `review` |
| `8_relocation/magnitude/` | `magnitude` |
| `8_relocation/` | `relocate` |
| `9_plots/` | run-level diagnostics |
| `logs/`, `config_snapshot.yaml` | every step |

3 and 6 are unused: OpenMEQ has no separate duplicate-removal or
pick-refinement stage.

Every step is idempotent: run standalone with `openmeq <step> -c config.yaml`,
or chained with `openmeq run -c config.yaml`. Re-running a step (or the whole
chain) with nothing changed does almost no work -- each step checks its own
inputs/config against a manifest and skips recomputation
(`openmeq.manifest.StepManifest`, "skip unchanged" below); pass `--force` to
recompute anyway.

Every step also writes `<step>/diagnostics/*.png` (and a CSV/text summary)
and `logs/<step>.log`, so you can always see what a step actually did without
re-running it. Optional extended QAQC figures (`diagnostics:` in the config)
are described in `docs/diagnostics_extended.md`.

## `openmeq run` -- the whole chain

```bash
openmeq run -c config.yaml
```

Chains every step's own `run(cfg, paths, **opts)` in order, sharing one
loaded config across all of them. On the synthetic demo dataset (8 stations,
2 days, ~20 events, `n_workers: 1`) a full chain takes about 80-95 seconds; a
second, no-op run (nothing changed) takes about 2-3 seconds, since every step
skips via its own manifest check. At the end it prints a summary table:

```
step       status    wall time  detail
--------------------------------------
qc         ok           37.80s  done
detect     ok           30.67s  done
chunk      ok            1.30s  done
pick       ok           12.48s  done
locate     ok            4.75s  done
review     skipped       0.01s  cfg.review.enabled is false
magnitude  ok            3.94s  done
relocate   ok            4.64s  done
--------------------------------------
total                   95.59s
```

Options:

- `--force` -- forward `--force` to every chained step (recompute even if
  nothing changed).
- `--from STEP` / `--to STEP` -- run only an inclusive sub-range of
  `qc, detect, chunk, pick, locate, review, magnitude, relocate` (e.g.
  `--from locate` to resume after fixing a config problem in the location
  step, or `--to locate` to stop before magnitudes/relocation). There is no
  `--only` option; use `--from X --to X` to run a single step through the
  `run` command's summary/logging path, or just call `openmeq X -c
  config.yaml` directly.
- `--skip-review` -- never touch the review step, even if
  `cfg.review.enabled: true`.

**If a step fails, the chain stops immediately** (later steps are never
attempted) and the error message tells you exactly how to resume, e.g.:

```
Error: openmeq run: step 'locate' failed: <cause>. Resume with `openmeq run -c config.yaml --from locate`.
```

Fix the underlying problem, then re-run with `--from <that step>` -- every
step before it already completed and will be skipped as unchanged (unless
you also pass `--force`).

### The `review` step's branching rule

`openmeq review`'s batch loop launches ObsPyck, an interactive GUI
subprocess, once per event (see "Manual review" below) -- `openmeq run`
**never launches it automatically** (it would hang a scripted/CI run waiting
for a human). Instead, on the `review` link in the chain:

- **`cfg.review.enabled: false`** (the default), or `--skip-review` was
  passed: nothing to do; `magnitude` falls through to
  `7_location/catalog_auto.xml` directly.
- **`cfg.review.enabled: true`, and nothing has been reviewed yet** (no
  `7b_review/events/*.xml` and no `7b_review/catalog_final.xml`): `openmeq run`
  logs a message telling you to run `openmeq review -c config.yaml`
  separately, then continues the rest of the chain using
  `7_location/catalog_auto.xml` for that run. Re-run `openmeq run` later (after
  reviewing some events) and it will pick up your review automatically.
- **`cfg.review.enabled: true`, and a prior separate `openmeq review`
  session already produced some reviewed events**: `openmeq run` safely
  (re)compiles `7b_review/catalog_final.xml` (`openmeq review --compile`,
  which never launches ObsPyck) so it reflects everything reviewed so far,
  then continues.

`magnitude` always runs against `7b_review/catalog_final.xml` if it exists,
else `7_location/catalog_auto.xml` -- this is the one and only place that
catalogue-selection rule is decided (`openmeq.magnitude.pipeline
.select_catalog_path`), so it applies identically whether you call `openmeq
magnitude` directly or through `openmeq run`.

## Step 1: `openmeq qc` -- data index and QA/QC

**Reads:** `cfg.data.waveform_dirs` (raw miniSEED), `cfg.stations` (StationXML
file/folder and/or CSV).

**Writes** (under `0_setup/` and `1_qc/`):

- `0_setup/index.json` -- a header-only miniSEED index every later step
  uses to find files, instead of re-scanning directories.
- `0_setup/stations.xml` / `.csv` -- the merged station inventory.
- `0_setup/metadata_report.json` -- channels with data but no metadata,
  and the reverse.
- `1_qc/availability.csv` -- per channel/day coverage, gaps, overlaps.
- `1_qc/daily_stats.csv` -- per channel/day RMS, mean absolute amplitude,
  isolated-impulse spike count, kurtosis, flat/clip run statistics.
- `1_qc/monthly_stats.csv` -- each station's stats against the network
  median, for outlier detection.
- `1_qc/channel_flags.json` -- the good/bad channel-per-day verdict
  (`"ok": true/false`, with `"reasons"` when not ok). This is a report for
  you to review: no later step drops data because of it. To keep a channel
  out of processing, list it in `qc.exclude_channels`.

**Diagnostics** (`1_qc/diagnostics/`): heatmaps with one labelled row per
channel (`STA.CHA`), grouped by instrument type (the StationXML sensor model,
or `qc.instrument_groups`) with the group name beside each block:

- `availability_heatmap.png`: % coverage per channel-day.
- `daily_spike_count.png` (as spike fraction) and `daily_kurtosis.png`:
  these don't depend on gain, so all instruments share one scale.
- `monthly_<stat>.png`: each channel's monthly value as a ratio to the median
  (log colour scale centred on 1×, value in each cell). For `rms`/`mean_abs`
  the values are divided by sensitivity and compared only within the
  channel's instrument group; a group with one station (marked `*`) is
  compared with its own median over the run. Daily RMS isn't plotted here:
  the noise heatmaps below show it per instrument group.
- `flag_overview.png`: each channel-day coloured by its highest-priority flag
  reason (low availability, flat, clipped, spikes, network outlier) or ok.

`qc.plot_exclude_stations`
leaves stations out of every QC plot (the tables keep them), and
`openmeq diagnostics qc -c config.yaml` redraws all of `1_qc/diagnostics/`
from the existing tables without re-reading any waveforms.

**PPSD (optional)** (`qc.ppsd.enabled: true`, or `openmeq diagnostics ppsd -c
config.yaml [--force]` at any time): an obspy probabilistic PSD per channel
(vertical only by default), stored per channel-day in `1_qc/ppsd/` so reruns
only add new days. `1_qc/diagnostics/ppsd/` gets, per channel, the PPSD
histogram with NLNM/NHNM (`<seed_id>_ppsd.png`), a spectrogram of every PSD,
and PSD level through time at a few periods, plus `ppsd_medians.png`: every
channel's median PSD, one panel per instrument group. It needs instrument
responses; channels without one are skipped and named in the log.

**Noise by instrument** (`1_qc/diagnostics/noise/`, data in
`1_qc/noise_daily.csv`): the MicroQuake `noise_daily` figures. Each channel-day
is cut into hourly windows; each window gets a band-limited RMS
(`qc.noise.core_band`, default 5-25 Hz), kurtosis and crest factor, and the
day is summarised by the median and p5/p95 across its hours. Levels are
divided by each channel's scalar StationXML sensitivity (dB re 1 m/s; dB re 1
count where there is none, marked †). Channels are grouped by the StationXML
sensor model, or by `qc.instrument_groups` where you set it (marked ‡), and
each group gets its own figures on its own colour/axis range, since levels
only compare between like instruments:

- `<group>_heatmap.png` -- daily noise level and daily transient spread
  (p95 − p5), station × day.
- `<group>_strips.png` -- one panel per station, median with a p5–p95 band.
- `<group>_overlay.png` -- the group's stations on one axis.
- `impulsiveness.png` -- every station on one scale (kurtosis and crest
  factor are scale-invariant, so no response is needed).

Vertical channels only.

Run standalone to sanity-check new data before detecting anything on it; run
first in the chain always (`detect` reads the `0_setup/index.json` it builds).

**Tuning note:** `qc.max_spike_fraction` (and the other `qc.max_*`/`qc.min_*`
thresholds) are tuned against this project's own synthetic dataset
(see `docs/performance.md` for the measured numbers) -- **expect to retune
these against your own real data's noise characteristics** before trusting
the QC flags on a new deployment.

## Step 2: `openmeq detect` -- network coincidence detection

**Reads:** `0_setup/index.json`, `cfg.detection`. Uses every channel with
data except those in `cfg.qc.exclude_channels`; QC flags do not gate
detection.

**Writes:** `2_detection/detections.csv` (trigger time, duration, coincidence
sum, participating stations).

**Diagnostics** (`2_detection/diagnostics/`): detections per day/hour,
coincidence-sum histogram, station participation.

Filters each day's data, runs recursive STA/LTA per channel, and applies
ObsPy's `coincidence_trigger` across all channels. A detection needs at
least `detection.min_trace_coincidence` triggering **traces** (channels,
weighted by `station_weights`; a 3-C station counts up to 3) and at least
`detection.min_stations` distinct stations. Day boundaries are padded
(`detection.day_overlap_s`) so a trigger isn't lost at midnight, and
triggers within `detection.dedup_window_s` are merged into one detection
(STA/LTA commonly fires separately on P and S).

### Choosing the detection parameters: `openmeq diagnostics detect-sweep`

```bash
openmeq diagnostics detect-sweep -c config.yaml [--day 2026-01-15 ...] \
    [--reference catalogue.csv] [--windows-csv 2_detection/coincidence_windows.csv] [--workers 3]
```

Runs the detector over the `detection.sweep` grid on a few sample days (needs
`0_setup/index.json` from `openmeq qc`) and writes `2_detection/sweep/`;
`detections.csv` is not touched. Each sample day x filter band is read and
filtered once, STA/LTA is computed once per `sta` x `lta_factor`, and every
`trigger_on` x `trigger_off` x `min_stations` combination reuses it, so the
threshold axes can be dense. Each grid point gives exactly the detections
`openmeq detect` would give with those values on that day.

- `lcurve_<band>_off<trigger_off>.png`: detections per day against
  `trigger_on` (log-log), one panel per `sta` x `lta`, one line per
  `min_stations`. Noise triggers fall away steeply, then the count levels off
  on the real events; the knee (x) is a good `trigger_on`. Grey: saturated
  (too many single-channel triggers to coincidence-test; noise).
- `recall_<band>_off<trigger_off>.png` (with `--reference` /
  `detection.sweep.reference`, a CSV with a `time` column): recall against
  extra detections per day. Detections match reference events with
  `detection - event` time in `match_window_s`.
- `sweep.csv` (every grid point: `det_per_day`, `persist_frac` = share of
  detections still found at the next higher `trigger_on`, and `recall`,
  `extras_per_day`, `precision`, `f1` with a reference), `knees.csv` (one row
  per curve), `candidates.csv` (best F1 with a reference, else the sharpest
  knees) and `items.csv` (the sample days).

For triggered archives (only short event windows recorded), pass the
`coincidence_windows.csv` of `scripts/detect_windows.py --windows-only` to
sweep inside those windows; keep the `lta` grid short, as the first `edge_s`
(default 2 x the largest `lta`) of each window is ignored. A day of many
200 Hz channels can use over 10 GB per worker; keep `n_workers` small.

## Step 3: `openmeq chunk` -- per-event waveform chunks

**Reads:** `2_detection/detections.csv`, `0_setup/index.json`.

**Writes:** `4_data_chunks/chunks/<event_id>/<event_id>.mseed` (raw, all channels,
original sampling rate, `chunks.pre_s`/`post_s` around the trigger) and
`4_data_chunks/chunks/completeness.csv` (fraction of the expected window actually present
per station).

**Diagnostics** (`4_data_chunks/chunks/diagnostics/`): completeness heatmap and box plot.

These raw chunks are kept and reused by review, magnitude and
cross-correlation later -- they are never regenerated once written unless
`detections.csv`/`0_setup/index.json` change or you pass `--force`.

## Step 4: `openmeq pick` -- PhaseNet picking

**Reads:** `4_data_chunks/chunks/<event_id>/*.mseed`, `cfg.stations.channel_map`,
`cfg.phasenet`.

**Writes:**

- `4_data_chunks/chunks_pn/<event_id>/<event_id>.mseed` -- PhaseNet-ready waveforms
  (merged/gap-filled, demeaned/detrended, tapered, optional high-pass,
  anti-alias-filtered and decimated to `phasenet.target_rate_hz`, channels
  mapped to Z/1/2 or rotated to ZNE if `stations.orientations` gives a known
  azimuth).
- `4_data_chunks/chunks_pn/data_list.csv` -- the PhaseNet input list.
- `4_data_chunks/chunks_pn/prep_report.csv` -- per event/station, how many components were
  found and how many had to be zero-filled.
- `5_ml_picks/phasenet_raw/` -- PhaseNet's own native output.
- `5_ml_picks/catalog_picks.xml` -- one QuakeML `Event` per detection with
  automatic P/S picks, probabilities and pick uncertainties (no origins
  yet -- that's `locate`'s job).
  Only picks within `phasenet.pick_window_s` (default `[-5, 15]` s) of the
  detection trigger are kept -- PhaseNet fires on the start of every chunk
  -- then one pick per station and phase, the highest-probability one.

**Diagnostics** (`5_ml_picks/diagnostics/`): P/S pick counts per station,
pick-probability histograms, picks-per-event histogram, and example
waveform + probability plots for a sample of events.

PhaseNet runs on CPU in this project's environment (TensorFlow 2.11 has no
matching CUDA build available here) -- `parallel.phasenet_device: cpu` is the
supported/tested setting; `gpu` is accepted by the schema but not exercised.
Preparation is parallel (process pool); the PhaseNet model itself runs as one
TensorFlow process (`predict.py`), always invoked with `--batch_size=1`
regardless of `phasenet.batch_size` (see the `phasenet` section of
[config_reference.md](config_reference.md) for why).

## Step 5: `openmeq locate` -- two-pass NLLoc location and filtering

**Reads:** `5_ml_picks/catalog_picks.xml`, `cfg.nlloc` (velocity model, grid,
filter thresholds). The first time this step runs (or whenever
`cfg.nlloc.velocity_model`/`cfg.nlloc.grid` change), it builds/refreshes the
NLLoc travel-time grids under `7_location/nlloc/` (equivalent to running
`scripts/setup_nlloc.py` yourself -- see below).

**Writes:**

- `7_location/catalog_auto.xml` -- one preferred `Origin` per surviving event.
- `7_location/summary.csv` -- `event_id,time,lat,lon,depth_km,rms,gap,n_p,n_s,
  err_h_km,err_z_km,status,reason`, one row per input event, **never
  dropped** (a rejected or never-converged event still gets a row, with
  numeric fields left empty rather than the row omitted).
- `7_location/removal_log.csv` -- every pick removed at every pass/iteration and
  why.
- `7_location/passA/<event_id>/iter_NN/`, `7_location/passB/<event_id>/iter_NN/` --
  the raw NLLoc work directories (obs file, control file, `.hyp` output) for
  each relocation iteration, useful for debugging one event by hand.

Two passes, each iterating "relocate, drop outlier picks, repeat" until
stable or `max_iter` is reached:

- **Pass A (P only):** locate with P picks; drop P residuals beyond
  `nlloc.filter.p_only.max_residual_s` or `residual_sigma`·σ; reject events
  with fewer than `nlloc.filter.events.min_p_phases` P picks.
- **Pass B (P + S):** screen S picks against the S arrival time predicted by
  pass A's location (drop those beyond
  `nlloc.filter.p_and_s.s_predicted_window_s`); locate with P + S; filter P
  and S residuals separately (`nlloc.filter.p_and_s.p`/`.s`).
- **Event rejection** (checked after each relevant pass, first failure
  wins): `min_p_phases`, `min_phases`, `min_s_phases`, `max_rms_s`,
  `max_gap_deg`, `max_err_h_km`, `max_err_z_km` (`nlloc.filter.events`).

**Diagnostics** (`7_location/diagnostics/`, per pass, `passA_`/`passB_`
prefixes): P/S residual histograms (overall and per station), an event map +
E-W/N-S depth-section figure, RMS/gap/error histograms, and pick/event
removal-by-reason bar charts, plus `pass_comparison.pdf` (every one of
those figures with pass A and pass B together on shared axes, bins and colour
scale, plus the accepted events as a third map row; one figure per page) and
`summary.txt` with flat
`events_in`/`events_accepted`/`events_rejected`/per-pass counts.

### `scripts/setup_nlloc.py` -- NLLoc grid setup on its own

```bash
python scripts/setup_nlloc.py -c config.yaml [--force]
```

Builds the shared velocity/travel-time grids `locate` (and manual review)
need, without running detection/picking/location: for `nlloc.velocity_model.
type: 1d`, runs `Vel2Grid` once from the layer file; for `3d`, validates and
links in your own NLLoc grid instead. Then runs `Grid2Time` per
(station, phase) in parallel, and refreshes the model link ObsPyck's own
NLLoc button needs (`nlloc.obspyck_pluginpath`). Useful to run standalone
after changing the velocity model or the station list, on day one of a
project before any events exist, or to pre-build grids before a large
`locate` run. Idempotent (`--force` to rebuild regardless).

## Step 6: `openmeq review` -- manual review (optional)

**Reads:** `7_location/catalog_auto.xml`, the raw `4_data_chunks/chunks/<event_id>/*.mseed`,
`cfg.review`, `7b_review/state.json` (resume state).

**Writes:** `7b_review/events/<event_id>.xml` (one file per manually-reviewed
event that was actually re-picked/re-saved), `7b_review/state.json`, and, with
`--compile`, `7b_review/catalog_final.xml`.

This step is entirely optional (`cfg.review.enabled: false` by default) and
is the only interactive step in the pipeline. It launches
[ObsPyck](https://github.com/joachimwassermann/obspyck) once per
not-yet-reviewed accepted event, with the raw chunk and the automatic
QuakeML's picks preloaded, so you can re-pick and relocate by hand (ObsPyck
runs NLLoc internally; there is no separate NLLoc re-run afterward).

**A real display is required.** ObsPyck is a real Qt GUI: run `openmeq
review` from a machine/session with `$DISPLAY` set (a desktop session, X
forwarding, or VNC). If no display is available, it falls back to
`xvfb-run` (confirmed bit-identical to a real display in this project's own
testing, `docs/obspyck_notes.md` §1); if neither works, the whole `openmeq
review` run fails immediately with a clear error rather than hanging.
(`QT_QPA_PLATFORM=offscreen`, used by this project's own automated tests, is
not useful for a real interactive session -- there is nothing to look at.)

**ObsPyck key bindings** (from the generated `obspyck.cfg`'s `[keys]`
section -- spliced byte-identical from ObsPyck's own shipped `example.cfg`):

| Key | Action |
|---|---|
| `a` | Set P pick at cursor |
| `s` | Set pick uncertainty/error |
| `q` | Delete pick |
| `control` | Switch phase (P/S) |
| `y` / `x` | Previous / next waveform stream |
| `0`-`3` | Set pick weight 0-3 |
| `+` / `-` | Set polarity up / down |
| `i` / `e` | Set onset impulsive / emergent |
| `shift` | Switch wheel-zoom axis |
| `alt` + scroll wheel | Zoom |

The map view's "do NLLoc" button relocates the event with your edited picks
using the same NLLoc control file/grids `setup_nlloc.py` built. Saving
("save Event locally") writes a timestamped QuakeML file in that event's own
scratch work directory (`7b_review/work/<event_id>/`); the batch loop detects
the save by diffing that directory's file list before/after ObsPyck exits
(ObsPyck's own exit code cannot distinguish "saved" from "closed without
saving" -- both exit `0`), not by trusting the exit code alone.

**When no save is detected**, you're prompted at the terminal:

```
Event <event_id>: no save detected. [s]kip / [r]eject / [q]uit batch?
```

- **`s` (skip)** -- leave the automatic pick/location as-is; you can review
  this event again later with `--reset EVENT_ID` or `--reset-all`.
- **`r` (reject)** -- mark the event bad; `--compile` will flag it rejected
  in `catalog_final.xml` (never deleted, per ARCHITECTURE.md §5's "never
  delete, always flag" convention).
- **`q` (quit)** -- stop the batch run now. The in-progress event is left
  with **no** `state.json` entry (neither skipped nor done), so it's offered
  again first next time.

**Resuming:** `7b_review/state.json` records `"done"`/`"skipped"`/`"rejected"`
per event and is written after every event, not just at the end -- a killed
session loses at most the one event that was in progress when it died.
Once an event has a `state.json` entry it is **never** re-offered
automatically; use `--reset EVENT_ID` (repeatable) or `--reset-all` to clear
entries and have them offered again.

**Compiling:**

```bash
openmeq review -c config.yaml --compile
```

Merges `7b_review/events/*.xml` (each one replaces the automatic event
wholesale -- origin, picks, everything) and any `"rejected"` decisions over
`7_location/catalog_auto.xml`, into `7b_review/catalog_final.xml`. Safe to run any
time (never launches ObsPyck); `openmeq run` calls this automatically once
anything has been reviewed (see "The `review` step's branching rule" above).
An event never offered for review, or still `"skipped"`, passes through
unchanged.

## Step 7: `openmeq magnitude` -- local magnitude (ML)

**Reads:** `7b_review/catalog_final.xml` if it exists, else
`7_location/catalog_auto.xml` (override with `--catalog PATH`, e.g. to compute
magnitudes on hypoDD-relocated output), `0_setup/stations.xml` (needs a
real instrument response), `4_data_chunks/chunks/<event_id>/*.mseed`, `cfg.magnitude`.

**Writes:** `8_relocation/magnitude/catalog_mag.xml` (with `Amplitude`/`StationMagnitude`/
`Magnitude` objects added), `8_relocation/magnitude/station_magnitudes.csv`.

**Diagnostics** (`8_relocation/magnitude/diagnostics/`): station ML vs. distance, station
residuals (useful for deriving `magnitude.ml.station_corrections`), a
Gutenberg-Richter magnitude-frequency plot, and ML through time.

Per event: remove instrument response and simulate a Wood-Anderson
seismograph, measure peak amplitude in a window around the S pick (max of
horizontals, falls back to Z), convert to ML via the configured relation
(`hutton_boore` or `local`, `magnitude.ml.relation`), and combine station
MLs (median by default) into the event ML. Stations listed in
`magnitude.exclude_stations` are still measured and reported in
`station_magnitudes.csv` (`excluded=True`) but left out of the event ML;
stations with no instrument response are skipped (review gives ObsPyck a
placeholder response for them, `7b_review/stations_review.xml`, so they stay
pickable). `8_relocation/magnitude/catalog_mag.xml` is
what `relocate` reads next, so hypoDD's relocated catalogue carries
magnitudes through too.

## Step 8: `openmeq relocate` -- hypoDD relocation

**Reads:** `8_relocation/magnitude/catalog_mag.xml`, the raw chunks (for
cross-correlation), `cfg.hypodd`.

**Writes:** `8_relocation/catalog_dd.xml` (a new `smi:openmeq/hypodd` origin per
relocated event, `event.preferred_origin_id` updated, magnitudes carried
over unchanged), `8_relocation/summary.csv`, plus the hypoDD/ph2dt working files
under `8_relocation/work/` (`station.dat`, `phase.dat`, `event.dat`,
`ph2dt.inp`/`hypoDD.inp`, `dt.ct`/`dt.cc`, `hypoDD.log`, `hypoDD.reloc`) and
the per-event cross-correlation waveform cache under `8_relocation/cache/`.

**Diagnostics** (`8_relocation/diagnostics/`): before/after location maps and
depth sections (the "after" map coloured by relocation shift), a `dt.cc`
CC-value histogram, a relative-shift histogram, and per-iteration RMS/
connectivity/air-quake-and-condition-number plots parsed from `hypoDD.log`.

Chains: build the cross-correlation waveform cache for every accepted event
-> `ph2dt` (catalogue differential times, `dt.ct`) -> cross-correlation
(`dt.cc`) -> `hypoDD` itself -> merge `hypoDD.reloc` back into QuakeML. Each
sub-step has its own skip-unchanged manifest key, so re-running after only
changing (say) the weighting schedule re-runs just `hypoDD`, not the whole
chain.

### Read this before trusting the default `hypodd`/`ph2dt` tuning on real data

**The `hypodd`/`ph2dt` distance and weighting defaults (the `hypodd` section
of [config_reference.md](config_reference.md)) are tuned against this
project's own synthetic test network, whose 20 events
are spread over about 19 km (median nearest-neighbour distance ~2.3 km, e.g.
`hypodd.ph2dt.maxsep_km: 10`, `weighting_rows`' final-row `WDCC`/`WDCT`
bottoming out at 6 km).** Real microseismic sequences are usually **much
tighter clusters -- tens to hundreds of metres**, not kilometres. Applying
these defaults unscaled to a genuinely tight swarm will not break anything
outright, but will under-resolve the fine-scale double-difference structure
those distance cutoffs are meant to control.

**Read [docs/hypodd_tuning.md](hypodd_tuning.md) before running `relocate` on
real data** -- it explains every distance/weighting default's derivation and
gives the explicit scaling rule (measure your own network's aperture and
event spread, then rescale `ph2dt.maxsep_km`/`maxdist_km`, the weighting
schedule's final-row `WDCC`/`WDCT`, and `ph2dt.maxobs`/`minobs` accordingly)
plus a troubleshooting table for symptoms like "most events end up removed
as air-quakes" or "hypoDD reports many small clusters instead of one".

The cross-correlation filter band (`hypodd.xcorr.filter`, default 2-20 Hz)
was likewise chosen from this project's own **synthetic** P/S pulse spectra
-- retune it against your real events' own spectra (`docs/ARCHITECTURE.md`
§6g has the analysis this default came from).

### `scripts/setup_hypodd.py` -- hypoDD setup on its own

```bash
python scripts/setup_hypodd.py -c config.yaml [--nlloc-to-simul] [--force]
```

Writes `station.dat`, `ph2dt.inp` and `hypoDD.inp`, refreshes the persisted
event-id/station-label maps, and checks the dataset against hypoDD/ph2dt's
currently *compiled* array limits (`include/hypoDD.inc`, `ph2dt.inc`,
`vel3d.inc`), recompiling with headroom if exceeded. For a 3D run, also
validates that every station/event falls inside the 3D model's node domain
(a real memory-safety hazard in hypoDD's own ray tracer otherwise --
`docs/hypodd_3d_notes.md`). `--nlloc-to-simul` converts a configured NLLoc 3D
grid into hypoDD's SIMUL2000 `vel3D.dat` format first. Like
`setup_nlloc.py`, this is meant to be runnable independently and before any
events exist (e.g. right after choosing a velocity model), not only as part
of `relocate`.

## Resuming and skip-unchanged

Every step's `run(cfg, paths, **opts)` checks a `StepManifest`
(`<step>/manifest.json`) keyed on a fingerprint of the relevant config
section plus its input files' mtimes/sizes, and skips recomputation when
nothing relevant changed -- this is what makes a second `openmeq run`
(nothing changed) take a couple of seconds instead of a couple of minutes.
Pass `--force` to any single-step command, or to `openmeq run`, to recompute
regardless. `openmeq relocate`'s own sub-steps (waveform cache, `ph2dt`,
cross-correlation, `hypoDD`) each have their own manifest key, so changing
only `hypodd.weighting_rows` re-runs only the `hypoDD` invocation itself, not
`ph2dt`/cross-correlation.

`openmeq run --from STEP` combines with this naturally: everything before
`STEP` is untouched (its own manifest still applies if you also re-run it
directly), and the chain resumes exactly where a failure or an interrupted
run left off.

## Parallelism settings

The `parallel` config section (see its entry in
[config_reference.md](config_reference.md)) controls every step's process
pool:

- `n_workers: auto` uses all physical cores; set an explicit integer to cap
  it (the synthetic demo's own `config.yaml` uses `1`, since the dataset is
  tiny and process-pool overhead would dominate).
- `threads_per_worker` caps BLAS/OpenMP/TensorFlow threads *inside* each
  worker process, to avoid oversubscribing the machine when `n_workers` > 1
  (each worker's environment is set before NumPy/BLAS is imported, via the
  `spawn` process-pool start method).
- `chunk_days` batches QC/detection's per-task work.
- `phasenet_device: cpu` is this project's tested/supported setting (see
  "Step 4" above).

Some steps are inherently memory-bandwidth-bound rather than CPU-bound at
high worker counts (see `docs/performance.md` for a measured example on QC's
daily-statistics computation) -- more workers helps up to a point, then
plateaus or even regresses; there is no one right `n_workers` for every step
on every machine, so it's worth benchmarking (`scripts/benchmark.py`) your
own hardware and dataset size rather than assuming `auto` is always fastest.

## Troubleshooting

### `ExternalToolMissing: External tool '<name>' not found`

Raised by `openmeq.external.bin_path` when a required binary (`NLLoc`,
`Vel2Grid`, `Grid2Time`, `hypoDD`, `ph2dt`, ...) isn't found in
`<external_dir>/bin/` or on `$PATH`. `external_dir` resolves, in order:

1. `$OPENMEQ_EXTERNAL_DIR`, if set.
2. `<repo root>/external` (the standard `pip install -e .` layout).

Common causes:

- `./install.sh` was never run, or was run with `--skip-external` (which
  deliberately skips the PhaseNet clone, NonLinLoc build and hypoDD build --
  fine for CI/offline dev, but every NLLoc/hypoDD-dependent step will then
  raise this). Re-run `./install.sh` without `--skip-external`.
- `$OPENMEQ_EXTERNAL_DIR` is set but points at the wrong directory (e.g. a
  different worktree/checkout's `external/`).
- The binary exists but isn't executable (a permissions issue after copying
  `external/` around by hand) -- `bin_path` checks `os.access(path, os.X_OK)`
  as well as existence.

### `requires_nlloc` / `requires_hypodd` / `requires_phasenet` / `requires_gui` / `slow` test markers

These `pytest` markers (declared in `pyproject.toml`) document, for a
developer running the test suite, which external piece a test needs; they
are labels, not automatic skips on their own. Each test module that needs
one of these computes its own availability check (e.g. "is
`external/bin/hypoDD` present", "is `obspyck`/PyQt5 importable") and pairs it
with an explicit `pytest.mark.skipif(..., reason=...)` alongside the marker,
so the test cleanly skips (rather than erroring) when that piece isn't
available in the current environment -- e.g. a `requires_gui` test skips on a
headless CI runner with no display and no `xvfb-run`, and a `requires_hypodd`
test skips if `external/HYPODD`/`external/bin/hypoDD` wasn't built. `slow` is
purely advisory (a long-running test, e.g. a full synthetic-dataset
end-to-end run) -- use `pytest -m "not slow"` to skip those for quick local
iteration.

### A config key raises a validation error

Every config section is `extra="forbid"` (`openmeq.config`), so a misspelled
or unknown key raises a validation error naming the exact bad path (e.g.
`hypodd.ph2dt.max_sep_km` instead of `maxsep_km`) rather than being silently
ignored. Check the key's exact spelling/path against
[config_reference.md](config_reference.md).

### A relative path in my config doesn't resolve where I expect

Every relative path in the YAML resolves against **the config file's own
directory**, not your current working directory when you run `openmeq`
(`docs/ARCHITECTURE.md` §2). Use an absolute path if you need something
resolved elsewhere.
