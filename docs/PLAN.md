# OpenMEQ — Microseismic Processing Workflow: Build Plan

Status: **DRAFT v3 for review** · 2026-09-28

> Add comments anywhere, e.g. as `> **Comment:** ...` blocks or `<!-- ... -->`.

## Changes in v3

- **HYPODD.zip slimmed** to source, include files, docs and example 1: 1.7 MB, down from 26 MB. It was test-compiled after slimming. The original is kept locally as `HYPODD_full.zip` and git-ignored.
- **Magnitudes:** ML with a choice of Hutton & Boore (1987) or a local calibration (§3 `magnitude`).
- **hypoDD step** uses the local hypoddpy code as a template only, with each part checked before it is used (§6 Step 8, §10).
- **Parallel processing** throughout, with speed as a design goal: new §4 and a `parallel` config block.

## Changes in v2 (from review comments)

- **hypoDD:** built from the bundled `HYPODD.zip` (hypoDD 2.1 with 3D ray tracing, SIMUL2000-format models). Both 1D and 3D models are supported.
- **Location:** a P-only NLLoc pass with P filtering runs first, then a P + S pass. P and S residual filters are set separately.
- **Diagnostics:** plots at every step, and at every filtering pass during location (residual distributions, event locations).
- **New magnitude step** (Step 7).
- **Station metadata:** accepts a StationXML file, a folder of StationXMLs, or a CSV.
- **Borehole orientations:** treated as mostly unknown.
- **3D models:** NLLoc grid format for NLLoc, SIMUL2000 format for hypoDD.
- **hypoDD licence:** confirmed.

---

## 1. Key constraints (from investigation)

| Topic | Finding | Consequence |
|---|---|---|
| PhaseNet | AI4EPS/PhaseNet is **not on PyPI**; pins `tensorflow==2.11.0`, `numpy==1.26.4`. TF 2.11 supports Python ≤ 3.10. Pretrained weights (`model/190703-214543`) live in the git repo, not the package. | Project uses **Python 3.10**. Installer clones PhaseNet at a pinned commit. |
| PhaseNet input | Model runs at **100 Hz**. Built-in resampling is linear interpolation. | We decimate 200 → 100 Hz ourselves (anti-alias filter + factor 2) during chunk prep. |
| ObsPyck | On PyPI (0.8.2). PyQt5 not declared as a dependency. CLI supports `--config`, `--time`, `--duration`, `--event`. Runs NLLoc internally from a models directory; can save events locally as QuakeML. | Batch wrapper drives ObsPyck per event; no separate NLLoc re-run after manual picking. PyQt5 added explicitly. |
| NonLinLoc | Not on PyPI or conda-forge. C, CMake build (`alomax/NonLinLoc`). | Compiled by installer. |
| hypoDD | `HYPODD.zip` (repo root, slimmed) contains hypoDD 2.1 source with 3D ray tracing (`ray_3d`, `get_vel3d`, SIMUL2000 `vel3D.dat`) plus `ph2dt`. **Test build with gfortran 11.4 succeeded** (`hypoDD`, `ph2dt`). Array sizes are fixed at compile time (`include/hypoDD.inc`, e.g. `MAXEVE=8000`; 3D grid sizes in `include/vel3d.inc`). | Installer unpacks and compiles from the zip. `setup_hypodd.py` checks dataset/model sizes against the compiled limits and recompiles with larger limits if needed. |
| Other packages | ObsPy, PyQt5, ObsPyck all compatible with Python 3.10. | — |

> **HYPODD.zip** was slimmed to `src/`, `include/`, `util/`, `doc/` and `examples/example1` (used by the install smoke test): 1.7 MB, 122 files. Site runs, site models, the other examples and the prebuilt `.o` files were removed. The full original stays local as `HYPODD_full.zip` (git-ignored).

---

## 2. Installation (single command)

```bash
git clone https://github.com/StevenSewell83/OpenMEQ.git && cd OpenMEQ && ./install.sh
```

`install.sh`:

1. Creates a Python 3.10 environment in `.venv` using `uv` (fetches 3.10 if absent).
2. `pip install -e .` — ObsPy, TensorFlow 2.11, PyQt5, ObsPyck, PyYAML, etc.
3. Clones PhaseNet (pinned commit) into `external/PhaseNet` and installs it, so the model weights are present.
4. Downloads and compiles NonLinLoc (cmake) into `external/bin`.
5. Unpacks `HYPODD.zip` into `external/HYPODD`, runs `make clean && make`, and copies `hypoDD` and `ph2dt` to `external/bin`.
6. Runs a smoke check: all imports succeed, all binaries execute, and hypoDD example 1 runs.

System prerequisites: `gcc`, `gfortran`, `cmake`, `git`, `unzip`.

---

## 3. User configuration (single YAML file)

All key parameters in one well-commented file, e.g. `config/example.yaml`. Sketch:

```yaml
project:
  name: my_survey
  output_dir: ./runs/my_survey

parallel:
  n_workers: auto                       # auto = all physical cores; or an integer
  threads_per_worker: 1                 # BLAS/OpenMP threads inside each worker (avoids oversubscription)
  phasenet_device: cpu                  # cpu | gpu
  chunk_days: 1                         # unit of work for QC and detection

data:
  waveform_dirs: [/data/mseed]          # miniSEED, searched recursively
  file_pattern: "*.mseed"
  start: 2025-01-01
  end:   2025-12-31

stations:
  # Any of these; they are merged. StationXML (with response) is required for magnitudes.
  stationxml: ./meta/stations.xml       # a single file ...
  stationxml_dir: ./meta/stationxml/    # ... and/or a folder of StationXML files
  csv: ./meta/stations.csv              # net,sta,loc,lat,lon,elev_m,depth_m
  channel_map:                          # borehole / non-standard components → Z/1/2
    DPZ: Z
    DP1: "1"                            # orientation unknown → treated as unoriented horizontals
    DP2: "2"
  orientations: {}                      # optional known azimuths, e.g. {XX.BH01..DP1: 37.0}

qc:
  daily_stats: [rms, mean_abs, spike_count, kurtosis]
  spike_threshold_mad: 10
  min_availability_pct: 80
  exclude_channels: []

detection:
  sta: 0.05
  lta: 1.0
  trigger_on: 3.5
  trigger_off: 1.5
  filter:
    type: bandpass
    freqmin: 5
    freqmax: 80
  coincidence_sum: 4
  station_weights: {}

chunks:
  pre_s: 30                             # default 30 s either side of detection
  post_s: 30

phasenet:
  target_rate_hz: 100
  taper_pct: 0.05
  highpass_hz: 1.0
  min_p_prob: 0.3
  min_s_prob: 0.3
  batch_size: 20

nlloc:
  velocity_model:
    type: 1d                            # 1d | 3d
    file: ./models/vel1d.txt            # 1d: layer file; 3d: NLLoc grid (.hdr/.buf) root name
  grid:
    origin: [lat, lon]
    size_km: [20, 20, 10]
    spacing_km: 0.05
  filter:
    p_only:                             # pass A: P picks only
      max_iter: 5
      max_residual_s: 0.10
      residual_sigma: 3
    p_and_s:                            # pass B: P + S picks
      max_iter: 5
      s_predicted_window_s: 0.3         # drop S picks this far from the S time predicted by the pass-A location
      p:
        max_residual_s: 0.10
        residual_sigma: 3
      s:
        max_residual_s: 0.20
        residual_sigma: 3
    events:                             # applied after pass A (min_p_phases) and pass B (rest)
      min_p_phases: 4
      min_phases: 6
      min_s_phases: 1
      max_rms_s: 0.1
      max_gap_deg: 270
      max_err_h_km: 1.0
      max_err_z_km: 2.0

review:
  enabled: false
  obspyck_config: ./config/obspyck.cfg

magnitude:
  type: ml                              # ml (Mw from spectral fitting could be added later)
  ml:
    wood_anderson: true                 # simulate Wood-Anderson from StationXML response
    amplitude_window: s                 # window around the S pick (or P→S+coda)
    component: horizontal               # max of horizontals, falls back to Z
    relation: hutton_boore              # hutton_boore | local
    # ML = log10(A) + a·log10(R/100) + b·(R-100) + c   (A in mm Wood-Anderson, R hypocentral km)
    hutton_boore: {a: 1.110, b: 0.00189, c: 3.0}
    local:                              # used when relation: local
      coefficients: {a: null, b: null, c: null}
      file: null                        # or a CSV table of -logA0 vs distance (interpolated)
    station_corrections: {}             # {NET.STA: correction}, or a CSV path
  min_stations: 2
  statistic: median                     # event ML = median of station MLs

hypodd:
  velocity_model:
    type: 1d                            # 1d | 3d
    file: ./models/vel1d.txt            # 1d: layer file; 3d: SIMUL2000 vel3D.dat
  xcorr:
    window_p: [-0.05, 0.2]
    window_s: [-0.05, 0.3]
    min_cc: 0.7
    maxlag_s: 0.05
    components: {P: [Z], S: ["1", "2", N, E]}
  ph2dt:
    maxsep_km: 1.0
    maxngh: 10
    minlnk: 8
```

CLI: `openmeq <step> -c config.yaml` runs one step; `openmeq run -c config.yaml` runs the whole chain. Each step writes to its own subfolder of `output_dir`, so any step can be re-run in isolation.

---

## 4. Parallel processing and performance

Speed is a design goal. Every step is split into independent units of work, which run in a process pool (`concurrent.futures.ProcessPoolExecutor`) sized by `parallel.n_workers`. Workers are given file paths and parameters, not waveform objects, so little data is copied between processes. Each worker sets its BLAS/OpenMP/TensorFlow thread counts from `threads_per_worker`, so the machine isn't oversubscribed.

| Step | Unit of parallel work | Notes |
|---|---|---|
| Index | miniSEED file | Header-only reads (`obspy.read(..., headonly=True)`) |
| QC | channel × day | Vectorised NumPy statistics; plots rendered in parallel afterwards |
| Detect | day (all stations) | Each worker reads its day plus overlap so triggers aren't lost at day boundaries; results merged and de-duplicated |
| Chunk | detection | Detections grouped by source file, so each file is opened once per worker |
| PhaseNet prep | event | Preprocessing in the process pool |
| PhaseNet run | batch | One TensorFlow process, batched (`batch_size`), CPU or GPU. Prep runs ahead of prediction so the model is never idle |
| Locate | event | Each worker runs NLLoc in its own temporary directory, with the travel-time grids shared read-only; the P-only → P+S passes are iterated per event, independently |
| Review | — | Interactive, run one event at a time |
| Magnitude | event | Instrument responses cached per channel in each worker |
| xcorr (dt.cc) | event pair (batched) | The heaviest step. Chunks are pre-loaded and pre-filtered once into an on-disk cache (per-event `.npz`); pairs are processed in batches per worker; results are written incrementally so a run can resume |
| hypoDD | cluster (optional) | hypoDD itself is single-threaded; independent clusters (from ph2dt/linking) can be run as separate hypoDD jobs in parallel |

Also for speed:
- File lookups go through the index, so directories are never re-scanned.
- Steps only reprocess what has changed: they skip outputs already produced for the same inputs and config.
- Progress bars and timing per step are logged.
- A benchmark script on the test dataset tracks runtime as the code grows.

---

## 5. Diagnostics (all steps)

Every step writes a `<step>/diagnostics/` folder with PNG plots and a short text/CSV summary, generated automatically at the end of the step. Plotting code lives in `openmeq/plotting/` so the same map, section and histogram functions are reused across steps.

| Step | Diagnostic plots |
|---|---|
| QC | Availability heatmap; daily statistics time series per channel; monthly station box plots and rankings |
| Detect | Detections per day/hour; coincidence-sum histogram; station participation |
| Chunk | Chunk completeness per station (fraction of the window present) |
| Pick | P/S pick counts per station; pick-probability histograms; picks per event; example waveform + probability plots (sample of events) |
| Locate | **Per pass:** P and S residual distributions (all, per station); event map + E–W and N–S depth sections; RMS/gap/error histograms; picks and events removed per pass |
| Review | Auto vs manual pick-time differences; location shifts |
| Magnitude | Station ML vs distance; station residuals (for corrections); magnitude–frequency (Gutenberg–Richter) plot; ML through time |
| hypoDD | Before/after map and sections; dt.cc CC distribution; residuals by iteration; relative location shift histogram |

---

## 6. Workflow steps

### Step 1 — QA/QC and data index (`openmeq qc`)

- **Data index:** scan miniSEED headers only (no waveform loading) and build
  `{seed_id: [(file, start, end), ...]}`, saved to `inventory/index.json`.
  All later steps use this index to find files instead of re-scanning directories.
- **Station metadata:** merge the StationXML file, StationXML folder and CSV into one inventory (`inventory/stations.xml` + `inventory/stations.csv`). Report channels that have data but no metadata, and the reverse.
- **Availability:** percentage coverage per channel per day, gaps and overlaps.
- **Daily noise statistics per channel:**
  - RMS
  - mean absolute amplitude
  - spikiness: count of samples > N × MAD, kurtosis
  - flat-lined and clipped sample counts
- **Monthly cross-station comparison:** each station's statistics against the network median.
- **Output:** `qc/channel_flags.json` (good/bad channel per day), used by detection to exclude bad channels.

### Step 2 — Network detection (`openmeq detect`)

- Read data one day at a time via the index, excluding flagged channels.
- Filter → recursive STA/LTA → ObsPy `coincidence_trigger` across stations.
- **Output:** `detections/detections.csv` (time, stations, coincidence sum).

### Step 3 — Chunk writing (`openmeq chunk`)

- Cut `pre_s` / `post_s` around each detection (default 30 s either side) and write miniSEED to `chunks/<event_id>/<event_id>.mseed`.
- These raw chunks are kept for review, magnitudes and cross-correlation.

### Step 4 — PhaseNet picking (`openmeq pick`)

- **Preparation** (written to `chunks_pn/`):
  1. merge and fill small gaps
  2. demean and detrend
  3. taper
  4. optional high-pass
  5. anti-alias filter and decimate 200 → 100 Hz
  6. map channels to Z/1/2 using `channel_map`; unknown borehole orientations are passed as unoriented horizontals (PhaseNet's E/N inputs). Known orientations in `orientations` are rotated to ZNE.
  7. ensure 3 components per station
- **Run:** PhaseNet `predict.py` on the prepared chunks.
- **Output:** PhaseNet picks converted to QuakeML, one Event per detection with P/S picks, probabilities and pick uncertainties.

### Step 5 — Location and filtering (`openmeq locate`)

Two passes, with separate P and S residual filters:

**Pass A — P only**
1. Run NLLoc with P picks only.
2. Drop P picks with |residual| > `p_only.max_residual_s` or more than `residual_sigma`·σ from the rest.
3. Relocate; repeat until no picks are removed or `max_iter` is reached.
4. Reject events with fewer than `min_p_phases` P picks.

**Pass B — P + S**
1. Screen S picks against the S arrival predicted from the pass-A location; drop those outside `s_predicted_window_s`.
2. Run NLLoc with the remaining P + S picks.
3. Filter P and S residuals separately (`p_and_s.p`, `p_and_s.s`).
4. Relocate; repeat until stable or `max_iter` is reached.

**Event rejection:** RMS, number of phases (total, P, S), azimuthal gap, horizontal/vertical error.

**Output:**
- `locate/catalog_auto.xml` and a CSV summary
- `locate/passA/`, `locate/passB/` with per-iteration NLLoc output and diagnostics
- a log of which picks/events were removed at each pass and why

### Step 6 — Manual review, ObsPyck batch mode (`openmeq review`, optional)

- Launches ObsPyck event by event with the raw chunk and the automatic QuakeML. The user re-picks and relocates with NLLoc inside ObsPyck (no separate NLLoc re-run).
- Each event is saved as `review/events/<event_id>.xml` when picked. Already-reviewed events are skipped, so a session can be stopped and resumed. There is also a skip / mark-as-rejected option.
- The ObsPyck config (station list, NLLoc models folder) is generated from the YAML.
- **Compile** (`openmeq review --compile`): merge reviewed events over the automatic catalogue → `review/catalog_final.xml`.

### Step 7 — Magnitudes (`openmeq magnitude`)

- Runs on the best available catalogue: `review/catalog_final.xml` if review was run, otherwise `locate/catalog_auto.xml`. `--catalog` overrides this, e.g. to re-run on hypoDD output.
- **ML:**
  1. remove instrument response and simulate Wood-Anderson (needs StationXML responses)
  2. measure peak amplitude in a window around the S pick (max of horizontals, Z fallback)
  3. hypocentral distance from the location
  4. station ML from the configurable −log A₀ relation plus optional station corrections
  5. event ML = median of stations
- The −log A₀ term comes from the relation chosen in `magnitude.ml.relation`: Hutton & Boore (1987), or local coefficients or a distance table.
- Written into the QuakeML as `Amplitude`, `StationMagnitude` and `Magnitude` objects, with the relation used recorded in `method_id`, and into the CSV summary.
- **Output:** `magnitude/catalog_mag.xml`, which is the input to hypoDD (so magnitudes appear in `hypoDD.reloc`).

### Step 8 — hypoDD relocation (`openmeq relocate`)

- **Catalogue differential times:** write `phase.dat` (with magnitudes) from the catalogue, then run `ph2dt` → `dt.ct`.
- **Cross-correlation differential times (`dt.cc`):**
  - for linked event pairs sharing a station and phase, cross-correlate raw chunks with `obspy.signal.cross_correlation.xcorr_pick_correction`
  - P on Z; S on horizontals (as available)
  - keep pairs with CC ≥ `min_cc`
- **Velocity model:** 1D layers or 3D SIMUL2000 (`vel3D.dat`), per `hypodd.velocity_model`.
- **Run and output:** run hypoDD, then read `hypoDD.reloc` back into `relocate/catalog_dd.xml` (magnitudes carried over).
- **Template:** built from the local hypoddpy code, used as a template only (see §10).

---

## 7. Utility scripts

- **`scripts/setup_nlloc.py -c config.yaml`**
  - Writes the NLLoc control file and station positions (borehole stations placed by depth).
  - 1D: builds the model grid from the layer file with `Vel2Grid`.
  - 3D: uses the supplied NLLoc model grid (`.hdr`/`.buf`) directly, after checking it covers the location grid.
  - Runs `Grid2Time` for P and S, per station in parallel.
  - Links the model directory where ObsPyck expects it.
- **`scripts/setup_hypodd.py -c config.yaml`**
  - Writes `station.dat`, `ph2dt.inp` and `hypoDD.inp`, including the per-iteration weighting scheme.
  - 1D: writes the layered model into `hypoDD.inp`.
  - 3D: points `hypoDD.inp` at the SIMUL2000 `vel3D.dat` and selects 3D ray tracing.
  - Checks event/station/pair counts and 3D grid dimensions against the compiled limits (`hypoDD.inc`, `vel3d.inc`). If they're exceeded, it rewrites the include files and recompiles into `external/bin`.
  - Optional `--nlloc-to-simul`: converts an NLLoc 3D grid into SIMUL2000 format, so one 3D model can serve both programs.

---

## 8. Repository layout

```
install.sh
pyproject.toml
HYPODD.zip             # hypoDD 2.1 (3D) source, unpacked + compiled by install.sh
config/example.yaml
scripts/setup_nlloc.py
scripts/setup_hypodd.py
openmeq/
  config.py            # YAML loading + validation
  cli.py               # openmeq <step>
  index.py             # miniSEED index
  stations.py          # StationXML file/folder + CSV → merged inventory
  qc/
  detect/
  chunk.py
  pick/
  locate/
  review/
  magnitude/
  relocate/
  plotting/            # shared map/section/histogram functions for diagnostics
  parallel.py          # process-pool helper: worker init, thread limits, progress, resume
tests/
external/              # git-ignored: PhaseNet clone, HYPODD build, compiled binaries (bin/)
docs/
```

---

## 9. Build order (each milestone becomes a beads epic)

| # | Milestone | Depends on |
|---|---|---|
| M0 | Project setup: `install.sh` (incl. hypoDD from zip), `pyproject.toml`, YAML config (loading, validation, commented example), CLI, logging, shared plotting module, parallel helper, small test dataset, benchmark script | — |
| M1 | Data index, station metadata merge, QA/QC, with diagnostics | M0 |
| M2 | Network detection | M1 |
| M3 | Chunk writing | M2 |
| M4 | PhaseNet prep, running PhaseNet, converting picks to QuakeML | M3 |
| M5 | `setup_nlloc.py` (1D, then 3D grid input) | M0 |
| M6 | Two-pass location and filtering (P-only → P+S), per-pass diagnostics | M4, M5 |
| M7 | ObsPyck batch review and compile step | M6 |
| M8 | Magnitudes (ML) | M6 (uses M7 output if review is run) |
| M9 | `setup_hypodd.py` (1D + 3D SIMUL2000, limit checks, NLLoc→SIMUL conversion) | M0 |
| M10 | Review of the hypoddpy template (§10), then parallel cross-correlation with caching, `dt.ct`/`dt.cc`, running hypoDD, reading results back | M8, M9 |
| M11 | `openmeq run` for the whole chain, end-to-end test, performance profiling on a realistic data volume, documentation | all |

M5 and M9 can be built at any time after M0. M7 and M8 can be built in parallel.

---

## 10. Reuse of hypoddpy (template only)

- **Source:** `~/seismic/hypoDDpy-master/hypoddpy/hypodd_relocator_parallel.py`, which already uses joblib and `xcorr_pick_correction`. Also `~/seismic/hypoddpy/hypodd_relocator.py`, which is the same as `~/PycharmProjects/hypoddpy_original`. The `*_bup.py` and `*_broken.py` variants are ignored.
- **Parts to use as templates:**
  - station, phase and event file writers
  - `ph2dt.inp` and `hypoDD.inp` writers
  - event-pair selection from `dt.ct`
  - the cross-correlation loop
  - reading `hypoDD.reloc` back
- **Rules:**
  - Nothing is copied unchanged. Each ported function is rewritten to OpenMEQ's config and data structures.
  - Each is checked against the hypoDD 2.1 user guide (`HYPODD/doc/hypoDD2_UserGuide.pdf`) for file formats and parameter meaning.
  - Each gets a unit test. The written `phase.dat`, `station.dat`, `ph2dt.inp` and `hypoDD.inp` are checked against the example-1 files in the zip, and the full pipeline must reproduce the example-1 `hypoDD.reloc`.
  - Checks for the cross-correlation code:
    - sign convention of the time lag
    - how the correlation coefficient and lag correction are turned into a `dt.cc` differential time
    - window alignment on the pick
    - handling of sampling rate
    - pairs with a missing phase
  - Any defects found in the template are recorded in the beads issue for M10.

---

## 11. Open questions

None blocking. Items still to settle during the build:
- Local ML calibration: the file format of the local −log A₀ table can be decided when M8 starts.
- Whether hypoDD should be split into clusters by default (§4), or run on the whole catalogue in one go.

---

## Resolved

- **HYPODD.zip:** slimmed zip approved and created.
- **Magnitude:** ML, with Hutton & Boore or local calibration selectable.
- **hypoddpy:** can be reused as a template only, with every part checked (§10).
- **Performance:** everything that can run in parallel does (§4).

### Resolved in v2

- **Station metadata:** StationXML file, folder of StationXMLs, and CSV are all supported (§3 `stations`).
- **Borehole orientations:** mostly unknown. Treated as unoriented horizontals, with known azimuths optional.
- **3D models:** NLLoc grid for NLLoc; SIMUL2000 for hypoDD.
- **hypoDD licence:** confirmed OK.
