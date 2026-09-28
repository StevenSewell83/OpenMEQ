# OpenMEQ — Architecture and Data Contracts

This document is the **contract between modules**. Every task in the build (see `docs/PLAN.md` and the beads issues) must follow it, so that pieces built independently fit together. If a task needs to change a contract, it updates this file in the same commit and says so in its beads issue notes.

## 1. Platform and conventions

- **Python 3.10.** This is fixed by PhaseNet's pin of `tensorflow==2.11.0` and `numpy==1.26.4`.
- **Layout:** flat package `openmeq/` at the repo root, installed with `pip install -e .`, and `pyproject.toml` (setuptools). CLI entry point: `openmeq = openmeq.cli:main`.
- **Libraries:** `click` for the CLI; `pydantic` v2 for config validation; `PyYAML`; `obspy`; `numpy`; `scipy`; `pandas`; `matplotlib` (Agg backend in batch code); `tqdm`.
- **Code style:** `ruff` (lint + format); type hints on public functions; NumPy-style docstrings on public functions.
- **Tests:** `pytest` under `tests/`, mirroring the package (`tests/test_index.py` etc.).
- **Markers** for tests needing external pieces, skipped automatically if the piece is absent:
  - `requires_nlloc`
  - `requires_hypodd`
  - `requires_phasenet`
  - `requires_gui`
  - `slow`
- **Times:** all times are UTC `obspy.UTCDateTime` in code, and ISO-8601 strings with a `Z` suffix in files.
- **Units:** distances in km in the location and hypoDD steps; metres for station elevation and depth in metadata.

## 2. Shared modules (built in M0; later tasks use them, not their own copies)

| Module | API |
|---|---|
| `openmeq.config` | `load_config(path) -> Config` (pydantic). One attribute per YAML section: `cfg.project`, `cfg.parallel`, `cfg.data`, `cfg.stations`, `cfg.qc`, `cfg.detection`, `cfg.chunks`, `cfg.phasenet`, `cfg.nlloc`, `cfg.review`, `cfg.magnitude`, `cfg.hypodd`. Relative paths are resolved against the config file's directory. The schema follows `config/example.yaml`, which is the documented reference. |
| `openmeq.paths` | `RunPaths(cfg)`: properties for every path in §3 (e.g. `p.index_json`, `p.chunk_file(event_id)`); `ensure()` creates the folders. |
| `openmeq.log` | `configure_logging(paths: RunPaths, step: str, level="INFO")` is called once by the CLI before a step runs, and sets up the console handler plus `<output_dir>/logs/<step>.log`. `get_logger(name: str) -> logging.Logger` returns a child of the `openmeq` logger; until `configure_logging` is called it logs to the console only (as in library use and tests). |
| `openmeq.parallel` | `pmap(func, items, pcfg, desc="", chunksize=1, ordered=True) -> list`. It uses a `ProcessPoolExecutor` with `n_workers` (`auto` = physical cores), and a worker initialiser sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/TF threads to `threads_per_worker`. Because the thread environment variables must be set **before** NumPy/BLAS is imported in a worker, the pool uses the `spawn` start method, and the parent also exports the variables before creating the pool. Shows a tqdm progress bar. `func` must be a top-level, picklable function, and items should be paths or parameters, not waveform objects. `n_workers=1` runs serially in-process (for debugging and tests). `pmap_incremental(func, items, pcfg, out_path, key=...)` appends results to a JSON-lines file as they finish and skips keys already present, so runs can resume. |
| `openmeq.cli` | Each step is registered by defining `run(cfg, paths, **opts)` in that step's module (`openmeq.qc`, `openmeq.detect`, `openmeq.chunk`, `openmeq.pick`, `openmeq.locate`, `openmeq.review`, `openmeq.magnitude`, `openmeq.relocate`). The CLI imports the module lazily and calls it; `cli.py` needs no edit. Extra options: `review` gets `compile: bool`, and `magnitude` gets `catalog: Path or None`. `RunPaths.diagnostics_dir(step)` and `manifest_json(step)` take the step's output-folder name (e.g. `detections`, `chunks_pn`), not the CLI command name. |
| `openmeq.manifest` | `StepManifest(paths, step)`: `.is_done(key, fingerprint)` and `.mark_done(key, fingerprint)`. `fingerprint(*objs)` is a stable hash of a config section plus input file mtimes/sizes. Used for "skip unchanged". |
| `openmeq.external` | The external folder is resolved in this order: `$OPENMEQ_EXTERNAL_DIR`, then `<repo root>/external`, where the repo root is `Path(openmeq.__file__).parents[1]` (valid for the editable install). `bin_path(name) -> Path`, looking in `external/bin` first, then `PATH`, and raising `ExternalToolMissing` if absent. `phasenet_dir() -> Path` (`external/PhaseNet`). `phasenet_model_dir() -> Path`. `run(cmd, cwd, log, timeout)` runs a subprocess, logging its output. |
| `openmeq.catalog` | QuakeML helpers from §5: `event_id_from_time(t: UTCDateTime) -> str` (the only implementation of the §4.4 format), `event_id(event)`, `make_event(event_id)`, `accepted(catalog)`, `reject_pick(pick, arrival, reason, step)`, `reject_event(event, reason, step)`, `set_extra(obj, key, value)`, `get_extra(obj, key)`, `read_catalog(path)`, `write_catalog(cat, path)`. |
| `openmeq.index` | `DataIndex.load(path)`; `.files(seed_id, t1, t2) -> list[Path]`; `.seed_ids(pattern="*")`; `.read(seed_ids, t1, t2, pad=0) -> Stream`, which reads only the needed files with `starttime`/`endtime` slicing and merges them. |
| `openmeq.stations` | `load_inventory(cfg.stations) -> obspy.Inventory` (StationXML file/folder + CSV merge); `station_table(cfg.stations) -> pandas.DataFrame` with columns `network,station,location,latitude,longitude,elevation_m,depth_m`; `apply_channel_map(stream, cfg.stations) -> obspy.Stream`; `write_merged_inventory(inv, paths, cfg.stations=None)` writes `inventory/stations.xml`/`.csv`; `check_coverage(seed_ids: Iterable[str], inv) -> dict` (takes a plain iterable of seed ids, e.g. `DataIndex.seed_ids()` — does not import `openmeq.index`) plus `write_metadata_report(report, paths)` for `inventory/metadata_report.json` (M1.2). |
| `openmeq.plotting` | `setup_style()`, `save_fig(fig, path)`, `plot_map_sections(df, stations, path, ...)`, `plot_hist(...)`, `plot_heatmap(...)`, `plot_timeseries(...)`, `plot_boxplot(...)`. |
| `openmeq.review.obspyck_config` (M7.2) | ObsPyck ini + pluginpath generator, backing M7.3's batch loop (docs/obspyck_notes.md). `generate_obspyck_config(cfg, paths, force=False) -> Path` writes `cfg.review.obspyck_config` (`[base]`/`[seismic_phases]`/`[seed_id_lookup]`/`[nonlinloc]` from `cfg`, `[keys]`/`[misc]`/`[gui_defaults]`/`[matplotlibrc]`/`[ar_picker]` spliced byte-identical from a packaged copy of ObsPyck's own `example.cfg`; no `server_*`/`[station_combinations]`), calls `openmeq.nlloc_setup.link_obspyck_model` (reused, not duplicated) and adds the empty `hyp_2000/`/`focmec/` dirs `ObsPyck.__init__` requires alongside `nlloc/`, gated by `openmeq.manifest.StepManifest` on `cfg.review`+`cfg.nlloc`+the station table. `openmeq.review._obspyck_compat.apply_obspyck_compat_patches(cfg)` is the one entry point for the two ObsPyck 0.8.2 monkeypatches (`gk2lonlat` re-derived from `cfg.nlloc.grid.origin`'s `TRANS SIMPLE`, and stripping NLLoc 7.1.05's stray `>` `PHASE`-line token) -- M7.3's launcher must call it once after import and again immediately after every `ObsPyck(...)` construction. |
| `openmeq.nlloc_setup` (M5.2, 3D grids M5.3) | 1D/3D NLLoc setup, backing `scripts/setup_nlloc.py` (PLAN.md §7). `write_control_common(cfg, paths) -> str` (shared `CONTROL`/`TRANS` boilerplate every Vel2Grid/Grid2Time/NLLoc control file in the project uses, including M6.1's). `write_station_list(station_table) -> list[str]` (one `GTSRCE` line per unique `NET.STA`, raises `ConfigError` on a bare-station-code clash). `location_grid_bounds(cfg, station_table) -> LocationGridBounds` (the convention for the eventual `LOCGRID` bounds -- §6d). `build_1d_grid(cfg, paths, station_table)` (1D: runs `Vel2Grid`). `read_grid3d_hdr(hdr_path) -> Grid3DHeader` (parses an NLLoc 3D grid `.hdr`). `check_3d_grid_coverage(hdr_path, cfg, bounds)` (raises `ConfigError` unless the grid strictly covers `bounds` and its own `TRANSFORM` matches `cfg.nlloc.grid.origin` -- §6d). `link_3d_grid(cfg, paths, station_table)` (3D: validates then copies/symlinks the user-supplied grid in, same `<root>.{P,S}.mod.{hdr,buf}` convention as `build_1d_grid`). `run_grid2time(cfg, paths, station_table)` (runs `Grid2Time` per (station, phase) via `openmeq.parallel.pmap`; auto-detects `GTMODE GRID2D`/`GRID3D` per phase from its own model grid's `.hdr`, so it's agnostic to which of the two setup paths produced its input). `link_obspyck_model(cfg, paths, station_table)` (refreshes ObsPyck's model link, docs/obspyck_notes.md). `setup_nlloc(cfg, paths, force=False) -> bool` (top-level orchestration, dispatches on `cfg.nlloc.velocity_model.type`, with `openmeq.manifest.StepManifest` skip-unchanged). |

## 3. Run directory layout (`cfg.project.output_dir`)

```
inventory/index.json            # miniSEED index (§4.1)
inventory/stations.xml          # merged StationXML
inventory/stations.csv          # station table (§2 station_table columns)
inventory/metadata_report.json  # {"data_without_metadata": [seed_id], "metadata_without_data": [seed_id]}
qc/availability.csv             # seed_id,date,coverage_pct,n_gaps,n_overlaps,gap_s
qc/daily_stats.csv              # seed_id,date,rms,mean_abs,spike_count,kurtosis,flat_count,clip_count,npts
qc/monthly_stats.csv            # seed_id,month,stat,value,network_median,ratio,rank
qc/channel_flags.json           # (§4.2)
qc/diagnostics/
detections/detections.csv       # (§4.3)
detections/diagnostics/
chunks/<event_id>/<event_id>.mseed      # raw chunk, all channels, original sampling
chunks/completeness.csv         # event_id,seed_id,fraction_present
chunks_pn/<event_id>/<event_id>.mseed   # PhaseNet-ready: 100 Hz, 3-comp per station
chunks_pn/data_list.csv         # PhaseNet input list
chunks_pn/prep_report.csv       # event_id,seed_id_base,n_components_found,zero_filled_components (M4.2)
pick/phasenet_raw/              # PhaseNet native output
pick/catalog_picks.xml          # events with automatic picks, no origins
nlloc/VELOCITY_MODELS/layer_time/layer.{P,S}.mod.{hdr,buf}        # Vel2Grid 1D velocity grid (M5.2)
nlloc/VELOCITY_MODELS/layer_time/layer.{P,S}.<STA>.time.{hdr,buf} # Grid2Time per-station travel-time grids (M5.2)
nlloc/manifest.json             # setup_nlloc.py skip-unchanged (M5.2)
locate/passA/<event_id>/        # NLLoc work per iteration
locate/passB/<event_id>/
locate/catalog_auto.xml
locate/summary.csv              # event_id,time,lat,lon,depth_km,rms,gap,n_p,n_s,err_h_km,err_z_km,status,reason
locate/removal_log.csv          # event_id,pass,iteration,seed_id,phase,residual_s,reason
review/state.json               # {event_id: "done"|"skipped"|"rejected"}
review/events/<event_id>.xml
review/catalog_final.xml
magnitude/catalog_mag.xml
magnitude/station_magnitudes.csv
relocate/work/                  # hypoDD run folder(s): station.dat, phase.dat, event.dat,
                                 # ph2dt.inp, hypoDD.inp, event.sel/station.sel (ph2dt output),
                                 # dt.ct (ph2dt output, M10.2), dt.cc (M10.3), ...
relocate/work/event_id_map.csv         # event_id,cuspid -- persisted, append-only (§6c)
relocate/work/station_label_map.csv    # net_sta,label -- persisted, append-only (§6c)
relocate/cache/<event_id>.npz   # pre-filtered waveforms for xcorr
relocate/catalog_dd.xml
<step>/diagnostics/             # every step
<step>/manifest.json            # every step
logs/<step>.log
```

**`relocate/work/` layout, resolved (M10.2 / OpenMEQ-16d.2).** An earlier
sketch of this layout showed `dt.ct`/`dt.cc` as top-level `relocate/dt.ct`,
`relocate/dt.cc`. That was ambiguous: `hypoDD.inp`'s `dt.cc`/`dt.ct`
input-file-selection lines (§6c) are bare filenames, read relative to
hypoDD's own `cwd`, and hypoDD is always run with `relocate/work/` as that
`cwd` (alongside `hypoDD.inp`, `event.sel`, `station.sel`). So `dt.ct`
(written by `ph2dt`, M10.2) and `dt.cc` (written by cross-correlation,
M10.3) are colocated in `relocate/work/` -- there is no separate top-level
copy. `openmeq.paths.RunPaths.dt_ct`/`.dt_cc` resolve to
`relocate/work/dt.ct`/`relocate/work/dt.cc` accordingly.

**The `dt.ct` file and the event-pair list (M10.2).**
`openmeq.relocate.ph2dt_step.run_ph2dt(cfg, paths)` runs the real `ph2dt`
binary in `relocate/work/` (inputs: `station.dat`, `phase.dat`, `ph2dt.inp`,
written by `openmeq.relocate.setup.setup_hypodd`/`openmeq.relocate.hypodd_files`
from `magnitude/catalog_mag.xml`'s accepted events/non-rejected picks only)
and parses its `relocate/work/dt.ct` output into the in-memory **event-pair
list** M10.3's cross-correlation consumes. `dt.ct` itself (`ph2dt.f`'s own
format, confirmed against `examples/example1/dt.ct`) is:

```
# CUSPID1 CUSPID2
STA  TT1  TT2  WGHT  PHA
STA  TT1  TT2  WGHT  PHA
...
# CUSPID1 CUSPID2
...
```

- A `#`-prefixed header line starts each linked event pair: two whitespace-
  separated integer fields, `CUSPID1 CUSPID2` (the two events' hypoDD
  `cuspid`s, `build_event_id_map`'s persisted `event_id_map.csv` mapping,
  §6c) -- no `OTC` third field (that's `dt.cc`'s header, `# ID1 ID2 OTC`,
  `docs/hypoddpy_audit.md` §2.1; `dt.ct`'s header is ground-truth-confirmed
  as exactly 2 fields, `getdata.f:485`).
- Each following non-`#` line, until the next `#` line or end of file, is
  one linked observation: station label (`STA`, `station_label_map.csv`'s
  mapped `character*7` label), `TT1`/`TT2` (each event's travel time to
  `STA`, seconds), `WGHT` (ph2dt's own averaged pick-quality weight, 0-1),
  `PHA` (`P` or `S`).
- `run_ph2dt`'s return value, `list[tuple[int, int]]`, is the ordered list
  of `(cuspid1, cuspid2)` pairs from every `#` header line, in the order
  `ph2dt` wrote them (ph2dt has already applied `ph2dt.inp`'s
  `MAXSEP`/`MAXNGH`/`MINLNK`/etc. filtering by the time `dt.ct` exists, so
  this is the exact, already-filtered set of pairs to cross-correlate --
  not the full O(n_events²) combinatorial space, `docs/hypoddpy_audit.md`
  §3/§6 defect #6's "port-rewritten" guidance). M10.3 resolves a `cuspid`
  back to an `event_id` via `openmeq.relocate.hypodd_files.invert_event_id_map`
  over the same persisted `event_id_map.csv`, and re-reads `dt.ct` itself
  for each pair's per-station/phase data -- the returned list is only the
  pair identities, not a duplicate copy of `dt.ct`'s per-line content.

## 4. File formats

### 4.1 `inventory/index.json`

```json
{"version": 1, "created": "2026-09-28T00:00:00Z", "roots": ["/data/mseed"],
 "channels": {"NET.STA.LOC.CHA": [["/abs/path/file.mseed", "2025-01-01T00:00:00.000000Z", "2025-01-01T23:59:59.995000Z", 200.0], ...]}}
```

Segments are sorted by start time. A file holding several channels appears under each of them. Paths are absolute.

### 4.2 `qc/channel_flags.json`

```json
{"version": 1, "flags": {"NET.STA.LOC.CHA": {"2025-01-01": {"ok": true, "reasons": []},
                                             "2025-01-02": {"ok": false, "reasons": ["availability<80", "spikes"]}}}}
```

A channel/day with no entry counts as `ok: false` (no data).

### 4.3 `detections/detections.csv`

Columns:
- `event_id`
- `time` (trigger on-time, ISO)
- `duration_s`
- `coincidence_sum`
- `n_stations`
- `stations` (`;`-separated `NET.STA`)
- `trace_ids` (`;`-separated)

### 4.4 Event IDs

The ID is `ev` followed by the detection trigger time as `YYYYMMDDTHHMMSS_mmm` (UTC, milliseconds), e.g. `ev20250101T123456_789`. It is created in detection and never changes afterwards. The QuakeML `Event.resource_id` is `smi:openmeq/event/<event_id>`.

## 5. QuakeML conventions

The chain passes ObsPy `Catalog` objects saved as QuakeML. Every step reads the previous step's catalogue and writes a new file; nothing is edited in place.

- **Picks:**
  - `Pick.waveform_id` is the full SEED id **after** channel mapping back to the original channel code (picks always refer to real channels).
  - `phase_hint` is `"P"` or `"S"`.
  - `method_id` is `smi:openmeq/phasenet`, `smi:openmeq/manual/obspyck`, etc.
  - `evaluation_mode` is `automatic` or `manual`.
  - `time_errors.uncertainty` is in seconds.
  - PhaseNet probability goes in `extra` (below), key `probability`.
  - **Channel for PhaseNet picks:** PhaseNet picks per station (instrument), not per channel. P picks go on the station's Z channel; S picks go on the first horizontal present, in the order N, E, 1, 2. The original channel code is used.
- **Origins:**
  - `method_id` is `smi:openmeq/nlloc/passA` or `.../passB`, or `smi:openmeq/hypodd`.
  - `Origin.arrivals` carry `time_residual`, `distance` (degrees), `azimuth` and `time_weight`.
  - `event.preferred_origin_id` points to the latest origin.
- **Rejection (never delete, always flag):**
  - Picks: `reject_pick` sets `pick.evaluation_status="rejected"`, sets `arrival.time_weight=0` on the current origin, and adds a `Comment` of the form `"rejected:<step>:<reason>"`.
  - Events: `reject_event` sets the preferred origin's `evaluation_status="rejected"` and adds the same kind of comment. `accepted(catalog)` returns only events that are not rejected.
- **Magnitudes:** `Amplitude` (type `"AML"`, generic amplitude in metres, Wood-Anderson), `StationMagnitude` (`"ML"`), and `Magnitude` (`"ML"`, `method_id` `smi:openmeq/ml/hutton_boore` or `smi:openmeq/ml/local`).
- **`extra` namespace:** `http://openmeq.org/xmlns/1.0` with prefix `openmeq`. Always use `openmeq.catalog.set_extra` / `get_extra` so values survive a QuakeML round-trip.

## 6. Channels and stations

- **Component mapping:** after `apply_channel_map`, the last character of the channel code is `Z`, `N`/`E` (oriented) or `1`/`2` (unoriented). The original code is kept in `trace.stats.openmeq_original_channel` so picks can be mapped back.
- **Station depth:** station elevation for location is `elevation_m - depth_m` (borehole sensors have `depth_m > 0`). NLLoc and hypoDD receive this effective elevation (negative depth below the datum).
- **Station keys:** `NET.STA` everywhere; location codes are ignored when two share a site.
- **Station codes must be unique across networks.** NLLoc obs files and ObsPy's NLLoc reader and writer match on the bare station code, and hypoDD allows at most 7 characters. `openmeq.stations` checks this at load time and fails with a clear error.

## 6a. Velocity model files

**1D layer file** (`*.txt`), used by NLLoc setup (`LAYER`), hypoDD 1D and the synthetic generator:

```
# depth_top_km  vp_km_s  vs_km_s  [rho_g_cc]
0.0   3.50  2.00  2.3
1.0   4.50  2.60  2.5
3.0   5.80  3.35  2.7
```

- Whitespace-separated; `#` starts a comment.
- Depth is to the top of each layer, below sea level (negative = above sea level), and increases down the file.
- The first layer applies from the top of the model to the next depth.
- The layers are constant velocity; NLLoc receives zero gradients.
- `rho` is optional (default 2.7).
- Parser: `openmeq.velocity.read_layers(path) -> pandas.DataFrame[depth_km, vp, vs, rho]`, built in M0.8.

**3D models:** an NLLoc grid (`<root>.P.mod.hdr/.buf`, `<root>.S.mod...`) for NLLoc, and SIMUL2000 `vel3D.dat` for hypoDD (format as in `HYPODD/examples/example1/vel3D.dat`).

## 6b. Local ML calibration file format

Resolves PLAN.md §11's open question, per M8.2 / OpenMEQ-625.2.

When `magnitude.ml.relation: local` and `magnitude.ml.local.coefficients` is
not set (all three of `a`/`b`/`c` null), `magnitude.ml.local.file` points to a
CSV distance/`-logA0` calibration table instead of a 3-coefficient formula:

```csv
distance_km,neg_log_a0
0,1.4
10,1.6
50,2.2
100,3.0
200,3.4
```

- **Header** is exactly `distance_km,neg_log_a0` (both columns required, no
  extra columns).
- **`distance_km`** is hypocentral distance in km, **strictly increasing**
  down the file (no duplicate or repeated distances), at least 2 rows.
- **No missing/NaN values** in either column.
- **`ML = log10(A_mm) + neg_log_a0(R)`**, where `neg_log_a0(R)` is
  **linearly interpolated** between the two bracketing rows for a hypocentral
  distance `R` that falls inside the table's range.
- **Outside the table's range** (`R` below the first row's `distance_km` or
  above the last row's), `neg_log_a0(R)` is **clamped** to the nearest end
  row's value (i.e. held constant, not linearly extrapolated). This is
  `numpy.interp`'s default out-of-range behaviour, relied on rather than
  reimplemented, and is intentional here even though it looks like a bug at
  a glance: without it, calibration tables that only cover a network's
  useful aperture would extrapolate wildly for occasional far/near stations.
  Every clamped lookup increments a counter so the magnitude step's
  diagnostics (M8.3) can report how many stations were extrapolated.
- Loading is `openmeq.magnitude.relations.load_neg_log_a0_table(path)`;
  interpolation is `openmeq.magnitude.relations.interp_neg_log_a0(distance_km,
  table)`. A malformed table raises `ConfigError` naming the file path and
  the offending row number.
- The table (and, separately, a CSV-backed `magnitude.ml.station_corrections`)
  is loaded once per process and cached at module level, keyed by resolved
  file path — each `pmap` worker process gets its own cache (no cross-process
  sharing), matching M8.1's response-cache pattern.

## 6c. hypoDD/ph2dt file writers (1D)

Built in M9.2 (`openmeq/relocate/hypodd_files.py`), validated against
`external/HYPODD/examples/example1`. Every field/unit/quirk below was
confirmed directly against `external/HYPODD` source (not generic hypoDD
docs) — see `docs/hypoddpy_audit.md` §6 for the full defect list this
module fixes and `bd show OpenMEQ-6ld` (the M9 epic's design note) for
file:line citations.

- **ID mappings are persisted, append-only CSVs**, not regenerated
  per-run (hypoddpy's template does the latter, an audited resume-safety
  defect): `relocate/work/event_id_map.csv` (`event_id,cuspid`, cuspids
  assigned 1..N by sorting *newly seen* accepted events by origin time —
  existing rows are never renumbered) and
  `relocate/work/station_label_map.csv` (`net_sta,label`, a `NET.STA` ->
  `character*7` hypoDD station label, bare `STA` code where unique across
  networks, else a truncated/digit-suffixed fallback; a genuine collision
  within 7 characters raises `ConfigError`).
- **Units:** hypoDD/ph2dt files use **km** for depth/distance and **metres**
  for station elevation (hypoDD itself divides by 1000 internally,
  `getdata.f:274`); QuakeML depth is metres, so every writer converts via
  `openmeq.relocate.hypodd_files.depth_m_to_km`, the one shared conversion
  helper (also used by M10.4's `hypoDD.reloc` reader).
- **Station elevation quirk:** hypoDD silently zeroes any negative station
  elevation unless `IMOD=5` (`getdata.f:275-278`). Station elevation here is
  the §6 effective elevation (`elevation_m - depth_m`), so boreholes
  (`depth_m > 0`) are commonly negative. M9.2 targets `IMOD=1` and does
  **not** auto-switch to `IMOD=5`: it writes the true (negative) value and
  logs one `WARNING` per run naming every affected station, rather than
  silently replicating hypoDD's own silent clamp.
- **Rejected picks are excluded from `phase.dat` entirely** (checked via
  `pick.evaluation_status != "rejected"`), not merely weighted to zero —
  writing `time_weight=0` (§5's rejection convention) as a `phase.dat`
  weight would look like a deliberately-low-quality *present* pick to
  ph2dt's `MINWGHT` filter (`ph2dt.f:485`, which only skips weights that are
  `>= 0` and `< minwght`), i.e. a rejected pick would still pass the default
  filter. A pick with no `Arrival.time_weight` at all is written as `-1.0`,
  hypoDD's own "no info" sentinel (always bypasses `MINWGHT`), not `0.0`.
- **`hypodd.ph2dt`** (extends the block above): `minwght` (0), `maxdist_km`
  (200), `minobs` (8), `maxobs` (20) — ph2dt.inp's own bracketed-comment
  defaults.
- **`hypodd.weighting_rows`**: a list of `{niter, wtccp, wtccs, wrcc, wdcc,
  wtctp, wtcts, wrct, wdct, damp}` rows for hypoDD.inp's iterative
  data-weighting table; defaults to `examples/example1/hypoDD.inp`'s real
  5-row scheme (not hypoddpy's minimal 2-row default).
- **`hypodd.istart`/`isolv`/`iaq`**: solution-control defaults `2`
  (network sources), `2` (LSQR), `1` (remove airquakes).
- **`hypodd.idat`/`ipha`/`max_station_dist_km`**: hypoDD.inp's `IDAT IPHA
  DIST` line (data type, phase selection, max cluster-centroid-to-station
  distance); defaults `3`/`3`/`200`.
- **`hypodd.clustering`** (`obscc`, `obsct`, `mindist_km`, `maxdist_km`,
  `maxgap_deg`): hypoDD.inp's event-clustering line. hypoddpy's template
  hardcodes `OBSCC=OBSCT=0`, disabling clustering entirely; defaults here
  are `0`/`8`/`-999`/`-999`/`-999`, matching example1's own working config.
- `write_hypodd_inp_1d` always targets `IMOD=1` (1D, per-layer variable
  vp/vs ratio — the modern, backward-compatible choice per HYPODD's
  "Main changes in 2.1b"), references `event.sel`/`station.sel` (ph2dt's
  filtered output, not the raw `event.dat`/`station.dat`), and always
  writes `CID=0` (relocate all) with a blank `ID` list; per-cluster
  relocation is M10.4's `parallel_clusters` option.

**3D (SIMUL2000, `IMOD=9`)**, built in M9.3 (`write_hypodd_inp_3d`, same
module, sharing every section but the model block with `write_hypodd_inp_1d`
via `_write_hypodd_inp_common`/`_write_hypodd_inp_footer`):

- Requires `cfg.hypodd.velocity_model.origin_3d`, `.rotation_3d_deg` and
  `.simul2000` all set (raises `ConfigError` otherwise).
- The `IMOD=9` block's three data lines and field order — `vel3D.dat`
  filename (basename only, matching `write_ph2dt_inp`'s bare-filename
  convention since hypoDD always runs with the run folder as `cwd`);
  `LAT_3D LON_3D ROT_3D`; `IPHA NDIP ISKIP SCALE1 SCALE2 XFAC TLIM
  NITPB` — are confirmed directly against `get_vel3d.f:6-9`'s subroutine
  signature, `hypoDD.f:449-452`'s call site, and `getinp2.f:231-237`'s
  `hypoDD.inp` read (all three agree; see `write_hypodd_inp_3d`'s
  docstring for the full citation trail). This resolves the `TODO` M0.3
  left on `openmeq.config.Simul2000RaytraceConfig`'s field order — it was
  already correct; no config schema change was needed.
- `openmeq.relocate.hypodd_files.Vel3DGrid` (dataclass: `bld`, `nx`, `ny`,
  `nz`, `xn`, `yn`, `zn`) and `read_vel3d_header(path)` parse just a
  SIMUL2000 `vel3D.dat`'s grid header (not the velocity values), mirroring
  `get_vel3d.f:68-82`'s four list-directed Fortran reads exactly (including
  tokens left over on the header line, e.g. `examples/example1/vel3D.dat`'s
  own `1.0 13 19 11 2  computed`, being discarded rather than misread).
  Used by this module's own limit-check (below) and, in M9.4, to validate
  an NLLoc-grid-to-SIMUL2000 conversion.

**Compiled-array-limit checking and recompilation**, built in M9.3
(`openmeq/relocate/hypodd_limits.py`, a separate module from the file
writers above — a distinct concern, not a competing writer):

- `check_compiled_limits(cfg, catalog, station_table, vel3d_grid=None) ->
  LimitCheckResult` parses the *currently compiled* values out of
  `include/hypoDD.inc`/`ph2dt.inc`/`vel3d.inc` at check time (never
  hardcodes the shipped defaults — a previous recompile may have already
  changed them) and checks: event count vs. `min(MAXEVE, MEV)`; station
  count vs. `min(MAXSTA, MSTA)`; total observation count (catalog phases,
  rejected picks excluded, plus an optional `xcorr_obs_count` once M10.1's
  `dt.cc` exists) vs. `MAXDATA`; largest per-event phase count vs. `MOBS`
  (ph2dt only); and, if a `Vel3DGrid` is given, each grid axis's node count
  vs. `vel3d.inc`'s `maxnx`/`maxny`/`maxnz`, **and** (M9.5 / OpenMEQ-6ld.5)
  each axis's physical map span, in `bld` units, vs. `vel3d.inc`'s
  `ixkms`/`iykms`/`izkms` — a distinct compiled limit from `maxnx`/`maxny`/
  `maxnz` (node count vs. physical span of the `bldmap`-built node-lookup
  arrays, `get_vel3d.f`'s `bldmap` subroutine, which stops fatally if
  exceeded), checked instead of relying on
  `openmeq.relocate.nlloc_to_simul`'s own hardcoded stand-in value for this
  same limit. `LimitCheckResult` holds a list of `LimitCheck(name, current,
  limit, exceeded)`, with names `event_count`/`station_count`/
  `observation_count`/`phases_per_event`/`grid_nx`/`grid_ny`/`grid_nz`/
  `grid_span_x`/`grid_span_y`/`grid_span_z`.
- `compiled_vel3d_node_limits(include_dir=None) -> (maxnx, maxny, maxnz)`
  (M9.5 / OpenMEQ-6ld.5): the currently compiled node-count limits alone,
  for a caller about to build a `vel3D.dat`
  (`nlloc_to_simul.nlloc_grid_to_vel3d`) that wants the real current values
  rather than that function's own shipped-default fallbacks.
- `recompile_hypodd(new_limits, external_hypodd_dir, external_bin_dir,
  log)` rewrites just the changed `.inc` files' numeric values via a
  regex substitution that preserves Fortran continuation lines and
  comments (including `hypoDD.inc`'s own commented-out "small size
  problems" alternate block, which every parse/rewrite here explicitly
  skips), then runs `make clean && make` in `<external_hypodd_dir>/src`
  (`openmeq.external.run`) and installs the rebuilt `hypoDD`/`ph2dt` into
  `external_bin_dir`. `ensure_compiled_limits(...)` composes the two:
  checks, and recompiles only if something is exceeded (idempotent — a
  dataset that already fits never triggers a rebuild), raising each
  exceeded parameter to `ceil(current * 1.5)` (`DEFAULT_HEADROOM_FACTOR`)
  headroom so small further growth doesn't immediately force another
  recompile.
- **Never run any of these three against the main checkout's
  `external/HYPODD`/`external/bin` from a test** — copy the source tree to
  a scratch directory first (`recompile_hypodd` itself is directory-
  agnostic; this is a caller discipline, enforced by every test in
  `tests/test_relocate_limits.py`). A full `make clean && make` of the
  whole tree takes well under 2 seconds, so tests recompile for real
  rather than mocking the compiler.
- `scripts/setup_hypodd.py`'s CLI wiring around `ensure_compiled_limits`
  (reading the real catalogue/station table, deciding when to call it in
  the pipeline) is `openmeq.relocate.setup.setup_hypodd`, M9.5 — see below.

**3D (IMOD=9) model coverage/registration check**, built in M9.6 /
OpenMEQ-6ld.6 (`openmeq.relocate.hypodd_limits.check_vel3d_coverage`, same
module as the compiled-limit checks above — a distinct concern from the
file writers, per this section's own convention). Root-caused a real
segfault (confirmed with a `-fcheck=all`/`-fbacktrace` debug build, never
against the main checkout's `external/HYPODD`): hypoDD's `IMOD=9` ray
tracer (`ray_3d.f`'s `intmap` subroutine, called from `vel3`) computes a
node-grid lookup index directly from a query point's raw Cartesian
coordinate with **no bounds check** before indexing the compiled
`ixloc`/`iyloc`/`izloc` lookup arrays (`vel3d.inc`) — so any event or
station that, once projected under hypoDD.inp's `LAT_3D LON_3D ROT_3D`
(hypoDD's own runtime projection, `setorg.f`/`dist.f` — **not**
`setorg_3d.f`/`disto.f`, which is unused dead code), falls outside the
supplied `vel3D.dat`'s node domain corrupts memory instead of failing
cleanly. This is an input problem (nothing previously checked that a
supplied `vel3D.dat` actually covers the real network under the declared
`LAT_3D`/`LON_3D`/`ROT_3D`), not a small-aperture-network limitation: the
same small-aperture (~15 km, borehole+surface) geometry relocates
successfully once the 3D model is correctly registered and sized. See
`docs/hypodd_3d_notes.md` for the full backtrace and reproduction recipe.

- `check_vel3d_coverage(vel3d_grid, lat_3d, lon_3d, rot_3d, catalog,
  station_table)` projects every accepted event's and every station's
  location into the model's frame (`_hypodd_sdc_xy_km`, a direct port of
  `setorg.f`/`dist.f`, empirically verified against a real debug hypoDD
  run's own `hypoDD.loc` output) and raises `openmeq.config.ConfigError`,
  naming every offending station/event and which axis it falls outside of,
  **before** hypoDD ever runs — this is the "add validation ... fails with
  a clear error" fix path (the alternative, patching the vendored Fortran,
  was rejected: the compiled binaries are shared build artifacts and a
  clean input check catches the same failure earlier with a far more
  actionable message).
- A station's z-coordinate for this check is its §6 effective elevation,
  clamped to `0.0` whenever negative — mirroring `getdata.f:275-278`'s own
  silent clamp for any `IMOD` other than `5` (which includes `IMOD=9`), so
  the check validates the coordinate hypoDD will *actually* query the
  model at.
- Wired into `openmeq.relocate.setup.setup_hypodd` (M9.5, below): called for
  every 3D run, immediately after `ensure_compiled_limits`, using whatever
  catalogue is currently available (even an empty one pre-detection —
  station-only coverage is still checked from day one).

## 6c-2. `scripts/setup_hypodd.py` / `openmeq.relocate.setup` (M9.5 / OpenMEQ-6ld.5)

The importable orchestration this section's writers/checkers above (M9.2,
M9.3, M9.4, M9.6) are wired together by, matching M5.2's
`openmeq.nlloc_setup.setup_nlloc` pattern:
`openmeq.relocate.setup.setup_hypodd(cfg, paths, *, force=False,
nlloc_to_simul=False) -> bool`, with `scripts/setup_hypodd.py` a thin Click
CLI (`-c/--config`, `--nlloc-to-simul`, `--force`) around it — so M10.2/
M10.4 can `import openmeq.relocate.setup` and call the same function
directly rather than shelling out to the script (PLAN.md §7/§9: M9 must be
usable independently of, and before, the rest of the pipeline).

Sequence: **(a)** if `nlloc_to_simul`, resolve `<cfg.nlloc.velocity_model.
file>.{P,S}.mod.hdr` (S optional — required only if `cfg.hypodd.
velocity_model.simul2000.iphase != 1`, else a `ConfigError`) and call
`nlloc_to_simul.nlloc_grid_to_vel3d`, writing `cfg.hypodd.velocity_model.
file` (the `vel3D.dat`), passing `hypodd_limits.compiled_vel3d_node_limits()`
for `max_nx`/`max_ny`/`max_nz` rather than that function's own shipped
defaults; **(b)** refresh `relocate/work/event_id_map.csv`/
`station_label_map.csv` (`build_event_id_map`/`build_station_label_map`,
§6c) from whatever catalogue is currently available (an empty one if
`paths.catalog_mag_xml` doesn't exist yet); **(c)** `write_station_dat`;
**(d)** `write_ph2dt_inp`; **(e)** `write_hypodd_inp_1d` (reads
`cfg.hypodd.velocity_model.file` via `openmeq.velocity.read_layers`) or
`write_hypodd_inp_3d` (requires `cfg.hypodd.velocity_model.file`, the
`vel3D.dat`, to already exist on disk — from step (a) or user-supplied —
checked explicitly since the writer itself doesn't touch the filesystem);
**(f)** if the catalogue is available, `ensure_compiled_limits` (passing a
`Vel3DGrid` from `read_vel3d_header` for a 3D run — folds in step (g)'s
recompile-if-needed), else log a `WARNING` and skip the check entirely
(never a hard failure — PLAN.md §9: this script must run on day one, before
any events exist); **(g)** for a 3D run, `check_vel3d_coverage` (§6c,
above) immediately after, regardless of catalogue availability.

Idempotent via `openmeq.manifest.StepManifest(paths, "relocate")`, key
`"hypodd_setup"` (a distinct key from M10.2/M10.4's own `"relocate"`-step
manifest entries, sharing the same `relocate/manifest.json` file — the
manifest contract is per-`key`, not per-file) — fingerprinted on
`cfg.hypodd`, `cfg.stations`, the station table's contents, the velocity
model input file(s)' mtime/size (the layer file for 1D; the NLLoc `.{P,S}.
mod.{hdr,buf}` files for `nlloc_to_simul`, or the `vel3D.dat` itself
otherwise), the `nlloc_to_simul` flag, and `paths.catalog_mag_xml`'s mtime/
size (so a newly-appeared or changed catalogue always re-triggers the limit
check even if nothing else changed).

## 6d. NLLoc 1D/3D grids and the location-grid-bounds convention

Built in M5.2 (`openmeq/nlloc_setup.py`), verified against
`docs/nlloc_interface.md` (the M5.1 spike). `docs/nlloc_interface.md`'s
`LOCGRID` note: NLLoc silently locates 0 events (exit code 0, no error) if
the location search grid's z-range (or, in `GRID3D` mode, x/y range)
exceeds the travel-time grid's own coverage. Since M6.1's own `LOCGRID`
statement is built from the same `cfg.nlloc.grid` M5.2 already has,
`openmeq.nlloc_setup.location_grid_bounds(cfg, station_table)` is the single
place that computes the eventual location grid's bounds, so both M5.2
(sizing the shared Vel2Grid/Grid2Time grid generously around it) and M6.1
(writing a matching `LOCGRID` statement) agree:

- **Horizontal** (x, y): centered on `cfg.nlloc.grid.origin`, spanning
  `+-size_km[0]/2` and `+-size_km[1]/2`.
- **Vertical** (z, NLLoc's positive-down convention): from
  `-Z_MARGIN_ABOVE_SEA_LEVEL_KM` (2.0 km above sea level, comfortably
  covering every station's negative effective elevation, §6) down to
  `size_km[2]`; widened further if any station's own effective elevation
  would otherwise fall outside that range (a deep borehole).

M6.1's per-event NLLoc runner must call `location_grid_bounds` (not
recompute this itself) to build its own `LOCGRID xNum yNum zNum xOrig yOrig
zOrig dx dy dz gridType saveFlag` statement, so the grid it searches is
always the one M5.2's travel-time grids were actually sized to cover.

`openmeq.nlloc_setup.write_control_common(cfg, paths)` is the one place
that renders `CONTROL`/`TRANS` (`TRANS SIMPLE latOrig lonOrig 0.0`, centered
on `cfg.nlloc.grid.origin`) -- every control file for one project must use
byte-identical `TRANS`/`MAPLINE` statements (`docs/nlloc_interface.md` §4),
so M6.1 must call this function too rather than re-deriving the line.

**3D model grids (M5.3 / OpenMEQ-mev.3).** `cfg.nlloc.velocity_model.type ==
"3d"` skips `build_1d_grid`/`Vel2Grid` entirely: `openmeq.nlloc_setup.link_3d_grid`
validates and then copies/symlinks a user-supplied NLLoc grid
(`cfg.nlloc.velocity_model.file` is a *root name*; `<file>.{P,S}.mod.{hdr,buf}`
must already exist -- PLAN.md §1, 3D grids are always user-supplied, never
built by this project) into the same `<root>.{P,S}.mod.{hdr,buf}` convention
`build_1d_grid` produces, so `run_grid2time` and everything downstream is
unaware of which path produced its input.

- `openmeq.nlloc_setup.read_grid3d_hdr(hdr_path) -> Grid3DHeader` parses an
  NLLoc 3D grid `.hdr` file (format verified directly against the installed
  `external/NonLinLoc/src/GridLib.c` source, not a doc page -- no page states
  it as a single worked example). The grid's own `TRANSFORM` line is kept
  verbatim (`Grid3DHeader.transform_line`).
- `openmeq.nlloc_setup.check_3d_grid_coverage(hdr_path, cfg, bounds)` raises
  `ConfigError` (naming the specific shortfall, e.g. `"3D grid covers z in
  [0.000, 8.000] km but location grid needs [-2.000, 10.000] km"`) unless the
  grid's own extent **strictly contains** `bounds` (a `LocationGridBounds`
  from `location_grid_bounds`, §6d above) on every axis -- **zero tolerance
  margin** (NLLoc's own out-of-grid behaviour should not be relied upon;
  confirmed empirically for the 3D case too, `external/NonLinLoc/src/NLLocLib.c:1904-1911`:
  a genuine 3D travel-time grid that doesn't entirely contain the initial
  `LOCGRID` gets its observation *silently* rejected, exit code 0, the same
  failure mode as the already-documented 2D/`LOCGRID` case above). It also
  validates the grid's own `TRANSFORM` against `cfg.nlloc.grid.origin`: the
  `.hdr` format *does* self-describe its transform (resolving this task's own
  design-note hedge), so rather than a separate user-supplied config field,
  a 3D grid built with anything other than a matching `TRANS SIMPLE
  latOrig lonOrig 0.0` (no rotation) is rejected outright -- this project has
  no general point-transform code to reconcile two different projections,
  and guessing wrong would silently misplace every travel time.
- `Grid2Time`'s `GTMODE` `gridMode` field (`GRID2D` vs `GRID3D`) cannot be
  hardcoded for both paths: `external/NonLinLoc/src/Grid2Time1.c:288-296`
  hard-errors `GTMODE GRID2D` unless the model grid's own `numx == 2`
  exactly (`build_1d_grid`'s own convention), while `GRID3D` accepts any
  `numx`. `run_grid2time` therefore reads each phase's own model `.hdr` at
  run time (`_grid_mode_for_model`) rather than being told which setup path
  produced it -- this is the one internal change M5.3 made inside
  `run_grid2time`; its public signature is unchanged, and it still does not
  call `build_1d_grid`/`link_3d_grid` itself.

## 7. External tools

| Tool | Location | Built by |
|---|---|---|
| PhaseNet | `external/PhaseNet` (git clone at a pinned commit, `pip install -e`); model `external/PhaseNet/model/190703-214543` | `install.sh` |
| NonLinLoc | `external/NonLinLoc` source → `external/bin/{NLLoc,Vel2Grid,Vel2Grid3D,Grid2Time,...}` | `install.sh` (cmake) |
| hypoDD 2.1 (3D) | `HYPODD.zip` → `external/HYPODD` → `external/bin/{hypoDD,ph2dt}` | `install.sh` (make); `openmeq.relocate.hypodd_limits.ensure_compiled_limits` (M9.3) may recompile with larger limits, wired into `setup_hypodd.py`'s CLI in M9.5 |

## 8. Test data

`tests/data/make_synthetic.py` builds a deterministic small dataset:
- 8 stations: 6 surface `HH?`, and 2 borehole `DP?` with unknown orientation
- 2 days at 200 Hz
- one gap, one spiky channel, one flat-lined channel
- about 20 synthetic events with P/S arrivals computed in a 1D model from known hypocentres
- StationXML with a flat response and a CSV
- the 1D model file and an example config

The pytest fixture `synthetic_project` (in `tests/conftest.py`) generates it once per session into a temporary folder and returns `(config_path, truth)`. `truth` holds, for each event, `event_id_hint` (true origin time), `origin` (time, lat, lon, depth_km), `ml` (the magnitude used to scale the synthetic amplitudes through the Hutton & Boore relation and the WA response), and `arrivals` (`{"NET.STA": {"P": t, "S": t}}`, keyed by **station**, not channel). `truth["known_issues"]` is a list of `{"seed_id", "kind": "gap"|"spikes"|"flat", "start", "end"}`, which the QC tests assert against. Each step's tests use it to check recovery. Travel times are straight-ray through the layers. They're a few percent (tens of ms) off NLLoc's travel times, so location tests should allow ~20–50 ms residuals and a few hundred m to ~1 km location error. Synthetic ML is accurate to about ±0.15. Details are in the `tests/data/make_synthetic.py` docstring. One generation is about 500 MB and takes about 8 s.
