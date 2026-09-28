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
| `openmeq.stations` | `load_inventory(paths) -> obspy.Inventory`; `station_table(paths) -> pandas.DataFrame` with columns `network,station,location,latitude,longitude,elevation_m,depth_m`; `apply_channel_map(stream, cfg.stations)`. |
| `openmeq.plotting` | `setup_style()`, `save_fig(fig, path)`, `plot_map_sections(df, stations, path, ...)`, `plot_hist(...)`, `plot_heatmap(...)`, `plot_timeseries(...)`, `plot_boxplot(...)`. |

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
pick/phasenet_raw/              # PhaseNet native output
pick/catalog_picks.xml          # events with automatic picks, no origins
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
relocate/work/                  # hypoDD run folder(s)
relocate/cache/<event_id>.npz   # pre-filtered waveforms for xcorr
relocate/dt.ct, relocate/dt.cc
relocate/catalog_dd.xml
<step>/diagnostics/             # every step
<step>/manifest.json            # every step
logs/<step>.log
```

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

## 7. External tools

| Tool | Location | Built by |
|---|---|---|
| PhaseNet | `external/PhaseNet` (git clone at a pinned commit, `pip install -e`); model `external/PhaseNet/model/190703-214543` | `install.sh` |
| NonLinLoc | `external/NonLinLoc` source → `external/bin/{NLLoc,Vel2Grid,Vel2Grid3D,Grid2Time,...}` | `install.sh` (cmake) |
| hypoDD 2.1 (3D) | `HYPODD.zip` → `external/HYPODD` → `external/bin/{hypoDD,ph2dt}` | `install.sh` (make); `setup_hypodd.py` may recompile with larger limits |

## 8. Test data

`tests/data/make_synthetic.py` builds a deterministic small dataset:
- 8 stations: 6 surface `HH?`, and 2 borehole `DP?` with unknown orientation
- 2 days at 200 Hz
- one gap, one spiky channel, one flat-lined channel
- about 20 synthetic events with P/S arrivals computed in a 1D model from known hypocentres
- StationXML with a flat response and a CSV
- the 1D model file and an example config

The pytest fixture `synthetic_project` (in `tests/conftest.py`) generates it once per session into a temporary folder and returns `(config_path, truth)`. `truth` holds, for each event, `event_id_hint` (true origin time), `origin` (time, lat, lon, depth_km), `ml` (the magnitude used to scale the synthetic amplitudes through the Hutton & Boore relation and the WA response), and `arrivals` (`{seed_id: {"P": t, "S": t}}`). `truth["known_issues"]` is a list of `{"seed_id", "kind": "gap"|"spikes"|"flat", "start", "end"}`, which the QC tests assert against. Each step's tests use it to check recovery.
