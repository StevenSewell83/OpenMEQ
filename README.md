# OpenMEQ

OpenMEQ is a microseismic processing pipeline: it takes raw continuous
miniSEED waveforms and turns them into a relocated, magnitude-tagged
earthquake catalogue. It chains network detection (STA/LTA + coincidence
triggering), automatic P/S picking with [PhaseNet](https://github.com/AI4EPS/PhaseNet),
two-pass event location and filtering with [NonLinLoc](https://github.com/alomax/NonLinLoc),
optional interactive manual review with [ObsPyck](https://github.com/joachimwassermann/obspyck),
local (ML) magnitudes, and final double-difference relocation with
[hypoDD](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) -- with QC and
diagnostic plots generated automatically at every step.

## Quick start

### Install

```bash
git clone https://github.com/StevenSewell83/OpenMEQ.git && cd OpenMEQ && ./install.sh
```

`install.sh` creates a Python 3.10 virtual environment (`.venv`, via
[`uv`](https://docs.astral.sh/uv/)), installs OpenMEQ and its dependencies
(ObsPy, TensorFlow 2.11, PyQt5, ObsPyck, ...), clones and installs PhaseNet
at a pinned commit (with its pretrained model weights), builds NonLinLoc from
source with CMake, unpacks and builds hypoDD 2.1 from the bundled
`HYPODD.zip`, and finishes with a smoke check (imports, binaries run,
hypoDD's bundled example relocates end to end).

**System prerequisites** (checked up front; the script fails fast naming
whatever is missing):

- `git`, `unzip` -- fetch PhaseNet/NonLinLoc sources, unpack `HYPODD.zip`
- `cmake`, `gcc` -- build NonLinLoc
- `gfortran`, `make` -- build hypoDD/ph2dt
- [`uv`](https://docs.astral.sh/uv/) on `PATH` or at `~/.local/bin/uv` --
  `install.sh` does not install `uv` itself

Other useful flags: `./install.sh --dev` also installs the `[dev]` extra
(pytest, ruff); `./install.sh --skip-external` installs just the Python
package (venv + `pip install -e .`), with no PhaseNet/NonLinLoc/hypoDD
build -- for CI or offline development, but no pipeline step that shells out
to an external tool will work until you run a full install. Re-running
`install.sh` is safe (idempotent).

Once installed:

```bash
source .venv/bin/activate
openmeq --help
```

### Try it on the synthetic demo dataset

A deterministic synthetic dataset (8 stations, 2 days at 200 Hz, ~20 events,
with an injected data gap, a spiky channel and a flat-lined channel) ships
with the repo for exactly this purpose:

```bash
python tests/data/make_synthetic.py /tmp/openmeq-demo
openmeq run -c /tmp/openmeq-demo/config.yaml
```

That's the whole chain -- QC, detection, chunking, PhaseNet picking,
location, magnitudes and hypoDD relocation -- run end to end (about 80-95
seconds on a modest machine; a second run with nothing changed takes about
2-3 seconds, since every step skips recomputation once its inputs and
config are unchanged). Results land under
`/tmp/openmeq-demo/runs/synthetic/`; see
[docs/usage.md](docs/usage.md) for what each output file/diagnostic means.

### Run on your own data

1. Copy [`config/example.yaml`](config/example.yaml) and edit `data`,
   `stations`, `nlloc.grid` and `nlloc.velocity_model`/`hypodd.velocity_model`
   for your deployment. Every key is documented, with its type, default and
   unit, in [docs/config_reference.md](docs/config_reference.md).
2. `openmeq run -c your_config.yaml` for the whole chain, or `openmeq <step>
   -c your_config.yaml` (`qc`, `detect`, `chunk`, `pick`, `locate`, `review`,
   `magnitude`, `relocate`) to run one step at a time. See
   [docs/usage.md](docs/usage.md) for what each step reads/writes and when
   to run it standalone.
3. **Before trusting the hypoDD/ph2dt relocation on real data**, read
   [docs/hypodd_tuning.md](docs/hypodd_tuning.md): the shipped `hypodd`/
   `ph2dt` distance and weighting defaults are tuned against this project's
   own synthetic test network, whose events are spread over roughly 19 km.
   Real microseismic sequences are usually much tighter clusters -- tens to
   hundreds of metres -- and the defaults need rescaling accordingly; that
   document gives the explicit scaling rule and a troubleshooting table.

## Pipeline overview

```
qc -> detect -> chunk -> pick -> locate -> [review] -> magnitude -> relocate
```

| Step | Command | What it does |
|---|---|---|
| QA/QC | `openmeq qc` | miniSEED index, station metadata merge, availability, noise statistics, channel flags |
| Detection | `openmeq detect` | STA/LTA + network coincidence triggering |
| Chunking | `openmeq chunk` | per-event raw waveform chunks |
| Picking | `openmeq pick` | PhaseNet P/S picks, converted to QuakeML |
| Location | `openmeq locate` | two-pass (P-only, then P+S) NonLinLoc location and residual filtering |
| Review *(optional)* | `openmeq review` | interactive manual re-picking/relocation with ObsPyck |
| Magnitude | `openmeq magnitude` | local magnitude (ML) |
| Relocation | `openmeq relocate` | hypoDD double-difference relocation (cross-correlation + catalog data) |

Run the whole chain with `openmeq run -c config.yaml` (supports `--force`,
`--from`/`--to` a sub-range, and `--skip-review`), or any step on its own.
Every step writes its own numbered subfolder of `cfg.project.output_dir`
(`0_setup/`, `1_qc/`, `2_detection/`, `4_data_chunks/`, `5_ml_picks/`,
`7_location/`, `8_relocation/`, `9_plots/`, matching the MicroQuake run
layout), each with a `diagnostics/` folder of PNG plots and CSV/text
summaries, plus a `logs/<step>.log` and a `config_snapshot.yaml`. Two setup scripts (`scripts/setup_nlloc.py`,
`scripts/setup_hypodd.py`) can be run independently of the rest of the
pipeline, e.g. to build travel-time grids or check compiled array limits
before any events exist. Full detail, including exact inputs/outputs/options
per step and the review step's interactive/resume behaviour, is in
[docs/usage.md](docs/usage.md).

## Repository layout

```
install.sh                  # single-command installer
pyproject.toml
HYPODD.zip                  # hypoDD 2.1 (3D) source, unpacked + compiled by install.sh
config/example.yaml         # documented reference config
scripts/
  setup_nlloc.py            # NLLoc grid setup, standalone
  setup_hypodd.py           # hypoDD/ph2dt setup, standalone
  gen_config_reference.py   # generates docs/config_reference.md
  benchmark.py               # pipeline timing harness
openmeq/
  config.py                 # YAML loading + pydantic validation
  cli.py                    # `openmeq <step>` / `openmeq run`
  paths.py                  # run directory layout
  index.py                  # miniSEED index
  stations.py                # StationXML file/folder + CSV -> merged inventory
  catalog.py                 # QuakeML helpers (event ids, rejection, extras)
  external.py                 # external tool discovery + subprocess runner
  manifest.py                 # skip-unchanged step manifests
  parallel.py                 # process-pool helper
  log.py                      # logging setup
  velocity.py                  # 1D layer-file parsing
  nlloc_setup.py                # NLLoc grid setup (backs scripts/setup_nlloc.py)
  chunk.py, chunk_diagnostics.py
  qc/, detect/, pick/, locate/, review/, magnitude/, relocate/
  plotting/                    # shared map/section/histogram/etc. plot functions
tests/
  data/make_synthetic.py       # synthetic dataset generator
  locate/, pick/, integration/
external/                      # git-ignored: PhaseNet clone, NonLinLoc/hypoDD builds
docs/
```

## Documentation

- [docs/usage.md](docs/usage.md) -- every command's inputs/outputs/options
  and diagnostics, the manual review workflow, resuming and skip-unchanged,
  parallelism settings, troubleshooting.
- [docs/config_reference.md](docs/config_reference.md) -- every config key,
  its type, default and meaning/unit (generated from the live config
  schema).
- [docs/hypodd_tuning.md](docs/hypodd_tuning.md) -- hypoDD/ph2dt tuning for
  small-aperture microseismic networks; **read before relocating real
  data.**
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) -- the binding module/data
  contracts (run directory layout, QuakeML conventions, units).
- [docs/PLAN.md](docs/PLAN.md) -- the original design/build plan.
- [docs/performance.md](docs/performance.md) -- QC daily-statistics
  performance notes and benchmarks.
- [docs/validation.md](docs/validation.md) -- end-to-end validation results
  against the synthetic dataset.
- [docs/nlloc_interface.md](docs/nlloc_interface.md),
  [docs/phasenet_interface.md](docs/phasenet_interface.md),
  [docs/obspyck_notes.md](docs/obspyck_notes.md),
  [docs/hypoddpy_audit.md](docs/hypoddpy_audit.md),
  [docs/hypodd_3d_notes.md](docs/hypodd_3d_notes.md) -- implementation notes
  on each external tool's interface, empirically verified against the
  installed versions this project builds.
