# PhaseNet `predict.py` interface (M4.1 spike)

**Status:** verified against the actually-installed clone at
`external/PhaseNet` (pinned commit `62005c6`, model
`model/190703-214543`), Python 3.10.19 / TensorFlow 2.11.0 / NumPy 1.26.4,
on this machine (28 cores, RTX 4070, no matching CUDA libraries). Every
claim below was checked by actually running `predict.py` as a subprocess,
not assumed from the reference-clone read that produced the original spec.
Automated coverage: `tests/pick/test_phasenet_spike.py` (marker
`requires_phasenet`).

This document is the reference for **M4.2** (chunk preprocessing) and
**M4.3** (batch runner). Where it says "must", later code should treat that
as a hard constraint, not a suggestion.

## 1. Environment

Confirmed inside the project's `.venv` (Python 3.10 via `uv`):

| Package | Pinned | Confirmed installed |
|---|---|---|
| Python | 3.10 | 3.10.19 |
| tensorflow | ==2.11.0 | 2.11.0 |
| numpy | ==1.26.4 | 1.26.4 |

```
$ python --version
Python 3.10.19
$ python -c "import tensorflow as tf; print(tf.__version__)"
2.11.0
$ python -c "import numpy; print(numpy.__version__)"
1.26.4
```

### GPU vs CPU

- `tf.config.list_physical_devices('GPU')` returns `[]` on this machine
  **even without** `CUDA_VISIBLE_DEVICES` set. The RTX 4070 is visible to
  the NVIDIA driver (`cuInit` succeeds enough to enumerate it in some
  paths), but TensorFlow 2.11's CUDA op kernels need `libcudnn.so.8` and
  (for TensorRT integration) `libnvinfer.so.7`/`libnvinfer_plugin.so.7`,
  none of which are installed for this driver/CUDA generation. TensorFlow
  logs warnings (`Cannot dlopen some GPU libraries ... Skipping
  registering GPU devices...`) and transparently falls back to CPU.
- Running with `CUDA_VISIBLE_DEVICES=""` explicitly set produces **exactly
  the same picks** (verified byte-identical `picks.csv` for the same
  input) as running with it unset, just without the `cuInit:
  CUDA_ERROR_NO_DEVICE` / dlopen warning spam and a marginally faster
  startup (no wasted CUDA probing). **Recommendation for M4.3:** always
  set `CUDA_VISIBLE_DEVICES=""` in the subprocess environment regardless
  of `cfg.parallel.phasenet_device`, since GPU is not currently usable here
  either way; treat `phasenet_device: gpu` as a no-op/warning on this
  machine until matching CUDA/cuDNN libraries are installed.
- **Gap:** `openmeq.external.run(cmd, cwd=None, log=None, timeout=None)`
  has no `env` parameter. To force `CUDA_VISIBLE_DEVICES=""` (or override
  any other env var) for a `run()` call, the caller must mutate
  `os.environ` in the parent process first (subprocess inherits it when
  `env=None`) — there's no way to pass a one-off env override without
  touching global state. `tests/pick/test_phasenet_spike.py` works around
  this with `monkeypatch.setenv`. **Follow-up for M4.3:** either give
  `run()` an `env: dict | None` passthrough (merged over
  `os.environ`), or have the batch runner set `CUDA_VISIBLE_DEVICES` in
  `os.environ` itself before calling `run()`. Filed as a contract gap,
  not fixed here (out of scope for this spike).

## 2. Exact subprocess invocation recipe

```
CUDA_VISIBLE_DEVICES="" \
python <phasenet_dir>/phasenet/predict.py \
    --model_dir=<phasenet_model_dir> \
    --data_dir=<dir containing the mseed files> \
    --data_list=<csv with a "fname" column> \
    --format=mseed \
    --result_dir=<output dir> \
    --batch_size=1 \
    --sampling_rate=100 \
    --min_p_prob=0.3 --min_s_prob=0.3 \
    [--amplitude] [--save_prob] [--highpass_filter=<hz>]
```

using `openmeq.external.phasenet_dir()` / `phasenet_model_dir()` for the
first two paths, and `openmeq.external.run(cmd, cwd=..., timeout=...)` to
invoke it (it raises `ExternalToolError` with captured stdout/stderr on a
non-zero exit or timeout — no need to inspect the returned
`CompletedProcess.returncode` separately).

- **`--model_dir`, not `--model`.** The installed `predict.py`'s argparse
  only defines `--model_dir`; `docs/README.md` in the same clone shows an
  example using `--model=...`, which is **stale/wrong** for this commit —
  using it would silently leave `args.model_dir=None` (the default) and
  crash inside `tf.train.latest_checkpoint(None)`. Don't copy the README's
  example verbatim.
- **`model_dir` must be the checkpoint directory itself**
  (`external/PhaseNet/model/190703-214543`), confirmed: the code calls
  `tf.train.latest_checkpoint(args.model_dir)`, which reads that
  directory's `checkpoint` file to find `model_95.ckpt`.
- **`cwd` does *not* matter, empirically.** `predict.py` does
  `from data_reader import ...` (a bare, non-package-relative import).
  This resolves because Python inserts the running script's own directory
  at `sys.path[0]` regardless of the process's current working directory
  — confirmed by invoking `predict.py` by absolute path from `cwd=/tmp`
  and from this project's worktree root, both succeeding identically. It
  is **not** necessary to set `cwd` to `external/PhaseNet/phasenet/`, and
  **not** necessary to add it to `PYTHONPATH` either, as long as the
  script is invoked by its own absolute (or otherwise resolvable) path.
  M4.3 may still set `cwd=phasenet_dir()/"phasenet"` for
  clarity/convention (that's what `tests/pick/test_phasenet_spike.py`
  does), but it is not load-bearing.
- Because `predict.py` calls `tf.compat.v1.disable_eager_execution()` at
  **module import time**, it cannot be imported and re-invoked in-process
  more than once per Python process. It must be run as one **subprocess
  per pick run** (batching all events into one `data_list.csv`), never
  imported as a library, and never one subprocess per event (model load
  is ~2.7 s, see §6 — invoking it per-event would dominate total runtime).

## 3. `--format` and `data_list.csv`

`predict.py --format` accepts `numpy`, `mseed`, `sac`, `hdf5`,
`mseed_array` (confirmed in `data_reader.DataReader.__init__` and
`DataReader_mseed_array`). This project uses `mseed`.

For `format=mseed` (and `sac`, `numpy`), `data_list.csv` is read with
`pandas.read_csv(..., header=0, sep="[,|\s+]", engine="python")` (falling
back to `sep="\t"` on a parse error), and **only the `fname` column is
used** (`self.data_list = csv["fname"]`). Other columns are parsed but
ignored by the mseed code path. Confirmed: the reference clone's
`demo/fname.csv` has `fname,E,N,Z` columns, and `read_mseed()` never reads
`E`/`N`/`Z` — it discovers channels from the miniSEED file's own trace IDs.
`fname` is joined with `--data_dir` (`os.path.join(data_dir, fname)`), so
it can include a subdirectory component (e.g. `<event_id>/<event_id>.mseed`
for this project's `chunks_pn/` layout).

`--stations` is **only** consumed by `format=mseed_array`
(`DataReader_mseed_array`); it is unused (and not required) for
`format=mseed`.

## 4. `read_mseed()` behaviour

`DataReader_pred.__getitem__` calls `read_mseed(path, response=None,
sampling_rate=100, highpass_filter=0.0)` per data-list row (`response` is
`None` unless `--response_xml` is given — this project doesn't pass it,
confirmed unnecessary since we don't feed StationXML to PhaseNet).

1. `obspy.read(fname)`, then `.merge(fill_value="latest")` — confirmed:
   all traces in the file are merged; multi-segment files with small gaps
   fill with the last-seen value, not zero or interpolation.
2. Traces under 10 samples are dropped.
3. **Resampling:** `if abs(trace.stats.sampling_rate - sampling_rate) >
   0.1: logging.warning(...); trace.interpolate(sampling_rate,
   method="linear")`. Confirmed: supplying already-100 Hz data (as this
   project's `chunks_pn/` will) skips this silently (no warning, no
   interpolation call) — only a genuine rate mismatch (e.g. leftover
   200 Hz input) triggers it, and it's a `logging.warning`, not an
   exception; execution continues either way.
4. `trace.detrend("demean")` (always) and, if `--highpass_filter > 0`,
   `trace.filter("highpass", freq=...)`.
5. All traces trimmed/padded to `[min(starttime), max(endtime)]` with
   `pad=True, fill_value=0`.
6. **Station grouping:** `station_ids[trace.id[:-1]]` — keyed by
   `NET.STA.LOC.BandInst`, i.e. the full SEED id **minus the final
   component character**. Confirmed empirically (debug print patched into
   a throwaway copy of `data_reader.py`, run against real fixtures):
   `XX.STA1..HH` groups `{'E','N','Z'}`, `XX.STA3..DP` groups `{'1','2','Z'}`.
7. **Channel-to-slot assignment.** Components present for a station are
   sorted by the fixed priority list `["3","2","1","E","N","U","V","Z"]`.
   **When exactly 3 components are present** (the normal, expected case),
   the model's 3 input slots are filled by **position in that sorted
   order** (`enumerate()`), *not* by the `comp2idx` mapping (which is only
   used as a fallback when a station has fewer than 3 components).
   Confirmed empirically for both conventions this project uses:

   | Present components | Sorted order | slot 0 | slot 1 | slot 2 |
   |---|---|---|---|---|
   | `{E,N,Z}` (oriented) | E,N,Z | E | N | **Z** |
   | `{1,2,Z}` (unoriented, `channel_map`'s `"1"`/`"2"`) | 2,1,Z | 2 | 1 | **Z** |

   **Z always lands in slot 2** (it's last in the priority list, so it's
   always the last-sorted, hence highest-`enumerate()`-index component
   whenever exactly 3 are present) — this holds for both the oriented
   Z/N/E and unoriented Z/1/2 conventions, so it's self-consistent with
   however the model was itself trained (same `read_mseed` code path).
   The two horizontal channels land in slots 0/1 in the priority-list
   order shown above (E before N; "2" before "1" — note this is the
   opposite of numeric order for the unoriented pair).
8. A component whose stream has an empty match logs `Empty trace: ...`
   and leaves that slot as zeros — i.e. **PhaseNet does not require
   exactly 3 components to run**, but for `< 3` present the fallback
   `comp2idx` mapping is used instead of positional sorting (see the
   design note above about `demo/fname.csv`'s columns not being read: the
   mseed path infers components from the file's own trace IDs, so a file
   genuinely missing a channel will silently pick with a zeroed slot,
   with no line in `data_list.csv` needed to declare that).
9. `data.transpose([1, 2, 0])` gives the per-file array shape
   `(nt, nx, 3)`, where `nx` is **that file's own station count** — see
   §7 for why this matters.

`normalize_long(data, window=3000)` (30 s @ 100 Hz) then per-channel
normalizes with a sliding 30 s window (15 s shift, linear-interpolated
between window centres). **Caveat found while tuning test fixtures:** a
much larger later arrival within roughly the same 30 s window can inflate
the interpolated local std enough to suppress a genuine, smaller earlier
arrival's post-normalization amplitude — i.e. a strong S phase can, in
principle, reduce P-pick recoverability for very close-in stations with a
short S–P interval and a large amplitude contrast between phases. Not
something to "fix" (it's how the pretrained model was itself trained/
normalized) — just something to be aware of if P recall looks worse than
expected for high-magnitude/near-station events in real data.

## 5. Output: `picks.csv`

Written to `<result_dir>/<result_fname>.csv` (`result_fname` defaults to
`"picks"`, so `<result_dir>/picks.csv`). `<result_dir>` is **auto-created**
by `predict.py` if it doesn't already exist (confirmed: no need to
pre-`mkdir` it, though doing so is harmless).

Confirmed columns (no `--amplitude`):

```
station_id,begin_time,phase_index,phase_time,phase_score,phase_type,file_name
```

With `--amplitude`, two more columns are appended:
`phase_amplitude,phase_amp` (`phase_amp` is a duplicate — `df["phase_amp"]
= df["phase_amplitude"]` — kept for backward compatibility with an older
schema, no reason to prefer one over the other).

- `station_id` is the same `NET.STA.LOC.BandInst` string used for grouping
  (§4.6) — **no final component character**, confirmed
  (`not station_id.endswith((".Z",".N",".E",".1",".2"))` holds for every
  row observed). **There is no per-channel attribution**: a P or S pick
  cannot be traced back to which specific input channel produced it, only
  to the station/instrument as a whole.
- `phase_type` is `"P"` or `"S"` (from `extract_picks`'s `phases=["P",
  "S"]`, channel-1 / channel-2 of the 3-class softmax, channel 0 being
  noise).
- `phase_index` is the sample index into that station's trimmed/padded
  window; `phase_time = begin_time + phase_index / sampling_rate` (from
  `docs/README.md`, confirmed by inspection of `postprocess.extract_picks`,
  which literally computes `pick_time = begin_time +
  timedelta(seconds=phase_index * dt)`).
- `begin_time` is `read_mseed`'s computed `min(trace.starttime)` for that
  file, **not** necessarily the nominal chunk start if traces within the
  file have different start times before the trim/pad step.
- Confirmed **no per-channel amplitude/probability distinction** — one row
  per (file, station, phase-type, detected-peak), no channel column at
  all, consistent with ARCHITECTURE.md §5's convention that PhaseNet picks
  are per-station, mapped to a single representative channel (Z for P,
  first-present-of-N/E/1/2 for S) only later, during QuakeML conversion
  (M4.4) — `picks.csv` itself carries no channel information to map back.

Example real rows produced by this spike's fixtures (batch_size=1,
`--min_p_prob=0.3 --min_s_prob=0.3`, `CUDA_VISIBLE_DEVICES=""`):

```
station_id,begin_time,phase_index,phase_time,phase_score,phase_type,file_name
XX.STA1..HH,2025-06-01T00:00:00.000,2000,2025-06-01T00:00:20.000,0.811,P,single_zne.mseed
XX.STA1..HH,2025-06-01T00:00:00.000,3198,2025-06-01T00:00:31.980,0.604,P,single_zne.mseed
XX.STA1..HH,2025-06-01T00:00:00.000,3203,2025-06-01T00:00:32.030,0.347,S,single_zne.mseed
XX.STA3..DP,2025-06-01T00:00:00.000,2500,2025-06-01T00:00:25.000,0.768,P,single_z12.mseed
XX.STA3..DP,2025-06-01T00:00:00.000,3901,2025-06-01T00:00:39.010,0.614,S,single_z12.mseed
```

(injected truth: `single_zne.mseed` P@20.0s/S@32.0s; `single_z12.mseed`
P@25.0s/S@39.0s — both recovered within 0.03 s here. A spurious extra `P`
row sometimes appears very close to the true S arrival, i.e. the model can
double-fire the P channel near a strong S onset; downstream code (M4.4)
should not assume exactly one P and one S row per station.)

## 6. Timing (CPU, this machine)

Measured via repeated `subprocess` invocations of the whole `predict.py`
process (fixed cost includes Python startup, TensorFlow import,
`disable_eager_execution()`, session/graph construction, and checkpoint
restore):

| Chunks in one `data_list.csv` | `batch_size` | Wall time |
|---|---|---|
| 1 | 1 | 2.74 s |
| 3 | 1 | 2.80 s |
| 30 | 1 | 2.97 s |

- **Fixed overhead (model load + process startup): ~2.7 s.**
- **Marginal cost per extra 60 s/100 Hz/3-station chunk: ~8 ms.**
- This overwhelmingly confirms the design note: invoke `predict.py`
  **once per pick run**, batching every event into one `data_list.csv`,
  never once per event.
- `batch_size=20` (all-same-station-count chunks) vs `batch_size=1`: no
  measurable speedup on this CPU (3.16 s vs 3.00 s for 30 identical
  3-station chunks) — batching only helps GPU throughput; on CPU-only
  hardware there is no performance reason to prefer `batch_size>1`, only
  a correctness *risk* (see §7).
- `--save_prob` (writes `<result_dir>/result.h5` with one HDF5 dataset per
  input file, keyed by the `fname` string, shape `(nt, nx, 3)` per file)
  adds **no measurable overhead** for these small chunks (3.09 s without
  vs 3.10 s with, over 30 chunks) — cheap enough to always pass, **as
  long as every `fname` in a single `predict.py` invocation is unique**
  (see gotcha below). **Recommendation:** M4.3 should pass `--save_prob`
  by default so M4.4 can plot probability traces, since the cost is
  negligible at this chunk size.
  - **Gotcha confirmed:** the HDF5 dataset key is the raw `fname` string
    from `data_list.csv`, not a synthetic unique index. Two rows with the
    *same* `fname` value in one `data_list.csv` cause `save_prob_h5` to
    raise `ValueError: Unable to synchronously create dataset (name
    already exists)` and the whole run fails (exit 1) — confirmed by
    reproducing it directly. This project's `chunks_pn/<event_id>/
    <event_id>.mseed` naming already guarantees uniqueness (event ids are
    unique), so this shouldn't bite in practice, but M4.3 must not
    accidentally list the same chunk file twice in one run when
    `--save_prob` is on.

## 7. `batch_size` — the real constraint is per-file station count, not just time length

The spec's original assumption (from the reference-clone read) was that
fixed-length chunks are sufficient for `batch_size > 1` to work without
`drop_remainder` issues. **This is incomplete.** Confirmed by direct
reproduction:

- `DataReader_pred.dataset()` calls `.batch(batch_size,
  drop_remainder=False)` on a `tf.data.Dataset` whose per-item shape is
  `[None, None, 3]` — i.e. **both time length and station count (`nx`)
  are dynamic per item**, and TensorFlow's plain (non-padded) `.batch()`
  requires every item combined into one batch to have an **identical**
  concrete shape.
- Two chunks of the *same* 60 s/100 Hz duration but **different station
  counts** (e.g. a 1-station file and a 3-station file, both perfectly
  valid, fixed-length chunks per this project's chunk-prep convention)
  raise, at `batch_size=2`:
  ```
  InvalidArgumentError: Cannot add tensor to the batch: number of
  elements does not match. Shapes are: [tensor]: [6000,3,3], [batch]: [6000,1,3]
  ```
  reproduced exactly in `tests/pick/test_phasenet_spike.py::
  test_predict_py_batch_size_gt_1_requires_equal_station_count`.
- `batch_size=1` always works regardless of per-file `nx` or length,
  since no cross-item concatenation happens.
- **Real-world implication for M4.3:** this project's
  `chunks_pn/<event_id>/<event_id>.mseed` files will generally have
  **different station counts per event** (network availability differs
  event to event — that's the whole point of `chunks/completeness.csv`).
  Batching many different events' chunks into one `predict.py` call with
  `cfg.phasenet.batch_size` (default 20, per `config/example.yaml`) **will
  fail** the moment two events in the same batch have a different number
  of participating stations, which is the common case, not an edge case.
- **Recommendation:** M4.3 should invoke `predict.py` with
  **`--batch_size=1`** unconditionally, regardless of
  `cfg.phasenet.batch_size`. Given §6's timing (marginal cost ~8 ms/chunk,
  no measured CPU speedup from larger batches anyway), there is no
  performance reason to do otherwise on this hardware. If GPU batching
  ever matters on different hardware, the runner would need to
  pre-group events by identical station count (and possibly pad/mask)
  before raising `batch_size` — treat `cfg.phasenet.batch_size` as
  currently unused/aspirational and flag this as a contract note for
  whoever revisits GPU support.

## 8. Synthetic fixture design (for anyone extending the test)

A plain sinusoidal (Ricker-style) wavelet for both P and S did **not**
reliably get labelled correctly by the pretrained model: isolated bursts
(P-shaped or S-shaped, alone or paired) were almost always classified as
"P" regardless of frequency/amplitude/duration tuning — the model appears
to need an S arrival that's plausibly *low-frequency, broadband, and
appropriately timed/scaled relative to a real preceding P*, not just "big
and slow." Band-limited (Butterworth-filtered white noise) causal bursts
worked much better, but recovery of **both** phases, correctly labelled,
within 0.1 s, above the default `min_p_prob=min_s_prob=0.3` threshold, was
still sensitive to the specific noise realization: a grid search over 200
RNG seeds (same tuned envelope/frequency-band parameters, `predict.py` run
once per seed with `batch_size=1`) found only ~10% reliably recovered both
phases correctly-labelled within tolerance. `tests/pick/
test_phasenet_spike.py` and this doc use one verified seed (`base=660`).
**Do not change the amplitude/frequency/seed constants without re-running
`predict.py` against the result** — small changes can silently flip a
phase's label or drop it below threshold. See `build_fixtures.py` /
`onset_experiment*.py` / `seed_search.py` in this spike's scratch history
for the full tuning trail if this needs revisiting.

## 9. Summary table (work-to-do checklist)

| Claim | Verified |
|---|---|
| `--format` in `{numpy, mseed, sac, hdf5, mseed_array}` | yes |
| `data_list.csv` only needs `fname` for mseed; extra columns ignored | yes |
| `read_mseed` merges with `fill_value="latest"` | yes |
| Resampling: linear interpolation, warns if `\|rate-target\|>0.1`, silent at exact 100 Hz | yes |
| Station grouping key: `NET.STA.LOC.BandInst` (no component char) | yes |
| Slot order: Z always last of 3; horizontals in priority-list order | yes (empirical debug print) |
| `picks.csv` columns (+amplitude columns) | yes |
| `station_id` has no per-channel attribution | yes |
| CLI flags actually used | yes (see §2) |
| `batch_size>1` needs equal-length **and** equal-station-count inputs | yes (equal-station-count was the missing part) |
| `cfg.phasenet.batch_size=20` works for our fixed-length chunks | **no** — needs equal `nx` too; recommend `batch_size=1` |
| Checkpoint loaded via `tf.train.latest_checkpoint(model_dir)` | yes |
| `cwd`/`PYTHONPATH` requirement for the relative import | **not required**, contrary to the spec's assumption |
| Python 3.10 / TF 2.11.0 / NumPy 1.26.4 match pins | yes |
| GPU vs CPU | GPU present but unusable (missing cuDNN/TensorRT libs); CPU fallback silent and correct |
