# QC daily-stats performance (OpenMEQ-5nl.8)

Optimisation pass on `openmeq/qc/stats.py` (M1.4, `OpenMEQ-5nl.4`), which
measured ~1 channel-day/s serial and only ~1.7 channel-days/s with 8 workers
on full 200 Hz channel-days (17.28M samples each) -- too slow for a real
archive (e.g. 50 channels x 365 days ~= 18k channel-days, ~3 hours at the
original rate).

**Machine**: 28 logical cores, 94 GB RAM. Shared with other concurrent work
during this measurement (see "Measurement caveats" below) -- absolute
numbers are noisier than they would be on a quiet machine, but the
*relative* before/after and scaling comparisons below were taken with care
to stay fair under that noise (interleaved timing, or the same load window
for a before/after pair).

## 1. Profiling: where the time went

`stats_for_channel_day` on one synthetic full-day channel (17.28M samples,
int32 STEIM2-decoded), cProfile + manual phase timing, before any change:

| Phase | Cost (per channel-day) | Notes |
|---|---:|---|
| `DataIndex.read` (file read + STEIM2 decode + merge) | ~45-56 ms | obspy; not touched here |
| `stream.split()` | ~9-11 ms | deep-copies trace metadata even with no gap |
| `np.median(x)` (spike detection: median) | ~100 ms | full-array `np.partition`, incl. a hidden defensive copy |
| `np.median(abs(x - median))` (MAD) | ~100 ms | second full-array partition; `abs(x-median)` recomputed *again* for the spike-count comparison |
| `scipy.stats.kurtosis` | ~160-170 ms | re-demeans `x` twice (once per moment order) and uses `np.power(d, 4)` |
| run-length encoding (flat/clip) | ~140 ms | `cumsum` + `bincount` + a full per-sample gather (`run_len[run_id]`) just to threshold and count |
| pooled `sum(x**2)` / `sum(abs(x))` | ~30-40 ms | elementwise square + a separate reduction (extra full-size temp array) |
| **Total** | **~800-870 ms** | |

Two full `np.median` passes, a duplicated `abs()` computation, and a
duplicated demean inside `scipy.stats.kurtosis` line up closely with the
issue's suspicion ("two full np.median passes ... STEIM2 decode ...").
STEIM2 decode itself was **not** the dominant cost (~6% of the total) --
the per-sample statistics computation was.

## 2. Optimisations applied

All in `openmeq/qc/stats.py`; **no output-shape or column change**, and
every existing `tests/test_qc_stats.py` assertion still passes (see
"Correctness" below).

1. **`np.median(..., overwrite_input=True)`** for both the median and MAD
   partition calls (`_median_mad_spike_count`) -- skips `np.partition`'s
   internal defensive copy; safe because neither array's *order* matters
   afterwards (only kurtosis reads `x_all` again, and kurtosis doesn't
   care about order).
2. **`dev = abs(x_all - median)` computed once**, reused for both the MAD
   and the spike-count threshold comparison (previously recomputed from
   scratch for the spike count).
3. **Custom kurtosis (`_fast_kurtosis`)**: demeans once (not twice), gets
   the 4th power via `d2 = d*d; d2*d2` (elementwise multiply) instead of
   `np.power(d, 4)`, and runs in **float32** (kurtosis has no
   tight-tolerance consumer -- every assertion on it in the test suite is
   a wide-margin inequality; see "Correctness / tolerances" below).
4. **`np.dot(seg_f, seg_f)`** instead of `np.sum(seg_f**2)` for the pooled
   sum of squares -- one BLAS-level pass instead of materialising a
   same-size squared temp array plus a separate reduction.
5. **Run-length encoding rewritten around run *boundaries*, not a
   per-sample expansion** (`_segment_run_counts`): the old version built a
   full array the size of the input (`bincount` + a fancy-index gather,
   `run_len[run_id]`) purely to threshold and count it; the new version
   only ever allocates arrays sized by the *number of runs* and gets
   `flat_count`/`clip_count` from two masked `.sum()`s over that
   run-sized array. Also replaced `np.diff(..., append=n)` (a generic,
   axis-aware wrapper) with direct slicing into a pre-sized output array.
6. **Skip `Stream.split()` when nothing is masked** -- it unconditionally
   deep-copies trace metadata even with zero gaps, which is the
   overwhelming majority of channel-days.
7. **Avoid needless concatenation**: a single-segment (no-gap)
   channel-day no longer runs its lone segment through
   `np.concatenate([seg])`, which still copies.
8. **Per-worker-process caching of `DataIndex`/`QcConfig`**
   (`_cached_index`/`_cached_qc_cfg`, keyed by path): previously every
   single `(seed_id, day)` item re-parsed `inventory/index.json` and
   re-validated the config YAML from scratch, even though a
   `ProcessPoolExecutor` worker is long-lived and handles many items over
   its lifetime, all sharing the same index/config for one
   `compute_daily_stats` call.
9. **`chunk_days` batching**, mirroring `openmeq.qc.availability`'s
   existing M1.3 pattern exactly: one `pmap` task is now one seed id x up
   to `cfg.parallel.chunk_days` consecutive days, reducing per-task
   `ProcessPoolExecutor` scheduling/IPC overhead at production scale.

One considered-and-rejected idea, flagged for anyone tempted to try it
again: accumulating `mean_abs`'s pooled sum directly off the raw `int32`
array (half the bytes of the float64 cast) via `np.abs(seg).sum(...)`.
**This silently overflows** for exactly `-(2**31)` (two's complement has no
positive counterpart for that value) -- and `-(2**31)` is precisely one of
this module's own two int32 saturation values that the clip-detection path
explicitly checks for. A negatively-clipped channel would have silently
gotten a corrupted (negative) `mean_abs`. Caught by hand before it reached
a test; `mean_abs` stays on the float64 array for this reason (see the
comment left in `compute_channel_day_stats`).

## 3. Serial throughput: before vs. after

Two measurements, both against the same real synthetic channel-day data
(`tests/data/make_synthetic.py`'s generator, 200 Hz, one full day):

**A. Interleaved single-item comparison** (both implementations imported
side by side in the same process and timed alternately, to cancel out
host-load drift rather than compare two separate runs):

```
OLD: mean=929.1ms  median=902.3ms  min=824.1ms
NEW: mean=516.8ms  median=512.0ms  min=466.9ms
speedup (mean):   1.80x
speedup (median): 1.76x
speedup (min):    1.77x
```

**B. Full `tests/test_qc_stats.py` throughput print** (48 channel-days,
`n_workers=1`, the test suite's own config) -- consistent with (A):

```
before: 48 channel-days in 38.06s (1.3 channel-days/s)
after:  48 channel-days in 22-28s  (1.7-2.1 channel-days/s)
```

**Result: ~1.7-1.8x serial throughput, short of the 3x target.**
Documented reason: see section 5 below -- the remaining cost
(`np.partition`-based median/MAD, ~35% of the per-item total) is a
comparison/memory-bandwidth-bound operation on a 17.28M-element array that
does not meaningfully speed up under `float32` (measured ~5% faster only;
see section 5), and further speedup would require either changing the
statistic's exact definition (approximate/sampled median -- rejected: the
issue requires results to "stay identical, or within documented float
tolerance", and a sampled median is a different *algorithm*, not a
tolerance difference) or a compiled (Cython/numba) selection kernel, which
is a materially larger change than this issue's scope.

## 4. Parallel scaling: before vs. after

Repeated-real-channel-day benchmark (48 real channel-days x 4 = 192 items,
`chunk_days=1`, to reach a large-enough item count that per-pool spawn
overhead is amortised -- see the caveats below):

| n_workers | Before (channel-days/s) | Before, relative | After (channel-days/s) | After, relative |
|---:|---:|---:|---:|---:|
| 1  | 1.27 | 1.00x | 2.17 | 1.00x |
| 4  | 2.31 | 1.82x | 4.51 | 2.08x |
| 8  | 2.42 | 1.90x | 4.32 | 1.99x |
| 16 | 2.42 | 1.90x | 4.21 | 1.94x |

**Both before and after plateau at ~1.9-2.0x by 8 workers** -- the
optimisations raised the throughput at every worker count by roughly the
same ~1.7-1.8x (matching section 3's serial number), but did **not**
change the *shape* of the scaling curve. Scaling is not near-linear to 8
workers, before or after.

## 5. Documented bottleneck: memory bandwidth, not I/O or scheduling

To isolate *why* scaling plateaus, a separate benchmark
(`pure_compute_scaling.py`, not part of the repo -- a throwaway scratch
script) generates each channel-day's data **in-process** (no file I/O, no
`DataIndex`, no obspy read/decode) and runs only
`compute_channel_day_stats`'s per-segment math through the same
`ProcessPoolExecutor`-based `pmap`, at increasing worker counts:

```
n_workers=  1  throughput=1.35 items/s  (1.347 items/s/worker)
n_workers=  2  throughput=1.94 items/s  (0.970 items/s/worker)
n_workers=  4  throughput=2.94 items/s  (0.736 items/s/worker)
n_workers=  8  throughput=3.80 items/s  (0.475 items/s/worker)
n_workers= 16  throughput=4.24 items/s  (0.265 items/s/worker)
n_workers= 28  throughput=3.56 items/s  (0.127 items/s/worker)
```

With **no file I/O and no shared read-only state to reload**, per-worker
throughput still collapses steadily as worker count rises (1.35 down to
0.13 items/s/worker -- a >10x drop in per-worker efficiency), and
aggregate throughput actually *falls* at 28 workers versus 16. This rules
out I/O, `DataIndex`/config reloading, and `ProcessPoolExecutor`
scheduling overhead as the scaling bottleneck (this benchmark has none of
those) and points squarely at memory bandwidth: each channel-day's
statistics touch several full-array passes over 69-137 MB buffers
(`astype` to float64, `np.dot`, `abs()` for MAD/spike, the run-length
boundary scan, the two `np.partition` calls), and running many such
processes concurrently means they all contend for the same DRAM
controller. `threads_per_worker=1` (ARCHITECTURE.md's oversubscription
protection) means this isn't a BLAS/thread-count misconfiguration either
-- it's inherent to the workload being memory-bound rather than
compute-bound, on this 28-core/94 GB machine.

**Practical implication for a production run**: throughput will keep
improving with more workers up to roughly 4-8 (this machine's apparent
bandwidth "knee"), but requesting many more workers than that will not
help, and past ~16 can actively hurt (the 28-worker regression above).
`cfg.parallel.n_workers: auto` (all logical cores) is *not* the optimal
setting for this workload on a memory-bandwidth-bound machine; a follow-up
could cap `n_workers` for the QC-stats step specifically, or expose a
per-step override (out of scope here -- flagged as a follow-up, no issue
filed per the "don't run bd write commands" worker rule).

## 6. `DataIndex` per-worker caching: evidence at production scale

The 8-station/2-day synthetic test index is tiny (parses in a couple of
ms), so the "cache `DataIndex` per worker process" fix doesn't show up in
the 48-channel-day test timings above. It matters at the scale the issue
itself describes (50 channels x 365 days ~= 18k channel-days). A synthetic
index of that shape (54,750 segment entries, matching real production
`inventory/index.json` structure, 8.7 MB on disk) was built and timed:

```
DataIndex.load(): ~43 ms per call (mean of 10 loads)
```

Without per-worker caching, that cost was paid **once per channel-day
item**, even within the same long-lived worker process. At 18k
channel-days, that is ~774 s (~13 minutes) of pure, redundant re-parsing
across a run -- versus, with caching, once per worker process (e.g. ~28
worker processes x 43 ms =~ 1.2 s total). This is a one-off fixed cost
that does not show up in a throughput-per-item measurement but directly
shortens wall-clock time for a real archive-scale run.

## 7. Correctness

- `rms`/`mean_abs` are still computed on the exact same float64 pooled
  sums as before (only the *order* of summation for `sum_sq` differs --
  `np.dot` vs. elementwise-square-then-sum); `tests/test_qc_stats.py
  ::test_rms_mean_abs_matches_manual_calc` (`rel=1e-12` against a direct
  NumPy calculation) still passes unchanged.
- `spike_count`/`flat_count`/`clip_count` are exact integer counts from
  the same float64 values and the same threshold definitions; unaffected
  by the `float32` kurtosis path.
- `kurtosis` moved to `float32` (see item 3 above). Every consumer in the
  test suite already used a wide-margin inequality (`> 50`, `> 10`,
  `< 5`), never an exact/tight comparison, so this is safe.
  `tests/test_qc_stats.py::test_fast_kurtosis_matches_scipy_reference`
  documents and checks the tolerance directly: `rel=1e-3` against
  `scipy.stats.kurtosis` (float64), on both a plain-noise and a
  heavy-tailed sample.
- A dedicated regression test,
  `tests/test_qc_stats.py::test_optimised_output_matches_reference_implementation`,
  compares the optimised `compute_channel_day_stats` against a
  from-scratch, deliberately naive reference implementation written
  directly in the test file (plain `scipy.stats.kurtosis`, non-`
  overwrite_input` `np.median`, `np.sum(x**2)`, textbook per-sample
  run-length encoding) on one sample channel-day engineered to contain
  spikes, a flat run, and a clipped run simultaneously. Documented
  tolerances: `rel=1e-9` for `rms`/`mean_abs`, exact match for the three
  integer counts, `rel=1e-3` for `kurtosis`.
- All 14 pre-existing `tests/test_qc_stats.py` tests pass unchanged.

## 8. Test-suite speed

`tests/test_qc_stats.py`'s own `stats_world` fixture already computes
stats for every channel/day the test module needs (not more), so there
was no further "compute only the channels asserted on" trimming available
without changing what several tests assert against (the *_row helper
already selects individual rows out of one shared computation, mirroring
what M1.3's `availability.py` tests do). The serial throughput improvement
above (~1.7-1.8x) is what sped up the shared fixture itself; the module's
own per-test overhead (index build, CSV round-trip) was already small
relative to the 48-channel-day computation.

## Measurement caveats

This machine was shared with other concurrent agent sessions throughout
this benchmarking (`uptime` load average 44-55 against 28 logical cores),
so absolute wall-clock numbers include some contention noise from
unrelated processes. Where that could bias a *comparison* (section 3), an
interleaved measurement was used specifically to cancel out slow drift in
host load between the "before" and "after" halves. Section 4's before/after
pair was measured as two separate full runs (not interleaved, since a
process-pool benchmark can't easily interleave with another), taken
minutes apart; the *shape* of the scaling curve (early gains, plateau by
8, no further gain by 16) is the same in both, which is the qualitative
conclusion this document relies on, rather than the exact throughput
figures. Section 5's isolated benchmark was chosen specifically to give a
load-independent, structural explanation (per-worker efficiency collapsing
monotonically, and regressing at 28 workers) rather than relying on
absolute numbers at all.

## 9. Isolated-impulse spike metric (OpenMEQ-5nl.9): benchmark and cost

M1.9 replaced `spike_count`'s definition (global-MAD threshold on raw
samples -> isolated-impulse residual against a short running median --
`openmeq/qc/stats.py`'s module docstring and `docs/ARCHITECTURE.md` §3 have
the full rationale/definition) because the old metric could not tell a
glitch from a real seismic event on this project's own synthetic fixture:
every channel's old `spike_count` on a day with real events was already
1300-1850 (real quake pulses reach hundreds to low thousands of counts
against a background noise half-range of only 5 counts), swamping the
deliberately-spiky channel's signature (only ~600 above its own baseline, a
~1.15x network-outlier ratio). This section re-benchmarks §3's machine
(28 logical cores, shared with other concurrent work -- same caveats apply)
against the new metric, since a running-median filter is fundamentally more
expensive than a single global order-statistic.

### 9.1 Alternatives benchmarked for the running median

A full 17.28M-sample synthetic channel-day, `k=7` unless noted (isolated,
single-item timings, best of 3-7 repeats):

| Approach | Time |
|---|---:|
| `scipy.ndimage.median_filter`, float64, `mode="nearest"` | ~328 ms |
| `scipy.ndimage.median_filter`, float32 | ~311 ms |
| `scipy.ndimage.median_filter`, `k=5` (float32) | ~289 ms |
| `scipy.ndimage.rank_filter` (`rank=3`, equivalent to median-of-7) | ~324 ms |
| Hand-rolled median-of-7 sorting network (shifted-view compare/swap) | ~1053 ms |
| Strided-window `np.partition` (`sliding_window_view` + partition), 2M-sample subset, extrapolated to 17.28M | ~812 ms |

`scipy.ndimage.median_filter` won clearly -- both hand-rolled alternatives
lost because they materialise several full-size shifted copies/windows
(more passes over memory than the rank-order filter's single pass).
`float32` and `k=5` each save ~5-12%, not enough on their own to change the
choice of algorithm; the shipped default keeps `k=7` (per the issue's own
`k~5-7` guidance, and 7 gives a slightly wider isolation margin against
2-3-sample-wide glitches) and float64 (matching every other statistic in
this module, avoiding a second, `float32`-only code path for a single-digit
percent gain).

### 9.2 Candidate-window / subsampled-MAD optimisation: considered and rejected

The issue suggested computing the residual only in candidate windows where
`|x - median|` is already large, to avoid running the filter over the
*entire* day. This was seriously considered: a coarse global-MAD prescreen
could cheaply narrow the array down to a small candidate set, and the
residual/local-median computation is genuinely only needed at those
candidates -- but `MAD(r)`, the classifier's own scale estimate, needs a
*representative* population of "typical" residuals, which the candidate
set (by construction, already-extreme deviations) cannot supply. The only
way to get a global `MAD(r)` estimate cheaply is to compute it from a
*subsample* of ordinary points instead of the full array -- which makes
`spike_count` a function of a random subsample, not a deterministic
function of the input. That fails this project's own "identical, or within
documented float tolerance" correctness bar (`docs/performance.md` §3's
same rationale for rejecting a sampled median there) and would make the
optimised/reference regression test
(`tests/test_qc_stats.py::test_optimised_output_matches_reference_implementation`)
inherently unable to assert exact equality. Rejected in favour of running
`median_filter` over the full array, which is still fast enough in
absolute terms (below) without approximation.

### 9.3 Per-channel-day cost: before vs. after (interleaved, real data)

Same interleaved-single-item methodology as §3 (both implementations
imported side by side in the same process, timed alternately), on one
ordinary full 200 Hz synthetic channel-day (`XX.STA01.00.HHZ`,
2025-01-01, no gap), 7 repeats:

```
OLD (5nl.8, global-MAD spike_count):        mean=548.9ms  median=541.5ms  min=488.7ms
NEW (5nl.9, isolated-impulse spike_count):   mean=830.5ms  median=828.2ms  min=757.0ms
slowdown (mean):   1.51x
slowdown (median): 1.53x
```

**This is an expected, documented regression, not an oversight**: a
running-median filter over 17.28M samples is inherently more expensive
than a single global order-statistic pass, and §9.2 explains why the
cheaper candidate-window alternative was rejected on correctness grounds.
The new metric still stays within the same order of magnitude as 5nl.8's
own *pre-optimisation* baseline (§3: ~929 ms/item before 5nl.8, ~831 ms/item
here) -- i.e. this real correctness fix "costs back" roughly what 5nl.8
saved on the spike/MAD portion specifically, while every *other* 5nl.8
optimisation (run-length encoding, `np.dot`, per-worker caching, Stream.split
skip, batching) is unaffected and still fully in effect.

### 9.4 Test-suite throughput: before vs. after (real `synthetic_project` data)

Full `tests/test_qc_stats.py` + `tests/test_qc_monthly_flags.py`,
`n_workers=1` (both modules' own config default):

```
before (5nl.8, two independent module-scoped fixtures, each computing all
        48 channel-days): ~39s + ~39s of daily-stats computation alone
        (docs/performance.md §8's own note that each module fixture
        recomputes independently), full pair ~140s combined with the rest
        of the repo's suite per the M1.9 issue's own numbers.
after  (5nl.9, shared session-scoped `synthetic_qc_tables` fixture in
        tests/conftest.py, isolated-impulse spike metric):
        48 channel-days in 39.90s (1.2 channel-days/s) -- computed ONCE,
        not twice; both modules' 38 tests combined: 47.89s total wall time.
```

The daily-stats computation itself is slower per item (§9.3), but
computing it once instead of twice more than offsets that, so the *two
modules' combined* runtime dropped even though the underlying computation
got more expensive per channel-day.

### 9.5 Parallel scaling (real data, `synthetic_project`, 48 channel-days x N repeats where noted)

```
n_workers=1: 48 channel-days in ~41s   (~1.2 channel-days/s)
n_workers=4: 48 channel-days in 14.62s (3.28 channel-days/s, 2.7x)
n_workers=8: 48 channel-days in 11.11s (4.32 channel-days/s, 3.6x)
```

Scaling to 4-8 workers looks comparatively better than §4's post-5nl.8
table (which plateaued at ~1.9-2.0x by 8 workers) -- consistent with §5's
diagnosis that the old bottleneck was memory bandwidth, not compute: the
new metric shifts more of the per-item cost onto `scipy.ndimage.median_filter`
(a compute-bound rank-order filter, not a bandwidth-bound reduction), which
parallelises comparatively better across independent worker processes on
this 28-core machine. This is a small, single-machine data point (48
channel-days, some measurement noise per the shared-machine caveats below),
not a re-run of §4's full 1/4/8/16-worker repeated-channel-day study; a
follow-up could redo that fuller study for the new metric specifically
(flagged, no issue filed per the "don't run bd write commands" worker
rule).

### 9.6 Verified synthetic_project flagging numbers (default config)

Computed directly against `tests/data/make_synthetic.py`'s fixture with the
shipped `config/example.yaml` defaults (`spike_threshold_mad=10`,
`spike_median_window=7`, `max_spike_fraction=0.000012`,
`max_flat_run_s=60`, `max_flat_fraction=0.5`, `min_availability_pct=80`,
`monthly_outlier_ratio=3.0`):

- **Spikes**: `XX.STA06.00.HHE` 2025-01-02 (the known spiky channel/day)
  measures `spike_count=388`, `npts=17280000` (fraction ~2.245e-5) --
  crosses `max_spike_fraction`. Its own 2025-01-01 baseline (no anomaly)
  measures 80. The next-highest *ordinary* channel/day network-wide that
  same day (`XX.STA04.00.HHZ`) measures 119 (fraction ~6.89e-6) -- well
  below the threshold. The monthly cross-station `spike_count` ratio for
  the spiky channel is 2.66x the network median, below
  `monthly_outlier_ratio`'s default (3.0), so it flags via `"spikes"`
  directly, not `"network_outlier"`.
- **Flat**: `XX.STA03.00.HHN` 2025-01-01 (the known flat-lined channel/day)
  measures `longest_flat_run_s≈3600` (the whole 1-hour injected window) --
  crosses `max_flat_run_s` (60s) by a wide margin, while its
  `flat_count/npts` (~4.17%) still does *not* cross `max_flat_fraction`
  (50%) -- confirming the new duration criterion, not a retuned fraction,
  is what makes this flag. Every ordinary channel/day's `longest_flat_run_s`
  is under 1 second (background noise essentially never repeats an exact
  raw sample value that many times in a row).
- **Gap**: `XX.STA05.00.HHZ` 2025-01-01 (10-minute gap) measures
  `coverage_pct≈99.3%`, above `min_availability_pct`'s default (80%) -- not
  flagged, per the documented gap policy (`docs/ARCHITECTURE.md` §4.2).
- **Zero false positives**: running `compute_channel_flags` against every
  one of the 48 real synthetic channel-days with these defaults flags
  *exactly* the two channel-days above (`["flat"]` and `["spikes"]`
  respectively) and nothing else -- verified directly (also asserted in
  `tests/test_qc_monthly_flags.py::test_e2e_zero_false_positives_on_ordinary_channel_days`).

# M11.3: pipeline-wide profiling, parallel scaling, and further optimisation (`OpenMEQ-m1f.3`)

Scope: the *whole* pipeline (M11.2 already proved it correct end to end),
profiled and optimised on this project's own dev machine (28 logical / 20
physical cores -- a hybrid performance/efficiency-core CPU, see §5 below --
94 GB RAM). Measured baseline (`docs/validation.md`, `n_workers=1` on the
tiny 8-station/2-day `synthetic_project` fixture): `openmeq run` took
**108 s** total -- qc 41 s, detect 42 s, chunk 1 s, pick 10 s, locate 5 s,
magnitude 3 s, relocate 5 s.

Real archives are ~50 channels x 365 days (~18k channel-days), so
per-channel-day and per-day costs matter far more than this tiny fixture's
absolute numbers -- `docs/performance.md`'s existing §§1-9 (`OpenMEQ-5nl.8`/
`OpenMEQ-5nl.9`) already did this profiling pass for `qc`; this section
covers `detect` (not previously profiled), re-verifies the machine-level
concerns flagged in earlier notes (thread oversubscription, `auto`
resolving to logical vs. physical cores, `DataIndex.read`'s decode
behaviour), and adds a parallel-scaling measurement at a larger,
purpose-generated dataset.

## 1. `detect_window` profiling and optimisation

cProfile against one real `detect_window` call (25 traces, one full
200 Hz day, 8-station `synthetic_project` fixture), before any change:

| Function | Cumulative time | % of total |
|---|---:|---:|
| **Total** (`detect_window`) | 14.43 s | 100% |
| `stream.detrend("linear")` (-> `scipy.signal.detrend` -> `scipy.linalg.lstsq`) | 7.00 s | 48.5% |
| `coincidence_trigger` (incl. `recursive_sta_lta`/`trigger_onset`) | 2.86 s | 19.8% |
| `stream.filter("bandpass")` (`scipy.signal.sosfilt`) | 1.99 s | 13.8% |
| `DataIndex.read` (miniSEED I/O) | 1.34 s | 9.3% |
| everything else (astype casts, gap-split, etc.) | ~1.24 s | 8.6% |

**`stream.detrend("linear")` is the single largest cost, at essentially
half the function's total time**, and it is calling a generic solver where
this call site never needs the generality it pays for.

### 1.1 Root cause

ObsPy's `Trace.detrend("linear")` dispatches to
`scipy.signal.detrend(data, type="linear")`, which -- even for the
single-segment (no breakpoints) case this call site always uses -- solves
the least-squares fit via `scipy.linalg.lstsq` (an SVD-based solver
general enough for an arbitrary-column design matrix and multiple
breakpoints). Neither generality is used here: it is always exactly one
straight-line OLS fit against the sample index.

### 1.2 Fix: closed-form OLS detrend (`_fast_linear_detrend`)

`openmeq/detect/worker.py`'s new `_fast_linear_detrend` replaces the call
with a closed-form solve: a mean-centred index (`sum(x) == 0` by
construction) collapses the normal equations to two `float64`-accumulated
reductions (`slope = sum(x*y) / sum(x**2)`, `intercept = mean(y)`) -- no
matrix factorization. Least-squares regression against a single predictor
is invariant to affine reparametrization of that predictor, so this fits
the **mathematically identical line** `scipy.signal.detrend` does; the two
differ only by floating-point accumulation rounding.

**Accuracy** (`tests/test_detect_fast_detrend.py`, synthetic arrays up to
200,001 samples and a real synthetic-project channel-day, `float32`/
`float64`): max relative difference against the `scipy.signal.detrend`
reference stayed under `1e-4` of the input's own amplitude scale in every
case tested -- a downstream-irrelevant difference against a
bandpass/STA-LTA pipeline that already discards DC/near-DC content.

**Speed** (isolated micro-benchmark, one 17.28M-sample channel):

| dtype | `scipy.signal.detrend` | `_fast_linear_detrend` | speedup |
|---|---:|---:|---:|
| float32 | ~299 ms | ~139 ms | ~2.1x |
| float64 | ~486 ms | ~126 ms | ~3.9x |

**In situ** (same `detect_window` call as above, all 25 traces, float32):
`stream.detrend("linear")`'s share dropped from 7.00 s to 3.60 s (~1.9x),
and **`detect_window`'s total wall time dropped from 14.43 s to 11.58 s
(~20% faster)**, with `coincidence_trigger` (3.21 s) now the largest single
component. No change to detection results: `tests/test_detect_worker.py`'s
full existing suite (event recovery, flagged-channel exclusion, gap
handling, day-boundary events) passes unchanged after the swap.

### 1.3 Second fix: per-worker-process `DataIndex`/`Config` caching

Profiling this module also surfaced a gap versus the pattern
`openmeq.qc.stats`/`openmeq.chunk` already established (`OpenMEQ-5nl.8`):
`detect_window` called `DataIndex.load(index_path)` and
`load_config(cfg_path)` **on every single window**, even though a
long-lived `ProcessPoolExecutor` worker handles many windows over its
lifetime, all sharing the same `index_path`/`cfg_path`. This doesn't show
up in a single-window profile (it's a fixed cost, not proportional to data
volume), but at production scale (e.g. one window per day over a
multi-month survey, `cfg.parallel.chunk_days=1`) it is a real, avoidable
per-run cost: `openmeq.qc.stats`'s own measurement of `DataIndex.load()`
against a realistic 50x365 index was ~43 ms/call, so re-loading it once per
day over a year is ~15.7 s of pure redundant re-parsing that per-worker
caching eliminates entirely (paid once per worker process instead).
Mirrored `qc`/`chunk`'s exact `_cached_index`/`_cached_*` pattern into
`openmeq/detect/worker.py`; verified with a call-counting monkeypatch test
(`tests/test_detect_fast_detrend.py::test_detect_window_caches_index_and_config_per_worker_process`)
that the index/config are loaded exactly once across multiple windows.

## 2. `qc` daily-stats: confirmatory profiling, no further change

`openmeq/qc/stats.py` already went through a dedicated optimisation pass
(`OpenMEQ-5nl.8`/`OpenMEQ-5nl.9`, this document's own §§1-9 above). Per
this task's "profile first" design note, it was re-profiled rather than
assumed unchanged:

```
_fast_kurtosis                    ~3-7%
partition (median/MAD)            ~9%
_segment_run_counts (run-length)  ~9%
rank_filter (isolated-impulse median filter, k=7)   ~32-38%
```

This matches §9's own documented finding exactly (the isolated-impulse
spike metric's `scipy.ndimage.median_filter` call is now the dominant
cost, already investigated and its cheaper "candidate-window" alternative
already rejected there on correctness grounds -- §9.2). No new hot spot
appeared and no further optimisation was applied here: the already-shipped
5nl.8/5nl.9 work covers this step, and re-deriving an already-answered
question is out of this task's scope.

## 3. Chunk memory / `DataIndex.read` (M3.2 follow-up): resolved, no bug

A note on this issue asked whether the ~830 MB RSS still observed while
chunking a sparse day (M3.2, after its own sub-window-splitting fix) meant
`DataIndex.read` decodes a whole day file even for a short window read.
Measured directly: reading a 60 s window out of a full 17.28M-sample/day
miniSEED file costs **~3.4 ms / ~1.1 MB** of Python-visible allocation,
versus **~83 ms / ~70 MB** for the whole file -- proportional to the
*window*, not the file. ObsPy's native miniSEED selective decode (passing
`starttime`/`endtime` straight through, exactly as `openmeq/index.py`'s
own docstring already documented) does not decode whole-day files. No bug;
no code change needed.

The 830 MB figure is explained instead by `resource.getrusage(...)
.ru_maxrss` being a **whole-process, monotonically non-decreasing**
high-water mark (`scripts/benchmark.py`'s own docstring already flags this
caveat): by the time `chunk` runs, `qc` and `detect` have already touched
several GB in the same process (especially at `n_workers=1`, where `pmap`
runs every window in-process rather than in a subprocess), and that
earlier peak is still counted against `chunk`'s number even though `chunk`
itself never allocated it. A direct measurement with `scripts/benchmark.py`
(now that it can be run with `psutil` installed for a stage-*local*
sampled peak, not just `ru_maxrss`) on the small fixture confirmed this:

| stage | `ru_maxrss` (whole-process, cumulative) | `psutil`-sampled (stage-local) |
|---|---:|---:|
| qc | 1,232 MB | 1,199 MB |
| detect | 4,948 MB | 5,068 MB |
| chunk | 4,948 MB (unchanged -- never exceeded detect's peak) | **293 MB** |

`chunk`'s own actual contribution (293 MB) is far below the 830 MB
figure; the apparent "830 MB in chunk" was really `detect`'s own peak
(driven by holding a full day's worth of 25 float32/float64 traces through
detrend/filter/STA-LTA in one process at `n_workers=1`), attributed to the
next stage that happened to run afterward in the same process. Documented
here and in `openmeq/chunk.py`'s own module docstring for future readers.

## 4. BLAS/OMP thread-oversubscription: checked, confirmed not an issue

M0.5 flagged (and this task's design notes repeated) the risk of each
`pmap` worker spawning `threads_per_worker` BLAS threads *on top of*
`n_workers` processes -- verified here under actual `pmap` execution (not
just `openmeq.parallel`'s own unit test), using `threadpoolctl` inside a
real worker process at `n_workers=4, threads_per_worker=2`:

```json
{"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
 "threadpool_info": [{"internal_api": "openblas", "num_threads": 2, ...}]}
```

OpenBLAS's own runtime thread count (`threadpool_info`, not just the env
var) matches `threads_per_worker` exactly in every one of 4 worker
processes. No oversubscription found; `openmeq.parallel`'s `spawn`-based
thread-env-var mechanism (module docstring) works as designed.

## 5. `n_workers: "auto"`: physical cores, not logical (bug fix)

`docs/ARCHITECTURE.md` and `config/example.yaml` already documented
`n_workers: auto` as resolving to **physical** cores (`"auto = all
physical cores"`) -- but `openmeq/parallel.py`'s actual implementation
resolved it to `len(os.sched_getaffinity(0))`, the **logical** (hyperthread)
count, contradicting its own project's already-published contract (M0.5's
own note flagged exactly this as an open question for this task).

This dev machine makes the distinction concrete: a hybrid
performance/efficiency-core CPU (Intel Core i7-14700F) with **28 logical /
20 physical** cores -- 8 performance cores each exposing 2 hyperthread
siblings, plus 12 efficiency cores exposing 1 each. A naive `logical // 2`
would get this wrong (14, not 20); `openmeq.parallel._physical_core_count`
instead counts unique `(physical id, core id)` pairs from `/proc/cpuinfo`,
restricted to whatever `os.sched_getaffinity(0)` allows (so a cgroup/CPU
quota restriction, e.g. inside a container, is still respected). Falls
back to the logical count if `/proc/cpuinfo` is unavailable/unparseable
(non-Linux, or any other platform where this can't be determined) --
`_resolve_n_workers` never raises or returns 0 because of this.

This also lines up with this document's original §5 ("Documented
bottleneck: memory bandwidth, not I/O or scheduling") own earlier finding
(that QC-stats plateaus by ~4-8 workers and *regresses* by 28 on this
machine, i.e. its old bottleneck is memory bandwidth, not core count) --
requesting every
logical core by default was actively counter-productive for at least one
profiled step, and physical cores is a better single default across the
whole pipeline without this module needing to know which step is calling
it. A user who has measured their own workload/machine can still always
set `n_workers` to an explicit integer.

New tests: `tests/test_parallel_physical_cores.py` (synthetic
`/proc/cpuinfo` content covering the hybrid-topology and
affinity-restricted cases, plus the missing/malformed-file fallback).

## 6. Larger benchmark dataset + parallel scaling (1/4/8/16/28 workers)

`tests/data/make_synthetic.generate` gained `n_stations`/`n_days`
parameters (M11.3), reusing its existing generator code rather than
building a second one (`generate()`'s own docstring has the full contract;
left at their default `None`, output is byte-identical to every pre-M11.3
call -- verified directly, diffing every generated file). `scripts/
benchmark.py` gained matching `--n-stations`/`--n-days`/`--tmp-dir` flags.

Generated a 20-station x 7-day dataset (60 channels x 7 days = 420
channel-days, 70 events, ~4.4 GB) into the scratchpad (deleted after this
measurement -- disk on this machine has ~10-15 GB free). Machine load
checked with `uptime` before running (`load average: ~3-8` against 28
logical cores -- not perfectly idle, some background activity, but well
under saturation; noted as a measurement caveat below, matching this
document's own §"Measurement caveats" precedent).

Both `qc` and `detect` re-ran from a clean output directory at each
`n_workers` (forced, not skip-cached), on the same generated data (qc's
channel flags -- and therefore which channels `detect` reads -- are
deterministic given the same data/config, independent of `n_workers`, so
these five runs are a fair comparison of the exact same work at each
worker count):

**`qc` (420 channel-days):**

| n_workers | wall time | channel-days/s | speedup |
|---:|---:|---:|---:|
| 1 | 306.04 s | 1.37 | 1.00x |
| 4 | 123.02 s | 3.41 | 2.49x |
| 8 | 102.37 s | 4.10 | 2.99x |
| 16 | 100.57 s | 4.18 | 3.04x |
| 28 | 103.83 s | 4.05 | 2.95x |

Plateaus by 8 workers and *regresses slightly* by 28 -- reproducing, at
this larger 420-channel-day scale, this document's original §5
("Documented bottleneck: memory bandwidth...") finding
(memory-bandwidth-bound, not I/O/scheduling-bound) almost exactly (the
original §4's post-optimisation table showed the same 8-worker plateau and
16+-worker regression shape on the tiny 48-channel-day fixture). This is
direct supporting evidence for the "`n_workers: auto` -> physical-cores"
fix above (§5 of *this* M11.3 section): on this machine, physical cores
(20) sits right at the plateau, not past it -- a better default than
logical cores (28), which measurably regresses.

**`detect` (7 windows/days, one task per day at `chunk_days=1`):**

| n_workers | wall time | speedup |
|---:|---:|---:|
| 1 | 187.09 s | 1.00x |
| 4 | 142.96 s | 1.31x |
| 8 | 98.94 s | 1.89x |
| 16 | 89.99 s | 2.08x |
| 28 | 87.07 s | 2.15x |

`detect`'s scaling is capped well below `qc`'s not because of a bug, but
by its own unit of parallelism: PLAN.md §4 documents `detect` as "day (all
stations)" -- one `pmap` task per day. With only 7 days in this benchmark
dataset, no more than 7 tasks can ever run concurrently, and the
per-window costs are themselves wildly uneven (event-dependent: a
quiet day finishes in ~4-13 s, the busiest in this run took ~85-137 s
*regardless of `n_workers`*, since one window is never split across
workers). Wall time above `n_workers=8` is essentially bounded by that one
slowest single window, not by available parallelism -- a classic
load-imbalance-with-few-tasks ceiling, not a scaling defect. At production
scale (e.g. a full year, 365 potential day-tasks with `chunk_days=1`),
this ceiling is far less binding; this benchmark's 7-day size (chosen to
keep dataset generation/runtime within this task's budget) is simply too
short to demonstrate `detect`'s scaling past ~8 workers on its own. This
is exactly what `cfg.parallel.chunk_days`/finer day-splitting would need
to address for a short survey with very uneven per-day event rates -- noted
as a follow-up, not implemented here (out of scope: no profiling evidence
that *this* project's own realistic archives are that short/bursty).

**Machine load during this sweep** (`uptime`, checked before starting and
at the start of each configuration): load average ranged ~3-10 against 28
logical cores -- some background activity present (this is a shared dev
machine), but well under saturation throughout; re-measuring on a
perfectly idle machine might shift absolute numbers slightly but is
unlikely to change the qualitative shape (plateau/regression for `qc`,
task-count-capped scaling for `detect`) given how closely it reproduces
this document's original §4/§5, separately measured months earlier on a
different (tiny) dataset.

## 7. Recommendations

1. **`n_workers: auto` now resolves to physical cores** (§5) -- matches the
   already-documented contract; no user action needed, but anyone who had
   set `auto` on a hyperthreaded machine expecting the *logical* count
   should pin an explicit integer if they specifically want that.
2. For memory-bandwidth-bound steps (QC stats, per this document's original
   §5) on a machine where physical cores still oversubscribes
   available memory bandwidth, pin `n_workers` explicitly lower (e.g. 4-8)
   in that project's config -- a full per-step override is a larger
   `ParallelConfig` contract change, flagged as a follow-up, not applied
   here (single shared `cfg.parallel` block, out of this task's scope to
   restructure).
3. `detect`'s linear detrend and per-worker caching optimisations (§1) are
   safe to rely on for production-scale runs; no output change.
4. `DataIndex.read`/chunk memory: no action needed (§3) -- already correct.
5. Follow-up (not filed as a bd issue per the "don't run bd write commands"
   worker rule): a true per-step `n_workers`/`threads_per_worker` override
   in `ParallelConfig`, if a future profiling pass finds another step with
   a meaningfully different optimal worker count than the pipeline-wide
   default.

## Test-suite runtime (OpenMEQ-m1f.6)

A clean run on an idle machine (28 logical / 20 physical cores) after the shared `full_pipeline_run` session fixture:

| Suite | Before | After |
|---|---|---|
| `pytest -q -m "not slow"` | 7 min 44 s | 2 min 21 s |
| `pytest -q` (full, 660 tests) | ~15–17 min | 13 min 20 s |

The full suite stays above the 6-minute target because several tests legitimately need their own real pipeline runs: the skip-unchanged, `--force` and `--from`/`--to` checks in `tests/test_run_all.py`, the benchmark-script tests, and the pick and locate recovery reruns. Use `-m "not slow"` for day-to-day development, and run the full suite before merging.
