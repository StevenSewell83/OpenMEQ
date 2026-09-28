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
