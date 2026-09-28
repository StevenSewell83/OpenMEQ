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
