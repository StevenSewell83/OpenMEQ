# hypoddpy Template Audit

**Plan key:** M9.1. This document is the single source of truth for M9.2–M9.5 and
M10.1–M10.5: what to port (rewritten, never copied), what to fix, and what to drop
from the local hypoddpy template. Every claim below was independently re-verified
against the bundled `external/HYPODD` source/docs during this audit (not copied
from the M9 epic design note, though that note's own claims are cross-checked and
corrected where wrong — see §6).

## Sources read

- Template (primary): `/home/steve/seismic/hypoDDpy-master/hypoddpy/hypodd_relocator_parallel.py`
  (1562 lines, joblib + `xcorr_pick_correction`, Python 2 syntax throughout —
  `except Exception, err:`, `.iteritems()`, `print` statements, `basestring` — it
  cannot run under Python 3.10 as-is).
- Template (secondary): `/home/steve/seismic/hypoddpy/hypodd_relocator.py` (1469
  lines, same lineage, non-parallel, older). `*_bup.py`/`*_broken.py` variants were
  ignored per the issue's instructions.
- hypoDD 2.1 ground truth: `external/HYPODD/include/hypoDD.inc`, `vel3d.inc`,
  `ph2dt.inc`; `src/hypoDD/getdata.f`, `hypoDD.f`, `getinp2.f`, `get_vel3d.f`;
  `src/ph2dt/ph2dt.f`; `doc/hypoDD2_UserGuide.pdf` (extracted with
  `pdftotext -layout` to `/tmp/.../scratchpad/OpenMEQ-6ld.1/userguide.txt`,
  1366 lines); `examples/example1/*` (every file).

All file:line citations below are against these files at their paths above.

---

## 1. File writers (station/phase/event/ph2dt.inp/hypoDD.inp)

### 1.1 `_write_station_input_file` (parallel: 327–346; non-parallel: 310–329)

Writes one line per station: `"%s %.6f %.6f %i" % (label, lat, lon, elevation)`.

- **Format order correct**: matches `getdata.f:271–274`'s free-format
  `read(line,*) sta_lab(i), sta_lat(i), sta_lon(i), sta_elv(i)` (4-field case).
- **Elevation unit correct**: written in metres; `getdata.f:274` divides by 1000
  internally (`sta_elv(i) = sta_elv(i)/1000`) — matches ARCHITECTURE.md §1 units
  convention (station elevation in metres in OpenMEQ metadata).
- **Confirmed defect — no negative-elevation/IMOD handling at all.** The writer
  does a bare `%i` of `value["elevation"]` (parallel:341–342; non-parallel:
  325) with no check of sign and no relationship to which `IMOD` is selected.
  `getdata.f:275–278` (and the `IMOD=4` branch at `getdata.f:286–288`) silently
  zeroes any negative elevation unless `IMOD=5`:
  ```fortran
  if(imod.ne.5.and.sta_elv(i).lt.0.and.sta_elv(i).ne.-999) then
     sta_elv(i)=0.0
     ierr=ierr+1
  endif
  ```
  Since OpenMEQ's effective station elevation is `elevation_m - depth_m`
  (ARCHITECTURE.md §6) and boreholes have `depth_m > 0`, every borehole station
  with a positive depth will silently get zero relative elevation in the
  relocation unless `IMOD=5` is selected. The template is completely silent on
  this — it neither warns nor selects `IMOD=5`. **Confirmed, not previously
  handled anywhere in either template.**
- **Station label length**: the template's `station_id` keys are
  `"%s.%s" % (network_code, station_call_letters)` (parallel:307–308,
  316–317), i.e. `NET.STA`, written unmodified into station.dat. `getdata.f:51`
  declares `character sta_lab(MAXSTA)*7` — 7 characters max. A `NET.STA` label
  routinely exceeds 7 chars (e.g. `NZ.WIZ` = 6, but `XX.STA01` = 8). The
  template does **not** truncate, map, or even validate label length — Fortran
  free-format character reads silently truncate to the declared width, so
  overlong labels would collide or read as garbage. **Confirmed additional
  defect**, consistent with the epic note's requirement (see §6) that M9.2 must
  build a persisted label-mapping.

**Verdict: port-rewritten.** OpenMEQ's writer must (a) implement a
NET.STA→≤7-char label map with collision detection, and (b) decide and
document IMOD=5-vs-clamp for negative effective elevations (ARCHITECTURE.md
§6 already states the convention; the writer must act on it, not silently
replicate hypoDD's own zeroing). → **M9.2 must not reproduce this silence.**

### 1.2 `_write_catalog_input_file` (parallel: 348–433; non-parallel: 331–415) — writes `phase.dat`

Per-event header:
```python
"# {year} {month} {day} {hour} {minute} {second:.6f} {latitude:.6f} " \
"{longitude:.6f} {depth:.4f} {magnitude:.6f} {horizontal_error:.6f} " \
"{depth_error:.6f} {travel_time_residual:.6f} {event_id}"
```
(parallel:371–374). Depth divided by 1000 (m→km, line 386) — correct, since
QuakeML `Origin.depth` is metres.

- **Format order correct** against `ph2dt.f`'s documented header
  `# YR MO DY HR MN SEC LAT LON DEPTH_KM MAG HERR VERR RES CUSPID`; the
  template's field order matches exactly.
- Phase line: `"{station_id} {travel_time:.6f} {weight:.2f} {phase}"`
  (line 403), `travel_time = pick_time - origin_time` (line 404) — matches
  `ph2dt.f:483–484`'s documented `STA TRAVEL_TIME WGHT PHA` with travel time as
  a relative (not absolute) time.
- **Negative travel times are dropped with only a log warning** (parallel:
  405–415) rather than being written with `-1` or otherwise flagged. This is a
  defensive check against clock/ordering errors upstream, not a hypoDD format
  violation — acceptable behaviour, but OpenMEQ's writer should promote this to
  a hard, catalog-checkable rejection rather than a print-and-skip, since a
  discarded phase here is invisible to any later diagnostic.
- **Weight handling has a design gap** (parallel: 89, 416–422): the default
  `phase_weighting` is `lambda ...: 1.0` — every pick without an explicit
  `"weight"` key gets weight `1.0`. `ph2dt.f:485` only filters a phase when
  `weight >= 0 and weight < minwght`; a negative weight (hypoDD's own "no
  quality info" sentinel, confirmed via `examples/example1/El16.pha`'s
  ubiquitous `-1.000` column and the doc's own description) always survives the
  filter regardless of `minwght`. The template has no concept of this sentinel
  at all — it always writes an affirmative numeric weight. In practice a
  hardcoded `1.0` is functionally almost equivalent to `-1` (it always passes
  any `minwght <= 1`), but it is **not the same semantic** (a real quality-1.0
  pick and an unweighted pick become indistinguishable downstream, e.g. in any
  future weighted statistics). **Confirmed gap, not a functional bug for the
  default config, but must not be ported as-is**: M9.2 should derive the weight
  from QuakeML `Arrival.time_weight` clamped to `[0, 1]`, defaulting genuinely
  unweighted picks to `-1.0`, exactly as the epic note already directs.

**Verdict: port-rewritten** (correct format skeleton, but rebuild weight
derivation from `openmeq.catalog`'s QuakeML `Arrival.time_weight`, not the
template's constant-1.0/ad-hoc-comment-based weight).

### 1.3 `_write_ph2dt_inp_file` (parallel: 563–649; non-parallel: 538–621)

Writes `station.dat`, `phase.dat`, then one numeric line
`MINWGHT MAXDIST MAXSEP MAXNGH MINLNK MINOBS MAXOBS` — 7 fields, matching
`ph2dt.f`'s free-format read and the epic note's confirmed order.

- `MAXDIST` is **auto-computed** from the bounding box of all
  event/station lat/lon/depth (parallel:571–595) so that no event-station
  pair is ever excluded by distance — a reasonable default-safe choice, worth
  keeping as an option (though it scales badly and should be user-overridable,
  which the template already supports via `set_forced_configuration_value`).
- `MAXSEP` is **auto-computed** as the 10th percentile of the full O(N²)
  inter-event distance matrix (parallel: 609–627) — correct in intent but
  `O(n_events^2)` in memory/time (`distances` list holds every pair including
  self-pairs); acceptable for hypoDD-scale problems (thousands of events) but
  should be replaced with a vectorised (numpy) computation in the rewrite
  rather than a nested Python loop.
- Remaining defaults (`MINWGHT=0.0, MAXNGH=10, MINLNK=8, MINOBS=8, MAXOBS=50`,
  parallel: 604–608) are reasonable but **not** the `MAXOBS=20`/`MAXOBS=100`
  documented default the ph2dt.inp file itself brackets, nor example1's
  actual `MAXOBS=100` (see `examples/example1/ph2dt.inp`, confirmed:
  `0 120 10 10 8 1 100`). The epic note directs M9.2 to use `ph2dt.inp`'s
  bracketed-comment defaults, not the template's ad-hoc values — confirmed
  correct guidance, template disagrees with it in `MINOBS` (8 vs. 1) and
  `MAXOBS` (50 vs. 20/100).

**Verdict: port-rewritten.** Field order and free-format layout are right;
defaults must come from `ph2dt.inp`'s own bracketed comments /
`examples/example1`, not the template's constants, and the O(n²) MAXSEP
computation should be vectorised.

### 1.4 `_write_hypoDD_inp_file` (parallel: 999–1070; non-parallel: 1181–1246)

Section skeleton (parallel: 1008–1025):
```
hypoDD_2
dt.cc
dt.ct
event.sel
station.sel
""
""
hypoDD.sta
hypoDD.res
hypoDD.src
{IDAT} {IPHA} {DIST}
{OBSCC} {OBSCT} {MINDS} {MAXDS} {MAXGAP}
{ISTART} {ISOLV} {IAQ} {NSET}
{DATA_WEIGHTING_AND_REWEIGHTING}
{FORWARD_MODEL}
{CID}
{ID}
```

- **The two blank lines (`""`, `""`) in the output-file-selection block were
  investigated as a suspected additional defect and are *not* a bug.**
  `getinp2.f:126–139` reads lines 5 and 6 of the control file positionally into
  `fn_loc`/`fn_reloc` with no special-casing, but `getinp2.f:287–288` then
  applies exactly the fallback the user guide documents:
  ```fortran
  600   if (trimlen(fn_loc).lt.2) fn_loc= 'hypoDD.loc'
        if (trimlen(fn_reloc).lt.2) fn_reloc= 'hypoDD.reloc'
  ```
  and the user guide (`hypoDD2_UserGuide.pdf`, §B.3.1) states verbatim:
  *"filename of initial hypocenter output (if blank: output to hypoDD.loc)"*
  and *"filename of relocated hypocenter output (if blank: output to
  hypoDD.reloc)"*. So writing blank lines is fully spec-compliant, and
  `_run_hypodd`'s later check for `hypoDD.loc`/`hypoDD.reloc` (parallel:
  678–689) is consistent with this default. **Refuted as a defect** — but
  worth keeping the explicit filenames in OpenMEQ's own writer for
  readability/robustness rather than relying on the fallback.
- `IDAT=3, IPHA=3` (cross+catalog, P&S) hardcoded (parallel:1029–1031) —
  reasonable default, should be config-driven in the rewrite (a
  catalog-only or cc-only run is a legitimate use case).
- **`OBSCC=0, OBSCT=0` (parallel:1035–1037) disables event clustering
  entirely** (`OBSCC`/`OBSCT` are "min # obs/pair" thresholds for building
  clusters; `0` means "no clustering" per `ph2dt.inp`'s own comment and
  example1's `hypoDD.inp:41-42` comment `OBSCC: min # of obs/pair for
  crosstime data (0= no clustering)`). Example1 itself uses `OBSCT=8`
  (`hypoDD.inp:42`, matching the epic note). This is a real behavioural
  choice (not a format bug) that the rewrite must not silently inherit:
  clustering is what allows well-connected sub-networks of events to be
  relocated independently and is one of hypoDD's core features; defaulting it
  off is a meaningful deviation from example1's own working configuration.
- **`IAQ=2` hardcoded** (parallel:1046; non-parallel:1228). The pre-identified
  concern in the M9.1 issue text claims "hypoDD.inp's own documented values
  for IAQ ... are 0 (no) or 1 (yes) — 2 is not a documented valid value."
  **This is refuted.** The current user guide (`hypoDD2_UserGuide.pdf`,
  §B.3.1, extracted text) states explicitly:
  ```
  IAQ    1 = Remove air-quakes. Remove from the event list for all subsequent
             iterations. The last iteration is repeated.
         2 = Keep air-quakes. Reset depths to the depths of the previous
             (successful) iteration, and continue with subsequent iterations.
  ```
  `IAQ=2` is a documented, valid, and meaningfully different mode ("keep and
  reset depth" vs. "remove"). **Additional discrepancy found and corrected
  here (see §6):** `examples/example1/hypoDD.inp:47`'s own inline comment says
  `IAQ: remove airquakes: 0=NO, 1=YES` — this is a *stale* comment from an
  older hypoDD version that does not match the current 2.1b user guide's
  documented 1/2 semantics. Interestingly, `hypoDD.f:803`
  (`if(iaq.eq.0) mbad= -mbad ! don't remove air quakes`) shows `IAQ=0` still
  has a real, working legacy effect in the compiled binary — so `0`, `1`, and
  `2` are all functional values, but only `1` and `2` are the currently
  *documented* ones. **Verdict: not a bug** — `IAQ=2` (keep-and-reset) is a
  legitimate design choice, but it differs from example1's own `IAQ=1`
  (remove), and OpenMEQ should expose `IAQ` as a config choice rather than
  hardcode either value, defaulting to `1` (matches the shipped, validated
  example) unless a documented reason favours `2`.
- **`ISOLV=2` (LSQR, conjugate gradients) hardcoded** (parallel:1045) —
  matches example1 and is the recommended solver for larger problems per the
  user guide ("LSQR solves the damped least squares problem", confirmed at
  user guide line 363); reasonable default, should stay configurable.
- **Confirmed defect — minimal/static weighting-and-reweighting default.**
  parallel:1050–1057:
  ```python
  iterations = [
      "100 1 0.5 -999 -999 0.1 0.05 -999 -999 30",
      "100 1 0.5 6 -999 0.1 0.05 6 -999 30"]
  ```
  a 2-row scheme with no progressive tightening of the outlier/distance
  cutoffs (`WRCC`/`WRCT`/`WDCC`/`WDCT` essentially only tighten once). Compare
  `examples/example1/hypoDD.inp:60-65`'s real, working 5-row scheme, which
  starts with very low weights on catalog/cc data and *very* wide residual/
  distance cutoffs (`-999` = inactive) and only progressively tightens both
  the weights and the cutoffs over 5 stages (`NITER=1,3,3,3,3`, `WRCC`
  4→‑999→6→5→4, `WDCC` -999→-999→3→2→1, etc.). This 5-row scheme is
  purpose-tuned for stable convergence (loose-then-tight), which the
  template's flat 2-row scheme does not reproduce. **Confirmed: do NOT copy
  the template's minimal default.** OpenMEQ's default should be modelled on
  example1's actual values, exposed as a configurable list of
  `{niter, wtccp, wtccs, wrcc, wdcc, wtctp, wtcts, wrct, wdct, damp}` rows
  (10 fields per row, confirmed field order from
  `getinp2.f:9,42-55` argument list and user guide §B.3.1's
  `NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP` header).
- `FORWARD_MODEL` comes from `setup_velocity_model` / `_get_forward_model_string`
  — see §1.5.
- `CID=0` (all clusters), `ID=""` (all events) (parallel:1059–1062) —
  reasonable defaults, config-overridable already.

**Verdict: port-rewritten**, with three concrete changes M9.2/M9.5 must make
relative to the template: (1) make `OBSCC`/`OBSCT` clustering thresholds
config-driven (do not default to 0/0), (2) make `IAQ` config-driven,
defaulting to `1`, (3) base the default weighting/reweighting scheme on
`examples/example1/hypoDD.inp`'s 5-row progressive scheme, exposed as a list
of rows in config, not a hardcoded 2-row string.

### 1.5 `setup_velocity_model` / `_get_forward_model_string` (parallel: 1072–1129; non-parallel: 1247–1305)

```python
forward_model = [
    "0",  # IMOD
    "%i %.2f" % (len(layers), ratio),
    " ".join(depths),
    " ".join(velocities)]
```
(parallel: 1101–1105), preceded by the docstring/comment:
*"Use imod 5 which allows for negative station elevations by using straight
rays."* (parallel: 1099–1100; identically at non-parallel: 1274–1275).

- **Confirmed defect — docstring/code mismatch.** The code unconditionally
  writes `"0"` as `IMOD`, i.e. the legacy `IMOD=0` mode
  ("1D layered P-velocity model with constant Vp/Vs ratio, hypoDD 1.0",
  user guide §B.3.1) — it does **not** use `IMOD=5` as the comment claims, and
  it does not even use `IMOD=1` (the modern, per-layer-ratio mode the epic
  note recommends, confirmed exercised live in `examples/example1/hypoDD.inp:
  100-104`). Under `IMOD=0`, `mod_ratio` is a single constant Vp/Vs for all
  layers (matches the template's single `ratio` kwarg), so the *code* is
  internally consistent with `IMOD=0`'s own format — the bug is specifically
  that the docstring's claim about negative-elevation handling
  (`IMOD=5`-only behaviour) is simply false for this code path: the negative-
  borehole-elevation problem is **not** solved by this template despite the
  comment, exactly as flagged in the pre-identified defect list. **Confirmed.**
- Also note `IMOD=0`'s format block is `NLAY RATIO` / `TOP` / `VEL` (2 lines
  after the `IMOD` line, no `-9` terminators, confirmed at user guide §B.3.1
  "For IMOD=0" and matching the commented-out `examples/example1/hypoDD.inp:
  85-91`), which is what the template actually writes — so the *format* is at
  least self-consistent with `IMOD=0`, just not with the "IMOD=5" the comment
  promises.

**Verdict: do-not-port** the `IMOD=0` choice or the false-claim comment.
OpenMEQ's 1D writer must use `IMOD=1` (per-layer Vp/Vs ratio, matches
example1's live-exercised block, format: three lines each terminated by a
literal `-9`) as the epic note already directs, and must explicitly implement
the `IMOD=5`-vs-clamp decision for negative effective elevations rather than
silently defaulting to a mode (`IMOD=0`) that guarantees the problem occurs.

---

## 2. Cross-correlation loop

### 2.1 The confirmed dt.cc sign-convention bug

**Ground truth.** `hypoDD2_UserGuide.pdf` §B.3.2 (extracted text, userguide.txt:
979–994) states verbatim:
```
DT       Differential time (s) between event 1 and event 2 at station STA.
         DT = T1-T2.
```
where "event 1"/"event 2" are `ID1, ID2` from the pair header
`#, ID1, ID2, OTC` immediately above. `getdata.f:369-402` reads the header
(`read (line,*,err=1051) str1, ic1, ic2, otc`, line 371) and then, for each
data line, reads `dt_dt(i)` **directly** from the file
(`read (line,*,err=1051) dt_sta(i), dt_dt(i), dt_qual(i), pha`, line 398) and
only applies an origin-time correction (`dt_dt(i) = dt_dt(i) - otc`, line 399)
— it never recomputes the sign from two separate travel times the way it does
for `dt.ct` (`dt_dt(i) = dt1 - dt2`, line 485, matching the `dt.ct` header's
`ID1, ID2` order exactly, confirmed at `getdata.f:453,485`). This means the
**file itself** must already contain the correctly-signed
`T(event1) - T(event2)` value; hypoDD trusts it verbatim.

**Template code** (`cross_correlate_evt_pairs`, module-level function,
parallel: 1296–1561):
- Pair header write (parallel: 1332–1335):
  ```python
  current_pair_strings.append(
      "# {event_id_1}  {event_id_2} 0.0".format(
          event_id_1=event_1, event_id_2=event_2))
  ```
  `event_1, event_2 = evt_pair` (line 1301) — so `ID1=event_1`,
  `ID2=event_2`, exactly matching hypoDD's own `#, ID1, ID2, OTC` convention.
- Differential time computation, **live code** (parallel: 1542–1544):
  ```python
  diff_travel_time = (pick_2["pick_time"] + pick2_corr -
                       event_2_dict["origin_time"]) - (pick_1["pick_time"] -
                        event_1_dict["origin_time"])
  ```
  i.e. `TT(event_2) - TT(event_1)` — the **opposite** of the required
  `T1 - T2 = TT(event_1) - TT(event_2)`.
- Immediately below, **commented-out** (parallel: 1546–1547):
  ```python
      #((pick_1["pick_time"] - event_1_dict["origin_time"]) -
      #               (pick_2["pick_time"] + pick2_corr - event_2_dict["origin_time"]))
  ```
  which computes exactly `TT(event_1) - TT(event_2)` — the **correct** sign.

**This is a confirmed, real sign inversion.** The presence of the correct
formula immediately below, commented out, strongly suggests the sign was
flipped (introduced or reintroduced) at some point after the correct version
was written, and never fixed live. The identical bug (same live formula, but
*without* the commented-out correct alternative) is present in the
non-parallel template at `hypodd_relocator.py:1127–1129` — the same defect
propagated to both templates from a common ancestor, and the parallel
version's comment is the only trace of anyone having previously noticed it.

Before/after pseudocode:
```
# before (template, both files) — WRONG, writes T2 - T1:
dt = (pick2_time + pick2_lag_corr - origin2) - (pick1_time - origin1)

# after (matches user guide DT = T1 - T2 and getdata.f's dt.ct convention):
dt = (pick1_time - origin1) - (pick2_time + pick2_lag_corr - origin2)
```

**Verdict: do-not-port as-is; the sign must be fixed in the rewrite (a named,
documented fix, not a silent copy).** → **M10.3 must not reproduce this.**

### 2.2 Hardcoded stream-read window (parallel only)

`cross_correlate_evt_pairs` (parallel: 1392–1395):
```python
for waveform_file in data_files_1:
    stream_1 += read(waveform_file, starttime=pick_1["pick_time"]-1, endtime=pick_1["pick_time"]+5)
for waveform_file in data_files_2:
    stream_2 += read(waveform_file, starttime=pick_2["pick_time"]-1, endtime=pick_2["pick_time"]+5)
```
uses a **fixed 1s-before/5s-after** window, while the surrounding code (lines
1377–1385 for `_find_data`'s search window, and 1414–1421 for the
`max_starttime`/`min_endtime` trace-selection bounds) uses the **configured**
`cc_param["cc_time_before"]`/`["cc_time_after"]`. If a project's configured
`cc_time_before` exceeds 1s or `cc_time_after` exceeds 5s, the actual samples
available to `xcorr_pick_correction` (which itself windows with the
*configured* `t_before`/`t_after`, parallel: 1493–1494) are silently
truncated to the smaller hardcoded range — `xcorr_pick_correction` would then
either raise (if it can't find enough samples) or, worse, silently operate on
a narrower window than requested, degrading or invalidating the
cross-correlation lag estimate without a warning tied to the actual cause. The
shipped example script's defaults (`hypoDDpy-master/hypoddpy/example/
run_relocator.py:8`, `cc_time_before=0.05, cc_time_after=0.2`) happen to be
comfortably inside `1`/`5`, which is presumably why this was never caught in
practice — but it is a latent, confirmed bug that any larger
`cc_time_before`/`cc_time_after` configuration (plausible for OpenMEQ's
microseismic use case, e.g. longer time windows for lower-SNR picks) will
silently trip.

**Notably absent from the non-parallel template**: `hypodd_relocator.py:
989,991` reads the **entire file** with no `starttime`/`endtime` slicing at
all (`stream_1 += read(waveform_file)`), so it does not have this specific
bug (at the cost of always reading full files, less efficient but not
incorrect). This strongly suggests the hardcoded window was introduced
specifically when the parallel/joblib version was optimised for I/O, and the
person who added it forgot to source the bounds from
`self.cc_param["cc_time_before"/"cc_time_after"]` used two lines below at
1414–1417 for the same purpose.

**Verdict: do-not-port.** → **M10.3 must always slice using the configured
`cc_time_before`/`cc_time_after`** (with margin, not the constant 1/5).

### 2.3 Sampling-rate handling — confirmed correct, not a bug

`cross_correlate_evt_pairs` (parallel: 1468–1475) checks
`trace_1.stats.sampling_rate != trace_2.stats.sampling_rate` and skips the
pick pair (logs a warning) rather than resampling. This is a reasonable,
conservative choice — no resampling is attempted, so no aliasing/interpolation
artefacts are introduced into the correlation. **Not a bug.** OpenMEQ's own
waveform cache (M10.1) can do strictly better by resampling to a common rate
up front (at cache-build time, where the choice is auditable and can be
logged once per station-pair rather than silently dropping data pair-by-pair
during the O(n_pairs) cross-correlation loop) — recorded here as a design
improvement for M10.1/M10.3, not a defect fix.

### 2.4 Multi-channel S-pick combination — design choice, not a bug

For each configured channel with nonzero weight (parallel: 1403–1523), the
loop cross-correlates independently and then combines
(parallel: 1527–1534):
```python
pick2_corr = sum(lag_i * weight_i for lag_i, cc_i, weight_i in results) / sum(weight_i)
cross_corr_coeff = sum(cc_i * weight_i for ...) / sum(weight_i)
```
i.e. a **fixed per-channel weighted average** of independently-measured lags
and coefficients (not weighted by the actual correlation quality of each
channel). This is a legitimate design choice used consistently by both
templates, not a defect — but it means a poorly-correlating horizontal
channel can drag down (or bias) an otherwise-good lag estimate from a
well-correlating one. **Flagged as a design decision to revisit in M10.3**
(e.g., picking the single best-correlating channel, or weighting by
`cross_corr_coeff` itself rather than a fixed per-channel weight), not a bug
to "fix."

### 2.5 dt.cc weight column — confirmed correct

The written weight is the raw `cross_corr_coeff` (0..1) after the
`cc_min_allowed_cross_corr_coeff` threshold filter (parallel: 1536–1554,
`weight=cross_corr_coeff` at line 1553). `getdata.f:398`'s `dt_qual(i)`
field reads this same column position, and the user guide (§B.3.2)
documents `WGHT` as *"Weight of measurement (0-1; e.g. squared coherency)"* —
exactly what a cross-correlation coefficient is. **Confirmed correct, no fix
needed.**

### 2.6 Additional defect found — misleading variable name for the dt.cc path

`_cross_correlate_picks` (parallel: 909): the output path variable is named
`ct_file_path` but is actually `os.path.join(self.paths["input_files"],
"dt.cc")` — a `dt.cc` (cross-correlation) path named as if it were `dt.ct`
(catalog). The value used is correct (`"dt.cc"`), so this is not behaviourally
wrong, but it is confusing enough that it is worth flagging so the rewrite
doesn't propagate the naming confusion into a place where it *would* matter
(e.g. a future refactor that assumes the variable name reflects the file
content).

---

## 3. Event-pair selection from dt.ct

`get_event_id_pairs` (parallel: 886–900) parses `dt.ct`'s `#`-prefixed header
lines (`event_id_1, event_id_2 = map(int, line.split())` after stripping the
leading `#`, line 895) to build the list of event-id pairs to
cross-correlate, and skips any pair whose `cc_dir/{id1}_{id2}.txt` cache file
already exists (`_cross_correlate_picks`, parallel: 924; `evt_pair_file`
existence check inside `cross_correlate_evt_pairs`, parallel: 1303–1306).
This correctly derives the exact set of pairs hypoDD itself will use (ph2dt
has already filtered them, per `MAXSEP`/`MAXNGH`/etc., by the time `dt.ct`
exists) rather than cross-correlating the full O(n_events²) combinatorial
space — a good, worth-keeping design choice.

**Confirmed additional defect — the per-pair cache is unsafe across
re-runs with a changed event set**, directly corroborating (with a concrete
mechanism) the pre-identified "event-id mapping is in-memory only" concern:

1. `_create_event_id_map` (parallel: 546–561) assigns `self.event_map[event_id]
   = index+1` sequentially over `self.events`, which is itself sorted by
   origin time (parallel: 532) — deterministic *only if the exact same set of
   events is present in the exact same order every run*.
2. `event.dat`/`phase.dat` are written with these transient integer IDs as the
   `CUSPID` (parallel: 395), so `dt.ct`'s pair headers, and hence
   `get_event_id_pairs`'s parsed `(event_id_1, event_id_2)` tuples, are always
   expressed in terms of *this run's* transient mapping.
3. The per-pair cross-correlation cache filename is exactly
   `"%i_%i.txt" % (event_1, event_2)` (parallel: 1303–1304) — i.e., keyed
   purely by the transient integers, with **no validation that the integers
   still refer to the same two events** as when the cache file was written.
4. Compounding this: `_read_event_information` (parallel: 442–455),
   `_parse_station_files` (parallel: 289–294) and `_parse_waveform_files`
   (parallel: 817–827) all use an identical "if the serialized JSON cache file
   exists on disk, load it and return — ignoring whatever was just passed to
   `add_event_files`/`add_station_files`/`add_waveform_files`" pattern, with no
   fingerprint/hash check against the current input file list.

Net effect: if a project's event catalogue changes between runs (new events
added, an event rejected/removed, or even just a change in file read order)
while the `working_dir`'s `working_files/*.json` and `working_files/cc_files/
*.txt` caches from a previous run are still present, the pipeline will
silently mix up which cross-correlation numbers belong to which event pair —
this is a correctness risk, not merely an efficiency one, and it is the
**opposite of what a resumable/incremental pipeline needs.**

**Verdict: do-not-port the caching mechanism.** M9.2's event-id map must be
persisted deterministically to disk (e.g. `relocate/work/event_id_map.csv`,
`event_id,cuspid`, as the epic note already directs) and any per-pair /
per-step cache (M10.1's waveform cache, M10.3's cross-correlation cache) must
be keyed by content-stable identifiers (`event_id` strings, or a
config+input fingerprint via `openmeq.manifest.StepManifest`/
`pmap_incremental`, per ARCHITECTURE.md §2) — never by a transient,
run-order-dependent integer, and never by bare file-existence with no
fingerprint check. → **M10.1/M10.2/M10.3 must not reproduce this.**

---

## 4. Reading `hypoDD.reloc` back

`_create_output_event_file` (parallel: 1131–1180; non-parallel: 1306–1356)
parses each line of `hypoDD.reloc` with a 24-field unpack:
```python
event_id, lat, lon, depth, _, _, _, _, _, _, year, month, \
    day, hour, minute, second, _, _, _, _, _, _, _, \
    cluster_id = line.split()
```
This matches the confirmed 24-field write statement at `hypoDD.f:1283-1298`
(`CUSPID LAT LON DEPTH_KM X_M Y_M Z_M EX_M EY_M EZ_M YR MO DY HR MI SEC MAG
NCCP NCCS NCTP NCTS RCC RCT CID`) exactly in field count and position:
`event_id`↔CUSPID (position 1), `lat/lon/depth`↔positions 2-4, six
underscore-discarded fields↔X,Y,Z,EX,EY,EZ (positions 5-10), `year..second`↔
positions 11-16, seven more discarded fields↔MAG,NCCP,NCCS,NCTP,NCTS,RCC,RCT
(positions 17-23), `cluster_id`↔CID (position 24). **Confirmed correct** —
the template reads the real `hypoDD.reloc` (opened once, `hypoDD.f:479`,
closed at `hypoDD.f:1317`), not the superficially similar 18-field
per-iteration scratch file `mdat.reloc.NNN.NNN` (`hypoDD.f:958-976`) that
some hypoDD tutorials mistakenly conflate with the final output — the epic
note's warning about this distinction is itself confirmed accurate.

- The template discards magnitude and all per-event/residual counts
  (positions 17-23) — reasonable for hypoddpy's scope, but OpenMEQ's own
  reader (M10.4) should **not** discard `NCCP/NCCS/NCTP/NCTS/RCC/RCT`, since
  these are exactly the kind of per-event QC fields ARCHITECTURE.md's
  `locate/summary.csv`-style diagnostics need, and `MAG` should be carried
  through into the output QuakeML per ARCHITECTURE.md §7's "magnitudes appear
  in hypoDD.reloc" note.
- The seconds-rounding-to-60 workaround (parallel: 1158–1168,
  `sec = int(float(second)); if sec >= 60: sec = 0; add_minute = True`) is a
  correct, worth-keeping defensive fix for a documented real hypoDD quirk
  (rounding `SEC` to two decimal places can round up to `60.00`). Verified
  plausible given `hypoDD.f`'s write format `f6.3` for `SEC` (6 total digits,
  3 decimal places, i.e. rounds to millisecond precision) — a value of
  `59.9996` could format-round to `"60.000"`. Keep this fix.
- `event = res_id.getReferredObject()` (parallel: 1155) relies on ObsPy's
  global resource-id registry resolving `ResourceIdentifier(event_id)` back to
  the in-memory `Event` object read earlier in the same process from
  `self.event_files` — a fragile pattern (depends on object identity/registry
  state, not a lookup by explicit id in a dict) that will not survive being
  split across OpenMEQ's step boundaries (write catalogue → separate process
  runs hypoDD → read catalogue back). **Do not port this mechanism.**
  OpenMEQ's reader must look up events explicitly by `event_id` (via
  `openmeq.catalog`'s persisted event-id↔cuspid map), not via ObsPy's
  resource-id registry.

**Verdict: port-rewritten.** Field layout is fully verified and correct;
rebuild the event lookup on the persisted event-id map (§3) instead of
ObsPy's resource-id registry, keep the seconds-rounding fix, and carry through
magnitude/NCCP-NCCS-NCTP-NCTS/RCC/RCT rather than discarding them.

---

## 5. Parallelism (joblib) patterns worth keeping vs. Python-2-isms to discard

**Worth keeping (the underlying idea, reimplemented in OpenMEQ's own
concurrency primitives):**
- Splitting the *dt.ct-derived event-pair list* (not the full combinatorial
  event-pair space) into per-core batches before cross-correlating
  (parallel: 930–957) — the batching logic itself is a manual, hand-rolled
  round-robin split; OpenMEQ's `openmeq.parallel.pmap`/`pmap_incremental`
  (ARCHITECTURE.md §2) already provides this via `ProcessPoolExecutor` with
  proper `spawn`-based thread-count control, and should be used directly
  instead of hand-rolling batch assignment.
- Writing one small text file per event pair (`cc_dir/{id1}_{id2}.txt`,
  parallel: 1556–1558) and reassembling them at the end (parallel: 962–969)
  is a reasonable resumability mechanism in isolation — but see §3: it must be
  rekeyed on stable event ids and ideally replaced by
  `pmap_incremental`'s JSON-lines "append as completed, skip already-present
  keys" mechanism (ARCHITECTURE.md §2), which already solves this problem
  correctly and atomically.
- Passing `self` explicitly into a **module-level** function
  (`cross_correlate_evt_pairs`, parallel: 1296, called via
  `Parallel(n_jobs=ncores)(delayed(cross_correlate_evt_pairs)(self, cc_dir,
  cc_plot_dir, evt_pairs) for evt_pairs in evt_pair_jobs)`, parallel: 953–955)
  rather than calling a bound method directly is a deliberate (if
  unusual-looking) pattern to keep the callable's own closure minimal and
  `dill`-picklable (the file imports `dill as pickle`, parallel: 22, which
  handles pickling instance methods/complex objects that stock `pickle`
  cannot) — this workaround is unnecessary with `pmap`'s
  `ProcessPoolExecutor` + top-level-picklable-function contract
  (ARCHITECTURE.md §2: *"func must be a top-level, picklable function, and
  items should be paths or parameters, not waveform objects"*) — OpenMEQ's
  rewrite should follow that contract directly rather than needing `dill`.

**Python-2-isms to discard entirely** (confirms the issue's framing that this
template cannot run as-is under Python 3.10):
- `except Exception, err:` (parallel: 1514) — Python 2 exception syntax,
  a `SyntaxError` under Python 3.
- `.iteritems()` (parallel: 340, 405, 881, 1405; non-parallel equivalents) —
  removed in Python 3 (`dict.items()` only).
- Bare `print "..."` statements (parallel: 274, 278, 944, 949, 957, 959–960,
  1311, 1482, 1559–1560) — `SyntaxError` under Python 3.
- `basestring` (parallel: 165, 168, 183, 186, 201, 204, 241) — removed in
  Python 3 (`str` only).
- `err.message` (parallel: 1516, 1518) — `BaseException.message` was removed
  in Python 3; must use `str(err)`.
- Absolute hardcoded local paths to load `hypodd_compiler.py` via
  `imp.load_source` (parallel: 25–26, referencing
  `/home/steve/seismic/hypoDDpy-master/hypoddpy/hypodd_compiler.py`, with a
  second, commented-out macOS path immediately below) — confirms this is
  genuinely a personal working script, not a packaged library; `imp` itself
  is also removed in modern Python (deprecated since 3.4, removed 3.12).

---

## 6. Master defect list (cross-referenced to downstream tasks)

| # | Defect / finding | Status | Evidence (template) | Evidence (hypoDD ground truth) | Downstream task that must not reproduce it |
|---|---|---|---|---|---|
| 1 | dt.cc differential-time sign is `T2-T1` instead of `T1-T2` | **Confirmed bug** | `hypodd_relocator_parallel.py:1542-1544` (live), correct alternative commented at `:1546-1547`; identical bug at `hypodd_relocator.py:1127-1129` | `getdata.f:398-399` (reads dt.cc value verbatim + OTC only, no recompute) + user guide §B.3.2 "DT = T1-T2" (userguide.txt:992) | M10.3 |
| 2 | Hardcoded 1s-before/5s-after stream read window ignores configured `cc_time_before`/`cc_time_after` | **Confirmed bug** (parallel only; non-parallel reads whole file, no bug) | `hypodd_relocator_parallel.py:1392-1395` vs. configured bounds used at `:1377-1385,1414-1421` | — (internal template inconsistency; corroborated by `example/run_relocator.py:8`'s narrow defaults masking it) | M10.3 |
| 3 | `setup_velocity_model` docstring claims IMOD=5 (negative-elevation-safe) but code hardcodes IMOD=0 | **Confirmed bug** | `hypodd_relocator_parallel.py:1099-1102` (and `hypodd_relocator.py:1274-1277`) | user guide §B.3.1 IMOD definitions (userguide.txt:924-931); `examples/example1/hypoDD.inp:100-104` shows the correct IMOD=1 block actually exercised | M9.2 |
| 4 | IAQ=2 hardcoded, claimed by issue text to be an undocumented value (0/1 only) | **Refuted** — IAQ=2 ("keep air-quakes, reset depths") is documented and valid in the current user guide; template's choice of 2 vs. example1's 1 is a design difference, not a bug. Also found: `examples/example1/hypoDD.inp:47`'s own inline comment ("0=NO,1=YES") is a stale mismatch with the current guide — noted as its own discrepancy. | `hypodd_relocator_parallel.py:1046`, `hypodd_relocator.py:1228` | user guide §B.3.1 IAQ (userguide.txt:914-917); `hypoDD.f:803` (`iaq.eq.0` still functional); `examples/example1/hypoDD.inp:47` (stale comment) | M9.2/M9.5 (expose IAQ as config, default 1) |
| 5 | Static/minimal 2-row `DATA_WEIGHTING_AND_REWEIGHTING` default, vs. example1's real 5-row progressive scheme | **Confirmed, do not copy** | `hypodd_relocator_parallel.py:1050-1057` | `examples/example1/hypoDD.inp:60-65` | M9.2 |
| 6 | Event-id mapping is in-memory only, regenerated per run, and used as the key for an unvalidated per-pair cross-correlation cache | **Confirmed bug** — correctness risk for a resumable pipeline, not just an efficiency concern | `hypodd_relocator_parallel.py:546-561` (map), `:1303-1306` (cache key), `:442-455,289-294,817-827` (identical "trust cache blindly" pattern in three other loaders) | — (internal design flaw, no hypoDD-side corroboration needed) | M10.1/M10.2/M10.3 |
| 7 | `_write_station_input_file`'s elevation write has no negative-elevation/IMOD-5 handling at all | **Confirmed, real quirk not addressed** | `hypodd_relocator_parallel.py:341-342` | `getdata.f:275-278,286-288` (negative elevation silently zeroed unless IMOD=5) | M9.2 |
| 8 | Sampling-rate mismatch causes pair to be skipped, no resampling attempted | **Confirmed correct, not a bug** | `hypodd_relocator_parallel.py:1468-1475` | — | (M10.1 may improve on it by resampling at cache-build time) |
| 9 | Multi-channel S-pick lag/coefficient combination uses fixed per-channel weights, not correlation-quality weighting | **Confirmed as-designed, flagged for reconsideration** | `hypodd_relocator_parallel.py:1403-1523,1527-1534` | user guide §B.3.2 WGHT description (userguide.txt:993) | M10.3 (design decision, not a bug fix) |
| 10 | dt.cc weight column is the raw 0..1 cross-corr coefficient | **Confirmed correct, matches `dt_qual`** | `hypodd_relocator_parallel.py:1549-1554` | `getdata.f:398` (`dt_qual(i)` read position); user guide §B.3.2 (userguide.txt:993) | — |
| 11 (additional) | `station.dat` labels are unmapped `NET.STA` (routinely >7 chars) written straight into a `character*7` field | **Confirmed additional defect** | `hypodd_relocator_parallel.py:307-308,341-342` | `getdata.f:51` (`character sta_lab(MAXSTA)*7`) | M9.2 |
| 12 (additional) | `phase.dat` weight defaults to a constant `1.0` rather than hypoDD's documented `-1` "no info" sentinel | **Confirmed design gap, not currently a functional bug** | `hypodd_relocator_parallel.py:89,416-422` | `ph2dt.f:485` (`.ge.0.and..lt.minwght` filter, negative always bypasses) + user guide §B.3.2/B.3.3 WGHT description | M9.2 |
| 13 (additional) | `_cross_correlate_picks`'s dt.cc output path is misleadingly named `ct_file_path` | **Confirmed, harmless naming bug** | `hypodd_relocator_parallel.py:909` | — | M10.3 (rename only, no behaviour change) |
| 14 (additional) | `OBSCC=0, OBSCT=0` hardcoded, disabling hypoDD's event-clustering feature entirely | **Confirmed, do not silently inherit** | `hypodd_relocator_parallel.py:1035-1037` | `examples/example1/hypoDD.inp:41-42` (OBSCT=8 in the shipped, working example) | M9.2/M9.5 |
| 15 (additional) | `_create_output_event_file` resolves events via ObsPy's global `ResourceIdentifier` registry rather than an explicit id lookup | **Confirmed fragile pattern, do not port** | `hypodd_relocator_parallel.py:1151,1155` | — | M10.4 |
| 16 (investigated, refuted) | Blank `hypoDD.loc`/`hypoDD.reloc` filename lines in `hypoDD.inp` | **Refuted as a defect** — matches documented blank-defaults-to-`hypoDD.loc`/`hypoDD.reloc` behaviour | `hypodd_relocator_parallel.py:1014-1015` | `getinp2.f:287-288` (fallback logic) + user guide §B.3.1 (userguide.txt:809-812) | — |

---

## 7. Corrections to the M9 epic design note (OpenMEQ-6ld)

Per work item 6/acceptance criterion 4, the following corrects the epic's
design note where this audit's closer reading found it wrong:

1. **IAQ documented values.** The epic note states *"IAQ 0/1 remove
   airquakes"*. The current `hypoDD2_UserGuide.pdf` (§B.3.1) documents `IAQ`
   as `1 = Remove air-quakes` / `2 = Keep air-quakes (reset depths, continue)`
   — not `0`/`1`. (`0` remains a functional legacy value in the compiled
   source, `hypoDD.f:803`, but is not part of the currently documented
   interface.) This appears to derive from `examples/example1/hypoDD.inp:47`'s
   own stale inline comment (`"IAQ: remove airquakes: 0=NO, 1=YES"`), which
   itself does not match the current user guide. **No other part of the epic
   note's compiled-array-limits, file-format, or `hypoDD.reloc` field-layout
   tables was found to be incorrect** — every other checked claim (`MAXEVE`,
   `MAXDATA`, `MAXLAY`, `MAXSTA`, `MAXCL` in `hypoDD.inc`; `maxnx/maxny/maxnz/
   ixkms/iykms/izkms` in `vel3d.inc`; `MEV/MSTA/MOBS` in `ph2dt.inc`;
   `get_vel3d.f`'s `iphase,lat_3d,lon_3d,rot_3d,indip,iiskip,iscale1,iscale2,
   ixfac,itlim,initpb` argument order; the `hypoDD.reloc` 24-field layout; the
   `station.dat`/`event.dat`/`phase.dat`/`ph2dt.inp`/`hypoDD.inp` format
   tables) was independently re-verified byte-for-byte against the source and
   found accurate.
2. No changes to `docs/ARCHITECTURE.md` were required by this audit — the
   architecture document does not itself encode the IAQ value table (that
   lives only in the M9 epic's design note), so no update to
   ARCHITECTURE.md is needed in this commit.

---

## 8. Summary verdicts

| Component | Verdict |
|---|---|
| `station.dat` writer | port-rewritten (add label mapping + negative-elevation policy) |
| `phase.dat` writer | port-rewritten (rebuild weight from `Arrival.time_weight`) |
| `ph2dt.inp` writer | port-rewritten (defaults from `ph2dt.inp`'s own comments/example1) |
| `hypoDD.inp` writer | port-rewritten (config-driven OBSCC/OBSCT, IAQ; 5-row default weighting scheme; IMOD=1 not IMOD=0) |
| Cross-correlation loop | do-not-port as-is; fix sign bug, fix window, keep sampling-rate check, revisit channel-combination weighting |
| Event-pair selection (dt.ct-driven) | port-rewritten (keep the "only cross-correlate ph2dt-selected pairs" idea; replace the cache mechanism) |
| `hypoDD.reloc` reader | port-rewritten (keep field layout + seconds-rounding fix; replace resource-id lookup; carry through MAG/NCCP.../RCC/RCT) |
| joblib parallelism pattern | do-not-port mechanism; reimplement the underlying idea with `openmeq.parallel.pmap`/`pmap_incremental` |
| Python-2 syntax throughout | do-not-port; full rewrite required for Python 3.10 |
