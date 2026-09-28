# hypoDD/ph2dt tuning for small-aperture microseismic networks

OpenMEQ-16d.6. Explains every distance- and weight-related `cfg.hypodd`/
`cfg.hypodd.ph2dt` default (`openmeq/config.py`, mirrored with the same
justification in `config/example.yaml`), why they differ from
`external/HYPODD/examples/example1`'s own regional-network values, how to
rescale them for a different network, what "air-quakes" mean for a shallow
or borehole-referenced network, and a troubleshooting table. Every claim
about hypoDD/ph2dt's own behaviour below is sourced from
`external/HYPODD/doc/hypoDD2_UserGuide.pdf` (extracted with `pdftotext
-layout`) and cross-checked against `docs/hypoddpy_audit.md` and
`docs/ARCHITECTURE.md` §6c/§6c-2/§6g/§6h, which this document summarises but
does not duplicate.

## Why this doc exists

M10.4 (OpenMEQ-16d.4) found that this project's original hypoDD defaults —
copied from `example1`, a regional earthquake catalogue spanning tens to
over a hundred km — sent most events of the 20-event, 8-station synthetic
network (`tests/data/make_synthetic.py`, ~14.9 km station aperture) into
"air-quake" removal by the final iteration. The end-to-end relocation test
(`tests/test_relocate_hypodd_runner.py::
test_synthetic_end_to_end_relocation_improves_on_perturbed_input`) had to
override `cfg.hypodd.weighting_rows` and three `ph2dt` fields to pass.
OpenMEQ-16d.6 replaces those per-test overrides with new *defaults*, tuned
for a small-aperture (~10-15 km) network, and this test now passes on the
default config with all 20/20 relocated events improved (mean 3D error
~0.04 km against a 0.5 km per-event perturbation).

The root cause, confirmed empirically against the real `hypoDD`/`ph2dt`
binaries (not just read off the source): several defaults are either an
absolute distance in km that assumes a regional network's own scale
(`ph2dt`'s `MAXSEP`, hypoDD's `WDCC`/`WDCT`/`DIST`/`max_station_dist_km`),
or a "how many observations exist" count that assumes a much larger
station/phase count (`ph2dt`'s `MINOBS`/`MAXOBS`). None of these are
inherently wrong — they are `example1`'s own working values, or ph2dt's own
bracket-comment suggestions — they are simply the wrong *scale* for a
network an order of magnitude smaller in aperture. The residual-threshold
parameters (`WRCC`/`WRCT`) need no such rescaling: read on, they are already
scale-free.

## This project's reference network

Every "why this number" justification below is anchored to
`tests/data/make_synthetic.py`'s synthetic network, since that is what
`config/example.yaml` and the default-config end-to-end test are tuned
against, and it is representative of the aperture OpenMEQ targets:

- 8 stations (6 surface, 2 borehole), **~14.9 km station-station aperture**.
- 20 events, depths 0.3-8 km (drawn from `EVENT_DEPTH_KM_RANGE`), horizontal
  offsets up to 9 km from the network centre in each direction
  (`EVENT_OFFSET_KM_RANGE`) — i.e. events are not confined to a tight
  sub-km swarm in this test fixture, they can be spread across most of the
  monitored volume.
- Measured directly from this fixture's own truth locations (script used to
  derive the numbers below is not checked in; re-derive with
  `tests/data/make_synthetic.py`'s `_build_events` if you need to redo this
  for a different seed/network): pairwise 3D inter-event distance ranges
  0.8-19.1 km (median ~9.4 km); **nearest-neighbour distance per event**
  ranges 0.8-5.2 km (median ~2.3 km); **event-to-station hypocentral
  distance** up to ~17.5 km.

A genuinely tight microseismic swarm (real induced seismicity often
clusters within tens to hundreds of metres) would need *tighter* distance
cutoffs than this project's own defaults below for optimal fine-scale
resolution — see "Scaling for a different network" for how to retune. The
shipped defaults are deliberately generous enough to keep this test
fixture's wider spread fully connected, since that is what
`config/example.yaml` and the default-config test both use.

## ph2dt parameters (`cfg.hypodd.ph2dt`)

ph2dt's job (`external/HYPODD/doc/hypoDD2_UserGuide.pdf` §"Data
Preprocessing Using ph2dt") is to search catalog P/S phase data for event
pairs with common-station arrivals and subsample them into a *linked
network*: each event gets up to `MAXNGH` neighbours, found within search
radius `MAXSEP`, where a neighbour counts as "strong" only once linked by
at least `MINLNK` phase observations. `MINOBS`/`MAXOBS` then bound how many
observations are kept per linked pair, and `MAXDIST`/`MINWGHT` filter which
individual phase picks are eligible at all. The user guide's own advice
(directly relevant here): *"It is recommended to be rather liberal in the
selection of data using ph2dt, and to be more restrictive in hypoDD
(through `OBSCT` and `WDCT`) if necessary."* ph2dt is a *candidate-pair*
filter; hypoDD's own clustering and re-weighting is where the real
selectivity should happen.

| Field | Meaning | ph2dt.inp bracket default | Project default | Why |
|---|---|---|---|---|
| `maxsep_km` | Max. hypocentral separation (km) between a candidate event pair | 10 | **10.0** | The user guide's own "~10 km is appropriate to start with in many regions" is *not* scaled to network aperture — it is about a geophysically meaningful separation (small compared to velocity heterogeneity scale length), so it is kept unscaled rather than shrunk for a small network. Empirically necessary at this value for this project's own synthetic network: two of its 20 events have no true neighbour within 8 km of each other (only of themselves, at 5.24 km) — below ~8.9 km, ph2dt gives that pair no second link into the rest of the network, and hypoDD's clustering (`OBSCC`/`OBSCT`) then splits them into an isolated 2-event cluster, which hypoDD refuses to relocate ("Cluster has less than 2 events."). This is the single largest previously-tested-around defect: the un-audited old default was `1.0` km, well below even the bracket comment's own "10". |
| `maxdist_km` | Max. distance (km) between an event pair and a station (filters individual picks) | 500 (generic bracket) / 200 (this project's old default, matching ph2dt's other bracket line) | **25.0** | ~1.5x this project's own ~15 km network aperture — comfortably covers every station regardless of event depth or borehole offset (measured worst case ~17.5 km hypocentral distance) while excluding a station well outside the monitored volume on a larger multi-site deployment, rather than the regional-network 200 km value silently including everything regardless of relevance. |
| `maxngh` | Max. number of neighbours per event | 10 | **10** (unchanged) | Matches both the bracket default and `example1`; no small-network-specific reason to change it — `maxsep_km`/`minobs` were the actual bottlenecks (see above/below). |
| `minlnk` | Min. number of phase observations to count a neighbour as "strong" | 8 | **8** (unchanged) | The user guide: *"A strong link is typically defined by eight or more observations (one observation for each degree of freedom)"* — a network-size-independent stability floor, not a scale parameter. Must stay `>= hypodd.clustering.obsct` (see below). |
| `minwght` | Min. pick weight kept (`[0,1]`, `>=` comparison; hypoDD's own `-1` "no info" sentinel always bypasses this regardless of value, per `ph2dt.f:485` and `docs/ARCHITECTURE.md` §6c) | 0 | **0.0** (unchanged) | No reason to discard any positively-weighted pick for a data-scarce small network. |
| `minobs` | Min. observations kept per linked pair (pairs below this are discarded entirely) | 8 | **1** | The user guide, verbatim, for exactly this situation: *"for a small number of events that form a tight cluster one might select all phase pairs available by setting `MINOBS = 1`, `MAXOBS` to the number of stations, and `MAXNGH` to the number of events."* A small network cannot afford to throw away already-scarce inter-event links; `MINOBS=8` (the bracket default, aimed at "10,000 well connected events") would discard most of this project's own pairs. |
| `maxobs` | Max. observations kept per linked pair (nearest stations first) | 20 (bracket) / 50 (regional-network rule of thumb) | **16** | 2 phases (P, S) x 8 stations = every possible pick, per the user guide's own small-network recommendation above ("`MAXOBS` to the number of stations"), doubled here since this project always requests P&S (`ipha: 3`). Scale this to `2 x n_stations` for a different station count. |

## hypoDD weighting/re-weighting schedule (`cfg.hypodd.weighting_rows`)

hypoDD's `NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT DAMP` table
(`WeightingRow`, one row per weighting "stage", `NITER` = last iteration
that stage's weights apply to) controls, per iteration: the a-priori
weights on cross-correlation (`WTCCP`/`WTCCS`) and catalog (`WTCTP`/
`WTCTS`) P/S data; the outlier-residual cutoffs (`WRCC`/`WRCT`); the
max-linked-pair-separation cutoffs (`WDCC`/`WDCT`, km); and the LSQR
damping factor (`DAMP`). The recommended structure (user guide, "Data
Weighting and Re-weighting") is *loose-then-tight*: start with wide/inactive
cutoffs so the solver can find the large-scale structure, then progressively
tighten both the a-priori weights and the residual/distance cutoffs to
resolve fine detail and reject outliers.

**Two field families need opposite treatment when adapting a schedule to a
different network:**

- **`WRCC`/`WRCT` (residual thresholds) are already scale-free.** Per the
  user guide's own field description (`hypoDD2_UserGuide.pdf` §B.3.1):
  a value in `(0, 1)` is an absolute-seconds static cutoff, but **`>= 1` is
  a multiplier of the *current* residual standard deviation** (a dynamic
  cutoff). Every row in this project's default schedule uses values `>= 1`
  (4-10), so they auto-scale to whatever residual spread the real data has
  — a noise-free synthetic dataset with millisecond residuals and a noisy
  real dataset with decisecond residuals both get a sensible cutoff with
  the *same* numbers. Do not rescale these for network size.
- **`WDCC`/`WDCT` (distance cutoffs, km) are absolute and must be rescaled
  to network aperture.** `example1`'s own final-row value (1-2 km) is a
  *large fraction* of its own network's inter-event spread — appropriate
  there, since the goal in the final iterations is deliberately to collapse
  onto near-duplicate "repeating earthquake" pairs (user guide Figure 3).
  Applied unscaled to a small-aperture network whose events are not
  confined to a sub-km patch, that same 1-2 km cutoff instead disconnects
  most event pairs by the final iteration — this **is** the "most events
  become air-quakes" failure M10.4 hit, confirmed by re-running this
  project's own end-to-end test with the old (example1-derived) schedule:
  hypoDD's own per-iteration `EV` column (fraction of events still
  connected) visibly drops once the schedule reaches its final,
  over-tight `WDCT=2`/`WDCC=1` row.

### Chosen default schedule

```
NITER WTCCP  WTCCS  WRCC  WDCC  WTCTP  WTCTS  WRCT  WDCT  DAMP
  5    0.01   0.005  -999  -999  1.00   0.5    -999  -999  20
  5    0.01   0.005  -999  -999  1.00   0.5    10    10    20
  5    1      0.5     6     8    1.00   0.5     8     8    20
  5    1      0.5     5     6    0.1    0.05    6     6    20
  5    1      0.5     4     6    0.01   0.005   6     6    20
```

- **Structure** (rows 1-2: catalog-only, no cutoffs, establish large-scale
  locations; row 3: bring in cross-correlation data at full weight, wide
  cutoffs; rows 4-5: shift weight from catalog to cross-correlation data as
  outlier control tightens) mirrors `example1`'s own progression –
  loose-then-tight, catalog-then-cross-correlation.
- **`WDCC`/`WDCT` bottom out at 6 km** (~40% of this network's ~15 km
  aperture) instead of `example1`'s 1-2 km. 6 km was chosen empirically:
  this project's synthetic network's worst-case "only neighbour" separation
  is 5.24 km (see "Reference network" above); the final row must stay
  *above* that or those two events lose their only link exactly when the
  schedule tightens the most, and get dropped as unconnected /
  under-determined. 6 km is the smallest round number with headroom above
  5.24 km — tightening further (e.g. back to `example1`'s 1-2 km) is safe
  *only* once you know your own network's events form a genuinely tighter
  cluster than this project's own spread-out test fixture (see below).
- **`NITER=5` per row** (25 iterations total) rather than `example1`'s
  `1,3,3,3,3` (13 total): a few more iterations per stage for LSQR to
  converge stably at each weighting level before the next tightening step;
  inexpensive for a 20-event problem (the whole end-to-end test, real
  `ph2dt`+`hypoDD` binaries, runs in ~12 s).
- **`DAMP=20`** throughout, unchanged from `example1`. The user guide:
  *"Generally a damping between 1 and 100 is appropriate, resulting in a
  condition number that is between about 40 and 80."* The real run against
  this project's synthetic network reports `CND` (condition number) in the
  low-to-mid 20s-40s throughout — comfortably inside the recommended range,
  so no retuning was needed here. **Check your own `hypoDD.log`'s `CND`
  column when adapting this elsewhere** (see troubleshooting table).

## hypoDD solution-control and data-selection fields

- **`istart=2`** (start from catalog locations, not cluster centroid) and
  **`isolv=2`** (LSQR, not SVD) — unchanged from the M9.2 design note;
  LSQR scales to much larger event counts than SVD (`example1` itself uses
  SVD, appropriate only for its own ~100-event scale) and is this project's
  documented choice regardless of network aperture, so this issue does not
  revisit either.
- **`idat=3`/`ipha=3`** (cross-correlation + catalog data, P&S phases) —
  unchanged; a deliberate default (both data types combined), not related
  to network size.
- **`max_station_dist_km` (hypoDD.inp's `DIST` field, max. distance between
  cluster centroid and a station) = 25.0`**, matching `ph2dt.maxdist_km`
  above for the same reasoning (~1.5x this network's own aperture; was 200,
  `example1`'s own smaller 120 already hinted the regional 200 default
  was too loose even there).
- **`clustering.obscc=0`, `clustering.obsct=8`** (hypoDD's *own* clustering,
  `src/hypoDD/cluster1.f` — not a ph2dt feature, `docs/ARCHITECTURE.md` §6h)
  — unchanged. `OBSCT` must stay `<= ph2dt.minlnk` (user guide: *"The value
  for `OBSCT` in hypoDD should generally be equal to or less than the value
  for `MINLNK` in ph2dt to ensure enough strong links between closeby
  events"*) — both are 8 here, at the boundary the guide recommends, so
  this issue leaves them as-is. Lowering `OBSCT` below 8 would admit more
  weakly-linked events into a single cluster at some cost to stability;
  raising it (or `MINLNK`) further would split the network into more,
  smaller clusters — see the troubleshooting table.
- **`iaq=1`** (remove air-quakes) — unchanged; see the next section for what
  this means for a small/shallow/borehole network specifically.

## Scaling for a larger or denser network

The defaults above are tuned to *this* project's own ~15 km-aperture, 8
-station, 20-event reference network. Rescale as follows for a different
deployment:

1. **Measure your own network's aperture** (max station-station distance)
   and a rough sense of your expected event spread (a tight injection-well
   swarm vs. events spread across the whole monitored volume — the latter
   needs the more generous end of these ranges, matching this project's own
   choice above).
2. **Distance parameters scale roughly linearly with aperture:**
   - `ph2dt.maxsep_km`: keep the user guide's unscaled "~10 km to start"
     figure as an upper bound for a network aperture of that size or
     larger; for a much smaller aperture (e.g. a single injection well
     monitored by a tight downhole array, aperture < 1 km), scale down
     proportionally, but keep it above your own network's worst-case
     nearest-neighbour separation (re-derive this the way "Reference
     network" above did, from your own expected event locations) or you
     will reproduce the exact isolated-event-pair failure this issue fixed.
   - `ph2dt.maxdist_km` / `hypodd.max_station_dist_km`: ~1.5x aperture,
     enough to cover every station at your network's typical event depths
     without pulling in stations from an unrelated, more distant deployment.
   - Weighting schedule's final-row `WDCC`/`WDCT`: 30-40% of aperture is a
     reasonable starting point (this project's own choice, 6 km / ~40% of
     15 km); tighten it further, down toward `example1`'s 1-2 km, only once
     you have empirical evidence (e.g. this project's own end-to-end test
     pattern: check `n_improved`/mean error, and hypoDD's own per-iteration
     `EV`/`AQ` columns in `hypoDD.log` for events dropping out) that your
     actual event population is tightly enough clustered to support it —
     a genuinely tight swarm (inter-event distances of tens to hundreds of
     metres) benefits from a *tighter* final cutoff than this project's
     defaults, for better fine-scale double-difference resolution.
3. **Observation-count parameters scale with station/phase count and
   catalogue size:**
   - `ph2dt.maxobs`: `2 x n_stations` (P&S) as a starting point for a small
     network (this project's own reasoning); for a large, dense network
     where every event-pair link would otherwise carry far more
     observations than needed for a well-determined solution, cap it back
     down near the classic `MAXOBS=50` rule of thumb instead.
   - `ph2dt.minobs`: `1` only while your catalogue is small/data-scarce
     (this project's regime). Once you have "a large number of events"
     (user guide's own phrase, its own worked example uses ~10,000), switch
     back to `MINOBS ~= MINLNK` (e.g. 8) so ph2dt keeps only strongly-linked
     pairs rather than flooding hypoDD with weak ones.
   - `hypodd.clustering.obscc`/`obsct` and `ph2dt.minlnk`: the "8 = degrees
     of freedom" floor from the user guide is not itself a scale parameter,
     but a denser network can afford to raise it (more stable clusters) at
     the cost of potentially splitting into more/smaller clusters — watch
     `hypoDD.log`'s own "# clusters" / cluster-size report when tuning this.
4. **`DAMP` and `isolv`**: re-check `hypoDD.log`'s reported `CND` (condition
   number) after any change above — the user guide's target range is
   40-80; adjust `DAMP` up if higher, down if lower. Switch `isolv` from
   LSQR (`2`) to SVD (`1`) only if your event count stays under roughly
   100-200 events and you specifically want SVD's proper covariance
   estimates (LSQR's reported errors are "grossly under-estimated", per the
   user guide) — otherwise leave it at LSQR regardless of aperture, this is
   an event-count, not a distance-scale, decision.

## Air-quakes near a shallow datum or borehole reference

**What hypoDD means by "air-quake":** the user guide defines it exactly —
*"Air-quakes occur when event pairs locate near the surface, and the
control in vertical offset between the events is poor"* — and, in the
`IAQ` field description, *"Air-quakes are events that locate above the
model surface."* Concretely, hypoDD's 1D layered model (`IMOD=1`, this
project's default per `docs/ARCHITECTURE.md` §6c) is referenced to
**depth `z=0`, i.e. the top of its shallowest layer** — not literal ground
level. An air-quake is simply any event whose iterated hypocentre has
`z < 0` in that model frame. `IAQ=1` (this project's default) removes such
an event from the event list for all subsequent iterations and repeats the
last iteration; `IAQ=2` instead resets its depth to the last successful
iteration's value and keeps it in the solve.

**Why this matters more for a shallow/microseismic network than a regional
one:** a regional catalogue's events are typically many km deep relative to
its stations, so `z=0` (roughly sea level) is far from any real event depth
and an air-quake genuinely signals a location that has run away to an
unphysical result. A small-aperture microseismic array is different in two
related ways:

1. **The velocity model's `z=0` datum rarely coincides with true ground
   level.** This project's own layer-file convention
   (`docs/ARCHITECTURE.md` §6a) puts depth `0.0` at *sea level*, and its own
   synthetic reference network sits at 535-705 m elevation (station
   elevations, `tests/data/make_synthetic.py`'s `STATIONS` table) with
   events as shallow as 0.44 km depth (i.e. still ~130-260 m *below* true
   local ground level at that depth, but only a few hundred metres above
   the `z=0` datum). A real, physically unremarkable near-surface
   microseismic event can therefore sit much closer to hypoDD's `z=0`
   "air-quake" threshold than the same event would on a regional network —
   fewer km of headroom, so ordinary location noise (the exact "poor
   vertical control" the guide describes) is more likely to occasionally
   push an iteration's trial depth negative, even for a genuinely real,
   underground event.
2. **A borehole/downhole reference station makes this asymmetric, not
   symmetric.** This project's own station-elevation convention
   (`docs/ARCHITECTURE.md` §6, "Station depth": effective elevation =
   `elevation_m - depth_m`) gives a downhole sensor (`depth_m > 0`) a
   *negative* effective elevation — i.e. the station itself sits below the
   `z=0` datum, at the same or greater depth as some shallow events it
   monitors. hypoDD's own reader (`getdata.f:275-278`,
   `docs/ARCHITECTURE.md` §6c) silently clamps any negative station
   elevation to 0 unless `IMOD=5` — a station physically below the model's
   surface is treated by the ray tracer as if it sat exactly on that
   surface. Combined with air-quake removal, a genuinely well-located event
   near a borehole receiver's own depth can be geometrically indistinguishable,
   from the solver's perspective, from a location that has drifted above
   the true surface — both sit close to `z=0` in the model frame, and only
   the true (unknown) answer distinguishes them.

**Practical guidance:**

- Do not treat "some events removed as air-quakes" as automatically a bug
  or automatically evidence of an unstable network — check whether the
  removed events' *last successful* depth (visible in `hypoDD.log`'s
  per-iteration `DZ`/`AQ` columns, or by temporarily setting `iaq=2` and
  inspecting the reset depths in `hypoDD.reloc`) was plausible for your
  network (i.e. within a few hundred metres of `z=0`, consistent with a
  genuinely shallow source) rather than wildly negative (a real divergence).
- If air-quake removal is discarding events you have independent reason to
  believe are real and shallow (e.g. corroborated by NLLoc single-event
  locations, `catalog_mag.xml`'s pre-hypoDD origins), prefer `iaq=2` (keep,
  reset depth) over `iaq=1` (remove) for that run — you keep the event's
  horizontal double-difference information instead of losing it outright,
  at the cost of that event's depth being pinned rather than free to
  iterate further.
- If a large fraction of events are affected, first suspect **vertical
  control**, not `IAQ` itself, per the user guide: *"This problem can be
  addressed by increasing damping ... or by improving the geometrical
  control by including more stations close to the event pair. Also, a
  near-surface velocity that is higher than the true velocity pushes
  hypocentres up, while a lower than true velocity pushes hypocentres
  down."* For a borehole-referenced network specifically, check that the
  shallow layer velocities in your 1D model (`cfg.hypodd.velocity_model`)
  are not biased in a way that would systematically push events toward (or
  past) `z=0`.
- `IAQ=0` (`hypoDD.f:803`, functional but undocumented in the current user
  guide, `docs/hypoddpy_audit.md` defect #4) disables air-quake handling
  entirely. This project's config schema still permits it
  (`Literal[0, 1, 2]`) for a synthetic/test scenario whose brief
  mid-iteration negative-depth excursions would otherwise shrink a cluster
  below hypoDD's "less than 2 events" floor, but it is **not** the
  recommended default for a real deployment: it silently keeps events at
  physically impossible depths rather than surfacing the vertical-control
  problem described above.

## Troubleshooting

Read `hypoDD.log`'s own per-iteration table first (`EV`/`CT`/`CC` = percent
of events/catalog-links/cc-links still active, `AQ` = air-quakes removed
that iteration, `CND` = condition number) — it usually points straight at
which symptom below applies.

| Symptom | Likely cause | Parameter(s) to change |
|---|---|---|
| Most/all events end up removed as air-quakes by the final iteration | `WDCC`/`WDCT`'s final-row cutoff is tighter than your network's real inter-event spread, disconnecting events right when the schedule tightens most (this issue's own root cause) | Widen the final row(s)' `wdcc`/`wdct` (`cfg.hypodd.weighting_rows`); re-check against your own network's worst-case nearest-neighbour distance (see "Reference network"/"Scaling" above) |
| `ph2dt` produces very few/no linked pairs (`dt.ct` small or empty) | `maxsep_km` too tight for your event spread, or `minobs`/`minlnk` too strict for a data-scarce small network | Increase `ph2dt.maxsep_km`; lower `ph2dt.minobs` (down to 1); check `ph2dt.maxdist_km` isn't excluding real stations |
| `hypoDD` reports many small clusters instead of one connected network (`# clusters` > 1 in the log) | `OBSCC`/`OBSCT` requiring more links than the network's own connectivity supports, or `ph2dt`'s own `maxngh`/`minlnk` too restrictive to generate enough candidate links in the first place | Lower `hypodd.clustering.obscc`/`obsct` (keeping `obsct <= ph2dt.minlnk`); or loosen `ph2dt.maxsep_km`/`maxngh`/`minlnk` to generate more candidate links for hypoDD's clustering to use |
| `hypoDD.reloc` is empty or a whole cluster is dropped ("Cluster has less than 2 events") despite events reaching `hypoDD` | A cluster (often just 1-2 outlier events) loses all its links once the weighting schedule's distance cutoff tightens past their only real separation | Same fix as the first row above (widen final-row `WDCC`/`WDCT`); as a diagnostic, temporarily set `iaq=2` to see whether the events are being air-quake-removed vs. genuinely losing all data links |
| Condition number (`CND` in `hypoDD.log`) far outside ~40-80 | `DAMP` mismatched to the problem's own conditioning (user guide's own empirical range) | Increase `DAMP` if `CND` is high; decrease if `CND` is unusually low (over-damped, converging too slowly) |
| Relocated locations jump around between iterations / diverge instead of converging | Outlier residual cutoffs (`WRCC`/`WRCT`) too loose, letting bad data dominate early iterations; or `DAMP` too low | Tighten `WRCC`/`WRCT` in the early rows (still `>= 1` to keep the dynamic-cutoff behaviour); increase `DAMP` |
| Cross-correlation data appears to have no effect on final locations | `WTCCP`/`WTCCS` left near zero (or `-999`, "data not used") in the later rows where they should dominate; or `dt.cc` is empty/near-empty upstream | Check the final rows' `wtccp`/`wtccs` are non-trivial (this project's default schedule raises them to 1/0.5 from row 3 onward); check `paths.dt_cc` was actually built (`run_xcorr`) and is non-empty, and `cfg.hypodd.xcorr.min_cc` isn't excluding nearly everything |
| A genuinely nearby/regional station is diluting the local solution, or an irrelevant far station shows up in `station.sel` | `ph2dt.maxdist_km` / `hypodd.max_station_dist_km` too loose for your monitored volume | Tighten both toward ~1.5x your own network's aperture |
| Fine-scale structure (closely repeating events) not well resolved even though the overall relocation looks reasonable | Final-row `WDCC`/`WDCT` too loose for your actual (tighter-than-this-project's-default) event clustering | Tighten the final row(s)' `wdcc`/`wdct` toward `example1`'s own 1-2 km, but only once you've confirmed (per "Scaling" above) your events are tightly enough clustered to support it without losing connectivity |
| Real events near a downhole/borehole receiver's own depth are ambiguous with air-quakes | Model `z=0` datum does not coincide with true ground level, and/or the borehole station's own negative effective elevation puts it near shallow events in the model frame (see "Air-quakes" section) | Consider `iaq=2` for that run; check the shallow-layer velocities in `cfg.hypodd.velocity_model` are not biased toward pushing events past `z=0`; add closer stations for depth control if possible |
