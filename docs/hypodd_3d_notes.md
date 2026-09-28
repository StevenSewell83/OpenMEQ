# hypoDD 3D (IMOD=9) segfault investigation notes

OpenMEQ-6ld.6 (M9.6). Investigates the hypoDD 3D (SIMUL2000/`IMOD=9`)
ray-tracer segfault reported by the M9.4 worker (OpenMEQ-6ld.4) on a small
synthetic network, and either fixes it or validates inputs before running.
See `docs/ARCHITECTURE.md` §6c for the summarised contract this task adds
(`openmeq.relocate.hypodd_limits.check_vel3d_coverage`).

## Summary

The segfault is **not** an inherent limitation of hypoDD's `IMOD=9` ray
tracer on small-aperture networks. It is a genuine, unguarded
out-of-bounds array access in the vendored hypoDD 2.1 Fortran
(`src/hypoDD/ray_3d.f`'s `intmap` subroutine), triggered whenever an event
or station — once projected into the 3D model's own Cartesian frame —
falls outside the supplied `vel3D.dat`'s node domain. This happens
whenever hypoDD.inp's `LAT_3D LON_3D ROT_3D` doesn't actually match the
frame/extent the `vel3D.dat` file was built for (a registration mismatch),
which is exactly the M9 epic design note's leading hypothesis, not the
M9.4 worker's "pre-existing Fortran fragility" guess.

**A correctly registered and adequately-sized 3D model relocates the same
small-aperture (~15 km, 6 surface + 2 borehole stations) network
successfully**, with no crash (`tests/test_relocate_limits.py::
test_imod9_relocates_small_aperture_borehole_network`). The fix implemented
here is an input-validation function,
`openmeq.relocate.hypodd_limits.check_vel3d_coverage`, that catches the bad
case with a clear, actionable error before hypoDD ever runs.

## Reproduction

Never done against the main checkout's `external/HYPODD`/`external/bin` —
always a scratch copy.

1. Copy `external/HYPODD` to a scratch directory.
2. Rebuild `hypoDD`/`ph2dt` there with debugging flags:

   ```sh
   cd <scratch>/HYPODD/src
   make clean
   make FFLAGS="-g -O0 -fcheck=all -fbacktrace -I../../include"
   ```

   `-fcheck=all` turns an out-of-bounds Fortran array access into a clean
   runtime error with a `-fbacktrace` traceback (source file:line), instead
   of undefined behaviour — this is exactly what makes the crash easy to
   pin down; the *production* build (`install.sh`'s plain `-O`, no bounds
   checking) has no such protection, so the same input segfaults instead of
   erroring cleanly.

3. Build a small-aperture network: `tests/data/make_synthetic.py`'s own
   8-station geometry (`STATIONS`, 6 surface `HH?` + 2 borehole `DP?`,
   ~14.9 km aperture) plus a handful of well-separated synthetic events
   (offsets/depths in the same range as `EVENT_OFFSET_KM_RANGE`/
   `EVENT_DEPTH_KM_RANGE`), each with straight-ray P picks to every station.
4. Build a small NLLoc 3D grid covering the network with margin
   (`tests/helpers_nlloc3d.build_small_3d_velocity_grid`), convert it to a
   SIMUL2000 `vel3D.dat` (`openmeq.relocate.nlloc_to_simul.
   nlloc_grid_to_vel3d`), and write `station.dat`/`phase.dat`/`event.dat`/
   `hypoDD.inp` (`IMOD=9`) via `openmeq.relocate.hypodd_files`. Run `ph2dt`
   to get `dt.ct`.
5. Run the debug `hypoDD` binary against this input:
   - With `lat_3d`/`lon_3d` matching the `vel3D.dat`'s own frame (the frame
     `nlloc_grid_to_vel3d` built the model in): **no crash**, hypoDD
     relocates all events.
   - With `lat_3d`/`lon_3d` deliberately mismatched (`+5.0` degrees each —
     simulating a `vel3D.dat` copied in from a different survey/frame, or
     any other registration mistake): **reproduces the crash**, below.

## The crash (debug build, `-fcheck=all -fbacktrace`)

```
At line 586 of file ray_3d.f
Fortran runtime error: Index '-3388' of dimension 1 of array 'ixloc' below lower bound of 1

Error termination. Backtrace:
#3  0x... in intmap_
	at .../HYPODD/src/hypoDD/ray_3d.f:586
#4  0x... in vel3_
	at .../HYPODD/src/hypoDD/ray_3d.f:541
#5  0x... in ttime_3d_
	at .../HYPODD/src/hypoDD/ray_3d.f:307
#6  0x... in rayweb_
	at .../HYPODD/src/hypoDD/ray_3d.f:133
#7  0x... in path_
	at .../HYPODD/src/hypoDD/ray_3d.f:37
#8  0x... in partials_3d_
	at .../HYPODD/src/hypoDD/partials_3d.f:84
#9  0x... in hypodd
	at .../HYPODD/src/hypoDD/hypoDD.f:700
#10 0x... in main
	at .../HYPODD/src/hypoDD/hypoDD.f:1327
```

(Full absolute paths point at the scratch copy used during this
investigation; the source file:line citations above are what matters and
are stable across copies.)

Under the *production* build (no `-fcheck=all`), the exact same input
segfaults (`SIGSEGV`) instead of printing this message — the out-of-bounds
read/write into `ixloc`'s backing memory is undefined behaviour rather
than a caught error.

## Root cause

`src/hypoDD/ray_3d.f`'s `intmap` subroutine (called from `vel3`, itself
called from every point along every trial ray path the `IMOD=9` simulps
ray tracer builds — source-receiver endpoints, curved trial-path points,
and pseudo-bending perturbations):

```fortran
      subroutine intmap(x,y,z,ip,jp,kp)
c  Modified by W. Prothero so a single call can get the indices
c  common block variables:
      include 'vel3d.inc'
      ip=int((x+xl)/bld)
      ip=ixloc(ip)
      jp=int((yl+y)/bld)
      jp=iyloc(jp)
      kp=int((z+zl)/bld)
      kp=izloc(kp)
c  If an array element=0, the position is off the map.
      return
      end
```

`ip`/`jp`/`kp` are computed directly from the query point's raw Cartesian
coordinate, in units of `bld` km, and used **immediately** to index
`ixloc`/`iyloc`/`izloc` (`vel3d.inc`, each declared size 10000 —
`ixkms`/`iykms`/`izkms`) — with no check that the index is within
`[1, ixkms]` (or `[1, iykms]`/`[1, izkms]`) before the array read. The
comment "If an array element=0, the position is off the map" documents
that `bldmap` (which fills these lookup arrays, `get_vel3d.f`) deliberately
zero-fills entries for the physically-off-the-map range **within**
`[1, ixmax]..[ixmax, ixkms]`, but that is a separate concern from an index
that is itself `<1` or `>ixkms` — which is exactly what a registration
mismatch (or any other wildly-wrong Cartesian coordinate) produces.

**Which projection computes that Cartesian coordinate**: `hypoDD.f`'s
`imod.eq.9` branch (~lines 550–563) sets `sdc0_lat=lat_3d,
sdc0_lon=lon_3d` and calls `setorg(sdc0_lat, sdc0_lon, rot_3d, 0)`, then
every event's/station's position is `call SDC2(x, y, lat, lon, -1)`
(`partials_3d.f`), which dispatches to `dist.f` — the *ordinary* hypoDD
short-distance-conversion (SDC) machinery, the same one the 1D case uses.
**Not** `setorg_3d.f`/`disto.f`: despite the `_3d` name, those two
subroutines are dead code in this hypoDD build — `get_vel3d.f`'s own
`call setorg_3d(...)` is commented out, and grepping every call site in
`src/hypoDD/*.f` confirms `setorg_3d`/`disto` are never called from
anywhere.

`openmeq.relocate.hypodd_limits._hypodd_sdc_xy_km` is a direct Python port
of `setorg.f`/`dist.f` (not an approximation), empirically verified against
a real debug hypoDD run's own `hypoDD.loc` x/y output (agreement to about
1 m at a 6 km offset — the residual is single- vs. double-precision
rounding, not a formula discrepancy). Both NLLoc's `TRANS SIMPLE`/`SDC`
projection (`external/NonLinLoc/src/GridLib.c`'s `latlon2rect`) and
hypoDD's own `dist.f` use the same sign convention (x increases eastward, y
increases northward, rotation anticlockwise-positive) — despite `setorg.f`'s
own comment claiming "positive x-axis toward WEST", which is stale/
misleading relative to the actual formula once traced through with a
standard signed longitude input (confirmed empirically, not just read off
the comment). This means `openmeq.relocate.nlloc_to_simul`'s existing
`_check_coordinate_frame` (which cross-checks `rot_3d` against the source
NLLoc grid's `RotCW`, accounting for the CW/CCW sign flip) does not also
need an east/west axis mirror — there is none.

## Fix: `check_vel3d_coverage` (input validation, not a Fortran patch)

`openmeq.relocate.hypodd_limits.check_vel3d_coverage(vel3d_grid, lat_3d,
lon_3d, rot_3d, catalog, station_table)` projects every accepted event's
and every station's location into the model's frame via
`_hypodd_sdc_xy_km` and raises `openmeq.config.ConfigError` — naming every
offending station/event, which axis (x/y/z) it falls outside of, by how
much, and the domain it was checked against — if any of them fall outside
the `Vel3DGrid`'s own node domain (`xn[0]..xn[-1]`, `yn[0]..yn[-1]`,
`zn[0]..zn[-1]`), **before** hypoDD ever runs.

A station's z-coordinate is its ARCHITECTURE.md §6 effective elevation
(`elevation_m - depth_m`, km, negative for a downhole borehole sensor), but
clamped to `0.0` whenever negative — mirroring `getdata.f:275-278`'s own
silent clamp of a negative station elevation for any `IMOD` other than `5`
(which includes `IMOD=9`) — so the check validates the coordinate hypoDD
will *actually* query the model at, not the pre-clamp value (which
`openmeq.relocate.hypodd_files.write_station_dat` already logs a warning
for separately).

### Why input validation, not a Fortran source patch

Patching `intmap` to bounds-check `ip`/`jp`/`kp` before indexing (and fail
with a clear message, or clamp to the nearest legal index) was considered.
Rejected for this task because:

- The compiled `hypoDD`/`ph2dt` binaries in `external/bin` are shared build
  artifacts, built once by `install.sh` from `HYPODD.zip`
  (`docs/ARCHITECTURE.md`'s external-tools table). A source patch would
  need a `install.sh`-applied patch file to reach every environment
  (including CI and any existing installs), and would still leave the
  *underlying* registration mistake undetected until deep inside a hypoDD
  run.
- An input-validation check runs *before* hypoDD ever starts, with a far
  more actionable error message (naming the specific offending
  station/event and axis) than a Fortran `stop`/runtime error would, and
  needs no recompilation or patch-application step.
- This matches the issue's own guidance: "input problem: add validation
  ... that fails with a clear error before hypoDD runs".

If a genuine *unrelated* Fortran defect is found in a future task (e.g. one
that reproduces with a demonstrably well-registered, adequately-sized 3D
model), a minimal patch file applied by `install.sh` remains the right
mechanism — this task did not find that to be necessary here.

## What was ruled out

- **Small aperture / near-coincident hypocentres, as an inherent ray-tracer
  limitation**: ruled out. A correctly registered, adequately-sized 3D
  model relocates a real ~15 km-aperture, borehole+surface network with no
  crash (see the regression test below). The M9.4 worker's own tiny
  network reportedly crashed with *both* its own model and example1's —
  consistent with a registration mismatch in both cases (example1's own
  `vel3D.dat` is registered to example1's own `LAT_3D`/`LON_3D`, ~9 km from
  a differently-located small network would already be a mismatch relative
  to *that* network's own correct origin, unless the mismatch happened to
  still land inside example1's generous domain — the M9.4 report doesn't
  specify enough detail to reconstruct exactly which case was hit, but the
  mechanism identified here fully explains a segfault under a registration
  mismatch, and no other candidate mechanism reproduced).
- **`bld=0.1` scaling with small node spacing**: not implicated. The
  `ixkms`/`iykms`/`izkms` map-span limit (10000 `bld`-units) is already
  checked by `openmeq.relocate.nlloc_to_simul._check_map_span`, and this
  investigation's crash reproduced with a well-formed, correctly-sized
  small-aperture `vel3D.dat` — the trigger was the *registration*
  (`LAT_3D`/`LON_3D`), not the node grid's own internal spacing/scaling.
- **Station elevation/depth sign**: not implicated as a crash cause. The
  existing negative-effective-elevation clamp (`getdata.f:275-278`,
  documented in `docs/ARCHITECTURE.md` §6c) applies uniformly regardless of
  `IMOD`, and `check_vel3d_coverage` accounts for it (see above).
- **NLLoc vs. hypoDD axis convention (east/west mirror)**: checked and
  ruled out (see "Which projection computes that Cartesian coordinate"
  above) — both use the same east-positive/north-positive/
  anticlockwise-rotation convention; `setorg.f`'s own "positive x-axis
  toward WEST" comment is stale/misleading relative to the actual formula.
- **Events above the surface / `ixkms`/`iykms`/`izkms` overflow /
  array bounds in `ray_3d`/`get_vel3d` generally**: the specific bug found
  *is* an array-bounds issue in `ray_3d.f`, but the trigger is a
  registration mismatch, not events genuinely above the surface (z<0 is a
  legal, expected coordinate for a surface station and is well inside any
  reasonably-registered model's z-domain).

## Regression test

`tests/test_relocate_limits.py`:

- `test_hypodd_sdc_xy_km_matches_real_hypodd_output` — pins
  `_hypodd_sdc_xy_km` against a real debug hypoDD run's `hypoDD.loc`
  output.
- `test_check_vel3d_coverage_passes_for_covered_small_aperture_network`,
  `test_check_vel3d_coverage_raises_for_event_outside_domain`,
  `test_check_vel3d_coverage_raises_for_station_outside_domain`,
  `test_check_vel3d_coverage_clamps_negative_station_elevation_like_getdata`
  — fast unit tests for `check_vel3d_coverage` (no external tool).
- `test_imod9_relocates_small_aperture_borehole_network`
  (`requires_hypodd`, `slow`) — builds `tests/data/make_synthetic.py`'s own
  8-station (~14.9 km aperture, 2 borehole) network geometry with a handful
  of synthetic events, a correctly-registered small 3D model (via
  `openmeq.relocate.nlloc_to_simul`), and runs the *real* `ph2dt`+`hypoDD`
  binaries (`external/bin`, never a scratch/debug build) end-to-end with
  `IMOD=9`, asserting a complete relocation of every event with no crash.

## Follow-ups / contract gaps

- `check_vel3d_coverage` is a library function; wiring it into
  `scripts/setup_hypodd.py` (M9.5, out of scope for this task) should call
  it immediately after `check_compiled_limits` whenever
  `cfg.hypodd.velocity_model.type == "3d"`, before ph2dt/hypoDD run.
- `openmeq.relocate.nlloc_to_simul`'s own `_check_coordinate_frame` only
  validates the *building* of a `vel3D.dat` from an NLLoc grid; it does
  nothing for a user-supplied `vel3D.dat` (`cfg.hypodd.velocity_model.file`
  set directly, bypassing `nlloc_grid_to_vel3d`). `check_vel3d_coverage`
  covers this gap generically (it only needs the parsed `Vel3DGrid` header,
  not provenance), but a future task wiring the CLI together should make
  sure it runs on *every* 3D path, not only ones built via
  `nlloc_grid_to_vel3d`.
- This task did not attempt to bound the ray tracer's *mid-path* excursion
  beyond the literal source/receiver endpoints (the trial-curve dip vectors
  and pseudo-bending perturbations in `ray_3d.f`'s `rayweb`/`bend` can, in
  principle, probe slightly outside the straight source-receiver segment).
  `check_vel3d_coverage` checks event/station endpoints only, which is what
  this task's reproduction shows is sufficient to catch the failure mode
  found here; the `vel3D.dat`'s own "infinity boundary nodes" convention
  (`nlloc_to_simul`'s docstring, default `800 * bld` km margin) already
  provides a wide safety margin for mid-path excursions once endpoints
  themselves are correctly inside the domain. If a future task ever
  reproduces a crash with correctly-registered *endpoints* but a genuinely
  undersized/tight domain, that would be the next thing to investigate.
