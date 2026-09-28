# NonLinLoc interface notes (OpenMEQ-mev.1 / M5.1)

This is the spike for M5.1: verify the installed NonLinLoc build and its
control-file syntax empirically (the spec's own research could not reach
alomax.free.fr/alomax.net), so that M5.2 (`scripts/setup_nlloc.py`) and M6.1
(the per-event location runner) can be built against confirmed facts instead
of assumptions. Every claim below is checked against the actually-installed
`external/NonLinLoc` source/docs/samples and the actually-installed
`external/bin` binaries, and cited by file + line. Where the original spec's
"seed" notes turned out to be right, that's noted too; where they turned out
wrong, the correction is called out explicitly.

**No online docs were used.** `external/NonLinLoc/docs/programs/control.ControlFile.rst`
turned out to be a complete, local Sphinx-source copy of the same control-file
reference that alomax.net publishes online (it's built from the `docs/programs/`
tree into the project's Read the Docs site) -- so nothing here required network
access.

## 1. Installed version

Confirmed by running an installed binary directly (`external/bin/NLLoc`,
`external/bin/Vel2Grid`, `external/bin/Grid2Time`, `external/bin/Time2EQ` --
any NLLoc-family binary prints this banner when it actually runs a control
file; running with no arguments only prints a `Usage:` line, no banner):

```
NLLoc (NonLinLoc v7.1.05 18Apr2025)
```

Source of the version string: `external/NonLinLoc/src/include/GridLib.h:37-38`:

```c
#define PVER  "7.1.05"
#define PDATE "18Apr2025"
```

(A stray sibling file `external/NonLinLoc/src/include/GridLib-2024-10-07-08-50-02.h`
has an older `7.1.04` / `05Sep2024` -- not the one actually compiled into
`external/bin`, confirmed by the runtime banner above; it looks like a
leftover backup copy, not a second live header.)

There is no `.git` metadata inside `external/NonLinLoc` in this checkout (it's
vendored, not a submodule), so "commit 7a62fa4" from the worker instructions
could not be independently re-confirmed from within the tree; the PVER/PDATE
above is what actually matters for reproducing behaviour and is unambiguous.

## 2. What's actually in `external/NonLinLoc/doc/` and `docs/`

Two different directories, both present, format **not** consistent (confirms
the spec's warning that this wasn't guaranteed):

- `doc/` -- PDFs and OpenOffice docs: `1_NonLinLoc_Location_0.1.pdf`,
  `2_NonLinLoc_ControlFile_0.1.pdf`, `NLL_doc.odt`/`.pdf`, plus NLDiffLoc guides.
- `docs/` -- a Sphinx documentation source tree (`conf.py`, `index.rst`,
  `programs.rst`, `programs/*.rst`). **`docs/programs/control.ControlFile.rst`
  (1408 lines) is the authoritative, complete control-statement reference** --
  every `KEYWORD` syntax, parameter, type, min/max and default in this
  document is cited from there unless stated otherwise. There's also a stale
  `docs/build/doctrees` (Sphinx's binary doctree cache from a previous local
  build), not source, ignored.

## 3. Sample control files used as worked examples

- `external/NonLinLoc/nlloc_sample/run/nlloc_sample.in` -- a full, real
  1D-layered-model + Grid2Time + Time2EQ + NLLoc control file (Alaska
  Mww7.0 example), plus `nlloc_sample/obs/station_coordinates.txt` (74
  `GTSRCE` lines) and `nlloc_sample/run_nlloc_sample.sh`.
- Ran this sample end to end with the installed binaries (`Vel2Grid`,
  `Grid2Time`, `Time2EQ`, `NLLoc`) as an initial sanity check before building
  the hand-written minimal example for the pytest spike test. It produced 7
  located events; `obspy.read_events(..., format="NLLOC_HYP")` read the first
  one back with a depth of 44.937 km against the `EQSRCE` truth of
  44.998047 km -- confirms the whole toolchain works before layering on the
  project's own 1D layer format and a borehole station (§4 below).

## 4. Control statement reference (verbatim from the installed docs)

All line numbers below are in
`external/NonLinLoc/docs/programs/control.ControlFile.rst`.

### Generic statements

**`CONTROL messageFlag randomNumberSeed`** (line 83-95, *required*)
`messageFlag` (int, min -1, default 1): verbosity (-1 silent .. 2+ verbose).
`randomNumberSeed` (int): seed for Metropolis/Time2EQ noise. Example used:
`CONTROL 1 54321`.

**`TRANS ...`** (line 96-165, *required*). Seven syntaxes:

1. `TRANS GLOBAL`
2. `TRANS SIMPLE latOrig longOrig rotAngle`
3. `TRANS NONE`
4. `TRANS SDC latOrig longOrig rotAngle`
5. `TRANS LAMBERT refEllipsoid latOrig longOrig firstStdParal secondStdParal rotAngle`
6. `TRANS TRANS_MERC refEllipsoid latOrig longOrig rotAngle [useFalseEasting] [falseEasting] [mapScaleFactor]`
7. `TRANS AZIMUTHAL_EQUIDIST refEllipsoid latOrig longOrig rotAngle`

`SIMPLE`'s algorithm is given verbatim (line 151-153, Note 3):
```
x = (long - longOrig) * 111.111 * cos(lat_radians);
y = (lat - latOrig) * 111.111;
lat = latOrig + y / 111.111;
long = longOrig + x / (111.111 * cos(lat_radians));
```
i.e. `SIMPLE` corrects only longitudinal distance for latitude, using a fixed
111.111 km/deg (not a real ellipsoid) -- fine for small regional networks, not
appropriate at high latitude over large distances (that's what `LAMBERT` /
`TRANS_MERC` / `AZIMUTHAL_EQUIDIST` are for, and why the nlloc_sample Alaska
example -- 60N -- uses `LAMBERT`, not `SIMPLE`).
`refEllipsoid` choices (line 119-121): `WGS-84 GRS-80 WGS-72 Australian
Krasovsky International Hayford-1909 Clarke-1880 Clarke-1866 Airy Bessel
Hayford-1830 Sphere`.
`rotAngle` is clockwise-from-Y-axis in degrees (line 116-118, 139-144).
`TRANS` (and `MAPLINE`) must be identical across every control file used for
one project (line 66-69) -- important since M5.2/M6.1 will generate separate
control files per program invocation.

### Vel2Grid statements

**`VGOUT fileRoot`** (227-238, *required*). Output is
`fileRoot.waveType.mod.{buf,hdr}`.

**`VGTYPE waveType`** (239-244, *required, repeatable*): `P` or `S`, one
statement per wave type.

**`VGGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType`** (245-269,
*required*). For a 1D model use a 2D grid: `xNum=2`, `xOrig=yOrig=0.0` (line
267-269) -- verified empirically (§6): `Vel2Grid` ran fine with
`VGGRID 2 401 161 0.0 0.0 -3.0 0.05 0.05 0.05 SLOW_LEN`. `gridType` choices
(256-261): `VELOCITY VELOCITY_METERS SLOWNESS VEL2 SLOW2 SLOW_2_METERS
SLOW_LEN`; the bundled sample and Podvin-Lecomte (`GT_PLFD`) both use
`SLOW_LEN`.

**`LAYER depth VpTop VpGrad VsTop VsGrad rhoTop rhoGrad`** (271-286,
*optional, repeatable*). This is exactly ARCHITECTURE.md §6a's 1D layer file
shape (`depth_top_km vp_km_s vs_km_s rho_g_cc`) plus a zero gradient per
layer -- **confirms ARCHITECTURE.md §6a's file maps 1:1 onto repeated `LAYER`
statements with `*Grad` fields all `0`.** Layers must be in increasing-depth
order (Note 1); the deepest layer's top extends to infinite depth (Note 2).
Verified empirically that the *shallowest* layer's velocity likewise extends
upward with no error for grid nodes above its top depth (the ARCHITECTURE.md
§6a layer file's first entry is `depth_top_km=0.0`, but the grid used here
spans `z` from -3.0 to +5.0 km; `Vel2Grid` produced grid files with no
complaint) -- so a negative `VGGRID`/`LOCGRID` `zOrig` does **not** require an
extra negative-depth `LAYER` line.

### Grid2Time statements

**`GTFILES ttimeFileRoot outputFileRoot waveType iSwapBytesOnInput`**
(385-413, *required*). One invocation of `Grid2Time` computes **one** wave
type; P and S each need their own control file/invocation (confirmed: ran
`Grid2Time` twice, once per wave type, differing only in this line's
`waveType` field).

**`GTMODE gridMode angleMode`** (415-427, *required*).
`gridMode`: `GRID3D` or `GRID2D` (2D, i.e. depth+radial-distance, radially
symmetric per source -- this is what a 1D velocity model implies, and what
the bundled sample and this spike both use). `angleMode`: `ANGLES_YES
ANGLES_NO ANGLES_INCLINATION`.

**`GTSRCE label <XYZ|LATLON|LATLONDM|LATLONDS> ...`** (429-461, *required,
repeatable*) -- **this is the critical statement for the borehole-elevation
question (work item #4).** For `LATLON`: `GTSRCE label LATLON latSrce
longSrce zSrce elev`; for `XYZ`: `GTSRCE label XYZ xSrce ySrce zSrce elev`.
Per the docs: `zSrce` is "z grid position (depth, positive DOWN) in
kilometers", `elev` is "elevation above z grid position (positive UP) in
kilometers". Nothing in the prose states a sign restriction on `elev`. See
§5 below for the source-code confirmation and the empirical run.

**`GT_PLFD hs_eps_init message_flag`** (462-478, *required for Podvin &
Lecomte FD, must be absent otherwise*). Used throughout this spike:
`GT_PLFD 1.0e-3 0`.

### Time2EQ statements (used here only to generate ground-truth picks for testing, out of production scope)

**`EQFILES ttimeFileRoot outputFileName`** (485-499). **`EQSRCE label
LATLON lat lon zEvent elev`** (552-582, same `GTSRCE` semantics). **`EQSTA
label phase errorType error errorReportType errorReport probActive`**
(514-550) -- `errorType GAU`, `error 0.00` gives exact (noise-free) synthetic
arrival times, used to build the test's ground truth. **`EQVPVS VpVsRatio`**
(634-646): negative value forces Time2EQ to read the real S time grids
(rather than deriving S from a fixed Vp/Vs ratio on the P grid) -- used here
so the test genuinely exercises the S grids Grid2Time built from `vs`, not an
approximation. **`EQQUAL2ERR`** (621-632, *required*) is for HYPO71-style
output only but the program still requires the statement to be present.

### NLLoc statements

**`LOCFILES obsFiles obsFileType ttimeFileRoot outputFileRoot iSwapBytes`**
(679-701, *required*). `obsFileType` includes `NLLOC_OBS` -- the format
`obspy.io.nlloc.core.write_nlloc_obs` produces (§5).

**`LOCHYPOUT fileType1 ...`** (703-739, *optional*, default
`SAVE_NLLOC_ALL SAVE_HYPOINVERSE_Y2000_ARC`). **`SAVE_NLLOC_ALL` means "save
summary AND event files"** (line 710-712) -- i.e. it is not just the per-event
files. Empirically confirmed output filenames (running both the bundled
`nlloc_sample` and this spike's own minimal example; the exact template is
not spelled out as a single line anywhere in the control-file reference, so
this section is sourced from the actual files NLLoc wrote, not a doc quote):

- Per-event: `<outputFileRoot>.<YYYYMMDD>.<HHMMSS>.grid0.loc.hyp` (+ matching
  `.hdr` and, for `LOCSEARCH OCT` with `LOCGRID ... SAVE`, `.scat`). The
  `<YYYYMMDD>.<HHMMSS>` is the **earliest observation time in the input phase
  file for that event**, to integer-second precision by default (`.190000.`
  style) -- `LOCHYPOUT FILENAME_DEC_SEC` switches to 2-decimal-second
  filenames instead, "to avoid overwriting of output files for multiple
  events ... with earliest observation time in same second" (line 717-719).
  **M6.1 must not guess this pattern from `outputFileRoot` alone: it has to
  glob `<outputFileRoot>.*.*.grid0.loc.hyp`** (or, more robustly, read
  NLLoc's own stdout line `Finished location: <path>`, which `openmeq.external.run`
  already captures) to find the file it just wrote, matching what this spike's
  test does.
- Run-wide summary (all located events from that run, appended across
  multiple `NLLoc` invocations against the same `outputFileRoot` if reused):
  `<outputFileRoot>.sum.grid0.loc.hyp` (+ `.hdr/.stat/.stations/.stat_totcorr/.csv`,
  and `.hypo_inv` if `SAVE_HYPOINV_SUM`/`SAVE_HYPOINVERSE_Y2000_ARC` requested).
- Convenience copy of the most recent event: `<dir>/last.hyp` (+ same
  siblings), independent of `outputFileRoot`'s basename.
- `is_nlloc_hyp` (obspy, §5) only checks the file starts with `b'NLLOC '`, so
  any of the `.hyp` files above round-trip through `obspy.read_events(...,
  format="NLLOC_HYP")` --- confirmed for the per-event file in this spike's test.

**`LOCSEARCH OCT initNumCells_x initNumCells_y initNumCells_z minNodeSize
maxNumNodes numScatter useStationsDensity stopOnMinNodeSize`** (741-816,
*required*). This is the statement `openmeq.config.NllocLocSearchConfig`
models. **The existing schema's field names, order and types were already
correct** (`init_num_cells_x/y/z: int`, `min_node_size: float`,
`max_num_nodes: int`, `num_scatter: int`, `use_stations_density: bool`
(NLLoc-side 0/1), `stop_on_min_node_size: bool` (NLLoc-side 0/1, default 1)) --
no schema change needed here, just removed the TODO now that it's verified.
Note for M6.1's control-file writer: NLLoc's `useStationsDensity` and
`stopOnMinNodeSize` are literal `0`/`1` integers on the line, so a pydantic
`bool` must be rendered as `1 if v else 0`, not Python's `True`/`False`.

**`LOCMETH method maxDistStaGrid minNumberPhases maxNumberPhases
minNumberSphases VpVsRatio maxNum3DGridMemory minDistStaGrid
iRejectDuplicateArrivals`** (818-872, *required*). `method` one of
`GAU_ANALYTIC EDT EDT_OT_WT EDT_OT_WT_ML`. Used in this spike:
`LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1.0 6 -1.0 1` (`VpVsRatio=-1.0` so real S
grids are used, matching the `EQVPVS` choice above). Not currently modelled
in `openmeq.config` (`NllocConfig` has no `locmeth` field) -- flagged as a
follow-up for M6.1 (§8).

**`LOCGAU SigmaTime CorrLen`** (874-892, *required*). Matches
`openmeq.config.NllocLocGauConfig` (`sigma_time_s`, `corr_len_km`) exactly --
no change needed.

**`LOCGAU2 SigmaTfraction SigmaTmin SigmaTmax`** (893-912, *optional*).
**Correction:** the pre-existing schema (`NllocLocGau2Config`) had a third
field `corr_len_km`, but per the docs `LOCGAU2` takes exactly three
parameters and none of them is a correlation length (unlike `LOCGAU`) --
the third parameter is `SigmaTmax`, "maximum trave-time error in seconds"
(line 911-912). Fixed in this commit: `corr_len_km` -> `sigma_time_max_s`
(default `2.0`, matching the bundled sample's `LOCGAU2 0.02 0.05 2.0`).

**`LOCPHASEID stdPhase phaseCode1 ...`** (914-932, *optional, repeatable*).
Used as `LOCPHASEID P P` / `LOCPHASEID S S` in this spike (the obspy-written
obs file already uses bare `P`/`S`).

**`LOCQUAL2ERR Err0 ... ErrN`** (934-952). **Critical correction for
`openmeq.config.NllocConfig.locqual2err_thresholds_s`**: the statement is,
verbatim, *"required, non-repeatable, for phase/observation file formats that
do not include time uncertainties ; ignored, non-repeatable, otherwise"*
(line 935-937). OpenMEQ's NLLoc runner (M6.1) will always use
`LOCFILES ... NLLOC_OBS` written by `Catalog.write(..., format="NLLOC_OBS")`,
and that writer *always* emits a literal numeric `GAU <uncertainty>` field
per pick (confirmed in §5, and in the format string at
`obspy/io/nlloc/core.py:464`) -- **so for every observation file OpenMEQ ever
hands to NLLoc, `LOCQUAL2ERR`'s values are ignored**; NLLoc uses the picks'
own per-arrival uncertainty directly. The statement must still be present
(NLLoc calls it "required" and the whole run aborts if a required statement
is missing -- confirmed by the "required" wording and by every worked control
file, including the bundled sample, including one). Fixed in this commit:
kept the field (M6.1 still must emit *some* syntactically valid ascending
list) but corrected the docstring, and appended the conventional large
sentinel value the NonLinLoc/`EQQUAL2ERR` samples use for a "reject this
phase" bucket (`99999.9`) to the default list for consistency, even though
functionally it will never be read.

**`LOCGRID xNum yNum zNum xOrig yOrig zOrig dx dy dz gridType saveFlag`**
(954-1005, *required, repeatable*). **Empirically-discovered constraint not
stated as a single explicit rule in the prose**: NLLoc rejects every
observation whose station's 2D time grid doesn't vertically span the initial
`LOCGRID`'s full `z` range, with the exact message `"depth range of initial 3D
location search grid exceeds that of 2D time grid size, rejecting
observation: <STA> <PHASE>"`. Hit this directly while building the spike's
mini example (`LOCGRID` z-range `-1.0..7.0` vs the velocity/time grid's
`-3.0..5.0`) -- every one of 10 arrivals was silently rejected and NLLoc
reported `"1 events read, 0 events located"` with **no error exit code**.
**Flag for M5.2/M6.1**: the location search grid's z-range (and, in `GRID3D`
mode, x/y range) must be validated against the `VGGRID`/`Grid2Time` grid's
coverage *before* running `NLLoc`, or a misconfigured grid will silently
produce zero locations rather than a clear failure.

**`LOCANGLES angleMode qualtiyMin`** (1039-1052, *optional*, default
`ANGLES_YES 5`). Not exercised for output in this spike (`ANGLES_NO`, since
`GTMODE ... ANGLES_NO` was also used and no take-off angles were generated) --
flagged as a follow-up for M6.1 if per-arrival take-off angles are needed.

**`LOCMAG`** (1054-1085, *optional*) -- not used by the location step; ML is
computed separately by OpenMEQ's own magnitude step (ARCHITECTURE.md §… Step
7), not by NLLoc's built-in magnitude calculator. Not modelled in
`openmeq.config.NllocConfig` and not needed there.

## 5. GTSRCE and the borehole / negative-elevation question (work item #4)

**Finding: yes, `GTSRCE`'s `elev` parameter accepts negative values, and
this is exactly the mechanism ARCHITECTURE.md §6 needs. No ARCHITECTURE.md
change is required.**

Source proof (`external/NonLinLoc/src/GridLib.c`, function `GetSource`):
for every coordinate syntax, the source's working depth is computed as
`zSrce - elev`, with **no `checkRangeDouble` call on `elev`** (range checks
are only applied to `Lat`/`Long`, see e.g. lines 358-364):

- XYZ: `srce_in->z = val3 - val4;` (line 343)
- LATLON: `srce_in->depth = val3 - val4;` (line 360)
- LATLONDM: `srce_in->depth = val3 - val4;` (line 392)
- LATLONDS: `srce_in->depth = val3 - val4;` (line 425)

where `val3` is `zSrce` and `val4` is `elev`. So **ARCHITECTURE.md §6's rule
maps directly onto `GTSRCE`'s `elev` field**: set `zSrce = 0.0` and
`elev = (elevation_m - depth_m) / 1000` (km); a borehole sensor
(`depth_m > 0`, effective elevation negative) simply gets a negative `elev`,
which `sscanf`s as an ordinary negative float and is never range-checked.

Empirically verified (this spike's test, `tests/test_nlloc_spike.py`) with a
station `BH1` at `elev = -0.5` (a 500 m-deep borehole at 0 m surface
elevation): `Grid2Time` printed

```
Calculating travel times for source: BH1  X 0.3931  Y 0.5560  Z 0.5000 (lat/lon/depth  45.005000  10.005000  0.500000) ...
Finished calculation, time grid output files: ./time/vel1d.P.BH1.*
```

i.e. `Z = zSrce - elev = 0.0 - (-0.5) = 0.5` km (below the sea-level datum,
positive-down convention) -- exactly as predicted from the source, ran with
no warning/error, and produced a travel-time grid used successfully by
`NLLoc` (BH1's synthetic P arrival, 0.596 s, was in fact the *shortest* of
the five stations, consistent with it being both shallow and close to the
true epicenter -- a physically sane result, not just "didn't crash").

## 6. The end-to-end worked example (work item #6)

Built entirely by hand (literal control-file text and `subprocess`, no
OpenMEQ NLLoc-setup code) in `tests/test_nlloc_spike.py`
(`test_nlloc_end_to_end_recovers_hypocenter`, marked `requires_nlloc`):

1. A 3-layer 1D model in the exact ARCHITECTURE.md §6a text format ->
   `Vel2Grid` (2D grid, `SLOW_LEN`, per §4 above).
2. `Grid2Time` for P and S, run separately, for 5 stations: 4 surface + 1
   borehole (`BH1`, `elev=-0.5` km, §5 above).
3. `Time2EQ` against a known hypocenter (`lat 45.008, lon 10.008, depth 3.0
   km`) with `EQSTA ... GAU 0.00 ...` (zero-noise synthetic arrivals) to get
   ground-truth P/S travel times self-consistent with the same grids NLLoc
   will invert.
4. Those times, as absolute UTC picks, written to an `NLLOC_OBS` file via
   `obspy.core.event.Catalog.write(..., format="NLLOC_OBS")` (exercises the
   real writer, not a hand-written obs file).
5. `NLLoc` against that file (`LOCMETH EDT_OT_WT`, `LOCSEARCH OCT`).
6. The resulting `.hyp` file read back with
   `obspy.read_events(..., format="NLLOC_HYP")`, including the `picks=`
   round trip (§7).

Result: recovered epicenter within **32 m** horizontally and **31 m** in
depth of the true hypocenter (test asserts `< 5 km`, so this passes with
large margin -- the tight recovery reflects the noise-free synthetic picks,
not an assertion that NLLoc is this precise on real data).

One dead end worth recording: an earlier version of the test used
`1900-01-01T00:00:00` as the reference origin time (matching the bundled
sample's Time2EQ default epoch). NLLoc's location for that run printed odd
internal debug output (`DEBUG: reset_hypodatetime: phypo time in:
1900-01-01T00:00:-0.005130` / `phypo time out: 1970-01-01T00:59:58.994870`),
though the actual written `.hyp` file's origin time was still correct to
~5 ms. Switching the reference time to a realistic date
(`2025-06-15T03:00:00`) made the debug print disappear entirely (same ~5 ms
search-precision residual, no anomalous branch). Not investigated further
since no real OpenMEQ event will ever use a 1900 timestamp, but noted here in
case a future test happens to pick an origin time near the epoch and sees
strange debug output.

## 7. obspy NLLOC_OBS / NLLOC_HYP, confirmed against the installed version

Installed: **obspy 1.5.1** (`obspy/io/nlloc/core.py` in this worktree's
`.venv`). All of the spec's seed claims about this module were **confirmed
correct** for this version:

- `write_nlloc_obs(catalog, filename)` (core.py:428-505):
  - Raises `ValueError` if `len(catalog) > 1` (lines 453-457): "only
    supported for Catalogs with a single Event".
  - Writes **every** pick in `catalog[0].picks` (loop at line 465) --
    no filtering by `evaluation_status`/rejection at all. **OpenMEQ's NLLoc
    runner (M6.1) must filter to non-rejected picks itself before calling
    this** (e.g. via `openmeq.catalog.accepted`-style logic on the picks, not
    just the event), or rejected picks will silently be sent to NLLoc.
  - Uses only `wid.station_code` (line 467), **never** the network code --
    confirmed by direct test: writing a pick with `network_code="XX"`
    produced a line starting `BH1    ?    ...` (network dropped entirely,
    `?` in that column is a fixed placeholder, not the network). **Station
    codes must be unique across the whole deployment** for OpenMEQ's
    station-naming scheme, exactly as the spec's seed research said.
  - Always writes a literal `GAU <uncertainty>` error field (format string at
    line 464); if `pick.time_errors.uncertainty` is `None` and there are no
    upper/lower uncertainties either, it still writes and warns, using `0.0`
    (lines 479-486) rather than raising. Every OpenMEQ pick must carry
    `time_errors.uncertainty` for this to be meaningful (PhaseNet picks
    already do, per ARCHITECTURE.md §5).

- `read_nlloc_hyp(filename, picks=None, ...)` (core.py:47-425):
  - Without `picks=`, reconstructs "the rather basic" `Pick` objects from the
    `.hyp` file's `PHASE` block (station, channel, phase, time, uncertainty,
    onset, polarity) -- confirmed: reading this spike's `.hyp` file with no
    `picks=` gave 10 picks/10 arrivals, matching the 10 written.
  - **With `picks=original_picks`**, re-links each arrival to the matching
    original `Pick` by `station_code` + `phase_hint`, **first match wins**
    (core.py:401-406, `for pick_ in original_picks: if station ==
    wid.station_code and phase == pick_.phase_hint: pick = pick_; break`) --
    confirmed by test: passing the exact 10 original `Pick` objects back in
    gives `event.picks` containing those same Python objects (`pick in
    original_picks`), not reconstructed copies. **Confirms the spec's seed
    claim exactly, including the "first match wins" duplicate-station/phase
    hazard**: if a catalog ever has two picks with the same `(station_code,
    phase_hint)` (e.g. two location passes' picks not yet disambiguated), the
    wrong one can silently get linked. M6.1's per-event runner should pass
    only the picks actually used in that specific NLLoc run (already the
    natural thing to do), not a whole catalog's history.
  - `is_nlloc_hyp` (core.py:31-42) just checks the file starts with `b'NLLOC '`.

## 8. Follow-ups (file separately, not done here -- out of scope for M5.1)

- **M6.1**: no `locmeth` config section exists yet in `openmeq.config`
  (`LOCMETH method maxDistStaGrid minNumberPhases maxNumberPhases
  minNumberSphases VpVsRatio maxNum3DGridMemory minDistStaGrid
  iRejectDuplicateArrivals`, §4 above) -- needed before the per-event runner
  can be written.
- **M6.1**: must filter to non-rejected picks before
  `Catalog.write(..., format="NLLOC_OBS")` (§7); `write_nlloc_obs` does not
  do this itself.
- **M6.1**: must pass only that event's own picks as `picks=` to
  `read_nlloc_hyp`, and must render `NllocLocSearchConfig`'s
  `use_stations_density`/`stop_on_min_node_size` bools as `1`/`0` literals,
  not `True`/`False` (§4 above).
- **M6.1**: must locate the per-event `.hyp` file NLLoc actually wrote by
  globbing `<outputFileRoot>.*.*.grid0.loc.hyp` (or capturing NLLoc's
  `Finished location: <path>` stdout line), not by guessing a fixed suffix
  (§4 `LOCHYPOUT` above).
- **M5.2/M6.1**: validate the location search grid's spatial coverage against
  the travel-time grid's coverage before running `NLLoc` -- a mismatch is
  silently swallowed as "0 events located" with exit code 0, not an error
  (§4 `LOCGRID` above).
- Station naming: confirm at the OpenMEQ station-table / config level (not
  part of this spike) that station codes really are unique network-wide, per
  §7 above -- this is an implicit requirement on the whole project's station
  naming, not just NLLoc's config, per the issue's design notes.

## 9. Config schema corrections made in this commit

`openmeq/config.py` and `config/example.yaml`:

- `NllocLocGau2Config`: renamed `corr_len_km` -> `sigma_time_max_s` (default
  `2.0`); `LOCGAU2` has no correlation-length parameter (§4).
- `NllocConfig.locqual2err_thresholds_s`: default extended with a trailing
  `99999.9` sentinel bucket, and docstring corrected to explain the statement
  is ignored for `NLLOC_OBS` input (§4/§7).
- `NllocLocSearchConfig` / `NllocLocGauConfig`: field names/order/types were
  already correct; only the stale `TODO(OpenMEQ-mev.1)` / `TODO(OpenMEQ-xdf.1)`
  docstring notes were replaced with the verified citation.
- `tests/test_config.py`: updated the two assertions on the changed
  `locgau2`/`locqual2err_thresholds_s` defaults.
