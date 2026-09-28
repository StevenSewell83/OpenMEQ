# ObsPyck spike notes (OpenMEQ-c3l.1)

Plan key M7.1. This is the empirical spike required before M7.2 (config
generator) and M7.3 (batch review loop) are built. Everything below was
actually run, not inferred from source, under
`QT_QPA_PLATFORM=offscreen` (and cross-checked once with `xvfb-run`) in
this project's own venv. Screenshots:
`docs/images/obspyck_spike_waveform_picks.png` (waveform view, automatic
P/S picks preloaded) and `docs/images/obspyck_spike_nlloc_done.png` (map
view after "do NLLoc", scatter cloud + relocated hypocentre). Full terminal
transcripts for every scenario below are reproduced inline.

**Deviation from the task's step 3:** `tests/conftest.py`'s
`synthetic_project` fixture (OpenMEQ-qlg.8) had not landed yet when this
spike ran (in progress in a sibling worktree). This spike built its own
throwaway local event (2-3 stations, hand-crafted mseed/StationXML/QuakeML)
following the same conventions described in `docs/ARCHITECTURE.md` (event
id format §4.4, QuakeML conventions §5, `NET.STA` station keys §6) instead.
`scripts/obspyck_spike_smoke.py` (the required smoke script) builds its own
such fixture inline for the same reason, and should be pointed at the real
fixture once M0.8/OpenMEQ-qlg.8 lands, to confirm nothing about the real
dataset's conventions changes these findings.

## 1. Launch confirmed, offscreen and under Xvfb

`obspyck` (console-script entry point installed by `pip install -e ".[dev]"`
per M0.2/install.sh) imports and launches cleanly under
`QT_QPA_PLATFORM=offscreen`, no Xvfb required. Cross-checked once with
`xvfb-run -a` (unsetting `QT_QPA_PLATFORM` first) against the same
fixture/config: identical result (same NLLoc origin, screenshot capture
succeeded the same way). **Recommendation for M11/CI: use
`QT_QPA_PLATFORM=offscreen` rather than `xvfb-run`** — it needs no extra
system package (Xvfb was not installed on this dev machine by default; had
to `apt-get install xvfb` to even test the comparison) and gave bit-identical
results in this spike.

Platform tag actually computed by this Python/Qt build (util.py's
`setup_external_programs`, `platform.system()` / `platform.architecture()`):

```
$ python3 -c "import platform; print(platform.system(), platform.architecture())"
Linux ('64bit', 'ELF')
```

So the required NLLoc binary name is exactly **`NLLoc__Linux__64bit`**
(`"__".join(["NLLoc", "Linux", "64bit"])`), matching the design doc's
prediction.

## 2. Exact launch command line used

```
obspyck -c obspyck.cfg -t 2026-01-01T00:00:00 -d 30 \
    --event ev20260101T000010_100.xml \
    ev20260101T000010_100.mseed stations.xml
```

Run with `cwd` = a fresh empty per-event working directory (confirmed
required — see §5). `QT_QPA_PLATFORM=offscreen` and
`OPENMEQ_EXTERNAL_DIR=<repo>/external` set in the environment.

## 3. Exact minimal config file used

Confirms the M7 design doc's "config sections required" list exactly: all
of `[base]`, `[misc]`, `[seismic_phases]`, `[seed_id_lookup]` (empty),
`[gui_defaults]`, `[keys]` (copied verbatim from obspyck's `example.cfg`),
`[nonlinloc]`, `[matplotlibrc]`, `[ar_picker]` were needed; no
`[station_combinations]` or `server_*` sections. Full file (also produced
programmatically by `scripts/obspyck_spike_smoke.py`'s `_write_config`):

```ini
[base]
duration = 30
starttime_offset = 0
pluginpath = /abs/path/to/pluginpath
merge =
event_server =
ignore_gse2_chksum_error = true
verbosity = normal
zero_mean = true
normalization = true
no_metadata = false
spectrogram_colormap = viridis
magnitude_pick_color = green
magnitude_picker_width = 10
physical_units = velocity
taper_max_length = 5
taper_max_percentage = 0.05
taper_type = cosine

[misc]
allow_multiple_picks_with_same_seed_id = false
scrollWheelPercentage = 0.30
scrollWheelInvert = false
merge_events_in_catalog = false

[seismic_phases]
P = red
S = blue

[seed_id_lookup]

[gui_defaults]
filter = false
filter_type = bandpass
filter_highpass = 0.01
filter_lowpass = 0.05
filter_corners = 3
sta = 0.5
lta = 10.0
ar_f1 = 1.0
ar_f2 = 20.0
ar_sta_p = 0.1
ar_lta_p = 1.0
ar_sta_s = 1.0
ar_lta_s = 4.0
ar_m_p = 2
ar_m_s = 8
ar_l_p = 0.1
ar_l_s = 0.2
spectrogram_wlen = 0.4
spectrogram_overlap = 0.9
spectrogram_log = false

[keys]
setPick = a
setPickError = s
delPick = q
setMagMin = a
setMagMax = s
delMagMinMax = q
switchPhase = control
prevStream = y
nextStream = x
switchWheelZoomAxis = shift
scrollWheelZoom = alt
setWeight0 = 0
setWeight1 = 1
setWeight2 = 2
setWeight3 = 3
setPolU = +
setPolD = -
setOnsetI = i
setOnsetE = e

[nonlinloc]
default_pick_uncertainty = 0.05

[matplotlibrc]
lines.linewidth = 1.0
font.size = 10

[ar_picker]
f1 = 1.0
f2 = 20.0
lta_p = 1.0
sta_p = 0.1
lta_s = 4.0
sta_s = 1.0
m_p = 2
m_s = 8
l_p = 0.1
l_s = 0.2
```

## 4. Pitfall not in the original design doc: `pluginpath` needs `hyp_2000/` and `focmec/` too

`setup_external_programs` (util.py:790 loop over `PROGRAMS.items()`) does
`shutil.copytree(pluginpath/<prog_basename>, ...)` **unconditionally for
every entry in `PROGRAMS`** (`nlloc`, `hyp_2000`, `focmec`), with no
try/except. If `pluginpath/hyp_2000/` or `pluginpath/focmec/` don't exist,
`ObsPyck.__init__` raises `FileNotFoundError` and the GUI never
appears — even though only NLLoc is used. **M7.2's config generator must
create empty `hyp_2000/` and `focmec/` directories alongside `nlloc/` in
the generated pluginpath**, not just the `nlloc/` dir the design doc
describes.

## 5. Exact pluginpath / nlloc / VELOCITY_MODELS layout used

```
pluginpath/
  nlloc/
    NLLoc__Linux__64bit       # copy of external/bin/NLLoc, chmod +x
    locate_BY.nlloc           # NLLoc control file, see below
  hyp_2000/                  # empty — see pitfall above
  focmec/                    # empty — see pitfall above
  VELOCITY_MODELS/
    layer_time/
      layer.P.mod.{hdr,buf}
      layer.S.mod.{hdr,buf}
      layer.P.<STA>.time.{hdr,buf}   # per station, P and S
      layer.P.<STA>.angle.{hdr,buf}
      layer.S.<STA>.time.{hdr,buf}
      layer.S.<STA>.angle.{hdr,buf}
```

`VELOCITY_MODELS/` is **symlinked** (not copied) alongside a fresh copy of
`nlloc/` into a per-launch temp dir (confirmed: `/tmp/obspyck-XXXXXXXX/`),
exactly as the design doc predicted from source. `locate_BY.nlloc`'s
`LOCFILES` ttime-grid-root is therefore relative and reaches up one level:

```
LOCFILES  nlloc.obs  NLLOC_OBS  ../VELOCITY_MODELS/layer_time/layer  nlloc
```

Full working control file (produced by NLLoc v7.1.05, the version this
project's own `install.sh` builds into `external/bin`):

```
CONTROL 1 54321
TRANS  SIMPLE  -38.00 176.00  0.0
LOCSIG OpenMEQ spike (OpenMEQ-c3l.1)
LOCCOM Throwaway 1D model / 3-station spike control file
LOCFILES  nlloc.obs  NLLOC_OBS  ../VELOCITY_MODELS/layer_time/layer  nlloc
LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2 SAVE_HYPOINV_SUM
LOCSEARCH  OCT 10 10 4 0.01 20000 5000 0 1
LOCGRID  41 41 21  -10.0 -10.0 -2.0  0.5 0.5 0.5   PROB_DENSITY  SAVE
LOCMETH EDT_OT_WT 9999.0 4 -1 -1 1.68 6 -1.0 1
LOCGAU 0.2 0.0
LOCGAU2 0.02 0.05 2.0
LOCPHASEID  P   P p Pn Pg
LOCPHASEID  S   S s Sn Sg
LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9
LOCANGLES ANGLES_YES 5
LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
```

The travel-time grids themselves were produced with NonLinLoc's own
`Vel2Grid`/`Grid2Time` (from `external/bin`, per M0.2's install), using a
throwaway 1D layer model and `TRANS SIMPLE -38.00 176.00 0.0` (a
non-Bavaria reference point chosen specifically to make the gk2lonlat bug,
§7, visible). **M5's `setup_nlloc.py` builds the production version of this
same layout** — this spike only had to hand-build one because no example
ships in ObsPyck 0.8.2's source tree, confirming the design doc's claim.

**Combobox model-name decision:** used `locate_BY.nlloc`, matching the
hard-coded combobox's first item, `"BY"` (`qt_designer.py:857`, set via
`setItemText`, not `addItem`, confirmed by grep). **This is item index 0,
which is the combobox's default selection at startup — confirmed in
`docs/images/obspyck_spike_nlloc_done.png`, which shows "BY" selected
without any code touching the combobox.** This is the lower-risk option
the design doc asked this spike to decide between: **ship
`locate_<BY|RH|UH>.nlloc` control files matching the stock names,
preferring `BY` since it needs zero UI interaction to select** — do not
patch `qt_designer.py`.

**Station-naming pitfall (not in the original design doc, found
empirically):** `dicts2NLLocPhases()` writes only the **bare SEED station
code** (`pick.waveform_id.station_code`, e.g. `"ST1"`, no network/location)
into `nlloc.obs`'s "Station name" field. The travel-time grid files must
therefore be named `<ttime_root>.P.<STA>.time.*` using that same bare
code. An initial attempt using `NET_STA_LOC`-style grid labels (mimicking
NonLinLoc's own sample data) silently produced **no location at all** (an
empty `nlloc.sum.grid0.loc.hyp`, no renamed output file, followed by a
`FileNotFoundError` inside `doNLLoc`'s `catFile` call) because NLLoc could
not find a matching grid for `"ST1"`. **M5's `setup_nlloc.py` must name its
per-station travel-time grids using the bare station code** (matching
ARCHITECTURE.md §6's `NET.STA` convention's STA part), not `NET.STA` or any
other combination, or ObsPyck's NLLoc button will silently fail to locate.

## 6. Two ObsPyck 0.8.2 bugs found (beyond the reported gk2lonlat one)

### 6a. `gk2lonlat` (reported bug, confirmed)

`obspyck.util.gk2lonlat` (util.py:901-920) hard-codes `pyproj.Transformer.from_crs(31468, 4326)`
— Gauss-Krüger zone 4 / DHDN, Bavaria's regional projection — **regardless
of the `TRANS` statement actually used to build the NLLoc grids**. It is
used in two places: `loadNLLocOutput` (obspyck.py:2462), which sets
`Origin.longitude`/`Origin.latitude` directly from its output — **this
corrupts the saved event's actual coordinates**, not just a map display —
and `readNLLocScatter` (util.py:944, scatter-plot popup only, not
persisted, lower priority).

**Confirmed empirically** with this spike's real NLLoc output (`TRANS
SIMPLE -38.00 176.00 0.0`, HYPOCENTER x=-3.875 y=-6.875 km):

```
>>> from obspyck.util import gk2lonlat
>>> gk2lonlat(-3.875, -6.875)
(-25.444931253689845, -0.04551970701750636)
```

— nonsensical (the true reference point is `-38.0, 176.0`; raw NLLoc
`TRANS SIMPLE` km-offsets aren't valid Gauss-Krüger-4 eastings/northings at
all, so the "conversion" isn't just off, it's meaningless for any project
not actually using that specific German projection).

**Also confirmed missing dependency**: `pyproj` is not installed by
default — `obspyck` does not declare it, and it isn't in this project's
`pyproject.toml` either, so `gk2lonlat` crashes with `ModuleNotFoundError`
the first time it's called, in a stock `pip install -e ".[dev]"` venv.
**Added `pyproj` to `pyproject.toml`'s dependencies** (minimal, additive —
see the deviations/shared-files note in the report) so `doNLLoc` doesn't
crash outright even before the coordinate bug matters.

**Cleanest patch point: monkeypatch, not a wrapper around the CLI.**
`obspyck.py` does `from .util import (..., gk2lonlat, ...)`
(obspyck.py:55-61) — a `from-import` copies the name into `obspyck.py`'s
own module namespace. **Patching `obspyck.util.gk2lonlat` alone does not
affect the call at obspyck.py:2462** (confirmed by testing both
separately); both `obspyck.util.gk2lonlat` **and** `obspyck.obspyck.gk2lonlat`
must be reassigned to the same replacement function. Because this only
works within the same Python process that later calls
`obspyck.obspyck.main()`, **the batch wrapper (M7.3) must launch ObsPyck by
importing and calling `obspyck.obspyck.main()` in-process from a small
OpenMEQ-owned launcher module (e.g. run as its own subprocess via
`python -m openmeq.review._launch_obspyck ...`), not by shelling out to the
pip-installed `obspyck` console-script executable** — a monkeypatch
applied in the parent process has no effect on a `subprocess.Popen(["obspyck",
...])` child. The per-event subprocess isolation the design already
requires (§5 below) is unaffected — the launcher module just replaces the
console-script as the subprocess's entry point.

The replacement should implement the inverse of whatever `TRANS` the
project's own NLLoc control file uses (known to M7.2's config generator,
since it writes that file), not a fixed EPSG code. Verified working
end-to-end in this spike (flat-Earth inverse of `TRANS SIMPLE`, see
`patch_gk2lonlat`/`apply_bugfix_patches` in `scripts/obspyck_spike_smoke.py`):
with the patch applied, the same NLLoc run's origin came out as
`lon=175.956, lat=-38.062` (matching NLLoc's own `GEOGRAPHIC` line exactly),
vs. the nonsensical unpatched result above.

### 6b. NLLoc phase-line parser incompatible with the installed NonLinLoc version (new finding, not in the original bug report)

`ObsPyck.loadNLLocOutput` (obspyck.py:2685-2721) parses each `PHASE` output
line by fixed whitespace-split index (e.g. `float(line[15])` for predicted
travel time, `line[21]` for epicentral distance). **The NonLinLoc build
this project's own `install.sh` produces (v7.1.05) inserts an extra,
literal standalone `>` token** in every phase line, between the `PriorWt`
and `TTpred` columns:

```
PHASE ID ... PriorWt  >   TTpred    Res       Weight    StaLoc(X  Y ...
ST1 ... 1.0000 >    1.8940   -0.0068    1.0440    0.8760 ...
```

This shifts every field after index 14 by one position, so
`float(line[15])` grabs the literal `">"` instead of the travel time,
raising:

```
ValueError: could not convert string to float: '>'
```

**on every single real "do NLLoc" click** — NLLoc itself succeeds and
writes a correct location (confirmed: `GEOGRAPHIC OT ... Lat -38.062109
Long 175.956096 Depth 6.320312` was already on disk when the crash
happened); only ObsPyck's own re-parse of its output crashes. This makes
M7's whole design ("ObsPyck's own NLLoc button ... is the only relocation
path") non-functional out of the box against a modern NonLinLoc build, so
this is at least as load-bearing as the gk2lonlat bug.

**Patch point:** wrap `PROGRAMS['nlloc']['Call']` (util.py, reassigned
fresh inside `setup_external_programs()` on every `ObsPyck()` construction,
so must be re-applied after each instantiation) to strip stray lone `>`
tokens from `PHASE` data lines in the renamed summary file, immediately
after NLLoc runs and before ObsPyck reads it back — no change to ObsPyck's
own parsing code needed. Verified working end-to-end (see
`apply_bugfix_patches` in `scripts/obspyck_spike_smoke.py`).

Both patches are applied together, immediately after `ObsPyck(...)` is
constructed (or by wrapping `ObsPyck.__init__`, as this spike's driver
scripts do), before entering `qApp.exec_()`.

## 7. Picks preloaded, waveform shown, NLLoc runs — confirmed

`--event <file>.xml` loads the automatic QuakeML's picks onto the
respective traces before the window is shown — confirmed by inspecting
`window.catalog[0].picks` (6 picks for 3 stations x P/S, all present,
correct times) and visually in
`docs/images/obspyck_spike_waveform_picks.png` (P pick in red on `HHZ`,
S pick in blue on `HHN`, exactly at the injected synthetic arrival times).
Clicking "do NLLoc" (`window.on_qToolButton_doNlloc_clicked()`, the same
slot the real toolbutton calls) relocates the event and switches to the
map view; `docs/images/obspyck_spike_nlloc_done.png` shows the scatter
cloud, the three stations with per-station residuals, and the corrected
Origin panel (`Longitude 175.95613`, `Latitude -38.06211`, matching the
project's actual reference point once both bugs above are patched).

## 8. Save-locally path — confirmed

Made one pick-time change (`+0.37s` on one pick) directly on
`window.catalog[0]`, called `window.setXMLEventID()` then
`window.save_event_locally()` (the exact method + button target, invoked
the same way `on_qToolButton_save_...` would). Result:

```
writing xml as obspyck_20260928004317.xml
```

**Observed save-filename pattern: `obspyck_<YYYYMMDDHHMMSS>.xml`** (UTC,
no separators) — i.e. `"obspyck_" + UTCDateTime().strftime('%Y%m%d%H%M%S')
+ ".xml"`, exactly `setXMLEventID`'s timestamp, confirmed **not** related
to the input `event_id` in any way. Re-confirms the design doc's warning:
**the batch wrapper must detect the save by new-file-appearance in a
per-event cwd** (snapshot the dir before `Popen`, diff after `.wait()`),
never by expecting a particular filename. The saved file was valid,
readable QuakeML (`obspy.read_events` round-tripped it) and contained the
modified pick time.

## 9. Exit codes — empirically determined (cases a-d)

Tested by running the **real production entry point**
(`obspyck.obspyck.main()`, which ends in `os._exit(qApp.exec_())`) with a
`QTimer.singleShot` hook that performs the scripted action and then calls
`QApplication.quit()` the same way a real window close would — this is the
real exit path, not an artifact of a simplified test driver (an earlier,
simpler driver that never entered `qApp.exec_()` gave a spurious exit code
120 from interpreter-shutdown noise; discard that number, it does not
reflect real usage).

| Case | Scenario | Exit code | GUI appeared? |
|---|---|---|---|
| (a) | Normal close **after** a save (`save_event_locally()` called, then window closed) | **0** | yes |
| (b) | Normal close with **no save at all** | **0** | yes |
| (c1) | Missing `-t`/`--time` | **1** | **no** (raised before `QApplication` is constructed, obspyck.py:4878) |
| (c2) | Broken config (`[keys]` section removed) | **1** | **no** (raised while building `KEYS`, obspyck.py:4890, before `QApplication`) |
| (d) | Forced uncaught exception mid-session (broke internal state, then called the NLLoc button slot directly) | **0** | yes (app kept running after the exception, then closed normally) |

**Confirms the static-reading prediction exactly: (a) and (b) are
genuinely indistinguishable by exit code alone** (both exit 0) — the batch
wrapper (M7.3) has no choice but file-appearance polling plus its own
terminal prompt to tell save/skip/reject apart, as the design doc
specified. **(c) is distinguishable (exit 1)** — a config/argument error
that never gets to the GUI at all can be detected and treated differently
from a normal close (e.g. abort the whole batch run rather than mark one
event "skipped"). (d) confirms the custom `sys.excepthook` behavior:
the process does **not** terminate or change its eventual exit code when
an exception happens mid-session; the event loop keeps running.

Full transcript, case (a):

```
$ QT_QPA_PLATFORM=offscreen python cli_exit_code_driver.py normal_save \
    -c obspyck.cfg -t 2026-01-01T00:00:00 -d 30 \
    --event ev20260101T000010_100.xml ev20260101T000010_100.mseed stations.xml
...
writing xml as obspyck_20260928004004.xml
$ echo $?
0
```

Case (c1):

```
$ obspyck -c obspyck.cfg -d 30 --event ev20260101T000010_100.xml \
    ev20260101T000010_100.mseed stations.xml
Traceback (most recent call last):
  ...
  File ".../obspyck/obspyck.py", line 4878, in main
    raise Exception(msg)
Exception: Time option ("-t", "--time") must be specified.
$ echo $?
1
```

## 10. Requires_gui smoke script

`scripts/obspyck_spike_smoke.py` reproduces steps 3-5 end to end (builds
its own throwaway fixture + NLLoc grids via `openmeq.external.bin_path`,
launches ObsPyck in-process under offscreen, applies both bugfix
monkeypatches from §6, asserts picks preloaded / NLLoc relocates near the
reference point / a save file appears and round-trips). Skips cleanly
(exit 0, `SKIP: ...`) if obspyck/PyQt5 or the NLLoc toolchain aren't
available, the same way a `pytest.mark.requires_gui` test would be
auto-skipped. Run:

```
QT_QPA_PLATFORM=offscreen OPENMEQ_EXTERNAL_DIR=/path/to/external \
    python scripts/obspyck_spike_smoke.py
```

`ruff check`/`ruff format --check` both pass on it.

## 11. Summary of decisions and follow-ups for M7.2/M7.3/M5/M11

1. **Launch via a small OpenMEQ-owned Python launcher module run with
   `python -m`**, not the `obspyck` console script, so the two
   monkeypatches (§6a, §6b) can be applied in-process before
   `obspyck.obspyck.main()` runs. (M7.2/M7.3)
2. **Generated pluginpath must include empty `hyp_2000/` and `focmec/`
   dirs** alongside `nlloc/`, or `ObsPyck.__init__` crashes before the GUI
   appears. (M7.2)
3. **Ship `locate_BY.nlloc`** (not `RH`/`UH`, not a patched combobox) —
   needs zero UI/code changes since it's the default-selected item.
   (M7.2)
4. **Travel-time grid files must be named with the bare SEED station code
   only** (no network/location), matching what ObsPyck writes into its own
   `nlloc.obs`. **M5's `setup_nlloc.py` must follow this convention.**
   (M5, cross-cutting — flag to M5's owner)
5. **`pyproj` added to `pyproject.toml`** (shared file, minimal/additive
   edit) — required transitively by `obspyck.util.gk2lonlat`, not declared
   by `obspyck` itself.
6. **CI (M11) should use `QT_QPA_PLATFORM=offscreen`**, not `xvfb-run` —
   confirmed equivalent behavior, one fewer system dependency.
7. Batch wrapper (M7.3) exit-code handling: treat exit code `!=0` from the
   launcher as a hard failure distinct from "skipped" (per §9's case c);
   exit code `0` is save/skip/reject-ambiguous and needs the file-polling +
   terminal-prompt mechanism the design doc already specifies.
8. Suggest filing a follow-up issue for M7.4 (or M11) to track the two
   ObsPyck bugs' monkeypatches as a small, tested `openmeq.review._obspyck_compat`
   (or similar) module once M7.2/M7.3 exist, rather than re-deriving them —
   this spike's `scripts/obspyck_spike_smoke.py` has working reference
   implementations of both.
