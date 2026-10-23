# MacBan 2018–2019: processing guide

How the MacBan (Mak-Ban geothermal field, Laguna/Batangas, Philippines)
2018–2019 microseismic dataset is processed with OpenMEQ: from the original
PC-SUDS event files to a relocated, magnitude-tagged catalogue, the exports,
and the tomography catalogue. It covers the dataset-specific scripts, the
three MacBan configs, the instrument response and the decisions behind each
setting.

For the generic pipeline (every step's inputs, outputs and options) see
[usage.md](usage.md); for every config key, [config_reference.md](config_reference.md);
for choosing values, [parameter_guide.md](parameter_guide.md).

Contents

1. [Dataset](#1-dataset)
2. [File locations](#2-file-locations)
3. [Configs](#3-configs)
4. [Step 1 – SUDS to miniSEED](#4-step-1--suds-to-miniseed)
5. [Step 2 – instrument response](#5-step-2--instrument-response)
6. [Step 3 – the main run (detect → locate)](#6-step-3--the-main-run-detect--locate)
7. [Step 4 – magnitudes and station corrections](#7-step-4--magnitudes-and-station-corrections)
8. [Step 5 – hypoDD relocation](#8-step-5--hypodd-relocation)
9. [Step 6 – export](#9-step-6--export)
10. [Manual catalogue](#10-manual-catalogue)
11. [Tomography catalogue](#11-tomography-catalogue)
12. [Results summary](#12-results-summary)
13. [Known issues and caveats](#13-known-issues-and-caveats)
14. [Quick reference](#14-quick-reference)

---

## 1. Dataset

* **Network:** 16 station codes (network code `MB`) at 10 sites over
  roughly 9 × 6 km. Several sites were reoccupied under a new code: MB04 →
  MB4A, MB06 → MB6A → MB6B, MB07 → MB7A, MB09 → MB9A → MB9B.
* **Recording:** triggered. The original data are about 5,900 extended
  PC-SUDS event files, each roughly 90 s long, 3-component, 200 Hz, int32
  samples. Data exist only inside these event windows, never as continuous
  days.
* **Sensor:** Geospace GS-ONE LF 4.5 Hz geophone (vertical and horizontals).
* **Recorder:** Geotech SMART-24R, 40 Vpp input, gain 1.
* **Channels:** `EHZ`, `EHN`, `EHE`. SUDS channel names `chz`/`chn`/`che`
  map to these. Station names map as `MB-02` → `MB02`.
* **Timing:** UTC. This was checked against the 2019-04-22 M6.1 Zambales
  earthquake.

| Station | Lat | Lon | Elev (m) | Data span | Recorder epochs |
|---|---|---|---|---|---|
| MB01 | 14.10526 | 121.21752 | 354 | 2018-01-11 – 2019-12-31 | 2 |
| MB02 | 14.12245 | 121.24950 | 139 | 2018-01-03 – 2019-12-31 | 1 |
| MB03 | 14.09625 | 121.23479 | 275 | 2018-01-03 – 2019-12-31 | 1 |
| MB04 | 14.10323 | 121.20398 | 360 | 2018-01-03 – 2019-11-19 | 1 |
| MB05 | 14.12499 | 121.16825 | 294 | 2018-01-03 – 2019-12-31 | 2 |
| MB06 | 14.08571 | 121.21939 | 265 | 2018-01-03 – 2018-03-06 | 1 |
| MB07 | 14.07765 | 121.25191 | 172 | 2018-01-03 – 2019-08-07 | 1 |
| MB08 | 14.12865 | 121.21003 | 563 | 2018-02-21 – 2019-12-31 | 1 |
| MB09 | 14.07624 | 121.17040 | 222 | 2018-01-05 – 2018-02-22 | 1 |
| MB10 | 14.11155 | 121.23692 | 242 | 2018-01-03 – 2019-12-31 | 1 |
| MB4A | 14.10030 | 121.20573 | 337 | 2019-11-20 – 2019-12-31 | 1 |
| MB6A | 14.08717 | 121.21505 | 270 | 2018-03-09 – 2019-10-02 | 3 |
| MB6B | 14.08480 | 121.21618 | 263 | 2019-11-07 – 2019-12-10 | 1 |
| MB7A | 14.07606 | 121.24909 | 177 | 2019-08-08 – 2019-12-31 | 2 |
| MB9A | 14.07301 | 121.17616 | 228 | 2018-02-23 – 2019-09-03 | 2 |
| MB9B | 14.07396 | 121.17033 | 221 | 2019-09-04 – 2019-12-31 | 1 |

"Recorder epochs" is the number of different recorder units a station used.
A station gets a new date range in the StationXML each time its recorder was
swapped (see [§5](#5-step-2--instrument-response)).

---

## 2. File locations

All data and run folders are on the `fast9100` drive, which is the configs'
`project.base_dir`.

**Data**, under `/media/steve/fast9100/datasets/MacBan/`:

| Path | Contents |
|---|---|
| `MEQ Exercise MB 2018-2019(1)/` | Original PC-SUDS event files (the raw data) |
| `mseed/YEAR/STA/CHA.D/MB.STA..CHA.D.YEAR.DOY.mseed` | miniSEED day files converted from SUDS |
| `mseed/stations.xml` | Station coordinates only, written by the SUDS converter. Never edited. |
| `mseed/stations_resp.xml` | **Station metadata with instrument response.** This is the file all configs use. |
| `man_picks/` | The operator's manual catalogue (`.evl`, `.pha` and `.suds` per event) |
| `man_picks_extracted/` | The manual catalogue converted to QuakeML, CSV and per-event miniSEED |
| `models/` | 1-D velocity models (`macban_vel1d_velest_graded.txt` is the one in use) |

**Runs**, under `/media/steve/fast9100/runs/`:

| Path | Config | Contents |
|---|---|---|
| `macban_2018-01-01_2019-12-31/` | `config/macban_full.yaml` | **Main run**: 2018-01-01 to 2019-12-31 |
| `macban_tomo_2018-2019/` | `config/macban_tomo.yaml` | Tomography catalogue: QC'd and weighted picks, station delays, VELEST |
| `macban_2018-03-09_2019-08-07/` | `config/macban_old.yaml` | Earlier, shorter run, superseded |

**Repository** (`/home/steve/PycharmProjects/OpenMEQ`):

| Path | Purpose |
|---|---|
| `config/macban_full.yaml`, `macban_tomo.yaml`, `macban_old.yaml` | Run configs |
| `config/macban_station_corrections.csv` | ML station corrections |
| `scripts/suds2mseed.py` | SUDS → miniSEED, run outside the pipeline |
| `scripts/detect_windows.py` | Detection limited to the recording windows of a triggered archive |
| `scripts/macban_stationxml.py` | Builds `stations_resp.xml` |
| `scripts/macban_station_terms.py` | Derives ML station corrections from a run |
| `scripts/macban_man_picks.py` | Manual catalogue → QuakeML, CSV and miniSEED |
| `scripts/macban_tomo_qc.py` | Matches manual and automatic events; QCs the automatic picks |
| `scripts/macban_tomo_catalog.py` | Pick weights, NLLoc station delays, VELEST export |

---

## 3. Configs

The three configs are identical except for the keys in the table below.

| Key | `macban_old.yaml` | `macban_full.yaml` | `macban_tomo.yaml` |
|---|---|---|---|
| `project.output_dir` | `runs/macban_2018-03-09_2019-08-07` | `runs/macban_2018-01-01_2019-12-31` | `runs/macban_tomo_2018-2019` |
| `data.start` / `end` | 2018-03-09 / 2019-08-07 | 2018-01-01 / 2019-12-31 | 2018-01-01 / 2019-12-31 |
| `review.enabled` | true | false | false |
| `nlloc.station_delays_csv` | – | – | `runs/macban_tomo_2018-2019/tomo/station_delays.csv` |

The settings they share, and why:

* **Stations:** `stations.stationxml` points at `stations_resp.xml`.
* **Detection:** 1–10 Hz bandpass, STA 0.2 s, LTA 4 s, trigger on at 4.9
  and off at 1.5. A detection needs 6 traces on at least 3 stations. These
  were chosen with `openmeq diagnostics detect-sweep` on the 10 busiest days.
* **PhaseNet:** P picks need probability ≥ 0.3, S picks ≥ 0.5. Data are
  decimated from 200 to 100 Hz with a zero-phase anti-alias filter
  (OpenMEQ-tr9; the old causal filter made every pick 20–30 ms late).
* **NLLoc:**
  - The 1-D model is `macban_vel1d_velest_graded.txt`, from a VELEST Monte
    Carlo, smoothed and graded at 2 km.
  - Residual cuts are 0.20 s for P and 0.30 s for S, at 3σ.
  - An accepted event needs ≥ 5 phases, RMS ≤ 0.15 s, gap ≤ 285°,
    horizontal error ≤ 1.2 km and depth error ≤ 2.0 km. S is not required.
* **hypoDD:** see [§8](#8-step-5--hypodd-relocation).
* **Magnitude:** Hutton–Boore ML with Wood-Anderson sensitivity 2080,
  measured in an S window from 1 s before to 4 s after the pick. Station
  corrections come from `config/macban_station_corrections.csv`.

Paths in the configs are resolved against `project.base_dir`
(`/media/steve/fast9100`). That's why the station-corrections CSV, which
lives in the repository, is given as an absolute path.

---

## 4. Step 1 – SUDS to miniSEED

This has already been done, and the output is in `datasets/MacBan/mseed/`.
To redo it:

```bash
SRC="/media/steve/fast9100/datasets/MacBan/MEQ Exercise MB 2018-2019(1)"
python scripts/suds2mseed.py "$SRC" /media/steve/fast9100/datasets/MacBan/mseed
```

* The triggered windows are merged into one STEIM2 day file per channel.
  Gaps stay as gaps.
* `stations.xml` is written with each station at the median of its GPS fixes
  (bad or far-off fixes are dropped). It has coordinates only, no response.
* Existing day files are kept unless you pass `--overwrite`.
* The same conversion can run automatically at the start of `openmeq qc` if
  `data.convert` is set. It's commented out in the MacBan configs because the
  archive is already converted.

The Feb 2021 recorders wrote CD-1.1 files (`.cd11`) instead of SUDS.
`config/macban_2021.yaml` has a `data.convert` block (`format: cd11`) for
them, so this converts them to `MacBan/mseed_2021`:

```bash
openmeq convert -c config/macban_2021.yaml
```

`scripts/cd11tomseed.py IN_DIR OUT_DIR` does the same without a config.
The files carry no coordinates or gains, so no `stations.xml` is written,
and the 2018-2019 `stations_resp.xml` epochs don't reach 2021.

Two known quirks: two SUDS files have a corrupt block (one trace lost from
each), and 11 day files keep conflicting 60 s overlaps as separate segments.
The SUDS `FEATURE` picks (about 147k) are not converted.

---

## 5. Step 2 – instrument response

### 5.1 What the response is

`stations_resp.xml` gives every channel a two-stage response.

**Stage 1 – geophone (M/S → V).** This is the GS-ONE LF 4.5 Hz, from
brochure 592-15720-01 Rev. B. The vertical and horizontal specifications are
identical. The response has two zeros at 0 and the poles
`−h·ω0 ± i·ω0·√(1−h²)`, with `ω0 = 2π·4.5`.

Damping and sensitivity depend on the electrical load across the coil. The
script works them out from the brochure's two specification points, with
coil resistance Rc = 2450 Ω:

| | Damping h | Sensitivity |
|---|---|---|
| Open circuit | 0.42 | 100.4 V/(m/s) |
| 20 kΩ load | 0.735 | 89.4 V/(m/s) |

The formulas are `h = h0 + K/(Rc + R)` and `S = S0·R/(Rc + R)`, where h0 and
S0 are the open-circuit values and K is fixed by the 20 kΩ point.

The load R is the recorder's input resistance. The SMART-24 40 Vpp boards
come in three versions (SMART-24 manual §2.1.3):

| ADC board | Input R | Damping | Sensitivity |
|---|---|---|---|
| **Ver. 06** (standard since Nov 2011): **default** | 2.0 MΩ | 0.424 | 100.3 V/(m/s) |
| Ver. 01 | 1.7 MΩ | 0.424 | 100.3 V/(m/s) |
| Ver. 03 (low-impedance option) | 17.2 kΩ | 0.78 | 87.9 V/(m/s) |

**Stage 2 – recorder (V → COUNTS).** This is a gain-only stage:
`1 / con_mvolts`. `con_mvolts` is the PC-SUDS STATIONCOMP field
"conversion to mvolts" (a float at byte 56; data units `d` mean digital
counts; USGS OFR 89-188). The MacBan files record a calibrated value per
channel of 3.22–3.31 µV/count. That matches the SMART-24's nominal
3.271096 µV/count at 40 Vpp, gain 1, so there is no hidden amplifier gain.

The values belong to physical recorder units, and units were moved between
sites. For example, the MB07 recorder also appears at MB01, MB6A, MB6B and
MB7A at other times. So each channel gets a separate date range for every
run of constant `con_mvolts`.

The overall sensitivity is about 3.24 × 10⁷ counts/(m/s), flat above about
6 Hz.

Not included: the SMART-24's digital filters. They are linear phase with
unity passband gain, so they don't affect amplitudes or ML.

### 5.2 Rebuilding it

```bash
python scripts/macban_stationxml.py \
    "/media/steve/fast9100/datasets/MacBan/MEQ Exercise MB 2018-2019(1)/MEQ Exercise MB 2018-2019" \
    /media/steve/fast9100/datasets/MacBan/mseed/stations.xml \
    /media/steve/fast9100/datasets/MacBan/mseed/stations_resp.xml \
    [--load-ohm 2.0e6] [--plot response.png]
```

The script takes about 10 s. It prints every channel's date ranges with
µV/count and counts/(m/s), and it validates the StationXML before writing.

* If the units turn out to have the Ver. 03 board, or an external resistor
  across the geophone, rerun with that `--load-ohm`. Two resistors in
  parallel combine as `R = 1/(1/R1 + 1/R2)`.
* After changing the response, rerun magnitude **with `--force`** (see
  [§13](#13-known-issues-and-caveats)).

### 5.3 How it was checked

* The response plot is flat in velocity above about 6 Hz and rolls off as f²
  below 4.5 Hz.
* For a typical small event at MB04, removing the response gives a P peak of
  about 1.3 µm/s, which is plausible.
* Across the full run, ML agrees with the independent manual catalogue to
  +0.12 ± 0.15 (see [§7.3](#73-comparison-with-the-manual-catalogue)).

---

## 6. Step 3 – the main run (detect → locate)

The main run is `runs/macban_2018-01-01_2019-12-31` with
`config/macban_full.yaml`. Because the archive is triggered, detection
doesn't run over whole days. Instead it runs only inside the windows where
enough stations record at once:

```bash
C=config/macban_full.yaml
openmeq qc -c $C                                   # index the archive, merge station metadata, QC
python scripts/detect_windows.py -c $C --min-stations 5
openmeq run -c $C --from chunk --skip-review       # chunk, pick, locate, magnitude, relocate
```

* `detect_windows.py` writes `2_detection/detections.csv`, plus
  `coincidence_windows.csv` listing the windows it searched.
* Don't run `openmeq detect` or `openmeq run` from the start afterwards.
  Either one replaces `detections.csv` with a whole-day detection. Always
  start from `--from chunk`.
* Review is off in this config, so magnitude reads
  `7_location/catalog_auto.xml`.

The result: 3,194 detections, 2,929 events with picks, and 796 accepted NLLoc locations. The run
detects 98% of the manual events.

**Station S delays.** Locate's residual filter (S at most 0.3 s, 3σ) drops
most S picks at MB07/MB7A. This is a real site effect, not bad picks: site 7
S arrives about 0.4–0.47 s late, consistently across both occupations, and
site 9 about 0.2 s. The tomography run handles it with station delays
instead of rejection (see [§11](#11-tomography-catalogue)).

---

## 7. Step 4 – magnitudes and station corrections

### 7.1 Running

```bash
openmeq magnitude -c config/macban_full.yaml --force
```

For each event the step:
1. removes the instrument response and simulates a Wood-Anderson instrument
   (two zeros, sensitivity 2080);
2. takes the peak amplitude in the S window on the larger horizontal;
3. converts it to ML with Hutton & Boore (1987):
   `ML = log10(A mm) + 1.110·log10(R/100) + 0.00189·(R−100) + 3.0`, with R
   the hypocentral distance in km;
4. adds the station correction;
5. takes the median over stations, with at least 2 stations required.

Output is in `8_relocation/magnitude/`: `catalog_mag.xml`,
`station_magnitudes.csv`, and `diagnostics/` (ML against distance, station
residuals, magnitude–frequency, ML through time, and `ml_vs_manual.png`).

The step reads station metadata from the config's `stations.stationxml`.
It does **not** read `0_setup/stations.xml`.

### 7.2 Station corrections

The corrections are station terms solved from the run's own data, not from
the manual catalogue. The model is `station_ml = M_event + S_station`,
solved by least squares over every included measurement. The terms are
constrained to average zero, so the overall Hutton–Boore level is
unchanged. The correction is `−S`.

```bash
python scripts/macban_station_terms.py -c config/macban_full.yaml \
    config/macban_station_corrections.csv [--min-count 20]
openmeq magnitude -c config/macban_full.yaml --force
```

* The script first removes the corrections already in the config from each
  station ML, so rerunning it re-derives the same terms instead of stacking
  them. (Checked: a second pass reproduced them to within 0.001.)
* A station with fewer than `--min-count` measurements gets no correction.
  At present that's MB6B, with 9.

Current corrections, from the full run on 2026-10-23:

| Station | Correction | Station | Correction |
|---|---|---|---|
| MB9A | +0.44 | MB07 | −0.07 |
| MB05 | +0.16 | MB08 | −0.07 |
| MB7A | +0.15 | MB10 | −0.08 |
| MB9B | +0.14 | MB4A | −0.10 |
| MB09 | +0.11 | MB6A | −0.12 |
| MB04 | +0.02 | MB03 | −0.13 |
| MB06 | 0.00 | MB01 | −0.18 |
| MB6B | 0 (too few) | MB02 | −0.25 |

The corrections cut the spread of station MLs within an event from 0.23 to
0.17. Event MLs moved by 0.07 (median), at most 0.25.

**Why there's no local distance relation.** A joint fit wants a stronger
distance term than Hutton–Boore (coefficient about 1.65 instead of 1.11),
but:
- it reduces the scatter only from 0.164 to 0.157;
- it makes agreement with the manual catalogue slightly worse;
- the data span only 1–12 km, so the term is poorly constrained and trades
  off against the station terms.

Hutton–Boore is kept for that reason.

### 7.3 Comparison with the manual catalogue

Automatic events were matched one-to-one to manual events with an ML,
closest origin time first, within 2 s. 747 of the 796 matched.

| | No corrections | With station corrections |
|---|---|---|
| Median (OpenMEQ − manual) | +0.16 | +0.12 |
| Std | 0.15 | 0.15 |
| Correlation | 0.957 | 0.957 |
| Best fit | 0.91·manual + 0.21 | 0.90·manual + 0.17 |

Above manual ML 0.5 the difference is about +0.05 to +0.15. Below ML 0 it
grows to +0.3 to +0.4. Noise in the peak-amplitude reading of the smallest
events is the likely cause, but this hasn't been checked. The manual
catalogue's magnitude formula is unknown, so it's a cross-check, not an
absolute reference.

---

## 8. Step 5 – hypoDD relocation

```bash
openmeq relocate -c config/macban_full.yaml     # reads 8_relocation/magnitude/catalog_mag.xml
```

The settings were tuned on 2026-10-22:

* **Event pairs (ph2dt):** MAXSEP 2 km, MINLNK = MINOBS-CT 6, MINOBS 4,
  MAXOBS 20.
  - The earlier MINLNK 8 was too strict for about 7 stations per event. It
    produced 21k weak pairs out to 4 km and about 340 m of drift of the
    whole cluster.
  - The current settings give about 5.8k pairs with a mean separation of
    0.47 km, and about 80 m of drift.
* **Weighting:** fixed residual cutoffs rather than dynamic ones (CC
  0.04/0.03 s, CT 0.15/0.12 s). Distance weighting WDCT 2 km and WDCC 1 km.
  Damping 100 (condition number 42–56).
  - The dynamic cutoff (WRCC 6×SD) ran away and kept only 55% of the CC
    data instead of 80%.
* **Cross-correlation lag:** separate maximum lags for S and P (0.10 s and
  0.05 s), via `hypodd.xcorr.s_maxlag_s`.

Full-run result: 745 events relocated in one cluster, 64% of CC data kept,
final RMS 28 ms for catalogue times and 4 ms for cross-correlation times.
`8_relocation/catalog_dd.xml` holds all 2,929 events. The 745 relocated
ones carry the ML from the magnitude step.

---

## 9. Step 6 – export

```bash
E=/media/steve/fast9100/runs/macban_2018-01-01_2019-12-31/8_relocation
openmeq export -c config/macban_full.yaml --catalog $E/catalog_dd.xml --accepted-only
(cd $E && zip -qr export.zip export)
```

Writes to `8_relocation/export/`:
- `catalog_dd_events.csv`, `_arrivals.csv` and `_stations.csv`;
- `catalog_dd_fields.csv`, a column dictionary;
- WGS84 shapefiles of hypocentres, error ellipses and stations.

The current bundle has 796 events and 10,700 picks, all with ML.

Columns to know:
- `mag` is the event ML;
- `mag_n_sta` is the number of stations it uses;
- `hypodd_mag` is the magnitude as passed through hypoDD;
- `mag_err` is empty, because the magnitude step doesn't produce an
  uncertainty.

---

## 10. Manual catalogue

```bash
python scripts/macban_man_picks.py \
    /media/steve/fast9100/datasets/MacBan/man_picks \
    /media/steve/fast9100/datasets/MacBan/man_picks_extracted [--plot | --plot-only]
```

This converts 1,389 manually picked and located events (`.evl` locator
printouts) to `catalog.xml`, `events.csv`, `picks.csv` and
`waveforms/<event_id>.mseed` (with `.png` plots if `--plot` is given).

* The `events YYYY all` folders are byte-identical copies of the monthly
  folders, and are de-duplicated.
* `*_reg*.suds` files are unpicked regional recordings, and are skipped.
* The `.evl` file is authoritative. The `.pha` file disagrees with it in
  only two events: the MB07 S pick in 20180321_004107 evt2, and swapped N/E
  at MB10 in 20190305_122248 evt1.
* The catalogue contains near-duplicates: the same earthquake located from
  two SUDS files, less than 1 s apart.

**Don't use the manual picks as a timing reference.** They were placed to
reduce the locator RMS (the analyst confirmed this; checked on 2026-10-21):

- Manual P residuals track independent onset picks with slope only 0.34.
  For manual S the slope is 0.06, i.e. no information beyond the 1-D model.
- Manual station terms come out near 0. Onset picks show real terms: MB03 P
  about −0.05 s, MB07 S about +0.23 s, MB05 S about −0.16 s.

Use the manual catalogue for matching events and stations, flagging gross
errors, and as a magnitude cross-check only. Further notes: manual P sits on
the onset (±10 ms of an AIC pick), except at MB03 (about 45 ms late; it
picks the second, larger swing, so polarities there are suspect) and MB07
(about 20 ms late). MB9A is very noisy, and PhaseNet picks there run 70–90
ms late.

---

## 11. Tomography catalogue

Run: `runs/macban_tomo_2018-2019`, with `config/macban_tomo.yaml`. It reuses
the full run's raw chunks through a symlink at `4_data_chunks/chunks`.
Tracking issue: OpenMEQ-qgn.

```bash
R=/media/steve/fast9100/runs/macban_2018-01-01_2019-12-31
T=/media/steve/fast9100/runs/macban_tomo_2018-2019
M=/media/steve/fast9100/datasets/MacBan/man_picks_extracted

# 1. Match manual/automatic events and QC the automatic picks -> $R/tomo/qc/
python scripts/macban_tomo_qc.py $R $M

# 2. Weight classes 0-3 -> the tomo run's pick catalogue
python scripts/macban_tomo_catalog.py weights $R/tomo/qc $T/5_ml_picks/catalog_picks.xml

# 3. Locate, then iterate the station delays (seed S from the QC Wadati terms)
openmeq locate -c config/macban_tomo.yaml
python scripts/macban_tomo_catalog.py delays $T $T/tomo/station_delays.csv --seed-qc $R/tomo/qc
openmeq locate -c config/macban_tomo.yaml --force
python scripts/macban_tomo_catalog.py delays $T --previous $T/tomo/station_delays.csv $T/tomo/station_delays.csv
# ... repeat until the delays stop changing (rounds kept as station_delays_r1..r4.csv)

# 4. Export for VELEST (openSeisVelocity)
python scripts/macban_tomo_catalog.py export $T $T/tomo/catalog_velest.xml

# 5. Magnitudes and hypoDD on the tomo catalogue
openmeq magnitude -c config/macban_tomo.yaml --force
openmeq relocate  -c config/macban_tomo.yaml
```

**Pick QC (`macban_tomo_qc.py`):**
- **P:** checked against an AIC onset picker (2–30 Hz, zero-phase). Each
  station's PhaseNet lag behind the onset is estimated and removed.
- **S:** checked with a Wadati fit (S−P against P time), with per-station
  terms. A consistent site delay therefore goes into the term rather than
  causing rejection.
- **MB9A** has stricter thresholds because it's noisy. **MB9B:** about half
  of its PhaseNet S picks are a strong later phase about 3 s after P, not
  direct S.

**Weight classes** run from 0 to 3, with uncertainties 0.02, 0.05, 0.10 and
0.20 s. NLLoc and VELEST weight a pick the same way.

**State as of 2026-10-23:**

* **NLLoc:** 662 accepted events after 3 rounds of station delays. RMS
  0.030 s, horizontal/depth errors 0.42/0.37 km. The MB07 S delay is
  +0.42 s, and 183 S picks are kept there.
* **hypoDD:** 644 relocated, all with ML (−0.16 to 3.64, median 0.67), using
  the tuned settings from [§8](#8-step-5--hypodd-relocation). The whole
  cluster shifts only +9 m E / −24 m N / −7 m in depth relative to NLLoc,
  with a median horizontal shift of 121 m. The earlier result with the old
  settings shifted +275 / −152 / −48 m and is kept in
  `8_relocation_prev_nomag/`.
* **VELEST** (275 events, 15% held out for validation), two runs, neither
  acceptable:
  - (a) fixed Vp/Vs 1.768: RMS 0.041. The S station corrections are forced
    to Vp/Vs × the P corrections, so the separate site-7 S delay can't be
    represented.
  - (b) independent S model: RMS 0.025, but layer Vp/Vs swings from 1.48 to
    1.90 with no constraint, and layers 0 and 6 are unresolved.
* **Open decision (qgn.6):** how to constrain the minimum 1-D model.
  Proposed: independent P and S with Vp/Vs damped between layers, the
  unresolved layers tied to their neighbours, and the site S delays left to
  the NLLoc station delays. After that come qgn.7 (event selection and
  declustering) and qgn.8 (VELEST/FMTOMO export).

---

## 12. Results summary

| | Full run | Tomography run |
|---|---|---|
| Detections | 3,194 | (reuses full run) |
| Events with picks | 2,929 | 1,199 |
| Accepted NLLoc locations | 796 | 662 |
| hypoDD relocated | 745 | 644 |
| Events with ML | 796 | 662 |
| ML range (relocated) | −0.21 to 3.36 | −0.16 to 3.64 |
| ML vs manual | +0.12 ± 0.15, r 0.957 (747 matched) | – |

---

## 13. Known issues and caveats

* **Magnitude doesn't notice a changed response or corrections file
  (OpenMEQ-812).** The step's "already done" check covers only the
  magnitude settings and the input catalogue. After you change
  `stations_resp.xml` or edit the corrections CSV, it reports "unchanged"
  and skips. **Always use `openmeq magnitude --force`** after such a change,
  then rerun `relocate` so `catalog_dd.xml` picks up the new ML.
* **usage.md is wrong about the magnitude step's station file.** It says
  magnitude reads `0_setup/stations.xml`. It actually reads the config's
  `stations.stationxml`.
* **Rerunning `detect` overwrites the windowed detections.** See
  [§6](#6-step-3--the-main-run-detect--locate).
* **Recorder board version is assumed.** The response assumes ADC Ver. 06
  (2.0 MΩ). If it's Ver. 03 (17.2 kΩ), damping is 0.78 instead of 0.42,
  which changes the response by up to a factor of 1.8 near 4.5 Hz.
* **ML calibration.** Station corrections make the catalogue internally
  consistent but don't calibrate its absolute level. Hutton–Boore is
  California-derived and used here at 1–12 km.
* **Backups in the full run.** These are kept, in case you need to go back:
  - `0_setup/stations_coords_only.xml.bak`
  - `8_relocation/magnitude_prev_noresp/` (from before the response; no
    station amplitudes)
  - `8_relocation/magnitude_hb_nocorr/` (Hutton–Boore without station
    corrections)
  - `8_relocation_prev_nomag/` (hypoDD without ML)
  - `8_relocation/export_prev_nomag/` and `export_prev_nomag.zip`
* **Not in git.** `stations_resp.xml` and everything under `runs/` live on
  the data drive. The scripts that produce them are committed.

---

## 14. Quick reference

```bash
cd /home/steve/PycharmProjects/OpenMEQ
C=config/macban_full.yaml
E=/media/steve/fast9100/runs/macban_2018-01-01_2019-12-31/8_relocation

# Rebuild the response, then redo everything downstream of it
python scripts/macban_stationxml.py "<SUDS dir>" <stations.xml> <stations_resp.xml>
openmeq magnitude -c $C --force
python scripts/macban_station_terms.py -c $C config/macban_station_corrections.csv
openmeq magnitude -c $C --force          # again, with the new corrections
openmeq relocate  -c $C
openmeq export    -c $C --catalog $E/catalog_dd.xml --accepted-only
(cd $E && rm -f export.zip && zip -qr export.zip export)
```
