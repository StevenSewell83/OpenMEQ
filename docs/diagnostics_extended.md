# Extended QAQC diagnostics

Every step always writes a baseline set of figures to `<step>/diagnostics/`
(see `docs/usage.md`). On top of that, each step except `qc` has an opt-in
**extended** set, aimed at diagnosing whether the processing is working and at
tuning it. Extended figures go to `<step>/diagnostics/extended/`. A failing
extended figure only logs a warning; it never fails the step.

## Switching it on

```yaml
diagnostics:
  extended: true          # default for every step
  n_examples: 10          # sample events for per-event waveform figures
  relocate: {extended: false}   # per-step override (null = inherit)
  pick:
    pick_zoom: {half_window_s: 2.0, n_events: 10, show_rejected: false}
```

Rerun behaviour: turning `extended` on for a step that has already run does
not recompute it in most steps; rerun that step with `--force` to get the
figures (OpenMEQ-02o.10 tracks making this uniform). The run-level funnel can
be regenerated any time with `openmeq diagnostics funnel -c config.yaml`.

## What each step writes

| Step | File | What to look for |
|---|---|---|
| detect | `detections/station_triggers.csv` | Per-station STA/LTA triggers (only written when enabled; can be large on noisy data). |
| detect | `station_trigger_rate.png` | Station x hour trigger heatmap: noisy stations driving false coincidences. |
| detect | `cf_examples_NN.png` | Filtered trace + STA/LTA ratio with trigger_on/off lines, for sampled detections and 2 quiet windows: tune `sta`, `lta`, thresholds. |
| chunk | `component_coverage.png/.csv` | Station x Z/1/2 presence in raw chunks and `chunks_pn`: `channel_map` mistakes, missing horizontals. |
| chunk | `raw_vs_pn_<event>.png` | Raw vs PhaseNet-prepped traces (written during `pick`, once `chunks_pn/` exists). |
| pick | `pick_zoom_<event>.png` | +/- `half_window_s` around each pick; Z for P, horizontals for S; PhaseNet probability on a twin axis. `show_rejected` adds duplicate/unresolved picks and sub-threshold probability peaks. |
| pick | `wadati_prelocate.png` | S-P vs P arrival; Vp/Vs = 1 + slope; outliers are mis-associated S picks. |
| pick | `pick_offset_hist.png` | Pick time relative to detection time, with the chunk `pre_s`/`post_s` edges. |
| locate | `passX_res_vs_distance.png` | P/S residual vs epicentral distance with binned median: velocity-model bias. |
| locate | `passX_station_res_map.png`, `passX_station_residuals.csv` | Mean residual per station: candidate station corrections. |
| locate | `passX_depth_hist_grid.png` | Depths against LOCGRID limits and 1D layer interfaces: pile-ups on edges/interfaces. |
| magnitude | `res_vs_distance.png` | Station ML residual vs distance: does the -log A0 relation fit the region? |
| magnitude | `gr_mc_bvalue.png/.csv` | Mc (max curvature + 0.2), Aki/Utsu b-value, bootstrap uncertainty. |
| magnitude | `amp_window_<event>.png` | Wood-Anderson trace, window and measured peak: is the amplitude on S, noise or a spike? |
| relocate | `dtcc_vs_dtct.png/.csv` | (dt.cc - OTC) - (dt.ct) for matching observations: cycle skips, wrong-phase correlations. |
| relocate | `res_final_by_type.png` | Final `hypoDD.res` residuals split CT-P/CT-S/CC-P/CC-S: weighting and outlier cut-offs. |
| relocate | `iteration_shifts.png` | Mean DX/DY/DZ/DT/OS per iteration with weighting sets shaded: convergence. |
| run | `diagnostics/event_funnel.png/.csv` | Event counts through detect -> chunk -> pick -> locate A/B -> review -> ML -> relocate. |

Known limitation: a single hypoDD run over several clusters keeps only the
last cluster's `hypoDD.res`, and `parallel_clusters` mode has no merged
`hypoDD.log` (OpenMEQ-02o.9); the figure titles say when coverage is partial.
