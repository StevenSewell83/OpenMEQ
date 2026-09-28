# Benchmarks

`scripts/benchmark.py` times each *implemented* `openmeq` CLI pipeline step
(`qc`, `detect`, `chunk`, `pick`, `locate`, `review`, `magnitude`,
`relocate`) against the synthetic test dataset
(`tests/data/make_synthetic.py`), plus dataset generation and config
loading, and writes a timestamped report here:

```
python scripts/benchmark.py --help
```

See `scripts/benchmark.py`'s own module docstring for full usage, the
`STAGES` registry (this script's one-line extension point for new timed
stages as M1-M10 land), and the JSON schema. Summary of the schema
(`schema_version: 1`), reproduced here so it survives independently of the
script (M11.3, which extends this script for the full pipeline, plots
trends across these files by `git_commit` + `generated_at`):

- `schema_version` (int), `generated_at` (ISO-8601 UTC), `git_commit`
  (str or null), `python_version`, `platform`, `argv`
- `n_workers_requested` (`"auto"` or int), `n_workers_resolved` (int or
  null), `threads_per_worker`
- `dataset`: `{"source": "generated"|"config", "seed"?, "n_events"?,
  "n_stations"?, "n_days"?, "scratch_dir"?, "config_path"?}` (`n_days` added
  in M11.3, alongside `--n-stations`/`--n-days`/`--tmp-dir` for an opt-in
  larger benchmark dataset -- see `scripts/benchmark.py`'s own docstring)
- `stages`: list of
  `{"name", "status": "ran"|"skipped"|"error", "duration_s",
  "peak_rss_kb", "peak_rss_kb_delta", "peak_rss_bytes_sampled", "note"}`
  - `peak_rss_kb` is `resource.getrusage(RUSAGE_SELF).ru_maxrss`
    (kilobytes on Linux -- this project's only supported platform). It is a
    whole-process, monotonically non-decreasing high-water mark, not a
    per-stage measurement; `peak_rss_kb_delta` (this stage's value minus
    the previous stage's) only approximates that stage's own contribution.
  - `peak_rss_bytes_sampled` is a tighter, stage-local peak (bytes) from
    polling `psutil.Process().memory_info().rss` while the stage ran, and
    is `null` when `psutil` isn't installed (it's optional, not a project
    dependency).

## `benchmarks/results/`

Git-ignored: run output is timestamped, machine- and run-dependent, and not
meaningful to diff in git. Only this README and `scripts/benchmark.py`
(the schema and the harness that produces it) are committed.
