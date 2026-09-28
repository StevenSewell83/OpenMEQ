"""PhaseNet batch runner (M4.3, ``OpenMEQ-1m7.3``).

PLAN.md §4's "PhaseNet run" row ("One TensorFlow process, batched
(``batch_size``), CPU or GPU") -- this module is that one subprocess call.
It assumes ``chunks_pn/`` is already fully materialized (M4.2's
:func:`openmeq.pick.prep.prepare_all`, which also writes
``chunks_pn/data_list.csv``) before :func:`run_phasenet` is ever invoked.

Pipelining choice (prep vs. predict)
-------------------------------------
PLAN.md's "PhaseNet run" row says "Prep runs ahead of prediction so the
model is never idle". This milestone does **not** implement worker-level
pipelining (interleaving remaining ``pmap`` prep workers with an
already-running ``predict.py`` subprocess) -- it runs ``prepare_all``
(fully, via its own ``pmap`` over events) and *then* a single
``run_phasenet`` call, strictly sequential. Two things justify this over
building an incremental/streaming pipeline here:

1. **``predict.py`` cannot be pipelined internally.** It calls
   ``tf.compat.v1.disable_eager_execution()`` at *module import time*
   (``docs/phasenet_interface.md`` §2), so it can only run as one
   subprocess per pick run over the whole ``data_list.csv`` -- never
   imported and re-invoked per batch within one process. The only way to
   "stream" batches into PhaseNet at all would be several separate
   subprocess invocations (each paying the ~2.7s fixed model-load/startup
   cost again), which is strictly slower than one call over everything
   once prep is done, for any realistic per-run event count.
2. **The measured costs make it not worth building.** Per
   ``docs/phasenet_interface.md`` §6, ``predict.py``'s own marginal cost is
   ~8ms/chunk; M4.2's own prep step is the more expensive part per event
   (filtering + decimation), but M4.2 already parallelizes prep across all
   CPU cores via ``openmeq.parallel.pmap`` before this module ever runs.
   Overlapping "the last few prep workers finish" with "predict.py has
   started on the events already done" would only ever save, at best, one
   worker's tail latency for one batch of chunks (~milliseconds to a few
   seconds for realistic run sizes) -- dwarfed by the model's own fixed
   ~2.7s startup cost, which is paid exactly once either way. There is no
   measured or plausible bottleneck this would fix at this milestone's
   scale.

**Deferred, not a defect**: true worker-level pipelining (starting
``predict.py`` on a first streamed batch while remaining events are still
being prepped, or running several ``predict.py`` calls over event
sub-batches so the model's input queue is never empty on a machine where
prep is slower than prediction) is flagged as a performance-pass follow-up
for **M11.3**, not fixed here -- M4.2 fully materializes ``chunks_pn/``
before this module's ``run_phasenet`` starts, by design, so there is
nothing to overlap yet at this milestone.

``--batch_size`` is always 1
-----------------------------
``cfg.phasenet.batch_size`` (``config/example.yaml``, default 20) is
**not** used to build the ``predict.py`` invocation here, regardless of
its configured value. ``docs/phasenet_interface.md`` §7 (M4.1, confirmed
by direct reproduction) found that ``predict.py --batch_size=N>1``
requires every chunk combined into one TensorFlow batch to have both the
same time length **and** the same per-file station count -- this
project's ``chunks_pn/<event_id>/<event_id>.mseed`` files generally have
different station counts event to event (network availability differs
per event, ``chunks/completeness.csv``), so batching by
``cfg.phasenet.batch_size`` would fail the moment two events in the same
run have a different station count, which is the common case here, not an
edge case. There is also no measured CPU speedup from ``batch_size>1``
(§6: 3.16s vs. 3.00s for 30 identical-shape chunks) -- batching only helps
GPU throughput, and this machine's GPU is unusable anyway (no matching
cuDNN, see below). ``--batch_size=1`` is therefore hardcoded here;
``cfg.phasenet.batch_size`` is currently unused/aspirational (documented
on the field itself and in ``config/example.yaml``), a contract note for
whoever revisits GPU support and per-station-count batching.

Timing (this machine, CPU, ``docs/phasenet_interface.md`` §6)
-----------------------------------------------------------------
Fixed overhead (Python/TensorFlow startup, checkpoint restore): ~2.7s per
``predict.py`` invocation. Marginal cost: ~8ms per 60s/100Hz/3-station
chunk. This is why :func:`run_phasenet` makes **exactly one** subprocess
call over the whole ``data_list.csv`` -- one call per event would cost
``n_events * 2.7s`` instead of ``2.7s + n_events * 0.008s``, i.e. roughly
350x slower at 100 events and worse at scale.

Output
------
``pick/phasenet_raw/picks.csv`` (:attr:`openmeq.paths.RunPaths.phasenet_raw_dir`),
plus (since ``--save_prob`` is always passed, negligible extra cost per
``docs/phasenet_interface.md`` §6) ``pick/phasenet_raw/result.h5`` for
M4.4's probability-trace diagnostics. ``picks.csv``'s exact columns (with
``--amplitude``, always passed) are documented in
``docs/phasenet_interface.md`` §5 and reproduced here as
:data:`PICKS_CSV_COLUMNS` for **M4.4** (PhaseNet picks -> QuakeML
conversion, out of this task's scope) to import rather than re-derive:

``station_id, begin_time, phase_index, phase_time, phase_score,
phase_type, file_name, phase_amplitude, phase_amp``

-- ``station_id`` is ``NET.STA.LOC.BandInst`` (no final component
character: PhaseNet picks per-station, not per-channel, ARCHITECTURE.md
§5); ``phase_type`` is ``"P"`` or ``"S"``; ``phase_amp`` duplicates
``phase_amplitude`` (PhaseNet's own backward-compatibility alias, kept
verbatim); there is no per-channel attribution and no guarantee of exactly
one P/one S row per station (see ``docs/phasenet_interface.md`` §5 for the
full write-up, including a real example of a spurious double-fired P pick
near a genuine S onset).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openmeq.external import ExternalToolError, phasenet_dir, phasenet_model_dir
from openmeq.external import run as run_external
from openmeq.log import get_logger

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("pick")

#: ``pick/phasenet_raw/picks.csv``'s exact columns, always written with
#: ``--amplitude`` (this module always passes it) -- see the module
#: docstring's "Output" section and ``docs/phasenet_interface.md`` §5.
#: Exposed for **M4.4** (PhaseNet picks -> QuakeML conversion) to import
#: rather than re-derive.
PICKS_CSV_COLUMNS = [
    "station_id",
    "begin_time",
    "phase_index",
    "phase_time",
    "phase_score",
    "phase_type",
    "file_name",
    "phase_amplitude",
    "phase_amp",
]

#: Timeout floor (seconds) for the single ``predict.py`` subprocess call,
#: regardless of how few events are in ``data_list.csv`` -- generous
#: headroom over the ~2.7s fixed model-load cost for a slow/loaded CI box.
_MIN_TIMEOUT_S = 300.0

#: Additional timeout budget (seconds) per event in ``data_list.csv``, on
#: top of :data:`_MIN_TIMEOUT_S`. ``docs/phasenet_interface.md`` §6 measured
#: ~8ms/chunk marginal cost; 2s/event is generous (250x) headroom for a
#: slower machine or larger real chunks than the synthetic fixtures.
_TIMEOUT_PER_EVENT_S = 2.0

#: Number of trailing stderr lines kept in a :class:`PhaseNetRunError`
#: message -- enough to show the actual Python traceback line without
#: dumping an entire (possibly very long) TensorFlow import/warning log.
_STDERR_TAIL_LINES = 20


class PhaseNetRunError(RuntimeError):
    """Raised when the PhaseNet ``predict.py`` batch run fails.

    Covers three distinct failure modes, all surfaced as this one exception
    type so a caller has a single thing to catch:

    1. ``chunks_pn/data_list.csv`` is missing or empty (nothing to run).
    2. The ``predict.py`` subprocess exits non-zero or times out
       (wraps :class:`openmeq.external.ExternalToolError`, carrying its
       captured stderr tail in the message).
    3. ``predict.py`` exits 0 but ``pick/phasenet_raw/picks.csv`` is
       missing, or is missing one of :data:`PICKS_CSV_COLUMNS`.
    """


def _count_events(data_list_csv: Path) -> int:
    """Number of data rows in ``data_list.csv`` (excluding the header)."""
    df = pd.read_csv(data_list_csv)
    return len(df)


def _stderr_tail(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def run_phasenet(paths: RunPaths, cfg: Config) -> Path:
    """Run PhaseNet's ``predict.py`` once over the whole ``chunks_pn/`` batch.

    Builds the exact CLI invocation documented in
    ``docs/phasenet_interface.md`` §2 (M4.1's spike) and runs it as a single
    subprocess via :func:`openmeq.external.run` -- this *is* PLAN.md §4's
    "PhaseNet run: batch" unit of parallel work: PhaseNet's own internal
    ``--batch_size`` batching inside this one process (not
    :func:`openmeq.parallel.pmap`, which M4.2's :func:`openmeq.pick.prep.prepare_all`
    already used for prep). See the module docstring for why prep and
    predict are strictly sequential at this milestone, and why
    ``--batch_size`` is always ``1`` regardless of ``cfg.phasenet.batch_size``.

    Parameters
    ----------
    paths : openmeq.paths.RunPaths
        ``paths.chunks_pn_dir`` (and its ``data_list.csv``) must already
        exist -- normally from :func:`openmeq.pick.prep.prepare_all`.
        Output goes to ``paths.phasenet_raw_dir``
        (``pick/phasenet_raw/``).
    cfg : openmeq.config.Config
        Uses ``cfg.phasenet.min_p_prob``/``min_s_prob``/``target_rate_hz``
        and ``cfg.parallel.phasenet_device``. ``cfg.phasenet.batch_size``
        is *not* used (see module docstring).

    Returns
    -------
    pathlib.Path
        ``paths.phasenet_raw_dir / "picks.csv"``, guaranteed to exist and
        to have every column in :data:`PICKS_CSV_COLUMNS` before this
        function returns successfully.

    Raises
    ------
    PhaseNetRunError
        If ``data_list.csv`` is missing/empty, the subprocess fails or
        times out, or ``picks.csv`` is missing/malformed afterwards. The
        full subprocess stdout/stderr is always logged via ``logger``
        (``get_logger("pick")``, routed to ``logs/pick.log`` once
        ``openmeq.log.configure_logging`` has been called for the ``pick``
        step) regardless of outcome -- :func:`openmeq.external.run` logs it
        before this function ever sees a failure.
    """
    data_list_csv = paths.chunks_pn_dir / "data_list.csv"
    if not data_list_csv.is_file():
        raise PhaseNetRunError(
            f"{data_list_csv} not found -- run openmeq.pick.prep.prepare_all first"
        )

    try:
        n_events = _count_events(data_list_csv)
    except Exception as exc:  # pragma: no cover - pandas parse errors are rare/varied
        raise PhaseNetRunError(f"could not read {data_list_csv}: {exc}") from exc

    if n_events == 0:
        raise PhaseNetRunError(f"{data_list_csv} has no rows -- nothing to pick")

    result_dir = paths.phasenet_raw_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    result_fname = "picks"
    picks_csv = result_dir / f"{result_fname}.csv"

    predict_py = phasenet_dir() / "phasenet" / "predict.py"
    cmd = [
        sys.executable,
        str(predict_py),
        f"--model_dir={phasenet_model_dir()}",
        f"--data_dir={paths.chunks_pn_dir}",
        f"--data_list={data_list_csv}",
        "--format=mseed",
        "--amplitude",
        # --save_prob writes pick/phasenet_raw/result.h5 (one dataset per
        # input file, keyed by its `fname`), for M4.4's probability-trace
        # diagnostics. Negligible overhead (docs/phasenet_interface.md
        # §6). Its one gotcha -- every `fname` in a single predict.py call
        # must be unique, else save_prob_h5 raises -- is satisfied here
        # because chunks_pn/<event_id>/<event_id>.mseed is keyed by
        # event_id, which is unique by construction (ARCHITECTURE.md §4.4).
        "--save_prob",
        f"--min_p_prob={cfg.phasenet.min_p_prob}",
        f"--min_s_prob={cfg.phasenet.min_s_prob}",
        # Always 1 -- see the module docstring's "--batch_size is always 1"
        # section. cfg.phasenet.batch_size is intentionally not read here.
        "--batch_size=1",
        f"--sampling_rate={cfg.phasenet.target_rate_hz}",
        f"--result_dir={result_dir}",
        f"--result_fname={result_fname}",
    ]

    # docs/phasenet_interface.md §1: this machine's GPU is unusable (no
    # matching cuDNN/TensorRT libs), but that's a fact about *this*
    # machine, not something this function should assume about every
    # machine it runs on -- so it follows the bd issue's contract exactly:
    # force CPU-only only when the user's config asked for cpu; otherwise
    # (phasenet_device == "gpu") leave the subprocess's GPU visibility
    # exactly as the parent process's own environment already has it.
    env = {"CUDA_VISIBLE_DEVICES": ""} if cfg.parallel.phasenet_device == "cpu" else None

    timeout = max(_MIN_TIMEOUT_S, _TIMEOUT_PER_EVENT_S * n_events)

    logger.info(
        "Running PhaseNet predict.py once over %d event(s) from %s (timeout=%.0fs, device=%s)",
        n_events,
        data_list_csv,
        timeout,
        cfg.parallel.phasenet_device,
    )
    t0 = time.monotonic()
    try:
        run_external(
            cmd,
            cwd=phasenet_dir() / "phasenet",
            log=logger,
            timeout=timeout,
            env=env,
        )
    except ExternalToolError as exc:
        raise PhaseNetRunError(
            f"PhaseNet predict.py failed (exit {exc.returncode}) over {n_events} event(s) "
            f"from {data_list_csv}; stderr tail:\n{_stderr_tail(exc.stderr)}"
        ) from exc
    elapsed = max(time.monotonic() - t0, 1e-9)

    if not picks_csv.is_file():
        raise PhaseNetRunError(f"PhaseNet predict.py exited 0 but did not write {picks_csv}")

    header_df = pd.read_csv(picks_csv, nrows=0)
    missing = [c for c in PICKS_CSV_COLUMNS if c not in header_df.columns]
    if missing:
        raise PhaseNetRunError(
            f"{picks_csv} is missing expected column(s) {missing} (got {list(header_df.columns)})"
        )

    logger.info(
        "PhaseNet predict.py finished in %.2fs for %d event(s) (%.1f events/s, %.1f ms/event)",
        elapsed,
        n_events,
        n_events / elapsed,
        1000.0 * elapsed / n_events,
    )

    return picks_csv


__all__ = ["PhaseNetRunError", "PICKS_CSV_COLUMNS", "run_phasenet"]
