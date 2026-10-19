"""Optional probabilistic power spectral densities per channel (OpenMEQ-itl).

ObsPy's :class:`~obspy.signal.PPSD` (McNamara & Buland 2004) for every
selected channel, as an optional part of ``openmeq qc``
(``cfg.qc.ppsd.enabled``) or on its own with ``openmeq diagnostics ppsd``.

Computation (resumable)
------------------------
One :func:`openmeq.parallel.pmap` task per channel x batch of
``cfg.parallel.chunk_days`` days. Each channel-day is read through the
waveform index (``0_setup/index.json``), its PSDs computed against the
merged StationXML (``0_setup/stations.xml``, which must carry instrument
responses), and saved as ``1_qc/ppsd/<seed_id>/<YYYY-MM-DD>.npz``
(``PPSD.save_npz``). A day with no data, or too little for one
``ppsd_length_s`` segment, gets an empty ``<date>.empty`` marker instead.
Existing files are skipped, so a rerun only computes new days.

``1_qc/ppsd/settings.json`` records the computation settings (every
``qc.ppsd`` key except the plotting ones) and a hash of the merged
StationXML. When either changes -- or with ``force`` -- every stored day is
deleted and recomputed, so the files always match the current settings and
responses. Channels with no response are skipped (not marked, so they are
retried once their metadata is fixed) and listed in the log.

Figures (``1_qc/diagnostics/ppsd/``)
-------------------------------------
Per channel, from all of its days combined:

- ``<seed_id>_ppsd.png``: the PPSD histogram with mode, NLNM/NHNM noise
  models and data coverage;
- ``<seed_id>_spectrogram.png``: every PSD through time;
- ``<seed_id>_temporal.png``: PSD level through time at
  ``qc.ppsd.temporal_periods_s`` (periods outside the PSD's range dropped).

And ``ppsd_medians.png``: each channel's median PSD over the run, one panel
per instrument group (``noise_daily.csv``'s ``group``), against the noise
models -- the station-to-station comparison. Stations in
``qc.plot_exclude_stations`` are computed but not drawn.

``qc.ppsd`` is ``exclude=True`` in the config model, so it is not part of
the qc step's skip-unchanged fingerprint: enabling it computes PPSDs on the
next ``openmeq qc`` without recomputing the rest of QC.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_inventory

from openmeq.index import DataIndex
from openmeq.log import get_logger
from openmeq.parallel import pmap
from openmeq.qc.availability import date_range_inclusive

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("qc.ppsd")

#: ``qc.ppsd`` keys that change the stored PSDs (the rest only affect plots).
COMPUTE_KEYS = (
    "ppsd_length_s",
    "overlap",
    "db_bins",
    "period_limits",
    "period_smoothing_width_octaves",
    "period_step_octaves",
    "skip_on_gaps",
)

_SETTINGS_FILE = "settings.json"
_SETTINGS_VERSION = 1


# ---------------------------------------------------------------------------
# Settings stamp and channel selection
# ---------------------------------------------------------------------------


def _settings_stamp(cfg: Config, paths: RunPaths) -> dict:
    ppsd = cfg.qc.ppsd.model_dump(mode="json")
    digest = hashlib.sha256(Path(paths.stations_xml).read_bytes()).hexdigest()
    return {
        "version": _SETTINGS_VERSION,
        "ppsd": {k: ppsd[k] for k in COMPUTE_KEYS},
        "stations_xml_sha256": digest,
    }


def _prepare_store(ppsd_dir: Path, stamp: dict, force: bool) -> None:
    """Clear stored days if ``force`` or the settings/metadata changed."""
    settings_path = ppsd_dir / _SETTINGS_FILE
    previous = None
    if settings_path.is_file():
        try:
            previous = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    if ppsd_dir.is_dir() and (force or previous != stamp):
        reason = "--force" if force else "settings or station metadata changed"
        logger.info("qc.ppsd: clearing stored PPSD days in %s (%s)", ppsd_dir, reason)
        for child in ppsd_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    ppsd_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")


def select_seed_ids(seed_ids: list[str], cfg: Config) -> list[str]:
    """Channels to compute: ``qc.ppsd.components`` filter, minus ``qc.exclude_channels``."""
    components = cfg.qc.ppsd.components
    excluded = set(cfg.qc.exclude_channels)
    return sorted(
        s
        for s in seed_ids
        if s not in excluded and (not components or any(s.endswith(c) for c in components))
    )


def _plot_excluded(seed_id: str, exclude: list[str]) -> bool:
    net, sta = seed_id.split(".")[:2]
    return sta in exclude or f"{net}.{sta}" in exclude


# ---------------------------------------------------------------------------
# Per channel-day computation (pmap workers)
# ---------------------------------------------------------------------------

#: Per-worker-process caches (see openmeq.qc.stats for the same pattern).
_worker_index_cache: dict[str, DataIndex] = {}
_worker_inv_cache: dict = {}


def _cached(cache: dict, path: str, loader):
    value = cache.get(path)
    if value is None:
        value = loader(path)
        cache[path] = value
    return value


def _day_files(out_dir: Path, seed_id: str, day_iso: str) -> tuple[Path, Path]:
    base = out_dir / seed_id / day_iso
    return base.with_suffix(".npz"), base.with_suffix(".empty")


def ppsd_channel_day(
    seed_id: str, day_iso: str, index: DataIndex, inv, params: dict, out_dir: Path
) -> str:
    """Compute and store one channel-day's PSDs; returns a status string.

    ``"ok"``, ``"done"`` (already stored), ``"no_data"``, ``"no_segments"``
    (data shorter than one segment), ``"no_response"`` or ``"error: ..."``.
    """
    from obspy.signal import PPSD

    npz, empty = _day_files(out_dir, seed_id, day_iso)
    if npz.exists() or empty.exists():
        return "done"
    t1 = UTCDateTime(day_iso)
    t2 = t1 + 86400.0
    try:
        inv.get_response(seed_id, t1 + 43200.0)
    except Exception:  # noqa: BLE001 - obspy raises plain Exception for a missing response
        return "no_response"
    npz.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = index.read([seed_id], t1, t2)
        if not len(stream):
            empty.touch()
            return "no_data"
        ppsd = PPSD(
            stream[0].stats,
            metadata=inv,
            skip_on_gaps=params["skip_on_gaps"],
            db_bins=tuple(params["db_bins"]),
            ppsd_length=params["ppsd_length_s"],
            overlap=params["overlap"],
            period_smoothing_width_octaves=params["period_smoothing_width_octaves"],
            period_step_octaves=params["period_step_octaves"],
            period_limits=params["period_limits"],
        )
        ppsd.add(stream)
        if not ppsd.times_processed:
            empty.touch()
            return "no_segments"
        tmp = npz.with_name(f"{day_iso}.tmp.npz")
        ppsd.save_npz(str(tmp))
        os.replace(tmp, npz)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - one bad channel-day must not stop the run
        return f"error: {exc}"


def _ppsd_batch(item: tuple) -> list[tuple[str, str, str]]:
    """``pmap`` worker: one channel x a batch of days."""
    seed_id, days_iso, index_path, inv_path, params, out_dir = item
    index = _cached(_worker_index_cache, index_path, DataIndex.load)
    inv = _cached(_worker_inv_cache, inv_path, read_inventory)
    return [
        (seed_id, day, ppsd_channel_day(seed_id, day, index, inv, params, Path(out_dir)))
        for day in days_iso
    ]


def compute_ppsd(cfg: Config, paths: RunPaths, *, force: bool = False) -> dict[str, int]:
    """Compute every missing channel-day PSD (module docstring); returns status counts.

    Raises
    ------
    FileNotFoundError
        If the waveform index or merged StationXML is missing (run ``openmeq
        qc`` first).
    """
    missing = [str(p) for p in (paths.index_json, paths.stations_xml) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"qc.ppsd needs {missing}; run `openmeq qc` first")

    ppsd_dir = Path(paths.qc_ppsd_dir)
    _prepare_store(ppsd_dir, _settings_stamp(cfg, paths), force)

    index = DataIndex.load(paths.index_json)
    seed_ids = select_seed_ids(index.seed_ids(), cfg)
    days = [d.isoformat() for d in date_range_inclusive(cfg.data.start, cfg.data.end)]
    todo = {
        s: [d for d in days if not any(f.exists() for f in _day_files(ppsd_dir, s, d))]
        for s in seed_ids
    }
    params = {k: cfg.qc.ppsd.model_dump(mode="json")[k] for k in COMPUTE_KEYS}
    chunk = max(1, cfg.parallel.chunk_days)
    items = [
        (
            s,
            todo_days[i : i + chunk],
            str(paths.index_json),
            str(paths.stations_xml),
            params,
            str(ppsd_dir),
        )
        for s, todo_days in todo.items()
        for i in range(0, len(todo_days), chunk)
    ]
    n_todo = sum(len(v) for v in todo.values())
    logger.info(
        "qc.ppsd: %d channel(s) x %d day(s); %d channel-day(s) to compute (%d task(s))",
        len(seed_ids),
        len(days),
        n_todo,
        len(items),
    )
    results = [r for batch in pmap(_ppsd_batch, items, cfg.parallel, desc="qc-ppsd") for r in batch]

    counts: dict[str, int] = {}
    no_response: set[str] = set()
    for seed_id, day, status in results:
        key = "error" if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        if status == "no_response":
            no_response.add(seed_id)
        elif key == "error":
            logger.warning("qc.ppsd: %s %s: %s", seed_id, day, status)
    if no_response:
        logger.warning(
            "qc.ppsd: no instrument response for %d channel(s), skipped: %s",
            len(no_response),
            ", ".join(sorted(no_response)),
        )
    logger.info("qc.ppsd: computed %s", counts or "nothing new")
    return counts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_channel(item: tuple) -> dict | None:
    """``pmap`` worker: combine one channel's days and draw its figures.

    Returns the channel's median PSD (for ``ppsd_medians.png``) and the
    figures written, or ``None`` if it has no stored PSDs.
    """
    seed_id, files, out_dir, plot = item
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from obspy.signal import PPSD

    if not files:
        return None
    ppsd = PPSD.load_npz(files[0])
    for f in files[1:]:
        ppsd.add_npz(f)
    out_dir = Path(out_dir)
    periods = ppsd.period_bin_centers
    period_lim = (float(periods.min()), float(periods.max()))
    written = []

    path = out_dir / f"{seed_id}_ppsd.png"
    ppsd.plot(
        filename=str(path),
        show=False,
        show_mode=True,
        show_noise_models=True,
        xaxis_frequency=plot["xaxis_frequency"],
        period_lim=(1.0 / period_lim[1], 1.0 / period_lim[0])
        if plot["xaxis_frequency"]
        else period_lim,
    )
    written.append(str(path))
    plt.close("all")

    path = out_dir / f"{seed_id}_spectrogram.png"
    ppsd.plot_spectrogram(filename=str(path), show=False)
    written.append(str(path))
    plt.close("all")

    temporal = [p for p in plot["temporal_periods_s"] if period_lim[0] <= p <= period_lim[1]]
    if temporal:
        path = out_dir / f"{seed_id}_temporal.png"
        fig = ppsd.plot_temporal(temporal, show=False)
        fig.axes[0].set_title(f"{seed_id}: PSD through time at selected periods", loc="left")
        fig.savefig(str(path))
        written.append(str(path))
        plt.close("all")

    med_periods, med_db = ppsd.get_percentile(percentile=50)
    return {
        "seed_id": seed_id,
        "periods": np.asarray(med_periods).tolist(),
        "median_db": np.asarray(med_db).tolist(),
        "n_psd": len(ppsd.times_processed),
        "written": written,
    }


def _plot_medians(medians: list[dict], groups: dict[str, str], path: Path, cfg: Config) -> None:
    from openmeq.plotting import save_fig, setup_style
    from openmeq.plotting._style import COLORBLIND_PALETTE

    setup_style()
    import matplotlib.pyplot as plt
    from obspy.signal.spectral_estimation import get_nhnm, get_nlnm

    by_group: dict[str, list[dict]] = {}
    for m in medians:
        by_group.setdefault(groups.get(m["seed_id"], "all"), []).append(m)
    names = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    fig, axes = plt.subplots(
        len(names), 1, figsize=(11, 4.2 * len(names)), squeeze=False, sharex=True
    )
    freq = cfg.qc.ppsd.xaxis_frequency
    nlnm_p, nlnm = get_nlnm()
    nhnm_p, nhnm = get_nhnm()
    for ax, name in zip(axes[:, 0], names, strict=True):
        for i, m in enumerate(sorted(by_group[name], key=lambda m: m["seed_id"])):
            x = 1.0 / np.asarray(m["periods"]) if freq else np.asarray(m["periods"])
            label = ".".join(m["seed_id"].split(".")[1::2])  # STA.CHA
            ax.plot(
                x,
                m["median_db"],
                color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)],
                ls=["-", "--", ":"][(i // len(COLORBLIND_PALETTE)) % 3],
                lw=1.3,
                label=label,
            )
        for p, v in ((nlnm_p, nlnm), (nhnm_p, nhnm)):
            ax.plot(1.0 / p if freq else p, v, color="0.5", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylabel("power (dB re 1 (m/s²)²/Hz)")
        ax.set_title(f"{name}: median PSD per channel (grey: NLNM/NHNM)", loc="left")
        ax.legend(
            fontsize=7,
            ncol=1 + len(by_group[name]) // 24,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
        ax.grid(alpha=0.3, which="both")
    axes[-1, 0].set_xlabel("frequency (Hz)" if freq else "period (s)")
    all_x = np.concatenate(
        [1.0 / np.asarray(m["periods"]) if freq else np.asarray(m["periods"]) for m in medians]
    )
    axes[-1, 0].set_xlim(all_x.min(), all_x.max())
    fig.suptitle(f"{cfg.project.name} — PPSD medians over the run")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, path)


def plot_ppsd(cfg: Config, paths: RunPaths, instruments: pd.DataFrame | None = None) -> list[Path]:
    """Draw every channel's PPSD figures plus ``ppsd_medians.png``; returns the paths.

    ``instruments`` (``seed_id``, ``group``; e.g. ``noise_daily.csv``) groups
    the medians figure; defaults to reading ``noise_daily.csv`` if present.
    Figures from an earlier run that weren't redrawn are removed.
    """
    ppsd_dir = Path(paths.qc_ppsd_dir)
    out_dir = Path(paths.qc_ppsd_diagnostics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = cfg.qc.plot_exclude_stations
    plot = {
        "xaxis_frequency": cfg.qc.ppsd.xaxis_frequency,
        "temporal_periods_s": list(cfg.qc.ppsd.temporal_periods_s),
    }
    items = []
    if ppsd_dir.is_dir():
        for chan_dir in sorted(p for p in ppsd_dir.iterdir() if p.is_dir()):
            if _plot_excluded(chan_dir.name, exclude):
                continue
            files = sorted(str(f) for f in chan_dir.glob("*.npz") if ".tmp." not in f.name)
            if files:
                items.append((chan_dir.name, files, str(out_dir), plot))
    results = [r for r in pmap(_plot_channel, items, cfg.parallel, desc="qc-ppsd-plots") if r]

    written = [Path(w) for r in results for w in r["written"]]
    if results:
        if instruments is None and Path(paths.noise_daily_csv).is_file():
            instruments = pd.read_csv(paths.noise_daily_csv, usecols=["seed_id", "group"])
        groups = {}
        if instruments is not None and "group" in instruments:
            groups = (
                instruments.drop_duplicates("seed_id").set_index("seed_id")["group"].astype(str)
            ).to_dict()
        medians_path = out_dir / "ppsd_medians.png"
        _plot_medians(results, groups, medians_path, cfg)
        written.append(medians_path)

    keep = {p.name for p in written}
    for old in out_dir.glob("*.png"):
        if old.name not in keep:
            old.unlink()
    logger.info("qc.ppsd: %d figure(s) in %s", len(written), out_dir)
    return written


def run_ppsd(
    cfg: Config,
    paths: RunPaths,
    *,
    force: bool = False,
    instruments: pd.DataFrame | None = None,
) -> list[Path]:
    """Compute any missing PSDs, then redraw the PPSD figures."""
    compute_ppsd(cfg, paths, force=force)
    return plot_ppsd(cfg, paths, instruments)


__all__ = [
    "COMPUTE_KEYS",
    "compute_ppsd",
    "plot_ppsd",
    "ppsd_channel_day",
    "run_ppsd",
    "select_seed_ids",
]
