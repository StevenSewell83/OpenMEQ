"""Run directory layout (``RunPaths``).

``RunPaths(cfg)`` is the single place that knows the on-disk layout of a run
directory (``cfg.project.output_dir``), documented as the literal contract in
``docs/ARCHITECTURE.md`` §3. Every other module reaches a path under the run
directory through a ``RunPaths`` property or method instead of building the
string itself, so the layout can only change in one place.

Design
------
* ``RunPaths`` is thin and mostly declarative: every property/method is a
  pure function of ``self.root`` (``cfg.project.output_dir``, already
  resolved to an absolute path by ``openmeq.config``). Constructing a
  ``RunPaths`` or reading a property never touches the filesystem.
* :meth:`RunPaths.ensure` is the *only* method with a filesystem side effect:
  it creates every directory implied by the layout (never files),
  idempotently (``mkdir(parents=True, exist_ok=True)``), so a step can assume
  its output directories already exist.
* Per-key outputs (one file per event, per step, ...) are exposed as
  *methods* rather than properties, since they take an argument:
  :meth:`chunk_file`, :meth:`chunk_pn_file`, :meth:`review_event_file`,
  :meth:`relocate_cache_file`, :meth:`diagnostics_dir`,
  :meth:`manifest_json`, :meth:`log_file`.
* :meth:`diagnostics_dir` and :meth:`manifest_json` are generic: every step
  gets a ``diagnostics/`` folder and a ``manifest.json`` file in its own
  output folder (ARCHITECTURE.md §3's ``# every step`` rows). They take the
  step's *output key* (``"qc"``, ``"detections"``, ``"chunks"``,
  ``"chunks_pn"``, ``"pick"``, ``"locate"``, ``"review"``, ``"magnitude"``,
  ``"relocate"``), which is not always identical to the CLI subcommand name
  (the ``detect`` subcommand's key is ``detections``, and the ``chunk``
  subcommand's outputs are split across ``chunks`` and ``chunks_pn``).
  :meth:`log_file`, by contrast, always takes the CLI subcommand's own step
  name, since ``logs/<step>.log`` is keyed on that name directly
  (ARCHITECTURE.md §3, PLAN.md §3).
* Folder names (OpenMEQ-9wj): each output key maps to a numbered folder via
  :data:`STEP_DIRS`, following the MicroQuake run layout
  (``projects/RK-NM/run_2024_2026``) so the same stage carries the same
  number in both tools' runs: ``1_qc``, ``2_detection``, ``4_data_chunks``,
  ``5_ml_picks``, ``7_location``, ``8_relocation``, ``9_plots``. OpenMEQ has
  no separate duplicate-removal (3) or pick-refinement (6) stage, so those
  numbers are unused. Docstrings elsewhere that say e.g. ``qc/availability.csv``
  mean "the ``qc`` step's folder", not a literal ``qc/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmeq.config import Config

#: Step output key -> folder (relative to the run root). Every key gets a
#: ``diagnostics/`` subfolder and a ``manifest.json``, per ARCHITECTURE.md
#: §3's "every step" rows, so no two keys may share a folder: hence
#: ``chunks``/``chunks_pn`` as siblings under ``4_data_chunks/``, and
#: ``magnitude`` in its own subfolder of ``8_relocation/`` (the MicroQuake
#: layout keeps magnitudes with relocation). ``qc`` and ``detections`` also
#: get a dedicated named property (:attr:`RunPaths.qc_diagnostics_dir`,
#: :attr:`RunPaths.detect_diagnostics_dir`); the rest are reached through
#: :meth:`RunPaths.diagnostics_dir`.
STEP_DIRS: dict[str, str] = {
    "qc": "1_qc",
    "detections": "2_detection",
    "chunks": "4_data_chunks/chunks",
    "chunks_pn": "4_data_chunks/chunks_pn",
    "pick": "5_ml_picks",
    "locate": "7_location",
    "nlloc": "7_location/nlloc",
    "review": "7b_review",
    "relocate": "8_relocation",
    "magnitude": "8_relocation/magnitude",
}

#: Station inventory, file index and metadata report.
SETUP_DIR = "0_setup"
#: Run-level (cross-step) diagnostics.
PLOTS_DIR = "9_plots"


class RunPaths:
    """Every path under a run's output directory (``cfg.project.output_dir``).

    Parameters
    ----------
    cfg : openmeq.config.Config
        Only ``cfg.project.output_dir`` is used.

    Attributes
    ----------
    root : pathlib.Path
        ``cfg.project.output_dir``, the base of every path below.
    """

    def __init__(self, cfg: Config) -> None:
        self.root = Path(cfg.project.output_dir)

    def step_dir(self, step: str) -> Path:
        """A step's output folder, from its output key (see :data:`STEP_DIRS`).

        An unknown key falls back to a folder of that name, so an ad-hoc
        ``diagnostics_dir``/``manifest_json`` key never raises.
        """
        return self.root / STEP_DIRS.get(step, step)

    @property
    def setup_dir(self) -> Path:
        return self.root / SETUP_DIR

    @property
    def config_snapshot_yaml(self) -> Path:
        """``config_snapshot.yaml`` -- copy of the config the last step ran with."""
        return self.root / "config_snapshot.yaml"

    # -- inventory (Step 1 / M1) ------------------------------------------------

    @property
    def index_json(self) -> Path:
        return self.setup_dir / "index.json"

    @property
    def stations_xml(self) -> Path:
        return self.setup_dir / "stations.xml"

    @property
    def stations_csv(self) -> Path:
        return self.setup_dir / "stations.csv"

    @property
    def metadata_report_json(self) -> Path:
        return self.setup_dir / "metadata_report.json"

    # -- qc (Step 1 / M1) --------------------------------------------------------

    @property
    def availability_csv(self) -> Path:
        return self.step_dir("qc") / "availability.csv"

    @property
    def daily_stats_csv(self) -> Path:
        return self.step_dir("qc") / "daily_stats.csv"

    @property
    def monthly_stats_csv(self) -> Path:
        return self.step_dir("qc") / "monthly_stats.csv"

    @property
    def channel_flags_json(self) -> Path:
        return self.step_dir("qc") / "channel_flags.json"

    @property
    def qc_diagnostics_dir(self) -> Path:
        return self.diagnostics_dir("qc")

    @property
    def noise_daily_csv(self) -> Path:
        """Per channel-day hourly noise statistics with instrument group (OpenMEQ-h8j)."""
        return self.step_dir("qc") / "noise_daily.csv"

    @property
    def qc_noise_diagnostics_dir(self) -> Path:
        """Per-instrument-group noise figures (OpenMEQ-h8j)."""
        return self.qc_diagnostics_dir / "noise"

    @property
    def qc_ppsd_dir(self) -> Path:
        """Per channel-day PPSD ``.npz`` files, ``<seed_id>/<date>.npz`` (OpenMEQ-itl)."""
        return self.step_dir("qc") / "ppsd"

    @property
    def qc_ppsd_diagnostics_dir(self) -> Path:
        """Per-channel PPSD figures (OpenMEQ-itl)."""
        return self.qc_diagnostics_dir / "ppsd"

    # -- detections (Step 2 / M2, ``openmeq detect``) -----------------------------

    @property
    def detections_csv(self) -> Path:
        return self.step_dir("detections") / "detections.csv"

    @property
    def detect_diagnostics_dir(self) -> Path:
        return self.diagnostics_dir("detections")

    # -- chunks / chunks_pn (Steps 3-4 / M3-M4) -----------------------------------

    @property
    def chunk_dir(self) -> Path:
        """Parent of every per-event raw chunk dir (``chunks/<event_id>/``)."""
        return self.step_dir("chunks")

    @property
    def chunk_completeness_csv(self) -> Path:
        return self.chunk_dir / "completeness.csv"

    @property
    def chunks_pn_dir(self) -> Path:
        """Parent of every per-event PhaseNet-ready chunk dir (``chunks_pn/<event_id>/``)."""
        return self.step_dir("chunks_pn")

    @property
    def data_list_csv(self) -> Path:
        return self.chunks_pn_dir / "data_list.csv"

    @property
    def prep_report_csv(self) -> Path:
        """PhaseNet chunk-prep zero-fill report (M4.2 / OpenMEQ-1m7.2, ARCHITECTURE.md §3)."""
        return self.chunks_pn_dir / "prep_report.csv"

    # -- pick (Step 4 / M4) --------------------------------------------------------

    @property
    def phasenet_raw_dir(self) -> Path:
        return self.step_dir("pick") / "phasenet_raw"

    @property
    def catalog_picks_xml(self) -> Path:
        return self.step_dir("pick") / "catalog_picks.xml"

    # -- nlloc setup (M5.2, ``scripts/setup_nlloc.py`` / ``openmeq.nlloc_setup``) ----
    # Not one of the registered pipeline steps in ``openmeq/cli.py`` (it is a
    # one-off utility script per PLAN.md §7, also run by ``openmeq locate``),
    # but it keeps its own ``manifest.json``, so it has a ``STEP_DIRS`` key
    # (``"nlloc"``) placing it inside ``7_location/``.

    @property
    def nlloc_dir(self) -> Path:
        """NLLoc setup output directory (``nlloc/``, inside the locate step's folder)."""
        return self.step_dir("nlloc")

    @property
    def nlloc_velocity_models_dir(self) -> Path:
        """This run's NLLoc model root -- what ``link_obspyck_model`` symlinks
        as ObsPyck's own ``pluginpath/VELOCITY_MODELS`` (docs/obspyck_notes.md §5)."""
        return self.nlloc_dir / "VELOCITY_MODELS"

    @property
    def nlloc_grid_root(self) -> Path:
        """Shared ``VGOUT``/``GTFILES`` file-root basename (no extension).

        Both the Vel2Grid velocity grid (``<root>.{P,S}.mod.{hdr,buf}``) and
        every station's Grid2Time travel-time grid
        (``<root>.{P,S}.<STA>.time.{hdr,buf}``) live side by side under
        ``nlloc_velocity_models_dir/layer_time/``, matching the layout
        ObsPyck's own generated control file expects
        (docs/obspyck_notes.md §5).
        """
        return self.nlloc_velocity_models_dir / "layer_time" / "layer"

    # -- locate (Step 5 / M6) -------------------------------------------------------

    @property
    def passA_dir(self) -> Path:
        return self.step_dir("locate") / "passA"

    @property
    def passB_dir(self) -> Path:
        return self.step_dir("locate") / "passB"

    @property
    def catalog_auto_xml(self) -> Path:
        return self.step_dir("locate") / "catalog_auto.xml"

    @property
    def locate_summary_csv(self) -> Path:
        return self.step_dir("locate") / "summary.csv"

    @property
    def removal_log_csv(self) -> Path:
        return self.step_dir("locate") / "removal_log.csv"

    # -- review (Step 6 / M7) --------------------------------------------------------

    @property
    def review_state_json(self) -> Path:
        return self.step_dir("review") / "state.json"

    @property
    def review_events_dir(self) -> Path:
        return self.step_dir("review") / "events"

    @property
    def catalog_final_xml(self) -> Path:
        return self.step_dir("review") / "catalog_final.xml"

    @property
    def review_stations_xml(self) -> Path:
        """ObsPyck's StationXML (OpenMEQ-tch): ``0_setup/stations.xml`` with a
        labelled unit-sensitivity placeholder on every channel lacking a
        response (:func:`openmeq.review.batch.write_review_inventory`)."""
        return self.step_dir("review") / "stations_review.xml"

    def review_work_dir(self, event_id: str) -> Path:
        """Per-event ObsPyck subprocess working directory
        (``review/work/<event_id>/``, ARCHITECTURE.md §3, added by M7.3).

        This is the batch review loop's (:mod:`openmeq.review.batch`) chosen
        ``cwd`` for launching ``python -m openmeq.review._launch_obspyck``:
        ObsPyck's "save locally" writes ``obspyck_<timestamp>.xml`` into its
        own current working directory, so each event gets an isolated
        directory the loop can snapshot before launch and diff after exit to
        detect a save (never trusting the process exit code alone, see the
        module's own docstring). Deliberately **not** cleaned up between runs
        (kept for debugging a bad review session) -- only stale
        ``obspyck_*.xml`` files from a previous crashed attempt at the same
        event are removed, and only immediately before relaunching.
        """
        return self.step_dir("review") / "work" / event_id

    # -- magnitude (Step 7 / M8) -------------------------------------------------------

    @property
    def catalog_mag_xml(self) -> Path:
        return self.step_dir("magnitude") / "catalog_mag.xml"

    @property
    def station_magnitudes_csv(self) -> Path:
        return self.step_dir("magnitude") / "station_magnitudes.csv"

    # -- relocate / hypoDD (Step 8 / M10) -----------------------------------------------

    @property
    def relocate_work_dir(self) -> Path:
        return self.step_dir("relocate") / "work"

    @property
    def relocate_cache_dir(self) -> Path:
        return self.step_dir("relocate") / "cache"

    @property
    def dt_ct(self) -> Path:
        """``relocate/work/dt.ct`` (ph2dt's output, M10.2 / OpenMEQ-16d.2).

        Lives in :attr:`relocate_work_dir`, not a separate top-level
        ``relocate/dt.ct`` -- ``hypoDD.inp``'s ``dt.ct``/``dt.cc`` lines are
        bare filenames read relative to hypoDD's own ``cwd`` (§6c), so
        ``dt.ct``/``dt.cc`` are colocated with ``event.sel``/``station.sel``/
        ``hypoDD.inp`` in one working directory rather than duplicated
        (ARCHITECTURE.md §3).
        """
        return self.relocate_work_dir / "dt.ct"

    @property
    def dt_cc(self) -> Path:
        """``relocate/work/dt.cc`` (M10.3's cross-correlation output). See :attr:`dt_ct`."""
        return self.relocate_work_dir / "dt.cc"

    @property
    def xcorr_results_jsonl(self) -> Path:
        """``relocate/work/xcorr_results.jsonl`` (M10.3 / OpenMEQ-16d.3): the
        ``openmeq.parallel.pmap_incremental`` resumable results file
        ``openmeq.relocate.xcorr.run_xcorr`` assembles ``dt.cc`` from -- one
        JSON-lines record per cross-correlated pair, so a killed/interrupted
        run resumes without recomputing already-written pairs.
        """
        return self.relocate_work_dir / "xcorr_results.jsonl"

    @property
    def catalog_dd_xml(self) -> Path:
        return self.step_dir("relocate") / "catalog_dd.xml"

    # -- run-level diagnostics (OpenMEQ-02o.8, ``openmeq.run_diagnostics``) ------------

    @property
    def run_diagnostics_dir(self) -> Path:
        """Run-level ``9_plots/`` -- cross-step figures (e.g. the event
        funnel), as opposed to every step's own :meth:`diagnostics_dir`.
        Not part of :meth:`ensure`; its writer creates it on demand."""
        return self.root / PLOTS_DIR

    # -- per-key helper methods ----------------------------------------------------

    def chunk_file(self, event_id: str) -> Path:
        """Raw waveform chunk for one event (``chunks/<event_id>/<event_id>.mseed``)."""
        return self.chunk_dir / event_id / f"{event_id}.mseed"

    def chunk_pn_file(self, event_id: str) -> Path:
        """PhaseNet-ready chunk for one event (``chunks_pn/<event_id>/<event_id>.mseed``)."""
        return self.chunks_pn_dir / event_id / f"{event_id}.mseed"

    def review_event_file(self, event_id: str) -> Path:
        """Reviewed QuakeML for one event (``review/events/<event_id>.xml``)."""
        return self.review_events_dir / f"{event_id}.xml"

    def relocate_cache_file(self, event_id: str) -> Path:
        """Pre-filtered waveform cache for one event (``relocate/cache/<event_id>.npz``)."""
        return self.relocate_cache_dir / f"{event_id}.npz"

    def locate_pass_dir(self, pass_name: str, event_id: str) -> Path:
        """Per-event NLLoc work directory root for one pass (M6.1 / OpenMEQ-xdf.1).

        ``locate/passA/<event_id>/`` or ``locate/passB/<event_id>/``
        (ARCHITECTURE.md §3). ``openmeq.locate.nlloc_runner.locate_event``
        appends its own ``iter_NN`` subdirectory
        (``locate_pass_dir(pass_name, event_id) / f"iter_{iteration:02d}"``),
        one per location iteration within that pass.

        Parameters
        ----------
        pass_name : str
            ``"passA"`` or ``"passB"``.
        event_id : str

        Raises
        ------
        ValueError
            If ``pass_name`` is not ``"passA"`` or ``"passB"``.
        """
        if pass_name == "passA":
            base = self.passA_dir
        elif pass_name == "passB":
            base = self.passB_dir
        else:
            raise ValueError(f"pass_name must be 'passA' or 'passB', got {pass_name!r}")
        return base / event_id

    def diagnostics_dir(self, step: str) -> Path:
        """``diagnostics/`` inside a step's folder -- every step's diagnostics folder.

        ``step`` is that step's output key (see the module docstring and
        :data:`STEP_DIRS`), e.g. ``"qc"``, ``"detections"``, ``"chunks"``,
        ``"chunks_pn"``, ``"pick"``, ``"locate"``, ``"review"``,
        ``"magnitude"`` or ``"relocate"``.
        """
        return self.step_dir(step) / "diagnostics"

    def manifest_json(self, step: str) -> Path:
        """A step's :class:`openmeq.manifest.StepManifest` file (``manifest.json`` in its folder).

        ``step`` follows the same convention as :meth:`diagnostics_dir`.
        """
        return self.step_dir(step) / "manifest.json"

    def log_file(self, step: str) -> Path:
        """``logs/<step>.log`` -- the CLI subcommand's log file.

        Unlike :meth:`diagnostics_dir`/:meth:`manifest_json`, ``step`` here
        is always the CLI subcommand name (``qc``, ``detect``, ``chunk``,
        ``pick``, ``locate``, ``review``, ``magnitude``, ``relocate``,
        ``run``), matching ``openmeq.log.configure_logging``'s usage from
        ``openmeq/cli.py``.
        """
        return self.root / "logs" / f"{step}.log"

    # -- setup -----------------------------------------------------------------------

    def ensure(self) -> None:
        """Create every directory implied by this layout (never files).

        Idempotent: safe to call at the start of every step (and is called
        exactly there, by ``openmeq/cli.py``, before a step's body runs).
        """
        dirs = [
            self.root,
            self.setup_dir,
            self.qc_diagnostics_dir,
            self.detect_diagnostics_dir,
            self.chunk_dir,
            self.diagnostics_dir("chunks"),
            self.chunks_pn_dir,
            self.diagnostics_dir("chunks_pn"),
            self.phasenet_raw_dir,
            self.diagnostics_dir("pick"),
            self.passA_dir,
            self.passB_dir,
            self.diagnostics_dir("locate"),
            self.review_events_dir,
            self.diagnostics_dir("review"),
            self.catalog_mag_xml.parent,
            self.diagnostics_dir("magnitude"),
            self.relocate_work_dir,
            self.relocate_cache_dir,
            self.diagnostics_dir("relocate"),
            self.root / "logs",
        ]
        for step in STEP_DIRS:
            dirs.append(self.step_dir(step))
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
