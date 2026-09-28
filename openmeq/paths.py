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
  gets a ``<step>/diagnostics/`` folder and a ``<step>/manifest.json`` file
  (ARCHITECTURE.md §3's ``# every step`` rows), but ``<step>`` there is the
  step's own *top-level output folder name* (e.g. ``"qc"``, ``"detections"``,
  ``"chunks"``, ``"chunks_pn"``, ``"pick"``, ``"locate"``, ``"review"``,
  ``"magnitude"``, ``"relocate"``), which is not always identical to the CLI
  subcommand name (the ``detect`` subcommand's outputs live under
  ``detections/``, and the ``chunk`` subcommand's outputs are split across
  ``chunks/`` and ``chunks_pn/``). :meth:`log_file`, by contrast, always
  takes the CLI subcommand's own step name, since ``logs/<step>.log`` is
  keyed on that name directly (ARCHITECTURE.md §3, PLAN.md §3).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmeq.config import Config

#: Top-level output folders that get a ``<name>/diagnostics/`` subfolder,
#: per ARCHITECTURE.md §3's "``<step>/diagnostics/`` # every step" row.
#: ``qc`` and ``detections`` also get a dedicated named property
#: (:attr:`RunPaths.qc_diagnostics_dir`, :attr:`RunPaths.detect_diagnostics_dir`);
#: the rest are reached through :meth:`RunPaths.diagnostics_dir`.
_STEP_OUTPUT_DIRS = (
    "qc",
    "detections",
    "chunks",
    "chunks_pn",
    "pick",
    "locate",
    "review",
    "magnitude",
    "relocate",
)


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

    # -- inventory (Step 1 / M1) ------------------------------------------------

    @property
    def index_json(self) -> Path:
        return self.root / "inventory" / "index.json"

    @property
    def stations_xml(self) -> Path:
        return self.root / "inventory" / "stations.xml"

    @property
    def stations_csv(self) -> Path:
        return self.root / "inventory" / "stations.csv"

    @property
    def metadata_report_json(self) -> Path:
        return self.root / "inventory" / "metadata_report.json"

    # -- qc (Step 1 / M1) --------------------------------------------------------

    @property
    def availability_csv(self) -> Path:
        return self.root / "qc" / "availability.csv"

    @property
    def daily_stats_csv(self) -> Path:
        return self.root / "qc" / "daily_stats.csv"

    @property
    def monthly_stats_csv(self) -> Path:
        return self.root / "qc" / "monthly_stats.csv"

    @property
    def channel_flags_json(self) -> Path:
        return self.root / "qc" / "channel_flags.json"

    @property
    def qc_diagnostics_dir(self) -> Path:
        return self.diagnostics_dir("qc")

    # -- detections (Step 2 / M2, ``openmeq detect``) -----------------------------

    @property
    def detections_csv(self) -> Path:
        return self.root / "detections" / "detections.csv"

    @property
    def detect_diagnostics_dir(self) -> Path:
        return self.diagnostics_dir("detections")

    # -- chunks / chunks_pn (Steps 3-4 / M3-M4) -----------------------------------

    @property
    def chunk_dir(self) -> Path:
        """Parent of every per-event raw chunk dir (``chunks/<event_id>/``)."""
        return self.root / "chunks"

    @property
    def chunk_completeness_csv(self) -> Path:
        return self.root / "chunks" / "completeness.csv"

    @property
    def chunks_pn_dir(self) -> Path:
        """Parent of every per-event PhaseNet-ready chunk dir (``chunks_pn/<event_id>/``)."""
        return self.root / "chunks_pn"

    @property
    def data_list_csv(self) -> Path:
        return self.root / "chunks_pn" / "data_list.csv"

    # -- pick (Step 4 / M4) --------------------------------------------------------

    @property
    def phasenet_raw_dir(self) -> Path:
        return self.root / "pick" / "phasenet_raw"

    @property
    def catalog_picks_xml(self) -> Path:
        return self.root / "pick" / "catalog_picks.xml"

    # -- nlloc setup (M5.2, ``scripts/setup_nlloc.py`` / ``openmeq.nlloc_setup``) ----
    # Not one of the registered pipeline steps in ``openmeq/cli.py`` (it is a
    # one-off utility script per PLAN.md §7), so these are not part of
    # ``ensure()``/``_STEP_OUTPUT_DIRS``; ``openmeq.nlloc_setup`` creates its
    # own directories as needed.

    @property
    def nlloc_dir(self) -> Path:
        """Top-level NLLoc setup output directory (``nlloc/``)."""
        return self.root / "nlloc"

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
        return self.root / "locate" / "passA"

    @property
    def passB_dir(self) -> Path:
        return self.root / "locate" / "passB"

    @property
    def catalog_auto_xml(self) -> Path:
        return self.root / "locate" / "catalog_auto.xml"

    @property
    def locate_summary_csv(self) -> Path:
        return self.root / "locate" / "summary.csv"

    @property
    def removal_log_csv(self) -> Path:
        return self.root / "locate" / "removal_log.csv"

    # -- review (Step 6 / M7) --------------------------------------------------------

    @property
    def review_state_json(self) -> Path:
        return self.root / "review" / "state.json"

    @property
    def review_events_dir(self) -> Path:
        return self.root / "review" / "events"

    @property
    def catalog_final_xml(self) -> Path:
        return self.root / "review" / "catalog_final.xml"

    # -- magnitude (Step 7 / M8) -------------------------------------------------------

    @property
    def catalog_mag_xml(self) -> Path:
        return self.root / "magnitude" / "catalog_mag.xml"

    @property
    def station_magnitudes_csv(self) -> Path:
        return self.root / "magnitude" / "station_magnitudes.csv"

    # -- relocate / hypoDD (Step 8 / M10) -----------------------------------------------

    @property
    def relocate_work_dir(self) -> Path:
        return self.root / "relocate" / "work"

    @property
    def relocate_cache_dir(self) -> Path:
        return self.root / "relocate" / "cache"

    @property
    def dt_ct(self) -> Path:
        return self.root / "relocate" / "dt.ct"

    @property
    def dt_cc(self) -> Path:
        return self.root / "relocate" / "dt.cc"

    @property
    def catalog_dd_xml(self) -> Path:
        return self.root / "relocate" / "catalog_dd.xml"

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
        """``<step>/diagnostics/`` -- every step's diagnostics folder.

        ``step`` is that step's own top-level output folder name (see the
        module docstring), e.g. ``"qc"``, ``"detections"``, ``"chunks"``,
        ``"chunks_pn"``, ``"pick"``, ``"locate"``, ``"review"``,
        ``"magnitude"`` or ``"relocate"``.
        """
        return self.root / step / "diagnostics"

    def manifest_json(self, step: str) -> Path:
        """``<step>/manifest.json`` -- every step's :class:`openmeq.manifest.StepManifest` file.

        ``step`` follows the same convention as :meth:`diagnostics_dir`.
        """
        return self.root / step / "manifest.json"

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
            self.root / "inventory",
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
        for step in _STEP_OUTPUT_DIRS:
            dirs.append(self.root / step)
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
