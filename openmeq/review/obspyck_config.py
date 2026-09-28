"""Generate ObsPyck's ini config + pluginpath layout from OpenMEQ's own YAML
(M7.2 / OpenMEQ-c3l.2).

:func:`generate_obspyck_config` is this module's one public entry point:
it writes ``cfg.review.obspyck_config`` (an ObsPyck ``-c`` ini file) and
finishes populating ``cfg.nlloc.obspyck_pluginpath`` (ObsPyck's ``[base]
pluginpath``) so both are ready for M7.3's batch loop to launch
``obspyck -c <this file> -t ... -d ... --event ... <chunk>.mseed
stations.xml`` against, offline, with no server section and no hand-edited
ini.

Everything here is mechanical generation of what the OpenMEQ-c3l.1 spike
(``docs/obspyck_notes.md``) empirically confirmed working -- this module's
own job is *not* to re-derive any of those findings.

Division of labour with ``openmeq.nlloc_setup`` (M5.2)
--------------------------------------------------------
``openmeq.nlloc_setup.link_obspyck_model`` already creates, inside
``cfg.nlloc.obspyck_pluginpath``:

* ``nlloc/NLLoc__<system>__<architecture>`` (symlinked from
  ``external/bin/NLLoc``)
* ``nlloc/locate_BY.nlloc`` (the control file, referencing this run's own
  travel-time grids)
* ``VELOCITY_MODELS`` (symlinked to ``paths.nlloc_velocity_models_dir``)

:func:`generate_obspyck_config` calls it directly (reusing it, not
duplicating its logic) and adds exactly the two things
``link_obspyck_model`` explicitly documents as *not* its job
(``docs/obspyck_notes.md`` pitfall #4 / its own docstring):

* empty ``hyp_2000/`` and ``focmec/`` directories alongside ``nlloc/``
  (``ObsPyck.__init__`` calls ``shutil.copytree`` unconditionally for
  *every* entry in ``PROGRAMS``, with no try/except -- a missing
  ``hyp_2000``/``focmec`` directory raises ``FileNotFoundError`` before the
  GUI ever appears, even though only NLLoc is used)
* the ObsPyck ``.cfg`` ini itself

Config content
---------------
``[base]``, ``[seismic_phases]``, ``[seed_id_lookup]`` and ``[nonlinloc]``
are generated from ``cfg`` (see :func:`_build_config`). ``[keys]``,
``[misc]``, ``[gui_defaults]``, ``[matplotlibrc]`` and ``[ar_picker]`` are
spliced byte-identical from a packaged copy of ObsPyck's own shipped
``example.cfg`` (``openmeq/review/data/obspyck_example.cfg``) -- these are
static ObsPyck GUI/tooling boilerplate OpenMEQ's own YAML has no say over,
and inventing bindings risks a ``KeyError`` deep in a review session
(this task's own design note). The OpenMEQ-c3l.1 spike's own minimal
working config (``docs/obspyck_notes.md`` §3) used exactly this section
list; ``[misc]``/``[gui_defaults]``/``[matplotlibrc]``/``[ar_picker]`` are
included even though this task's original "work to do" text names only
``[base]``/``[seismic_phases]``/``[seed_id_lookup]``/``[keys]``/
``[nonlinloc]`` as load-bearing, because the spike's later, empirical
finding supersedes that earlier guess and because the source confirms it:
``ObsPyck.__init__`` reads ``config.getfloat("gui_defaults", ...)``
unconditionally (``obspyck.py``, right after ``drawAxes()``), so omitting
``[gui_defaults]`` crashes construction before the window ever appears,
exactly like the ``hyp_2000``/``focmec`` pitfall above. No ``server_*``
section and no ``[station_combinations]`` are ever emitted (confirmed
unnecessary for this project's local-file-positional-args batch design).

**Deviation from the issue text's literal ``merge = true``**: ObsPyck's
``merge`` key is a three-way *string* enum, not a boolean --
``merge_check_and_cleanup_streams`` (``obspyck/util.py``) accepts only
``""`` (no merging), ``"safe"`` or ``"overwrite"``, and raises
``Exception("Unrecognized config value for merging traces...")`` for any
other value, including the literal string ``"true"``. This module emits
``merge = safe`` instead -- the closest actual enum value to "merging
enabled" without silently overwriting overlapping data
(``"overwrite"``'s documented behaviour) -- so the generated config is
actually loadable by ObsPyck rather than crashing on first use.

Regenerate-on-config-change
-----------------------------
Gated by ``openmeq.manifest.StepManifest`` (the ``"review"`` step's
manifest, key ``"obspyck_config"``) fingerprinted on ``cfg.review``,
``cfg.nlloc`` and the merged station table's contents (not the raw
``cfg.stations`` section alone, so a StationXML/CSV file edited in place
without changing ``cfg.stations`` itself is still detected -- the same
reasoning ``openmeq.nlloc_setup.setup_nlloc``'s own fingerprint uses).
ObsPyck itself does not care about staleness, but M7.3 calls this once per
reviewed event, and redoing ``link_obspyck_model`` (which touches the
filesystem: relinking the NLLoc binary, rewriting the control file,
re-symlinking ``VELOCITY_MODELS``) plus rewriting the ini on every single
event would be wasteful over a run of hundreds of events.
"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING

from openmeq.manifest import StepManifest, fingerprint
from openmeq.nlloc_setup import link_obspyck_model
from openmeq.stations import station_table as load_station_table

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

#: ``StepManifest`` step name (shares ``review/manifest.json`` with M7.3/M7.4's
#: own per-event manifest entries -- ``StepManifest`` is a generic
#: ``{key: fingerprint}`` mapping, so a distinct key here does not collide
#: with theirs, and this stays inside ARCHITECTURE.md §3's existing
#: ``<step>/manifest.json`` layout instead of inventing a new top-level dir).
STEP_NAME = "review"
MANIFEST_KEY = "obspyck_config"

#: Packaged, byte-identical copy of ObsPyck's own shipped ``example.cfg``
#: (see the module docstring's "Config content" section).
_EXAMPLE_CFG_PATH = Path(__file__).parent / "data" / "obspyck_example.cfg"

#: Sections spliced verbatim from :data:`_EXAMPLE_CFG_PATH` rather than
#: generated from ``cfg`` (static ObsPyck GUI/tooling boilerplate).
SPLICED_SECTIONS = ("keys", "misc", "gui_defaults", "matplotlibrc", "ar_picker")

#: ``[base]`` keys this task's issue and the OpenMEQ-c3l.1 spike confirm are
#: read unconditionally, in the order they are documented (used by tests to
#: check every mandatory key is present).
MANDATORY_BASE_KEYS = (
    "no_metadata",
    "starttime_offset",
    "duration",
    "ignore_gse2_chksum_error",
    "merge",
    "zero_mean",
    "pluginpath",
    "verbosity",
    "event_server",
    "magnitude_pick_color",
    # Added by OpenMEQ-c3l.3 (M7.3): `ObsPyck.drawAxes()` (called
    # unconditionally from `__init__`) does `self.config.getboolean("base",
    # "normalization")` with no fallback -- missing it crashes construction
    # before the GUI ever appears, exactly like the `[gui_defaults]`/
    # `hyp_2000`/`focmec` pitfalls this module's own docstring already
    # documents (confirmed empirically by M7.3's requires_gui end-to-end
    # test: a real launch against this module's own generated config
    # crashed with `configparser.NoOptionError: No option 'normalization'`
    # until this was added). `magnitude_picker_width` is similarly read with
    # `self.config.getint(...)` and no fallback, but only from the
    # `setMagMin`/`setMagMax` key handler (a real reviewer's normal
    # workflow, not construction) -- added for the same reason before it
    # bites a real review session instead of a test.
    "normalization",
    "magnitude_picker_width",
)


def _new_parser() -> ConfigParser:
    parser = ConfigParser(allow_no_value=True)
    parser.optionxform = str  # preserve case (ObsPyck's own keys are mixed-case)
    return parser


def _load_example_cfg() -> ConfigParser:
    parser = _new_parser()
    read = parser.read(_EXAMPLE_CFG_PATH)
    if not read:
        raise FileNotFoundError(f"packaged ObsPyck example.cfg not found: {_EXAMPLE_CFG_PATH}")
    return parser


def _build_config(cfg: Config, pluginpath: Path) -> ConfigParser:
    """Build the full generated ini, as a :class:`configparser.ConfigParser`.

    Parameters
    ----------
    cfg : openmeq.config.Config
    pluginpath : pathlib.Path
        ``cfg.nlloc.obspyck_pluginpath``, resolved.

    Returns
    -------
    configparser.ConfigParser
    """
    duration_s = cfg.chunks.pre_s + cfg.chunks.post_s

    parser = _new_parser()
    parser["base"] = {
        "no_metadata": "false",
        "starttime_offset": "0",
        # At least the chunk's own length, so ObsPyck's local-file ingestion
        # (``fetch_waveforms_with_metadata``'s ``obspy.read(..., starttime=t1,
        # endtime=t2)``) never truncates it -- this is only a fallback value
        # (M7.3's launch command always passes ``-d`` explicitly, which wins,
        # ``obspyck/util.py``: ``if options.duration is None: ... else:
        # t2 = t1 + options.duration``), but must still be present and sane.
        "duration": str(duration_s),
        "ignore_gse2_chksum_error": "false",
        # See the module docstring's "Deviation" note: ObsPyck's own `merge`
        # enum has no boolean form; "safe" is the closest match to "merging
        # enabled" that doesn't silently overwrite overlapping data.
        "merge": "safe",
        "zero_mean": "false",
        "pluginpath": str(pluginpath),
        # Required so the "writing xml as %s" stdout line M7.3's file-polling
        # fallback greps for is actually emitted (obspyck's own verbosity
        # gate, util.py, SplitWriter; not `quiet`).
        "verbosity": "normal",
        # Blank: Jane is never contacted (obspyck.py: `if event_server_name:`),
        # matching the local-file-only batch design.
        "event_server": "",
        "magnitude_pick_color": "green",
        # See MANDATORY_BASE_KEYS' own comment: read unconditionally by
        # `drawAxes()`/the magnitude-pick key handler, no fallback default.
        # `normalization=false` matches this module's own `zero_mean=false`/
        # `no_metadata=false` choice (no implicit reshaping of the raw
        # counts ObsPyck displays); `magnitude_picker_width` is ObsPyck's
        # own shipped example.cfg value (a plausible pixel width, with no
        # OpenMEQ-specific meaning to derive it from).
        "normalization": "false",
        "magnitude_picker_width": "10",
    }
    parser["seismic_phases"] = {"P": "red", "S": "blue"}
    parser["seed_id_lookup"] = {}

    example = _load_example_cfg()
    for section in SPLICED_SECTIONS:
        parser[section] = dict(example.items(section))

    parser["nonlinloc"] = {"default_pick_uncertainty": "0.1"}
    return parser


def _fingerprint_inputs(cfg: Config) -> str:
    table = load_station_table(cfg.stations)
    return fingerprint(cfg.review, cfg.nlloc, table.to_dict("records"))


def generate_obspyck_config(cfg: Config, paths: RunPaths, *, force: bool = False) -> Path:
    """Write ``cfg.review.obspyck_config`` and finish populating
    ``cfg.nlloc.obspyck_pluginpath``, skipping the work if nothing relevant
    changed since the last call (see the module docstring's
    "Regenerate-on-config-change" section).

    Parameters
    ----------
    cfg : openmeq.config.Config
    paths : openmeq.paths.RunPaths
    force : bool, keyword-only
        Regenerate even if the manifest says nothing changed.

    Returns
    -------
    pathlib.Path
        ``cfg.review.obspyck_config`` (the generated/existing ini file).
    """
    pluginpath = Path(cfg.nlloc.obspyck_pluginpath)
    target = Path(cfg.review.obspyck_config)

    fp = _fingerprint_inputs(cfg)
    manifest = StepManifest(paths, STEP_NAME)

    already_laid_out = (
        target.exists()
        and (pluginpath / "hyp_2000").is_dir()
        and (pluginpath / "focmec").is_dir()
        and (pluginpath / "nlloc").is_dir()
    )
    if not force and already_laid_out and manifest.is_done(MANIFEST_KEY, fp):
        return target

    table = load_station_table(cfg.stations)
    link_obspyck_model(cfg, paths, table)

    (pluginpath / "hyp_2000").mkdir(parents=True, exist_ok=True)
    (pluginpath / "focmec").mkdir(parents=True, exist_ok=True)

    parser = _build_config(cfg, pluginpath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        parser.write(f)

    manifest.mark_done(MANIFEST_KEY, fp)
    return target
