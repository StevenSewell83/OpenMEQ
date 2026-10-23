"""Raw-data conversion at the start of a run (``data.convert``, OpenMEQ-6zu).

:func:`convert_raw_data` is called first thing by the ``qc`` step
(``openmeq qc``, and so ``openmeq run``), before the qc fingerprint and the
waveform index are computed, so freshly converted day files are indexed in
the same run (``openmeq convert`` runs it on its own). ``data.convert.format``
selects the converter: ``"suds"`` (:mod:`openmeq.suds`) or ``"cd11"``
(:mod:`openmeq.cd11`).

Skip-unchanged: the conversion is recorded in the qc step's manifest under
``convert`` with a fingerprint of the ``data.convert`` settings and every
input file's (path, size, mtime). It reruns when any of those change, when
``force`` is set, or when its output (any day file, plus ``stations.xml`` if
enabled) has gone missing. A rerun never overwrites existing day files
unless ``data.convert.overwrite`` is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openmeq.log import get_logger
from openmeq.manifest import StepManifest, fingerprint

if TYPE_CHECKING:
    from openmeq.config import Config
    from openmeq.paths import RunPaths

logger = get_logger("ingest")

_MANIFEST_STEP = "qc"
_MANIFEST_KEY = "convert"


def _outputs_present(out_dir: Path, stationxml: bool) -> bool:
    if stationxml and not (out_dir / "stations.xml").is_file():
        return False
    return next(out_dir.rglob("*.mseed"), None) is not None


def convert_raw_data(cfg: Config, paths: RunPaths, *, force: bool = False) -> bool:
    """Run ``cfg.data.convert`` if configured and needed; returns whether it ran."""
    conv = cfg.data.convert
    if conv is None:
        return False
    if not conv.input_dir.is_dir():
        raise FileNotFoundError(f"data.convert.input_dir does not exist: {conv.input_dir}")

    inputs = sorted(conv.input_dir.rglob(conv.pattern))
    fp = fingerprint(conv, inputs)
    manifest = StepManifest(paths, _MANIFEST_STEP)
    if (
        not force
        and manifest.is_done(_MANIFEST_KEY, fp)
        and _outputs_present(conv.output_dir, conv.stationxml)
    ):
        logger.info("data.convert: %d input files unchanged, skipping", len(inputs))
        return False

    logger.info("data.convert: %s %s -> %s", conv.format, conv.input_dir, conv.output_dir)
    if conv.format == "suds":
        from openmeq.suds import Codes, convert

        convert(
            conv.input_dir,
            conv.output_dir,
            cfg.parallel,
            Codes(conv.network, conv.location, conv.band, conv.instrument),
            overwrite=conv.overwrite,
            stationxml=conv.stationxml,
            pattern=conv.pattern,
        )
    elif conv.format == "cd11":
        from openmeq import cd11
        from openmeq.suds import Codes

        cd11.convert(
            conv.input_dir,
            conv.output_dir,
            cfg.parallel,
            Codes(conv.network, conv.location, conv.band, conv.instrument),
            overwrite=conv.overwrite,
            pattern=conv.pattern,
        )
    manifest.mark_done(_MANIFEST_KEY, fp)
    return True
