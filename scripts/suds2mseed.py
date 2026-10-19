#!/usr/bin/env python
"""CLI wrapper around ``openmeq.suds``: PC-SUDS event files -> miniSEED day files.

Writes ``OUT_DIR/YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed`` plus
``OUT_DIR/stations.xml``; see ``openmeq/suds.py`` for the format details.
The same conversion runs automatically at the start of the ``qc`` step (and
so of ``openmeq run``) when ``data.convert`` is set in the config -- this
script is for converting outside a pipeline run.

Usage
-----
::

    python scripts/suds2mseed.py IN_DIR OUT_DIR [--network MB] [--band E]
        [--inst H] [--location ""] [--n-workers 8] [--overwrite] [--dry-run]
        [--no-stationxml]

Example (MacBan 2018-2019)::

    SRC="/media/steve/fast9100/datasets/MacBan/MEQ Exercise MB 2018-2019(1)"
    python scripts/suds2mseed.py "$SRC" /media/steve/fast9100/datasets/MacBan/mseed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from openmeq.config import ParallelConfig
from openmeq.suds import Codes, convert


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert PC-SUDS event files to a miniSEED day-file archive "
        "(YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed) + stations.xml.",
    )
    ap.add_argument("in_dir", type=Path, help="directory searched recursively for SUDS files")
    ap.add_argument("out_dir", type=Path, help="archive root to write")
    ap.add_argument("--network", default="MB", help="SEED network code (default: %(default)s)")
    ap.add_argument("--location", default="", help="SEED location code (default: empty)")
    ap.add_argument(
        "--band", default="E", help="band code; E = short period >=80 Hz (default: %(default)s)"
    )
    ap.add_argument(
        "--inst", default="H", help="SEED instrument code; H = seismometer (default: %(default)s)"
    )
    ap.add_argument("--pattern", default="*.suds", help="input glob (default: %(default)s)")
    ap.add_argument("--n-workers", type=int, default=8, help="processes (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing day files")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
    ap.add_argument("--no-stationxml", action="store_true", help="skip writing stations.xml")
    args = ap.parse_args(argv)
    if not args.in_dir.is_dir():
        ap.error(f"{args.in_dir} is not a directory")
    for name, val in (("network", args.network), ("location", args.location)):
        if len(val) > 2 or not re.fullmatch(r"[A-Za-z0-9]*", val):
            ap.error(f"--{name} must be up to 2 alphanumeric characters")
    convert(
        args.in_dir,
        args.out_dir,
        ParallelConfig(n_workers=args.n_workers, threads_per_worker=1),
        Codes(args.network, args.location, args.band, args.inst),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        stationxml=not args.no_stationxml,
        pattern=args.pattern,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
