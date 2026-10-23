#!/usr/bin/env python
"""CLI wrapper around ``openmeq.cd11``: CD-1.1 recorder files -> miniSEED day files.

Writes ``OUT_DIR/YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed``; see
``openmeq/cd11.py`` for the format details. With ``data.convert.format:
cd11`` in a config, ``openmeq convert -c config.yaml`` (or the ``qc`` step)
does the same conversion -- this script is for converting without a config.

Usage
-----
::

    python scripts/cd11tomseed.py IN_DIR OUT_DIR [--network MB] [--band E]
        [--inst H] [--location ""] [--pattern "*.cd11"] [--n-workers 8]
        [--overwrite] [--dry-run]

Example (MacBan Feb 2021; ``--pattern "*_C_*.cd11"`` skips the event files)::

    SRC="/media/steve/fast9100/datasets/MB MEQ Feb 10, 2021 Cont Raw Data"
    python scripts/cd11tomseed.py "$SRC" /media/steve/fast9100/datasets/MacBan/mseed_2021
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from openmeq.cd11 import convert
from openmeq.config import ParallelConfig
from openmeq.suds import Codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert CD-1.1 (.cd11) files to a miniSEED day-file archive "
        "(YEAR/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY.mseed).",
    )
    ap.add_argument("in_dir", type=Path, help="directory searched recursively for .cd11 files")
    ap.add_argument("out_dir", type=Path, help="archive root to write")
    ap.add_argument("--network", default="MB", help="SEED network code (default: %(default)s)")
    ap.add_argument("--location", default="", help="SEED location code (default: empty)")
    ap.add_argument(
        "--band", default="E", help="band code; E = short period >=80 Hz (default: %(default)s)"
    )
    ap.add_argument(
        "--inst", default="H", help="SEED instrument code; H = seismometer (default: %(default)s)"
    )
    ap.add_argument("--pattern", default="*.cd11", help="input glob (default: %(default)s)")
    ap.add_argument("--n-workers", type=int, default=8, help="processes (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing day files")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
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
        pattern=args.pattern,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
