#!/usr/bin/env python
"""Redraw a finished run's map + depth-section figures from its on-disk
outputs, e.g. to add the ``diagnostics.wells`` overlay (OpenMEQ-7zt) without
re-running ``openmeq locate``/``relocate``.

Redraws, in place (or under ``--out-dir``), with the same titles/colouring
as the pipeline:

- ``locate/diagnostics/pass{A,B}_map_sections.png`` -- each pass's final
  location per event, rebuilt from the last ``iter_NN`` with a per-event
  NLLoc ``.hyp`` under ``locate/pass{A,B}/<event_id>/`` (the same "last
  successful iteration" origin ``openmeq.locate.pass_a`` keeps);
- ``relocate/diagnostics/locations_{before,after}.png`` -- from
  ``relocate/diagnostics/summary.csv``;
- ``review/diagnostics/location_shift_map.png`` -- from
  ``review/diagnostics/summary.csv``, if the run has one;
- page 1 (the map page) of ``locate/diagnostics/pass_comparison.pdf``,
  spliced in front of the existing PDF's other pages (needs ``pypdf`` or
  ``PyPDF2``; skipped with a warning otherwise). "Accepted" = pass B rows
  with ``status == accepted`` in ``locate/summary.csv``.

Not redrawn: the extended ``pass{A,B}_station_res_map.png`` (it needs
in-memory per-arrival data).

Usage
-----
    python scripts/regen_map_figures.py CONFIG.yaml [--wells-dir DIR] [--out-dir DIR]

``--wells-dir`` enables the well overlay from that directory, overriding
``diagnostics.wells`` in the config.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from openmeq.config import WellsConfig, load_config
from openmeq.locate.diagnostics import _plot_map
from openmeq.locate.diagnostics_compare import PASS_COMPARISON_PDF, _page_map
from openmeq.log import get_logger
from openmeq.paths import RunPaths
from openmeq.plotting import plot_map_sections
from openmeq.wells import load_wells

logger = get_logger("regen_map_figures")

_GEO_RE = re.compile(r"Lat\s+(\S+)\s+Long\s+(\S+)\s+Depth\s+(\S+)")
_RMS_RE = re.compile(r"\bRMS\s+(\S+)")


def _read_hyp(path: Path) -> dict | None:
    """``lat``/``lon``/``depth_km``/``rms`` from an NLLoc ``.hyp`` file (any
    NLLOC status, as obspy's reader -- and so the pipeline -- accepts it)."""
    out: dict = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("GEOGRAPHIC"):
            m = _GEO_RE.search(line)
            if m:
                out.update(lat=float(m[1]), lon=float(m[2]), depth_km=float(m[3]))
        elif line.startswith("QUALITY"):
            m = _RMS_RE.search(line)
            if m:
                out["rms"] = float(m[1])
    return out if "lat" in out else None


def pass_locations(pass_dir: Path) -> pd.DataFrame:
    """One row per event: its last successful iteration's location."""
    rows = []
    for event_dir in sorted(p for p in pass_dir.iterdir() if p.is_dir()):
        for iter_dir in sorted(event_dir.glob("iter_*"), reverse=True):
            hyps = sorted(iter_dir.glob("out.*.*.grid0.loc.hyp"))
            loc = _read_hyp(hyps[0]) if len(hyps) == 1 else None
            if loc is not None:
                rows.append({"event_id": event_dir.name, **loc})
                break
    return pd.DataFrame(rows, columns=["event_id", "lat", "lon", "depth_km", "rms"])


def replace_map_page(
    pdf_in: Path,
    pdf_out: Path,
    frames: dict[str, pd.DataFrame],
    stations: pd.DataFrame | None,
    wells,
) -> bool:
    """Write ``pdf_out`` = a freshly drawn map page + ``pdf_in``'s pages 2..N."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            logger.warning("pypdf/PyPDF2 not installed; %s not updated", pdf_in)
            return False
    import io

    import matplotlib.pyplot as plt  # noqa: F401  (Agg already selected by openmeq.plotting)
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _page_map(pdf, frames, stations, wells)
    old = PdfReader(str(pdf_in))
    writer = PdfWriter()
    writer.add_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
    for page in old.pages[1:]:
        writer.add_page(page)
    data = io.BytesIO()
    writer.write(data)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.write_bytes(data.getvalue())  # after reading pdf_in, so in place is safe
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--wells-dir", type=Path, help="enable the well overlay from DIR")
    parser.add_argument("--out-dir", type=Path, help="write here instead of over the run's PNGs")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = RunPaths(cfg)
    wells_cfg = (
        WellsConfig(
            **{**cfg.diagnostics.wells.model_dump(), "enabled": True, "dir": args.wells_dir}
        )
        if args.wells_dir
        else cfg.diagnostics.wells
    )
    wells = load_wells(wells_cfg)
    if wells is None:
        logger.warning("no well overlay (diagnostics.wells disabled or unreadable)")

    stations = pd.read_csv(paths.stations_csv) if paths.stations_csv.exists() else None

    def out(step: str, name: str) -> Path:
        return (args.out_dir / step / name) if args.out_dir else paths.diagnostics_dir(step) / name

    written: list[Path] = []
    pass_frames: dict[str, pd.DataFrame] = {}
    for label, pass_dir in (("A", paths.passA_dir), ("B", paths.passB_dir)):
        if not pass_dir.is_dir():
            logger.warning("%s missing; skipping pass %s map", pass_dir, label)
            continue
        df = pass_locations(pass_dir)
        pass_frames[label] = df
        logger.info("pass %s: %d located events", label, len(df))
        path = out("locate", f"pass{label}_map_sections.png")
        _plot_map(df, stations, label, path, wells)
        written.append(path)

    pdf_in = paths.diagnostics_dir("locate") / PASS_COMPARISON_PDF
    if len(pass_frames) == 2 and pdf_in.exists() and paths.locate_summary_csv.exists():
        summary = pd.read_csv(paths.locate_summary_csv)
        accepted = set(summary.loc[summary["status"] == "accepted", "event_id"])
        frames = {
            "Pass A": pass_frames["A"],
            "Pass B": pass_frames["B"],
            "Accepted": pass_frames["B"][pass_frames["B"]["event_id"].isin(accepted)],
        }
        path = out("locate", PASS_COMPARISON_PDF)
        if replace_map_page(pdf_in, path, frames, stations, wells):
            written.append(path)

    reloc_csv = paths.diagnostics_dir("relocate") / "summary.csv"
    if reloc_csv.exists():
        s = pd.read_csv(reloc_csv)
        moved = s[s["relative_shift_km"].notna()]
        before = moved.rename(
            columns={"lat_before": "lat", "lon_before": "lon", "depth_km_before": "depth_km"}
        )[["event_id", "lat", "lon", "depth_km"]]
        after = moved.rename(
            columns={"lat_after": "lat", "lon_after": "lon", "depth_km_after": "depth_km"}
        )[["event_id", "lat", "lon", "depth_km", "relative_shift_km"]]
        path = out("relocate", "locations_before.png")
        plot_map_sections(before, stations, path, title="Pre-hypoDD event locations", wells=wells)
        written.append(path)
        path = out("relocate", "locations_after.png")
        plot_map_sections(
            after,
            stations,
            path,
            color_by="relative_shift_km",
            cbar_label="Relative shift (km)",
            title="Post-hypoDD (relocated) event locations (colour: relative shift)",
            wells=wells,
        )
        written.append(path)

    review_csv = paths.diagnostics_dir("review") / "summary.csv"
    if review_csv.exists():
        path = out("review", "location_shift_map.png")
        plot_map_sections(
            pd.read_csv(review_csv),
            stations,
            path,
            color_by="horizontal_shift_km",
            cbar_label="Horizontal shift (km)",
            title="Reviewed event locations (colour: horizontal shift, auto -> manual)",
            wells=wells,
        )
        written.append(path)

    for path in written:
        print(path)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
