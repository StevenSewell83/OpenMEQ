#!/usr/bin/env python
"""MacBan tomography catalogue: pick weights, relocation inputs, station delays, VELEST export.

Follows ``scripts/macban_tomo_qc.py`` (OpenMEQ-qgn.3/.4) for OpenMEQ-qgn.5/.6.

``weights QC_DIR OUT_XML`` (qgn.5)
    Weight class 0 (best) .. 3 for every accepted pick in
    ``QC_DIR/picks_qc.csv``; a pick at a ``--noisy`` station (default MB9A)
    is moved down one class (max 3)::

        class  P                                    S                                  unc. (s)
        0      AIC-confirmed, prob >= 0.9, SNR >= 10  Wadati |dev| <= 1 sigma, prob >= 0.8   0.02
        1      AIC-confirmed, prob >= 0.7, SNR >= 5   Wadati |dev| <= 2 sigma, prob >= 0.6   0.05
        2      other AIC-confirmed                    other Wadati-checked                   0.10
        3      unconfirmed (emergent onset)           distance-checked (no P at station)     0.20

    The uncertainties are the upper limits of openSeisVelocity's default
    ``pick_uncertainty_to_weight`` ({0.02: 0, 0.05: 1, 0.10: 2, 0.20: 3}),
    so NLLoc (which uses the uncertainty) and VELEST (which uses the class)
    weight a pick alike. sigma is the robust spread of the accepted Wadati
    deviations. Writes ``OUT_XML``: one Event per QC event with at least
    ``--min-p`` accepted P (``smi:openmeq/event/<id>``, picks only, as
    ``openmeq pick`` writes ``catalog_picks.xml``), pick time = ``t_final``
    (P: PhaseNet minus its station lag; S: PhaseNet), extras
    ``weight_class``, ``probability``, ``qc_check``, ``confirmed``,
    ``manual_confirmed_event``; plus ``picks_weighted.csv`` next to it.

``delays RUN [--previous CSV] OUT_CSV``
    Station delays for ``nlloc.station_delays_csv`` from ``RUN/7_location``:
    per station and phase, the median residual of class 0-2 arrivals of
    accepted events (>= ``--min-n``, else 0), added to ``--previous`` (the
    delays that run used) so repeated rounds converge. ``--seed-qc QC_DIR``
    (first round): S delays from the QC Wadati station terms instead
    (P delay + S-P term - median term), because the residual filter drops
    most S picks of a station with a large S delay (MB07) and the survivors
    would miss it.

``export RUN OUT_XML``
    The accepted events of ``RUN/7_location/catalog_auto.xml`` for
    openSeisVelocity (``inputs.quakeml``, ``use_arrivals: true``): each
    keeps its preferred origin, the picks it used (raw times -- VELEST
    estimates its own station delays) and their arrivals.

Usage (MacBan 2018-2019)::

    R=/media/steve/fast9100/runs/macban_2018-01-01_2019-12-31
    T=/media/steve/fast9100/runs/macban_tomo_2018-2019
    python scripts/macban_tomo_catalog.py weights $R/tomo/qc $T/5_ml_picks/catalog_picks.xml
    openmeq locate -c config/macban_tomo.yaml
    python scripts/macban_tomo_catalog.py delays $T $T/tomo/station_delays.csv
    # set nlloc.station_delays_csv, relocate (--force), repeat delays with --previous
    python scripts/macban_tomo_catalog.py export $T $T/tomo/catalog_velest.xml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Catalog, UTCDateTime
from obspy.core.event import Pick, ResourceIdentifier, WaveformStreamID

from openmeq.catalog import get_extra, make_event, read_catalog, set_extra, write_catalog

UNCERTAINTY_S = {0: 0.02, 1: 0.05, 2: 0.10, 3: 0.20}
CHANNEL = {"P": "EHZ", "S": "EHN"}


# --------------------------------------------------------------------------- #
# weights (qgn.5)
# --------------------------------------------------------------------------- #


def weight_classes(q: pd.DataFrame, noisy: set[str]) -> tuple[pd.Series, float]:
    """Class 0-3 for accepted picks (module docstring) and the S Wadati sigma."""
    sig = 1.4826 * float(
        (q.dev[(q.phase == "S") & q.accept & (q.check == "wadati")]).abs().median()
    )
    is_p = q.phase == "P"
    conf = q.confirmed.astype(str).eq("True")
    p_cls = np.select(
        [conf & (q.prob >= 0.9) & (q.snr >= 10), conf & (q.prob >= 0.7) & (q.snr >= 5), conf],
        [0, 1, 2],
        3,
    )
    wad = q.check == "wadati"
    adev = q.dev.abs()
    s_cls = np.select(
        [wad & (adev <= sig) & (q.prob >= 0.8), wad & (adev <= 2 * sig) & (q.prob >= 0.6), wad],
        [0, 1, 2],
        3,
    )
    cls = pd.Series(np.where(is_p, p_cls, s_cls), index=q.index)
    cls = (cls + q.station.isin(noisy).astype(int)).clip(upper=3)
    return cls.where(q.accept), sig


def cmd_weights(args) -> int:
    q = pd.read_csv(args.qc_dir / "picks_qc.csv", low_memory=False)
    q["accept"] = q.accept.astype(bool)
    q["weight_class"], sig = weight_classes(q, set(args.noisy.split(",")))
    q = q[q.accept].copy()
    q["weight_class"] = q.weight_class.astype(int)
    q["uncertainty_s"] = q.weight_class.map(UNCERTAINTY_S)
    n_p = q[q.phase == "P"].groupby("event_id").station.nunique()
    keep = n_p.index[n_p >= args.min_p]
    q = q[q.event_id.isin(keep)]

    cat = Catalog()
    for eid, g in q.groupby("event_id"):
        ev = make_event(eid)
        for r in g.itertuples():
            pick = Pick(
                resource_id=ResourceIdentifier(f"smi:openmeq/pick/{eid}/{r.station}.{r.phase}/qc"),
                time=UTCDateTime(r.t_final),
                waveform_id=WaveformStreamID(
                    network_code="MB",
                    station_code=r.station,
                    location_code="",
                    channel_code=CHANNEL[r.phase],
                ),
                phase_hint=r.phase,
                evaluation_mode="automatic",
                method_id=ResourceIdentifier("smi:openmeq/method/phasenet_qc"),
            )
            pick.time_errors.uncertainty = r.uncertainty_s
            set_extra(pick, "weight_class", int(r.weight_class))
            set_extra(pick, "probability", float(r.prob))
            set_extra(pick, "qc_check", "aic" if r.phase == "P" else r.check)
            set_extra(pick, "confirmed", bool(r.confirmed) if r.phase == "P" else True)
            set_extra(pick, "manual_confirmed_event", bool(r.manual_confirmed_event))
            ev.picks.append(pick)
        cat.events.append(ev)
    args.out_xml.parent.mkdir(parents=True, exist_ok=True)
    write_catalog(cat, args.out_xml)
    q.to_csv(args.out_xml.parent / "picks_weighted.csv", index=False)
    tab = q.groupby(["phase", "weight_class"]).size().unstack(fill_value=0)
    print(f"S Wadati sigma {sig:.3f} s; {len(cat)} events (>= {args.min_p} accepted P)")
    print(f"{len(q)} picks")
    print(tab.to_string())
    return 0


# --------------------------------------------------------------------------- #
# delays
# --------------------------------------------------------------------------- #


def arrivals_table(run: Path, accepted_only: bool = True) -> pd.DataFrame:
    rows = []
    for ev in read_catalog(run / "7_location/catalog_auto.xml"):
        o = ev.preferred_origin()
        if o is None or (accepted_only and o.evaluation_status == "rejected"):
            continue
        pk = {str(p.resource_id): p for p in ev.picks}
        for a in o.arrivals:
            p = pk[str(a.pick_id)]
            if any(c.text.startswith("rejected:") for c in p.comments):
                continue
            rows.append(
                dict(
                    event_id=str(ev.resource_id).split("/")[-1],
                    station=p.waveform_id.station_code,
                    phase=a.phase,
                    residual=a.time_residual,
                    weight_class=get_extra(p, "weight_class"),
                )
            )
    return pd.DataFrame(rows)


def cmd_delays(args) -> int:
    a = arrivals_table(args.run)
    a = a[a.weight_class <= 2]
    g = a.groupby(["station", "phase"]).residual
    d = pd.DataFrame({"median": g.median(), "n": g.size(), "std_s": g.std()}).reset_index()
    d["step_s"] = np.where(d.n >= args.min_n, d["median"], 0.0)
    if args.seed_qc:
        # S: the locate residual filter drops most picks of a station with a
        # large S delay (MB07), so the surviving residuals miss it. Seed from
        # the QC Wadati terms instead: S delay = P delay + (S-P term - median).
        t = pd.read_csv(args.seed_qc / "station_terms.csv").set_index("station")
        t = t[t.index != "NETWORK"]
        c = t.s_minus_p_term_s[t.n_s_accepted >= args.min_n]
        p_step = d[d.phase == "P"].set_index("station").step_s
        seed = (p_step.reindex(c.index).fillna(0.0) + c - c.median()).rename("seed")
        is_s = (d.phase == "S") & d.station.isin(seed.index)
        d.loc[is_s, "step_s"] = d.loc[is_s, "station"].map(seed)
    prev = pd.read_csv(args.previous) if args.previous else None
    if prev is not None:
        d = d.merge(
            prev[["station", "phase", "delay_s"]].rename(columns={"delay_s": "prev_s"}), how="left"
        )
    else:
        d["prev_s"] = 0.0
    d["prev_s"] = d.prev_s.fillna(0.0)
    d["delay_s"] = (d.prev_s + d.step_s).round(4)
    d = d.sort_values(["phase", "station"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    d[["station", "phase", "delay_s", "n", "std_s", "prev_s", "step_s"]].to_csv(
        args.out_csv, index=False
    )
    print(
        d[["station", "phase", "n", "prev_s", "step_s", "delay_s"]].round(3).to_string(index=False)
    )
    print(f"max |step| {d.step_s.abs().max():.3f} s")
    return 0


# --------------------------------------------------------------------------- #
# export for openSeisVelocity
# --------------------------------------------------------------------------- #


def cmd_export(args) -> int:
    out = Catalog()
    for ev in read_catalog(args.run / "7_location/catalog_auto.xml"):
        o = ev.preferred_origin()
        if o is None or o.evaluation_status == "rejected":
            continue
        used = {
            str(a.pick_id)
            for a in o.arrivals
            if not any(
                c.text.startswith("rejected:")
                for p in ev.picks
                if str(p.resource_id) == str(a.pick_id)
                for c in p.comments
            )
        }
        ev.picks = [p for p in ev.picks if str(p.resource_id) in used]
        o.arrivals = [a for a in o.arrivals if str(a.pick_id) in used]
        ev.origins = [o]
        out.events.append(ev)
    args.out_xml.parent.mkdir(parents=True, exist_ok=True)
    write_catalog(out, args.out_xml)
    n = sum(len(e.picks) for e in out)
    print(f"{len(out)} accepted events, {n} picks -> {args.out_xml}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights")
    w.add_argument("qc_dir", type=Path)
    w.add_argument("out_xml", type=Path)
    w.add_argument("--noisy", default="MB9A")
    w.add_argument("--min-p", type=int, default=4, help="min accepted P stations per event")
    d = sub.add_parser("delays")
    d.add_argument("run", type=Path)
    d.add_argument("out_csv", type=Path)
    d.add_argument("--previous", type=Path, help="delays CSV the run was located with")
    d.add_argument("--min-n", type=int, default=10)
    d.add_argument("--seed-qc", type=Path, help="QC dir: seed S delays from its Wadati terms")
    e = sub.add_parser("export")
    e.add_argument("run", type=Path)
    e.add_argument("out_xml", type=Path)
    args = ap.parse_args(argv)
    return {"weights": cmd_weights, "delays": cmd_delays, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
