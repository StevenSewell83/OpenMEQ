#!/usr/bin/env python
"""MacBan ML station corrections from a run's own station magnitudes.

Solves ``station_ml = M_event + S_station`` by least squares over every
included measurement in ``RUN/8_relocation/magnitude/station_magnitudes.csv``,
with the station terms constrained to zero mean (so the network-average ML
scale -- Hutton & Boore -- is unchanged), and writes ``correction = -S`` as
the ``net,sta,correction`` CSV read by ``magnitude.ml.station_corrections``.

The corrections currently in the config are removed from ``station_ml``
first, so rerunning after the corrections are applied re-derives them rather
than compounding them. Stations with fewer than ``--min-count`` measurements
are left out (correction 0).

Usage::

    python scripts/macban_station_terms.py -c config/macban_full_.yaml OUT_CSV [--min-count 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr

from openmeq.config import load_config
from openmeq.magnitude.relations import load_station_corrections
from openmeq.paths import RunPaths


def station_terms(st: pd.DataFrame) -> pd.Series:
    """Zero-mean station terms ``S`` from ``station_ml = M_event + S_station``."""
    ev = st.event_id.astype("category").cat.codes.to_numpy()
    sa = st.key.astype("category")
    ne, ns, n = ev.max() + 1, len(sa.cat.categories), len(st)
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[ev, ne + sa.cat.codes.to_numpy()]
    a = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n, ne + ns))
    zero_mean = coo_matrix(
        (np.full(ns, 10.0), (np.zeros(ns), ne + np.arange(ns))), shape=(1, ne + ns)
    )
    b = np.r_[st.station_ml.to_numpy(), 0.0]
    x = lsqr(vstack([a, zero_mean]), b, atol=1e-10, btol=1e-10)[0]
    return pd.Series(x[ne:], index=sa.cat.categories)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("out", type=Path)
    ap.add_argument("--min-count", type=int, default=20)
    a = ap.parse_args()

    cfg = load_config(a.config)
    csv = RunPaths(cfg).step_dir("magnitude") / "station_magnitudes.csv"
    st = pd.read_csv(csv)
    st = st[st.included].copy()
    st["key"] = st.net + "." + st.sta
    st["station_ml"] -= st.key.map(load_station_corrections(cfg)).fillna(0.0)
    counts = st.key.value_counts()
    st = st[st.key.map(counts) >= a.min_count]

    s = station_terms(st)
    net, sta = zip(*(k.split(".") for k in s.index), strict=True)
    out = pd.DataFrame({"net": net, "sta": sta, "correction": (-s.to_numpy()).round(3)})
    out.to_csv(a.out, index=False)
    for (k, v), n in zip(s.sort_values().items(), counts[s.sort_values().index], strict=True):
        print(f"{k:10s} n={n:4d} term {v:+.3f} -> correction {-v:+.3f}")
    print(f"wrote {a.out} ({len(out)} stations, from {csv})")


if __name__ == "__main__":
    main()
