#!/usr/bin/env python
"""MacBan tomography catalogue: manual/automatic association and pick QC (OpenMEQ-qgn.3/.4).

The MacBan manual picks were placed to reduce the locator RMS (confirmed
2026-10-21: their residuals track independent onsets with slope 0.34 for P
and 0.06 for S, and station terms are erased), so they are never used as a
timing reference or to reject an automatic pick. Here they only confirm
events and stations and flag gross differences for review.

Inputs
------
* ``RUN/7_location/catalog_auto.xml`` + ``summary.csv`` -- every event with
  PhaseNet picks (picks removed by the locate residual filter are kept, so
  they can be reconsidered here) and its NonLinLoc origin.
* ``RUN/2_detection/detections.csv`` and ``RUN/4_data_chunks/chunks/`` (raw
  200 Hz waveforms).
* ``MANUAL/events.csv``, ``MANUAL/picks.csv`` from
  ``scripts/macban_man_picks.py``.

Association (qgn.3)
-------------------
Manual duplicates (two manual events < 1 s and < 1 km apart: the same
earthquake located from two SUDS files) are flagged ``manual_duplicate``;
the one with more stations is kept. The rest are matched one-to-one,
closest first: to a located automatic event by origin time (within
``--match-s``), else to an unlocated one by detection time (``origin - 5 s``
.. ``origin + 15 s``). Status: ``accepted`` / ``located_rejected`` /
``not_located`` / ``detected_unmatched`` (a detection but no match) /
``undetected``. For matched events each station is compared per phase
(manual S on N and E is one pick): manual only, automatic only, or both
with ``dt = auto - manual``; ``|dt| > 0.3 s`` is a gross difference listed
for review.

Pick QC (qgn.4), events with a NonLinLoc origin only
----------------------------------------------------
Duplicates: one pick per event/station/phase (the one locate used, else the
most probable).

P: an AIC onset on the raw vertical (2-30 Hz zero-phase) in
``[t_pn - 0.35, t_pn + 0.15] s``, and an SNR (RMS 0.3 s after the onset /
RMS 1 s before it). PhaseNet's lag behind the onset is estimated per station
(median ``t_pn - t_aic`` of picks with probability >= 0.9, SNR >= 5 and
``|t_pn - t_aic| < 0.1 s``; the network median where a station has < 20 such
picks) and removed: ``t_final = t_pn - lag``. Accepted when the corrected
pick agrees with the AIC onset within ``--p-agree-s`` and probability >=
0.5 (``confirmed``), or -- unconfirmed, for emergent onsets where AIC finds
none or disagrees -- probability >= 0.9 with SNR >= 3 or probability >= 0.7
with SNR >= 5 (SNR at the PhaseNet time when there is no AIC onset).

S: the slope is fixed first, then each station's term is the median offset
of all its points (fitting both together trades slope against terms, as
each station sees a narrow distance range), so a consistent site delay
(MB07, site 9) goes into the term instead of being rejected. Vp/Vs is the
median of per-event Wadati slopes (S-P vs absolute P time across >= 4
stations, independent of the origin time); the apparent S slowness for the
distance check is the median of ``(tS - t0) / r`` for r > 2 km.
With an accepted P at the station: Wadati check, ``y = tS - tP`` against
``x = tP - t0``, ``y = k x + c_station``; reject ``|dev| > max(--s-dev-min-s,
3 sigma)``. Without: S travel time ``tS - t0`` against hypocentral distance,
``tS - t0 = r / Vs + d_station`` (terms from every S), threshold
``max(1.5 --s-dev-min-s, 3 sigma)``. S times are PhaseNet's (no onset
correction).

Noisy stations (``--noisy``, default MB9A): P must agree with the AIC onset
and have probability >= 0.7; S needs probability >= 0.8.

Outputs (``RUN/tomo/qc``)
-------------------------
``association.csv``, ``station_compare.csv``, ``review_candidates.csv``,
``picks_qc.csv``, ``events_qc.csv``, ``station_terms.csv``, ``summary.txt``
and figures ``wadati.png``, ``p_onset_lag.png``, ``s_examples_<STA>.png``.

Usage::

    python scripts/macban_tomo_qc.py \\
        /media/steve/fast9100/runs/macban_2018-01-01_2019-12-31 \\
        /media/steve/fast9100/datasets/MacBan/man_picks_extracted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

from openmeq.catalog import get_extra, read_catalog, time_from_event_id
from openmeq.config import ParallelConfig
from openmeq.parallel import pmap

_AIC_WIN = (-0.35, 0.15)
_FILTER = dict(freqmin=2.0, freqmax=30.0, corners=4, zerophase=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_auto(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, picks) from ``7_location``; picks carry ``used``/``residual``."""
    summ = pd.read_csv(run / "7_location/summary.csv").set_index("event_id")
    evs, pks = [], []
    for ev in read_catalog(run / "7_location/catalog_auto.xml"):
        eid = str(ev.resource_id).split("/")[-1]
        o = ev.preferred_origin()
        has_origin = o is not None and o.latitude is not None
        row = summ.loc[eid] if eid in summ.index else None
        evs.append(
            dict(
                event_id=eid,
                t_det=time_from_event_id(eid).timestamp,
                has_origin=has_origin,
                t0=o.time.timestamp if has_origin else np.nan,
                lat=o.latitude if has_origin else np.nan,
                lon=o.longitude if has_origin else np.nan,
                depth_km=o.depth / 1000.0 if has_origin and o.depth is not None else np.nan,
                status=row.status if row is not None else "",
                reason=str(row.reason).split(":")[0]
                if row is not None and row.reason == row.reason
                else "",
            )
        )
        arr = {str(a.pick_id): a for a in o.arrivals} if o is not None else {}
        for p in ev.picks:
            a = arr.get(str(p.resource_id))
            rejected = any(c.text.startswith("rejected:") for c in p.comments)
            pks.append(
                dict(
                    event_id=eid,
                    pick_id=str(p.resource_id),
                    station=p.waveform_id.station_code,
                    phase=p.phase_hint,
                    t_pn=p.time.timestamp,
                    prob=get_extra(p, "probability"),
                    unc_s=p.time_errors.uncertainty,
                    used=a is not None and not rejected,
                    locate_reject=next(
                        (
                            c.text.split(":", 2)[-1]
                            for c in p.comments
                            if c.text.startswith("rejected:")
                        ),
                        "",
                    ),
                    residual=a.time_residual if a is not None else np.nan,
                )
            )
    return pd.DataFrame(evs), pd.DataFrame(pks)


def load_manual(man: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    me = pd.read_csv(man / "events.csv")
    me["ot"] = me.origin_time.map(lambda s: UTCDateTime(s).timestamp)
    mp = pd.read_csv(man / "picks.csv")
    mp["t"] = mp.time.map(lambda s: UTCDateTime(s).timestamp)
    mp = mp.groupby(["event_id", "station", "phase"], as_index=False).t.mean()  # S on N/E: one pick
    return me, mp


# --------------------------------------------------------------------------- #
# qgn.3 association
# --------------------------------------------------------------------------- #


def _greedy_pairs(a_t, b_t, tol):
    """One-to-one (i, j) pairs with ``|b_t[j] - a_t[i]| <= tol``, closest first."""
    pairs = []
    order = np.argsort(b_t)
    bs = b_t[order]
    for i, t in enumerate(a_t):
        lo, hi = np.searchsorted(bs, t - tol), np.searchsorted(bs, t + tol, side="right")
        pairs += [(abs(bs[k] - t), i, order[k]) for k in range(lo, hi)]
    used_a, used_b, out = set(), set(), {}
    for _, i, j in sorted(pairs):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out[i] = j
    return out


def associate(ev, pk, det_times, me, mp, match_s):
    """One row per manual event (see module docstring, Association)."""
    me = me.sort_values("ot").reset_index(drop=True)
    # manual duplicates: the same earthquake located from two SUDS files
    # (origins < 1 s and < 1 km apart); keep the one with more stations
    dup_of = {}
    for i in range(1, len(me)):
        a, b = me.iloc[i - 1], me.iloc[i]
        dkm = np.hypot((a.latitude - b.latitude) * 111.19, (a.longitude - b.longitude) * 107.9)
        if b.ot - a.ot < 1.0 and dkm < 1.0:
            keep, drop = (a, b) if a.n_stations >= b.n_stations else (b, a)
            dup_of[drop.event_id] = dup_of.get(keep.event_id, keep.event_id)
    prim = me[~me.event_id.isin(dup_of)].reset_index(drop=True)

    loc = ev[ev.has_origin].reset_index(drop=True)
    unloc = ev[~ev.has_origin].reset_index(drop=True)
    by_origin = _greedy_pairs(prim.ot.values, loc.t0.values, match_s)
    rest = [i for i in range(len(prim)) if i not in by_origin]
    # unlocated events: detection time -5..+15 s of the manual origin
    by_det = _greedy_pairs(prim.ot.values[rest] + 5.0, unloc.t_det.values, 10.0)
    det = np.sort(det_times)

    rows = []
    for i, m in prim.iterrows():
        auto, how = None, ""
        if i in by_origin:
            auto, how = loc.iloc[by_origin[i]], "origin"
        elif i in rest and rest.index(i) in by_det:
            auto, how = unloc.iloc[by_det[rest.index(i)]], "detection"
        if auto is not None:
            status = {True: "accepted", False: "located_rejected"}[auto.status == "accepted"]
            if not auto.has_origin:
                status = "not_located"
        elif ((det >= m.ot - 5) & (det <= m.ot + 15)).any():
            status = "detected_unmatched"  # no picks, or located > match_s away
        else:
            status = "undetected"
        rows.append(
            dict(
                manual_id=m.event_id,
                manual_time=m.origin_time,
                manual_n_stations=m.n_stations,
                manual_quality=m.quality,
                auto_id=auto.event_id if auto is not None else "",
                matched_by=how,
                status=status,
                auto_reason=auto.reason if auto is not None else "",
                dt0_s=(auto.t0 - m.ot) if auto is not None and auto.has_origin else np.nan,
                duplicate_of="",
            )
        )
    for d, k in dup_of.items():
        m = me[me.event_id == d].iloc[0]
        rows.append(
            dict(
                manual_id=d,
                manual_time=m.origin_time,
                manual_n_stations=m.n_stations,
                manual_quality=m.quality,
                auto_id="",
                matched_by="",
                status="manual_duplicate",
                auto_reason="",
                dt0_s=np.nan,
                duplicate_of=k,
            )
        )
    assoc = pd.DataFrame(rows).sort_values("manual_time").reset_index(drop=True)

    # one auto pick per event/station/phase for the comparison (most probable)
    best = pk.sort_values("prob", ascending=False).drop_duplicates(["event_id", "station", "phase"])
    m2a = dict(zip(assoc.manual_id, assoc.auto_id, strict=True))
    mpm = mp[mp.event_id.isin(assoc.manual_id[assoc.auto_id != ""])].assign(
        auto_id=lambda d: d.event_id.map(m2a)
    )
    cmp_ = mpm.merge(
        best.rename(columns={"event_id": "auto_id"})[
            ["auto_id", "station", "phase", "t_pn", "prob"]
        ],
        on=["auto_id", "station", "phase"],
        how="outer",
    )
    cmp_ = cmp_[cmp_.auto_id.isin(set(assoc.auto_id) - {""})]
    cmp_["manual_id"] = cmp_.auto_id.map({a: m for m, a in m2a.items() if a})
    cmp_["dt_s"] = cmp_.t_pn - cmp_.t
    cmp_["kind"] = np.select(
        [cmp_.t.isna(), cmp_.t_pn.isna(), cmp_.dt_s.abs() > 0.3],
        ["auto_only", "manual_only", "gross"],
        "both",
    )
    cmp_ = cmp_.drop(columns=["event_id"]).rename(columns={"t": "t_manual"})
    return assoc, cmp_


# --------------------------------------------------------------------------- #
# qgn.4 P onsets
# --------------------------------------------------------------------------- #


def _aic(x: np.ndarray) -> int:
    n = len(x)
    k = np.arange(1, n - 1)
    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    m1 = c1[k - 1] / k
    v1 = c2[k - 1] / k - m1**2
    m2 = (c1[-1] - c1[k - 1]) / (n - k)
    v2 = (c2[-1] - c2[k - 1]) / (n - k) - m2**2
    a = k * np.log(np.maximum(v1, 1e-20)) + (n - k - 1) * np.log(np.maximum(v2, 1e-20))
    return int(k[np.argmin(a)])


def onset_worker(item: tuple[str, list[tuple[int, str, float]]]) -> list[dict]:
    """AIC onset + SNR for every P pick of one event (raw chunk, vertical)."""
    path, picks = item
    out = []
    try:
        st = read(path).select(component="Z")
    except Exception:  # noqa: BLE001 -- unreadable chunk: no onsets
        return [dict(idx=i, t_aic=np.nan, snr=np.nan) for i, _, _ in picks]
    st.merge(fill_value=0)
    st.detrend("demean")
    st.filter("bandpass", **_FILTER)
    for idx, sta, t in picks:
        tr = st.select(station=sta)
        t_aic = snr = np.nan
        if tr:
            tr = tr[0]
            w = tr.slice(UTCDateTime(t + _AIC_WIN[0]), UTCDateTime(t + _AIC_WIN[1]))
            if w.stats.npts > 40:
                i = _aic(w.data.astype(float))
                if 8 < i < w.stats.npts - 8:  # minimum at the window edge: no clear onset
                    t_aic = w.stats.starttime.timestamp + i / w.stats.sampling_rate
            # SNR at the onset, or at the PhaseNet time when there is none
            sr = tr.stats.sampling_rate
            ref = t_aic if t_aic == t_aic else t
            k = int(round((ref - tr.stats.starttime.timestamp) * sr))
            noise = tr.data[max(0, k - int(1.0 * sr)) : max(0, k - int(0.05 * sr))].astype(float)
            sig = tr.data[max(0, k) : k + int(0.3 * sr)].astype(float)
            if len(noise) > 20 and len(sig) > 20:
                snr = float(np.sqrt(np.mean(sig**2) / max(np.mean(noise**2), 1e-20)))
        out.append(dict(idx=idx, t_aic=t_aic, snr=snr))
    return out


def qc_p(p: pd.DataFrame, agree_s: float, noisy: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = p.copy()
    p["dt_aic"] = p.t_pn - p.t_aic
    good = p[(p.prob >= 0.9) & (p.snr >= 5) & (p.dt_aic.abs() < 0.1)]
    net_lag = good.dt_aic.median()
    lags = good.groupby("station").dt_aic.agg(["median", "size"])
    lag = {s: (r["median"] if r["size"] >= 20 else net_lag) for s, r in lags.iterrows()}
    p["lag"] = p.station.map(lag).fillna(net_lag)
    p["t_final"] = p.t_pn - p.lag
    p["confirmed"] = (p.t_final - p.t_aic).abs() <= agree_s
    noisy_m = p.station.isin(noisy)
    ok_agree = p.confirmed & (p.prob >= np.where(noisy_m, 0.7, 0.5))
    ok_conf = ~noisy_m & (((p.prob >= 0.9) & (p.snr >= 3)) | ((p.prob >= 0.7) & (p.snr >= 5)))
    p["accept"] = ok_agree | ok_conf
    p["qc_reason"] = np.select(
        [p.accept, p.t_aic.isna() & ~p.accept, noisy_m & ~p.accept],
        ["", "no_clear_onset", "noisy_station"],
        "onset_disagree_or_low_prob",
    )
    terms = pd.DataFrame({"p_lag_s": pd.Series(lag), "p_lag_n": lags["size"]})
    terms.loc["NETWORK", "p_lag_s"] = net_lag
    return p, terms


# --------------------------------------------------------------------------- #
# qgn.4 S Wadati check
# --------------------------------------------------------------------------- #


def event_wadati_slope(tp, ts, event_id, min_pairs=4):
    """Median over events of the Wadati slope ``d(tS - tP) / d(tP)`` across
    stations (independent of origin time and of station terms' mean);
    Vp/Vs = 1 + slope. Returns ``(slope, n_events)``."""
    d = pd.DataFrame(dict(tp=tp, y=ts - tp, e=event_id)).dropna()
    slopes = []
    for _, g in d.groupby("e"):
        if len(g) >= min_pairs and np.ptp(g.tp) > 0.2:
            slopes.append(np.polyfit(g.tp - g.tp.min(), g.y, 1)[0])
    return float(np.median(slopes)), len(slopes)


def fit_with_terms(x, y, sta, dev_min, k, gross, n_iter=4):
    """``y = k x + c[sta]`` with ``k`` fixed: ``c`` = median station offset
    of the points within ``gross`` s of the network-wide relation (so a
    station where PhaseNet mostly picks a later phase, e.g. MB9B S, cannot
    pull its term onto that phase), ``dev`` the residual, and the rejection
    threshold ``max(dev_min, 3 sigma)`` from the inliers' MAD.
    Returns ``(c, dev, threshold, sigma)``."""
    r0 = y - k * x
    pre = (r0 - r0.median()).abs() <= gross
    c = r0[pre].groupby(sta[pre]).median()
    dev = r0 - sta.map(c)
    inl = pd.Series(True, index=x.index)
    for _ in range(n_iter):
        sig = 1.4826 * float(np.median(np.abs(dev[inl] - np.median(dev[inl]))))
        thr = max(dev_min, 3 * sig)
        inl = dev.abs() <= thr
    return c, dev, thr, sig


def qc_s(s, p, ev, coords, dev_min, noisy):
    """S check: Wadati (S-P vs P travel time) when the station has an accepted
    P, else S travel time vs hypocentral distance; both with station terms."""
    s = s.copy()
    pk = p[p.accept].set_index(["event_id", "station"]).t_final
    s["tp"] = [pk.get((e, st), np.nan) for e, st in zip(s.event_id, s.station, strict=True)]
    e = ev.set_index("event_id")
    s["t0"] = s.event_id.map(e.t0)
    lat0, lon0 = s.event_id.map(e.lat), s.event_id.map(e.lon)
    dz = s.event_id.map(e.depth_km) + s.station.map(coords.elevation_m) / 1000.0
    dx = (s.station.map(coords.longitude) - lon0) * 111.19 * np.cos(np.radians(lat0))
    dy = (s.station.map(coords.latitude) - lat0) * 111.19
    s["r_km"] = np.sqrt(dx**2 + dy**2 + dz**2)
    s["x"] = s.tp - s.t0
    s["y"] = s.t_pn - s.tp

    w = s[s.tp.notna() & (s.x > 0.1) & (s.y > 0)]
    k, n_ev = event_wadati_slope(w.tp, w.t_pn, w.event_id)
    c, dev, thr, sig = fit_with_terms(w.x, w.y, w.station, dev_min, k, gross=0.8)
    s["check"] = np.where(s.index.isin(w.index), "wadati", "distance")
    s["dev"] = dev.reindex(s.index)
    s["c_station"] = s.station.map(c)
    s["threshold"] = thr

    d = s[(s.check == "distance") & s.r_km.notna() & (s.t_pn > s.t0)]
    all_d = s[s.r_km.notna() & (s.t_pn > s.t0)]
    # fit the distance relation on every S with an origin, apply it to those without P
    tt = all_d.t_pn - all_d.t0
    kd = float(np.median(tt[all_d.r_km > 2.0] / all_d.r_km[all_d.r_km > 2.0]))
    cd, devd, thrd, sigd = fit_with_terms(
        all_d.r_km, tt, all_d.station, 1.5 * dev_min, kd, gross=1.2
    )
    s.loc[d.index, "dev"] = devd.reindex(d.index)
    s.loc[d.index, "threshold"] = thrd

    noisy_m = s.station.isin(noisy)
    in_fit = s.dev.abs() <= s.threshold
    s["accept"] = in_fit & (s.prob >= np.where(noisy_m, 0.8, 0.5))
    s["qc_reason"] = np.select(
        [s.accept, s.dev.isna(), noisy_m & in_fit],
        ["", "no_check", "noisy_station"],
        s.check + "_outlier",
    )
    s["wadati_dev"] = np.where(s.check == "wadati", s.dev, np.nan)
    fit = dict(
        vpvs=1.0 + k,
        n_wadati_events=n_ev,
        sigma_s=sig,
        threshold_s=thr,
        vs_app_kms=1.0 / kd,
        dist_sigma_s=sigd,
        dist_threshold_s=thrd,
    )
    return s, c, cd, fit


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_wadati(s, k, c, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stas = sorted(s.station[s.wadati_dev.notna()].unique())
    nc = 4
    fig, axs = plt.subplots(
        -(-len(stas) // nc), nc, figsize=(16, 3.2 * -(-len(stas) // nc)), squeeze=False
    )
    xx = np.linspace(0, 2.5, 10)
    for ax, sta in zip(axs.flat, stas, strict=False):  # more axes than stations
        d = s[(s.station == sta) & s.wadati_dev.notna()]
        ax.plot(d.x[d.accept], d.y[d.accept], ".", ms=3, color="tab:blue", label="accepted")
        ax.plot(d.x[~d.accept], d.y[~d.accept], "x", ms=3, color="tab:red", label="rejected")
        ax.plot(xx, k * xx + c.get(sta, 0.0), "k-", lw=1)
        ax.plot(xx, k * xx, "k:", lw=0.8)
        ax.set_title(
            f"{sta}: c = {c.get(sta, 0.0):+.3f} s  ({d.accept.sum()}/{len(d)})", fontsize=9
        )
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 2.5)
    for ax in axs.flat[len(stas) :]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=7)
    fig.supxlabel("P travel time tP - t0 (s)")
    fig.supylabel("S - P (s)")
    fig.suptitle(
        f"S-P vs P travel time, Vp/Vs = {1 + k:.3f}; solid = with station term, dotted = without"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_lag(p, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = p[p.dt_aic.abs() < 0.2]
    stas = sorted(d.station.unique())
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.boxplot([d.dt_aic[d.station == s] * 1000 for s in stas], showfliers=False)
    ax.set_xticks(range(1, len(stas) + 1), stas)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("PhaseNet P - AIC onset (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def plot_s_examples(s, chunks: Path, sta: str, out, n=8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = s[(s.station == sta) & s.accept & (s.prob >= 0.8) & (s.check == "wadati")]
    if d.empty:
        return
    d = d.sample(min(n, len(d)), random_state=0)
    fig, axs = plt.subplots(len(d), 1, figsize=(12, 2.3 * len(d)), squeeze=False)
    for ax, r in zip(axs[:, 0], d.itertuples(), strict=True):
        st = read(str(chunks / r.event_id / f"{r.event_id}.mseed")).select(station=sta)
        st.detrend("demean")
        st.filter("bandpass", **_FILTER)
        t0 = UTCDateTime(r.tp - 0.3)
        for i, comp in enumerate("ZNE"):
            for tr in st.select(component=comp):
                w = tr.slice(t0, t0 + r.y + 1.5)
                ax.plot(
                    w.times(reftime=t0),
                    w.data / (np.abs(w.data).max() or 1) * 0.45 + i,
                    "k",
                    lw=0.6,
                )
            ax.text(-0.05, i, comp, ha="right", va="center")
        ax.axvline(0.3, color="tab:blue")
        ax.axvline(0.3 + r.y, color="tab:red")
        ax.axvline(0.3 + r.y - r.c_station, color="tab:red", ls=":")
        ax.set_yticks([])
        ax.set_title(
            f"{sta} {r.event_id}  S p={r.prob:.2f}  S-P={r.y:.2f} s  dev={r.dev:+.3f}", fontsize=8
        )
    axs[-1, 0].set_xlabel(
        "s (P at 0.3 s; red = S pick, dotted = S predicted without the station term)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="OpenMEQ run directory")
    ap.add_argument("manual", type=Path, help="output of scripts/macban_man_picks.py")
    ap.add_argument("--out", type=Path, help="default RUN/tomo/qc")
    ap.add_argument("--match-s", type=float, default=2.0, help="origin-time match tolerance (s)")
    ap.add_argument("--p-agree-s", type=float, default=0.05, help="PhaseNet (lag-corrected) vs AIC")
    ap.add_argument("--s-dev-min-s", type=float, default=0.10, help="minimum S Wadati threshold")
    ap.add_argument("--noisy", default="MB9A", help="comma-separated noisy stations")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args(argv)
    out = args.out or args.run / "tomo" / "qc"
    out.mkdir(parents=True, exist_ok=True)
    noisy = {s for s in args.noisy.split(",") if s}

    ev, pk = load_auto(args.run)
    me, mp = load_manual(args.manual)
    det = pd.read_csv(args.run / "2_detection/detections.csv")
    det_t = det.time.map(lambda s: UTCDateTime(s).timestamp).values

    # qgn.3
    assoc, cmp_ = associate(ev, pk, det_t, me, mp, args.match_s)
    assoc.to_csv(out / "association.csv", index=False)
    cmp_.to_csv(out / "station_compare.csv", index=False)

    # qgn.4 -- events with an origin; one pick per event/station/phase
    pk = pk[pk.event_id.isin(ev.event_id[ev.has_origin])]
    pk = pk.sort_values(["used", "prob"], ascending=False).drop_duplicates(
        ["event_id", "station", "phase"]
    )
    p = pk[pk.phase == "P"].reset_index(drop=True)
    chunks = args.run / "4_data_chunks/chunks"
    items = [
        (str(chunks / e / f"{e}.mseed"), [(i, r.station, r.t_pn) for i, r in g.iterrows()])
        for e, g in p.groupby("event_id")
    ]
    res = pmap(
        onset_worker, items, ParallelConfig(n_workers=args.n_workers), desc="P onsets", chunksize=4
    )
    on = pd.DataFrame([r for rr in res for r in rr]).set_index("idx")
    p = p.join(on)
    p, terms = qc_p(p, args.p_agree_s, noisy)
    coords = (
        pd.read_csv(args.run / "0_setup/stations.csv")
        .drop_duplicates("station")
        .set_index("station")
    )
    s, c, cd, fit = qc_s(
        pk[pk.phase == "S"].reset_index(drop=True), p, ev, coords, args.s_dev_min_s, noisy
    )
    terms["s_minus_p_term_s"] = c
    terms["s_tt_term_s"] = cd
    terms["n_s_accepted"] = s[s.accept].groupby("station").size()
    terms["n_p_accepted"] = p[p.accept].groupby("station").size()
    terms.to_csv(out / "station_terms.csv", index_label="station")

    m_conf = {a for a in assoc.auto_id if a}
    picks = pd.concat([p, s], ignore_index=True)
    picks.loc[picks.phase == "S", "t_final"] = picks.t_pn  # S: PhaseNet time (no onset correction)
    picks["manual_confirmed_event"] = picks.event_id.isin(m_conf)
    for col in ("t_pn", "t_aic", "t_final"):
        picks[col.replace("t_", "time_")] = picks[col].map(
            lambda v: "" if v != v else UTCDateTime(v).isoformat()
        )
    picks.to_csv(out / "picks_qc.csv", index=False)

    acc = picks[picks.accept].groupby(["event_id", "phase"]).station.nunique().unstack(fill_value=0)
    evq = (
        ev[ev.has_origin]
        .set_index("event_id")
        .join(acc.rename(columns={"P": "n_p_ok", "S": "n_s_ok"}))
        .fillna({"n_p_ok": 0, "n_s_ok": 0})
    )
    evq["manual_id"] = evq.index.map(
        {a: m for m, a in zip(assoc.manual_id, assoc.auto_id, strict=True) if a}
    )
    evq.to_csv(out / "events_qc.csv")

    # review list: accepted picks that differ grossly from the manual pick, and
    # manual P at stations with no accepted automatic P in accepted events
    g = cmp_[cmp_.kind == "gross"].merge(
        picks[["event_id", "station", "phase", "accept"]].rename(columns={"event_id": "auto_id"}),
        how="left",
    )
    mo = cmp_[(cmp_.kind == "manual_only") & (cmp_.phase == "P")]
    mo = mo[mo.auto_id.isin(evq.index[evq.status == "accepted"])]
    pd.concat([g.assign(review="gross_dt"), mo.assign(review="manual_p_no_auto")]).to_csv(
        out / "review_candidates.csv", index=False
    )

    plot_wadati(s, fit["vpvs"] - 1.0, c, out / "wadati.png")
    plot_lag(p, out / "p_onset_lag.png")
    for sta in ("MB07", "MB7A", "MB9B"):
        plot_s_examples(s, chunks, sta, out / f"s_examples_{sta}.png")

    kinds = cmp_.groupby("phase").kind.value_counts().unstack(fill_value=0).to_dict("index")
    by_check = s.groupby("check").accept.agg(["size", "sum"]).to_dict("index")
    good = (evq.n_p_ok >= 5) & (evq.n_s_ok >= 3)
    lines = [
        f"manual events: {len(assoc)}; status: {assoc.status.value_counts().to_dict()}",
        f"station comparison kinds: {kinds}",
        f"P picks: {len(p)}, accepted {int(p.accept.sum())} "
        f"(AIC-confirmed {int((p.accept & p.confirmed).sum())}); "
        f"reasons {p.qc_reason.value_counts().to_dict()}",
        f"P lag (PhaseNet - AIC): network {terms.loc['NETWORK', 'p_lag_s'] * 1000:.1f} ms",
        f"S picks: {len(s)}, accepted {int(s.accept.sum())}; "
        f"reasons {s.qc_reason.value_counts().to_dict()}",
        f"Wadati: Vp/Vs {fit['vpvs']:.3f} (median of {fit['n_wadati_events']} event fits), "
        f"sigma {fit['sigma_s']:.3f} s, threshold {fit['threshold_s']:.3f} s",
        f"S tt vs distance: Vs_app {fit['vs_app_kms']:.2f} km/s, "
        f"sigma {fit['dist_sigma_s']:.3f} s, threshold {fit['dist_threshold_s']:.3f} s; "
        f"S by check {by_check}",
        f"events with an origin: {len(evq)}; n_p_ok>=5: {int((evq.n_p_ok >= 5).sum())}; "
        f"n_p_ok>=5 & n_s_ok>=3: {int(good.sum())}; "
        f"of those manual-confirmed: {int((good & evq.manual_id.notna()).sum())}",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n\n" + terms.round(3).to_string() + "\n")
    print("\n".join(lines))
    print(terms.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
