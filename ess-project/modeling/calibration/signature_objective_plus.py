#!/usr/bin/env python3
"""
signature_objective_plus.py -- extended signature objective (v2). Imports signature_objective.py UNCHANGED.

Why: the evaluation of the aquifer refits showed the v1 terms cannot see two things that matter for the paper:
  * total volume  (v1 has none; every run sat at about +4-5%)
  * annual peak flow -- a KGE-style refit of the baseline aquifer raised the 7-day annual peak by ~20 points
    (+9.7% -> +30%) and nothing in v1 or in KGE objected.
v2 therefore scores (each a normalised error, 0 = perfect, clipped at 3):
  com_timing      directional centre-of-mass timing, mean |d| / 7 d                          [v1 term]
  monthly_volume  volume-significant monthly bias                                             [v1 term]
  total_volume    |sum(sim)/sum(obs) - 1| / 5%
  peak_flow       mean over WY of |annual max 7-day flow: sim - obs| / obs, / 15%
  peak_timing     mean over WY of |date of annual max 7-day flow: sim - obs| / 7 d
  lowflow_sepnov  mean |d log10| of the annual Sep-Nov 7-day minimum                          [v1 term]
  recession       |tau_sim - tau_target| / tau_target, default weight 0.5                      [v1 term]
  swe_peak        Butte-HRU peak SWE |rel err| beyond a 20% dead band, / 15%                 (soft, weight 0.5; only if swe given)
  swe_melt        mean of melt-out-date and melt-season-length errors outside [-5 d, +15 d] (model minus SNOTEL; later is allowed), / 7 d   (soft, weight 0.5)
  ASO is NEVER in the objective (about 6 flights): see swe_evaluation() for the evaluation-only basin metrics.
`concentration` (directional R) is dropped: peak_flow / peak_timing measure the same thing directly.
The SWE terms need the model's per-HRU daily SWE; with snow parameters frozen they are constant and carry no weight in a fit.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import signature_objective as S
OBS = HERE/"observations"
Q_FLOOR = S.Q_FLOOR
DEFAULT_WEIGHTS_V2 = {"com_timing": 1.0, "monthly_volume": 1.0, "total_volume": 1.0, "peak_flow": 1.0, "peak_timing": 1.0,
                      "lowflow_sepnov": 1.0, "recession": 0.5, "swe_peak": 0.5, "swe_melt": 0.5}   # SWE terms are SOFT (weight 0.5, dead-banded)
def _wy(idx): return idx.year + (idx.month >= 10)

# ---------------- streamflow terms ----------------
def annual_peak7(q: pd.Series):
    """per water year: (annual max of the centred 7-day mean flow, its day of water year)."""
    r = q.rolling(7, center=True, min_periods=7).mean().dropna(); out = {}
    for y, g in r.groupby(_wy(r.index)):
        if len(g) < 300: continue
        t = g.idxmax(); out[y] = (float(g.max()), (t - pd.Timestamp(int(y)-1, 10, 1)).days + 1)
    return pd.DataFrame(out, index=["peak", "dowy"]).T
def sig_total_volume(sim, obs, norm=0.05): return float(np.clip(abs(sim.sum()/obs.sum()-1.0)/norm, 0, 3))
def sig_peak_flow(sim, obs, norm=0.15):
    s, o = annual_peak7(sim), annual_peak7(obs); c = s.index.intersection(o.index)
    return 3.0 if len(c) == 0 else float(np.clip(((s.peak[c]-o.peak[c]).abs()/o.peak[c]).mean()/norm, 0, 3))
def sig_peak_timing(sim, obs, norm_days=7.0):
    s, o = annual_peak7(sim), annual_peak7(obs); c = s.index.intersection(o.index)
    return 3.0 if len(c) == 0 else float(np.clip((s.dowy[c]-o.dowy[c]).abs().mean()/norm_days, 0, 3))

# ---------------- SWE: soft calibration terms (SNOTEL) + evaluation-only metrics (ASO) ----------------
# Design (2026-10-14): ASO has ~6 East flight dates (5-6 paired years): far too few to calibrate to without overfitting, so it is
# EVALUATION ONLY. The 42-year Butte SNOTEL record informs the objective weakly, through DEAD-BANDED terms: no penalty inside the
# tolerance, linear beyond it, small weight. The tolerances are deliberately loose (Butte HRU 13 sits 90 m below the pillow; a point
# is compared with an HRU), so the terms only push back on a clearly wrong snowpack.
# melt-out and season-length errors are model MINUS observed (days), so + means the model is LATER than SNOTEL. The band is asymmetric on
# purpose (2026-10-14): ASO shows snow persisting around the pillow after the pillow itself melts out, and a later melt also pushes part of the
# melt pulse later. Zero penalty for a melt-out from 5 d early to 15 d late (per year); linear beyond, /7 d.
SWE_TOL = dict(peak_rel=0.20, melt_lo=-5.0, melt_hi=15.0, dur_lo=-5.0, dur_hi=15.0)       # edit to taste
MELT_TH, MELT_CONFIRM, MELT_MIN_DOWY = 25.4, 14, 152               # same definitions as utils/snotel_metrics.py
def _melt_out(g, ipk, oct1):
    w = g.loc[ipk:]; above = w[w > MELT_TH]
    if len(above) == 0 or above.index[-1] == w.index[-1]: return np.nan
    last = above.index[-1]
    if (w.index[-1]-last).days < MELT_CONFIRM: return np.nan
    d = (last-oct1).days + 1
    return d if d >= MELT_MIN_DOWY else np.nan
_BUTTE = pd.read_csv(OBS/"east_butte_swe_annual.csv"); _ASO = pd.read_csv(OBS/"east_aso_hru_swe.csv", parse_dates=["date"])
def _deadband(err, tol, norm): return float(max(0.0, err-tol)/norm)
def _band_penalty(e, lo, hi, norm):
    """mean over years of the distance outside [lo, hi] (days), / norm. NaN years ignored."""
    e = pd.Series(e).dropna()
    return float((np.maximum(0.0, lo-e) + np.maximum(0.0, e-hi)).mean()/norm) if len(e) else np.nan
def butte_swe_stats(swe_hru: pd.DataFrame, butte_hru: int = 13):
    m = swe_hru[butte_hru].dropna(); rows = []
    for y, o in _BUTTE.set_index("wy").iterrows():
        g = m[_wy(m.index) == y]
        if len(g) < 300: continue
        oct1 = pd.Timestamp(int(y)-1, 10, 1); ipk = g.idxmax(); tm = _melt_out(g, ipk, oct1); tp = (ipk-oct1).days
        ok = np.isfinite(tm)
        rows.append(dict(wy=y, peak_rel=abs(g.max()-o.peak_obs)/_BUTTE.peak_obs.mean(), peak_bias=(g.max()-o.peak_obs)/_BUTTE.peak_obs.mean(),
                         melt_err=(tm-o.tmelt_obs) if ok else np.nan, dur_err=((tm-tp)-(o.tmelt_obs-o.tpeak_obs)) if ok else np.nan))
    return pd.DataFrame(rows)
def swe_signatures(swe_hru: pd.DataFrame, hru_area: pd.Series = None, butte_hru: int = 13):
    """Soft SWE terms for the objective (Butte SNOTEL only; dead-banded). swe_hru: daily-mean SWE (mm), columns = hruId.
    Returns (swe_peak, swe_melt). swe_melt averages the melt-out-date and the peak->melt-out duration terms (duration = melt-season length)."""
    b = butte_swe_stats(swe_hru, butte_hru)
    peak = _deadband(b.peak_rel.mean(), SWE_TOL["peak_rel"], 0.15)
    melt = 0.5*_band_penalty(b.melt_err, SWE_TOL["melt_lo"], SWE_TOL["melt_hi"], 7.0) + 0.5*_band_penalty(b.dur_err, SWE_TOL["dur_lo"], SWE_TOL["dur_hi"], 7.0)
    return float(np.clip(peak, 0, 3)), float(np.clip(melt, 0, 3))
def swe_evaluation(swe_hru: pd.DataFrame, hru_area: pd.Series, hru_elev: pd.Series, butte_hru: int = 13):
    """EVALUATION ONLY (never in the objective): Butte point-vs-HRU stats and ASO basin-scale metrics.
    Returns a dict: Butte peak/melt-out/duration bias and MAE; ASO volume bias in spring (Mar-Apr) and in the melt season (May-Jun);
    ASO elevation-band bias; and, for years with a spring AND a melt-season flight, the fraction of spring SWE remaining (a melt-season
    progress / length proxy) in ASO vs model."""
    b = butte_swe_stats(swe_hru, butte_hru); out = dict(butte_n=len(b), butte_peak_bias_pct=100*b.peak_bias.mean(), butte_peak_nmae_pct=100*b.peak_rel.mean(),
        butte_melt_bias_d=b.melt_err.mean(), butte_melt_mae_d=b.melt_err.abs().mean(), butte_dur_bias_d=b.dur_err.mean(),
        butte_melt_in_band_frac=float(b.melt_err.dropna().between(SWE_TOL['melt_lo'], SWE_TOL['melt_hi']).mean()), butte_dur_in_band_frac=float(b.dur_err.dropna().between(SWE_TOL['dur_lo'], SWE_TOL['dur_hi']).mean()))
    rows = []
    for dt, g in _ASO[(_ASO.covf >= 0.7)].groupby("date"):
        if dt not in swe_hru.index: continue
        h = g[g.hru.isin(swe_hru.columns)]
        if len(h) < 12: continue
        w = hru_area.reindex(h.hru).values; w = w/w.sum()
        rows.append(dict(date=dt, aso=float((h.aso_mm.values*w).sum()), sim=float((swe_hru.loc[dt, h.hru].values*w).sum())))
    f = pd.DataFrame(rows)
    if len(f):
        f["yr"] = f.date.dt.year; f["spring"] = f.date.dt.month <= 4
        sp, ml = f[f.spring], f[~f.spring]
        out.update(aso_spring_bias_pct=100*(sp.sim.sum()/sp.aso.sum()-1), aso_spring_mae_pct=100*((sp.sim-sp.aso).abs()/sp.aso).mean(), n_aso_spring=len(sp),
                   aso_melt_bias_pct=100*(ml.sim.sum()/ml.aso.sum()-1), aso_melt_mae_pct=100*((ml.sim-ml.aso).abs()/ml.aso).mean(), n_aso_melt=len(ml))
        pr = []
        for y, g in f.groupby("yr"):
            if g.spring.any() and (~g.spring).any():
                e, l = g[g.spring].iloc[0], g[~g.spring].iloc[-1]; pr.append(dict(yr=y, aso_remaining=l.aso/e.aso, mod_remaining=l.sim/e.sim))
        pr = pd.DataFrame(pr)
        if len(pr): out.update(n_pairs=len(pr), remaining_mae_pts=100*(pr.mod_remaining-pr.aso_remaining).abs().mean(), remaining_bias_pts=100*(pr.mod_remaining-pr.aso_remaining).mean(), pairs=pr)
    sp_ = _ASO[(_ASO.covf >= 0.7) & (_ASO.date.dt.month <= 4) & _ASO.date.isin(swe_hru.index)].copy()
    if len(sp_):
        sp_["sim"] = [swe_hru.loc[d, h] for d, h in zip(sp_.date, sp_.hru)]; sp_["elev"] = sp_.hru.map(hru_elev)
        sp_["band"] = pd.cut(sp_.elev, [0, 2800, 3200, 3500, 4000], labels=["~2700 m", "~3000 m", "~3350 m", "~3700 m"])
        out["aso_band_ratio"] = sp_.groupby("band").apply(lambda x: x["sim"].sum()/x.aso_mm.sum()).to_dict()
    return out

# ---------------- combined objective ----------------
def signature_objective_plus(sim: pd.Series, obs: pd.Series, tau_target_days: float, weights: Optional[Dict[str, float]] = None,
                             area_m2: Optional[float] = None, swe: Optional[tuple] = None, lowflow_over_weight: float = 1.0,
                             exclude_months: tuple = ()) -> S.SignatureResult:
    """swe: optional (swe_peak, swe_melt) from swe_signatures(); the SWE terms are included only when given."""
    w = dict(DEFAULT_WEIGHTS_V2); w.update(weights or {})
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if len(j) < 60: return S.SignatureResult(objective=1e6, parts={"error": np.nan})
    s, o = j.s.clip(lower=Q_FLOOR), j.o.clip(lower=Q_FLOOR)
    parts = {"com_timing": S.sig_com_timing(s, o), "monthly_volume": S.sig_monthly_volume_significant(s, o, area_m2=area_m2, exclude_months=exclude_months),
             "total_volume": sig_total_volume(s, o), "peak_flow": sig_peak_flow(s, o), "peak_timing": sig_peak_timing(s, o),
             "lowflow_sepnov": S.sig_lowflow_sepnov(s, o, over_weight=lowflow_over_weight), "recession": S.sig_recession(s, tau_target_days)}
    if swe is not None: parts["swe_peak"], parts["swe_melt"] = swe
    wsum = sum(w[k] for k in parts); obj = sum(w[k]*parts[k] for k in parts)/wsum
    return S.SignatureResult(objective=float(obj), parts=parts, diagnostics={"tau_sim_days": S.recession_constant(s)})


# ---------------- band-resolved ASO evaluation (EVALUATION ONLY; never in the objective) ----------------
def aso_band_evaluation(swe_hru: pd.DataFrame, aso: pd.DataFrame, hru_area: pd.Series, hru_band: pd.Series, band_order: list, min_cov: float = 0.7):
    """Independent check that the model's precipitation/snow elevation pattern mimics ASO. swe_hru: daily SWE (mm), columns = hruId.
    aso: columns date, hru, aso_mm, covf (per-HRU flight means). hru_band: hruId -> band label; band_order: labels low->high.
    Returns spring (<= Apr) and melt-season (>= May) model/ASO ratio per band (area-weighted over HRU-flights), the normalised spring elevation profile of both
    (band SWE / all-band mean) with their correlation and the band holding the maximum, and the melt-season retention ratio per band."""
    a = aso[(aso.covf >= min_cov) & aso.aso_mm.notna() & aso.date.isin(swe_hru.index)].drop_duplicates(["date", "hru"])
    a = a[a.hru.isin(swe_hru.columns) & a.hru.isin(hru_area.index)]
    if not len(a): return {}
    a = a.assign(mdl=[swe_hru.at[d, h] for d, h in zip(a.date, a.hru)], area=hru_area.reindex(a.hru).values, band=hru_band.reindex(a.hru).values)
    out = {"n_flights": int(a.date.nunique())}
    for nm, sub in (("spring", a[a.date.dt.month <= 4]), ("melt", a[a.date.dt.month >= 5])):
        if not len(sub): continue
        g = sub.groupby("band").apply(lambda x: pd.Series(dict(aso=np.average(x.aso_mm, weights=x.area), mdl=np.average(x.mdl, weights=x.area))))
        g = g.reindex([b for b in band_order if b in g.index]); out[f"{nm}_ratio"] = (g.mdl/g.aso.where(g.aso > 5)).round(2).to_dict()
        out[f"{nm}_n_flights"] = int(sub.date.nunique())
        if nm == "spring" and len(g) >= 3:
            pa, pm = g.aso/g.aso.mean(), g.mdl/g.mdl.mean()
            out["spring_profile_corr"] = float(np.corrcoef(pa, pm)[0, 1]); out["spring_peak_band_aso"] = str(pa.idxmax()); out["spring_peak_band_model"] = str(pm.idxmax())
            out["spring_profile_aso"] = pa.round(2).to_dict(); out["spring_profile_model"] = pm.round(2).to_dict()
    return out


# ---------------- control objective (2026-10-20): within-year melt-season terms only ----------------
# The control is the no-storage baseline for the memory experiments (deep store in East, perennial snow in Tuolumne), so it is fitted ONLY on
# within-year behaviour. Late-season terms (lowflow_sepnov, recession) and the absolute annual peak are reported, never optimised: the memory test
# regresses low flow on same-year peak flow, so per-year volume errors (largely forcing) are absorbed there and chasing them distorts parameters.
# Every scored term is computed in the basin's MELT WINDOW (East Mar-Jul, Tuolumne Apr-Jul) so winter baseflow / forcing-limited winter months
# cannot pull timing or volume (East's winter-baseflow excess moved the 10%-volume date ~45 d early; Tuolumne's COM included the Dec-Mar months
# that monthly_volume already excluded, and the optimiser warmed the forcing to make up winter flow it cannot produce).
DEFAULT_WEIGHTS_CONTROL = {"com_melt": 1.0, "peak_timing_melt": 1.0, "rising_limb": 1.0, "monthly_volume_melt": 1.0, "total_volume": 1.0,
                           "peak_shape": 1.0, "swe_peak": 0.5, "swe_melt": 0.5}
def _melt(q, months): return q[q.index.month.isin(months)]
def sig_com_melt(sim, obs, months, norm_days=7.0):
    """mean over WY of |flow-weighted centroid date within the melt window: sim - obs| / 7 d."""
    def ct(q):
        q = _melt(q, months); out = {}
        for y, g in q.groupby(_wy(q.index)):
            if len(g) < 25*len(months): continue
            t = (g.index - pd.Timestamp(int(y)-1, 10, 1)).days.values; out[y] = float((t*g.values).sum()/g.values.sum())
        return pd.Series(out)
    s, o = ct(sim), ct(obs); c = s.index.intersection(o.index)
    return 3.0 if len(c) == 0 else float(np.clip((s[c]-o[c]).abs().mean()/norm_days, 0, 3))
def _peak7_melt(q, months):
    r = q.rolling(7, center=True, min_periods=7).mean(); r = _melt(r, months).dropna(); out = {}
    for y, g in r.groupby(_wy(r.index)):
        if len(g) < 25*len(months): continue
        t = g.idxmax(); out[y] = (float(g.max()), (t - pd.Timestamp(int(y)-1, 10, 1)).days + 1, float(_melt(q, months)[_wy(_melt(q, months).index) == y].mean()))
    return pd.DataFrame(out, index=["peak", "dowy", "mean"]).T
def sig_peak_timing_melt(sim, obs, months, norm_days=7.0):
    s, o = _peak7_melt(sim, months), _peak7_melt(obs, months); c = s.index.intersection(o.index)
    return 3.0 if len(c) == 0 else float(np.clip((s.dowy[c]-o.dowy[c]).abs().mean()/norm_days, 0, 3))
def sig_peak_shape(sim, obs, months, norm=0.15):
    """peakiness independent of volume: per WY, (melt-window 7-day max / melt-window mean flow), mean |sim/obs - 1| / 15%."""
    s, o = _peak7_melt(sim, months), _peak7_melt(obs, months); c = s.index.intersection(o.index)
    if len(c) == 0: return 3.0
    rs, ro = s.peak[c]/s["mean"][c], o.peak[c]/o["mean"][c]
    return float(np.clip((rs/ro-1.0).abs().mean()/norm, 0, 3))
def sig_rising_limb_melt(sim, obs, months, fracs=(0.10, 0.25), norm_days=7.0):
    """mean over WY and over the 10% / 25% melt-window-volume dates of |sim date - obs date| / 7 d."""
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna(); j = _melt(j, months); errs = []
    for y, g in j.groupby(_wy(j.index)):
        if len(g) < 25*len(months): continue
        for fr in fracs:
            d = [np.argmax((g[k].cumsum()/g[k].sum()).values >= fr) for k in ("s", "o")]; errs.append(abs(d[0]-d[1]))
    return 3.0 if not errs else float(np.clip(np.mean(errs)/norm_days, 0, 3))
def lowflow_peak_residual_lag1(q):
    """memory diagnostic (REPORTED ONLY): SPEARMAN lag-1 correlation of the residuals of ln(Aug 1-Nov 30 7-day-min flow) regressed on
    ln(Apr-Jul 7-day peak), per calendar year. Same definition as emu_core.memory (res1) used for the deep-store results -- keep them identical.
    The control should give about 0 here; the storage experiments are judged on matching the observed value."""
    from scipy.stats import spearmanr
    mn, pk = {}, {}
    for y in sorted(set(q.index.year)):
        a = q.loc[f"{y}-08-01":f"{y}-11-30"]
        if a.isna().mean() <= 0.1 and len(a) > 100:
            m = a.rolling(7, center=True, min_periods=7).mean().min()
            if np.isfinite(m) and m > Q_FLOOR: mn[y] = m
        wv = q.loc[f"{y}-04-01":f"{y}-07-31"]
        if len(wv) > 100 and wv.isna().mean() <= 0.05: pk[y] = wv.interpolate(limit=5).rolling(7, center=True, min_periods=7).mean().max()
    d = pd.concat([np.log(pd.Series(mn)).rename("y"), np.log(pd.Series(pk)).rename("x")], axis=1).dropna()
    if len(d) < 6: return np.nan
    r = d.y - np.polyval(np.polyfit(d.x, d.y, 1), d.x)
    p = pd.DataFrame({"a": r.reindex(r.index - 1).values, "b": r.values}).dropna()
    return float(spearmanr(p.a, p.b)[0]) if len(p) > 5 else np.nan
def signature_objective_control(sim, obs, melt_months, weights=None, area_m2=None, swe=None, exclude_months=(), tau_target_days=None):
    w = dict(DEFAULT_WEIGHTS_CONTROL); w.update(weights or {})
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if len(j) < 60: return S.SignatureResult(objective=1e6, parts={"error": np.nan})
    s, o = j.s.clip(lower=Q_FLOOR), j.o.clip(lower=Q_FLOOR)
    kept = s.index.month.isin([m for m in range(1, 13) if m not in exclude_months])
    non_melt = tuple(m for m in range(1, 13) if m not in melt_months)
    parts = {"com_melt": sig_com_melt(s, o, melt_months), "peak_timing_melt": sig_peak_timing_melt(s, o, melt_months),
             "rising_limb": sig_rising_limb_melt(s, o, melt_months),
             "monthly_volume_melt": S.sig_monthly_volume_significant(s, o, area_m2=area_m2, exclude_months=tuple(set(exclude_months) | set(non_melt))),
             "total_volume": sig_total_volume(s[kept], o[kept]), "peak_shape": sig_peak_shape(s, o, melt_months)}
    if swe is not None: parts["swe_peak"], parts["swe_melt"] = swe
    wsum = sum(w[k] for k in parts); obj = sum(w[k]*parts[k] for k in parts)/wsum
    diag = {"REPORTED_lowflow_sepnov": S.sig_lowflow_sepnov(s, o), "REPORTED_peak_flow": sig_peak_flow(s, o), "REPORTED_com_timing_fullyear": S.sig_com_timing(s, o),
            "resid_lag1_sim": lowflow_peak_residual_lag1(s), "resid_lag1_obs": lowflow_peak_residual_lag1(o),
            "KGE": 1.0 - S._kge_cost(s.values, o.values), "NSE": 1.0 - S._nse_cost(s.values, o.values), "tau_sim_days": S.recession_constant(s)}
    if tau_target_days: diag["REPORTED_recession"] = S.sig_recession(s, tau_target_days)
    return S.SignatureResult(objective=float(obj), parts=parts, diagnostics=diag)


# ---------------------------------------------------------------------------------------------------------------------------------
# ROUTING/TIMING objective (East control, 2026-10-22). Water volume is taken as right (precipitation and temperature frozen); only
# routing and timing are scored. Five equally weighted terms, NO low-flow terms:
#   peak_timing   mean |date of annual 7-day peak, sim - obs| / 7 d
#   peak_volume   mean |7-day peak, sim/obs - 1| / 15%
#   falling_limb  mean |days from the 7-day peak until 7-day flow falls to 25% of the peak, sim - obs| / 7 d
#   com_timing    mean |water-year centre-of-mass date, sim - obs| / 7 d
#   monthly_bias  sum|monthly-mean bias| / sum(obs monthly mean) over Apr-Oct (Nov-Mar excluded) / 10%
# Each term clipped at 3. Reported (not scored): post-peak recession e-folding time (ln-linear fit, peak -> 25%), KGE/NSE, memory lag-1.
DEFAULT_WEIGHTS_ROUTING = dict(peak_timing=1.0, peak_volume=1.0, falling_limb=1.0, com_timing=1.0, monthly_bias=1.0)
ROUTING_BIAS_MONTHS = (4, 5, 6, 7, 8, 9, 10)


def _per_wy_peak(q):
    r = q.rolling(7, center=True, min_periods=7).mean(); out = {}
    for y, g in r.groupby(_wy(r.index)):
        g = g.dropna()
        if len(g) < 300: continue
        t = g.idxmax(); post = g.loc[t:]; b = post[post <= 0.25*g.max()]
        lim = post.loc[:b.index[0]] if len(b) else None
        tau = np.nan
        if lim is not None and len(lim) >= 5:
            tau = -1.0/np.polyfit(np.arange(len(lim)), np.log(lim.clip(lower=Q_FLOOR).values), 1)[0]
        out[y] = dict(date=t, peak=float(g.max()), d25=float((b.index[0] - t).days) if len(b) else np.nan, tau=tau)
    return pd.DataFrame(out).T


def _com_doy(q):
    return q.groupby(_wy(q.index)).apply(lambda g: float((np.arange(len(g))*g.values).sum()/g.values.sum()) if len(g) >= 300 else np.nan).dropna()


def signature_objective_routing(sim, obs, weights=None, bias_months=ROUTING_BIAS_MONTHS):
    w = dict(DEFAULT_WEIGHTS_ROUTING); w.update(weights or {})
    j = pd.concat([sim.rename("s"), obs.rename("o")], axis=1).dropna()
    if len(j) < 365: return S.SignatureResult(objective=1e6, parts={"error": np.nan})
    s, o = j.s.clip(lower=Q_FLOOR), j.o.clip(lower=Q_FLOOR)
    ps, po = _per_wy_peak(s), _per_wy_peak(o); c = ps.index.intersection(po.index)
    dt = np.array([(ps.loc[y, "date"] - po.loc[y, "date"]).days for y in c], dtype=float)
    pv = (ps.loc[c, "peak"].astype(float)/po.loc[c, "peak"].astype(float) - 1.0).abs()
    dl = (ps.loc[c, "d25"].astype(float) - po.loc[c, "d25"].astype(float)).abs().dropna()
    cs, co = _com_doy(s), _com_doy(o); cc = cs.index.intersection(co.index)
    m = s.index.month.isin(bias_months); sm, om = s[m].groupby(s[m].index.month).mean(), o[m].groupby(o[m].index.month).mean()
    parts = {"peak_timing": float(np.clip(np.abs(dt).mean()/7.0, 0, 3)), "peak_volume": float(np.clip(pv.mean()/0.15, 0, 3)),
             "falling_limb": float(np.clip(dl.mean()/7.0, 0, 3)) if len(dl) else 3.0,
             "com_timing": float(np.clip((cs[cc] - co[cc]).abs().mean()/7.0, 0, 3)),
             "monthly_bias": float(np.clip((sm - om).abs().sum()/om.sum()/0.10, 0, 3))}
    obj = sum(w[k]*parts[k] for k in parts)/sum(w.values())
    kge = 1 - np.sqrt((np.corrcoef(s, o)[0, 1] - 1)**2 + (s.std()/o.std() - 1)**2 + (s.mean()/o.mean() - 1)**2)
    diag = {"peak_timing_bias_d": float(np.median(dt)), "peak_volume_bias_pct": float(100*np.median(ps.loc[c, "peak"].astype(float)/po.loc[c, "peak"].astype(float) - 1)),
            "d25_sim_d": float(ps.loc[c, "d25"].astype(float).median()), "d25_obs_d": float(po.loc[c, "d25"].astype(float).median()),
            "tau_post_peak_sim_d": float(ps.loc[c, "tau"].astype(float).median()), "tau_post_peak_obs_d": float(po.loc[c, "tau"].astype(float).median()),
            "com_bias_d": float((cs[cc] - co[cc]).median()), "total_vol_bias_pct": float(100*(s.sum()/o.sum() - 1)),
            "KGE": float(kge), "NSE": float(1 - ((s - o)**2).sum()/((o - o.mean())**2).sum()),
            "resid_lag1_sim": lowflow_peak_residual_lag1(s), "resid_lag1_obs": lowflow_peak_residual_lag1(o)}
    return S.SignatureResult(objective=float(obj), parts=parts, diagnostics=diag)
