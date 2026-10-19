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
