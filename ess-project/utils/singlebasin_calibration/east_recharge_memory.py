"""
Where does the fast part's NEGATIVE year-to-year low-flow memory come from? (2026-10-27)
All emulator models share the SUMMA routing-control recharge (soil drainage to the buckets, Rch) and surface runoff (SR). This checks the year-to-year
structure of those inputs: per water year, basin-mean recharge and surface runoff by season, residual on the same-year ln Apr-Jul 7-d flow peak
(treatment v2 flow; observed peak as a check), Spearman lag-1; plus how each relates to the treatment's fast-part fall minimum (ablation run).
usage: python east_recharge_memory.py
"""
import json, logging; logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import east_ctl_treat_cal as C, east_ctl_treat_cal_v2 as V

E, st, W = C.E, C.st, C.W
tr = [json.loads(l) for l in open(V.OUTD/"trials_treatment.jsonl")]; p = min([x for x in tr if x["objective"] < 10], key=lambda x: x["objective"])["params"]
Qs, Qd, Sd = C.simulate(p, p["f"]); q_tr = C.flow(Qs, Qd, p); Qs0, Qd0, _ = C.simulate(p, 0.0); q_fast = C.flow(Qs0, Qd0, p)
day = lambda a: pd.Series((a*W).sum(1)*3600.0*1000.0, index=st["tt"]).resample("D").sum()          # mm/day basin mean
R, SR = day(st["Rch"]), day(st["SR"])
def wy(i): return i.year + (i.month >= 10)
def season(x, y, m0, d0, m1, d1, yo=0): return x[f"{y-1+yo}-{m0:02d}-{d0:02d}":f"{y}-{m1:02d}-{d1:02d}"].sum()
yrs = range(1983, 2025)
A = pd.DataFrame({y: dict(R_OctMar=season(R, y, 10, 1, 3, 31), R_AprJul=R[f"{y}-04-01":f"{y}-07-31"].sum(), R_AugNov=R[f"{y}-08-01":f"{y}-11-30"].sum(),
                          R_year=season(R, y, 10, 1, 9, 30), SR_AugNov=SR[f"{y}-08-01":f"{y}-11-30"].sum(), SR_year=season(SR, y, 10, 1, 9, 30),
                          R_centroid_doy=float((np.arange(len(R[f"{y-1}-10-01":f"{y}-09-30"]))*R[f"{y-1}-10-01":f"{y}-09-30"].values).sum()/R[f"{y-1}-10-01":f"{y}-09-30"].sum()))
                  for y in yrs}).T
for nm, q in (("LF_fast", q_fast), ("LF_treatment", q_tr), ("LF_observed", E.obs)):
    lf, _ = E.series(q, 1983, 2024); A[nm] = np.exp(lf)
_, pk_m = E.series(q_tr, 1983, 2024); _, pk_o = E.series(E.obs, 1983, 2024); A["ln_peak_model"], A["ln_peak_obs"] = pk_m, pk_o
def lag1(s): s = s.dropna(); d = pd.concat([s.rename("y"), pd.Series(s.values, index=s.index + 1).rename("x")], axis=1).dropna(); return spearmanr(d.x, d.y)[0]
def resid(col, pk): d = A[[col, pk]].dropna(); y = np.log(d[col].clip(lower=1e-3)) if col != "R_centroid_doy" else d[col]; return y - np.polyval(np.polyfit(d[pk], y, 1), d[pk])
rows = []
for w, (y0, y1) in {"WY1987-2024": (1987, 2024), "WY2009-2024": (2009, 2024)}.items():
    B_ = A.loc[y0:y1]
    for col in ("R_OctMar", "R_AprJul", "R_AugNov", "R_year", "R_centroid_doy", "SR_AugNov", "LF_fast", "LF_treatment", "LF_observed"):
        r = resid(col, "ln_peak_model").loc[y0:y1]
        rows.append(dict(window=w, series=col, raw_lag1=lag1(np.log(B_[col].clip(lower=1e-3)) if col != "R_centroid_doy" else B_[col]), resid_lag1=lag1(r),
                         corr_with_peak=spearmanr(B_[col], B_["ln_peak_model"], nan_policy="omit")[0],
                         corr_resid_with_LF_fast_resid=spearmanr(r, resid("LF_fast", "ln_peak_model").loc[y0:y1], nan_policy="omit")[0]))
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nmean seasonal recharge (mm):", A.loc[1987:2024, ["R_OctMar", "R_AprJul", "R_AugNov", "R_year", "SR_AugNov"]].mean().round(0).to_dict())
A.to_csv(C.X.OUT/"east_recharge_memory.csv")
