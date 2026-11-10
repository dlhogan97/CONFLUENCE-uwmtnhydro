"""Does the decline of perennial snow account for the decline of low flow (after controlling for precipitation)?  Mediation on the long Hetch Hetchy record (1985-2024, n~39), clustered resampling of years.
  total  :  ln Q = a0 + b_P ln P + c  * decade                      (c  = precip-adjusted trend)
  path a :  ln A = d0 + d_P ln P + a  * decade                      (A = end-of-season snow area; a = precip-adjusted snow-area trend)
  direct :  ln Q = e0 + e_P ln P + e_A ln A + c' * decade           (c' = trend not explained by snow area; b = e_A)
  indirect = c - c' (= a*b);  proportion mediated = (c - c')/c.  Tiers of snow area: strict / shadow-inclusive / ASO-anchored. Also with WY temperature as an extra control.
Caveats: A is measured with error (attenuates b -> understates mediation); A, P, Q are collinear; the HH record is a reconstruction.  -> figures/draft/lowflow_trend_mediation.csv/.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"; OBS = U+"modeling/calibration/observations/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"
src = open(U+"utils/tuo_baseflow_trend_precip.py").read(); i = src.index("rows = []; keep = {}"); ns = {}; exec(compile(src[:i], "b", "exec"), ns); metrics, hh, P, tair = ns["metrics"], ns["hh"], ns["P"], ns["tair"]
eos = pd.read_csv(FS+"eos_final.csv").set_index("year"); M = metrics(hh); rng = np.random.default_rng(3)
def ols(y, cols): X = np.column_stack([np.ones(len(y))]+cols); return np.linalg.lstsq(X, y, rcond=None)[0]
def med(df, ctrl):
    t = (df.index.values-2000)/10; y = np.log(df.Q.values); a = np.log(df.A.values); cp = [np.log(df.P.values)]+([df.Tm.values] if ctrl == "P+T" else [])
    c = ols(y, cp+[t])[-1]; ap = ols(a, cp+[t])[-1]; bb = ols(y, cp+[a, t]); cprime = bb[-1]; b = bb[-2]; return dict(total=c, a=ap, b=b, direct=cprime, indirect=c-cprime, prop=(c-cprime)/c if abs(c) > 1e-9 else np.nan)
rows = []; pd.set_option("display.width", 220)
for metric in ["WY min 7-d", "WY Q10", "Aug-Nov min 7-d", "Sep mean"]:
    for tier in ["A_strict", "A_shadow", "A_aso"]:
        for ctrl in ["P", "P+T"]:
            df = pd.concat([M[metric].rename("Q"), eos[tier].clip(lower=0.02).rename("A"), P.rename("P"), tair.rename("Tm")], axis=1).dropna(); df = df[(df.index >= 1985) & (df.index <= 2024)]
            base = med(df, ctrl); boots = []
            for _ in range(2000):
                s = df.iloc[rng.integers(0, len(df), len(df))]
                if s.index.nunique() < 20: continue
                try: boots.append(med(s, ctrl))
                except Exception: pass
            B = pd.DataFrame(boots); lo = lambda k: np.nanpercentile(B[k], 5); hi = lambda k: np.nanpercentile(B[k], 95)
            rows.append(dict(metric=metric, snow_tier=tier, control=ctrl, n=len(df), total_pct_decade=100*(np.exp(base["total"])-1), direct_pct_decade=100*(np.exp(base["direct"])-1), snow_trend_pct_decade=100*(np.exp(base["a"])-1), elasticity_b=base["b"], b_lo=lo("b"), b_hi=hi("b"), prop_mediated=base["prop"], prop_lo=np.nanpercentile(B.prop.clip(-2, 3), 5), prop_hi=np.nanpercentile(B.prop.clip(-2, 3), 95), total_lo=100*(np.exp(lo("total"))-1), total_hi=100*(np.exp(hi("total"))-1)))
R = pd.DataFrame(rows); R.to_csv(D+"lowflow_trend_mediation.csv", index=False); print(R[R.control == "P"].round(2).to_string(index=False)); print("\nwith WY temperature as extra control, shadow-inclusive tier:"); print(R[(R.control == "P+T") & (R.snow_tier == "A_shadow")].round(2).to_string(index=False))
