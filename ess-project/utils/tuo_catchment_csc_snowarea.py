"""Late-season flow metrics vs catchment-specific Landsat end-of-season snow area (eos_by_catchment.csv; sensor rule NDSI 0.08 OLI / 0.11 TM-ETM+, NIR>0.11).
Pairs (reference = Dana Fork DFkBC; Dana's snowmelt-wave end defines the transition day, as in tuo_dmc_daily_csc.py):
  LBM    (Lyell below Maclure, 15 km2)   CSC = q_LBM - s0*q_Dana, snow area = LBM catchment
  LABvTB (Lyell above Twin Bridges, 107 km2; contains LBM, not Dana)  CSC = q_LABvTB - s0*q_Dana, snow area = LABvTB catchment
  DFkBC  (Dana, own late-season flow, no reference): flow after its own wave end, snow area = Dana catchment
Metrics: season CSC (mm, May1-Oct31), post-transition volume (mm), mean post flow. Predictors: area same year (end of that summer), prior year (start of the year), mean of both, WY precip same/prior. Spearman + permutation p; 2011 & 2017 optional exclusion.
-> figures/draft/tuo_catchment_csc_snowarea.png, .csv (data), _stats.csv"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
SC = "/scratch/dlhogan/ess-project-data/"; R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"
CLS = sys.argv[1] if len(sys.argv) > 1 else "sensor"; TIER = sys.argv[2] if len(sys.argv) > 2 else "shadow"
A = pd.read_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv").set_index("id").area_km2
n = pd.read_csv(SC+"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date").loc[:"2021-08-12"]
Q = {"LBM": n.Estimated_Discharge_LBM/A["LBM"]*86.4, "LABvTB": n.Estimated_Discharge_LABvTB/A["LABvTB"]*86.4, "DFkBC": n.Estimated_Discharge_DFkBC/A["DFkBC"]*86.4}
E = pd.read_csv(SC+"geospatial-data/snowpatch_fsca/eos_by_catchment.csv"); E = E[(E.cls == CLS) & (E.tier == TIER)].pivot(index="year", columns="catch", values="area_km2")
Pm = pd.read_csv(R+"modeling/calibration/observations/prism_tuolumne_monthly_precip.csv", index_col=0, parse_dates=True).precip_mm; wyp = Pm.groupby(Pm.index.year+(Pm.index.month >= 10)).sum()
FR = 0.10; rows = []
for g in ["LBM", "LABvTB", "DFkBC"]:
    for y in range(2005, 2022):
        w = pd.DataFrame({"G": Q[g], "N": Q["DFkBC"]}).loc[f"{y}-05-01":f"{y}-10-31"]; ok = w.notna().all(axis=1)
        if len(w) < 150 or (~ok).sum() > 12 or not ok.iloc[:30].all() or ok.iloc[-45:-14].sum() < 30: continue
        w = w.interpolate(limit=10).dropna(); n7 = w.N.rolling(7, center=True, min_periods=4).mean(); pk = n7.idxmax(); pv = n7.max(); af = n7.loc[pk:]
        if not (af < FR*pv).any(): continue
        tr = af.index[(af < FR*pv).values][0]; s0 = w.G.loc[:pk].sum()/w.N.loc[:pk].sum(); csc = w.G-s0*w.N
        r = dict(gauge=g, year=y, trans_doy=tr.dayofyear, total_mm=w.G.sum(), post_mm=w.G.loc[tr:].sum(), post_mean=w.G.loc[tr:].mean())
        r["csc_season_mm"] = csc.sum() if g != "DFkBC" else np.nan
        r.update(A_same=E[g].get(y, np.nan) if g in E else np.nan, A_prior=E[g].get(y-1, np.nan) if g in E else np.nan, P_same=wyp.get(y, np.nan), P_prior=wyp.get(y-1, np.nan)); rows.append(r)
X = pd.DataFrame(rows); X["A_mean2"] = X[["A_same", "A_prior"]].mean(axis=1); X.to_csv(D+f"tuo_catchment_csc_snowarea_{CLS}_{TIER}.csv", index=False)
pd.set_option("display.width", 220); print(X.round(2).to_string(index=False))
def sp(a, b): ra, rb = rankdata(a)-(len(a)+1)/2., rankdata(b)-(len(b)+1)/2.; return (ra*rb).sum()/np.sqrt((ra*ra).sum()*(rb*rb).sum())
def perm(a, b, k=3000, seed=0): rng = np.random.default_rng(seed); r = sp(a, b); return r, np.mean([abs(sp(a, rng.permutation(b))) >= abs(r) for _ in range(k)])
S = []
for g in ["LBM", "LABvTB", "DFkBC"]:
    for ym in ["csc_season_mm", "post_mm", "post_mean", "total_mm"]:
        for xm in ["A_same", "A_prior", "A_mean2", "P_same", "P_prior"]:
            for ex in (False, True):
                d = X[X.gauge == g][["year", ym, xm]].dropna()
                if ex: d = d[~d.year.isin([2011, 2017])]
                if len(d) >= 5: r, p = perm(d[ym].values, d[xm].values); S.append(dict(gauge=g, y=ym, x=xm, excl_mega=ex, n=len(d), rho=r, p=p))
S = pd.DataFrame(S); S.to_csv(D+f"tuo_catchment_csc_snowarea_stats_{CLS}_{TIER}.csv", index=False)
print(S.pivot_table(index=["gauge", "y", "x"], columns="excl_mega", values=["rho", "p", "n"]).round(2).to_string())
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(3, 3, figsize=(13, 11))
for i, g in enumerate(["LBM", "LABvTB", "DFkBC"]):
    for j, (xm, xl) in enumerate([("A_same", "catchment EOS snow area same year (km²)"), ("A_prior", "prior year (km²)"), ("P_same", "WY precip (mm)")]):
        ym = "post_mm"; a = ax[i, j]; d = X[X.gauge == g][["year", ym, xm]].dropna(); a.plot(d[xm], d[ym], "o", color="#2a78d6")
        for _, r in d.iterrows(): a.annotate(str(int(r.year))[2:], (r[xm], r[ym]), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#d55e00" if r.year in (2011, 2017) else "#52514e")
        rr, pp = perm(d[ym].values, d[xm].values) if len(d) > 4 else (np.nan, np.nan); a.set_title(f"{g}: ρ={rr:+.2f} (p={pp:.2f}, n={len(d)})", loc="left", fontsize=8); a.set_xlabel(xl); a.set_ylabel(f"{g} flow after Dana wave ends (mm)"); a.grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+f"tuo_catchment_csc_snowarea_{CLS}_{TIER}.png", dpi=120, bbox_inches="tight")
