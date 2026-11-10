"""Lyell post-transition flow (tuo_dmc_daily_csc_table.csv, Dana reference) vs Landsat end-of-season snow area (eos_thresh_compare.csv column given as argv[1], default sensor_shadow = NDSI 0.08 OLI / 0.11 TM-ETM+ with NIR>0.11; t40_shadow = old 0.40 record, km2) of the same year (end of THAT summer, after the flow) and of the prior year (the perennial snow present at the start of the year),
and vs WY precip (PRISM-sampled basin, Oct-Sep) same/prior. Spearman, 5000-perm p; n<=9 so exploratory. Also lag-1 pairs of the flow metric. -> figures/draft/tuo_lyell_csc_vs_snowarea.png, tuo_lyell_csc_vs_snowarea.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"
T = pd.read_csv(D+"tuo_dmc_daily_csc_table.csv"); T = T[T.ref == "DFkBC"].set_index("year")
COL = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "sensor_shadow"; SUF = "_"+COL
eos = pd.read_csv("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/eos_thresh_compare.csv").set_index("year"); eos["A_shadow"] = eos[COL]   # t40_shadow reproduces eos_final; sensor_* = NDSI 0.08 (OLI) / 0.11 (TM/ETM+) + NIR>0.11
P = pd.read_csv(R+"modeling/calibration/observations/prism_tuolumne_monthly_precip.csv", index_col=0, parse_dates=True).precip_mm; wyp = P.groupby([(P.index.year+(P.index.month >= 10))]).sum()
X = pd.DataFrame({"post_mm": T.lyell_post_mm, "post_mean": T.lyell_post_mean, "csc_season_mm": T.csc_season_frac*T.lyell_total_mm, "lyell_total_mm": T.lyell_total_mm, "trans_doy": T.trans_doy}, index=T.index)
for y in X.index:
    X.loc[y, "A_same"] = eos.A_shadow.get(y, np.nan); X.loc[y, "A_prior"] = eos.A_shadow.get(y-1, np.nan); X.loc[y, "P_same"] = wyp.get(y, np.nan); X.loc[y, "P_prior"] = wyp.get(y-1, np.nan)
X["A_mean2"] = X[["A_same", "A_prior"]].mean(axis=1)
for m in ["post_mm", "post_mean", "csc_season_mm"]: X[m+"_prior"] = [X[m].get(y-1, np.nan) for y in X.index]
X.to_csv(D+f"tuo_lyell_csc_vs_snowarea{SUF}.csv"); print(X.round(1).to_string())
def sp(a, b):
    ra, rb = rankdata(a)-(len(a)+1)/2., rankdata(b)-(len(b)+1)/2.; return (ra*rb).sum()/np.sqrt((ra*ra).sum()*(rb*rb).sum())
def perm(a, b, nperm=5000, seed=0):
    rng = np.random.default_rng(seed); r = sp(a, b); return r, np.mean([abs(sp(a, rng.permutation(b))) >= abs(r) for _ in range(nperm)])
rows = []
for yv in ["post_mm", "post_mean", "csc_season_mm"]:
    for xv in ["A_same", "A_prior", "A_mean2", "P_same", "P_prior", yv+"_prior"]:
        for ex, lab in [(False, "all"), (True, "excl 2011/2017")]:
            d = X[[yv, xv]].dropna(); d = d.drop([y for y in (2011, 2017) if y in d.index]) if ex else d
            if len(d) >= 5: r, p = perm(d[yv].values, d[xv].values); rows.append(dict(y=yv, x=xv, set=lab, n=len(d), rho=r, p=p))
S = pd.DataFrame(rows); print(S.pivot_table(index=["y", "x"], columns="set", values=["rho", "p", "n"]).round(2).to_string())
S.to_csv(D+f"tuo_lyell_csc_vs_snowarea_stats{SUF}.csv", index=False)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, (xv, xl) in zip(ax.ravel(), [("A_same", "Landsat EOS snow area, same year (km²)"), ("A_prior", "EOS snow area, prior year (km²)"), ("P_same", "WY precip same year (mm)"), ("P_prior", "WY precip prior year (mm)"), ("post_mm_prior", "Lyell post-transition volume, prior year (mm)"), ("A_mean2", "mean EOS area same+prior (km²)")]):
    d = X[["post_mm", xv]].dropna(); a.plot(d[xv], d.post_mm, "o", color="#2a78d6")
    for y, r in d.iterrows(): a.annotate(str(y)[2:], (r[xv], r.post_mm), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#d55e00" if y in (2011, 2017) else "#52514e")
    r, p = perm(d.post_mm.values, d[xv].values) if len(d) > 4 else (np.nan, np.nan); a.set_title(f"ρ={r:+.2f} (p={p:.2f}, n={len(d)})", fontsize=8, loc="left"); a.set_xlabel(xl); a.set_ylabel("Lyell flow after Dana wave ends (mm)"); a.grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+f"tuo_lyell_csc_vs_snowarea{SUF}.png", dpi=125, bbox_inches="tight")
