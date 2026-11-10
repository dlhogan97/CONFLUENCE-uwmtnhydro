"""August-only (images Aug 1 - Sep 5) Lyell snow area vs flow between images, from tuo_lyell_snow_flow_table.csv.
(1) within-year slope dlog q / dlog SCA (years demeaned) and within-year slope controlling for day of year; cluster (year) bootstrap CIs.
(2) control: Dana Fork flow (no perennial snow) against the same Lyell snow area, same model -> how much is shared seasonal recession.
(3) across years: mean August flow and Aug 7-d min vs August-mean Lyell snow area (Spearman).
-> figures/draft/tuo_lyell_august.png, tuo_lyell_august_stats.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
SC = "/scratch/dlhogan/ess-project-data/"; R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"
A = pd.read_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv").set_index("id").area_km2
n = pd.read_csv(SC+"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date").loc[:"2021-08-12"]
Q = {"LBM": n.Estimated_Discharge_LBM/A["LBM"]*86.4, "LABvTB": n.Estimated_Discharge_LABvTB/A["LABvTB"]*86.4, "Dana": n.Estimated_Discharge_DFkBC/A["DFkBC"]*86.4}
T = pd.read_csv(D+"tuo_lyell_snow_flow_table.csv", parse_dates=["date", "prev"])
T = T[(T.date >= pd.to_datetime(T.year.astype(str)+"-08-01")) & (T.date <= pd.to_datetime(T.year.astype(str)+"-09-05")) & (T.prev >= pd.to_datetime(T.year.astype(str)+"-07-22"))].copy()
def between(q, d0, d1): w = q.loc[d0+pd.Timedelta(days=1):d1]; return w.mean() if len(w) and w.notna().mean() >= .8 else np.nan
T["q_Dana"] = [between(Q["Dana"], r.prev, r.date) for r in T.itertuples()]; EPS = 0.02
T["ls_LBM"] = np.log10(T.snow_LBM+EPS); T["ls_LABvTB"] = np.log10(T.snow_LABvTB+EPS); T["gap_mid"] = (T.prev+(T.date-T.prev)/2).dt.dayofyear
print("Aug images used:", len(T), "years:", T.year.nunique()); T.to_csv(D+"tuo_lyell_august_table.csv", index=False)
def slopes(d, ycol, xcol, ctrl):
    d = d[[ycol, xcol, "year", "gap_mid"]].dropna(); d = d[d[ycol] > 0]; d = d[d.groupby("year").year.transform("size") >= 2]
    if len(d) < 8: return np.nan, np.nan, len(d)
    y = np.log10(d[ycol]); y = y-y.groupby(d.year).transform("mean"); x = d[xcol]-d.groupby("year")[xcol].transform("mean"); t = d.gap_mid-d.groupby("year").gap_mid.transform("mean")
    if not ctrl: return (x*y).sum()/(x*x).sum(), np.nan, len(d)
    Xm = np.c_[x, t]; b = np.linalg.lstsq(Xm, y, rcond=None)[0]; return b[0], b[1], len(d)
def boot(d, ycol, xcol, ctrl, k=2000, seed=1):
    rng = np.random.default_rng(seed); ys = d.year.unique(); out = []
    for _ in range(k):
        s = rng.choice(ys, len(ys)); dd = pd.concat([d[d.year == y].assign(year=i) for i, y in enumerate(s)]); out.append(slopes(dd, ycol, xcol, ctrl)[0])
    return np.nanpercentile(out, [2.5, 97.5])
rows = []
for lab, ycol, xcol in [("Lyell LBM flow vs LBM snow", "q_LBM", "ls_LBM"), ("Lyell LABvTB flow vs LABvTB snow", "q_LABvTB", "ls_LABvTB"), ("Dana flow vs LBM snow (control)", "q_Dana", "ls_LBM"), ("Dana flow vs LABvTB snow (control)", "q_Dana", "ls_LABvTB")]:
    for ctrl in (False, True):
        b, bt, nn = slopes(T, ycol, xcol, ctrl); d = T[[ycol, xcol, "year", "gap_mid"]].dropna(); d = d[d[ycol] > 0]; ci = boot(d, ycol, xcol, ctrl)
        rows.append(dict(model=lab, controls_doy=ctrl, n=nn, years=d.year.nunique(), slope=b, ci_lo=ci[0], ci_hi=ci[1], doy_coef_per_day=bt))
    r, _ = spearmanr(*T[[ycol, xcol]].dropna().values.T); rows.append(dict(model=lab+" [pooled Spearman]", n=T[[ycol, xcol]].dropna().shape[0], slope=r))
S = pd.DataFrame(rows); pd.set_option("display.width", 200); print(S.round(2).to_string(index=False))
# across years
Y = []
for y in sorted(T.year.unique()):
    e = T[T.year == y]; a = Q["LBM"].loc[f"{y}-08-01":f"{y}-08-31"]; ad = Q["Dana"].loc[f"{y}-08-01":f"{y}-08-31"]
    if a.notna().sum() >= 25: Y.append(dict(year=y, snow_LBM=e.snow_LBM.mean(), snow_LABvTB=e.snow_LABvTB.mean(), n_img=len(e), q_aug_LBM=a.mean(), q7min_LBM=a.rolling(7).mean().min(), q_aug_Dana=ad.mean() if ad.notna().sum() >= 25 else np.nan))
Y = pd.DataFrame(Y); Y.to_csv(D+"tuo_lyell_august_years.csv", index=False); print(Y.round(2).to_string(index=False))
def sp_(a, b): d = Y[[a, b]].dropna(); return (spearmanr(d[a], d[b])[0], len(d)) if len(d) > 4 else (np.nan, len(d))
for a, b in [("q_aug_LBM", "snow_LBM"), ("q7min_LBM", "snow_LBM"), ("q_aug_Dana", "snow_LBM")]:
    r, nn = sp_(a, b); print(f"across years  {a} vs {b}: rho={r:+.2f} n={nn}"); rows.append(dict(model=f"across-years {a} vs {b}", n=nn, slope=r))
    d = Y[Y.year.isin([2011, 2017]) == False]; dd = d[[a, b]].dropna(); print(f"   excl 2011/2017: rho={spearmanr(dd[a], dd[b])[0]:+.2f} n={len(dd)}")
S = pd.DataFrame(rows); S.to_csv(D+"tuo_lyell_august_stats.csv", index=False)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (yc, xc, ttl) in zip(ax[:2], [("q_LBM", "snow_LBM", "Lyell (LBM)"), ("q_Dana", "snow_LBM", "Dana flow vs LBM snow (control)")]):
    d = T[[yc, xc, "year", "gap_mid"]].dropna(); sc = a.scatter(d[xc]+EPS, d[yc], c=d.gap_mid, cmap="viridis", s=24); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("LBM snow area + eps (km²)"); a.set_ylabel("mean flow since previous image (mm/d)"); a.set_title(ttl, loc="left"); a.grid(color="#e3e2de", lw=.5, which="both")
    for y, e in d.groupby("year"): a.plot(e[xc]+EPS, e[yc], "-", color="#ccc", lw=.6, zorder=0)
    plt.colorbar(sc, ax=a, label="interval midpoint (doy)")
a = ax[2]; a.scatter(Y.snow_LBM+EPS, Y.q_aug_LBM, color="#2a78d6", label="Lyell Aug mean flow"); a.scatter(Y.snow_LBM+EPS, Y.q7min_LBM, color="#d55e00", marker="s", label="Lyell Aug min 7-d")
for _, r in Y.iterrows(): a.annotate(str(int(r.year))[2:], (r.snow_LBM+EPS, r.q_aug_LBM), fontsize=7, xytext=(3, 3), textcoords="offset points")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("mean LBM snow area in Aug images (km²)"); a.set_ylabel("mm/d"); a.legend(fontsize=7); a.set_title("across years", loc="left"); a.grid(color="#e3e2de", lw=.5, which="both")
fig.tight_layout(); fig.savefig(D+"tuo_lyell_august.png", dpi=120, bbox_inches="tight")
