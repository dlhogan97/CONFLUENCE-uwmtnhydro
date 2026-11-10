"""Has Tuolumne baseflow declined over time AFTER controlling for water-year precipitation?  (a drawdown of perennial stores would show as a falling low-flow-for-given-precip trend)
Model per gauge/metric:  log10(Q_metric) = a + b*log10(P_WY) + c*(year-2000)/10 ;  c converted to %/decade = 100*(10^c-1).  p from OLS t-test (df=n-3) and a rank partial correlation of the precip-adjusted residual with year.
Metrics: Aug / Sep / Oct mean Q, Sep-Oct mean, Aug-Nov min 7-d, water-year min 7-d, water-year Q10.  Precip control: PRISM-based basin WY precipitation (METSIM forcing, 1982-2025), also TUM gauge (1986-2024); long HH record also with its own WY runoff as the control (1970-2024).
Gauges: USGS 11274790 (WY2007-2025), HH unimpaired reconstruction (1970-2024), nested gauges (QC'd, to 2021-08-12). -> figures/draft/tuo_baseflow_trend_precip.png, tuo_baseflow_trend_precip.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import t as tdist, spearmanr, rankdata
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"; OBS = U+"modeling/calibration/observations/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"
src = open(U+"utils/tuo_augnov_lag1_acf_qc.py").read(); ns = {}; exec(compile(src[:src.index('L = pd.DataFrame({k: low7')], "base", "exec"), ns); G = ns["G"]
usgs = G["USGS11274790"]; hh = G["HH unimpaired"]; names = ["USGS11274790", "HH unimpaired", "T120", "LBM", "LABvTB", "DFkBC"]
pm = pd.read_csv(OBS+"prism_tuolumne_monthly_precip.csv", index_col=0, parse_dates=True)["precip_mm"]; wy = lambda i: i.year+(i.month >= 10); P = pm.groupby(wy(pm.index)).sum(); P = P[pm.groupby(wy(pm.index)).size() == 12]
TUM = pd.read_csv(FS+"tum_wy_precip.csv", comment="#").set_index("water_year")["wyppt_mm"]; tair = pd.read_csv(FS+"basin_wy_forcing.csv").set_index("wy")["Tuolumne_wytair_C"]
def metrics(q):
    q = q.asfreq("D"); r7 = q.rolling(7, center=True, min_periods=6).mean(); out = {}
    for y in range(int(q.index.year.min())+1, int(q.index.year.max())+1):
        w = q[(q.index >= pd.Timestamp(y-1, 10, 1)) & (q.index <= pd.Timestamp(y, 9, 30))]; rec = {}
        mm = lambda ms, need: (q[(q.index.year == y) & q.index.month.isin(ms)].mean() if q[(q.index.year == y) & q.index.month.isin(ms)].notna().sum() >= need else np.nan)
        rec["Aug mean"] = mm([8], 28); rec["Sep mean"] = mm([9], 27); rec["Oct mean"] = mm([10], 28); rec["Sep-Oct mean"] = mm([9, 10], 55)
        a = r7[(r7.index.year == y) & (r7.index.month >= 8) & (r7.index.month <= 11)]; qa = q[(q.index.year == y) & (q.index.month >= 8) & (q.index.month <= 11)]; rec["Aug-Nov min 7-d"] = a.min() if qa.notna().sum() >= 110 else np.nan
        if w.notna().sum() >= 350: rec["WY min 7-d"] = r7[w.index].min(); rec["WY Q10"] = w.quantile(0.10)
        out[y] = rec
    return pd.DataFrame(out).T
def fit(y, controls, yrs):
    d = pd.concat([np.log10(y.clip(lower=1e-3)).rename("y")]+[np.log10(c).rename(f"c{i}") if lg else c.rename(f"c{i}") for i, (c, lg) in enumerate(controls)], axis=1).dropna(); d = d[d.index.isin(yrs)]; n = len(d)
    if n < 10: return None
    X = np.column_stack([np.ones(n)]+[d[f"c{i}"].values for i in range(len(controls))]+[(d.index.values-2000)/10]); b, *_ = np.linalg.lstsq(X, d.y.values, rcond=None); res = d.y.values-X@b; s2 = res@res/(n-X.shape[1]); cov = s2*np.linalg.inv(X.T@X); se = np.sqrt(cov[-1, -1]); tt = b[-1]/se; p = 2*tdist.sf(abs(tt), n-X.shape[1])
    Xc = X[:, :-1]; ry = d.y.values-Xc@np.linalg.lstsq(Xc, d.y.values, rcond=None)[0]; rs, ps = spearmanr(ry, d.index.values); tcrit = tdist.ppf(0.975, n-X.shape[1])
    return dict(n=n, pct_per_decade=100*(10**b[-1]-1), lo=100*(10**(b[-1]-tcrit*se)-1), hi=100*(10**(b[-1]+tcrit*se)-1), p=p, b_precip=b[1], rho_resid_year=rs, p_rank=ps, resid=pd.Series(ry, index=d.index))
rows = []; keep = {}
SC = {"PRISM WY precip": ([(P, True)], range(1982, 2026)), "PRISM WY precip + WY temp": ([(P, True), (tair, False)], range(1982, 2026)), "TUM obs WY precip": ([(TUM, True)], range(1986, 2026))}
for g in names:
    M = metrics(G[g] if g != "HH unimpaired" else hh)
    ctr = dict(SC)
    if g == "HH unimpaired":
        wyvol = hh.groupby(wy(hh.index)).mean(); ctr["HH own WY runoff (1970-2024)"] = ([(wyvol, True)], range(1971, 2025))
    for cn, (cs, yr) in ctr.items():
        for m in M.columns:
            r = fit(M[m], cs, yr)
            if r: rows.append(dict(gauge=g, control=cn, metric=m, **{k: v for k, v in r.items() if k != "resid"})); keep[(g, cn, m)] = r["resid"]
R = pd.DataFrame(rows); R.to_csv(D+"tuo_baseflow_trend_precip.csv", index=False); pd.set_option("display.width", 220)
for cn in ["PRISM WY precip", "HH own WY runoff (1970-2024)"]:
    x = R[R.control == cn]; print("\n==", cn); print(x.pivot(index="gauge", columns="metric", values="pct_per_decade").round(0).reindex([g for g in names if g in set(x.gauge)]).to_string()); print("p:"); print(x.pivot(index="gauge", columns="metric", values="p").round(2).reindex([g for g in names if g in set(x.gauge)]).to_string()); print("n:", x.groupby("gauge").n.max().to_dict())
x = R[R.control == "PRISM WY precip + WY temp"]; print("\n== with WY temp too"); print(x.pivot(index="gauge", columns="metric", values="pct_per_decade").round(0).to_string()); print(x.pivot(index="gauge", columns="metric", values="p").round(2).to_string())
# figure: precip-adjusted residuals vs year for USGS and HH, 4 metrics + forest plot
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig = plt.figure(figsize=(17, 8)); mets = ["Aug mean", "Sep mean", "Aug-Nov min 7-d", "WY min 7-d"]
for i, g in enumerate(["USGS11274790", "HH unimpaired"]):
    for j, m in enumerate(mets):
        a = fig.add_subplot(2, 5, i*5+j+1); k = (g, "PRISM WY precip", m)
        if k not in keep: continue
        r = keep[k]; a.plot(r.index, r.values, "o", color="#2a78d6", ms=4); b, c = np.polyfit(r.index, r.values, 1); a.plot(r.index, c+b*r.index, color="#d55e00"); a.axhline(0, color="#999", lw=.7)
        q = R[(R.gauge == g) & (R.control == "PRISM WY precip") & (R.metric == m)].iloc[0]; a.set_title(f"{g} — {m}\n{q.pct_per_decade:+.0f}%/decade [{q.lo:+.0f},{q.hi:+.0f}], p={q.p:.2f}, n={int(q.n)}", fontsize=8.5, loc="left"); a.set_xlabel("year"); a.set_ylabel("residual log10 Q after WY precip") if j == 0 else None
a = fig.add_subplot(1, 5, 5) if False else None
ax = fig.add_axes([0.80, 0.08, 0.18, 0.84]); yy = 0; lab = []
for g in names:
    x = R[(R.gauge == g) & (R.control == "PRISM WY precip")]
    for _, r in x.iterrows():
        ax.errorbar(r.pct_per_decade, yy, xerr=[[r.pct_per_decade-r.lo], [r.hi-r.pct_per_decade]], fmt="o", color="#d55e00" if r.p < 0.05 else "#2a78d6", ms=3.5, lw=.9); lab.append(f"{g[:7]} {r.metric}"); yy += 1
ax.axvline(0, color="#52514e", lw=.8); ax.set_yticks(range(yy)); ax.set_yticklabels(lab, fontsize=5.5); ax.set_xlim(-250, 250); ax.set_xlabel("%/decade (95% CI; orange p<0.05)"); ax.invert_yaxis()
fig.suptitle("Precipitation-adjusted baseflow trends (PRISM-basin WY precip as control)", y=0.995); fig.subplots_adjust(left=0.05, right=0.78, hspace=0.45, wspace=0.35); fig.savefig(D+"tuo_baseflow_trend_precip.png", dpi=140, bbox_inches="tight")
