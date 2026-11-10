"""HOLD-OUT VERSION (WY/years 2011 and 2017 excluded, the two largest snow-area/wet years). Rank correlation of Tuolumne gauge low flows with the area of perennial (end-of-season) snow.
Snow area: /scratch/.../geospatial-data/snowpatch_fsca/eos_final.csv (Landsat, min over Aug 1-Oct 10 of basin snow area, km2): A_shadow = PRIMARY, A_strict = lower bound, A_aso = ASO-anchored.
Low-flow metrics per gauge/year: Sep volume, Sep-Oct volume, Aug-Nov minimum 7-day flow. Spearman rho raw and PARTIAL (controlling for the water-year peak 7-d flow of the HH record; snow area and low flow are both wet-year driven).
-> ess-project/figures/draft/tuo_lowflow_vs_snowpatch_area_holdout.png (LBM + LABvTB scatter), tuo_lowflow_vs_snowpatch_holdout_summary.png/.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist
base = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/tuo_augnov_lag1_acf.py"; src = open(base).read(); ns = {}; exec(compile(src[:src.index('L = pd.DataFrame({k: low7')], "base", "exec"), ns)
G, names, low7, D = ns["G"], ns["names"], ns["low7"], ns["D"]
def window_vol(q, months, need):
    q = q.asfreq("D"); r = {}
    for y in sorted(set(q.index.year)):
        w = (q.index.year == y) & q.index.month.isin(months); z = q[w].dropna()
        if len(z) >= need: r[y] = z.mean()*w.sum()*86400/1e6
    return pd.Series(r)
def peak7(q):
    q = q.asfreq("D"); r7 = q.rolling(7, min_periods=6).mean(); out = {}
    for y in sorted(set(q.index.year)):
        if q[(q.index.year == y) & q.index.month.isin([4, 5, 6, 7])].notna().sum() >= 100: out[y] = r7[(r7.index >= pd.Timestamp(y-1, 10, 1)) & (r7.index <= pd.Timestamp(y, 9, 30))].max()
    return pd.Series(out)
PEAK = peak7(G["HH unimpaired"])
EXCL = [2011, 2017]
eos = pd.read_csv("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/eos_final.csv").set_index("year")
TIER = {"A_shadow": "shadow-inclusive (primary)", "A_strict": "strict (lower bound)", "A_aso": "ASO-anchored"}
MET = {"Sep volume": lambda q: window_vol(q, [9], 25), "Sep-Oct volume": lambda q: window_vol(q, [9, 10], 55), "Aug-Nov min 7-d flow": low7}
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); n = len(x); f = lambda a, b: np.corrcoef(a, b)[0, 1]
    r = (f(rx, ry)-f(rx, rz)*f(ry, rz))/np.sqrt((1-f(rx, rz)**2)*(1-f(ry, rz)**2)); return r, 2*tdist.sf(abs(r*np.sqrt((n-3)/(1-r**2))), n-3)
rows = []; DATA = {}
for g in names:
    for mn, fn in MET.items():
        m = fn(G[g]); DATA[(g, mn)] = m
        for tier in TIER:
            d = pd.concat([m.rename("q"), eos[tier].rename("a"), PEAK.rename("pk")], axis=1).dropna().drop(index=EXCL, errors="ignore"); n = len(d)
            if n < 6: continue
            r, p = spearmanr(d.a, d.q); rp, pp = partial(d.a.values, d.q.values, d.pk.values) if n > 5 else (np.nan, np.nan)
            rows.append(dict(gauge=g, metric=mn, tier=tier, n=n, rho=r, p=p, rho_partial_peak=rp, p_partial=pp))
S = pd.DataFrame(rows); S.to_csv(D+"tuo_lowflow_vs_snowpatch_holdout_summary.csv", index=False); pd.set_option("display.width", 220)
print(S[S.tier == "A_shadow"].round(3).to_string(index=False)); print("\nstrict / aso, raw rho:"); print(S[S.tier != "A_shadow"].pivot_table(index=["gauge", "metric"], columns="tier", values="rho").round(2))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
gs = ["LBM", "LABvTB"]; ms = list(MET); fig, ax = plt.subplots(len(gs), len(ms), figsize=(4.3*len(ms), 3.9*len(gs)))
for i, g in enumerate(gs):
    for j, mn in enumerate(ms):
        a = ax[i, j]; d = pd.concat([DATA[(g, mn)].rename("q"), eos["A_shadow"].rename("a"), PEAK.rename("pk")], axis=1).dropna().drop(index=EXCL, errors="ignore"); r = S[(S.gauge == g) & (S.metric == mn) & (S.tier == "A_shadow")]
        a.scatter(d.a, d.q.clip(lower=1e-5 if "min" in mn else 1e-3), c=np.log10(d.pk), cmap="viridis", s=38, edgecolor="white", lw=.6)
        for y, row in d.iterrows(): a.annotate(str(y)[2:], (row.a, max(row.q, 1e-5 if "min" in mn else 1e-3)), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#52514e")
        a.set_yscale("log"); a.grid(color="#e3e2de", lw=.5)
        if len(r): rr = r.iloc[0]; a.set_title(f"{g} — {mn}\nρ={rr.rho:+.2f} (p={rr.p:.2f}); partial|peak ρ={rr.rho_partial_peak:+.2f} (p={rr.p_partial:.2f}); n={int(rr.n)}", fontsize=8.5, loc="left")
        a.set_xlabel("end-of-season snow area, km² (Landsat, shadow-inclusive)"); a.set_ylabel(("10⁶ m³" if "volume" in mn else "m³/s")+" (log)")
fig.suptitle("Gauge low flows vs perennial-snow area, 2011 and 2017 held out (points coloured by log10 water-year peak flow)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_lowflow_vs_snowpatch_area_holdout.png", dpi=140, bbox_inches="tight")
H = S[S.tier == "A_shadow"]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, (col, pc, ttl) in enumerate([("rho", "p", "Spearman ρ, raw"), ("rho_partial_peak", "p_partial", "partial ρ controlling for peak flow")]):
    M = H.pivot(index="gauge", columns="metric", values=col).reindex(index=names, columns=ms); P = H.pivot(index="gauge", columns="metric", values=pc).reindex(index=names, columns=ms); N = H.pivot(index="gauge", columns="metric", values="n").reindex(index=names, columns=ms)
    im = ax[k].imshow(M.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto"); ax[k].set_xticks(range(len(ms))); ax[k].set_xticklabels(ms, fontsize=8); ax[k].set_yticks(range(len(names))); ax[k].set_yticklabels(names); ax[k].set_title(ttl, fontsize=10, loc="left")
    for a_ in range(len(names)):
        for b_ in range(len(ms)):
            v = M.values[a_, b_]
            if not np.isnan(v): ax[k].text(b_, a_, f"{v:+.2f}{'*' if P.values[a_, b_] < 0.05 else ''}\nn={int(N.values[a_, b_])}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else "#222")
fig.colorbar(im, ax=ax, fraction=0.025); fig.suptitle("Rank correlation of low flow with end-of-season snow area, 2011 + 2017 held out (* p<0.05)", y=1.02); fig.savefig(D+"tuo_lowflow_vs_snowpatch_holdout_summary.png", dpi=140, bbox_inches="tight")
