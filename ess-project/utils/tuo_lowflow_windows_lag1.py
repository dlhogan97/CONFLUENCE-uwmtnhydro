"""Tuolumne observed low-flow volume in Sep-Oct and Oct-only windows: cross-gauge correlation plots (as tuo_sept_volume_xcorr.py) + lag-1 persistence after normalising for the
spring peak flow. Peak = max 7-day mean daily Q of the HH unimpaired record (used for ALL gauges; matches the USGS gauge peak at rho=0.99) in the water year ending in the same calendar year (Oct y-1 .. Sep y).
Lag-1 variants per gauge/window: raw V_y vs V_{y-1}; ratio-normalised (V/Qpeak) lag-1; partial Spearman of V_y on V_{y-1} controlling for Qpeak_y.
-> ess-project/figures/draft/tuo_{sepoct,oct}_volume_xcorr.png (+ scatter matrices), tuo_lowflow_lag1_summary.png/.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist
import importlib.util, sys
D = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
SC = "/scratch/dlhogan/ess-project-data/"; SF = SC+"streamflow-data/TuolumneRiver/"
def hh_unimpaired():
    d = pd.read_csv(SF+"TuolumneRiver_HH_unimpaired_19702024.csv", encoding="utf-8-sig").set_index("DATE"); d = d[d.index.astype(str).str.fullmatch(r"\d+/\d+")].apply(pd.to_numeric, errors="coerce"); out = []
    for c in d.columns:
        wy = int(c.split()[1])
        for md, v in d[c].items():
            m, dd = map(int, md.split("/")); y = wy-1 if m >= 10 else wy
            try: out.append((pd.Timestamp(y, m, dd), v*0.0283168))
            except ValueError: pass
    return pd.Series(dict(out)).sort_index()
nest = pd.read_csv(SF+"TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date"); nest.columns = [c.replace("Estimated_Discharge_", "") for c in nest.columns]
G = {k: nest[k].astype(float) for k in nest}; G["T120"] = G["T120"].where(G["T120"].index < "2016-10-01", G["T120"]/35.3146)
G["USGS11274790"] = pd.read_csv(SC+"domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_USGS11274790_dailyQ_WY2007_2025.csv", parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
G["HH unimpaired"] = hh_unimpaired(); names = list(G)
WIN = {"Sep-Oct": ([9, 10], 55), "Oct": ([10], 28), "Sep": ([9], 25)}
def window_vol(q, months, need):
    q = q.asfreq("D"); r = {}
    for y in sorted(set(q.index.year)):
        z = q[(q.index.year == y) & q.index.month.isin(months)].dropna()
        if len(z) >= need: r[y] = z.mean()*len(q[(q.index.year == y) & q.index.month.isin(months)])*86400/1e6
    return pd.Series(r)
def spring_peak(q):
    q = q.asfreq("D"); r7 = q.rolling(7, min_periods=6).mean(); r = {}
    for y in sorted(set(q.index.year)):
        aj = q[(q.index.year == y) & q.index.month.isin([4, 5, 6, 7])]
        if aj.notna().sum() >= 100: r[y] = r7[(r7.index >= pd.Timestamp(y-1, 10, 1)) & (r7.index <= pd.Timestamp(y, 9, 30))].max()
    return pd.Series(r)
_pk = spring_peak(G["HH unimpaired"]); PK = pd.DataFrame({k: _pk for k in G})   # common basin wetness index: headwater gauges lack many spring days; HH peak vs USGS peak rho=0.99 (n=16)
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); n = len(x); f = lambda a, b: np.corrcoef(a, b)[0, 1]
    r = (f(rx, ry)-f(rx, rz)*f(ry, rz))/np.sqrt((1-f(rx, rz)**2)*(1-f(ry, rz)**2)); tt = r*np.sqrt((n-3)/(1-r**2)); return r, 2*tdist.sf(abs(tt), n-3)
def xcorr_figs(T, tag, label):
    n = len(names); R = pd.DataFrame(np.nan, names, names); N = R.copy(); P = R.copy()
    for a in names:
        for b in names:
            if a == b: N.loc[a, b] = T[a].notna().sum(); R.loc[a, b] = 1.0; P.loc[a, b] = 0.0; continue
            j = T[[a, b]].dropna(); N.loc[a, b] = len(j)
            if len(j) >= 5: r, p = spearmanr(j[a], j[b]); R.loc[a, b] = r; P.loc[a, b] = p
    iu = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]; ns = sum(P.loc[a, b] < 0.05 for a, b in iu); print(f"[{label}] years/gauge", T.notna().sum().to_dict(), f"| significant pairs {ns}/{len(iu)}"); print(R.round(2))
    fig = plt.figure(figsize=(15, 6.6)); ax = fig.add_axes([0.04, 0.12, 0.30, 0.78]); im = ax.imshow(R.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_yticklabels(names)
    for i in range(n):
        for k in range(n):
            if not np.isnan(R.iloc[i, k]): ax.text(k, i, f"{R.iloc[i,k]:+.2f}{'*' if P.iloc[i,k] < 0.05 and i != k else ''}\nn={int(N.iloc[i,k])}", ha="center", va="center", fontsize=7.5, color="white" if abs(R.iloc[i, k]) > 0.6 else "#222")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"Spearman correlation of {label} flow volume\n(* p<0.05; pairwise over overlapping years)", fontsize=10, loc="left"); fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    axs = fig.add_axes([0.42, 0.12, 0.56, 0.78]); cols = plt.cm.tab10(np.arange(n))
    for k, c in enumerate(names):
        z = T[c].dropna(); axs.plot(z.index, z.values.clip(1e-4), "-o", ms=3.2, lw=1.1, color=cols[k], label=f"{c} (n={len(z)})")
    axs.set_yscale("log"); axs.set_xlim(1969, 2026); axs.set_ylabel(f"{label} flow volume (10⁶ m³, log)"); axs.grid(axis="y", color="#e3e2de", lw=.6); axs.legend(ncol=2, frameon=False, fontsize=8, loc="lower right")
    axs.set_title(f"{label} flow volume by year (mean Q × days in window); T120 divided by 35.3 after WY2016", fontsize=10, loc="left"); fig.savefig(D+f"tuo_{tag}_volume_xcorr.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig2, A = plt.subplots(n, n, figsize=(13, 12))
    for i, a in enumerate(names):
        for k, b in enumerate(names):
            ax = A[i, k]; ax.set_xticks([]); ax.set_yticks([])
            if i == k: ax.text(.5, .5, a, ha="center", va="center", fontsize=10, transform=ax.transAxes); ax.set_axis_off(); continue
            j = T[[b, a]].dropna()
            if i < k: ax.set_axis_off(); ax.text(.5, .5, f"ρ={R.loc[a,b]:+.2f}{'*' if P.loc[a,b]<0.05 else ''}\nn={int(N.loc[a,b])}" if len(j) >= 5 else f"n={len(j)}", ha="center", va="center", fontsize=12, transform=ax.transAxes); continue
            ax.loglog(j[b].clip(1e-4), j[a].clip(1e-4), "o", ms=4, color="#2a78d6")
            if len(j) >= 2: lo, hi = min(j.clip(1e-4).min()), max(j.max()); ax.plot([lo, hi], [lo, hi], color="#bbb", lw=.7)
            if k == 0: ax.set_ylabel(a, fontsize=8)
            if i == n-1: ax.set_xlabel(b, fontsize=8)
    fig2.suptitle(f"{label} flow volume, gauge vs gauge (10⁶ m³, log-log; grey = 1:1)", y=0.995); fig2.tight_layout(); fig2.savefig(D+f"tuo_{tag}_volume_scatter_matrix.png", dpi=130, bbox_inches="tight"); plt.close(fig2)
rows = []; VOL = {}
for wname, (months, need) in WIN.items():
    T = pd.DataFrame({k: window_vol(v, months, need) for k, v in G.items()}); VOL[wname] = T; T.to_csv(D+f"tuo_{wname.lower().replace('-','')}_volume_table.csv")
    if wname != "Sep": xcorr_figs(T, wname.lower().replace("-", ""), wname)
    for g in names:
        v = T[g].dropna(); pk = PK[g].dropna(); idx = [y for y in v.index if y-1 in v.index and y in pk.index and (y-1) in pk.index]
        if len(idx) < 7: rows.append(dict(window=wname, gauge=g, n=len(idx))); continue
        vy = np.array([v[y] for y in idx]); vp = np.array([v[y-1] for y in idx]); py = np.array([pk[y] for y in idx])
        r_raw, p_raw = spearmanr(vp, vy); ny = vy/py; npv = np.array([v[y-1]/pk[y-1] for y in idx]); r_rat, p_rat = spearmanr(npv, ny); r_par, p_par = partial_spearman(vp, vy, py)
        rows.append(dict(window=wname, gauge=g, n=len(idx), raw=r_raw, p_raw=p_raw, ratio=r_rat, p_ratio=p_rat, partial=r_par, p_partial=p_par))
S = pd.DataFrame(rows); S.to_csv(D+"tuo_lowflow_lag1_summary.csv", index=False); pd.set_option("display.width", 200); print(S.round(3).to_string())
# summary heatmap: gauges x (window, method)
meth = [("raw", "raw"), ("ratio", "V/peak"), ("partial", "partial | peak")]; cols = [(w, m) for w in ["Sep", "Sep-Oct", "Oct"] for m in meth]
M = np.full((len(names), len(cols)), np.nan); Pm = M.copy(); Nn = M.copy()
for ci, (w, (m, _)) in enumerate(cols):
    for gi, g in enumerate(names):
        r = S[(S.window == w) & (S.gauge == g)]
        if len(r) and m in r and not r[m].isna().all(): M[gi, ci] = r[m].values[0]; Pm[gi, ci] = r["p_"+m].values[0]; Nn[gi, ci] = r["n"].values[0]
fig, ax = plt.subplots(figsize=(11, 4.6)); im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{w}\n{l[1]}" for w, l in cols], fontsize=8); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for gi in range(len(names)):
    for ci in range(len(cols)):
        if not np.isnan(M[gi, ci]): ax.text(ci, gi, f"{M[gi,ci]:+.2f}{'*' if Pm[gi,ci] < 0.05 else ''}\nn={int(Nn[gi,ci])}", ha="center", va="center", fontsize=7.5, color="white" if abs(M[gi, ci]) > 0.6 else "#222")
for x in (2.5, 5.5): ax.axvline(x, color="white", lw=3)
ax.set_title("Lag-1 (year-to-year) Spearman correlation of low-flow volume, raw vs normalised for spring peak flow  (* p<0.05)", fontsize=10, loc="left"); fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); fig.savefig(D+"tuo_lowflow_lag1_summary.png", dpi=150); print("saved")
