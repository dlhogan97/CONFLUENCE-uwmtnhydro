"""Lag-1 low-flow ACF per Tuolumne gauge. Low-flow metric = minimum centred 7-day mean flow, Aug 1 - Nov 30 (needs >=110 of 122 days). Rank (Spearman) ACF, 5000-permutation p, +-1.96/sqrt(n) band.
Top row: scatter of year y vs y-1 (log-log). Bottom row: ACF lags 1-4. -> ess-project/figures/draft/tuo_augnov_lag1_acf.png, tuo_augnov_lowflow_table.csv, tuo_augnov_acf_summary.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
SC = "/scratch/dlhogan/ess-project-data/"; SF = SC+"streamflow-data/TuolumneRiver/"; D = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
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
def low7(q):
    q = q.asfreq("D"); r7 = q.rolling(7, center=True, min_periods=6).mean(); r = {}
    for y in sorted(set(q.index.year)):
        w = (q.index.year == y) & (q.index.month >= 8) & (q.index.month <= 11)
        if q[w].notna().sum() >= 110 and r7[w].notna().any(): r[y] = r7[w].min()
    return pd.Series(r)
def sp(a, b):
    ra = rankdata(a)-(len(a)+1)/2.; rb = rankdata(b)-(len(b)+1)/2.; den = np.sqrt((ra*ra).sum()*(rb*rb).sum()); return (ra*rb).sum()/den if den > 0 else 0.
def acf(s, nl=4, nperm=5000, seed=0):
    rng = np.random.default_rng(seed); s = s.dropna(); yrs = s.index.values.astype(int); v = s.values; n = v.size; pos = {int(y): i for i, y in enumerate(yrs)}; rho = np.full(nl, np.nan); p = np.full(nl, np.nan); m = np.zeros(nl, int)
    for k in range(1, nl+1):
        ic = np.array([i for i, y in enumerate(yrs) if int(y)-k in pos], int); ip = np.array([pos[int(yrs[i])-k] for i in ic], int); m[k-1] = ic.size
        if ic.size < 6: continue
        rho[k-1] = sp(v[ip], v[ic]); pm = np.array([abs(sp(r[ip], r[ic])) for r in (rng.permutation(v) for _ in range(nperm))]); p[k-1] = np.mean(pm >= abs(rho[k-1]))
    return rho, p, m, 1.96/np.sqrt(n), n
L = pd.DataFrame({k: low7(v) for k, v in G.items()}); L.index.name = "year"; L.to_csv(D+"tuo_augnov_lowflow_table.csv")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); n = len(names)
fig, ax = plt.subplots(2, n, figsize=(3.4*n, 7.2)); rows = []
for gi, g in enumerate(names):
    s = L[g].dropna(); rho, p, m, band, nn = acf(s); rows += [dict(gauge=g, lag=k+1, rho=rho[k], p=p[k], n_pairs=m[k], band=band, n_years=nn) for k in range(4)]
    a = ax[0, gi]; pairs = [(y, s[y-1], s[y]) for y in s.index if y-1 in s.index]
    if pairs:
        yy, x, y_ = zip(*pairs); a.loglog(x, y_, "o", ms=5, color="#2a78d6")
        for t, xx, yv in pairs: a.annotate(str(t)[2:], (xx, yv), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#52514e")
        lo, hi = min(min(x), min(y_)), max(max(x), max(y_)); a.plot([lo, hi], [lo, hi], color="#bbb", lw=.7)
    a.set_title(f"{g}\nlag-1 ρ = {rho[0]:+.2f}{'*' if p[0] < 0.05 else ''}  (p={p[0]:.2f}, n={m[0]})", fontsize=9, loc="left"); a.set_xlabel("min 7-d flow Aug–Nov, year y−1 (m³/s)"); a.grid(color="#e3e2de", lw=.5)
    if gi == 0: a.set_ylabel("year y (m³/s)")
    b = ax[1, gi]; cols = ["#d55e00" if (p[k] < 0.05) else "#2a78d6" for k in range(4)]; b.bar(range(1, 5), np.nan_to_num(rho), color=cols, width=0.6); b.axhline(0, color="#52514e", lw=.8)
    b.axhspan(-band, band, color="#999", alpha=.18, lw=0); b.set_ylim(-1, 1); b.set_xticks(range(1, 5)); b.set_xlabel("lag (years)")
    for k in range(4): b.text(k+1, (rho[k] if not np.isnan(rho[k]) else 0)+(0.06 if (rho[k] if not np.isnan(rho[k]) else 0) >= 0 else -0.12), f"n={m[k]}", ha="center", fontsize=6.5, color="#52514e")
    if gi == 0: b.set_ylabel("rank ACF (grey = ±1.96/√n; orange p<0.05)")
fig.suptitle("Lag-1 low-flow ACF, Aug–Nov minimum 7-day flow (T120 divided by 35.3 after WY2016)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_augnov_lag1_acf.png", dpi=140, bbox_inches="tight")
S = pd.DataFrame(rows); S.to_csv(D+"tuo_augnov_acf_summary.csv", index=False); print(S.pivot(index="gauge", columns="lag", values="rho").round(2).reindex(names)); print(S[S.lag == 1][["gauge", "rho", "p", "n_pairs", "n_years"]].round(3).to_string(index=False))
