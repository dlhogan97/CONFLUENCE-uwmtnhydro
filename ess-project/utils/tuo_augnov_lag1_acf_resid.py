"""Lag-1 ACF of low-flow RESIDUALS after removing the peak-flow relationship. Per gauge: OLS of log10(min 7-d flow, Aug 1-Nov 30 of year y) on log10(peak 7-d flow of water year y; HH record, used for all gauges), then the
rank ACF (5000-perm p) of the residual series. Low flows floored at 1e-5 m3/s before the log (dry streambeds). Rows: (1) low flow vs peak with fit, (2) residual y vs y-1, (3) residual ACF lags 1-4.
-> ess-project/figures/draft/tuo_augnov_lag1_acf_resid.png, tuo_augnov_resid_table.csv, tuo_augnov_acf_resid_summary.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("base", "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/tuo_augnov_lag1_acf.py")
src = open(spec.origin).read(); src = src[:src.index('L = pd.DataFrame({k: low7')]; ns = {}; exec(compile(src, "base", "exec"), ns)     # reuse loaders/metrics only
G, names, low7, acf, D = ns["G"], ns["names"], ns["low7"], ns["acf"], ns["D"]
def peak7(q):
    q = q.asfreq("D"); r7 = q.rolling(7, min_periods=6).mean(); out = {}
    for y in sorted(set(q.index.year)):
        if q[(q.index.year == y) & q.index.month.isin([4, 5, 6, 7])].notna().sum() >= 100: out[y] = r7[(r7.index >= pd.Timestamp(y-1, 10, 1)) & (r7.index <= pd.Timestamp(y, 9, 30))].max()
    return pd.Series(out)
PEAK = peak7(G["HH unimpaired"]); LOW = pd.DataFrame({k: low7(v) for k, v in G.items()}); RES = pd.DataFrame(index=LOW.index, columns=names, dtype=float); fits = {}
for g in names:
    d = pd.concat([np.log10(LOW[g].clip(lower=1e-5)).rename("y"), np.log10(PEAK).rename("x")], axis=1).dropna()
    b, a = np.polyfit(d.x, d.y, 1); res = d.y-(a+b*d.x); r2 = 1-res.var()/d.y.var(); fits[g] = (a, b, r2, len(d), d); RES.loc[d.index, g] = res
RES.index.name = "year"; RES.to_csv(D+"tuo_augnov_resid_table.csv")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); n = len(names); fig, ax = plt.subplots(3, n, figsize=(3.4*n, 10.4)); rows = []
for gi, g in enumerate(names):
    a_, b_, r2, nn, d = fits[g]; s = RES[g].dropna(); rho, p, m, band, ny = acf(s); rows += [dict(gauge=g, lag=k+1, rho=rho[k], p=p[k], n_pairs=m[k], band=band, n_years=ny, slope=b_, r2=r2) for k in range(4)]
    a = ax[0, gi]; a.plot(d.x, d.y, "o", ms=5, color="#2a78d6"); xs = np.array([d.x.min(), d.x.max()]); a.plot(xs, a_+b_*xs, color="#d55e00", lw=1.4)
    a.set_title(f"{g}\nlog low = {b_:.2f}·log peak + c   R²={r2:.2f} (n={nn})", fontsize=8.5, loc="left"); a.set_xlabel("log10 peak 7-d flow (m³/s)"); a.grid(color="#e3e2de", lw=.5)
    if gi == 0: a.set_ylabel("log10 low flow, Aug–Nov (m³/s)")
    a = ax[1, gi]; pairs = [(y, s[y-1], s[y]) for y in s.index if y-1 in s.index]
    if pairs:
        _, x, y_ = zip(*pairs); a.plot(x, y_, "o", ms=5, color="#2a78d6"); lo, hi = min(min(x), min(y_)), max(max(x), max(y_)); a.plot([lo, hi], [lo, hi], color="#bbb", lw=.7)
        for t, xx, yv in pairs: a.annotate(str(t)[2:], (xx, yv), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#52514e")
    a.axhline(0, color="#ddd", lw=.6); a.axvline(0, color="#ddd", lw=.6)
    a.set_title(f"residual lag-1 ρ = {rho[0]:+.2f}{'*' if p[0] < 0.05 else ''} (p={p[0]:.2f}, n={m[0]})", fontsize=9, loc="left"); a.set_xlabel("residual, year y−1 (log10 units)")
    if gi == 0: a.set_ylabel("residual, year y")
    b = ax[2, gi]; cols = ["#d55e00" if p[k] < 0.05 else "#2a78d6" for k in range(4)]; b.bar(range(1, 5), np.nan_to_num(rho), color=cols, width=0.6); b.axhline(0, color="#52514e", lw=.8); b.axhspan(-band, band, color="#999", alpha=.18, lw=0)
    b.set_ylim(-1, 1); b.set_xticks(range(1, 5)); b.set_xlabel("lag (years)")
    for k in range(4): v = rho[k] if not np.isnan(rho[k]) else 0; b.text(k+1, v+(0.06 if v >= 0 else -0.12), f"n={m[k]}", ha="center", fontsize=6.5, color="#52514e")
    if gi == 0: b.set_ylabel("rank ACF of residuals (grey = ±1.96/√n; orange p<0.05)")
fig.suptitle("Low-flow (Aug–Nov min 7-day) residual after peak-flow regression: lag-1 ACF (T120 divided by 35.3 after WY2016)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_augnov_lag1_acf_resid.png", dpi=140, bbox_inches="tight")
S = pd.DataFrame(rows); S.to_csv(D+"tuo_augnov_acf_resid_summary.csv", index=False); print(S.pivot(index="gauge", columns="lag", values="rho").round(2).reindex(names)); print(S[S.lag == 1][["gauge", "rho", "p", "n_pairs", "n_years", "slope", "r2"]].round(3).to_string(index=False))
