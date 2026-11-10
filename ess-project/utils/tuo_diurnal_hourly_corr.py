"""Is there ANY hour of the day at which the diurnal flow anomaly correlates positively with perennial snow area (melt needs time to travel)?
For each gauge-year (Aug 1-Oct 31, local clock): composite diurnal anomaly of discharge by hour over QUIET days (>=5 d after any >5% daily-mean rise, daily mean >= 0.01 m3/s, enough samples): anomaly_h = Q(h) - (daily mean + linear within-day trend), as RELATIVE (/daily mean) and ABSOLUTE (m3/s).
Then Spearman rho across years between each hour's annual composite anomaly and end-of-season snow area (and partial controlling for Aug-Oct T_high and SW_high). Gauges: 4 nested (QC'd, 30-min, UTC->UTC-7) + USGS 11274790 (15-min NWIS, PDT).
-> figures/draft/tuo_diurnal_hourly_corr.png, tuo_diurnal_hourly_corr.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"
DN = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/TuolumneStreamflow_2002_2021/data/"
NEST = {"T120": "Tuolumne120_streamflow.csv", "LBM": "LyellBlwMaclure_streamflow.csv", "LABvTB": "LyellAbvTB_streamflow.csv", "DFkBC": "DanaFkBugCamp_streamflow.csv"}
CUT = pd.Timestamp("2021-08-12 23:59"); QMIN = 0.01
def series(g):
    if g == "USGS11274790":
        d = pd.read_csv(OBS+"usgs11274790_iv_augoct.csv"); d = d[d.tz == "PDT"].copy(); d["loc"] = pd.to_datetime(d.datetime); d["q"] = pd.to_numeric(d.q_cms, errors="coerce"); return d[["loc", "q"]].dropna()
    d = pd.read_csv(DN+NEST[g], comment="#", parse_dates=["DateTime"]); d = d[(d.DateTime <= CUT) & d.Discharge_Flag.isin([0, 1])]; d["loc"] = d.DateTime-pd.Timedelta(hours=7); d["q"] = d.Estimated_Discharge; d = d[d["loc"].dt.month.isin([8, 9, 10])]; return d[["loc", "q"]].dropna()
def hourly_composites(d):
    d = d.copy(); d["day"] = d["loc"].dt.floor("D"); dm = d.groupby("day").q.mean().asfreq("D"); quiet = (dm.pct_change() > 0.05).astype(float).rolling(5, min_periods=5).sum().eq(0); out = {}
    for day, x in d.groupby("day"):
        if not quiet.get(day, False) or len(x) < 20: continue
        h = (x["loc"].dt.hour+x["loc"].dt.minute/60).values
        if np.max(np.diff(np.sort(h))) > 4: continue
        q = x.q.values; m = q.mean()
        if m < QMIN: continue
        X = np.column_stack([np.ones_like(h), h-12.0]); res = q-X@np.linalg.lstsq(X, q, rcond=None)[0]; b = pd.Series(res).groupby(np.floor(h).astype(int)).mean().reindex(range(24))
        out[day] = (b.values/m, b.values, m)
    return out
met = pd.read_csv(OBS+"tuolumne_daily_met.csv", index_col=0, parse_dates=True); sel = lambda s: s[s.index.month.isin([8, 9, 10])].groupby(lambda i: i.year).mean()
Tw, Sw = sel(met.T_high), sel(met.SW_high); eos = pd.read_csv(FS+"eos_final.csv").set_index("year").A_shadow
def pcor(x, y, Z):
    X = np.column_stack([np.ones(len(x))]+[rankdata(z) for z in Z]); rx, ry = rankdata(x), rankdata(y); ex = rx-X@np.linalg.lstsq(X, rx, rcond=None)[0]; ey = ry-X@np.linalg.lstsq(X, ry, rcond=None)[0]; r = np.corrcoef(ex, ey)[0, 1]; df = len(x)-2-len(Z)
    return r, 2*tdist.sf(abs(r*np.sqrt(df/(1-r**2))), df)
rows = []; comp = {}
for g in list(NEST)+["USGS11274790"]:
    c = hourly_composites(series(g)); yrs = sorted(set(k.year for k in c))
    for kind, idx in (("relative", 0), ("absolute", 1)):
        M = {}
        for y in yrs:
            v = np.array([val[idx] for k, val in c.items() if k.year == y])
            if len(v) >= 10: M[y] = np.nanmedian(v, axis=0)
        if len(M) < 8: continue
        A = pd.DataFrame(M).T; comp[(g, kind)] = A
        for h in range(24):
            d = pd.concat([A[h].rename("y"), eos.rename("x"), Tw.rename("Tw"), Sw.rename("Sw")], axis=1).dropna()
            if len(d) < 8: continue
            r, p = spearmanr(d.x, d.y); rp, pp = pcor(d.x.values, d.y.values, [d.Tw.values, d.Sw.values]); rows.append(dict(gauge=g, kind=kind, hour=h, n=len(d), rho=r, p=p, rho_partial=rp, p_partial=pp))
R = pd.DataFrame(rows); R.to_csv(D+"tuo_diurnal_hourly_corr.csv", index=False); pd.set_option("display.width", 220)
for kind in ("relative", "absolute"):
    P = R[R.kind == kind].pivot(index="hour", columns="gauge", values="rho"); print(kind, "rho by hour"); print(P.round(2).T.to_string())
    print("max rho per gauge (hour, rho, p):", {g: (int(x.loc[x.rho.idxmax(), "hour"]), round(x.rho.max(), 2), round(x.loc[x.rho.idxmax(), "p"], 3)) for g, x in R[R.kind == kind].groupby("gauge")})
fig, ax = plt.subplots(2, 5, figsize=(21, 7), sharex=True, sharey=True); G_ = list(NEST)+["USGS11274790"]
for j, g in enumerate(G_):
    for i, kind in enumerate(("relative", "absolute")):
        a = ax[i, j]; x = R[(R.gauge == g) & (R.kind == kind)]
        if len(x) == 0: a.set_title(f"{g} (insufficient years)", fontsize=9, loc="left"); continue
        a.plot(x.hour, x.rho, "o-", color="#2a78d6", ms=4, label="raw"); a.plot(x.hour, x.rho_partial, "s--", color="#d55e00", ms=3.5, lw=1, label="partial | T, SW"); a.axhline(0, color="#52514e", lw=.8)
        sg = x[x.p < 0.05]; a.scatter(sg.hour, sg.rho, s=95, facecolors="none", edgecolors="#0b0b0b", lw=1.2, label="p<0.05 (raw)"); a.set_title(f"{g} — {kind} anomaly (n={int(x.n.max())})", fontsize=9, loc="left"); a.set_xticks(range(0, 24, 3)); a.set_ylim(-1, 1); a.grid(color="#e3e2de", lw=.5)
        if i == 1: a.set_xlabel("local hour")
        if j == 0: a.set_ylabel("Spearman ρ with snow area")
ax[0, 0].legend(frameon=False, fontsize=7.5); fig.suptitle("Hour-of-day correlation of the diurnal flow anomaly with end-of-season snow area (quiet days, Aug–Oct)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_diurnal_hourly_corr.png", dpi=140, bbox_inches="tight")
