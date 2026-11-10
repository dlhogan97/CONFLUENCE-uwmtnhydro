"""Daily basin snow-covered fraction (SCA), May 1 - Sep 30, 2013-2025, from Landsat scenes + ASO flights.
Landsat: area-weighted fSCA over valid elevation bands (shadow-inclusive, scenes with valid_sh >= 0.5), from landsat_spring_scenes.csv + scenes_by_band3.csv.
ASO: basin snow-covered fraction at SWE > 0.01 m (aso_all_flights.csv, coverage >= 0.95), converted to Landsat-equivalent with a log-log fit on 28 coincident pairs (<= 4 d): log10 f_L = a + b log10 f_ASO (ASO overreads thin/patchy snow at low cover).
Series: interpolation in log10(f + 0.002) between all anchor points (Mar 1-Oct 31) on a MELT AXIS (cumulative positive degree-days of basin-mean air temperature from Mar 1; held-out RMSE 0.18 log10 vs 0.22 for plain time interpolation), gaps > 75 d left empty (back/forward fill <= 14 d); gap_days = size of the bracketing anchor gap for every day, non-increasing after the seasonal peak (cloud / new-snow spikes ignored). Source of each day's bracketing anchors is recorded.
-> modeling/calibration/observations/tuolumne_sca_daily_2013_2025.csv, figures/draft/tuo_sca_daily_series.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"; GS = "/scratch/dlhogan/ess-project-data/geospatial-data/"; P = GS+"snowpatch_fsca/"
BA = pd.read_csv(GS+"tuolumne_decmar_precip_bands.csv").set_index("elevClass")["area_km2"].to_dict(); BANDS = [8, 7, 6, 5, 4, 3, 2]; ATOT = sum(BA.values()); LOGOFF = 0.002
sp = pd.concat([pd.read_csv(P+"landsat_spring_scenes.csv", parse_dates=["date"]), pd.read_csv(P+"scenes_by_band3.csv", parse_dates=["date"]).query("year >= 2013")]).drop_duplicates("date"); sp = sp[(sp.status == "ok") & (sp.valid_sh >= 0.5)].copy()
def frac(r):
    num = den = 0.0
    for b in BANDS:
        v = r.get(f"fsca_b{b}_sh")
        if pd.notna(v): num += v*BA[b]; den += BA[b]
    return num/den if den >= 0.75*ATOT else np.nan
sp["f"] = sp.apply(frac, axis=1); sp = sp.dropna(subset=["f"]); L = sp[["date", "f"]].assign(src="Landsat")
aso = pd.read_csv(P+"aso_all_flights.csv", parse_dates=["date"]); aso = aso[aso.cov_basin >= 0.95].copy()
pairs = []
for _, a in aso.iterrows():
    d = (sp.date-a.date).abs(); i = d.idxmin()
    if abs((sp.loc[i, "date"]-a.date).days) <= 4 and sp.loc[i, "valid_sh"] >= 0.35: pairs.append((a["f_basin_t0.01"], sp.loc[i, "f"]))
pairs = np.array(pairs); xa = np.log10(np.clip(pairs[:, 0], LOGOFF, None)); xl = np.log10(np.clip(pairs[:, 1], LOGOFF, None)); b, a0 = np.polyfit(xa, xl, 1); rmse = np.sqrt(np.mean((xl-(a0+b*xa))**2)); print(f"ASO->Landsat-equivalent: log10 f_L = {a0:+.3f} + {b:.3f} log10 f_ASO  (n={len(pairs)}, rmse {rmse:.2f} log10 = x{10**rmse:.2f})")
aso["f"] = np.clip(10**(a0+b*np.log10(np.clip(aso["f_basin_t0.01"], LOGOFF, None))), 0, 1); A = aso[["date", "f"]].assign(src="ASO"); anch = pd.concat([L, A]).sort_values("date"); anch = anch[(anch.date.dt.month >= 3) & (anch.date.dt.month <= 10)]
met = pd.read_csv(OBS+"tuolumne_daily_met.csv", index_col=0, parse_dates=True); MAXGAP = 75
out = []; cover = {}
for y in range(2013, 2026):
    x = anch[anch.date.dt.year == y].drop_duplicates("date").sort_values("date").set_index("date"); idx = pd.date_range(pd.Timestamp(y, 5, 1), pd.Timestamp(y, 9, 30))
    if len(x) < 2: continue
    tt = pd.date_range(pd.Timestamp(y, 3, 1), pd.Timestamp(y, 10, 31)); C = met.T_basin.reindex(tt).clip(lower=0).cumsum()+1e-3*np.arange(len(tt))       # melt axis: cumulative positive degree-days (basin-mean air T) from Mar 1
    lf = np.log10(x.f.clip(lower=0)+LOGOFF); d = x.index.values; Ca = np.array([C[t] for t in x.index]); vals = []; gaps = []
    for t in idx:
        i = d.searchsorted(t.to_datetime64())
        if i < len(d) and d[i] == t.to_datetime64(): vals.append(lf.iloc[i]); gaps.append(0); continue
        if 0 < i < len(d):
            g = int((d[i]-d[i-1]).astype("timedelta64[D]").astype(int)); w = (C[t]-Ca[i-1])/(Ca[i]-Ca[i-1]); vals.append(lf.iloc[i-1]+w*(lf.iloc[i]-lf.iloc[i-1])); gaps.append(g)
        elif i == 0: g = int((d[0]-t.to_datetime64()).astype("timedelta64[D]").astype(int)); vals.append(lf.iloc[0] if g <= 14 else np.nan); gaps.append(g)
        else: g = int((t.to_datetime64()-d[-1]).astype("timedelta64[D]").astype(int)); vals.append(lf.iloc[-1] if g <= 14 else np.nan); gaps.append(g)
    f = pd.Series(10**np.array(vals, dtype=float)-LOGOFF, index=idx).clip(lower=0); gp = pd.Series(gaps, index=idx); f[gp > MAXGAP] = np.nan
    pk = x.f[(x.index.month <= 6)].idxmax() if (x.index.month <= 6).any() else None
    if pk is not None: f.loc[f.index >= pk] = f.loc[f.index >= pk].cummin()
    s_ = pd.DataFrame({"sca": f, "gap_days": gp}); s_["year"] = y; out.append(s_); cover[y] = (float(f.notna().mean()), int((x.src == "Landsat").sum()), int((x.src == "ASO").sum()), float((f.notna() & (gp <= 30)).mean()))
S = pd.concat(out); S.index.name = "date"; S.to_csv(OBS+"tuolumne_sca_daily_2013_2025.csv"); print("year: fraction of May1-Sep30 with SCA (gap<=75 d) | ... of which bracketing gap <=30 d | Landsat anchors | ASO anchors (Mar-Oct)")
for y, (c, nl, na, c30) in cover.items(): print(f"  {y}: {c:.2f} | {c30:.2f} | {nl} | {na}")
# compare with Landsat-only coverage
Lonly = {}
for y in range(2013, 2026):
    x = L[(L.date.dt.year == y) & (L.date.dt.month >= 3)].set_index("date"); idx = pd.date_range(pd.Timestamp(y, 5, 1), pd.Timestamp(y, 9, 30)); d = x.index.values
    Lonly[y] = np.mean([(d.searchsorted(t.to_datetime64()) > 0 and d.searchsorted(t.to_datetime64()) < len(d) and (d[d.searchsorted(t.to_datetime64())]-d[d.searchsorted(t.to_datetime64())-1]).astype("timedelta64[D]").astype(int) <= 24) for t in idx]) if len(x) > 1 else 0
print("Landsat-only coverage (previous approach, <=24 d gaps):", {y: round(v, 2) for y, v in Lonly.items()})
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(3, 5, figsize=(21, 10), sharey=True); ax = ax.ravel()
for k, y in enumerate(range(2013, 2026)):
    a = ax[k]; s = S[S.year == y]; a.plot(s.index.dayofyear, np.clip(s.sca*100, 0.05, None), "-", color="#444", lw=1.2)
    x = anch[anch.date.dt.year == y]
    for src, c, m in (("Landsat", "#2a78d6", "o"), ("ASO", "#d55e00", "s")):
        z = x[x.src == src]; a.plot(z.date.dt.dayofyear, np.clip(z.f*100, 0.05, None), m, color=c, ms=4.5, label=src)
    a.set_yscale("log"); a.set_xlim(60, 275); a.set_title(f"{y} (SCA coverage {cover.get(y, (0,))[0]*100:.0f}% of May–Sep)", fontsize=9, loc="left"); a.axvspan(121, 274, color="#f3f3f0", zorder=0); a.grid(color="#e3e2de", lw=.4)
    if k == 0: a.legend(frameon=False, fontsize=8)
    if k % 5 == 0: a.set_ylabel("basin SCA (%)")
a = ax[13]; a.plot(10**xa, 10**xl, "o", color="#2a78d6"); lim = np.array([0.005, 1]); a.plot(lim, lim, color="#999", lw=.8); a.plot(lim, 10**(a0+b*np.log10(lim)), color="#d55e00"); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("ASO fraction (SWE>0.01 m)"); a.set_ylabel("Landsat fraction"); a.set_title(f"ASO vs Landsat, {len(pairs)} pairs; fit slope {b:.2f}", fontsize=9, loc="left")
ax[14].axis("off"); fig.tight_layout(); fig.savefig(D+"tuo_sca_daily_series.png", dpi=130, bbox_inches="tight")
