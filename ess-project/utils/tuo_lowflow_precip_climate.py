"""What drives year-to-year Tuolumne low flow? Rank correlations of gauge low flows (Sep volume, Aug-Nov min 7-d flow; nested gauges cut at 2021-08-12 = last QC'd date) with
PRISM basin precipitation (WY, lagged WY, Nov-Mar, Apr-Jun, Aug-Nov), observed TUM precip, DAN Apr-1 SWE, WY temperature, end-of-season snow area, peak flow, and ENSO (ONI DJF) / PDO (Nov-Mar).
Also lag structure: low flow(y) vs WY precip (y-k), k=0..3, partial lag-1 | same-year precip, and the autocorrelation of precipitation itself.
-> figures/draft/tuo_lowflow_drivers_heatmap.png, tuo_lowflow_precip_lags.png, tuo_lowflow_drivers.csv; modeling/calibration/observations/prism_tuolumne_monthly_precip.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, glob, io, urllib.request, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, t as tdist
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBSD = U+"modeling/calibration/observations/"
src = open(U+"utils/tuo_augnov_lag1_acf_qc.py").read(); ns = {}; exec(compile(src[:src.index('L = pd.DataFrame({k: low7')], "base", "exec"), ns)
G, names, low7, D = ns["G"], ns["names"], ns["low7"], ns["D"]
SC = "/scratch/dlhogan/ess-project-data/"; FS = SC+"geospatial-data/snowpatch_fsca/"
# ---------- PRISM-based basin precipitation: METSIM forcing (PRISM-sampled precip, before the ASO gradient correction), area-weighted over the 28 HRUs, monthly totals (mm) 1981-2025
fn = OBSD+"prism_tuolumne_monthly_precip.csv"
try: pm = pd.read_csv(fn, index_col=0, parse_dates=True)["precip_mm"]
except Exception:
    area = np.array(xr.open_dataset(SC+"domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")["HRUarea"].values, float); w = area/area.sum(); rows = {}
    for f in sorted(glob.glob(SC+"domain_Tuolumne_River_distributed_elevAspect/forcing/basin_averaged_data/*_metsim_*.nc")):
        d = xr.open_dataset(f); p = d["pptrate"].values.astype(float)*3600.0; rows[pd.Timestamp(d["time"].values[0]).replace(day=1)] = float((p.sum(0)*w).sum()); d.close()
    pm = pd.Series(rows).sort_index(); pm.rename("precip_mm").to_csv(fn, header=True)
wy = lambda i: i.year+(i.month >= 10)
def agg(months, label):
    s = pm[pm.index.month.isin(months)]; yy = (s.index.year + (s.index.month >= 10)) if min(months) >= 10 or max(months) >= 10 else s.index.year
    return s.groupby(yy).sum().rename(label)
# months by water year: Oct(y-1)..Sep(y)
P_wy = pm.groupby(wy(pm.index)).sum(); cnt = pm.groupby(wy(pm.index)).size(); P_wy = P_wy[cnt == 12].rename("WY precip")
def sel(months_cal, label):   # calendar months within water year y, using calendar year = y (Jan-Sep) / y-1 (Oct-Dec)
    s = pm[pm.index.month.isin(months_cal)]; return s.groupby(wy(s.index)).sum().rename(label)
Pnm = sel([11, 12, 1, 2, 3], "Nov-Mar precip"); Paj = sel([4, 5, 6], "Apr-Jun precip")
Pan = pm[pm.index.month.isin([8, 9, 10, 11])]; Pan = Pan.groupby(Pan.index.year).sum().rename("Aug-Nov precip (low-flow window)")
Pwy1 = P_wy.shift(1, fill_value=np.nan); Pwy1 = P_wy.reindex(P_wy.index-1).set_axis(P_wy.index).rename("WY precip, previous WY"); Pwy2 = P_wy.reindex(P_wy.index-2).set_axis(P_wy.index).rename("WY precip, 2 WY back"); Pwy3 = P_wy.reindex(P_wy.index-3).set_axis(P_wy.index).rename("WY precip, 3 WY back")
P2 = (P_wy+Pwy1).rename("2-yr precip (WY + prev)")
tum = pd.read_csv(FS+"tum_wy_precip.csv", comment="#").set_index("water_year")["wyppt_mm"].rename("TUM gauge WY precip (obs)")
dan = pd.read_csv(FS+"dan_apr1_swe.csv", comment="#").set_index("water_year")["apr1_mm"].rename("DAN Apr-1 SWE (obs)")
tair = pd.read_csv(FS+"basin_wy_forcing.csv").set_index("wy")["Tuolumne_wytair_C"].rename("WY air temp (forcing)")
eos = pd.read_csv(FS+"eos_final.csv").set_index("year")["A_shadow"].rename("end-of-season snow area")
def nino():
    t = urllib.request.urlopen("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=30).read().decode().split("\n")[1:]; r = {}
    for l in t:
        p = l.split()
        if len(p) == 4 and p[0] == "DJF": r[int(p[1])+1] = float(p[3])          # DJF 1950 centred on Jan 1950 -> water year 1950 (Dec of previous year)
    return pd.Series(r).rename("ONI (DJF)")
def pdo():
    t = urllib.request.urlopen("https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat", timeout=30).read().decode().split("\n")[2:]; r = {}
    for l in t:
        p = l.split()
        if len(p) == 13 and p[0].isdigit(): r[int(p[0])] = [float(v) for v in p[1:]]
    out = {}
    for y, v in r.items():
        if y-1 in r: out[y] = np.nanmean([r[y-1][10], r[y-1][11], v[0], v[1], v[2]])      # Nov(y-1) .. Mar(y)
    return pd.Series(out).rename("PDO (Nov-Mar)")
ONI, PDO = nino(), pdo()
# ---------- peak flow (HH record) as in earlier analyses
def peak7(q):
    q = q.asfreq("D"); r7 = q.rolling(7, min_periods=6).mean(); out = {}
    for y in sorted(set(q.index.year)):
        if q[(q.index.year == y) & q.index.month.isin([4, 5, 6, 7])].notna().sum() >= 100: out[y] = r7[(r7.index >= pd.Timestamp(y-1, 10, 1)) & (r7.index <= pd.Timestamp(y, 9, 30))].max()
    return pd.Series(out).rename("peak flow (HH)")
def sepvol(q):
    q = q.asfreq("D"); r = {}
    for y in sorted(set(q.index.year)):
        w = (q.index.year == y) & (q.index.month == 9); z = q[w].dropna()
        if len(z) >= 25: r[y] = z.mean()*30*86400/1e6
    return pd.Series(r)
PRED = pd.concat([P_wy, Pwy1, Pwy2, Pwy3, P2, Pnm, Paj, Pan, tum, dan, tair, eos, peak7(G["HH unimpaired"]), ONI, PDO], axis=1)
TARG = {}
for g in names: TARG[(g, "Sep volume")] = sepvol(G[g]); TARG[(g, "Aug-Nov min 7-d")] = low7(G[g])
rows = []
for (g, mn), s in TARG.items():
    for c in PRED.columns:
        d = pd.concat([s.rename("q"), PRED[c].rename("x")], axis=1).dropna()
        if len(d) >= 7: r, p = spearmanr(d.x, d.q); rows.append(dict(gauge=g, metric=mn, predictor=c, n=len(d), rho=r, p=p))
R = pd.DataFrame(rows); R.to_csv(D+"tuo_lowflow_drivers.csv", index=False)
# ---------- heatmap
fig, ax = plt.subplots(1, 2, figsize=(20, 6.8), sharey=True)
for k, mn in enumerate(["Sep volume", "Aug-Nov min 7-d"]):
    M = R[R.metric == mn].pivot(index="gauge", columns="predictor", values="rho").reindex(index=names, columns=PRED.columns); P = R[R.metric == mn].pivot(index="gauge", columns="predictor", values="p").reindex(index=names, columns=PRED.columns); N = R[R.metric == mn].pivot(index="gauge", columns="predictor", values="n").reindex(index=names, columns=PRED.columns)
    im = ax[k].imshow(M.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto"); ax[k].set_xticks(range(M.shape[1])); ax[k].set_xticklabels(M.columns, rotation=45, ha="right", fontsize=8); ax[k].set_yticks(range(len(names))); ax[k].set_yticklabels(names); ax[k].set_title(f"{mn}: Spearman ρ with candidate drivers (* p<0.05)", fontsize=10, loc="left")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v): ax[k].text(j, i, f"{v:+.2f}{'*' if P.values[i, j] < 0.05 else ''}\nn={int(N.values[i, j])}", ha="center", va="center", fontsize=6.8, color="white" if abs(v) > 0.6 else "#222")
fig.colorbar(im, ax=ax, fraction=0.012); fig.savefig(D+"tuo_lowflow_drivers_heatmap.png", dpi=140, bbox_inches="tight")
# ---------- lag structure
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); n = len(x); f = lambda a, b: np.corrcoef(a, b)[0, 1]
    r = (f(rx, ry)-f(rx, rz)*f(ry, rz))/np.sqrt((1-f(rx, rz)**2)*(1-f(ry, rz)**2)); return r, 2*tdist.sf(abs(r*np.sqrt((n-3)/(1-r**2))), n-3)
lagrows = []; fig, ax = plt.subplots(2, len(names), figsize=(3.3*len(names), 6.4), sharey=True)
for j, g in enumerate(names):
    for i, mn in enumerate(["Sep volume", "Aug-Nov min 7-d"]):
        s = TARG[(g, mn)]; vals = []; ps = []; ns_ = []
        for k in range(4):
            x = P_wy.reindex(s.index-k).values; d = pd.DataFrame({"q": s.values, "x": x}).dropna(); r, p = spearmanr(d.x, d.q); vals.append(r); ps.append(p); ns_.append(len(d)); lagrows.append(dict(gauge=g, metric=mn, lag=k, n=len(d), rho=r, p=p))
        d = pd.DataFrame({"q": s.values, "x0": P_wy.reindex(s.index).values, "x1": P_wy.reindex(s.index-1).values}).dropna(); pr, pp = partial(d.x1.values, d.q.values, d.x0.values) if len(d) > 5 else (np.nan, np.nan); lagrows.append(dict(gauge=g, metric=mn, lag="1|0 partial", n=len(d), rho=pr, p=pp))
        a = ax[i, j]; a.bar(range(4), vals, color=["#d55e00" if p < 0.05 else "#2a78d6" for p in ps], width=.6); a.bar([4.3], [pr], color="#d55e00" if pp < 0.05 else "#56b4e9", width=.6, hatch="//"); a.axhline(0, color="#52514e", lw=.8); a.set_ylim(-1, 1)
        a.set_xticks([0, 1, 2, 3, 4.3]); a.set_xticklabels(["0", "1", "2", "3", "1|0"], fontsize=8); a.set_title(f"{g} — {mn}", fontsize=8.5, loc="left")
        for k in range(4): a.text(k, vals[k]+(0.06 if vals[k] >= 0 else -0.13), f"n={ns_[k]}", ha="center", fontsize=6.5, color="#52514e")
        if j == 0: a.set_ylabel("ρ with WY precip (y−lag)")
        if i == 1: a.set_xlabel("lag (water years); 1|0 = lag-1 partial on same-year")
fig.suptitle("Low flow vs PRISM water-year precipitation at lags 0–3 (orange p<0.05; hatched = lag-1 partial controlling for same-year precip)", y=1.0); fig.tight_layout(); fig.savefig(D+"tuo_lowflow_precip_lags.png", dpi=140, bbox_inches="tight")
pd.DataFrame(lagrows).to_csv(D+"tuo_lowflow_precip_lags.csv", index=False)
pa = [dict(lag=k, rho=spearmanr(P_wy.values[k:], P_wy.values[:len(P_wy)-k])[0] if k else 1.0) for k in range(1, 4)]
print("PRISM WY precip", P_wy.index.min(), P_wy.index.max(), len(P_wy), "| mean %.0f mm" % P_wy.mean(), "| precip ACF lag1-3:", [round(x["rho"], 2) for x in pa])
print("PRISM vs TUM obs WY precip: rho=%.2f n=%d" % (spearmanr(*[pd.concat([P_wy, tum], axis=1).dropna().iloc[:, i] for i in (0, 1)])[0], len(pd.concat([P_wy, tum], axis=1).dropna())))
print("ONI vs WY precip: rho=%.2f; PDO vs WY precip: rho=%.2f" % (spearmanr(*[pd.concat([P_wy, ONI], axis=1).dropna().iloc[:, i] for i in (0, 1)])[0], spearmanr(*[pd.concat([P_wy, PDO], axis=1).dropna().iloc[:, i] for i in (0, 1)])[0]))
pd.set_option("display.width", 250); print(R[R.metric == "Aug-Nov min 7-d"].pivot(index="gauge", columns="predictor", values="rho").reindex(names)[list(PRED.columns)].round(2).to_string()); print(R[R.metric == "Sep volume"].pivot(index="gauge", columns="predictor", values="rho").reindex(names)[list(PRED.columns)].round(2).to_string())
L = pd.DataFrame(lagrows); print(L[L.metric == "Sep volume"].pivot(index="gauge", columns="lag", values="rho").round(2)); print(L[L.metric == "Aug-Nov min 7-d"].pivot(index="gauge", columns="lag", values="rho").round(2))
