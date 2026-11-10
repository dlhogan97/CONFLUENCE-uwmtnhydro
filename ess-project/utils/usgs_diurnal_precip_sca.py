"""USGS 11274790 diurnal cycle (May-Sep) with precipitation-affected days removed, and its relation to basin snow-covered area (Landsat).
Wet day = PRISM-based basin precipitation (METSIM forcing) >= 0.5 mm OR CDEC TUM gauge daily increment >= 0.5 mm. Screened set = days with NO wet day on d-1, d or d+1 (and a clear 24-h cycle, power fraction >= 0.3).
Snow-covered area: Landsat scenes 2013-2025 (landsat_spring_scenes.csv + scenes_by_band3.csv merged), area-weighted fSCA over valid elevation bands (shadow-inclusive), daily by linear interpolation (gaps <= 24 d), non-increasing after the seasonal peak (cloud / new-snow spikes ignored).
-> figures/draft/usgs_diurnal_precip_sca.png, usgs_diurnal_precip_screened_days.csv, usgs_diurnal_sca_transition.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, theilslopes, binomtest, ttest_1samp
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"; SC = "/scratch/dlhogan/ess-project-data/geospatial-data/"
days = pd.read_csv(D+"usgs_diurnal_hour_maysep_v2_days.csv", parse_dates=["date"]); days = days[days.power_frac >= 0.3].set_index("date").sort_index()
met = pd.read_csv(OBS+"tuolumne_daily_met.csv", index_col=0, parse_dates=True); tum = pd.read_csv(OBS+"tum_daily_precip.csv", index_col=0, parse_dates=True)
def wetflags(thr_prism=0.5, thr_tum=0.5, use_tum=True):
    w = (met.P_basin >= thr_prism)
    if use_tum: w = w | (tum.increment_mm.reindex(met.index) >= thr_tum)
    w = w.fillna(False); return (w | w.shift(1, fill_value=False) | w.shift(-1, fill_value=False))
wet3 = wetflags(); days["wet3"] = wet3.reindex(days.index).values; S = days[~days.wet3].copy()
print("clear days:", len(days), "-> precip-screened:", len(S), "(removed %.0f%%)" % (100*(1-len(S)/len(days)))); print("screened days by month:", S.groupby(S.index.month).size().to_dict(), "| all clear:", days.groupby(days.index.month).size().to_dict())
def circ(h, w): a = np.array(h)*2*np.pi/24; return (np.degrees(np.arctan2((w*np.sin(a)).sum(), (w*np.cos(a)).sum()))/15) % 24
def conc(h, w): a = np.array(h)*2*np.pi/24; return np.hypot((w*np.sin(a)).sum(), (w*np.cos(a)).sum())/w.sum()
def windows(x): 
    g = ((x.index.dayofyear-121)//7)*7+121+3; return x.groupby(g).apply(lambda z: pd.Series(dict(n=len(z), hour=circ(z.hour_max, z.amp_abs.values), R=conc(z.hour_max, z.amp_abs.values), amp_abs=z.amp_abs.median(), amp_rel=z.amp_rel.median(), q=z.q_mean.median())))
W0, W1 = windows(days), windows(S); cmp_ = W0[["n", "hour", "R"]].join(W1[["n", "hour", "R"]], lsuffix="_all", rsuffix="_screened"); cmp_["date"] = (pd.Timestamp("2021-01-01")+pd.to_timedelta(cmp_.index-1, unit="D")).strftime("%b %d"); pd.set_option("display.width", 200); print(cmp_.round(2).to_string())
def flowclass(x):
    out = []
    for lo, hi in [(-1, 0), (0, .5), (.5, .8), (.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.5)]:
        z = x[(np.log10(x.q_mean) >= lo) & (np.log10(x.q_mean) < hi)]
        if len(z) >= 8: out.append(dict(flow=f"{10**lo:.1f}-{10**hi:.0f}", n=len(z), hour=circ(z.hour_max, z.amp_abs.values), R=conc(z.hour_max, z.amp_abs.values), amp_abs=z.amp_abs.median(), amp_rel=z.amp_rel.median()))
    return pd.DataFrame(out)
print("\nhour of max by flow class: ALL clear days"); print(flowclass(days).round(2).to_string(index=False)); print("\nhour of max by flow class: PRECIP-SCREENED"); print(flowclass(S).round(2).to_string(index=False))
# night-regime drift May1-Jul10 screened
S["u"] = ((S.hour_max-2+12) % 24)-12+2; nd = S[(S.doy <= 191) & (S.u > -6) & (S.u < 7)]; sl = []
for y, x in nd.groupby("year"):
    if len(x) >= 15: sl.append(theilslopes(x.u.values, x.doy.values)[0]*30)
sl = np.array(sl); print(f"\nnight-regime drift (screened), {len(sl)} years: median {np.median(sl):+.2f} h/30d, later in {int((sl>0).sum())}/{len(sl)}, sign p={binomtest(int((sl>0).sum()), len(sl), .5).pvalue:.3f}, t-test p={ttest_1samp(sl, 0).pvalue:.3f}")
S.to_csv(D+"usgs_diurnal_precip_screened_days.csv")
# sensitivity of the screen
for lab, w in (("any PRISM >=0.1mm or TUM>=0.25mm", wetflags(0.1, 0.25)), ("PRISM >=2mm or TUM>=2mm", wetflags(2, 2)), ("PRISM only >=0.5mm", wetflags(0.5, use_tum=False))):
    z = days[~w.reindex(days.index).values]; print(f"sensitivity [{lab}]: {len(z)} days; hour by flow class:", flowclass(z).set_index("flow").hour.round(1).to_dict())
# ---------------- snow covered area: daily series built by tuo_sca_daily_series.py (Landsat + ASO anchors, melt-axis interpolation)
ATOT = 774.5; scadf = pd.read_csv(OBS+"tuolumne_sca_daily_2013_2025.csv", index_col=0, parse_dates=True); sca = scadf.sca; gapd = scadf.gap_days; SCA = {y: g.sca for y, g in scadf.groupby("year")}
S["sca"] = sca.reindex(S.index).values; S["gap"] = gapd.reindex(S.index).values; S["T_high"] = met.T_high.reindex(S.index).values; S["SW_high"] = met.SW_high.reindex(S.index).values; Sy = S.dropna(subset=["sca"]).copy(); Sy["sca_km2"] = Sy.sca*ATOT; Sy30 = Sy[Sy.gap <= 30].copy()
print("screened USGS days with SCA:", len(Sy), "(of which bracketing gap <=30 d:", len(Sy30), ") years", sorted(Sy.year.unique()))
for a_, b_ in (("amp_abs", "sca"), ("amp_rel", "sca"), ("u", "sca"), ("q_mean", "sca")): print(f"   Spearman {a_:8s} vs SCA (pooled days): rho={spearmanr(Sy[a_], Sy[b_])[0]:+.2f}")
for lab, Z in (("all SCA days", Sy), ("gap<=30 d only", Sy30)):
    yv_ = np.log10(Z.amp_abs); lq_ = np.log10(Z.q_mean); ls_ = np.log10(Z.sca_km2.clip(lower=1))
    for cl_, cols in (("SCA", [ls_]), ("Q", [lq_]), ("SCA+Q", [ls_, lq_])):
        X_ = np.column_stack([np.ones(len(Z))]+cols); b_ = np.linalg.lstsq(X_, yv_, rcond=None)[0]; r_ = yv_-X_@b_; print(f"   [{lab}] log10 amp_abs ~ log10 {cl_}: coef {np.round(b_[1:], 2)} R2={1-r_.var()/yv_.var():.3f} (n={len(Z)})")
    print(f"   [{lab}] Spearman: amp~SCA {spearmanr(Z.amp_abs, Z.sca)[0]:+.2f}, amp~Q {spearmanr(Z.amp_abs, Z.q_mean)[0]:+.2f}, hour(u)~SCA {spearmanr(Z.u, Z.sca)[0]:+.2f}, hour(u)~Q {spearmanr(Z.u, Z.q_mean)[0]:+.2f}, SCA~Q {spearmanr(Z.sca, Z.q_mean)[0]:+.2f}")
X = np.column_stack([np.ones(len(Sy)), np.log10(Sy.sca_km2.clip(lower=1)), Sy.SW_high, Sy.T_high]); yv = np.log10(Sy.amp_abs); b, *_ = np.linalg.lstsq(X, yv, rcond=None); res = yv-X@b; print(f"   log10 amp_abs ~ log10 SCA_km2 + SW + T : coef {np.round(b[1:], 3)}, R2={1-res.var()/yv.var():.2f}")
Xq = np.column_stack([np.ones(len(Sy)), np.log10(Sy.q_mean)]); bq, *_ = np.linalg.lstsq(Xq, yv, rcond=None); rq = yv-Xq@bq; print(f"   log10 amp_abs ~ log10 Q alone: slope {bq[1]:.2f}, R2={1-rq.var()/yv.var():.2f}; ~ log10 SCA alone R2={np.corrcoef(np.log10(Sy.sca_km2.clip(lower=1)), yv)[0,1]**2:.2f}")
# transition per year
tr = []
for y, x in S.groupby("year"):
    x = x.sort_values("date"); u = pd.Series(x.u.values, index=x.index); qq = pd.Series(x.q_mean.values, index=x.index); r = u.rolling("9D", min_periods=3).median(); cand = r[(r >= 7) & (r.index >= pd.Timestamp(y, 5, 15))]
    for t in cand.index:
        nxt = u[(u.index > t) & (u.index <= t+pd.Timedelta(days=14))]
        if len(nxt) >= 3 and nxt.median() >= 7: tr.append(dict(year=y, transition=t, doy=t.dayofyear, q_med=float(qq[(qq.index >= t-pd.Timedelta(days=4)) & (qq.index <= t+pd.Timedelta(days=4))].median()), sca=float(sca.get(t, np.nan)), gap=float(gapd.get(t, np.nan)))); break
T = pd.DataFrame(tr); T.to_csv(D+"usgs_diurnal_sca_transition.csv", index=False); print("\ntransition from night-time to midday peak (screened days):"); print(T.round(3).to_string(index=False))
for c in ("doy", "q_med", "sca"): z = T[c].dropna(); print(f"   {c:6s} at transition: median {z.median():.2f}, CV {z.std()/z.mean():.2f} (n={len(z)})")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(2, 3, figsize=(19, 9)); xt = [121, 152, 182, 213, 244, 274]; xl = ["May 1", "Jun 1", "Jul 1", "Aug 1", "Sep 1", "Oct 1"]
a = ax[0, 0]; a.plot(W0.index, W0.hour, "o-", color="#999", label=f"all clear days (n={len(days)})"); a.plot(W1.index, W1.hour, "o-", color="#d55e00", label=f"precip-screened (n={len(S)})"); a.set_xticks(xt); a.set_xticklabels(xl); a.set_yticks(range(0, 25, 3)); a.set_ylabel("hour of max (7-day circular mean)"); a.legend(frameon=False, fontsize=8); a.set_title("Hour of maximum flow: effect of removing wet days", fontsize=10, loc="left"); a.grid(color="#e3e2de", lw=.5)
a = ax[0, 1]; sc_ = a.scatter(S.doy, S.hour_max, c=np.log10(S.q_mean), cmap="viridis", s=6, alpha=.6); a.set_xticks(xt); a.set_xticklabels(xl); a.set_yticks(range(0, 25, 3)); a.set_title("Screened days, coloured by log10 flow", fontsize=10, loc="left"); fig.colorbar(sc_, ax=a, label="log10 Q")
a = ax[0, 2]; fc0, fc1 = flowclass(days), flowclass(S); a.plot(range(len(fc0)), fc0.hour, "o-", color="#999", label="all"); a.plot(range(len(fc1)), fc1.hour, "o-", color="#d55e00", label="screened"); a.set_xticks(range(len(fc0))); a.set_xticklabels(fc0.flow, rotation=45, fontsize=7); a.set_xlabel("daily flow class (m³/s)"); a.set_ylabel("hour of max (circular mean)"); a.set_yticks(range(0, 25, 3)); a.legend(frameon=False, fontsize=8); a.set_title("Hour of maximum by flow", fontsize=10, loc="left"); a.grid(color="#e3e2de", lw=.5)
a = ax[1, 0]; sc_ = a.scatter(Sy.sca_km2, Sy.amp_abs, c=np.log10(Sy.q_mean), cmap="viridis", s=7, alpha=.6); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("basin snow-covered area (km², interpolated Landsat)"); a.set_ylabel("absolute 24-h amplitude (m³/s)"); a.set_title(f"Diurnal amplitude vs snow-covered area (ρ={spearmanr(Sy.amp_abs, Sy.sca)[0]:+.2f}, {len(Sy)} days, {Sy.year.nunique()} yr)", fontsize=10, loc="left")
a = ax[1, 1]; sc_ = a.scatter(Sy.sca_km2, Sy.hour_max, c=np.log10(Sy.q_mean), cmap="viridis", s=7, alpha=.6); a.set_xscale("log"); a.set_yticks(range(0, 25, 3)); a.set_xlabel("basin snow-covered area (km²)"); a.set_ylabel("hour of max flow"); a.set_title(f"Hour of maximum vs snow-covered area", fontsize=10, loc="left")
a = ax[1, 2]
for y, f in SCA.items(): a.plot(f.index.dayofyear, f.values*100, color=plt.cm.viridis((y-2013)/12), lw=1)
for r in T.itertuples():
    if r.year in SCA and pd.notna(r.sca): a.plot(r.doy, r.sca*100, "k|", ms=14, mew=2)
a.set_xticks(xt); a.set_xticklabels(xl); a.set_ylabel("basin snow-covered fraction (%)"); a.set_title("Landsat SCA by year (| = night→midday transition date)", fontsize=10, loc="left"); a.grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+"usgs_diurnal_precip_sca.png", dpi=140, bbox_inches="tight")
