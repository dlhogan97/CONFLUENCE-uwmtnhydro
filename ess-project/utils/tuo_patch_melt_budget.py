"""Feasibility water budget: could perennial snow patches supply a meaningful share of Aug-Sep streamflow at the basin (USGS 11274790, the 774.5 km2 calibration domain)?
Snow area A(t): Landsat shadow-inclusive basin snow area per scene (sum_b fSCA_b*Area_b), Aug 1 - Oct 10; daily series = linear interpolation of the running minimum (patch area cannot grow in Aug-Sep; new-snow/cloud spikes ignored), flat before the first / after the last scene.
Melt rate m(t) (mm SWE/day per unit patch area) at the high-elevation HRUs (3680 m) of the METSIM forcing, three scenarios: (a) degree-day 4 mm/C/d, (b) degree-day 8 mm/C/d, (c) energy limit (1-alpha)*SW/(rho*Lf), alpha=0.5, only when T>0 C (an UPPER bound: ignores longwave/turbulent losses).
V_melt = sum_t m(t)*A(t) over Aug 1 - Sep 30; V_flow = sum of USGS daily Q over the same days. ratio = V_melt/V_flow (>1 means the budget cannot be closed even if all melt reached the gauge; 0.05 means patches are a few % at most).
-> figures/draft/tuo_patch_melt_budget.png, tuo_patch_melt_budget.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
SC = "/scratch/dlhogan/ess-project-data/"; FS = SC+"geospatial-data/snowpatch_fsca/"; U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"
BA = pd.read_csv(SC+"geospatial-data/tuolumne_decmar_precip_bands.csv").set_index("elevClass")["area_km2"].to_dict(); BANDS = [8, 7, 6, 5, 4, 3, 2]
sc = pd.read_csv(FS+"scenes_by_band3.csv", parse_dates=["date"]); sc = sc[(sc.status == "ok") & (sc.valid_sh >= 0.35) & sc[["fsca_b8_sh", "fsca_b7_sh"]].notna().all(axis=1)]
sc["A"] = sum(sc[f"fsca_b{b}_sh"].fillna(0)*BA[b] for b in BANDS); sc = sc[(sc.date.dt.month >= 7) & ((sc.date.dt.month < 10) | (sc.date.dt.day <= 10))]
met = pd.read_csv(U+"modeling/calibration/observations/tuolumne_daily_met.csv", index_col=0, parse_dates=True)
usgs = pd.read_csv(SC+"domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_USGS11274790_dailyQ_WY2007_2025.csv", parse_dates=["datetime"]).set_index("datetime")["discharge_cms"]
eos = pd.read_csv(FS+"eos_final.csv").set_index("year"); rows = []; series = {}
for y in range(1985, 2026):
    s = sc[sc.year == y].sort_values("date")
    if len(s) < 2: continue
    d0, d1 = pd.Timestamp(y, 8, 1), pd.Timestamp(y, 9, 30); s = s[s.date >= pd.Timestamp(y, 7, 25)]
    A = pd.Series(s.A.values, index=s.date).cummin(); idx = pd.date_range(d0, d1); Ad = A.reindex(A.index.union(idx)).interpolate("time").reindex(idx).bfill().ffill()
    m = met.reindex(idx); T = m.T_high; SW = m.SW_high; rate = {"DDF4": 4*np.clip(T, 0, None), "DDF8": 8*np.clip(T, 0, None), "SWenergy": np.where(T > 0, 0.5*SW*86400/3.34e5, 0.0)}
    rec = dict(year=y, n_scenes=len(s), A_aug1=float(Ad.iloc[0]), A_sep30=float(Ad.iloc[-1]), A_min_eos=float(eos.A_shadow.get(y, np.nan)), Thigh_AugSep=float(T.mean()), SWhigh_AugSep=float(SW.mean()))
    for k, r in rate.items():
        r = pd.Series(r, index=idx); rec[f"melt_depth_{k}_mm"] = float(r.sum()); rec[f"V_{k}_Mm3"] = float((r*Ad).sum()/1e3)      # mm * km2 = 1e3 m3
    q = usgs.reindex(idx); rec["n_flow_days"] = int(q.notna().sum()); rec["V_flow_Mm3"] = float(q.mean()*len(idx)*86400/1e6) if q.notna().sum() >= 55 else np.nan
    for k in rate: rec[f"ratio_{k}"] = rec[f"V_{k}_Mm3"]/rec["V_flow_Mm3"] if rec["V_flow_Mm3"] == rec["V_flow_Mm3"] else np.nan
    rows.append(rec); series[y] = Ad
B = pd.DataFrame(rows); B.to_csv(D+"tuo_patch_melt_budget.csv", index=False); pd.set_option("display.width", 220)
print(B[["year", "A_aug1", "A_sep30", "A_min_eos", "melt_depth_DDF4_mm", "melt_depth_SWenergy_mm", "V_DDF4_Mm3", "V_DDF8_Mm3", "V_SWenergy_Mm3", "V_flow_Mm3", "ratio_DDF4", "ratio_DDF8", "ratio_SWenergy"]].round(2).to_string(index=False))
g = B.dropna(subset=["V_flow_Mm3"]); print("\nyears with flow:", len(g)); print(g[["ratio_DDF4", "ratio_DDF8", "ratio_SWenergy"]].describe().round(2))
for k in ("DDF4", "DDF8", "SWenergy"): print(k, "rank corr V_melt vs V_flow: rho=%.2f p=%.3f n=%d" % (*spearmanr(g[f"V_{k}_Mm3"], g.V_flow_Mm3), len(g)))
print("rank corr A_aug1 vs V_flow: rho=%.2f p=%.3f" % spearmanr(g.A_aug1, g.V_flow_Mm3), "| A_sep30:", "rho=%.2f p=%.3f" % spearmanr(g.A_sep30, g.V_flow_Mm3))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]; a.fill_between(B.year, B.V_DDF4_Mm3, B.V_SWenergy_Mm3, color="#2a78d6", alpha=.25, label="patch melt capacity: degree-day 4 → energy limit"); a.plot(B.year, B.V_DDF8_Mm3, color="#2a78d6", lw=1.2, label="degree-day 8"); a.plot(B.year, B.V_flow_Mm3, "o-", color="#0b0b0b", ms=4, label="USGS flow, Aug-Sep"); a.set_yscale("log"); a.set_ylabel("volume (10⁶ m³, log)"); a.legend(frameon=False, fontsize=7.5); a.set_title("Aug–Sep: patch melt capacity vs observed basin flow", fontsize=9.5, loc="left")
a = ax[1]; a.plot(B.year, B.ratio_DDF4, "o-", ms=4, color="#56b4e9", label="degree-day 4"); a.plot(B.year, B.ratio_DDF8, "o-", ms=4, color="#2a78d6", label="degree-day 8"); a.plot(B.year, B.ratio_SWenergy, "o-", ms=4, color="#d55e00", label="energy limit"); a.axhline(1, color="#999", lw=.8); a.set_yscale("log"); a.set_ylabel("V_melt / V_flow"); a.legend(frameon=False, fontsize=7.5); a.set_title("Share of Aug–Sep flow the patches could supply", fontsize=9.5, loc="left")
a = ax[2]; a.scatter(g.V_DDF8_Mm3, g.V_flow_Mm3, c=g.year, cmap="viridis", s=40); [a.annotate(str(int(r.year))[2:], (r.V_DDF8_Mm3, r.V_flow_Mm3), fontsize=7, xytext=(3, 3), textcoords="offset points") for r in g.itertuples()]; a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("patch melt capacity, degree-day 8 (10⁶ m³)"); a.set_ylabel("USGS Aug–Sep flow (10⁶ m³)"); r_, p_ = spearmanr(g.V_DDF8_Mm3, g.V_flow_Mm3); a.set_title(f"ρ={r_:+.2f} (p={p_:.2f}, n={len(g)})", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(D+"tuo_patch_melt_budget.png", dpi=140, bbox_inches="tight")
