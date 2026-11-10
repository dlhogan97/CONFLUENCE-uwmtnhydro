"""
Snowpatch carryover vs next-year low flow (2026-11-10, user: "lag-1 storage change with low flows"). Full-record wired runs (g35_suite7_full.py).
Per water year, bins (HRU 2 + 29-37) area-weighted, in BASIN mm (x HRU-2 area / basin area):
  S_oct1 = bin SWE on 1 Oct (carryover into the WY), dS = S_sep30 - S_oct1, AugSep_out = bins' scalarTotalRunoff Aug-Sep,
  added = patch_wired - control_wired outlet Aug-Sep, obs / cw / pw Aug-Sep outlet flow.
Lag-1 question: does last year's storage gain (dS[y-1]) or carryover (S_oct1[y]) set this year's low-flow addition?
-> figures/draft/tuo_patch_lag.csv, tuo_patch_lag.png
"""
import numpy as np, pandas as pd, netCDF4 as nc, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import g35_suite7_full as G, g35_suite7_full_table as T, g35_suite4_snowpatch as SP
FIG = T.FIG
A = nc.Dataset("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")
ah = np.array(A["hruId"][:]).ravel().astype(int); aa = np.array(A["HRUarea"][:]).ravel(); FRAC = aa[ah == 2][0]/aa.sum()


def bins():
    d = nc.Dataset(G.ROOT/"patch_wired"/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    hid = np.array(d["hruId"][:]).ravel().astype(int); b = np.isin(hid, SP.BINS); w = np.array([SP.AREA[h] for h in hid[b]]); w = w/w.sum()
    swe = pd.Series(np.array(d["scalarSWE_mean"][:])[:, b] @ w, index=t)*FRAC
    ro = pd.Series(np.array(d["scalarTotalRunoff_mean"][:])[:, b] @ w*86400e3, index=t)*FRAC
    deep = pd.DataFrame(np.array(d["scalarSWE_mean"][:])[:, b]/1000, index=t, columns=hid[b]); d.close(); return swe, ro, deep


def main():
    swe, ro, deep = bins(); q = pd.DataFrame({n: T.sim(n) for n in ("control_wired", "patch_wired")}); q["obs"] = T.obs_mm_full()
    q = q["1982-10-01":"2025-09-30"]; m = q.index.month; wy = q.index.year + (m >= 10); a = m.isin([8, 9])
    Y = q[a].groupby(wy[a]).sum().rename(columns={"control_wired": "cw", "patch_wired": "pw"}); Y["added"] = Y.pw - Y.cw
    Y["obs_ann"] = q.obs.groupby(wy).sum()
    s1 = swe[swe.index.strftime("%m-%d") == "10-01"]; s1.index = s1.index.year + 1; Y["S_oct1"] = s1
    se = swe[swe.index.strftime("%m-%d") == "09-30"]; se.index = se.index.year; Y["S_sep30"] = se; Y["dS"] = Y.S_sep30 - Y.S_oct1
    ra = ro[ro.index.month.isin([8, 9])]; Y["bin_AugSep_out"] = ra.groupby(ra.index.year).sum()
    dp = deep[deep.index.strftime("%m-%d") == "09-30"]; dp.index = dp.index.year; Y["deepest_bin_m_sep30"] = dp.max(axis=1)
    Y["dS_prev"] = Y.dS.shift(1); Y["obs_ann_prev"] = Y.obs_ann.shift(1); Y["gap_cw"] = Y.obs - Y.cw; Y["gap_closed_frac"] = Y.added/Y.gap_cw
    Y.to_csv(FIG + "tuo_patch_lag.csv"); pd.set_option("display.width", 250)
    print(f"HRU2 area frac {FRAC:.4f}; basin mm per bin-mean m SWE = {FRAC*1000:.1f}")
    print(Y[["obs_ann", "obs", "cw", "pw", "added", "S_oct1", "dS", "dS_prev", "bin_AugSep_out", "deepest_bin_m_sep30"]].round(2).to_string())
    Z = Y.dropna(subset=["dS_prev"])
    def sp(x, y): return Z[[x, y]].corr(method="spearman").iloc[0, 1]
    print("\nSpearman:")
    for x in ("dS_prev", "S_oct1", "obs_ann_prev", "dS", "obs_ann"):
        print(f"  {x:13s} vs added {sp(x,'added'):+.2f}   vs obs AugSep {sp(x,'obs'):+.2f}   vs gap(obs-cw) {sp(x,'gap_cw'):+.2f}")
    up, dn = Z[Z.dS_prev > 0], Z[Z.dS_prev <= 0]
    print(f"\nafter a storage GAIN (n={len(up)}): added {up.added.mean():.2f}, obs {up.obs.mean():.1f}, cw {up.cw.mean():.1f}, pw {up.pw.mean():.1f}")
    print(f"after a storage LOSS (n={len(dn)}): added {dn.added.mean():.2f}, obs {dn.obs.mean():.1f}, cw {dn.cw.mean():.1f}, pw {dn.pw.mean():.1f}")
    f, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(Y.index, Y.S_oct1, "k-", label="bin SWE 1 Oct (basin mm)"); ax[0].bar(Y.index, Y.dS, color=np.where(Y.dS > 0, "#2a6fb0", "#c0504d"), alpha=.6, label="ΔS over WY")
    ax[0].set_title("snowpatch storage (basin mm)"); ax[0].legend(fontsize=8); ax[0].axhline(0, c="0.6", lw=.6)
    ax[1].scatter(Z.dS_prev, Z.added, c="k", s=18); ax[1].set_xlabel("ΔS previous WY (basin mm)"); ax[1].set_ylabel("Aug–Sep added at outlet (mm)")
    ax[1].set_title(f"lag-1: ρ = {sp('dS_prev','added'):+.2f}")
    ax[2].scatter(Z.dS_prev, Z.gap_cw, c="0.5", s=18, label="obs − model (no patch)"); ax[2].scatter(Z.dS_prev, Z.added, c="#2a6fb0", s=18, label="added by patch")
    ax[2].set_xlabel("ΔS previous WY (basin mm)"); ax[2].set_ylabel("Aug–Sep (mm)"); ax[2].set_title(f"vs low-flow gap: ρ = {sp('dS_prev','gap_cw'):+.2f}"); ax[2].legend(fontsize=8)
    ax[2].axhline(0, c="0.6", lw=.6); f.tight_layout(); f.savefig(FIG + "tuo_patch_lag.png", dpi=130); print(FIG + "tuo_patch_lag.png")


if __name__ == "__main__":
    main()
