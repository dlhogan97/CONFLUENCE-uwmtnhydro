"""
Summarise g35_suite5_ground.py runs (2026-11-05): soil temperature / ice under snow by elevation band and month, and winter baseflow + scores vs observed.
-> figures/draft/tuo_suite5_ground.csv
usage: python g35_suite5_ground_table.py
"""
import json, numpy as np, pandas as pd, netCDF4 as nc
import g35_suite5_cal as C5
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
A = nc.Dataset("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")
area = np.array(A["HRUarea"][:]).ravel(); elev = np.array(A["elevation"][:]).ravel(); asp = np.array(A["aspect"][:]).ravel()
NORTH = (asp >= 300) | (asp <= 60); SOUTH = (asp >= 120) & (asp <= 240)                   # aspect groups (2026-11-05: frozen HRUs are the north-facing ones)
GROUPS = {"north <2900": NORTH & (elev < 2900), "south <2900": SOUTH & (elev < 2900), "all >2900": elev >= 2900}
BANDS = [("<2300", 0, 2300), ("2300-2900", 2300, 2900), (">2900", 2900, 5000)]
obs = C5.C4.obs_mm()["2012-10-01":"2020-09-30"]; base = lambda q: q.rolling(15, center=True, min_periods=8).min()
runs = {p.name.replace("lbt_", ""): p for p in sorted((C5.OUT/"ground").glob("lbt_*")) if (p/"cal_day.nc").exists() and (p/"cal_timestep.nc").exists()}
soil, flow, grp = {}, {}, {}
for k, p in runs.items():
    d = nc.Dataset(p/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    h = nc.Dataset(p/"cal_timestep.nc"); th = pd.DatetimeIndex(nc.num2date(h["time"][:], h["time"].units, only_use_cftime_datetimes=False))
    k0 = np.where((th >= pd.Timestamp("2012-10-01")) & (th.hour == 12))[0]; sl = slice(k0[0], k0[-1] + 1, 24); tt = th[sl]   # one noon sample per day
    ns = np.array(h["nSnow"][sl]).astype(int); T = np.array(h["mLayerTemp"][sl]); I = np.array(h["mLayerVolFracIce"][sl]); h.close()
    idx = ns[:, None, :] + np.arange(3)[None, :, None]                                    # soil layers follow the snow layers in midToto
    Ts = np.take_along_axis(T, idx, axis=1).mean(1) - 273.15; Is = np.take_along_axis(I, idx, axis=1).mean(1); Ttop = np.take_along_axis(T, idx[:, :1], axis=1)[:, 0] - 273.15
    for b, lo, hi in BANDS:
        m = (elev >= lo) & (elev < hi); w = area[m]/area[m].sum()
        for mo in [10, 11, 12, 1, 2, 3, 4]:
            s = tt.month == mo
            soil[(k, b, mo)] = dict(T_soil=float((Ts[s][:, m]@w).mean()), T_top=float((Ttop[s][:, m]@w).mean()), ice=float((Is[s][:, m]@w).mean()),
                                    frozen_days=float(((Ttop[s][:, m] < -0.05)@w).mean()*100))
    dj = np.isin(tt.month, [12, 1, 2])
    for gname, m in GROUPS.items():
        w = area[m]/area[m].sum()
        grp[(k, gname)] = dict(T_top_DJF=float((Ttop[dj][:, m]@w).mean()), ice_top_DJF=float((np.take_along_axis(I, idx[:, :1], axis=1)[:, 0][dj][:, m]@w).mean()),
                               ice_top_Mar=float((np.take_along_axis(I, idx[:, :1], axis=1)[:, 0][tt.month == 3][:, m]@w).mean()))
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=t)["2012-10-01":"2020-09-30"]; b_ = base(q); mo_ = q.index.month
    sc = json.loads((p/"score.json").read_text()); g = sc["diagnostics"]
    flow[k] = dict(objective=sc["objective"], **{c: g[c] for c in ("KGE_AprJul", "AprMay_bias", "JunJul_bias", "AugSep_bias", "aso_swe_nmae", "band_order_mae")},
                   **{f"base {l}": b_[mo_ == mm].mean() for mm, l in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]}, DecMar_mm=q[np.isin(mo_, [12, 1, 2, 3])].sum()/8)
bo = base(obs); mo_ = obs.index.month
flow["observed"] = {**{f"base {l}": bo[mo_ == mm].mean() for mm, l in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]}, "DecMar_mm": obs[np.isin(mo_, [12, 1, 2, 3])].sum()/8}
S = pd.DataFrame(soil).T; S.index.names = ["run", "band", "month"]; pd.set_option("display.width", 260)
print("top-soil-layer temperature (C), monthly mean"); print(S.T_top.unstack("month")[[10, 11, 12, 1, 2, 3, 4]].round(1).to_string())
print("\n% of days top soil layer below -0.05 C"); print(S.frozen_days.unstack("month")[[10, 11, 12, 1, 2, 3, 4]].round(0).to_string())
print("\nsoil ice (volumetric, 3-layer mean)"); print(S.ice.unstack("month")[[11, 12, 1, 2, 3, 4]].round(3).to_string())
G = pd.DataFrame(grp).T; G.index.names = ["run", "group"]; print("\nDec-Feb top-layer temperature (C) and top-layer ice by aspect group")
print(G.unstack("group").round(3).to_string())
F = pd.DataFrame(flow).T; print("\n", F.round(3).to_string()); F.to_csv(FIG + "tuo_suite5_ground.csv")
