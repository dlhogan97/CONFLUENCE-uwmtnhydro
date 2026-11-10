"""
Compare the round-5 control (OUT/suite5_cal/energy) with the cloud-LW runs (OUT/suite5_cal/cloudlw/s*), 2026-11-06.
Per group (north <2900 m, south <2900 m, 2900-3500 m, >3500 m), Dec-Feb snow-covered hours: incoming LW, net LW at the ground, surface minus air, snow base,
whole-pack mean, top-soil temperature and top-soil ice. Then flow/scores and winter baseflow (15-day rolling min) vs observed, and Apr 1 SWE by group.
-> figures/draft/tuo_suite5_cloudlw.csv
usage: python g35_suite5_cloudlw_table.py
"""
import json, numpy as np, pandas as pd, netCDF4 as nc
import g35_suite5_cal as C5
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
A = nc.Dataset("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")
area = np.array(A["HRUarea"][:]).ravel(); el = np.array(A["elevation"][:]).ravel(); asp = np.array(A["aspect"][:]).ravel()
N = (asp >= 300) | (asp <= 60); S = (asp >= 120) & (asp <= 240)
GROUPS = {"N <2900": N & (el < 2900), "S <2900": S & (el < 2900), "2900-3500": (el >= 2900) & (el < 3500), ">3500": el >= 3500}
RUNS = {"control": C5.OUT/"energy"} | {p.name: p for p in sorted((C5.OUT/"cloudlw").glob("s*")) if (p/"score.json").exists()}
obs = C5.C4.obs_mm()["2012-10-01":"2020-09-30"]; base = lambda q: q.rolling(15, center=True, min_periods=8).min()
E, F, W = [], {}, {}
for name, p in RUNS.items():
    h = nc.Dataset(p/"cal_timestep.nc"); th = pd.DatetimeIndex(nc.num2date(h["time"][:], h["time"].units, only_use_cftime_datetimes=False))
    k = np.where(th >= pd.Timestamp("2012-10-01"))[0]; sl = slice(k[0], k[-1] + 1); tt = th[sl]; g = lambda v: np.array(h[v][sl]).astype(float)
    ns = g("nSnow").astype(int); T = np.array(h["mLayerTemp"][sl]) - 273.15; I = np.array(h["mLayerVolFracIce"][sl])
    sb = np.take_along_axis(T, np.maximum(ns - 1, 0)[:, None, :], 1)[:, 0]; st = np.take_along_axis(T, ns[:, None, :], 1)[:, 0]; si = np.take_along_axis(I, ns[:, None, :], 1)[:, 0]
    cs = np.cumsum(np.where(np.arange(T.shape[1])[None, :, None] < ns[:, None, :], T, 0), axis=1); bulk = np.take_along_axis(cs, np.maximum(ns - 1, 0)[:, None, :], 1)[:, 0]/np.maximum(ns, 1)
    LWa = g("LWRadAtm"); LWn = g("scalarLWNetGround"); dT = g("scalarSurfaceTemp") - g("airtemp"); h.close()
    djf = np.isin(tt.month, [12, 1, 2])[:, None]; snow = ns > 0; djf_all = np.isin(tt.month, [12, 1, 2])
    for gname, m in GROUPS.items():
        s = djf & snow & m[None, :]; a = djf_all[:, None] & m[None, :]
        E.append(dict(run=name, group=gname, LWin=LWa[a].mean(), LWnet=LWn[s].mean(), Ts_Ta=dT[s].mean(), snow_base=sb[s].mean(), pack=bulk[s].mean(),
                      soil_top=st[a].mean(), soil_ice=si[a].mean()))
    d = nc.Dataset(p/"cal_day.nc"); td = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    swe = pd.DataFrame(np.array(d["scalarSWE_mean"][:]), index=td); q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=td)["2012-10-01":"2020-09-30"]; d.close()
    a1 = swe[(swe.index.month == 4) & (swe.index.day == 1)]["2013":"2020"].mean()
    W[name] = {gname: float((a1.values[m]*area[m]).sum()/area[m].sum()) for gname, m in GROUPS.items()}
    sc = json.loads((p/"score.json").read_text()) if (p/"score.json").exists() else None
    if sc is None: obj, parts, diag = C5.score(p/"cal_day.nc", C5.C4.obs_mm()); sc = dict(objective=obj, diagnostics=diag)
    gg = sc["diagnostics"]; b_ = base(q); mo = q.index.month
    F[name] = dict(objective=sc["objective"], **{c: gg[c] for c in ("KGE_AprJul", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias", "aso_swe_nmae", "band_order_mae", "vapour_mm")},
                   **{f"base {l}": b_[mo == mm].mean() for mm, l in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]}, DecMar_mm=q[np.isin(mo, [12, 1, 2, 3])].sum()/8)
bo = base(obs); mo = obs.index.month
F["observed"] = {**{f"base {l}": bo[mo == mm].mean() for mm, l in [(12, "Dec"), (1, "Jan"), (2, "Feb"), (3, "Mar")]}, "DecMar_mm": obs[np.isin(mo, [12, 1, 2, 3])].sum()/8}
pd.set_option("display.width", 250)
X = pd.DataFrame(E).set_index(["group", "run"]).sort_index(); print("Dec-Feb (snow-covered hours for LWnet/Ts-Ta/snow; all hours for LWin/soil)"); print(X.round(2).to_string())
print("\nApr 1 SWE (mm), WY2013-2020 mean"); print(pd.DataFrame(W).T.round(0).to_string())
Fd = pd.DataFrame(F).T; print("\n", Fd.round(3).to_string())
pd.concat([X.reset_index().assign(table="energy"), Fd.reset_index().rename(columns={"index": "run"}).assign(table="flow")]).to_csv(FIG + "tuo_suite5_cloudlw.csv", index=False)
print("\n", FIG + "tuo_suite5_cloudlw.csv")
