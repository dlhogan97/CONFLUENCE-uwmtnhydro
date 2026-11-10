"""
Compare the g35_lever_tests.py runs with the round-5 best and observed flow (2026-11-06), WY2013-2020.
  1. monthly flow (mm/month, 8-yr mean) and bias
  2. timing: day of water year when 25 / 50 / 75 % of the WY volume has passed
  3. scores: round-5 objective, Apr-Jul KGE, monthly Apr-Jul biases, Aug-Sep, ASO, band order, ET+sublimation
  4. soil water (mm, area-weighted) by elevation band in Oct / Apr / Jun / Sep
-> figures/draft/tuo_levers.csv
usage: python g35_lever_table.py
"""
import json, numpy as np, pandas as pd, netCDF4 as nc
import g35_suite5_cal as C5
import g35_lever_tests as L
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
A = nc.Dataset("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")
area = np.array(A["HRUarea"][:]).ravel(); el = np.array(A["elevation"][:]).ravel()
BANDS = {"<2300": el < 2300, "2300-2900": (el >= 2300) & (el < 2900), "2900-3400": (el >= 2900) & (el < 3400)}
RUNS = {"round5": C5.OUT/"best"/"best_obj"} | {v: L.OUT/v for v in L.VARIANTS if (L.OUT/v/"score.json").exists()} \
       | ({"round7": C5.F.OUT/"suite7_cal"/"best"} if (C5.F.OUT/"suite7_cal"/"best"/"score.json").exists() else {})
obs = C5.C4.obs_mm()["2012-10-01":"2020-09-30"]; ORDER = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def timing(q):
    wy = q.index.year + (q.index.month >= 10); out = []
    for _, g in q.groupby(wy):
        cs = g.cumsum()/g.sum(); out.append([(cs >= f).values.argmax() + 1 for f in (0.25, 0.5, 0.75)])
    return np.mean(out, 0)


M, S, W = {}, {}, {}
for name, p in RUNS.items():
    d = nc.Dataset(p/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=t)["2012-10-01":"2020-09-30"]
    soil = pd.DataFrame(np.array(d["scalarTotalSoilWat_mean"][:]), index=t)["2012-10-01":"2020-09-30"] if "scalarTotalSoilWat_mean" in d.variables else None; d.close()
    j = pd.concat([q.rename("s"), obs.rename("o")], axis=1).dropna()
    M[name] = (j.s.groupby(j.index.month).sum()/8).loc[ORDER]
    sc = json.loads((p/"score.json").read_text()); g = sc["diagnostics"]
    if name == "round5":
        import g35_suite6_cal as C6; _, _, d6 = C6.score(p/"cal_day.nc", C5.C4.obs_mm()); g.update({k: d6[k] for k in ("Apr_bias", "May_bias", "Jun_bias", "Jul_bias")})
        o5 = sc["objective"]
    else: o5 = sc.get("obj_round5") if sc.get("obj_round5") is not None else C5.score(p/"cal_day.nc", C5.C4.obs_mm())[0]
    tm = timing(j.s)
    S[name] = dict(obj_round5=o5, KGE_AprJul=g["KGE_AprJul"], **{k: g[k] for k in ("Apr_bias", "May_bias", "Jun_bias", "Jul_bias", "AugSep_bias", "OctNov_bias",
                   "aso_swe_nmae", "band_order_mae", "vapour_mm", "annual_mm")}, t25=tm[0], t50=tm[1], t75=tm[2])
    if soil is None: continue
    mo = soil.index.month
    W[name] = {f"{b} {m}": float((soil[mo == mm].mean().values[k]*area[k]).sum()/area[k].sum()) for b, k in BANDS.items() for m, mm in (("Oct", 10), ("Apr", 4), ("Jun", 6), ("Sep", 9))}
jo = obs.to_frame("o"); M["observed"] = (obs.groupby(obs.index.month).sum()/8).loc[ORDER]
to = timing(obs); S["observed"] = dict(t25=to[0], t50=to[1], t75=to[2])
Md = pd.DataFrame(M).T; Md.columns = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
B = (100*(Md/Md.loc["observed"] - 1)).drop("observed")
pd.set_option("display.width", 250)
print("monthly flow, mm (8-yr mean)"); print(Md.round(1).to_string())
print("\nmonthly bias, %"); print(B.round(0).to_string())
Sd = pd.DataFrame(S).T; print("\nscores + timing (day of water year)"); print(Sd.round(3).to_string())
Wd = pd.DataFrame(W).T; print("\nsoil water (mm) by band"); print(Wd.round(0).to_string())
pd.concat([Md.assign(table="flow_mm"), B.assign(table="bias_pct"), Sd.assign(table="scores"), Wd.assign(table="soil_mm")]).to_csv(FIG + "tuo_levers.csv")
print("\n", FIG + "tuo_levers.csv")
