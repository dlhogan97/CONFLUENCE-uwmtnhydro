"""
Full-year daily KGE (and r, alpha = sd ratio, beta = mean ratio) for round 5, the lever runs and round 7 (2026-11-09).
NOTE: rounds 5-7 calibrated on ALL of WY2013-2020 (Apr-Jul terms), so there is NO held-out evaluation period; the WY2013-2016 / WY2017-2020 split is for
information only. Also log-flow KGE (low-flow emphasis) and Apr-Jul KGE.
-> figures/draft/tuo_kge_periods.csv
"""
import numpy as np, pandas as pd, netCDF4 as nc
import g35_suite5_cal as C5, g35_lever_tests as L
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
RUNS = {"round5": C5.OUT/"best"/"best_obj"} | {v: L.OUT/v for v in L.VARIANTS} | {"round7": C5.F.OUT/"suite7_cal"/"best"}
obs = C5.C4.obs_mm()
def kge(s, o):
    r = np.corrcoef(s, o)[0, 1]; a = s.std()/o.std(); b = s.mean()/o.mean(); return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2), r, a, b
P = {"WY2013-2020": ("2012-10-01", "2020-09-30"), "WY2013-2016": ("2012-10-01", "2016-09-30"), "WY2017-2020": ("2016-10-01", "2020-09-30")}
rows = []
for n, p in RUNS.items():
    if not (p/"cal_day.nc").exists(): continue
    d = nc.Dataset(p/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=t); d.close()
    for pn, (a0, a1) in P.items():
        j = pd.concat([q.rename("s"), obs.rename("o")], axis=1)[a0:a1].dropna(); k, r, al, be = kge(j.s.values, j.o.values)
        aj = j[j.index.month.isin([4, 5, 6, 7])]; lk = kge(np.log(j.s.values + 0.01), np.log(j.o.values + 0.01))[0]
        rows.append(dict(run=n, period=pn, KGE=k, r=r, alpha=al, beta=be, KGE_AprJul=kge(aj.s.values, aj.o.values)[0], KGE_log=lk))
X = pd.DataFrame(rows).set_index(["period", "run"]); pd.set_option("display.width", 200); print(X.round(3).to_string()); X.to_csv(FIG + "tuo_kge_periods.csv")
