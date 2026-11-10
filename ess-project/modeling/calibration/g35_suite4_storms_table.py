"""
Compare the g35_suite4_storms.py runs (tempCritRain and snow liquid-water variants) with the round-4 best: objective/constraints, winter flow, and the four
large Dec-Mar storm responses. -> figures/draft/tuo_suite4_snow_liquid.csv
usage: python g35_suite4_storms_table.py
"""
import json, numpy as np, pandas as pd, netCDF4 as nc
import g35_suite4_cal as C4
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
EV = ["2017-01-12", "2017-02-11", "2018-03-25", "2019-02-17"]; W = ("2012-10-01", "2020-09-30")
obs = C4.obs_mm()[W[0]:W[1]]
runs = {"round-4 best (270.66)": (C4.OUT/"fate/nopatch/cal_day.nc", C4.OUT/"best/best_obj/trial.json")}
if (C4.OUT.parent/"suite5_cal/best/best_obj/score.json").exists():
    runs["round-5 best"] = (C4.OUT.parent/"suite5_cal/best/best_obj/cal_day.nc", C4.OUT.parent/"suite5_cal/best/best_obj/score.json")
for d in sorted((C4.OUT/"storms").glob("tcr_*")) + sorted((C4.OUT/"storms").glob("snow_*")):
    if (d/"score.json").exists(): runs[d.name] = (d/"cal_day.nc", d/"score.json")
def flows(q):
    m = q.index.month; n = 8
    r = {"DecApr_mm": q[np.isin(m, [12, 1, 2, 3, 4])].sum()/n, "JanMar_mm": q[np.isin(m, [1, 2, 3])].sum()/n, "JunJul_mm": q[np.isin(m, [6, 7])].sum()/n}
    for e in EV: e = pd.Timestamp(e); r[f"{e:%b%y}_5d"] = q[e - pd.Timedelta(days=4):e].sum()
    return r
rows = {}
for k, (p, js) in runs.items():
    d = nc.Dataset(p); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    q = pd.Series(np.array(d["averageRoutedRunoff_mean"][:]).ravel()*86400e3, index=t)[W[0]:W[1]]; d.close()
    s = json.loads(js.read_text()); g = s["diagnostics"]
    rows[k] = dict(objective=s["objective"], **{c: g[c] for c in ("KGE_AprJul", "AprMay_bias", "JunJul_bias", "AugSep_bias", "OctNov_bias", "aso_swe_nmae", "band_order_mae", "vapour_mm")}, **flows(q))
rows["observed"] = flows(obs)
T = pd.DataFrame(rows).T; T.to_csv(FIG + "tuo_suite4_snow_liquid.csv"); pd.set_option("display.width", 260); print(T.round(2).to_string())
