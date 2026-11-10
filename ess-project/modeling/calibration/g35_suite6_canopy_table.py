"""
Table for g35_suite6_canopy.py (BeersLaw vs UEB_2stream), 2026-11-06.
Groups: open (summerLAI < 2: HRUs 1-7) and forest (summerLAI >= 2) split at 2500 / 3000 m. Per group, area-weighted, WY2013-2020:
  Dec-Feb and Mar-May mean below-canopy SW, ground-absorbed SW, canopy LW to ground; Apr 1 and May 1 SWE; melt-out day (first day after Mar 1 with SWE < 5 mm,
  as day of water year) ; Then the flow/score diagnostics from score.json.
-> figures/draft/tuo_suite6_canopy.csv
usage: python g35_suite6_canopy_table.py
"""
import json, numpy as np, pandas as pd, netCDF4 as nc
import g35_suite6_canopy as G
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
SET = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/"
A = nc.Dataset(SET + "attributes.nc"); area = np.array(A["HRUarea"][:]).ravel(); el = np.array(A["elevation"][:]).ravel()
lai = np.array(nc.Dataset(SET + "trialParams.nc")["summerLAI"][:]).ravel(); forest = lai >= 2
GROUPS = {"open": ~forest, "forest <2500": forest & (el < 2500), "forest 2500-3000": forest & (el >= 2500) & (el < 3000), "forest >3000": forest & (el >= 3000)}
SCHEMES = ["BeersLaw", "UEB_2stream"]
rows, flow = [], {}
for s in SCHEMES:
    d = nc.Dataset(G.ROOT/s/"cal_day.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False)).normalize()
    V = lambda v: pd.DataFrame(np.array(d[f"{v}_mean"][:]).astype(float), index=t)["2012-10-01":"2020-09-30"]
    swe = V("scalarSWE"); bc = V("scalarBelowCanopySolar"); ga = V("scalarGroundAbsorbedSolar"); lwc = V("scalarLWRadCanopy2Ground"); d.close()
    mo = swe.index.month; wy = swe.index.year + (mo >= 10)
    mo_out = []
    for y in range(2013, 2021):
        x = swe[(wy == y) & (swe.index >= f"{y}-03-01")]; dd = (x < 5).idxmax().where((x < 5).any())
        mo_out.append(((pd.DatetimeIndex(dd.fillna(pd.Timestamp(f"{y}-09-30"))) - pd.Timestamp(f"{y - 1}-10-01")).days).values)
    melt = np.mean(mo_out, 0)
    a1 = swe[(mo == 4) & (swe.index.day == 1)].mean().values; m1 = swe[(mo == 5) & (swe.index.day == 1)].mean().values
    win = np.isin(mo, [12, 1, 2]); spr = np.isin(mo, [3, 4, 5])
    for gname, m in GROUPS.items():
        w = lambda v: float((v[m]*area[m]).sum()/area[m].sum())
        rows.append(dict(group=gname, scheme=s, BelowSW_DJF=w(bc[win].mean().values), BelowSW_MAM=w(bc[spr].mean().values), GroundSW_MAM=w(ga[spr].mean().values),
                         LWcan_DJF=w(lwc[win].mean().values), SWE_Apr1=w(a1), SWE_May1=w(m1), meltout_dowy=w(melt)))
    sc = json.loads((G.ROOT/s/"score.json").read_text()); g = sc["diagnostics"]
    flow[s] = dict(objective=sc["objective"], **{k: g[k] for k in ("KGE_AprJul", "Apr_bias", "May_bias", "Jun_bias", "Jul_bias", "AugSep_bias", "aso_swe_nmae",
                                                                      "band_order_mae", "vapour_mm") if k in g})
X = pd.DataFrame(rows).set_index(["group", "scheme"]); pd.set_option("display.width", 250)
print("HRUs per group:", {k: list(np.where(m)[0] + 1) for k, m in GROUPS.items()})
print(X.round(1).to_string()); Fd = pd.DataFrame(flow).T; print("\n", Fd.round(3).to_string())
pd.concat([X.reset_index().assign(table="snow"), Fd.reset_index().rename(columns={"index": "scheme"}).assign(table="flow")]).to_csv(FIG + "tuo_suite6_canopy.csv", index=False)
print("\n", FIG + "tuo_suite6_canopy.csv")
