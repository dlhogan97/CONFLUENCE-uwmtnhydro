"""
Snow-surface energy balance summary for g35_suite5_energy.py (2026-11-06). Groups: north <2900 m, south <2900 m, 2900-3500 m, >3500 m (aspect-based, as in
g35_suite5_ground_table.py). Snow-covered hours only (nSnow > 0), WY2013-2020, by month (Nov-Apr) and night (00-06 local) / day (10-16 local).
Columns (W/m2, positive = energy INTO the snow surface; signs checked against the night-time sensible heat, which must be positive over a surface colder than air):
  Ta, Ts, Ts-Ta (C); LWatm = forcing LWRadAtm; eps_eff = LWatm/(sigma Ta^4); LWsky = sky LW absorbed by ground; LWcan = canopy LW absorbed by ground;
  LWout = emitted; LWnet; SWabs; H; LE; ADV; NET
Also the pack temperature profile (top, base, whole-pack mean).
-> figures/draft/tuo_suite5_energy.csv
usage: python g35_suite5_energy_table.py
"""
import numpy as np, pandas as pd, netCDF4 as nc
import g35_suite5_cal as C5
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/"
SIG = 5.670374e-8
A = nc.Dataset("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/settings/SUMMA/attributes.nc")
area = np.array(A["HRUarea"][:]).ravel(); el = np.array(A["elevation"][:]).ravel(); asp = np.array(A["aspect"][:]).ravel()
N = (asp >= 300) | (asp <= 60); S = (asp >= 120) & (asp <= 240)
GROUPS = {"N <2900": N & (el < 2900), "S <2900": S & (el < 2900), "2900-3500": (el >= 2900) & (el < 3500), ">3500": el >= 3500}
h = nc.Dataset(C5.OUT/"energy"/"cal_timestep.nc"); th = pd.DatetimeIndex(nc.num2date(h["time"][:], h["time"].units, only_use_cftime_datetimes=False))
k = np.where(th >= pd.Timestamp("2012-10-01"))[0]; sl = slice(k[0], k[-1] + 1); tt = th[sl]
g = lambda v: np.array(h[v][sl]).astype(float)
Ta = g("airtemp"); Ts = g("scalarSurfaceTemp"); LWa = g("LWRadAtm"); ns = g("nSnow").astype(int)
V = dict(LWsky=g("scalarLWRadUbound2Ground"), LWcan=g("scalarLWRadCanopy2Ground"), LWout=g("scalarLWRadGround"), LWnet=g("scalarLWNetGround"),
         SWabs=g("scalarGroundAbsorbedSolar"), H=g("scalarSenHeatGround"), LE=g("scalarLatHeatGround"), ADV=g("scalarGroundAdvectiveHeatFlux"), NET=g("scalarGroundNetNrgFlux"))
T = np.array(h["mLayerTemp"][sl]) - 273.15; h.close()
top = T[:, 0, :]; base = np.take_along_axis(T, np.maximum(ns - 1, 0)[:, None, :], 1)[:, 0]
cs = np.cumsum(np.where(np.arange(T.shape[1])[None, :, None] < ns[:, None, :], T, 0), axis=1); bulk = np.take_along_axis(cs, np.maximum(ns - 1, 0)[:, None, :], 1)[:, 0]/np.maximum(ns, 1)
snow = ns > 0; hr = tt.hour.values; mo = tt.month.values
PART = {"night": (hr >= 0) & (hr <= 6), "day": (hr >= 10) & (hr <= 16), "all": np.ones(len(tt), bool)}
rows = []
for gname, m in GROUPS.items():
    for mon in (11, 12, 1, 2, 3, 4):
        for pname, p in PART.items():
            s = snow & (mo == mon)[:, None] & p[:, None] & m[None, :]
            if s.sum() < 50: continue
            r = dict(group=gname, month=mon, part=pname, Ta=(Ta[s] - 273.15).mean(), Ts=(Ts[s] - 273.15).mean(), eps_eff=(LWa[s]/(SIG*Ta[s]**4)).mean(), LWatm=LWa[s].mean())
            r["Ts-Ta"] = r["Ts"] - r["Ta"]
            r.update({kk: vv[s].mean() for kk, vv in V.items()}); r.update(top=top[s].mean(), base=base[s].mean(), bulk=bulk[s].mean())
            rows.append(r)
X = pd.DataFrame(rows).set_index(["group", "month", "part"]); X.to_csv(FIG + "tuo_suite5_energy.csv"); pd.set_option("display.width", 250)
C = ["Ta", "Ts", "Ts-Ta", "LWatm", "eps_eff", "LWsky", "LWcan", "LWout", "LWnet", "SWabs", "H", "LE", "ADV", "NET", "top", "base", "bulk"]
for part in ("night", "day", "all"):
    print(f"\n=== {part}, snow-covered hours, Dec-Feb mean ===")
    print(X.xs(part, level="part").loc[(slice(None), [12, 1, 2]), :].groupby(level="group")[C].mean().round(2).to_string())
print("\n=== all hours, by month: LWnet / H / NET / base ===")
print(X.xs("all", level="part")[["LWnet", "SWabs", "H", "LE", "NET", "base", "eps_eff"]].unstack("month").round(1).to_string())
print("\n", FIG + "tuo_suite5_energy.csv")
