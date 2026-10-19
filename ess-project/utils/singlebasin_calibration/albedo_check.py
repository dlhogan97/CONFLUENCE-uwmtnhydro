"""Model snow albedo (scalarSnowAlbedo, snow-covered days) vs measured albedo: Tuolumne -> CDEC FLV Vogelsang (3077 m, vs band b6); East -> ARM SAIL S3 (2856 m, vs the ~2700 and ~3000 m bands).
usage: albedo_check.py <East_River|Tuolumne_River> <best_full/rank0_ext dir>"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr
dom, R = sys.argv[1], sys.argv[2]
d = xr.open_dataset(R+"/out/best_timestep.nc"); t = pd.DatetimeIndex(d["time"].values); hid = d["hruId"].values.ravel().astype(int)
alb = pd.DataFrame(d["scalarSnowAlbedo_mean"].values if "scalarSnowAlbedo_mean" in d else d["scalarSnowAlbedo"].values, index=t, columns=hid).resample("D").mean()
swe = pd.DataFrame(d["scalarSWE"].values, index=t, columns=hid).resample("D").mean()
a = xr.open_dataset(R+"/attributes.nc"); E = pd.Series(a["elevation"].values.ravel(), index=a["hruId"].values.ravel().astype(int))
if dom == "Tuolumne_River":
    import geopandas as gpd
    g = gpd.read_file("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
    hrus = g[g.elevClass == 6].HRU_ID.astype(int).values
    o = pd.read_csv("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/cdec_FLV_radiation.csv", parse_dates=["time"], index_col="time")
    o.loc[(o.sw_down < 0) | (o.sw_down > 1400), "sw_down"] = np.nan; o.loc[(o.sw_up < 0) | (o.sw_up > 1400), "sw_up"] = np.nan; o.loc[o.sw_up > 1.02*o.sw_down, "sw_up"] = np.nan
    h = o[["sw_down", "sw_up"]].resample("1h").mean(); c = h.sw_down.resample("1D").count(); num, den = h.sw_up.resample("1D").sum(), h.sw_down.resample("1D").sum()
    obs = (num/den).where((den > 500) & (c >= 20)); name = "FLV 3077 m vs model b6"
else:
    hrus = E[(E > 2500) & (E < 3200)].index.values
    B = "/storage/dlhogan/synoptic_sublimation/sail_data/"; fr = []
    for f in (B+"winter_21_22/sebs_20211001_20220930.nc", B+"winter_22_23/sebs_20221001_20230930.nc"):
        x = xr.open_dataset(f); tt = pd.DatetimeIndex(x.time.values)
        fr.append(pd.DataFrame({"dn": pd.Series(x.down_short_hemisp.values.astype(float), index=tt).where(pd.Series(x.qc_down_short_hemisp.values, index=tt).isin([0, 2])).clip(lower=0),
                                "up": pd.Series(x.up_short_hemisp.values.astype(float), index=tt).where(pd.Series(x.qc_up_short_hemisp.values, index=tt).isin([0, 2])).clip(lower=0)}))
    h = pd.concat(fr).resample("1h").mean(); h.index = h.index - pd.Timedelta(hours=7); c = h.dn.resample("1D").count(); num, den = h.up.resample("1D").sum(), h.dn.resample("1D").sum()
    obs = (num/den).where((den > 500) & (c >= 20)); name = "SAIL S3 2856 m vs model 2700-3000 m bands"
snow = (swe[hrus] > 10)
mod = alb[hrus].where(snow).mean(axis=1); frac = snow.mean(axis=1)
df = pd.concat([mod.rename("model"), obs.rename("obs"), frac.rename("model_snow_frac")], axis=1).dropna(subset=["obs"]); df = df[df.index >= df.index[0]]
df["mon"] = df.index.month; df["wy"] = df.index.year + (df.index.month >= 10)
print(name, "| days compared:", len(df), "| years:", sorted(df.wy.unique()))
m = df.groupby("mon").agg(n=("obs", "count"), obs=("obs", "mean"), obs_p10=("obs", lambda x: x.quantile(.1)), model=("model", "mean"), model_p10=("model", lambda x: x.quantile(.1)), model_snow_frac=("model_snow_frac", "mean"))
m = m.reindex([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]).dropna(how="all"); pd.set_option("display.width", 200); print(m.round(2).to_string())
# melt season only, days the instrument sees snow (albedo > 0.45) AND the model has snow: how fast does albedo fall?
s = df[(df.obs > 0.45) & df.model.notna()]; print("\nSnow days both agree on (obs albedo>0.45, model SWE>10):", len(s), "mean obs", round(s.obs.mean(), 2), "mean model", round(s.model.mean(), 2))
for ms, nm in (([3, 4], "Mar-Apr"), ([5, 6], "May-Jun")):
    q = s[s.mon.isin(ms)]
    if len(q): print(f"  {nm}: obs {q.obs.mean():.2f} (p10 {q.obs.quantile(.1):.2f})  model {q.model.mean():.2f} (p10 {q.model.quantile(.1):.2f})  n={len(q)}")
