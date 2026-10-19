import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd, rasterio, re, netCDF4 as nc
from rasterio.mask import mask
from pathlib import Path
D=Path("/scratch/dlhogan/ess-project-data/geospatial-data/ASO/East")
SHP="/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp"
g=gpd.read_file(SHP); print(g.columns.tolist()[:14]); 
hid_col=[c for c in g.columns if c.lower() in ("hru_id","hruid","hru")][0]; g=g.sort_values(hid_col).reset_index(drop=True)
R="/scratch/dlhogan/ess-project-data/calibration/bestruns/East_River_baseline_1981_2025/run"
d=nc.Dataset(R+"/out/baseline_timestep.nc"); t=d["time"]; tt=pd.DatetimeIndex(nc.num2date(t[:],t.units,only_use_cftime_datetimes=False)); hid=np.array(d["hruId"][:]).ravel().astype(int)
swe=pd.DataFrame(np.array(d["scalarSWE"][:]),index=tt,columns=hid).resample("1D").mean()
A=np.array(nc.Dataset(R+"/attributes.nc")["HRUarea"][:]); elev=np.array(nc.Dataset(R+"/attributes.nc")["elevation"][:])
print("HRU ids shp:",g[hid_col].tolist()[:5],"nc:",hid[:5])
rows=[]
from rasterio.features import geometry_mask
for f in sorted(D.glob("*.tif")):
    m=re.search(r"(\d{8})",f.name); date=pd.Timestamp(m.group(1))
    with rasterio.open(f) as src:
        gg=g.to_crs(src.crs); nod=src.nodata
        for i,r in gg.iterrows():
            try: arr,tr=mask(src,[r.geometry],crop=True,filled=True,nodata=-9999.0)
            except Exception: continue
            a=arr[0].astype(float); inside=geometry_mask([r.geometry],out_shape=a.shape,transform=tr,invert=True)
            ok=inside&np.isfinite(a)&(a>=0)&(a<50)&(a!=-9999.0)
            if nod is not None: ok&=(a!=nod)
            rows.append(dict(file=f.name,date=date,hru=int(r[hid_col]),aso_mm=1000*a[ok].mean() if ok.any() else np.nan,covf=ok.sum()/max(inside.sum(),1)))
df=pd.DataFrame(rows); df.to_csv("aso_east_hru.csv",index=False)
df["mod_mm"]=[swe.loc[dt,h] if dt in swe.index else np.nan for dt,h in zip(df.date,df.hru)]
ar=dict(zip(hid,A)); df["area"]=df.hru.map(ar); df["elev"]=df.hru.map(dict(zip(hid,elev)))
out=[]
for (f,dt),gb in df.groupby(["file","date"]):
    gb=gb[(gb.covf>=0.7)&gb.aso_mm.notna()&gb.mod_mm.notna()]
    if len(gb)<4: out.append(dict(date=dt.date(),n_hru=len(gb),note="insufficient coverage / outside model run")); continue
    w=gb.area/gb.area.sum(); out.append(dict(date=dt.date(),n_hru=len(gb),area_frac=round(gb.area.sum()/A.sum(),2),ASO_mm=round((gb.aso_mm*w).sum()),model_mm=round((gb.mod_mm*w).sum()),bias_pct=round(100*((gb.mod_mm*w).sum()/(gb.aso_mm*w).sum()-1),1),r_hru=round(np.corrcoef(gb.aso_mm,gb.mod_mm)[0,1],2)))
pd.set_option("display.width",200); print(pd.DataFrame(out).sort_values("date").to_string(index=False))
# by elevation class: pooled across spring flights (Mar-Apr, pre-melt) 
sp=df[(df.date.dt.month<=4)&(df.covf>=0.7)&df.aso_mm.notna()&(df.date.dt.year<=2025)]
sp=sp.assign(band=pd.cut(sp.elev,[0,2800,3200,3500,4000],labels=["~2700 m","~3000 m","~3350 m","~3700 m"]))
print("\nMarch-April flights, model/ASO SWE by elevation band (sum over HRU-flights):")
print(sp.groupby("band").apply(lambda x: pd.Series(dict(n=len(x),ASO_mm=x.aso_mm.mean(),model_mm=x.mod_mm.mean(),ratio=x.mod_mm.sum()/x.aso_mm.sum()))).round(2).to_string())
lt=df[(df.date.dt.month>=5)&(df.covf>=0.7)&df.aso_mm.notna()&(df.date.dt.year<=2025)]
print("\nMay-June flights (melt season):"); print(lt.groupby("date").apply(lambda x: pd.Series(dict(n=len(x),ASO_mm=(x.aso_mm*x.area).sum()/x.area.sum(),model_mm=(x.mod_mm*x.area).sum()/x.area.sum()))).round(0).to_string())
