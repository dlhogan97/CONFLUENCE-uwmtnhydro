"""Tuolumne forcing SW vs Vogelsang (CDEC FLV, 3077 m) incoming solar, plus FLV observed albedo. Forcing: band b6 (~3060 m) HRU mean."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, glob, geopandas as gpd
OBS = '/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/cdec_FLV_radiation.csv'
o = pd.read_csv(OBS, parse_dates=['time'], index_col='time')
o.loc[(o.sw_down < 0) | (o.sw_down > 1400), 'sw_down'] = np.nan; o.loc[(o.sw_up < 0) | (o.sw_up > 1400), 'sw_up'] = np.nan
o.loc[o.sw_up > 1.02*o.sw_down, 'sw_up'] = np.nan
h = o[['sw_down', 'sw_up']].resample('1h').mean(); pass                                           # CDEC is PST; forcing files are local solar time (SW peaks at hour 12): same clock, no shift
print('hourly valid fraction sw_down', round(h.sw_down.notna().mean(), 3))
cnt = h.sw_down.resample('1D').count(); day_sw = h.sw_down.resample('1D').mean().where(cnt >= 20)
sw_num, sw_den = h.sw_up.resample('1D').sum(), h.sw_down.resample('1D').sum(); alb = (sw_num/sw_den).where((sw_den > 500) & (cnt >= 20))
g = gpd.read_file("/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp")
b6 = g[g.elevClass == 6].HRU_ID.values.astype(int); b5 = g[g.elevClass == 5].HRU_ID.values.astype(int); b7 = g[g.elevClass == 7].HRU_ID.values.astype(int)
FD = '/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/forcing/SUMMA_input'
rows = []; peak = None
for f in sorted(glob.glob(FD+'/*_2021[01][0-9].nc')+glob.glob(FD+'/*_202[2-5]*.nc')):
    d = xr.open_dataset(f); ids = list(d.hruId.values.astype(int)); t = pd.DatetimeIndex(d.time.values)
    for nm, hh in (('b6', b6),):
        s = pd.Series(d['SWRadAtm'].values[:, [ids.index(i) for i in hh]].mean(1), index=t)
        if peak is None and t[0].month in (6, 7): peak = s.groupby(s.index.hour).mean().idxmax()
        rows.append(s)
fs = pd.concat(rows).resample('1D').mean(); print('forcing diurnal SW peak hour in file time:', peak, '(20 => UTC, ~12 => local)')
df = pd.concat([day_sw.rename('obs'), fs.rename('forc'), alb.rename('alb')], axis=1).dropna(subset=['obs', 'forc'])
df['mon'] = df.index.month; df['yr'] = df.index.year
m = df.groupby('mon').agg(n=('obs', 'count'), obs=('obs', 'mean'), forc=('forc', 'mean'), alb=('alb', 'mean')); m['bias'] = m.forc-m.obs; m['ratio'] = m.forc/m.obs
pd.set_option('display.width', 200); print(m.round(2).to_string())
print(f"\nall n={len(df)}: obs {df.obs.mean():.0f} forc {df.forc.mean():.0f} bias {df.forc.mean()-df.obs.mean():+.0f} W/m2 r={np.corrcoef(df.obs,df.forc)[0,1]:.2f}")
for nm, ms in (('Mar-Jun', [3, 4, 5, 6]), ('Jul-Sep', [7, 8, 9]), ('Nov-Feb', [11, 12, 1, 2])):
    s = df[df.mon.isin(ms)]; print(f"{nm}: bias {s.forc.mean()-s.obs.mean():+.0f} W/m2 ({100*(s.forc.mean()/s.obs.mean()-1):+.0f}%), n={len(s)}, by year:", {int(y): round(x.forc.mean()-x.obs.mean()) for y, x in s.groupby('yr') if len(x) > 20})
