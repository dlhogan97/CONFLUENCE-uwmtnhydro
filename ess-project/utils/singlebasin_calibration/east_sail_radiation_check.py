"""East forcing SW/LW vs ARM SAIL S3 (Mt Crested Butte, 2856 m) observations, WY2022-23.  Monthly means of daily means, UTC days.
Forcing = East elevAspect SUMMA_input (METSIM remap) interpolated to 2856 m between the ~2700 m and ~3000 m bands (simple mean over each band's HRUs)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, glob
B = '/storage/dlhogan/synoptic_sublimation/sail_data/'
def load(f):
    d = xr.open_dataset(f); t = pd.DatetimeIndex(d.time.values); o = {}
    for k, ok in (('SW', (0, 2)), ('LW', (0,)), ('SWup', (0, 2)), ('LWup', (0,)), ('alb', (0,))):
        name = {'SW': 'down_short_hemisp', 'LW': 'down_long', 'SWup': 'up_short_hemisp', 'LWup': 'up_long', 'alb': 'albedo'}[k]
        v = pd.Series(d[name].values.astype(float), index=t); q = pd.Series(d['qc_'+name].values, index=t); v = v.where(q.isin(ok))
        if k in ('SW', 'SWup'): v = v.clip(lower=0)
        o[k] = v
    return pd.DataFrame(o)
obs = pd.concat([load(B+'winter_21_22/sebs_20211001_20220930.nc'), load(B+'winter_22_23/sebs_20221001_20230930.nc')]).sort_index()
h = obs.resample('1h').mean(); h.index = h.index - pd.Timedelta(hours=7)      # ARM is UTC; forcing is local (MST, UTC-7)
day = h.resample('1D').agg(lambda x: x.mean() if x.notna().sum() >= 20 else np.nan)
day['alb_day'] = (h.SWup.resample('1D').sum()/h.SW.resample('1D').sum()).where(h.SW.resample('1D').sum() > 500)       # SW-weighted daily albedo
a = xr.open_dataset('/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/settings/SUMMA/attributes.nc')
E = pd.Series(a.elevation.values, index=a.hruId.values); lo = E[(E > 2500) & (E < 2800)].index.values; hi = E[(E > 2850) & (E < 3200)].index.values
def forcing(path):
    out = []
    for f in sorted(glob.glob(path+'/*_2021[1][0-2].nc')+glob.glob(path+'/*_2022*.nc')+glob.glob(path+'/*_2023[0][1-9].nc')):
        d = xr.open_dataset(f); t = pd.DatetimeIndex(d.time.values); ids = d.hruId.values.astype(int)
        def band(v, hh): return pd.Series(d[v].values[:, [list(ids).index(i) for i in hh]].mean(1), index=t)
        w = (2856-E[lo].mean())/(E[hi].mean()-E[lo].mean())
        out.append(pd.DataFrame({k: (1-w)*band(k, lo) + w*band(k, hi) for k in ('SWRadAtm', 'LWRadAtm', 'airtemp')}))
    return pd.concat(out).resample('1D').mean()
f0 = forcing('/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/forcing/SUMMA_input')
cmp = pd.concat([day[['SW', 'LW', 'alb_day']], f0.rename(columns={'SWRadAtm': 'SW_f', 'LWRadAtm': 'LW_f', 'airtemp': 'T_f'})], axis=1).dropna(subset=['SW_f'])
cmp = cmp[(cmp.index >= '2021-10-05') & (cmp.index <= '2023-09-30')]
cmp['mon'] = cmp.index.month; cmp['yr'] = cmp.index.year
g = cmp.groupby(['yr', 'mon']).agg(n=('SW', 'count'), SW_obs=('SW', 'mean'), SW_forc=('SW_f', 'mean'), LW_obs=('LW', 'mean'), LW_forc=('LW_f', 'mean'), alb_obs=('alb_day', 'mean'))
g['dSW'] = g.SW_forc-g.SW_obs; g['dLW'] = g.LW_forc-g.LW_obs
pd.set_option('display.width', 200); print(g[g.n >= 10].round(1).to_string())
sub = cmp.dropna(subset=['SW', 'LW'])
print(f"\nall days n={len(sub)}: SW obs {sub.SW.mean():.0f} forc {sub.SW_f.mean():.0f} (bias {sub.SW_f.mean()-sub.SW.mean():+.0f}, r={np.corrcoef(sub.SW,sub.SW_f)[0,1]:.2f}); LW obs {sub.LW.mean():.0f} forc {sub.LW_f.mean():.0f} (bias {sub.LW_f.mean()-sub.LW.mean():+.0f}, r={np.corrcoef(sub.LW,sub.LW_f)[0,1]:.2f})")
for nm, ms in (('Mar-Jun melt season', [3, 4, 5, 6]), ('Nov-Feb', [11, 12, 1, 2])):
    s = sub[sub.mon.isin(ms)]; print(f"{nm}: SW bias {s.SW_f.mean()-s.SW.mean():+.0f} W/m2 ({100*(s.SW_f.mean()/s.SW.mean()-1):+.0f}%), LW bias {s.LW_f.mean()-s.LW.mean():+.0f} W/m2, n={len(s)}")
cmp.to_csv('/tmp/claude-1008/-home-dlhogan-projects-forked-repos-CONFLUENCE-uwmtnhydro/c0858c32-d024-48f4-b236-80c8fd0350ee/scratchpad/east_sail_daily.csv')
