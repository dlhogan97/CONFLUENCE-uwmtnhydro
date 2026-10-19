"""SNOTEL / CDEC pillow vs the SUMMA HRU that contains it.

Per station-water-year: peak SWE error, peak-timing error, melt-out date error.
Model SWE is the single HRU containing the pillow -- NOT the basin mean.

QA matters here. Butte (NRCS SNOTEL) is clean, but the three CDEC pillows carry
hard errors -- a 200 m SWE spike at Tuolumne Meadows in 2001, a 762 m spike at
Slide Canyon, and 850-2600 negative readings each. Rather than repair the series,
whole water years failing any physical check are dropped, since every metric here
is computed per year anyway.
"""
import warnings; warnings.filterwarnings("ignore")
import re
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
from shapely.geometry import Point

SN  = Path("/scratch/dlhogan/ess-project-data/meteorological-data/SNOTEL")
OUT = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
BR  = Path("/scratch/dlhogan/ess-project-data/calibration/bestruns")
FT, IN = 0.3048, 25.4

# 1 inch, a natural sensor unit. 10 mm is too tight: Butte (NRCS) has a perfect zero
# floor, but the CDEC pillows sit on an end-of-summer offset of up to 48 mm (Dana p90
# 17 mm), which a 10 mm threshold reads as a pack that never melts -- it cost 26 of 40
# Dana melt-out dates. 40 mm recovers only 5 more years and 50 mm none beyond that.
MELT_TH   = 25.4      # mm
MIN_PEAK  = 50.0      # mm; skip years with no real snowpack
CEIL      = 2500.0    # mm; above this is a sensor fault at these elevations
MAX_JUMP  = 200.0     # mm/d
MIN_NEG   = -50.0     # mm; below this is a fault, not drift
CORE      = ((12, 1), (7, 15))   # Dec 1 -> Jul 15, the window that must be covered
MIN_CORE  = 150       # valid days inside CORE
MELT_MIN_DOWY = 152   # 1 March in water-year days (Oct 1 = day 1)
MELT_CONFIRM  = 14    # days of below-threshold record required after the melt date
WY0, WY1  = 1984, 2025           # model analysis_start is 1983-10-01

BASINS = {
 "East": dict(
   shp="/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
   run=BR/"East_River_baseline_1981_2025/run/out/baseline_timestep.nc",
   sites=[("380_Butte_sntl_obs.csv", "Butte")]),
 "Tuolumne": dict(
   shp="/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/shapefiles/catchment/Tuolumne_River_distributed_elevAspect_HRUs_elevation_aspect.shp",
   run=BR/"Tuolumne_River_baseline_wbias10_1981_2025/run/out/baseline_timestep.nc",
   sites=[("DAN_DANA_MEADOWS_cdec_obs.csv", "Dana Meadows"),
          ("GYP_GAYLOR_PIT_cdec_obs.csv", "Gaylor Pit"),
          ("SLI_SLIDE_CANYON_cdec_obs.csv", "Slide Canyon"),
          ("TUM_TUOLUMNE_MEADOWS_cdec_obs.csv", "Tuolumne Meadows")]),
}
wy = lambda idx: idx.year + (idx.month >= 10)


def load_obs(fn):
    cols = pd.read_csv(SN/fn, nrows=0).columns
    if "SWE" not in cols:                      # Gaylor Pit records depth only
        return None, None, None, None
    d = pd.read_csv(SN/fn, usecols=["datetime", "geometry", "SWE", "SWE_units"])
    lon, lat, elev_ft = map(float, re.findall(r"[-\d.]+", d.geometry.iloc[0]))
    t = pd.to_datetime(d.datetime, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    unit = str(d.SWE_units.dropna().iloc[0]).lower()
    s = pd.to_numeric(d.SWE, errors="coerce") * (IN if unit.startswith("in") else 1.0)
    s = pd.Series(s.values, index=t).dropna()
    return s[~s.index.duplicated(keep="first")].sort_index(), lat, lon, elev_ft*FT


def qa_year(g, y):
    """None if the year passes, else the reason it fails."""
    lo = pd.Timestamp(y-1, *CORE[0]); hi = pd.Timestamp(y, *CORE[1])
    if len(g[(g.index >= lo) & (g.index <= hi)]) < MIN_CORE:
        return "coverage"
    if (g < MIN_NEG).any():
        return "negative"
    if g.max() > CEIL:
        return "ceiling"
    if g.diff().abs().max() > MAX_JUMP:
        return "jump"
    if g.max() < MIN_PEAK:
        return "no snow"
    return None


def melt_out(g, ipk, oct1):
    """FINAL melt-out: the day after the last reading above threshold.

    Taking the *first* drop below threshold gives a spurious early date whenever the
    pack melts, re-accumulates and melts again -- common at Tuolumne Meadows. This
    definition is unique by construction and always after the peak. Returns
    (day_of_wy, n_episodes) with NaN where the date cannot be trusted.
    """
    w = g.loc[ipk:]                                   # after the peak, by construction
    above = w[w > MELT_TH]
    if len(above) == 0 or above.index[-1] == w.index[-1]:
        return np.nan, 0                              # never melts, or record runs out
    last = above.index[-1]
    # censoring guard: need real below-threshold data after the melt date
    tail = w.loc[last:]
    if (tail.index[-1] - last).days < MELT_CONFIRM:
        return np.nan, 0
    dowy = (last - oct1).days + 1
    if dowy < MELT_MIN_DOWY:                          # must be after 1 March
        return np.nan, 0
    # count how many separate below-threshold excursions preceded the final one
    below = (w <= MELT_TH).astype(int)
    episodes = int(((below.diff() == 1).sum()))
    return dowy, episodes


def season_stats(s, qa):
    """peak SWE, day-of-WY of peak, day-of-WY of melt-out, per water year."""
    rows, drops = {}, {}
    for y, g in s.groupby(wy(s.index)):
        if not (WY0 <= y <= WY1):
            continue
        g = g.sort_index()
        if qa:
            bad = qa_year(g, y)
            if bad:
                drops[y] = bad; continue
            g = g.clip(lower=0.0)               # small negative drift -> 0
        elif g.max() < MIN_PEAK:
            continue
        oct1 = pd.Timestamp(y-1, 10, 1)
        ipk = g.idxmax()
        # a peak sitting on the first/last reading means the season is censored
        if qa and (ipk == g.index[0] or ipk == g.index[-1]):
            drops[y] = "censored peak"; continue
        tm, nep = melt_out(g, ipk, oct1)
        rows[y] = dict(peak=float(g.max()),
                       t_peak=(ipk - oct1).days,
                       t_melt=tm, n_episodes=nep)
    return pd.DataFrame(rows).T, drops


recs, notes, multi = [], [], 0
for basin, cfg in BASINS.items():
    g = gpd.read_file(cfg["shp"])
    ds = xr.open_dataset(cfg["run"])
    hid = np.asarray(ds.hruId.values).ravel()
    tt = pd.DatetimeIndex(ds.time.values)
    print(f"\n===== {basin} =====  model {tt[0].date()}..{tt[-1].date()}, {len(hid)} HRUs")

    # area-weighted basin mean SWE, for the "does the containing HRU beat the basin
    # mean?" comparison. Weights come from the shapefile, ordered to match hruId.
    area = g.set_index("HRU_ID").HRU_area.reindex(hid).to_numpy(float)
    wts = area / area.sum()
    bmean = pd.Series((np.asarray(ds["scalarSWE"].values) * wts).sum(axis=1),
                      index=tt).resample("D").mean()
    m_basin, _ = season_stats(bmean, qa=False)
    print(f"  basin mean: {len(m_basin)} WY with a peak >= {MIN_PEAK:.0f} mm, "
          f"{int(m_basin.t_melt.isna().sum())} of them never melt out (dropped)")

    for fn, name in cfg["sites"]:
        obs, lat, lon, elev = load_obs(fn)
        if obs is None:
            print(f"  {name:18s} -- no SWE column, skipped"); continue
        pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(g.crs)
        hru = gpd.sjoin(pt, g, predicate="within").iloc[0]
        h_id, h_elev = int(hru.HRU_ID), float(hru.elev_mean)
        k = int(np.where(hid == h_id)[0][0])
        mod = pd.Series(np.asarray(ds["scalarSWE"].isel(hru=k).values).ravel(),
                        index=tt).resample("D").mean()

        o, drops = season_stats(obs, qa=True)
        m_hru, _ = season_stats(mod, qa=False)
        multi += int((o.n_episodes.reindex(o.index.intersection(m_hru.index))
                      .fillna(0) > 1).sum())
        # the same pillow is scored twice: against its own HRU, and against the basin mean
        for src, m in (("hru", m_hru), ("basin", m_basin)):
            for y in o.index.intersection(m.index):
                recs.append(dict(basin=basin, source=src, site=name, wy=int(y), hru=h_id,
                                 site_elev_m=elev, hru_elev_m=h_elev,
                                 peak_obs=o.peak[y], peak_mod=m.peak[y],
                                 tpeak_obs=o.t_peak[y], tmelt_obs=o.t_melt[y],
                                 tpeak_mod=m.t_peak[y], tmelt_mod=m.t_melt[y],
                                 dur_obs=o.t_melt[y]-o.t_peak[y],
                                 dur_mod=m.t_melt[y]-m.t_peak[y],
                                 d_peak=m.peak[y]-o.peak[y],
                                 d_tpeak=m.t_peak[y]-o.t_peak[y],
                                 d_tmelt=m.t_melt[y]-o.t_melt[y],
                                 d_dur=(m.t_melt[y]-m.t_peak[y])-(o.t_melt[y]-o.t_peak[y]),
                                 obs_episodes=o.n_episodes[y],
                                 mod_episodes=m.n_episodes[y]))
        yrs = o.index.intersection(m_hru.index)
        why = pd.Series(drops).value_counts().to_dict()
        print(f"  {name:18s} {elev:4.0f} m -> HRU {h_id:2d} ({h_elev:4.0f} m, {h_elev-elev:+4.0f} m)"
              f"   n = {len(yrs):2d} WY   dropped {len(drops)}: {why}")
        notes.append(dict(basin=basin, site=name, hru=h_id, site_elev_m=round(elev),
                          hru_elev_m=round(h_elev), n_years=len(yrs), n_dropped=len(drops)))

df = pd.DataFrame(recs)

# ---- normalise each error by the mean OBSERVED value of its own record --------------
# Per station, so a 100 mm miss counts differently at Butte (mean peak 400 mm) than at
# Slide Canyon (mean peak 1000 mm). Dates are day-of-water-year, whose origin (1 Oct)
# is arbitrary, so `nd_season` additionally normalises the timing errors by the mean
# observed peak->melt-out interval, which is a real duration and the fairer scale.
base = df.groupby("site").agg(mu_peak=("peak_obs", "mean"),
                              sd_peak=("peak_obs", "std"),
                              q25_peak=("peak_obs", lambda s: s.quantile(0.25)),
                              q75_peak=("peak_obs", lambda s: s.quantile(0.75)),
                              mu_tpeak=("tpeak_obs", "mean"),
                              mu_tmelt=("tmelt_obs", "mean"))
base["iqr_peak"] = base.q75_peak - base.q25_peak
base["mu_season"] = base.mu_tmelt - base.mu_tpeak          # mean peak->melt duration
df = df.join(base, on="site")
df["nd_peak"]   = df.d_peak  / df.mu_peak                  # fraction of mean peak SWE
# Error in units of the record's own interannual spread. IQR is used rather than SD
# because a handful of extreme years (2017, 2023) inflate the SD and flatter the model.
df["rsr_peak"]  = df.d_peak  / df.sd_peak      # kept for reference; not plotted
df["niqr_peak"] = df.d_peak  / df.iqr_peak
df["nd_tpeak"]  = df.d_tpeak / df.mu_tpeak                 # fraction of mean day-of-WY
df["nd_tmelt"]  = df.d_tmelt / df.mu_tmelt
df["ns_tpeak"]  = df.d_tpeak / df.mu_season                # fraction of ablation season
df["ns_tmelt"]  = df.d_tmelt / df.mu_season
df.to_csv(OUT/"snotel_hru_swe_metrics.csv", index=False)
pd.DataFrame(notes).to_csv(OUT/"snotel_hru_sites.csv", index=False)
print(f"\nwrote {OUT/'snotel_hru_swe_metrics.csv'}  ({len(df)} station-years)")
print(f"observed seasons with more than one melt-out episode: {multi} "
      f"(resolved to the FINAL melt-out)")

# ---- per-site skill, and pick one site per basin ------------------------------------
def site_table(d):
    r = {}
    for s, g in d.groupby("site"):
        r[s] = dict(basin=g.basin.iloc[0], n=len(g),
                    RMSE_over_IQR=np.sqrt((g.niqr_peak**2).mean()),
                    nRMSE_pct=100*np.sqrt((g.nd_peak**2).mean()),
                    bias_pct=100*g.nd_peak.mean(),
                    melt_ME_d=g.d_tmelt.mean(),
                    dur_ME_d=g.d_dur.mean())
    return pd.DataFrame(r).T

st = site_table(df[df.source == "hru"])
print("\n--- per-site skill (all QA-passing station-years) ---")
print(st.round(2).to_string())

# one site per basin: the lowest peak-SWE RSR, i.e. the best against its own climatology
# Explicit, NOT auto-ranked. Ranking sites by a spread-normalised error rewards whichever
# site is most variable: Slide Canyon has the largest observed IQR (677 mm vs Butte's 182)
# so it wins on RMSE/IQR (0.58) despite the worst bias of the three (-34.9%), and it would
# cut the shared record from 25 years to 16. No Tuolumne site is best on every measure --
#   Slide Canyon     best RMSE/IQR 0.58 | worst bias -34.9% | n=21
#   Tuolumne Meadows      RMSE/IQR 0.66 |       bias +26.5% | n=27  <- longest record
#   Dana Meadows          RMSE/IQR 0.72 |  best bias  -4.3% | but melts out 40 d early
# Default keeps the longest record. Change this line to swap sites.
PICK = {"East": "Butte", "Tuolumne": "Tuolumne Meadows"}
print(f"\nsite used per basin: {PICK}")
df_pick = df[df.site.isin(PICK.values())].copy()

# ---- one value per basin-year, on the year set both basins share --------------------
MET = ["d_peak", "d_tpeak", "d_tmelt", "d_dur", "dur_obs", "dur_mod",
       "nd_peak", "rsr_peak", "niqr_peak", "nd_tpeak", "nd_tmelt", "ns_tpeak", "ns_tmelt"]
KEY = ["basin", "source", "wy"]
ann = df_pick.groupby(KEY)[MET].mean()
ann["n_sites"] = df_pick.groupby(KEY).site.nunique()
ann["site"] = df_pick.groupby(KEY).site.first()
# Every (basin, source) cell must have every metric that year -- so a year in which the
# basin mean never melts out is dropped everywhere, keeping all four rows comparable.
cells = sorted({(b, s) for b, s, _ in ann.index})
yr_sets = {c: {y for b, s, y in ann.index if (b, s) == c} for c in cells}
common = sorted(set.intersection(*yr_sets.values()))
ok = [y for y in common
      if all(ann.loc[(b, s, y), MET].notna().all() for b, s in cells)]
ann = ann.loc[[(b, s, y) for b, s in cells for y in ok]].reset_index()
ann.to_csv(OUT/"snotel_hru_swe_annual.csv", index=False)

print(f"\nper-cell year counts: { {f'{b}/{s_}': len(v) for (b, s_), v in yr_sets.items()} }")
print(f"common years: {len(common)}   with all metrics present: {len(ok)}  "
      f"[{min(ok)}-{max(ok)}]")
print(f"wrote {OUT/'snotel_hru_swe_annual.csv'}  ({len(ann)} basin-years)\n")

print("mean observed values used as the normaliser:")
print(base.round(1).to_string(), "\n")
for (basin, srcname), gb in ann.groupby(["basin", "source"]):
    print(f"{basin:9s} [{srcname:5s}] n={len(gb):3d} yr")
    print(f"   absolute   peak RMSE {np.sqrt((gb.d_peak**2).mean()):6.1f} mm   "
          f"peak bias {gb.d_peak.mean():+6.1f} mm   "
          f"melt-out ME {gb.d_tmelt.mean():+5.1f} d   "
          f"t-peak bias {gb.d_tpeak.mean():+5.1f} d")
    print(f"   /obs IQR   RMSE/IQR {np.sqrt((gb.niqr_peak**2).mean()):5.2f}   "
          f"median error {gb.niqr_peak.median():+5.2f} IQR")
    print(f"   /mean obs  nRMSE     {100*np.sqrt((gb.nd_peak**2).mean()):5.1f} %    "
          f"peak bias {100*gb.nd_peak.mean():+5.1f} %    "
          f"melt-out ME {100*gb.nd_tmelt.mean():+4.1f} %   "
          f"t-peak bias {100*gb.nd_tpeak.mean():+4.1f} %")
    print(f"   /season    "
          f"{'':22s}melt-out ME {100*gb.ns_tmelt.mean():+4.1f} %   "
          f"t-peak bias {100*gb.ns_tpeak.mean():+4.1f} %")
    print(f"   melt duration (peak->melt-out)  obs {gb.dur_obs.mean():5.1f} d   "
          f"model {gb.dur_mod.mean():5.1f} d   error {gb.d_dur.mean():+5.1f} d "
          f"(median {gb.d_dur.median():+.1f})")
