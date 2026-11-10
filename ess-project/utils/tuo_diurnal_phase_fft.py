"""Does the hour of daily maximum streamflow shift (season, year, snow retreat)?  FFT per day at T120 (QC'd, 30-min, UTC-7) and USGS 11274790 (15-min, PDT).
Per quiet day (>=5 d after any >5% daily-mean rise; daily mean >= 0.01 m3/s): resample to 48 half-hour bins (mean), interpolate gaps <=2 bins (need >=44 valid bins), remove the linear trend, rfft.
k=1 (24 h) component -> amplitude (fraction of daily mean), hour of maximum = -arg(X1)*24/(2 pi) mod 24, and power fraction = |X1|^2/sum_k |Xk|^2 (k>=1) (how dominant the 24-h cycle is).  Days with power fraction >= 0.3 count as 'clear diurnal days'.
Tests: (a) within-year drift of the hour of max with day of season (per-year Theil-Sen slope, h per 30 d; t-test and sign test over years), (b) monthly median hour of max, (c) year-to-year: median hour of max (Aug-Sep) and drift rate vs end-of-season snow area and snow area lost Aug-1 -> Sep-30, (d) vented vs unvented sensor era at T120.
-> figures/draft/tuo_diurnal_phase_fft.png, tuo_diurnal_phase_fft_days.csv, tuo_diurnal_phase_fft_years.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, theilslopes, ttest_1samp, binomtest
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"
DN = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/TuolumneStreamflow_2002_2021/data/"; CUT = pd.Timestamp("2021-08-12 23:59"); QMIN = 0.01
def load(g):
    if g == "USGS11274790":
        d = pd.read_csv(OBS+"usgs11274790_iv_augoct.csv"); d = d[d.tz == "PDT"].copy(); d["loc"] = pd.to_datetime(d.datetime); d["q"] = pd.to_numeric(d.q_cms, errors="coerce"); return d[["loc", "q"]].dropna()
    d = pd.read_csv(DN+"Tuolumne120_streamflow.csv", comment="#", parse_dates=["DateTime"]); d = d[(d.DateTime <= CUT) & d.Discharge_Flag.isin([0, 1])]; d["loc"] = d.DateTime-pd.Timedelta(hours=7); d["q"] = d.Estimated_Discharge; d = d[d["loc"].dt.month.isin([8, 9, 10])]; return d[["loc", "q"]].dropna()
def day_fft(x):
    t = (x["loc"].dt.hour*60+x["loc"].dt.minute).values/30.0; b = pd.Series(x.q.values).groupby(np.floor(t).astype(int)).mean().reindex(range(48))
    if b.notna().sum() < 44: return None
    b = b.interpolate(limit=2, limit_direction="both")
    if b.isna().any(): return None
    y = b.values; n = np.arange(48); y = y-np.polyval(np.polyfit(n, y, 1), n)+0*y; X = np.fft.rfft(y); P = np.abs(X[1:])**2
    # half-hour bin centres are at (i+0.5)/2 h: shift phase to hours
    ph = (-np.angle(X[1])*24/(2*np.pi)) % 24; ph = (ph+0.25) % 24; return dict(amp_rel=2*np.abs(X[1])/48/b.mean(), hour_max=ph, power_frac=P[0]/P.sum(), q_mean=b.mean())
rows = []
for g in ["T120", "USGS11274790"]:
    d = load(g); d["day"] = d["loc"].dt.floor("D"); dm = d.groupby("day").q.mean().asfreq("D"); quiet = (dm.pct_change() > 0.05).astype(float).rolling(5, min_periods=5).sum().eq(0)
    for day, x in d.groupby("day"):
        if len(x) < 20: continue
        r = day_fft(x)
        if r is None or r["q_mean"] < QMIN: continue
        rows.append(dict(gauge=g, date=day, quiet=bool(quiet.get(day, False)), doy=(day-pd.Timestamp(day.year, 8, 1)).days, year=day.year, **r))
Dd = pd.DataFrame(rows); Dd.to_csv(D+"tuo_diurnal_phase_fft_days.csv", index=False)
# snow area series from budget script (daily, Aug 1-Sep 30)
bs = open(U+"utils/tuo_patch_melt_budget.py").read(); bns = {}; exec(compile(bs[:bs.index("B = pd.DataFrame(rows)")], "b", "exec"), bns); AS = bns["series"]; eos = pd.read_csv(FS+"eos_final.csv").set_index("year").A_shadow
clear = Dd[Dd.quiet & (Dd.power_frac >= 0.3)].copy(); clear["h"] = ((clear.hour_max-13+12) % 24)-12+13       # unwrap around 13 h
yr = []
for (g, y), x in clear.groupby(["gauge", "year"]):
    xa = x[x.date.dt.month.isin([8, 9])]
    if len(xa) >= 12:
        sl = theilslopes(xa.h.values, xa.doy.values)[0]*30; med = xa.h.median(); amp = xa.amp_rel.median()
        yr.append(dict(gauge=g, year=y, n_days=len(xa), hour_max_med=med, drift_h_per_30d=sl, amp_rel_med=amp, snow_eos=eos.get(y, np.nan), snow_aug1=AS[y].iloc[0] if y in AS else np.nan, snow_lost=(AS[y].iloc[0]-AS[y].iloc[-1]) if y in AS else np.nan))
Y = pd.DataFrame(yr); Y.to_csv(D+"tuo_diurnal_phase_fft_years.csv", index=False); pd.set_option("display.width", 200)
print("days (all / quiet / clear-diurnal):"); print(Dd.groupby("gauge").agg(all=("date", "size"), quiet=("quiet", "sum")).join(clear.groupby("gauge").size().rename("clear")))
print("\npower fraction at 24 h, median:", Dd.groupby("gauge").power_frac.median().round(2).to_dict()); print("median hour of max (clear days) by month:"); print(clear.assign(month=clear.date.dt.month).groupby(["gauge", "month"]).h.median().unstack().round(1))
for g, x in Y.groupby("gauge"):
    s = x.drift_h_per_30d; t = ttest_1samp(s, 0); sg = binomtest(int((s > 0).sum()), len(s), 0.5)
    print(f"\n{g}: years {len(x)}; median hour of max {x.hour_max_med.median():.1f} h (range {x.hour_max_med.min():.1f}-{x.hour_max_med.max():.1f}); within-season drift {s.mean():+.2f} h/30d (SE {s.std()/np.sqrt(len(s)):.2f}; t-test p={t.pvalue:.3f}; {int((s > 0).sum())}/{len(s)} years later; sign-test p={sg.pvalue:.3f})")
    for a, b in (("hour_max_med", "snow_eos"), ("drift_h_per_30d", "snow_eos"), ("drift_h_per_30d", "snow_lost"), ("hour_max_med", "snow_aug1"), ("amp_rel_med", "snow_eos")):
        z = x[[a, b]].dropna(); r, p = spearmanr(z[a], z[b]); print(f"   {a:16s} vs {b:10s}: rho={r:+.2f} p={p:.3f} n={len(z)}")
x = Y[Y.gauge == "T120"]; 
for lab, z in (("unvented (<=2015)", x[x.year <= 2015]), ("vented (>=2016)", x[x.year >= 2016])): print(f"T120 {lab}: n={len(z)} median hour {z.hour_max_med.median():.1f}, drift {z.drift_h_per_30d.mean():+.2f} h/30d")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(2, 4, figsize=(20, 8))
for i, g in enumerate(["T120", "USGS11274790"]):
    x = clear[clear.gauge == g]; a = ax[i, 0]; sc = a.scatter(x.doy, x.h, c=x.year, cmap="viridis", s=7, alpha=.6); mm = x.assign(b=(x.doy//15)*15).groupby("b").h.median(); a.plot(mm.index+7, mm.values, "k-o", ms=4, lw=1.5); a.set_xlabel("days since Aug 1"); a.set_ylabel("hour of max flow (FFT, local)"); a.set_title(f"{g}: daily hour of max (clear diurnal days, n={len(x)})", fontsize=9, loc="left"); a.set_ylim(4, 24); a.grid(color="#e3e2de", lw=.5)
    y_ = Y[Y.gauge == g]; a = ax[i, 1]; a.bar(y_.year, y_.drift_h_per_30d, color=["#d55e00" if v > 0 else "#2a78d6" for v in y_.drift_h_per_30d]); a.axhline(0, color="#52514e", lw=.8); a.set_title(f"{g}: within-season drift of hour of max (h per 30 d, Aug-Sep)", fontsize=9, loc="left"); a.set_xlabel("year")
    a = ax[i, 2]; z = y_.dropna(subset=["snow_eos"]); a.scatter(z.snow_eos, z.hour_max_med, c=z.year, cmap="viridis", s=40); [a.annotate(str(int(r.year))[2:], (r.snow_eos, r.hour_max_med), fontsize=7, xytext=(3, 3), textcoords="offset points") for r in z.itertuples()]; a.set_xscale("symlog", linthresh=0.5); r_, p_ = spearmanr(z.snow_eos, z.hour_max_med); a.set_title(f"{g}: median hour of max vs snow area  ρ={r_:+.2f} (p={p_:.2f}, n={len(z)})", fontsize=9, loc="left"); a.set_xlabel("end-of-season snow area (km²)"); a.set_ylabel("hour of max")
    a = ax[i, 3]; z = y_.dropna(subset=["snow_lost"]); a.scatter(z.snow_lost, z.drift_h_per_30d, c=z.year, cmap="viridis", s=40); [a.annotate(str(int(r.year))[2:], (r.snow_lost, r.drift_h_per_30d), fontsize=7, xytext=(3, 3), textcoords="offset points") for r in z.itertuples()]; a.set_xscale("symlog", linthresh=0.5); a.axhline(0, color="#999", lw=.7); r_, p_ = spearmanr(z.snow_lost, z.drift_h_per_30d); a.set_title(f"{g}: drift vs snow area lost Aug1→Sep30  ρ={r_:+.2f} (p={p_:.2f}, n={len(z)})", fontsize=9, loc="left"); a.set_xlabel("snow area lost (km²)"); a.set_ylabel("drift (h/30 d)")
fig.colorbar(sc, ax=ax, fraction=0.012, label="year"); fig.savefig(D+"tuo_diurnal_phase_fft.png", dpi=140, bbox_inches="tight")
