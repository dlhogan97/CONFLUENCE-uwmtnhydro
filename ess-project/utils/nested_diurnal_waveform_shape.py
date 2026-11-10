"""Waveform shape at the four nested headwater gauges, Aug 1 - Sep 30 (QC'd record to 2021-08-12), same method as usgs_diurnal_waveform_shape.py: relative anomaly = Q / centred 24-h running mean - 1 on precipitation-screened days (no wet day within +-1 d).
-> figures/draft/nested_diurnal_waveform_shape.png"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"
DN = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/TuolumneStreamflow_2002_2021/data/"; F = {"T120": "Tuolumne120_streamflow.csv", "LBM": "LyellBlwMaclure_streamflow.csv", "LABvTB": "LyellAbvTB_streamflow.csv", "DFkBC": "DanaFkBugCamp_streamflow.csv"}
met = pd.read_csv(OBS+"tuolumne_daily_met.csv", index_col=0, parse_dates=True); tum = pd.read_csv(OBS+"tum_daily_precip.csv", index_col=0, parse_dates=True)
w = ((met.P_basin >= 0.5) | (tum.increment_mm.reindex(met.index) >= 0.5)).fillna(False); wet3 = w | w.shift(1, fill_value=False) | w.shift(-1, fill_value=False); CUT = pd.Timestamp("2021-08-12 23:59")
def smooth(c, n=5): k = np.ones(n)/n; ext = np.r_[c[-(n//2):], c, c[:n//2]]; return np.convolve(ext, k, "valid")
def metrics(C, per_day):
    c = smooth(np.median(C, axis=0), 3 if per_day == 24 else 5); th = np.arange(len(c))*24/len(c); imax, imin = int(np.argmax(c)), int(np.argmin(c)); rise = ((imax-imin) % len(c))*24/len(c); g = np.gradient(np.r_[c[-1], c, c[0]])[1:-1]*len(c)/24
    return dict(n_days=len(C), hour_max=th[imax], hour_min=th[imin], rise_h=rise, fall_h=24-rise, ratio=g.max()/-g.min(), amp_pct=100*(c.max()-c.min())/2, curve=c, th=th)
res = {}
for g, f in F.items():
    d = pd.read_csv(DN+f, comment="#", parse_dates=["DateTime"]); d = d[(d.DateTime <= CUT) & d.Discharge_Flag.isin([0, 1])]; d["loc"] = d.DateTime-pd.Timedelta(hours=7); ser = d.set_index("loc").Estimated_Discharge.sort_index(); ser = ser[~ser.index.duplicated()]
    step = ser.index.to_series().diff().median(); per = int(round(pd.Timedelta(hours=24)/step)); full = ser.resample(f"{int(step.total_seconds()//60)}min").mean().interpolate(limit=3); base = full.rolling(per, center=True, min_periods=int(.9*per)).mean(); anom = full/base-1; curves = {}
    for day in anom.index.floor("D").unique():
        if day.month not in (8, 9) or wet3.get(day, True): continue
        seg = anom[day:day+pd.Timedelta(hours=23, minutes=59)]
        if len(seg) == per and seg.notna().all() and ser[day:day+pd.Timedelta(hours=23, minutes=59)].mean() >= 0.01: curves[day] = seg.values
    if len(curves) >= 30: res[g] = metrics(np.array(list(curves.values())), per); print(f"{g}: days {len(curves)}, years {len(set(k.year for k in curves))}; step {step}; max {res[g]['hour_max']:.1f} h, min {res[g]['hour_min']:.1f} h, rise {res[g]['rise_h']:.1f} h vs fall {res[g]['fall_h']:.1f} h, steepest rise/fall {res[g]['ratio']:.2f}, amplitude ±{res[g]['amp_pct']:.1f}%")
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
for (g, r), c in zip(res.items(), ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]): ax.plot(r["th"], r["curve"]*100, color=c, lw=1.8, label=f"{g} (n={r['n_days']})")
ax.axhline(0, color="#999", lw=.7); ax.axvspan(5, 9, color="#d55e00", alpha=.08); ax.axvspan(15, 19, color="#2a78d6", alpha=.08); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("local hour (PDT)"); ax.set_ylabel("relative flow anomaly (% of daily mean)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Aug–Sep mean daily cycle at the nested headwater gauges (precip-screened)", fontsize=10, loc="left"); ax.grid(color="#e3e2de", lw=.5); fig.tight_layout(); fig.savefig(D+"nested_diurnal_waveform_shape.png", dpi=140)
