"""Shape of the mean daily flow cycle at USGS 11274790 by month (May-Sep), precipitation-screened clear-diurnal days (usgs_diurnal_precip_screened_days.csv).
Per day: 15-min discharge / centred 24-h running mean - 1 (relative anomaly; no calendar-day edge effects). Composite = median across days at each 15-min time of day (5-point circular smoothing).
Shape metrics: hour of max / min, rise duration (min->max, h), fall duration (max->min), asymmetry, steepest rise vs steepest fall, 2nd-harmonic share. ET-dominated streamflow: max in the morning, slow rise through the night, sharp morning fall.
Clustered bootstrap over years for the Aug-Sep metrics. -> figures/draft/usgs_diurnal_waveform_shape.png, usgs_diurnal_waveform_metrics.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"
S = pd.read_csv(D+"usgs_diurnal_precip_screened_days.csv", parse_dates=["date"]); days = set(S.date.dt.floor("D"))
d = pd.concat([pd.read_csv(OBS+"usgs11274790_iv_mayjul.csv"), pd.read_csv(OBS+"usgs11274790_iv_augoct.csv")]); d = d[d.tz == "PDT"].copy(); d["loc"] = pd.to_datetime(d.datetime, errors="coerce"); d["q"] = pd.to_numeric(d.q_cms, errors="coerce"); d = d.dropna(subset=["loc", "q"]).drop_duplicates("loc"); d["day"] = d["loc"].dt.floor("D"); d = d[d.day.isin(days)]
# continuous 15-min series; baseline = centred 24-h running mean (no calendar-day boundary, no trend assumption); anomaly = q / baseline - 1
ser = d.drop_duplicates("loc").set_index("loc").q.sort_index(); full = ser.resample("15min").mean(); full = full.interpolate(limit=3); base = full.rolling(96, center=True, min_periods=90).mean(); anom = (full/base-1)
curves = {}
for day in sorted(days):
    seg = anom[day:day+pd.Timedelta(hours=23, minutes=45)]
    if len(seg) == 96 and seg.notna().all(): curves[day] = seg.values
print("days with full 15-min coverage:", len(curves))
def smooth(c): k = np.ones(5)/5; ext = np.r_[c[-2:], c, c[:2]]; return np.convolve(ext, k, "valid")
def metrics(C):
    c = smooth(np.median(C, axis=0)); th = np.arange(96)/4; imax, imin = int(np.argmax(c)), int(np.argmin(c)); rise = ((imax-imin) % 96)/4; fall = 24-rise; g = np.gradient(np.r_[c[-1], c, c[0]])[1:-1]*4
    F = np.abs(np.fft.rfft(c-c.mean()))**2
    return dict(n_days=len(C), hour_max=th[imax], hour_min=th[imin], rise_h=rise, fall_h=fall, asym=(rise-fall)/24, steep_rise=g.max(), steep_fall=-g.min(), ratio=g.max()/-g.min(), amp_pct=100*(c.max()-c.min())/2, harm2_share=F[2]/F[1:].sum(), curve=c)
months = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}; M = {}; rows = []
for m, nm in months.items():
    C = np.array([v for k, v in curves.items() if k.month == m]); 
    if len(C) >= 20: M[m] = metrics(C); rows.append({k: v for k, v in M[m].items() if k != "curve"} | dict(month=nm))
Rm = pd.DataFrame(rows).set_index("month"); pd.set_option("display.width", 200); print(Rm.round(2).to_string())
# Aug-Sep clustered bootstrap over years
keys = [k for k in curves if k.month in (8, 9)]; yrs = sorted(set(k.year for k in keys)); rng = np.random.default_rng(1); boots = []
for _ in range(500):
    ys = rng.choice(yrs, len(yrs)); C = np.array([curves[k] for y in ys for k in keys if k.year == y]); mm = metrics(C); boots.append([mm["hour_max"], mm["hour_min"], mm["rise_h"], mm["steep_rise"]/mm["steep_fall"]])
B = np.array(boots); ASm = metrics(np.array([curves[k] for k in keys]))
print("\nAug-Sep (n=%d days, %d years): max at %.2f h (90%% CI %.1f-%.1f), min at %.2f h (%.1f-%.1f), rise %.1f h (%.1f-%.1f) vs fall %.1f h, steepest-rise/steepest-fall %.2f (%.2f-%.2f)" % (len(keys), len(yrs), ASm["hour_max"], *np.percentile(B[:, 0], [5, 95]), ASm["hour_min"], *np.percentile(B[:, 1], [5, 95]), ASm["rise_h"], *np.percentile(B[:, 2], [5, 95]), ASm["fall_h"], ASm["ratio"], *np.percentile(B[:, 3], [5, 95])))
# per-year Aug-Sep metrics
py = []
for y in yrs:
    C = np.array([curves[k] for k in keys if k.year == y])
    if len(C) >= 25: mm = metrics(C); py.append(dict(year=y, n=len(C), hour_max=mm["hour_max"], hour_min=mm["hour_min"], rise_h=mm["rise_h"], ratio=mm["ratio"]))
PY = pd.DataFrame(py); print("\nper-year Aug-Sep composite:"); print(PY.round(1).to_string(index=False)); print("years with rise longer than fall:", int((PY.rise_h > 12).sum()), "/", len(PY), "; median hour_max %.1f, hour_min %.1f" % (PY.hour_max.median(), PY.hour_min.median()))
Rm.to_csv(D+"usgs_diurnal_waveform_metrics.csv")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 3, figsize=(18, 4.8)); th = np.arange(96)/4; cols = plt.cm.viridis(np.linspace(0, .9, 5))
for (m, mm), c in zip(M.items(), cols): ax[0].plot(th, mm["curve"]*100, color=c, lw=1.8, label=f"{months[m]} (n={mm['n_days']})")
ax[0].axhline(0, color="#999", lw=.7); ax[0].set_xticks(range(0, 25, 3)); ax[0].set_xlabel("local hour (PDT)"); ax[0].set_ylabel("relative flow anomaly (% of daily mean)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Mean daily cycle by month (precip-screened days)", fontsize=10, loc="left"); ax[0].grid(color="#e3e2de", lw=.5)
for (m, mm), c in zip(M.items(), cols): cc = mm["curve"]; ax[1].plot(th, (cc-cc.mean())/(cc.max()-cc.min())*2, color=c, lw=1.8)
ax[1].axvspan(5, 9, color="#d55e00", alpha=.12, label="where an ET-dominated max would sit"); ax[1].axvspan(15, 19, color="#2a78d6", alpha=.12, label="where an ET-dominated min would sit"); ax[1].set_xticks(range(0, 25, 3)); ax[1].set_xlabel("local hour"); ax[1].set_ylabel("anomaly scaled to ±1"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Shapes compared (amplitude-normalised)", fontsize=10, loc="left"); ax[1].grid(color="#e3e2de", lw=.5)
for y, c in zip(PY.year, plt.cm.viridis(np.linspace(0, .95, len(PY)))): cc = smooth(np.median(np.array([curves[k] for k in keys if k.year == y]), axis=0)); ax[2].plot(th, (cc-cc.mean())*100, color=c, lw=1, alpha=.85)
ax[2].set_xticks(range(0, 25, 3)); ax[2].set_xlabel("local hour"); ax[2].set_ylabel("Aug–Sep composite (% of daily mean)"); ax[2].set_title("Aug–Sep composite, each year (colour = year)", fontsize=10, loc="left"); ax[2].grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+"usgs_diurnal_waveform_shape.png", dpi=140, bbox_inches="tight")
