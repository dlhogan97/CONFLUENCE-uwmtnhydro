"""(v2: circular statistics + hour-by-day density; the first version unwrapped around 14 h and mishandled the early-season night-time peak.)
Hour of maximum daily flow at USGS 11274790, May 1 - Sep 30, 2007-2025 (15-min NWIS, local PDT). Per day: resample to 48 half-hour bins (>=44 valid, gaps<=2 interpolated), remove linear trend, rfft; k=1 (24 h) -> hour of max, relative and absolute amplitude, power fraction.
All days are used (no quiet-day screen: the melt-season rising limb is the signal); 'clear' = 24-h component carries >=30% of the non-DC power. Phase aggregated with circular statistics (amplitude-weighted circular mean), unwrapped around 14 h for plotting.
-> figures/draft/usgs_diurnal_hour_maysep_v2.png, usgs_diurnal_hour_maysep_v2_days.csv, usgs_diurnal_hour_maysep_v2_windows.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import theilslopes, binomtest, ttest_1samp, spearmanr
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; OBS = U+"modeling/calibration/observations/"; D = U+"figures/draft/"
d = pd.concat([pd.read_csv(OBS+"usgs11274790_iv_mayjul.csv"), pd.read_csv(OBS+"usgs11274790_iv_augoct.csv")]); d = d[d.tz.isin(["PDT"])].copy(); d["loc"] = pd.to_datetime(d.datetime, errors="coerce"); d["q"] = pd.to_numeric(d.q_cms, errors="coerce"); d = d.dropna(subset=["loc", "q"]).drop_duplicates("loc")
d = d[(d["loc"].dt.month >= 5) & (d["loc"].dt.month <= 9)]; d["day"] = d["loc"].dt.floor("D"); rows = []
for day, x in d.groupby("day"):
    t = (x["loc"].dt.hour*60+x["loc"].dt.minute).values/30.0; b = pd.Series(x.q.values).groupby(np.floor(t).astype(int)).mean().reindex(range(48))
    if b.notna().sum() < 44: continue
    b = b.interpolate(limit=2, limit_direction="both")
    if b.isna().any(): continue
    n = np.arange(48); y = b.values-np.polyval(np.polyfit(n, b.values, 1), n); X = np.fft.rfft(y); P = np.abs(X[1:])**2
    rows.append(dict(date=day, year=day.year, doy=day.dayofyear, hour_max=((-np.angle(X[1])*24/(2*np.pi)) % 24+0.25) % 24, amp_abs=2*np.abs(X[1])/48, amp_rel=2*np.abs(X[1])/48/b.mean(), power_frac=P[0]/P.sum(), q_mean=b.mean()))
Dd = pd.DataFrame(rows); Dd["h"] = ((Dd.hour_max-14+12) % 24)-12+14; Dd.to_csv(D+"usgs_diurnal_hour_maysep_v2_days.csv", index=False)
cl = Dd[Dd.power_frac >= 0.3].copy(); print("days:", len(Dd), " clear:", len(cl), " by month clear/all:", (cl.groupby(cl.date.dt.month).size().astype(str)+"/"+Dd.groupby(Dd.date.dt.month).size().astype(str)).to_dict())
def circ(h, w=None):
    w = np.ones(len(h)) if w is None else np.asarray(w); a = np.array(h)*2*np.pi/24; return (np.degrees(np.arctan2((w*np.sin(a)).sum(), (w*np.cos(a)).sum()))/15) % 24

cl["u"] = ((cl.hour_max-2+12) % 24)-12+2            # unwrap around 02 h: night regime ~1-3 h, noon regime ~12 h
cl["win"] = ((cl.doy-121)//7)*7+121+3; ndays = {}
def resultant(h, w): a = np.array(h)*2*np.pi/24; return np.hypot((w*np.sin(a)).sum(), (w*np.cos(a)).sum())/w.sum()
W = cl.groupby("win").apply(lambda x: pd.Series(dict(n=len(x), hour_circ=circ(x.hour_max, x.amp_abs), R=resultant(x.hour_max, x.amp_abs.values), frac_night=((x.u < 7)).mean(), amp_rel=x.amp_rel.median(), amp_abs=x.amp_abs.median(), power=x.power_frac.median(), q_mean=x.q_mean.median()))); W["date"] = (pd.Timestamp("2021-01-01")+pd.to_timedelta(W.index-1, unit="D")).strftime("%b %d"); W.to_csv(D+"usgs_diurnal_hour_maysep_v2_windows.csv"); pd.set_option("display.width", 200)
print("7-day windows: amplitude-weighted circular mean hour of max, concentration R (1 = all days same hour), fraction of days with a night-time (<~07 h) peak, amplitudes, flow"); print(W[["date", "n", "hour_circ", "R", "frac_night", "amp_rel", "amp_abs", "power", "q_mean"]].round(2).to_string())
# transition date per year: first date from which the daily hour of max stays in 09-16 h for >=5 consecutive clear days
tr = []
for y, x in cl.groupby("year"):
    x = x.sort_values("date"); ok = ((x.hour_max >= 9) & (x.hour_max <= 16)).astype(int).values; run = np.convolve(ok, np.ones(5), "valid") == 5; idx = np.where(run)[0]
    # last time a night-regime day (hour<7 or >20) occurred, +1
    nt = x[(x.hour_max < 7) | (x.hour_max > 20)]; last_night = nt.date.max() if len(nt) else None; tr.append(dict(year=y, first_noon_run=x.date.iloc[idx[0]] if len(idx) else pd.NaT, last_night_day=last_night, q_at_last_night=float(x[x.date == last_night].q_mean.iloc[0]) if last_night is not None else np.nan))
T = pd.DataFrame(tr); T["last_night_doy"] = T.last_night_day.dt.dayofyear; print("\ntransition (last day with a night-time peak, hour<7 or >20) per year:"); print(T.round(2).to_string(index=False)); print("median last-night day:", pd.to_datetime("2021-01-01")+pd.to_timedelta(T.last_night_doy.median()-1, unit="D"), "| median flow that day: %.1f m3/s" % T.q_at_last_night.median())
# night-regime drift: May 1 - Jul 10, days whose peak is at night/early morning (u in [-6,7)), per-year Theil-Sen
nd = cl[(cl.doy <= 191) & (cl.u > -6) & (cl.u < 7)]; sl = []
for y, x in nd.groupby("year"):
    if len(x) >= 25: sl.append(dict(year=y, n=len(x), drift_h_per_30d=theilslopes(x.u.values, x.doy.values)[0]*30, hour_start=x[x.doy <= 150].u.median(), hour_end=x[x.doy > 170].u.median(), rho_u_logq=spearmanr(x.u, np.log10(x.q_mean))[0]))
S = pd.DataFrame(sl); print("\nNight-regime drift, May 1-Jul 10 (days with peak 20h-07h):"); print(S.round(2).to_string(index=False)); s = S.drift_h_per_30d
print(f"{len(S)} years: median drift {s.median():+.2f} h/30d (mean {s.mean():+.2f}); later in {int((s>0).sum())}/{len(s)}; sign-test p={binomtest(int((s>0).sum()), len(s), .5).pvalue:.4f}; t-test p={ttest_1samp(s, 0).pvalue:.4f}; within-year rho(hour, log flow): median {S.rho_u_logq.median():+.2f}")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(2, 2, figsize=(15, 9)); xt = [121, 152, 182, 213, 244, 274]; xl = ["May 1", "Jun 1", "Jul 1", "Aug 1", "Sep 1", "Oct 1"]
a = ax[0, 0]; H, xe, ye = np.histogram2d(cl.doy, cl.hour_max, bins=[np.arange(121, 275, 3), np.arange(0, 25, 1)]); im = a.imshow((H/np.maximum(H.sum(1, keepdims=True), 1)).T, origin="lower", aspect="auto", extent=[121, 274, 0, 24], cmap="magma_r"); a.plot(W.index, W.hour_circ, "c-o", ms=3, lw=1.3, label="amplitude-weighted circular mean (7-day)"); a.set_yticks(range(0, 25, 3)); a.set_xticks(xt); a.set_xticklabels(xl); a.set_ylabel("hour of max flow (local PDT)"); a.set_title(f"Where the daily flow maximum falls, USGS 11274790 (all years, {len(cl)} clear days)", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=7.5, loc="upper right"); fig.colorbar(im, ax=a, label="fraction of days in each 3-day window")
a = ax[0, 1]; sc = a.scatter(cl.doy, cl.hour_max, c=np.log10(cl.q_mean), cmap="viridis", s=5, alpha=.6); a.set_yticks(range(0, 25, 3)); a.set_xticks(xt); a.set_xticklabels(xl); a.set_ylabel("hour of max flow (local PDT)"); a.set_title("Each day, coloured by log10 daily flow", fontsize=10, loc="left"); fig.colorbar(sc, ax=a, label="log10 Q (m³/s)")
a = ax[1, 0]; a.plot(W.index, W.frac_night*100, "o-", color="#d55e00", label="% of days with a night/early-morning peak"); a.set_ylabel("% of clear days"); a.set_ylim(0, 105); b = a.twinx(); b.plot(W.index, W.amp_abs, "s--", color="#2a78d6", label="absolute 24-h amplitude (m³/s)"); b.set_yscale("log"); b.set_ylabel("absolute amplitude (m³/s, log)", color="#2a78d6"); b.spines["top"].set_visible(False); a.set_xticks(xt); a.set_xticklabels(xl); a.set_title("Night-time-peak regime fades as the melt wave decays", fontsize=10, loc="left"); a.legend(frameon=False, fontsize=8, loc="center left")
a = ax[1, 1]
for y, x in cl.groupby("year"): a.plot(x[x.doy <= 191].groupby((x[x.doy <= 191].doy//7)*7).u.median().index+3, x[x.doy <= 191].groupby((x[x.doy <= 191].doy//7)*7).u.median().values, color=plt.cm.viridis((y-2007)/18), lw=.9, alpha=.8)
a.set_xticks([121, 135, 152, 166, 182]); a.set_xticklabels(["May 1", "May 15", "Jun 1", "Jun 15", "Jul 1"]); a.set_ylabel("weekly median hour of max (unwrapped about 02 h)"); a.set_title("May–early July by year (colour = year)", fontsize=10, loc="left"); a.set_ylim(-6, 14); a.grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+"usgs_diurnal_hour_maysep_v2.png", dpi=140, bbox_inches="tight")
