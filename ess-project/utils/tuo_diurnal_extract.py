"""Diurnal cycles of discharge and water temperature at the four nested Tuolumne gauges (QC'd record, ends 2021-08-12), Aug 1 - Oct 31.
Per eligible day: least-squares fit Q(t) = a0 + a1 t + b1 cos(2 pi h/24) + c1 sin(2 pi h/24) on local time (UTC-7). Reported: relative amplitude = amp/mean Q, phase hour of the daily maximum,
and the SIGNED snowmelt-like projection = rel_amp * cos(2 pi (phase-18)/24)  (positive: afternoon/evening maximum, as snowmelt travelling to the gauge; negative: early-morning maximum, as evapotranspiration drawdown).
Eligible day: >=20 samples with no gap >4 h, daily mean Q >= QMIN, no >5% day-to-day rise on this day or the previous 4 (>=5 d after any increase).
-> modeling/calibration/observations/tuo_diurnal_days.csv (one row per gauge-day), tuo_diurnal_annual.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
D = "/scratch/dlhogan/ess-project-data/streamflow-data/TuolumneRiver/TuolumneStreamflow_2002_2021/data/"; OBS = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/"
F = {"T120": "Tuolumne120_streamflow.csv", "LBM": "LyellBlwMaclure_streamflow.csv", "LABvTB": "LyellAbvTB_streamflow.csv", "DFkBC": "DanaFkBugCamp_streamflow.csv"}
QMIN = 0.01; CUT = pd.Timestamp("2021-08-12 23:59")
def harm(h, y):
    X = np.column_stack([np.ones_like(h), h-12.0, np.cos(2*np.pi*h/24), np.sin(2*np.pi*h/24)]); b = np.linalg.lstsq(X, y, rcond=None)[0]; amp = np.hypot(b[2], b[3]); ph = (np.degrees(np.arctan2(b[3], b[2]))/15.0) % 24; return b[0], amp, ph
rows = []
for g, f in F.items():
    d = pd.read_csv(D+f, comment="#", parse_dates=["DateTime"]); d = d[d.DateTime <= CUT]; d = d[d.Discharge_Flag.isin([0, 1])]; d["loc"] = d.DateTime-pd.Timedelta(hours=7); d = d[d["loc"].dt.month.isin([8, 9, 10])]
    d["day"] = d["loc"].dt.floor("D"); dm = d.groupby("day").Estimated_Discharge.mean(); dm = dm.asfreq("D"); rise = (dm.pct_change() > 0.05).astype(float); quiet = rise.rolling(5, min_periods=5).sum().eq(0)
    for day, x in d.groupby("day"):
        q = x.Estimated_Discharge.values; h = (x["loc"].dt.hour+x["loc"].dt.minute/60).values; ok = ~np.isnan(q)
        if ok.sum() < 20 or np.nanmax(np.diff(np.sort(h[ok]))) > 4: continue
        m = np.nanmean(q); rec = dict(gauge=g, date=day, n=int(ok.sum()), q_mean=m, quiet=bool(quiet.get(day, False)))
        if m >= QMIN: _, a, ph = harm(h[ok], q[ok]); rec.update(q_rel_amp=a/m, q_phase=ph, q_proj=a/m*np.cos(2*np.pi*(ph-18)/24))
        w = x.Water_Temperature.values; okw = ~np.isnan(w)
        if okw.sum() >= 20:
            m0, aw, pw = harm(h[okw], w[okw]); rec.update(wt_mean=m0, wt_amp=aw, wt_phase=pw, wt_range=np.nanmax(w)-np.nanmin(w))
        rows.append(rec)
Dd = pd.DataFrame(rows); Dd.to_csv(OBS+"tuo_diurnal_days.csv", index=False)
Dd["year"] = Dd.date.dt.year; Dd["win"] = np.where(Dd.date.dt.month.isin([8, 9]), "Aug-Sep", "Oct")
ann = []
for (g, y), x in Dd.groupby(["gauge", "year"]):
    for wname, xs in (("Aug-Oct", x), ("Aug-Sep", x[x.date.dt.month.isin([8, 9])])):
        e = xs[xs.quiet & xs.q_proj.notna()]
        ann.append(dict(gauge=g, year=y, window=wname, n_days=len(xs), n_quiet_flow=len(e), q_rel_amp=e.q_rel_amp.median() if len(e) >= 10 else np.nan, q_proj=e.q_proj.median() if len(e) >= 10 else np.nan,
                        frac_melt_like=(e.q_proj > 0).mean() if len(e) >= 10 else np.nan, phase_med=e.q_phase.median() if len(e) >= 10 else np.nan, q_mean=e.q_mean.median() if len(e) >= 10 else np.nan,
                        wt_mean=xs.wt_mean.median() if xs.wt_mean.notna().sum() >= 15 else np.nan, wt_range=xs.wt_range.median() if xs.wt_range.notna().sum() >= 15 else np.nan))
A = pd.DataFrame(ann); A.to_csv(OBS+"tuo_diurnal_annual.csv", index=False); pd.set_option("display.width", 200)
print(Dd.groupby("gauge").agg(days=("n", "size"), quiet_days=("quiet", "sum"), flow_days=("q_rel_amp", "count")).T)
print(A[A.window == "Aug-Oct"].pivot(index="year", columns="gauge", values="q_proj").round(3)); print(A[A.window == "Aug-Oct"].pivot(index="year", columns="gauge", values="wt_mean").round(1))
