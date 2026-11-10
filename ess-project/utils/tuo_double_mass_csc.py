"""Double mass curve (DMC) of Lyell Fork below Maclure (LBM, glacierized: Lyell Glacier) vs Dana Fork at Bug Camp (DFkBC, non-glacierized reference); independent drainages.
Cumulative daily specific discharge (mm) from May 1 (both gauges present; QC'd data to 2021-08-12). Baseline slope s0 = OLS slope of the DMC over May 1 - day 180 (Jun 29) [literal] or the 10-60% span of Dana's cumulative May-Jul volume [meltspan]; late-summer onset (DateOn) = first date after which the 7-day local DMC slope
stays above s0 (+ tolerance) until termination = first date the local slope falls below s0 for 7 days. Cryospheric streamflow contribution CSC = sum over [DateOn, termination] of (q_LBM - s0*q_DFkBC) in mm of Lyell basin, error from the s0 standard error.
-> figures/draft/tuo_dmc_lyell_dana.png, tuo_dmc_lyell_dana_csc.png, tuo_dmc_lyell_dana_table.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC = "/scratch/dlhogan/ess-project-data/"; R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = R+"figures/draft/"
A = pd.read_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv").set_index("id").area_km2
n = pd.read_csv(SC+"streamflow-data/TuolumneRiver/TuolumneStreamflow_Hydroclimate_2002_2021.csv", parse_dates=["Date"]).set_index("Date")
n = n.loc[:"2021-08-12"]; q = pd.DataFrame({"G": n.Estimated_Discharge_LBM/A["LBM"]*86400/1e3, "N": n.Estimated_Discharge_DFkBC/A["DFkBC"]*86400/1e3})   # m3/s / km2 -> mm/d
TOL, W = 0.0, 7
MODE = __import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "meltspan"; SUF = "" if MODE == "literal" else "_meltspan"
rows = []; curves = {}
for y in range(2004, 2022):
    w = q.loc[f"{y}-05-01":f"{y}-10-31"].copy()
    if len(w) < 150: continue
    ok = w.notna().all(axis=1); gap = (~ok).sum()
    if gap > 12 or not ok.iloc[:30].all() or ok.iloc[-45:-14].sum() < 30: rows.append(dict(year=y, status=f"skip ({gap} missing d)")); continue
    w = w.interpolate(limit=10).dropna(); doy = w.index.dayofyear.values; cG = w.G.cumsum(); cN = w.N.cumsum()
    if MODE == 'literal': base = doy <= 180
    else:
        tot = cN[(w.index.month <= 7)].iloc[-1]; base = ((cN >= .10*tot) & (cN <= .60*tot)).values
    X = np.c_[np.ones(base.sum()), cN[base].values]; b, res, *_ = np.linalg.lstsq(X, cG[base].values, rcond=None); s0 = b[1]
    sig2 = res[0]/(base.sum()-2) if len(res) else np.nan; se = np.sqrt(sig2/((cN[base]-cN[base].mean())**2).sum()) if base.sum() > 2 else np.nan
    loc = (cG.diff(W)/cN.diff(W)).shift(-(W//2))     # centred 7-d local slope
    after = (doy > (180 if MODE == 'literal' else doy[np.where(base)[0][-1]])); hi = (loc > s0+TOL).values; on = term = None
    for i in np.where(after)[0]:
        if hi[i:i+W].all(): on = i; break
    if on is not None:
        for i in range(on+W, len(w)):
            if (~hi[i:i+W]).all(): term = i; break
        if term is None: term = len(w)-1
        sl = slice(on, term+1); csc = (w.G.values[sl]-s0*w.N.values[sl]); csc = csc.clip(min=None).sum(); cscse = se*w.N.values[sl].sum(); vol = w.G.values[sl].sum()
        rows.append(dict(year=y, status="ok", s0=s0, s0_se=se, date_on=w.index[on].date(), date_off=w.index[term].date(), days=term-on+1, lyell_mm=vol, csc_mm=csc, csc_err_mm=cscse, csc_frac=csc/vol, dana_mm=w.N.values[sl].sum()))
    else: rows.append(dict(year=y, status="ok (no onset)", s0=s0, s0_se=se))
    curves[y] = (w, cN, cG, s0, loc, on, term)
T = pd.DataFrame(rows); T.to_csv(D+f"tuo_dmc_lyell_dana_table{SUF}.csv", index=False); print(T.round(3).to_string(index=False))
ys = list(curves); nc = 5; nr = int(np.ceil(len(ys)/nc)); plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(nr, nc, figsize=(3.3*nc, 3.1*nr)); ax = ax.ravel()
for a, y in zip(ax, ys):
    w, cN, cG, s0, loc, on, term = curves[y]; a.plot(cN, cG, color="#2a78d6", lw=1.6); a.plot(cN, s0*cN+(cG.iloc[0]-s0*cN.iloc[0]), "--", color="#999", lw=.9)
    if on is not None: a.plot(cN.iloc[on], cG.iloc[on], "o", color="#d55e00", ms=5); a.plot(cN.iloc[term], cG.iloc[term], "s", color="#009e73", ms=5)
    r = T[T.year == y].iloc[0]; a.set_title(f"{y}  s0={s0:.2f}" + (f"  on {pd.Timestamp(r.date_on):%b %d}–{pd.Timestamp(r.date_off):%b %d}" if on is not None else ""), fontsize=8, loc="left"); a.grid(color="#e3e2de", lw=.5)
for a in ax[len(ys):]: a.axis("off")
fig.supxlabel("cumulative Dana Fork (DFkBC) specific discharge from May 1 (mm)"); fig.supylabel("cumulative Lyell (LBM) specific discharge (mm)"); fig.tight_layout(); fig.savefig(D+f"tuo_dmc_lyell_dana{SUF}.png", dpi=130, bbox_inches="tight")
ok = T[T.status == "ok"]; fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].errorbar(ok.year, ok.csc_mm, ok.csc_err_mm, fmt="o-", color="#2a78d6", capsize=2); ax[0].set_ylabel("CSC (mm of Lyell basin)"); ax[0].set_title("Cryospheric streamflow contribution", loc="left")
ax[1].plot(ok.year, ok.csc_frac, "o-", color="#d55e00"); ax[1].set_ylabel("CSC / Lyell flow in window"); ax[1].set_title("Fraction of late-summer Lyell flow", loc="left")
for _, r in ok.iterrows(): ax[2].plot([r.year]*2, [pd.Timestamp(r.date_on).dayofyear, pd.Timestamp(r.date_off).dayofyear], "-", color="#52514e", lw=3)
ax[2].set_ylabel("day of year"); ax[2].set_title("DateOn → termination", loc="left")
for a in ax: a.grid(color="#e3e2de", lw=.5)
fig.tight_layout(); fig.savefig(D+f"tuo_dmc_lyell_dana_csc{SUF}.png", dpi=130, bbox_inches="tight")
