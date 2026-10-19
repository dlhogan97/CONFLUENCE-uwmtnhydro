"""Tuolumne: snowpatch (anchor_m080, 37 HRUs) vs its MATCHED no-snowpatch control (baseline_wbias10, 28 HRUs): same params, same +10% winter precip, same +1.04 K offset. Key variables for the storage question."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xarray as xr, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/simulations/"; B = "/scratch/dlhogan/ess-project-data/calibration/bestruns/Tuolumne_River_baseline_wbias10_1981_2025/run/"
RUN = {"No snowpatch (control)": (B+"out/baseline_timestep.nc", B+"attributes.nc"), "With snowpatch bins": (S+"snowpatch_anchor_20260812/anchor_m080/output/anchor_m080_timestep.nc", S+"snowpatch_anchor_20260812/anchor_m080/attributes.nc")}
OBS = "/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/observations/streamflow/preprocessed/Tuolumne_River_merged_dailyQ_WY1971_2025.csv"
INK, BLUE, ORG, GRID = "#0b0b0b", "#2a78d6", "#eb6834", "#d9d8d4"; CL = {"No snowpatch (control)": BLUE, "With snowpatch bins": ORG}
def wy(i): return i.year + (i.month >= 10)
D = {}
for nm, (f, af) in RUN.items():
    d = xr.open_dataset(f); t = pd.DatetimeIndex(d.time.values); A = xr.open_dataset(af)["HRUarea"].values.ravel().astype(float); w = A/A.sum()
    dd = lambda v, avg=True: pd.DataFrame(d[v].values, index=t).resample("D").mean()
    q = (pd.Series(d["averageRoutedRunoff"].isel(gru=0).values, index=t)*A.sum()).resample("D").mean()
    swe = dd("scalarSWE"); flux = {k: (dd(v)*86400).mul(w, axis=1).sum(1) for k, v in (("precip", "pptrate"), ("ET", "scalarTotalET"), ("sublimation", "scalarSnowSublimation"))}
    D[nm] = dict(q=q, swe=swe, w=w, A=A, **flux)
o = pd.read_csv(OBS, parse_dates=["datetime"]).set_index("datetime"); obs = o["discharge_cms"]
def min7(q, months=(8, 9, 10)):
    r = q.where(q.index.month.isin(months)).rolling(7, center=True, min_periods=7).mean().resample("YS-OCT").min(); r.index = r.index.year+1; return r
def seasonal(q, months):
    x = q[q.index.month.isin(months)].resample("YS-OCT").mean(); x.index = x.index.year+1; return x
yrs = range(1984, 2026)
fig, ax = plt.subplots(3, 2, figsize=(13.5, 13)); ax = ax.ravel()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
# (a) annual Aug-Oct 7-day min
a = ax[0]; om = min7(obs).reindex(yrs); a.plot(om.index, om, color=INK, lw=2, marker="o", ms=3, label="Observed (HH-scaled to 2006, gauge after)")
for nm in D: m = min7(D[nm]["q"]).reindex(yrs); a.plot(m.index, m, color=CL[nm], lw=1.8, label=nm)
a.axvline(2006.5, color="#8a8984", ls=":", lw=1); a.text(2006.7, a.get_ylim()[1]*0.8 if False else 3.2, "real gauge →", fontsize=8, color="#52514e"); a.set_yscale("log"); a.set_title("(a) Annual Aug–Oct 7-day minimum flow", loc="left", fontsize=10); a.set_ylabel("m³/s (log)")
# (b) late-season flow difference
a = ax[1]; mm = {nm: seasonal(D[nm]["q"], [8, 9, 10]).reindex(yrs) for nm in D}; diff = mm["With snowpatch bins"]-mm["No snowpatch (control)"]; oo = seasonal(obs, [8, 9, 10]).reindex(yrs)
a.bar(diff.index, diff, color=ORG, width=0.8); a.plot(oo.index, oo*0+0, color=INK, lw=0.6); a.set_title(f"(b) Aug–Oct mean flow: snowpatch minus control  (mean {diff.mean():+.2f} m³/s = {100*diff.mean()/mm['No snowpatch (control)'].mean():+.0f}% of control)", loc="left", fontsize=10); a.set_ylabel("m³/s")
# (c) perennial store at Sep 30
a = ax[2]
for nm in D:
    sw = D[nm]["swe"]; sep = sw[(sw.index.month == 9) & (sw.index.day == 30)]; sep.index = sep.index.year; basin = (sep*D[nm]["w"]).sum(1) if False else sep.mul(D[nm]["w"], axis=1).sum(1)
    fsca = (sep > 10).mul(D[nm]["w"], axis=1).sum(1)*100; bb = basin.reindex(range(1984, 2026)); a.plot(bb.index, bb, color=CL[nm], lw=1.8, label=nm)
    a2 = a if False else None
a.set_title("(c) Perennial snow store: basin-mean SWE on 30 Sep (mm)", loc="left", fontsize=10); a.set_ylabel("mm"); 
# (d) mean Jun-Oct hydrograph
a = ax[3]; dowy = lambda q: q[(q.index >= "1984-10-01") & (q.index <= "2025-09-30")].groupby([q[(q.index >= '1984-10-01') & (q.index <= '2025-09-30')].index.month, q[(q.index >= '1984-10-01') & (q.index <= '2025-09-30')].index.day]).mean()
def clim(q):
    x = q[(q.index >= "1984-10-01") & (q.index <= "2025-09-30")]; g = x.groupby([x.index.month, x.index.day]).mean(); g.index = [pd.Timestamp(2001 if m < 10 else 2000, m, dday) for m, dday in g.index if not (m == 2 and dday == 29)] if False else g.index; return g
def clim_doy(q):
    x = q[(q.index >= "1984-10-01") & (q.index <= "2025-09-30")]; x = x[~((x.index.month == 2) & (x.index.day == 29))]; doy = (x.index - pd.to_datetime(np.where(x.index.month >= 10, x.index.year, x.index.year-1).astype(str)+"-10-01")).days
    return x.groupby(doy).mean()
oc = clim_doy(obs[obs.index >= "2007-10-01"]); a.plot(oc.index, oc, color=INK, lw=2, label="Observed (gauge era, WY2008–25)")
for nm in D: c = clim_doy(D[nm]["q"][D[nm]["q"].index >= "2007-10-01"]); a.plot(c.index, c, color=CL[nm], lw=1.8, label=nm)
a.set_xlim(240, 365); a.set_xticks([244, 274, 305, 335, 365][:0] or [244, 274, 305, 335]); a.set_xticklabels(["Jun 1", "Jul 1", "Aug 1", "Sep 1"][:0] or ["Jun 1", "Jul 1", "Aug 1", "Sep 1"]); a.set_xlim(240, 365); a.set_yscale("log"); a.set_title("(d) Mean summer hydrograph, WY2008–25 (log)", loc="left", fontsize=10); a.set_ylabel("m³/s")
# (e) water balance difference
a = ax[4]; names = ["precip", "ET", "sublimation"]; vals = {}
for k in names: vals[k] = [abs(D[nm][k][(D[nm][k].index >= "1984-10-01") & (D[nm][k].index <= "2025-09-30")].sum()/41.99) for nm in D]   # ET/sublimation are negative-up in SUMMA: report magnitudes
qa = [(D[nm]["q"][(D[nm]["q"].index >= "1984-10-01") & (D[nm]["q"].index <= "2025-09-30")].mean()*86400*365.25/D[nm]["A"].sum()*1000) for nm in D]; vals["streamflow"] = qa
labels = ["precipitation\n(mass source)", "ET", "sublimation", "streamflow"]; dv = [vals[k][1]-vals[k][0] for k in ["precip", "ET", "sublimation", "streamflow"]]
a.bar(range(4), dv, color=ORG); a.set_xticks(range(4)); a.set_xticklabels(labels); a.axhline(0, color=INK, lw=0.6); a.set_ylabel("mm / yr (snowpatch − control)"); a.set_title("(e) Where the added water goes (annual, WY1985–2025)", loc="left", fontsize=10)
for i, v in enumerate(dv): a.text(i, v, f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color="#52514e")
# (f) late-season flow vs Apr-Jul runoff
a = ax[5]; slopes = {}
for nm, q, c in (("Observed", obs, INK), *[(n, D[n]["q"], CL[n]) for n in D]):
    x = np.log(seasonal(q, [4, 5, 6, 7]).reindex(range(2008, 2026))); y = np.log(min7(q).reindex(range(2008, 2026))); ok = x.notna() & y.notna()
    X = np.column_stack([np.ones(ok.sum()), x[ok], (np.array(list(x[ok].index))-2000)/10.0]); beta = np.linalg.lstsq(X, y[ok].values, rcond=None)[0]; slopes[nm] = beta[2]
    a.scatter(np.exp(x[ok]), np.exp(y[ok]), color=c, s=22, alpha=0.9, label=f"{nm}: trend {beta[2]:+.2f}/decade")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Apr–Jul mean flow (m³/s)"); a.set_ylabel("Aug–Oct 7-day min (m³/s)"); a.set_title("(f) Late-season low flow vs spring runoff, WY2008–25 (trend given runoff)", loc="left", fontsize=10)
for i in (0, 1, 2, 3, 5): ax[i].grid(axis="y", color=GRID, lw=0.6); ax[i].set_axisbelow(True); ax[i].legend(frameon=False, fontsize=8)
ax[4].grid(axis="y", color=GRID, lw=0.6); ax[0].set_xlabel("water year"); ax[1].set_xlabel("water year"); ax[2].set_xlabel("water year")
fig.suptitle("Tuolumne: snowpatch vs matched no-snowpatch control (same parameters, +10% winter precip, +1.04 K)", fontsize=12, y=0.995); fig.tight_layout(rect=(0, 0, 1, 0.985))
out = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/tuolumne_snowpatch_vs_nopatch_key_variables.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
print("annual (mm/yr) control vs snowpatch:", {k: [round(float(v), 1) for v in vals[k]] for k in vals})
print("Aug-Oct mean flow m3/s: control %.2f snowpatch %.2f obs(gauge era) %.2f" % (mm["No snowpatch (control)"].mean(), mm["With snowpatch bins"].mean(), seasonal(obs, [8, 9, 10]).reindex(range(2008, 2026)).mean()))
print("conditional trend per decade (log7dmin ~ AprJul + decade), WY2008-25:", {k: round(float(v), 2) for k, v in slopes.items()})
for nm in D:
    sw = D[nm]["swe"]; sep = sw[(sw.index.month == 9) & (sw.index.day == 30) & (sw.index.year >= 1984)]; print(nm, "Sep-30 basin SWE mm: mean %.1f, last-10y mean %.1f; area with SWE>10mm: mean %.2f%%" % ((sep.mul(D[nm]["w"], axis=1).sum(1)).mean(), (sep.mul(D[nm]["w"], axis=1).sum(1)).iloc[-10:].mean(), 100*((sep > 10).mul(D[nm]["w"], axis=1).sum(1)).mean()))
