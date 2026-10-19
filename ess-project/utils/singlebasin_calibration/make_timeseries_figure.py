import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fit_models as F, emu_core as E
O = F.OUT; FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/singlebasin_wet_dry_average_year.png"
M = ["M0", "M1", "M2", "M3"]
LAB = {"M0": "M0 baseline (recalibrated)", "M1": "M1 single-basin", "M2": "M2 single-basin, linear", "M3": "M3 single-basin, memory-fit"}
COL = {"M0": "#7a7a76", "M1": "#2a78d6", "M2": "#1baf7a", "M3": "#eb6834"}; LS = {"M0": "-", "M1": "-", "M2": "--", "M3": ":"}; LW = {"M0": 1.6, "M1": 2.0, "M2": 1.5, "M3": 1.8}
q = {m: pd.read_csv(O/f"summa_daily_{m}_full.csv", index_col=0, parse_dates=True)["q_cms"] for m in M}
obs = E.obs.copy()
wy = lambda s: s.index.year+(s.index.month >= 10)
ann = obs.loc["1986-10-01":"2024-09-30"].groupby(wy(obs.loc["1986-10-01":"2024-09-30"])).mean()
pick = {}
for lab, pr in (("Dry year", 0.10), ("Average year", 0.50), ("Wet year", 0.90)):
    t = ann.quantile(pr); y = int((ann-t).abs().idxmin()); pick[lab] = (y, float(ann[y]), pr)
print({k: (v[0], round(v[1], 2)) for k, v in pick.items()}, "| WY mean obs median %.2f"%ann.median())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e"})
fig, ax = plt.subplots(2, 3, figsize=(15, 7.6), constrained_layout=True, sharex="col")
for c, (lab, (y, mean, pr)) in enumerate(pick.items()):
    sl = slice(f"{y-1}-10-01", f"{y}-09-30")
    for r in (0, 1):
        a = ax[r, c]
        for m in M: a.plot(q[m].loc[sl].index, q[m].loc[sl].values, color=COL[m], ls=LS[m], lw=LW[m], label=LAB[m], zorder=3 if m == "M1" else 2)
        a.plot(obs.loc[sl].index, obs.loc[sl].values, color="#0b0b0b", lw=1.5, label="Observed (USGS 09112500)", zorder=4)
        if r == 1: a.set_yscale("log"); a.set_ylim(0.15, None); a.grid(axis="y", color="#e4e3df", lw=0.6, which="both")
        else: a.set_ylim(0, None); a.grid(axis="y", color="#e4e3df", lw=0.6)
        if c == 0: a.set_ylabel("discharge (m³/s), "+("linear" if r == 0 else "log scale"))
        if r == 0: a.set_title(f"{lab}: WY{y}\n(observed mean {mean:.1f} m³/s, P{int(pr*100)} of WY1987–2024)", fontsize=10, loc="left")
        import matplotlib.dates as mdates
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10, 1, 4, 7])); a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("East River daily flow in a dry, an average and a wet water year: observations vs the four full-window SUMMA runs (new singleBasin build)", fontsize=10.5, x=0.01, ha="left", color="#52514e")
fig.savefig(FIG, dpi=140, bbox_inches="tight"); print("saved", FIG)
# numbers for the text: low-flow (Aug-Nov) mean and peak by model in each picked year
rows = []
for lab, (y, mean, pr) in pick.items():
    for nm, s in [("Observed", obs)]+[(m, q[m]) for m in M]:
        w = s.loc[f"{y}-08-01":f"{y}-11-30"]; p = s.loc[f"{y}-04-01":f"{y}-07-31"].rolling(7, center=True).mean().max(); wy_ = s.loc[f"{y-1}-10-01":f"{y}-09-30"]
        rows.append(dict(year=f"{lab} WY{y}", series=nm, WYmean=round(wy_.mean(), 2), peak7=round(p, 1), AugNov_mean=round(w.mean(), 2), AugNov_min7=round(w.rolling(7, center=True).mean().min(), 2)))
df = pd.DataFrame(rows); print(df.to_string(index=False))
