"""Diagnostic: 7-day local DMC slope (dQ_Lyell/dQ_ref, specific discharge) vs day of year, per year, for refs DFkBC and T120. Lyell cumulative-free: ratio of 7-d sums. -> figures/draft/tuo_dmc_local_slope.png"""
import warnings; warnings.filterwarnings("ignore")
exec(open(__file__.replace("tuo_dmc_local_slope", "tuo_dmc_csc_multi")).read().split("def run")[0])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); import matplotlib.cm as cm
for a, ref in zip(ax[:2], ["DFkBC", "T120"]):
    ys = [y for y in range(2005, 2022)]
    for y in ys:
        w = pd.DataFrame({"G": Q["LBM"], "N": Q[ref]}).loc[f"{y}-05-01":f"{y}-10-31"].dropna(); 
        if len(w) < 120: continue
        r = (w.G.rolling(7, center=True).sum()/w.N.rolling(7, center=True).sum()); mega = y in EXC
        a.semilogy(r.index.dayofyear, r.values, color="#d55e00" if mega else "#2a78d6", lw=1.8 if mega else .9, alpha=1 if mega else .7); None
    a.set_title(f"Lyell / {ref} 7-d slope (red = 2011, 2017)", loc="left"); a.set_ylim(0.3, 200); a.set_xlabel("day of year"); a.grid(color="#e3e2de", lw=.5, which="both")
w = pd.DataFrame({"L": Q["LBM"], "D": Q["DFkBC"], "T": Q["T120"]}).dropna(subset=["L", "D"])
for y in [2013, 2015, 2018]: 
    x = w.loc[f"{y}-05-01":f"{y}-10-31"]; ax[2].semilogy(x.index.dayofyear, x.L, label=f"Lyell {y}"); ax[2].semilogy(x.index.dayofyear, x.D, "--", label=f"Dana {y}")
ax[2].legend(fontsize=6); ax[2].set_ylim(0.01, 100); ax[2].set_title("specific discharge (mm/d)", loc="left"); ax[2].grid(color="#e3e2de", lw=.5, which="both")
fig.tight_layout(); fig.savefig(D+"tuo_dmc_local_slope.png", dpi=125, bbox_inches="tight")
