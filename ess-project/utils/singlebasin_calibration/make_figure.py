import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fit_models as F, emu_core as E
O = F.OUT; FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/singlebasin_calibration_summary.png"
M = ["M0", "M1", "M2", "M3"]; LAB = {"M0": "M0\nbaseline", "M1": "M1\nsingle-basin", "M2": "M2\nlinear", "M3": "M3\nmemory-fit"}
COL = {"M0": "#7a7a76", "M1": "#2a78d6", "M2": "#1baf7a", "M3": "#eb6834"}; HAT = {"M0": None, "M1": None, "M2": "///", "M3": "\\\\\\"}
fits = {(m, w): json.load(open(O/f"fit_{m}_{w}.json")) for m in M for w in ("full", "A", "B")}
obs = {w: E.memory(E.obs, *E.WINDOWS[w]["mem"]) for w in E.WINDOWS}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#0b0b0b", "ytick.color": "#0b0b0b"})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
x = np.arange(4); wd = 0.36
def held(metric, which):   # held-out skill: fit on A -> eval B ; fit on B -> eval A
    return [fits[(m, "A")]["eval"]["B"]["skill"][metric] if which == 0 else fits[(m, "B")]["eval"]["A"]["skill"][metric] for m in M]
for k, (metric, title) in enumerate((("KGE", "Held-out KGE"), ("logNSE", "Held-out logNSE"))):
    a = ax[k]
    v0 = held(metric, 0); v1 = held(metric, 1)
    for i, m in enumerate(M):
        a.plot([x[i], x[i]], [v1[i], v0[i]], color=COL[m], lw=2, alpha=0.5, zorder=1)
        a.plot([x[i]], [v0[i]], "o", ms=11, color=COL[m], mec="#fcfcfb", mew=1.5, zorder=3)
        a.plot([x[i]], [v1[i]], "o", ms=11, mfc="#fcfcfb", mec=COL[m], mew=2.4, zorder=3)
        a.text(x[i]+0.14, v0[i], f"{v0[i]:.2f}", va="center", fontsize=8.5); a.text(x[i]+0.14, v1[i], f"{v1[i]:.2f}", va="center", fontsize=8.5)
    a.set_xticks(x); a.set_xticklabels([LAB[m] for m in M]); a.set_xlim(-0.5, 3.6); a.set_ylim(0.62, 0.92 if metric == "logNSE" else 0.90)
    a.set_title(title+"\nfilled: fit 1986-2005, test 2006-2024\nopen: fit 2006-2024, test 1986-2005", fontsize=10, loc="left"); a.set_ylabel(metric); a.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
a = ax[2]
for i, m in enumerate(M):
    v = fits[(m, "full")]["eval"]["full"]["memory"]["res1"]; a.bar(x[i], v, 0.55, color=COL[m], hatch=HAT[m], edgecolor="#fcfcfb", linewidth=0.6)
    (a.text(x[i], v/2, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white") if v > 0.2 else a.text(x[i], max(v, 0)+0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8))
    for off, (cal, val) in ((-0.17, ("A", "B")), (0.17, ("B", "A"))):
        hv = fits[(m, cal)]["eval"][val]["memory"]["res1"]; a.plot([x[i]+off], [hv], marker="D", ms=6, color="white", mec="#0b0b0b", mew=1.2, zorder=5)
a.axhline(obs["full"]["res1"], color="#0b0b0b", lw=1.6); a.text(-0.45, obs["full"]["res1"]+0.012, f"observed (1986-2024): {obs['full']['res1']:.2f}", ha="left", fontsize=8.5)
a.axhline(0, color="#52514e", lw=0.8)
a.set_xticks(x); a.set_xticklabels([LAB[m] for m in M]); a.set_ylabel("residual lag-1 autocorrelation")
a.set_title("Memory (Aug-Nov low flow)\nbars: full-window fit; diamonds: held-out half\nobserved halves: -0.01 (1986-2005), 0.56 (2006-2024)", fontsize=10, loc="left")
fig.suptitle("East River: recalibrated baseline vs single-basin (leak to a basin-wide store) -- emulator, verified in SUMMA. Palette not machine-validated (no node available); bars and dots are value-labelled and M2/M3 bars are hatched.", fontsize=9, x=0.01, ha="left", color="#52514e")
fig.savefig(FIG, dpi=150); print("saved", FIG)
