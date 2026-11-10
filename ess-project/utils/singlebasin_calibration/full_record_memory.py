"""
Full-record (WY1982-2025) check of the East memory result on the NEW no-storage control (2026-10-22).

The control (run_calibration_control best) is rerun from 1981-10-01; its hourly recharge drives the aquifer emulator with
EXISTING parameter sets, no refitting:
  new M0 / M1      stage-B fits on the WY2009-2025 control state (objective included low-flow terms -- kept only for comparison)
  overnight M0/M1  the 2026-10-14 fits on the OLD baseline's recharge, WY1987-2024 (objective (1-KGE)+(1-logNSE)+|Aug-Nov bias|+|bias|)
plus the SUMMA control itself (its own fast aquifer). Memory = emu_core.memory (Spearman; ln Aug-Nov 7-d min on ln Apr-Jul peak),
reported for the full record, both halves and the recent period, with the main hydrograph metrics for WY1987-2024.
usage: STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30 python full_record_memory.py
"""
import json, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, netCDF4 as nc
import matplotlib.pyplot as plt
import stageB_emulator as B, emu_core as E

assert B.T0 == "1981-10-01" and B.END == "2025-09-30", "run with STAGEB_T0=1981-10-01 STAGEB_END=2025-09-30"
D = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/"
RUN = B.SRC["ctlfull"][0]
FIG = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/figures/draft/east_full_record_memory.png"
WINDOWS = {"WY1987-2024": (1987, 2024), "WY1987-2005": (1987, 2005), "WY2006-2024": (2006, 2024), "WY2009-2024": (2009, 2024)}
wy = lambda i: i.year + (i.month >= 10)


def summa_control():
    d = nc.Dataset(f"{RUN}/out/best_timestep.nc"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    A = float(np.array(nc.Dataset(f"{RUN}/attributes.nc")["HRUarea"][:]).sum())
    return (pd.Series(np.array(d["averageRoutedRunoff"][:, 0]), index=t)*A).resample("D").mean()


def main_metrics(s, o):
    j = pd.concat([s.rename("s"), o.rename("o")], axis=1).dropna(); s, o = j.s, j.o
    pk = lambda q: q.rolling(7, center=True).mean().groupby(wy(q.index)).idxmax()
    com = lambda q: q.groupby(wy(q.index)).apply(lambda g: (np.arange(len(g))*g.values).sum()/g.values.sum())
    sm, om = s.groupby(s.index.month).mean(), o.groupby(o.index.month).mean()
    return {"KGE": E.kge(o, s), "vol bias %": 100*(s.sum()/o.sum() - 1), "monthly |bias| %": 100*(sm - om).abs().sum()/om.sum(),
            "peak timing |err| d": np.mean([abs((a - b).days) for a, b in zip(pk(s), pk(o))]), "COM bias d": float((com(s) - com(o)).median())}


def main():
    st = B.build_state("ctlfull")
    series = {"observed": E.obs, "SUMMA control (fast aquifer)": summa_control()}
    sets = {"new M0": D+"stageB/fit_ctl_M0_to2025.json", "new M1": D+"stageB/fit_ctl_M1_to2025.json",
            "overnight M0": D+"overnight/fit_M0_full.json", "overnight M1": D+"overnight/fit_M1_full.json"}
    for lab, f in sets.items():
        p = json.load(open(f))["params"]; Qs, Qd, Sd = B.emulate(st, p); series[lab] = B.daily_Q(st, Qs, Qd, p["scale"])
        if lab == "overnight M1": deep = pd.Series(Sd, index=st["tt"]).resample("D").mean()
    rows = []
    for lab, q in series.items():
        r = dict(series=lab)
        for w, (y0, y1) in WINDOWS.items():
            m = E.memory(q, y0, y1); r[f"lag1 {w}"] = m["res1"]
        if lab != "observed": r.update(main_metrics(q["1986-10-01":"2024-09-30"], E.obs["1986-10-01":"2024-09-30"]))
        rows.append(r)
    T = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("Spearman residual lag-1 (ln Aug-Nov 7-d min on ln Apr-Jul peak) by window; hydrograph metrics WY1987-2024"); print(T.round(2).to_string(index=False))
    T.to_csv(D+"stageB/full_record_memory.csv", index=False)

    # figure: hydrograph (log) + residual series for observed / SUMMA control / overnight M1
    C = {"observed": "#222222", "SUMMA control (fast aquifer)": "#2a78c2", "overnight M1": "#d9822b"}
    fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    for k in C: ax.plot(series[k]["1986-10-01":"2025-09-30"], color=C[k], lw=0.9 if k != "observed" else 1.2, label=k)
    ax.set_yscale("log"); ax.set_ylabel("m³/s (log)"); ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("East River, WY1987–2025: new no-storage control and the overnight deep-store parameters on its recharge", loc="left", fontsize=12)
    ax2 = fig.add_subplot(gs[1])
    for k in C:
        lf, pk = E.series(series[k], 1987, 2024); r = E.resid(lf, pk)
        ax2.plot(r.index, r.values, color=C[k], lw=1.8, marker="o", ms=4, label=f"{k}: lag-1 {E.lag_r(r, 1):+.2f}")
    ax2.axhline(0, color="#999", lw=1); ax2.axvline(2005.5, color="#bbb", ls="--", lw=1)
    ax2.set_ylabel("residual, ln(Aug–Nov 7-d min) | peak"); ax2.set_xlabel("year"); ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.set_title("Low-flow residuals after removing the same-year peak (Spearman lag-1 over WY1987–2024)", loc="left", fontsize=11)
    for a in (ax, ax2):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.grid(True, color="#e5e5e5", lw=0.6); a.set_axisbelow(True)
    fig.savefig(FIG, dpi=125, bbox_inches="tight", facecolor="white"); print("wrote", FIG)


if __name__ == "__main__":
    main()
