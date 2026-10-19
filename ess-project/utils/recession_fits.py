"""
Nonlinear recession curves: EOS snow-covered fraction vs modelled peak SWE.

The linear fit is unphysical at both ends (negative below ~0.42 m, unbounded
above 1). Candidates below are all bounded in [0,1] and most carry a threshold.

The lognormal-survival model is not a generic curve -- it is DERIVED from the
subgrid depth distribution ASO actually measured (equal-area bins, lognormal,
CV~1.01). If end-of-season area is just the part of that distribution deeper
than the season's ablation E, then

    f(S) = P(d > E),   d ~ LN(mean = S, CV = c)

which with c FIXED at the ASO value has a single free parameter, E.
That makes it the model the 16-bin build should reproduce on its own.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt

D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
FIGDIR = Path("/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/"
              "ess-project/figures/draft")
ASO_CV = 1.01          # measured within-HRU-2 depth CV, 22 peak-season flights


# ----------------------------------------------------------------- models
def m_linear(S, a, b):
    return a + b * S


def m_power(S, a, b):
    return a * np.power(np.clip(S, 1e-9, None), b)


def m_thresh_power(S, S0, a, b):
    z = np.clip(S - S0, 0, None)
    return a * np.power(z, b)


def _lognorm_survival(S, E, cv):
    S = np.clip(S, 1e-9, None)
    sig = np.sqrt(np.log(1 + cv ** 2))
    mu = np.log(S) - 0.5 * sig ** 2
    return 1.0 - norm.cdf((np.log(max(E, 1e-9)) - mu) / sig)


def m_lognorm_fixedcv(S, E):
    return _lognorm_survival(S, E, ASO_CV)


def m_lognorm_freecv(S, E, cv):
    return _lognorm_survival(S, E, cv)


def m_logistic(S, L, k, S0):
    return L / (1.0 + np.exp(-k * (S - S0)))


def m_logistic1(S, k, S0):
    """Logistic with the ceiling pinned at 1 -- fSCA cannot exceed 1."""
    return 1.0 / (1.0 + np.exp(-k * (S - S0)))


def m_hill(S, S50, n):
    S = np.clip(S, 1e-9, None)
    return S ** n / (S50 ** n + S ** n)


MODELS = [
    ("linear",            m_linear,          [0.0, 0.5],        2),
    ("power  a·S^b",      m_power,           [0.3, 2.0],        2),
    ("threshold power",   m_thresh_power,    [0.35, 0.8, 1.5],  3),
    ("lognormal (CV fixed 1.01)", m_lognorm_fixedcv, [2.0],     1),
    ("lognormal (CV free)",       m_lognorm_freecv,  [2.0, 1.0], 2),
    ("logistic",          m_logistic,        [0.9, 4.0, 1.3],   3),
    ("logistic (L=1, bounded)", m_logistic1,  [4.0, 1.33],       2),
    ("Hill (bounded)",    m_hill,            [1.3, 4.0],        2),
]


def fit_all(x, y):
    out = []
    n = len(x)
    for name, fn, p0, k in MODELS:
        try:
            p, _ = curve_fit(fn, x, y, p0=p0, maxfev=200000)
            yh = fn(x, *p)
            res = y - yh
            sse = float((res ** 2).sum())
            rmse = np.sqrt(sse / n)
            r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
            aic = n * np.log(sse / n) + 2 * k
            aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
            out.append(dict(model=name, k=k, rmse=rmse, r2=r2, aic=aic,
                            aicc=aicc, params=p, fn=fn))
        except Exception as e:                     # noqa: BLE001
            out.append(dict(model=name, k=k, rmse=np.nan, r2=np.nan,
                            aic=np.nan, aicc=np.nan, params=None, fn=fn,
                            err=str(e)[:40]))
    best = np.nanmin([o["aicc"] for o in out])
    for o in out:
        o["dAICc"] = o["aicc"] - best
    return out


def main():
    eos = pd.read_csv(D / "eos_final.csv", index_col="year")
    peak = pd.read_csv(D / "model_hru2_peak_swe.csv", index_col="year").peak_swe_mm
    j = eos.join(peak.rename("peak_mm"), how="inner").dropna(subset=["peak_mm"])
    x = (j.peak_mm / 1000).values
    y = j.f_shadow.values
    print(f"n = {len(x)}   peak SWE {x.min():.2f}-{x.max():.2f} m   "
          f"fSCA {y.min():.3f}-{y.max():.3f}\n")

    res = fit_all(x, y)
    print(f"{'model':28s} {'k':>2s} {'RMSE':>7s} {'R2':>7s} {'AICc':>8s} {'dAICc':>7s}")
    for o in sorted(res, key=lambda d: (np.isnan(d["aicc"]), d["aicc"])):
        if o["params"] is None:
            print(f"{o['model']:28s} {o['k']:2d}   FAILED {o.get('err','')}")
            continue
        print(f"{o['model']:28s} {o['k']:2d} {o['rmse']:7.4f} {o['r2']:7.3f} "
              f"{o['aicc']:8.1f} {o['dAICc']:7.2f}   "
              f"params {np.round(o['params'], 3)}")

    ln = [o for o in res if o["model"].startswith("lognormal (CV fixed")][0]
    lnf = [o for o in res if o["model"].startswith("lognormal (CV free")][0]
    print(f"\nlognormal CV-fixed: effective seasonal ablation E = "
          f"{ln['params'][0]:.2f} m of SWE")
    print(f"lognormal CV-free : E = {lnf['params'][0]:.2f} m, CV = "
          f"{lnf['params'][1]:.2f}  (ASO measured {ASO_CV:.2f})")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    xf = np.linspace(0.05, 2.0, 400)
    colors = {"linear": "#999999", "power  a·S^b": "#E69F00",
              "threshold power": "#009E73",
              "lognormal (CV fixed 1.01)": "#7f7f7f",
              "lognormal (CV free)": "#bcbcbc", "logistic": "#CC79A7",
              "logistic (L=1, bounded)": "#0072B2", "Hill (bounded)": "#009E73"}

    for ax, (lo, hi, ttl) in zip(axes[:2], [(0.05, 2.0, "A · fitted range"),
                                            (0.0, 3.0, "B · extrapolated to 3 m")]):
        xf = np.linspace(lo, hi, 400)
        ax.axhspan(0, 1, color="#f4f4f4", zorder=0)
        ax.scatter(x, y, s=62, color="#0072B2", edgecolor="white", lw=.9,
                   zorder=5, label="observed")
        for o in sorted(res, key=lambda d: d["dAICc"]):
            if o["params"] is None:
                continue
            lw = 2.4 if o["dAICc"] == 0 else 1.4
            ls = "-" if o["dAICc"] == 0 else "--"
            ax.plot(xf, o["fn"](xf, *o["params"]), lw=lw, ls=ls,
                    color=colors[o["model"]], zorder=4 if o["dAICc"] == 0 else 3,
                    label=f"{o['model']}  (ΔAICc {o['dAICc']:.1f})")
        ax.axhline(0, color="#c0392b", lw=1)
        ax.axhline(1, color="#c0392b", lw=1)
        ax.set_xlabel("modelled HRU-2 peak SWE (m)")
        ax.set_ylabel("effective fSCA of HRU 2")
        ax.set_title(ttl, loc="left", fontsize=11.5)
        ax.set_ylim(-0.35, 1.35); ax.set_xlim(lo, hi)
        ax.grid(True, color="#d8d8d8", lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.6, loc="upper left")
    axes[1].annotate("linear goes NEGATIVE here", (0.06, -0.22), fontsize=8.5,
                     color="#c0392b")
    axes[1].annotate("power law blows past 1 →", (1.55, 1.15), fontsize=8.5,
                     color="#c0392b")

    # --- panel C: the physical model at the scale ASO measured it
    axc = axes[2]
    y8 = j.fsca_b8_sh.values
    p8, _ = curve_fit(m_lognorm_fixedcv, x, y8, p0=[2.0], maxfev=200000)
    pf, _ = curve_fit(m_lognorm_freecv, x, y8, p0=[2.0, 1.0], maxfev=200000)
    xf = np.linspace(0.05, 2.0, 400)
    axc.scatter(x, y8, s=62, color="#009E73", edgecolor="white", lw=.9, zorder=5)
    axc.plot(xf, m_lognorm_fixedcv(xf, *p8), color="#0072B2", lw=2.4,
             label=f"lognormal, CV fixed at ASO 1.01\n  E = {p8[0]:.2f} m")
    axc.plot(xf, m_lognorm_freecv(xf, *pf), color="#CC79A7", lw=1.6, ls="--",
             label=f"lognormal, CV free → {pf[1]:.2f}\n  E = {pf[0]:.2f} m")
    b8, a8 = np.polyfit(x, y8, 1)
    axc.plot(xf, a8 + b8 * xf, color="#999999", lw=1.2, ls=":", label="linear")
    axc.set_xlabel("modelled HRU-2 peak SWE (m)")
    axc.set_ylabel("band 8 snow-covered fraction")
    axc.set_title("C · physical model at the ASO scale (band 8)", loc="left",
                  fontsize=11.5)
    axc.legend(frameon=False, fontsize=8, loc="upper left")
    axc.set_ylim(-0.05, 0.55); axc.set_xlim(0.05, 2.0)
    axc.grid(True, color="#d8d8d8", lw=.6); axc.set_axisbelow(True)
    for s_ in ("top", "right"):
        axc.spines[s_].set_visible(False)
    print(f"\nband-8 lognormal: CV fixed 1.01 -> E={p8[0]:.2f} m;  "
          f"CV free -> E={pf[0]:.2f} m, CV={pf[1]:.2f}")
    fig.suptitle("Nonlinear recession curves — fSCA is bounded [0,1]; the "
                 "linear and power-law fits are not", x=0.005, ha="left",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGDIR / "recession_nonlinear.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    print(f"\nwrote {FIGDIR/'recession_nonlinear.png'}")

    # export fitted params
    rows = [dict(model=o["model"], k=o["k"], rmse=o["rmse"], r2=o["r2"],
                 aicc=o["aicc"], dAICc=o["dAICc"],
                 params=";".join(f"{v:.5f}" for v in o["params"]))
            for o in res if o["params"] is not None]
    pd.DataFrame(rows).sort_values("dAICc").to_csv(D / "recession_fits.csv",
                                                   index=False)
    print(f"wrote {D/'recession_fits.csv'}")


if __name__ == "__main__":
    main()
