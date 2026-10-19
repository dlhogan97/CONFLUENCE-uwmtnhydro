"""Anchor-bin snowpatch designs: never melt out, and keep the perennial bins distinct.

Two defects in the sweep design this fixes.

1. ALL FIVE perennial bins shared one drift multiplier, so their only differentiation
   was the initial condition. The 1987-92 drought emptied every one of them, and from
   1986 on they carried identical SWE forever after -- functionally a single 8% bin.
   That is the source of the horizontal banding in the fSCA-vs-SWE plot.
   Fix: give bins 2-5 a spread of multipliers so the FORCING maintains their spread,
   which is robust to any drought and to spin-up length.

2. The store empties completely in 7 of 41 years, where Landsat never reads zero.
   Fix: bin 1 becomes an ANCHOR -- deep initial mass and its own multiplier slightly
   above balance, so it persists through the worst drawdown on record (-9.17 m).
   At 0.5% of HRU 2 it pins the fSCA floor at 0.005 (observed 5th percentile 0.0067,
   observed minimum 0.0011) and costs 0.86 hm3 = 0.097% of one year of basin precip.

Decoupling these lets the GLOBAL multiplier be lowered for accuracy without killing
the store: the sweep's best member (0.90) still over-predicted fSCA by 1.52x, and the
response is near-linear, so ~0.80 should land near 1.0. Persistence is now the
anchor's job, not the multiplier's.
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path("/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca")
OUT = D/"designs_anchor_20260812"; OUT.mkdir(exist_ok=True)
K, S0 = 4.043, 1.331
A_HRU2, A_BASIN = 8.611236, 775.0
ABL = 2.09                 # m/yr, measured deep-pack annual ablation on the 44-yr run
S_THRESH = 2.10            # m available mass needed to last the summer
Q = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.56, 0.80, 1.00])
QPERM = 0.089              # perennial fraction: bins 1-5 (best member of the sweep)

MULT = [0.80, 0.85, 0.90]  # global multiplier on bins 2-10
ANCHOR_MULT = 1.05         # bin 1 only, fixed across members: slightly above balance
ANCHOR_INIT = 20.0         # m; worst drawdown on record is 9.17 m, so this has headroom
PEREN_SPREAD = np.array([1.00, 0.94, 0.88, 0.82])   # bins 2-5, x global multiplier
PEREN_INIT = np.array([6.0, 4.0, 2.5, 1.2])         # m

f_log = lambda S: 1/(1+np.exp(-K*(S-S0)))
f_inv = lambda q: S0 - np.log(1/np.clip(q, 1e-6, 1-1e-6) - 1)/K


def design(mult):
    S = (pd.read_csv(D/"model_hru2_peak_swe.csv", index_col="year")
         .peak_swe_mm/1000).loc[1981:2025]
    Sbar = float(S.mean())
    a_bal = ABL/Sbar                       # multiplier that balances a deep pack
    w = np.diff(np.r_[0.0, Q])
    Sm = f_inv(Q); Sm[-1] = np.inf
    per = Q <= QPERM + 1e-9                # bins 1-5

    a = np.where(per, a_bal*mult, (S_THRESH/Sm)*mult)
    a[-1] = 1.0                            # bin 10 is structurally bare
    init = np.zeros(len(Q))

    # bin 1: the anchor -- own multiplier, deep start, exempt from the global sweep
    a[0] = a_bal*ANCHOR_MULT
    init[0] = ANCHOR_INIT
    # bins 2-5: graded multipliers so the spread is maintained by forcing, not by init
    idx = np.where(per)[0][1:]
    a[idx] = a_bal*mult*PEREN_SPREAD[:len(idx)]
    init[idx] = PEREN_INIT[:len(idx)]

    b = pd.DataFrame(dict(
        bin=np.arange(1, len(Q)+1), q_cum=Q, area_frac=w, area_km2=w*A_HRU2,
        regime=np.where(np.arange(len(Q)) == 0, "anchor",
                        np.where(per, "perennial",
                                 np.where(np.isinf(Sm), "bare", "seasonal"))),
        drift_a=a, S_margin_m=Sm, init_m=init,
        hruId=[2]+list(range(29, 29+len(Q)-1))))
    return b, a_bal, Sbar


def main():
    for m in MULT:
        b, a_bal, Sbar = design(m)
        tag = f"anchor_m{int(round(m*100)):03d}"
        b.to_csv(OUT/f"{tag}.csv", index=False)
        awd = float((b.area_frac*b.drift_a).sum())
        net_anchor = a_bal*ANCHOR_MULT*Sbar - ABL
        print(f"=== {tag}  (global multiplier {m}) ===")
        print(b.round(3).to_string(index=False))
        print(f"  a_bal {a_bal:.3f} (Sbar {Sbar:.3f} m)   anchor drift {b.drift_a[0]:.3f}"
              f"  -> net {net_anchor:+.3f} m/yr, {ANCHOR_INIT + net_anchor*44:.1f} m after 44 yr")
        print(f"  area-weighted multiplier {awd:.3f}x  -> prescribed source "
              f"{(awd-1)*A_HRU2/A_BASIN*100:.2f}% of basin precipitation")
        print(f"  distinct perennial drifts: {np.round(b.drift_a[:5].values,3)}\n")
    (OUT/"members.txt").write_text("".join(
        f"{OUT}/anchor_m{int(round(m*100)):03d}.csv anchor_m{int(round(m*100)):03d}\n"
        for m in MULT))
    print("->", OUT)


if __name__ == "__main__":
    main()
