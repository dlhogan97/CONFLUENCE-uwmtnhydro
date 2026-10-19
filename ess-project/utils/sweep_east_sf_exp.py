"""East: does raising the exponent BUY BACK the autumn bias that a big scale factor costs?

Established so far, both one-dimensional:
  * uniform aquiferScaleFactor is the only thing that moves rho_1, and it costs autumn
    volume monotonically (sf 7 -> +104% PBIAS_low, sf 10 -> +115%, baseline +29%);
  * raising aquiferBaseflowExp on the valley-bottom store improved every low-flow metric
    monotonically (29.1% -> 23.6% at exp 10) but bought no memory, because 20% of the
    basin cannot carry an interannual signal.

This crosses them, UNIFORMLY across all 18 HRUs so the memory mechanism is live.

Which way it goes is genuinely unclear. Q = rate*(S/sf)**exp gives equilibrium storage
S = sf*(R/rate)**(1/exp), which INCREASES with exp since R/rate < 1, and tau = S/(exp*Q).
For sf=10 that is tau ~ 200 d at exp 2.06 and ~640 d at exp 6 -- i.e. MORE damping and
MORE autumn excess, opposite to what the valley-bottom sweep measured. The valley run
moved only a fifth of the basin, so the two effects need not share a sign at full scale.

exp 2.056814 (the calibrated value) is already covered for sf 7/10/14 by
sweep_sf_20260812, so this grid skips it.
"""
from pathlib import Path
import shutil
import numpy as np, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
E = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East"
TEMPLATE = E/"ens_bw_sf7p0_s0p0_fp"      # uniform sf across all HRUs, zero initial storage
OUT = E/"sweep_sf_exp_20260812"
SF = [7.0, 10.0, 14.0]
EXP = [4.0, 7.0, 10.0]                   # basinParamInfo bound is 10.0


def build_one(sf, exp):
    tag = f"sf{sf:04.1f}_e{exp:04.1f}".replace(".", "p")
    d = OUT/tag
    d.mkdir(parents=True, exist_ok=True); (d/"output").mkdir(exist_ok=True)
    for f in TEMPLATE.iterdir():
        if f.name in ("output", "summa.log"):
            continue
        t = d/f.name
        if t.exists() or t.is_symlink():
            t.unlink()
        if f.is_symlink():
            t.symlink_to(f.resolve())
        elif f.is_file():
            shutil.copy2(f, t)
    with nc4.Dataset(d/"trialParams.nc", "a") as t:
        for nm, val in (("aquiferScaleFactor", sf), ("aquiferBaseflowExp", exp)):
            a = np.array(t[nm][:]); a[...] = val; t[nm][:] = a     # uniform, all HRUs
    out = []
    for ln in (TEMPLATE/"fileManager.txt").read_text().splitlines():
        k = ln.split()[0] if ln.split() else ""
        if k == "settingsPath":
            ln = f"settingsPath         '{d}/'"
        elif k == "outputPath":
            ln = f"outputPath           '{d}/output/'"
        elif k == "outFilePrefix":
            ln = f"outFilePrefix        '{tag}'"
        out.append(ln)
    (d/"fileManager.txt").write_text("\n".join(out) + "\n")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = [build_one(sf, e) for sf in SF for e in EXP]
    (OUT/"jobs.txt").write_text("".join(f"{d}/fileManager.txt\n" for d in ds))
    print(f"built {len(ds)} members -> {OUT}")
    print(f"  aquiferScaleFactor {SF} x aquiferBaseflowExp {EXP}, uniform over all 18 HRUs")


if __name__ == "__main__":
    main()
