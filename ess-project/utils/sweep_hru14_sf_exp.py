"""East: the sf x exponent grid, applied to the FLAT VALLEY HRU ALONE (HRU 14, 6.95%).

The uniform version of this grid (sweep_sf_exp_20260812) produced real memory -- rho1 up
to +0.886 -- but only because it moved all 18 HRUs; autumn bias went to +19..+107% and KGE
fell from 0.867 to as low as 0.525. Restricting the store to the valley bottom removes
that cost entirely but, at the calibrated exponent, also removes the memory:

    store area   rho1     PBIAS_low   KGE
     6.95% sf=7  +0.177     29.5%     0.868   (flatvalley_20260812)
    20.4%  sf=7  +0.173     30.7%     0.870
    100%   sf=7  +0.286    104.1%     0.737

So memory is flat from 7% to 20% of area and only switches on nearer full coverage. The
open question this grid asks: can a MUCH slower store on that same 6.95% carry a signal
anyway? In autumn, total basin flow is small, so a deep slow store at the valley bottom
could supply a large FRACTION of late-season flow even at a small area fraction -- which
is the mechanism the area-scaling argument would miss.

HRU 14 is the only flat valley HRU: combined_e "1_0", slope 1.94 deg vs 11-12 deg for the
other four class-1 HRUs, lowest elevation (2650 m).

sf=7 x exp=2.056814 (calibrated) already exists as flatvalley_20260812/flatvalley_sf07p0.
basinParamInfo bounds: aquiferBaseflowExp in [1.0, 10.0].
"""
from pathlib import Path
import shutil
import numpy as np, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
E = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East"
TEMPLATE = E/"ens_scale_sf07p0_s0p00_fp"
OUT = E/"sweep_hru14_20260812"
SF  = [7.0, 10.0, 14.0]
EXP = [4.0, 7.0, 10.0]
HRU = 14
CAL_SF, CAL_EXP = 2.6577657921019333, 2.056814


def build_one(sf, exp):
    tag = f"h14_sf{sf:04.1f}_e{exp:04.1f}".replace(".", "p")
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
        hid = np.array(t["hruId"][:]).ravel()
        k = int(np.where(hid == HRU)[0][0])
        a = np.array(t["aquiferScaleFactor"][:]); fa = a.reshape(-1)
        fa[:] = CAL_SF; fa[k] = sf                 # clear the 5-HRU store, set HRU 14 only
        t["aquiferScaleFactor"][:] = a
        b = np.array(t["aquiferBaseflowExp"][:]); fb = b.reshape(-1)
        fb[:] = CAL_EXP; fb[k] = exp
        t["aquiferBaseflowExp"][:] = b
        assert int((fa != CAL_SF).sum()) == 1 and int((fb != CAL_EXP).sum()) == 1

    out = []
    for ln in (TEMPLATE/"fileManager.txt").read_text().splitlines():
        k0 = ln.split()[0] if ln.split() else ""
        if k0 == "settingsPath":
            ln = f"settingsPath         '{d}/'"
        elif k0 == "outputPath":
            ln = f"outputPath           '{d}/output/'"
        elif k0 == "outFilePrefix":
            ln = f"outFilePrefix        '{tag}'"
        out.append(ln)
    (d/"fileManager.txt").write_text("\n".join(out) + "\n")
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = [build_one(sf, e) for sf in SF for e in EXP]
    (OUT/"jobs.txt").write_text("".join(f"{d}/fileManager.txt\n" for d in ds))
    print(f"built {len(ds)} members -> {OUT}")
    print(f"  aquiferScaleFactor {SF} x aquiferBaseflowExp {EXP}, HRU {HRU} ONLY (6.95% of basin)")


if __name__ == "__main__":
    main()
