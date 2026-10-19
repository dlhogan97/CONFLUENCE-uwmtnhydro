"""East: put the slow store on the FLAT valley-bottom HRU alone.

HRU 14 is the only flat valley HRU -- combined_e "1_0" (elev class 1, aspect class 0),
slope 1.94 deg against 11-12 deg for the other four class-1 HRUs, lowest elevation
(2650 m), 6.95% of basin area.

The point is to shrink the contributing area. Established so far:
  20.4% (all 5 class-1 HRUs, sf=7)  -> rho1 +0.173, IDENTICAL to baseline; costs nothing
  100%  (uniform sf=7)              -> rho1 +0.286 (p=0.063) but PBIAS_low 104%, KGE 0.737
This run sits below the lower endpoint at 6.95%, so it should carry even less memory
than the 20.4% case -- worth having as the bottom anchor of the area-fraction curve.

aquiferBaseflowExp is left at the calibrated 2.056814 everywhere.
"""
from pathlib import Path
import shutil
import numpy as np, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
E = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East"
TEMPLATE = E/"ens_scale_sf07p0_s0p00_fp"      # sf=7 on HRUs 14-18; we reset then re-apply
OUT = E/"flatvalley_20260812"
SF, HRU = 7.0, 14
CAL_SF = 2.6577657921019333                    # calibrated aquiferScaleFactor


def main():
    tag = f"flatvalley_sf{SF:04.1f}".replace(".", "p")
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
        sf = np.array(t["aquiferScaleFactor"][:])
        flat = sf.reshape(-1)
        flat[:] = CAL_SF                         # clear the 5-HRU valley store
        k = int(np.where(hid == HRU)[0][0])
        flat[k] = SF                             # ...and put it on HRU 14 alone
        t["aquiferScaleFactor"][:] = sf
        ex = np.array(t["aquiferBaseflowExp"][:]).ravel()
        print(f"  sf: {np.unique(flat)}  modified {int((flat != CAL_SF).sum())} of {flat.size} HRUs")
        print(f"  exp: {np.unique(ex)} (left at calibrated)")

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
    (OUT/"jobs.txt").write_text(f"{d}/fileManager.txt\n")
    print(f"built {d}")


if __name__ == "__main__":
    main()
