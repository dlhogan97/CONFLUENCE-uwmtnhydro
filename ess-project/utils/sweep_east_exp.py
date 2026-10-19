"""East aquiferBaseflowExp sweep, valley-bottom store only.

Design (2026-08-12): keep the slow store where it is physically defensible -- the five
lowest-elevation HRUs (2650-2710 m, 20.4% of basin area, hruId 14-18) -- and hold
aquiferScaleFactor at 7 there, exactly as ens_scale_sf07p0. Vary ONLY the baseflow
exponent on those same HRUs; the other 13 keep the calibrated 2.056814.

Why the exponent is the interesting axis. SUMMA's bucket is
    Q = aquiferBaseflowRate * (S / aquiferScaleFactor) ** aquiferBaseflowExp
so the recession timescale is tau = S / (exp * Q) -- FLOW DEPENDENT. Raising exp gives
short tau at high flow (the spring peak is not damped, which is what inflated autumn
volume in the uniform-sf sweep) and long tau at low flow (where interannual memory
lives). Every previous East run held exp fixed at the calibrated value, so this is the
one knob that can plausibly break the memory-vs-bias trade rather than slide along it.

basinParamInfo bounds: aquiferBaseflowExp in [1.0, 10.0]; the sweep spans that range.
No aquifer-storage initialisation: the scan showed it is inert (identical results to 4
decimals across every family) and the 2.5 m version produced a WY1981 spin-up spike.
"""
from pathlib import Path
import shutil
import numpy as np, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
E = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East"
TEMPLATE = E/"ens_scale_sf07p0_s0p00_fp"          # sf=7 already set on the 5 valley HRUs
OUT = E/"sweep_exp_20260812"
EXPS = [1.5, 2.056814, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
SF_VALLEY = 7.0


def valley_mask(path):
    """The HRUs the valley-bottom store is applied to = those whose sf differs."""
    with nc4.Dataset(Path(path)/"trialParams.nc") as t:
        sf = np.array(t["aquiferScaleFactor"][:]).ravel()
    return sf != np.median(sf)


def build_one(exp, mask):
    tag = f"exp{exp:04.1f}".replace(".", "p")
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
        sf = np.array(t["aquiferScaleFactor"][:])
        sf.reshape(-1)[mask] = SF_VALLEY                 # re-assert, template already has it
        t["aquiferScaleFactor"][:] = sf
        ex = np.array(t["aquiferBaseflowExp"][:])
        ex.reshape(-1)[mask] = exp                       # valley HRUs only
        t["aquiferBaseflowExp"][:] = ex

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
    mask = valley_mask(TEMPLATE)
    with nc4.Dataset(TEMPLATE/"attributes.nc") as a:
        hid = np.array(a["hruId"][:]).ravel()
        area = np.array(a["HRUarea"][:]).ravel()
    print(f"valley-bottom store on {mask.sum()} HRUs: {sorted(hid[mask])}  "
          f"({100*area[mask].sum()/area.sum():.1f}% of basin area)")
    ds = [build_one(e, mask) for e in EXPS]
    (OUT/"jobs.txt").write_text("".join(f"{d}/fileManager.txt\n" for d in ds))
    print(f"built {len(ds)} members -> {OUT}")
    print(f"  aquiferBaseflowExp: {EXPS}   (baseline 2.056814)")
    print(f"  aquiferScaleFactor: {SF_VALLEY} on the valley HRUs, 2.6578 elsewhere")


if __name__ == "__main__":
    main()
