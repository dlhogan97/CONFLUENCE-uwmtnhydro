"""East: the sf x exponent grid on the 5-HRU class-1 valley store (20.4% of basin).

This fills the one extent that never got a grid. Coverage before this run:
    6.95% (HRU 14)      3x3 complete  -> rho1 +0.148..+0.198, none significant, SATURATES
   20.4%  (HRUs 14-18)  a CROSS only: sf {1,4,7,10} at calibrated exp, and
                        exp {1.5..10} at sf=7. Every off-axis cell empty; sf=14 untested.
  100%   (all 18)       3x3 complete -> rho1 +0.494..+0.886 but KGE 0.646..0.525

20.4% is the extent that decides whether the memory transition is gradual or abrupt, and
it is the least explored of the three. Running the full 3x3 (rather than only the 7 truly
missing cells) keeps it code-identical to the other two grids; sf=7 x exp={4,10} also
duplicate sweep_exp_20260812/exp04p0 and exp10p0, so they double as a reproducibility check.

Valley store = elevation class 1 = hruId 14,15,16,17,18 (2650-2710 m, 20.37% of area).
basinParamInfo bounds: aquiferBaseflowExp in [1.0, 10.0].
"""
from pathlib import Path
import shutil
import numpy as np, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
E = SC/"domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East"
TEMPLATE = E/"ens_scale_sf07p0_s0p00_fp"
OUT = E/"sweep_valley5_20260812"
SF  = [7.0, 10.0, 14.0]
EXP = [4.0, 7.0, 10.0]
VALLEY = [14, 15, 16, 17, 18]
CAL_SF, CAL_EXP = 2.6577657921019333, 2.056814


def build_one(sf, exp):
    tag = f"v5_sf{sf:04.1f}_e{exp:04.1f}".replace(".", "p")
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
        k = np.array([int(np.where(hid == h)[0][0]) for h in VALLEY])
        a = np.array(t["aquiferScaleFactor"][:]); fa = a.reshape(-1)
        fa[:] = CAL_SF; fa[k] = sf
        t["aquiferScaleFactor"][:] = a
        b = np.array(t["aquiferBaseflowExp"][:]); fb = b.reshape(-1)
        fb[:] = CAL_EXP; fb[k] = exp
        t["aquiferBaseflowExp"][:] = b
        assert int((~np.isclose(fa, CAL_SF)).sum()) == len(VALLEY)
        assert int((~np.isclose(fb, CAL_EXP)).sum()) == len(VALLEY)

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
    print(f"  aquiferScaleFactor {SF} x aquiferBaseflowExp {EXP} on hruId {VALLEY} (20.37%)")


if __name__ == "__main__":
    main()
