"""East: hold the store STRUCTURE fixed and sweep its EXTENT up the elevation profile.

Structure is the saturated optimum from sweep_valley5_20260812: aquiferScaleFactor 10,
aquiferBaseflowExp 10 on the store HRUs, calibrated values elsewhere. At 20.4% that is
the best East low-flow configuration measured -- PBIAS_low 23.40% vs baseline 29.14%,
KGE 0.8773 vs 0.8674, NSE 0.7584 vs 0.7567 -- and sf=14 is identical to 4 decimals, so
10 is the conservative pick.

What the three completed grids established: memory is FLAT at rho1 ~ 0.196 across both
6.95% and 20.4% (tripling the area changes nothing, and both saturate in sf and exp),
then jumps to >= 0.494 at 100% while KGE collapses 0.867 -> 0.575. The switch therefore
sits somewhere in 20-100%, and nothing between has ever been run.

HRUs are added in ASCENDING ELEVATION order -- the store grows upslope from the valley
bottom, which is the physically defensible way to expand it. Targets ~30/40/50/60%;
actual fractions fall where HRU boundaries allow and are recorded per member.

    cum %   adds hruId   elev (m)
     6.95   14           2650   <- done (sweep_hru14)
    20.37   18,17,16,15  2698-2710   <- done (sweep_valley5)
    22.11   9            2989
    31.78   10           3011   <- ~30
    39.75   11           3012   <- ~40
    50.28   12           3014   <- ~50
    57.53   13           3016   <- ~60 (next step is 64.6%, overshoots)
"""
from pathlib import Path
import shutil
import numpy as np, pandas as pd, geopandas as gpd, netCDF4 as nc4

SC = Path("/scratch/dlhogan/ess-project-data")
D = SC/"domain_East_River_distributed_elevAspect"
E = D/"simulations/store_memory_sandbox/East"
SHP = D/"shapefiles/catchment/East_River_distributed_elevAspect_HRUs_elevation_aspect.shp"
TEMPLATE = E/"ens_scale_sf07p0_s0p00_fp"
OUT = E/"sweep_extent_20260812"
SF, EXP = 10.0, 10.0
CAL_SF, CAL_EXP = 2.6577657921019333, 2.056814
TARGETS = [30.0, 40.0, 50.0, 60.0]


def hru_order():
    g = gpd.read_file(SHP).sort_values("elev_mean")
    g["cum"] = 100*g.HRU_area.cumsum()/g.HRU_area.sum()
    return g[["HRU_ID", "elev_mean", "cum"]].reset_index(drop=True)


def build_one(hrus, pct):
    tag = f"ext{pct:04.1f}".replace(".", "p")
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
        k = np.array([int(np.where(hid == h)[0][0]) for h in hrus])
        a = np.array(t["aquiferScaleFactor"][:]); fa = a.reshape(-1)
        fa[:] = CAL_SF; fa[k] = SF
        t["aquiferScaleFactor"][:] = a
        b = np.array(t["aquiferBaseflowExp"][:]); fb = b.reshape(-1)
        fb[:] = CAL_EXP; fb[k] = EXP
        t["aquiferBaseflowExp"][:] = b
        assert int((~np.isclose(fa, CAL_SF)).sum()) == len(hrus)
        assert int((~np.isclose(fb, CAL_EXP)).sum()) == len(hrus)

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
    return d, tag


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = hru_order()
    ds, rows = [], []
    for tgt in TARGETS:
        n = int(np.argmin(np.abs(g["cum"].values - tgt))) + 1   # HRU whose cum is nearest
        hrus = g.HRU_ID.iloc[:n].tolist()
        pct = float(g["cum"].iloc[n-1])
        d, tag = build_one(hrus, pct)
        ds.append(d)
        rows.append(dict(target=tgt, tag=tag, area_pct=round(pct, 2), n_hru=len(hrus),
                         max_elev=round(float(g.elev_mean.iloc[n-1])), hrus=hrus))
    (OUT/"jobs.txt").write_text("".join(f"{d}/fileManager.txt\n" for d in ds))
    t = pd.DataFrame(rows)
    t.to_csv(OUT/"extents.csv", index=False)
    print(f"built {len(ds)} members -> {OUT}   (sf={SF}, exp={EXP} on the store HRUs)")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
