#!/usr/bin/env python3
"""
apply_terrain_sw.py — Apply a slope/aspect correction to SWRadAtm in the forcing.

Why this is needed
------------------
SUMMA does NOT scale shortwave by terrain.  `sunGeomtry.f90` computes a proper
slope/aspect radiation index (`hri`), but `derivforce.f90` receives it and never
uses it — only `cosZenith` survives, and it feeds only the direct/diffuse split:

    spectralIncomingDirect  = SWRadAtm * fracDirect * ...
    spectralIncomingDiffuse = SWRadAtm * (1 - fracDirect) * ...

Those four components sum back to exactly SWRadAtm.  So the *magnitude* of
incident shortwave is whatever the forcing says, identically for a north- and a
south-facing HRU.  The terrain correction must therefore be applied here.

Method
------
Per HRU (slope beta from tan_slope, aspect alpha, basin-centroid solar position):

    cos_i  = cos(b)cos(z) + sin(b)sin(z)cos(gamma - alpha)   incidence on the slope
    SVF    = (1 + cos(b)) / 2                                sky-view factor
    SW'    = DNI * max(cos_i, 0) + DHI * SVF + albedo * GHI * (1 - SVF)

GHI is the forcing SWRadAtm; the direct/diffuse split uses the Erbs (1982)
decomposition from the clearness index.  The direct beam is what responds to
aspect; diffuse is only reduced by the sky-view factor.

Usage
-----
    python apply_terrain_sw.py --domain-dir /scratch/.../domain_X \\
        --src-forcing basin_averaged_data_gradient_corrected \\
        --dst-forcing SUMMA_input
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

GROUND_ALBEDO = 0.25          # bulk; snow-covered ground is handled by SUMMA itself
MAX_DIRECT_GAIN = 5.0         # cap the beam correction near sunrise/sunset


def hru_geometry(settings_dir: Path):
    a = xr.open_dataset(settings_dir / "attributes.nc")
    slope = np.arctan(np.abs(a["tan_slope"].values))        # radians
    aspect = np.deg2rad(a["aspect"].values)                 # radians, E of N
    lat = float(np.mean(a["latitude"].values))
    lon = float(np.mean(a["longitude"].values))
    return slope, aspect, lat, lon, a.sizes["hru"]


def correct_file(src: Path, dst: Path, slope, aspect, lat, lon) -> dict:
    with xr.open_dataset(src) as ds:
        ds = ds.load()
    times = pd.DatetimeIndex(ds["time"].values)
    ghi = ds["SWRadAtm"].values                              # (time, hru)

    # Two corrections to get the solar geometry in phase with the radiation:
    #  1. The METSIM forcing is in LOCAL SOLAR time (SWRadAtm peaks at hour 12 in
    #     every month, both basins), so shift by -lon/15 h to reach UTC.
    #  2. Timestamps are interval-START: SWRadAtm is symmetric about hour 11.5
    #     (h11 and h12 are both 930 W/m2), so the representative solar time for
    #     the bin labelled h is h + 0.5.  Without this, E-facing slopes are
    #     systematically favoured over W-facing ones.
    times_utc = times + pd.Timedelta(hours=-lon / 15.0 + 0.5)

    # Solar position at the basin centroid (the domain spans <0.5 deg; the
    # intra-basin difference in solar geometry is negligible next to slope/aspect).
    sp = pvlib.solarposition.get_solarposition(times_utc, lat, lon)
    zen = np.deg2rad(sp["apparent_zenith"].values)          # (time,)
    azi = np.deg2rad(sp["azimuth"].values)                  # (time,)
    cos_z = np.cos(zen)

    # Erbs decomposition needs a per-timestep GHI; use the basin-mean.
    ghi_mean = np.nanmean(ghi, axis=1)
    erbs = pvlib.irradiance.erbs(ghi_mean, sp["apparent_zenith"].values, times_utc)
    dhi_frac = np.where(ghi_mean > 0, erbs["dhi"].values / np.maximum(ghi_mean, 1e-6), 1.0)
    dhi_frac = np.clip(dhi_frac, 0.0, 1.0)

    dhi = ghi * dhi_frac[:, None]                           # diffuse   (time, hru)
    bhi = ghi - dhi                                         # beam on horizontal

    # Incidence angle on each tilted HRU surface
    cos_i = (np.cos(slope)[None, :] * cos_z[:, None]
             + np.sin(slope)[None, :] * np.sin(zen)[:, None]
             * np.cos(azi[:, None] - aspect[None, :]))
    cos_i = np.clip(cos_i, 0.0, None)

    # Beam correction factor = cos_i / cos_z, capped (cos_z -> 0 at sunrise/set)
    denom = np.maximum(cos_z, 0.05)[:, None]
    f_beam = np.clip(cos_i / denom, 0.0, MAX_DIRECT_GAIN)
    f_beam = np.where(cos_z[:, None] <= 0.0, 0.0, f_beam)

    svf = ((1.0 + np.cos(slope)) / 2.0)[None, :]
    sw_new = bhi * f_beam + dhi * svf + GROUND_ALBEDO * ghi * (1.0 - svf)
    sw_new = np.clip(sw_new, 0.0, 1400.0)

    ds["SWRadAtm"] = xr.DataArray(sw_new.astype(np.float32),
                                  dims=ds["SWRadAtm"].dims,
                                  attrs=ds["SWRadAtm"].attrs)
    ds["SWRadAtm"].attrs["terrain_correction"] = (
        "slope/aspect beam correction + sky-view factor (apply_terrain_sw.py)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dst)
    return {"mean_before": np.nanmean(ghi, axis=0),
            "mean_after": np.nanmean(sw_new, axis=0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain-dir", required=True, type=Path)
    p.add_argument("--src-forcing", required=True)
    p.add_argument("--dst-forcing", default="SUMMA_input")
    p.add_argument("--limit", type=int, default=0, help="process only N files (testing)")
    args = p.parse_args()

    dom = args.domain_dir
    settings = dom / "settings" / "SUMMA"
    slope, aspect, lat, lon, n_hru = hru_geometry(settings)

    src_dir = dom / "forcing" / args.src_forcing
    dst_dir = dom / "forcing" / args.dst_forcing
    files = sorted(src_dir.glob("*.nc"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No .nc under {src_dir}")

    print(f"{dom.name}: {n_hru} HRUs, centroid ({lat:.3f}, {lon:.3f})")
    print(f"  slope  {np.rad2deg(slope).min():.1f}-{np.rad2deg(slope).max():.1f} deg")
    print(f"  {len(files)} files: {src_dir.name} -> {dst_dir.name}")

    before = np.zeros(n_hru)
    after = np.zeros(n_hru)
    for i, f in enumerate(files, 1):
        r = correct_file(f, dst_dir / f.name, slope, aspect, lat, lon)
        before += r["mean_before"]
        after += r["mean_after"]
        if i % 60 == 0 or i == len(files):
            print(f"    {i}/{len(files)}")

    before /= len(files)
    after /= len(files)

    a = xr.open_dataset(settings / "attributes.nc")
    elev = a["elevation"].values
    asp_deg = a["aspect"].values
    print("\n  hru  elev   aspect        SW before   SW after   change")
    for h in np.argsort(asp_deg):
        print(f"  {h+1:>3}  {elev[h]:6.0f}  {asp_deg[h]:6.1f}deg  "
              f"{before[h]:9.1f}  {after[h]:9.1f}  {100*(after[h]-before[h])/before[h]:+6.1f}%")

    # The check that matters: north-facing must receive less than south-facing.
    north = np.abs(((asp_deg - 0) + 180) % 360 - 180) < 60
    south = np.abs(((asp_deg - 180) + 180) % 360 - 180) < 60
    if north.any() and south.any():
        print(f"\n  N-facing mean SW: {after[north].mean():.1f} W/m2  (n={north.sum()})")
        print(f"  S-facing mean SW: {after[south].mean():.1f} W/m2  (n={south.sum()})")
        d = after[south].mean() - after[north].mean()
        print(f"  S - N = {d:+.1f} W/m2   "
              f"{'OK: south > north' if d > 0 else '*** FAIL: north >= south ***'}")


if __name__ == "__main__":
    main()
