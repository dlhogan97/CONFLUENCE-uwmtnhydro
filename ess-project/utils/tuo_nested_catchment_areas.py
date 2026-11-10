"""Delineate nested Tuolumne gauge catchments (LBM, DFkBC, LABvTB, T120) on the MERIT-Hydro 90 m DEM with pysheds. Pour points snapped to the highest flow accumulation within 600 m.
-> modeling/calibration/observations/tuolumne_nested_gauge_areas.csv, figures/draft/tuo_nested_catchments.png"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, rasterio
from pysheds.grid import Grid
R = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"
dem = "/scratch/dlhogan/ess-project-data/geospatial-data/MERIT-Hydro/elv/elv_n30w120/n35w120_elv.tif"
pts = {"LBM": (-119.26180, 37.77821), "DFkBC": (-119.338341, 37.87672701), "LABvTB": (-119.3311127, 37.8694852), "T120": (-119.3547627, 37.8762952)}
bbox = (-119.6, 37.65, -119.1, 38.0); grid0 = Grid.from_raster(dem, data_name="dem"); grid = Grid.from_raster(dem, data_name="dem", window=bbox, window_crs=grid0.crs); d = grid.read_raster(dem, window=bbox, window_crs=grid0.crs)
pit = grid.fill_pits(d); fl = grid.fill_depressions(pit); inf = grid.resolve_flats(fl)
fdir = grid.flowdir(inf); acc = grid.accumulation(fdir)
lat = np.deg2rad(37.85); dx = 0.000833333*111320*np.cos(lat); dy = 0.000833333*110574; A = dx*dy/1e6
rows = []; masks = {}
for k, (x, y) in pts.items():
    xs, ys = grid.snap_to_mask(acc > 200, (x, y)); c = grid.catchment(x=xs, y=ys, fdir=fdir, xytype="coordinate"); m = np.asarray(c).astype(bool); masks[k] = m
    e = np.asarray(inf)[m]; rows.append(dict(id=k, lon=x, lat=y, snap_lon=xs, snap_lat=ys, area_km2=m.sum()*A, zmean=e.mean(), zmax=e.max(), zmin=e.min()))
T = pd.DataFrame(rows); print(T.round(2).to_string(index=False)); T.to_csv(R+"modeling/calibration/observations/tuolumne_nested_gauge_areas.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 7.5)); ext = grid.extent; ax.imshow(np.asarray(inf), cmap="Greys", extent=ext, origin="upper", vmin=2000, vmax=4000)
ax.imshow(np.where(np.asarray(acc) > 400, 1., np.nan), cmap="Blues", extent=ext, origin="upper", alpha=.8, vmin=0, vmax=1.5)
for k, c in zip(masks, ["#d55e00", "#2a78d6", "#009e73", "#cc79a7"]):
    ax.contour(masks[k].astype(float), levels=[.5], colors=c, extent=ext, origin="upper", linewidths=1.5); ax.plot(*T.set_index("id").loc[k, ["snap_lon", "snap_lat"]], "o", ms=6, mec="k", color=c, label=f"{k} {T.set_index('id').loc[k,'area_km2']:.0f} km²")
ax.legend(fontsize=8); ax.set_xlabel("lon"); ax.set_ylabel("lat"); fig.savefig(R+"figures/draft/tuo_nested_catchments.png", dpi=130, bbox_inches="tight")
fig, axs = plt.subplots(2, 2, figsize=(11, 10)); sub = (-119.45, -119.1, 37.68, 37.95)
for a, (k, c) in zip(axs.ravel(), zip(masks, ["#d55e00", "#2a78d6", "#009e73", "#cc79a7"])):
    a.imshow(np.asarray(inf), cmap="Greys", extent=ext, origin="upper", vmin=2000, vmax=4000); a.imshow(np.where(masks[k], 1., np.nan), cmap="Reds" if False else matplotlib.colors.ListedColormap([c]), extent=ext, origin="upper", alpha=.45)
    a.imshow(np.where(np.asarray(acc) > 400, 1., np.nan), cmap="Blues", extent=ext, origin="upper", alpha=.8, vmin=0, vmax=1.5); a.plot(*T.set_index("id").loc[k, ["lon", "lat"]], "k*", ms=11, label="gauge (given)"); a.plot(*T.set_index("id").loc[k, ["snap_lon", "snap_lat"]], "wo", mec="k", ms=5, label="snapped")
    a.set_xlim(sub[0], sub[1]); a.set_ylim(sub[2], sub[3]); a.set_title(f"{k}  {T.set_index('id').loc[k,'area_km2']:.0f} km²", loc="left"); a.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(R+"figures/draft/tuo_nested_catchments_panels.png", dpi=110, bbox_inches="tight")
# catchment polygons (EPSG:4326) for rasterising onto the Landsat grid
import geopandas as gpd; from rasterio.features import shapes; from shapely.geometry import shape
recs = []
for k, m in masks.items():
    geoms = [shape(g) for g, v in shapes(m.astype("uint8"), mask=m, transform=grid.affine) if v == 1]
    from shapely.ops import unary_union; recs.append(dict(id=k, area_km2=T.set_index("id").loc[k, "area_km2"], geometry=unary_union(geoms)))
gpd.GeoDataFrame(recs, crs="EPSG:4326").to_file(R+"modeling/calibration/observations/tuolumne_nested_catchments.geojson", driver="GeoJSON")
