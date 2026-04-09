"""
generate_multiHRU_files.py
Reads CONFLUENCE config YAML + HRU shapefile to generate:
  - coldState_multiHRU.nc
  - trialParams_multiHRU.nc
  - calib_bounds_multiHRU.json

Usage:
  python generate_multiHRU_files.py --config config_East_River_distributed_bigBuckt.yaml
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import netCDF4 as nc
import geopandas as gpd
import yaml

# ═══════════════════════════════════════════════════════════════
# SECTION 1: LOAD AND RESOLVE CONFIG
# ═══════════════════════════════════════════════════════════════

def load_config(yaml_path):
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def resolve_path(cfg, key, *subpath_parts):
    """Resolve a CONFLUENCE 'default' path using standard conventions.
    If the config value is not 'default', use it as-is."""
    value = cfg.get(key, 'default')
    if str(value).lower() != 'default':
        return Path(value)
    base  = Path(cfg['CONFLUENCE_DATA_DIR'])
    domain = cfg['DOMAIN_NAME']
    return base / f'domain_{domain}' / Path(*subpath_parts)

def resolve_summa_settings(cfg):
    return resolve_path(cfg, 'SETTINGS_SUMMA_PATH',
                        'settings', 'SUMMA')

def resolve_catchment_shp(cfg):
    shp_dir  = resolve_path(cfg, 'CATCHMENT_PATH',
                            'shapefiles', 'catchment')
    shp_name = cfg.get('CATCHMENT_SHP_NAME', 'default')
    if str(shp_name).lower() == 'default':
        # CONFLUENCE convention: {DOMAIN_NAME}_HRUs.shp
        shp_name = f"{cfg['DOMAIN_NAME']}_HRUs_elevation.shp"
    return shp_dir / shp_name

# ═══════════════════════════════════════════════════════════════
# SECTION 2: READ HRU SHAPEFILE
# Extracts HRU_IDs, GRU_IDs, areas, and mean elevations.
# Sorts by elevation descending (alpine first) to match the
# physical convention: HRU 0 = highest band.
# ═══════════════════════════════════════════════════════════════

# Candidate elevation field names from CONFLUENCE/gistool outputs
ELEV_FIELD_CANDIDATES = [
    'elev_mean', 'ELEV_MEAN', 'mean_elev', 'elevMean',
    'dem_mean',  'DEM_MEAN',  'elev',      'ELEV',
]

def read_hru_shapefile(cfg):
    shp_path = resolve_catchment_shp(cfg)
    print(f'Reading shapefile: {shp_path}')
    gdf = gpd.read_file(shp_path)

    hru_field  = cfg['CATCHMENT_SHP_HRUID']   # 'HRU_ID'
    gru_field  = cfg['CATCHMENT_SHP_GRUID']   # 'GRU_ID'
    area_field = cfg['CATCHMENT_SHP_AREA']    # 'HRU_area'
    lat_field  = cfg['CATCHMENT_SHP_LAT']     # 'center_lat'

    # Detect elevation field
    elev_field = None
    for candidate in ELEV_FIELD_CANDIDATES:
        if candidate in gdf.columns:
            elev_field = candidate
            break
    if elev_field is None:
        # Fall back: derive elevation from center_lat using DEM mean
        # or raise a clear error listing available fields
        available = list(gdf.columns)
        raise ValueError(
            f'No elevation field found. Available columns: {available}\n'
            f'Set ELEV_FIELD_CANDIDATES in the script to match your shapefile.'
        )
    print(f'  Using elevation field: {elev_field!r}')
    print(f'  {len(gdf)} HRUs found')

    # Sort descending by elevation (alpine → riparian)
    gdf = gdf.sort_values(elev_field, ascending=False).reset_index(drop=True)

    hru_ids  = gdf[hru_field].values.astype(int)
    gru_ids  = np.unique(gdf[gru_field].values.astype(int))
    areas    = gdf[area_field].values.astype(float)
    elevs    = gdf[elev_field].values.astype(float)

    # Area-weighted mean elevation → reference for lapse-rate deltas
    ref_elev = np.average(elevs, weights=areas)
    print(f'  Elevation range: {elevs.min():.0f} – {elevs.max():.0f} m')
    print(f'  Area-weighted mean elevation: {ref_elev:.1f} m  (lapse reference)')

    return {
        'hru_ids':  hru_ids,
        'gru_ids':  gru_ids,
        'areas':    areas,
        'elevs':    elevs,
        'ref_elev': ref_elev,
        'n_hru':    len(hru_ids),
        'gdf':      gdf,
    }

# ═══════════════════════════════════════════════════════════════
# SECTION 3: PARSE localParamInfo BOUNDS
# ═══════════════════════════════════════════════════════════════

def parse_local_param_info(settings_dir, cfg):
    fname = cfg.get('SETTINGS_SUMMA_LOCAL_PARAMS_FILE', 'localParamInfo.txt')
    path  = settings_dir / fname
    bounds = {}
    pattern = re.compile(
        r'^\s*(\w+)\s*\|\s*([\d.eEdD+\-]+)\s*\|\s*([\d.eEdD+\-]+)\s*\|\s*([\d.eEdD+\-]+)'
    )
    def ff(s): return float(s.replace('d','e').replace('D','e'))
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                bounds[m.group(1)] = {
                    'default': ff(m.group(2)),
                    'min':     ff(m.group(3)),
                    'max':     ff(m.group(4)),
                }
    print(f'Parsed {len(bounds)} parameter bounds from {path.name}')
    return bounds

# ═══════════════════════════════════════════════════════════════
# SECTION 4: PARAMETER SETUP — loaded from config
# PARAMS_TO_CALIBRATE drives what goes into trialParams and the
# calibration JSON. Split into shared vs. per-HRU here.
# ═══════════════════════════════════════════════════════════════

# Parameters that stay uniform across all HRUs.
# Add to this list to prevent a PARAMS_TO_CALIBRATE entry from
# being expanded per-HRU.
FORCE_SHARED = {
    'albedoDecayRate',  
    'aquiferBaseflowRate', 
    'aquiferBaseflowExp', 
    'aquiferScaleFactor',
    'routingGammaShape', 
    'routingGammaScale'
}

def build_param_tables(cfg, hru_info, bounds):
    """
    For each param in PARAMS_TO_CALIBRATE:
      - shared params → one value written to all HRU rows
      - per-HRU params → default value from localParamInfo,
        scaled by a simple elevation multiplier as a physically
        motivated starting point. Replace with your lumped
        calibrated value before running.
    """
    params_str = cfg.get('PARAMS_TO_CALIBRATE', '')
    param_names = [p.strip() for p in params_str.split(',') if p.strip()]
    elevs    = hru_info['elevs']
    ref_elev = hru_info['ref_elev']
    n_hru    = hru_info['n_hru']

    # Elevation-scaling functions for per-HRU initial values.
    # These are physically motivated starting points, not calibrated values.
    # Replace starting_value with your lumped optimum when available.
    def elev_scale(base, elev, ref, factor=0.3):
        # positive factor → higher value at elevation
        return base * (1. + factor * (elev - ref) / ref)

    shared_params  = {}
    per_hru_params = {}

    for name in param_names:
        if name not in bounds:
            print(f'  Warning: {name} not in localParamInfo — skipping')
            continue
        default = bounds[name]['default']

        if name in FORCE_SHARED:
            shared_params[name] = default
        else:
            # Generate per-HRU starting values scaled by elevation
            if name == 'k_soil':
                # higher k at elevation (coarser, shallower soils)
                vals = [np.clip(elev_scale(default, e, ref_elev, +0.5),
                                bounds[name]['min'], bounds[name]['max'])
                        for e in elevs]
            elif name == 'aquiferScaleFactor':
                # smaller storage at elevation
                vals = [np.clip(elev_scale(default, e, ref_elev, -0.4),
                                bounds[name]['min'], bounds[name]['max'])
                        for e in elevs]
            elif name == 'aquiferBaseflowRate':
                # faster drainage at elevation
                vals = [np.clip(elev_scale(default, e, ref_elev, +0.4),
                                bounds[name]['min'], bounds[name]['max'])
                        for e in elevs]
            elif name == 'qSurfScale':
                # lower scale at elevation (more abrupt saturation)
                vals = [np.clip(elev_scale(default, e, ref_elev, -0.3),
                                bounds[name]['min'], bounds[name]['max'])
                        for e in elevs]
            else:
                # no scaling guidance — use default for all HRUs
                vals = [default] * n_hru

            per_hru_params[name] = vals

    return shared_params, per_hru_params

# ═══════════════════════════════════════════════════════════════
# SECTION 5: GENERATE coldState_multiHRU.nc
# ═══════════════════════════════════════════════════════════════

def make_cold_state(cfg, settings_dir, hru_info):
    lapse    = cfg.get('LAPSE_RATE', 0.0065)          # K/m from config
    ref_elev = hru_info['ref_elev']
    elevs    = hru_info['elevs']
    hru_ids  = hru_info['hru_ids']
    n_hru    = hru_info['n_hru']

    in_path  = settings_dir / cfg['SETTINGS_SUMMA_COLDSTATE']
    out_path = settings_dir / 'coldState_multiHRU.nc'

    src = nc.Dataset(in_path,  'r')
    dst = nc.Dataset(out_path, 'w')

    for name, dim in src.dimensions.items():
        dst.createDimension(name, n_hru if name == 'hru' else len(dim))
    dst.setncatts({a: getattr(src, a) for a in src.ncattrs()})

    # Derive soil layer depths from existing cold state and scale by elevation.
    # If CALIBRATE_DEPTH is true in config, these are later updated by the
    # calibration loop — otherwise they stay fixed.
    src_depths = src.variables['mLayerDepth'][:, 0]   # (midToto,) from HRU 0
    total_depth = src_depths.sum()

    # Scale total soil depth by elevation (shallower at elevation)
    depth_scales = np.clip(
        1. - 1.5 * (elevs - ref_elev) / ref_elev, 0.3, 2.0
    )
    layer_depths = np.outer(depth_scales, src_depths / total_depth)  # (n_hru, nLayers)
    layer_depths *= depth_scales[:, None] * total_depth              # rescale each HRU total

    for name, var in src.variables.items():
        fv = getattr(var, '_FillValue', None)
        v  = dst.createVariable(name, var.datatype, var.dimensions,
                                fill_value=fv)
        v.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                     if a != '_FillValue'})

        if name == 'hruId':
            v[:] = hru_ids

        elif 'hru' not in var.dimensions:
            v[:] = var[:]

        elif name == 'mLayerDepth':
            v[:] = layer_depths.T               # (midToto, n_hru)

        elif name == 'iLayerHeight':
            for i in range(n_hru):
                v[:, i] = np.concatenate([[0.], -np.cumsum(layer_depths[i])])

        elif name in ('mLayerTemp', 'scalarCanairTemp', 'scalarCanopyTemp'):
            for i in range(n_hru):
                dT = -lapse * (elevs[i] - ref_elev)
                v[..., i] = src.variables[name][..., 0] + dT
        else:
            for i in range(n_hru):
                v[..., i] = src.variables[name][..., 0]

    src.close(); dst.close()
    print(f'Written: {out_path}')
    return layer_depths

# ═══════════════════════════════════════════════════════════════
# SECTION 6: GENERATE trialParams_multiHRU.nc
# ═══════════════════════════════════════════════════════════════

def make_trial_params(cfg, settings_dir, hru_info,
                      shared_params, per_hru_params):
    hru_ids = hru_info['hru_ids']
    gru_ids = hru_info['gru_ids']
    n_hru   = hru_info['n_hru']

    in_path  = settings_dir / cfg['SETTINGS_SUMMA_TRIALPARAMS']
    out_path = settings_dir / 'trialParams_multiHRU.nc'

    src = nc.Dataset(in_path,  'r')
    dst = nc.Dataset(out_path, 'w')

    for name, dim in src.dimensions.items():
        if   name == 'hru': dst.createDimension('hru', n_hru)
        elif name == 'gru': dst.createDimension('gru', len(gru_ids))
        else:               dst.createDimension(name, len(dim))
    dst.setncatts({a: getattr(src, a) for a in src.ncattrs()})

    all_params = {
        **{k: [v] * n_hru for k, v in shared_params.items()},
        **per_hru_params,
    }

    for name, var in src.variables.items():
        fv = getattr(var, '_FillValue', None)
        v  = dst.createVariable(name, var.datatype, var.dimensions,
                                fill_value=fv)
        v.setncatts({a: var.getncattr(a) for a in var.ncattrs()
                     if a != '_FillValue'})

        if name == 'hruId':
            v[:] = hru_ids
        elif name == 'gruId':
            v[:] = gru_ids
        elif name in all_params and 'hru' in var.dimensions:
            v[:] = all_params[name]
        elif 'hru' in var.dimensions:
            for i in range(n_hru):
                v[..., i] = src.variables[name][..., 0]
        else:
            v[:] = src.variables[name][:]

    src.close(); dst.close()
    print(f'Written: {out_path}')

# ═══════════════════════════════════════════════════════════════
# SECTION 7: EXPORT CALIBRATION BOUNDS JSON
# ═══════════════════════════════════════════════════════════════

def export_calib_bounds(cfg, settings_dir, hru_info,
                        shared_params, per_hru_params, bounds):
    hru_ids = hru_info['hru_ids']
    calib   = {'shared': [], 'per_hru': [], 'meta': {
        'domain':     cfg['DOMAIN_NAME'],
        'experiment': cfg['EXPERIMENT_ID'],
        'calib_period':    cfg['CALIBRATION_PERIOD'],
        'eval_period':     cfg['EVALUATION_PERIOD'],
        'optimizer':       cfg['ITERATIVE_OPTIMIZATION_ALGORITHM'],
        'metric':          cfg['OPTIMIZATION_METRIC'],
        'n_iterations':    cfg['NUMBER_OF_ITERATIONS'],
        'population_size': cfg['POPULATION_SIZE'],
    }}

    for name, val in shared_params.items():
        if name in bounds:
            calib['shared'].append({
                'param': name, 'value': val,
                'min': bounds[name]['min'], 'max': bounds[name]['max'],
            })

    for name, vals in per_hru_params.items():
        if name in bounds:
            for i, v in enumerate(vals):
                calib['per_hru'].append({
                    'param': name, 'hru_index': i,
                    'hru_id': int(hru_ids[i]),
                    'value': float(v),
                    'min': bounds[name]['min'],
                    'max': bounds[name]['max'],
                })

    out = settings_dir / 'calib_bounds_multiHRU.json'
    with open(out, 'w') as f:
        json.dump(calib, f, indent=2)

    n_total = len(calib['shared']) + len(calib['per_hru'])
    print(f'Written: {out}')
    print(f'  {len(calib["shared"])} shared + '
          f'{len(calib["per_hru"])} per-HRU = {n_total} calibration params')

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to CONFLUENCE YAML config file')
    args = parser.parse_args()

    cfg          = load_config(args.config)
    settings_dir = resolve_summa_settings(cfg)
    hru_info     = read_hru_shapefile(cfg)
    bounds       = parse_local_param_info(settings_dir, cfg)
    shared, per_hru = build_param_tables(cfg, hru_info, bounds)
    layer_depths = make_cold_state(cfg, settings_dir, hru_info)
    make_trial_params(cfg, settings_dir, hru_info, shared, per_hru)
    export_calib_bounds(cfg, settings_dir, hru_info, shared, per_hru, bounds)

    print('\nSummary:')
    print(f'  Domain:   {cfg["DOMAIN_NAME"]}')
    print(f'  HRUs:     {hru_info["n_hru"]}')
    print(f'  Elev:     {hru_info["elevs"].min():.0f} – '
          f'{hru_info["elevs"].max():.0f} m')
    print(f'  Soil depths (m): '
          + ', '.join(f'{d.sum():.2f}' for d in layer_depths))