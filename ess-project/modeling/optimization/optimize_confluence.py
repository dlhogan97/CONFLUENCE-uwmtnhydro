#!/usr/bin/env python3
"""
CONFLUENCE Parameter Calibration Optimization with Complete Preprocessing

Flexible optimization script supporting multiple algorithms and domains.

Standalone script that handles:
1. Configuration setup (any domain)
2. Parameter initialization  
3. Temperature lapsing and forcing adjustments
4. Model preprocessing
5. Parameter calibration optimization (DDS, PSO, SCE, GA, DE)
6. Results analysis

Usage:
    # DDS with single processor (recommended for sequential)
    python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1
    
    # PSO with parallel processing
    python optimize_confluence.py --config config_East_River.yaml --algorithm PSO --mpi-processes 4
    
    # Custom experiment name
    python optimize_confluence.py --config config_template.yaml --run-name "test_run" --algorithm DDS --mpi-processes 1
    
Or with bash wrapper:
    ./optimize.sh --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import yaml

# Add CONFLUENCE root to path (up 4 levels to CONFLUENCE-uwmtnhydro)
# optimization -> modeling -> ess-project -> CONFLUENCE-uwmtnhydro
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from CONFLUENCE import CONFLUENCE
from utils.custom.adjust_settings import edit_modelDecisions, update_and_reformat_parameter_file


def setup_logger(name, log_level=logging.INFO):
    """Configure logging"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger('CONFLUENCE Optimizer')


def empirical_lw_dilley_obrien(ta, ea, pa):
    """
    Calculate incoming longwave radiation using Dilley & O'Brien (2002) method.
    
    Based on Equation (6) in:
    Dilley, A.C., and D.M. O'Brien, 2002: Estimating downwelling longwave 
    irradiance at the surface from cloud amount and cloud type. 
    J. Geophys. Res., 107(D13), 4280, doi:10.1029/2001JD000822.
    
    Parameters:
    -----------
    ta : array-like
        Air temperature (K)
    ea : array-like
        Water vapor pressure (Pa)
    pa : array-like
        Air pressure (Pa)
        
    Returns:
    --------
    lw : array-like
        Downwelling longwave radiation (W/m²)
    """
    # Stefan-Boltzmann constant
    sigma = 5.670374419e-8  # W m^-2 K^-4
    
    # Calculate clear-sky emissivity (Equation 2)
    emiss_clear = 0.24 + 4.81e-4 * (ea / 100.0) ** 0.5  # ea converted from Pa to hPa
    
    # Cloud adjustment factor (Equation 3) - assuming average cloud cover
    # For lumped model, we'll use all-sky approach
    cls = 1.0 + 0.22 * ((pa / 101325.0) ** 2)  # Clear-sky adjustment
    
    # Calculate LW radiation
    lw = emiss_clear * cls * sigma * ta ** 4
    
    return lw


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_decision_option(option_value, default):
    """Return a single decision value from either a scalar or a candidate list."""
    if option_value is None:
        return default
    if isinstance(option_value, (list, tuple)):
        return option_value[0] if option_value else default
    return option_value


def setup_parameters(confluence, config_dict, use_previous=False):
    """Setup parameter files with values from base settings or previous optimization"""
    logger.info("\n" + "=" * 70)
    logger.info("PARAMETER AND MODEL DECISION SETUP")
    logger.info("=" * 70)
    
    # Path to base settings (up to ess-project level: 2 parents up)
    base_settings_dir = Path(__file__).parent.parent.parent / "0_base_settings" / "SUMMA"
    project_settings_dir = (
        Path(config_dict['CONFLUENCE_DATA_DIR']) / 
        f"domain_{config_dict['DOMAIN_NAME']}" / 
        "settings" / "SUMMA"
    )
    project_settings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📋 Copying base settings...")
    for fname in ["modelDecisions.txt", "localParamInfo.txt", "basinParamInfo.txt"]:
        src = base_settings_dir / fname
        dst = project_settings_dir / fname
        if src.exists():
            shutil.copy(src, dst)
        else:
            logger.warning(f"   ⚠️  {fname} not found")
    
    modelDecision_file = project_settings_dir / "modelDecisions.txt"
    localParamInfo_file = project_settings_dir / "localParamInfo.txt"
    basinParamInfo_file = project_settings_dir / "basinParamInfo.txt"
    
    # Update model decisions (read from config file if available)
    logger.info("✓ Updating model decisions...")
    
    # Extract model decisions from SUMMA_DECISION_OPTIONS in config
    summa_decisions = config_dict.get('SUMMA_DECISION_OPTIONS', {})
    modelDecision_updates = {
        'groundwatr': _resolve_decision_option(summa_decisions.get('groundwatr'), 'bigBuckt'),
        'bcLowrSoiH': _resolve_decision_option(summa_decisions.get('bcLowrSoiH'), 'drainage'),
        'spatial_gw': _resolve_decision_option(summa_decisions.get('spatial_gw'), 'localColumn'),
        'alb_method': _resolve_decision_option(summa_decisions.get('alb_method'), 'varDecay'),
    }
    logger.info(f"   - groundwatr: {modelDecision_updates['groundwatr']}")
    logger.info(f"   - bcLowrSoiH: {modelDecision_updates['bcLowrSoiH']}")
    logger.info(f"   - spatial_gw: {modelDecision_updates['spatial_gw']}")
    logger.info(f"   - alb_method: {modelDecision_updates['alb_method']}")
    edit_modelDecisions(modelDecision_file, modelDecision_updates)
    
    # Load parameters
    best_params_dict = {}
    
    if use_previous:
        logger.info("\n📊 Looking for previous best parameters...")
        opt_base_dir = Path(config_dict['CONFLUENCE_DATA_DIR']) / f"domain_{config_dict['DOMAIN_NAME']}" / "optimisation"
        opt_dirs = sorted(
            [d for d in opt_base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if opt_dirs:
            best_params_csv = opt_dirs[0] / "best_parameters.csv"
            if best_params_csv.exists():
                best_params_df = pd.read_csv(best_params_csv)
                best_params_dict = dict(zip(best_params_df['parameter'], best_params_df['value']))
                logger.info(f"   ✓ Loaded {len(best_params_dict)} optimized parameters")
    
    # Use defaults if no previous optimization
    if not best_params_dict:
        logger.info("\n✏️  Using default parameter values...")
        best_params_dict = {
            'tempCritRain': 274.1,
            'k_soil': 9.4e-6,
            'theta_sat': 0.5160,
            'theta_res': 0.0270,
            'rootingDepth': 6.876,
            "basin__aquiferHydCond": 0.0010,
            "basin__aquiferScaleFactor": 50.0,
            "routingGammaShape": 2.5,
            "routingGammaScale": 4.6e4,
        }
    
    # Update parameter files
    logger.info(f"📝 Updating parameter files...")
    
    local_updates = {k: v for k, v in best_params_dict.items() 
                    if not k.startswith('basin__') and not k.startswith('routing')}
    if local_updates:
        update_and_reformat_parameter_file(localParamInfo_file, local_updates, reformat_all=True)
    
    basin_updates = {k: v for k, v in best_params_dict.items() 
                    if k.startswith('basin__') or k.startswith('routing')}
    if basin_updates:
        update_and_reformat_parameter_file(basinParamInfo_file, basin_updates, reformat_all=True)
    
    logger.info(f"✓ Parameters synced to: {project_settings_dir}")
    return project_settings_dir


def adjust_forcing_data(config_dict):
    """
    Apply temperature lapsing and radiation correction to forcing data.
    This matches the preprocessing from the notebook.
    """
    logger.info("\n" + "=" * 70)
    logger.info("FORCING DATA ADJUSTMENT")
    logger.info("=" * 70)
    
    project_dir = (
        Path(config_dict['CONFLUENCE_DATA_DIR']) / 
        f"domain_{config_dict['DOMAIN_NAME']}"
    )
    
    # Load DEM for elevation calculations
    # Try multiple common DEM path patterns
    dem_candidates = [
        project_dir / 'attributes' / 'elevation' / 'dem' / f"domain_{config_dict['DOMAIN_NAME']}_elv.tif",
        project_dir / 'attributes' / 'elevation' / 'dem' / "dem.tif",
        project_dir / 'attributes' / 'dem.tif',
    ]
    
    dem_path = None
    for candidate in dem_candidates:
        if candidate.exists():
            dem_path = candidate
            break
    
    if dem_path is None:
        logger.warning(f"⚠️  DEM not found. Searched:")
        for c in dem_candidates:
            logger.warning(f"     {c}")
        logger.warning(f"   Skipping elevation adjustments")
        return
    
    logger.info(f"\n📊 Loading DEM from: {dem_path}")
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
    
    mean_elevation = np.nanmean(dem_data)
    
    # Get HRU mean elevation from shapefile
    hru_path = project_dir / 'shapefiles' / 'catchment' / f"{config_dict['DOMAIN_NAME']}_HRUs_{config_dict['DOMAIN_DISCRETIZATION']}.shp"
    
    if hru_path.exists():
        import geopandas as gpd
        hru_gdf = gpd.read_file(hru_path)
        mean_hru_elevation = hru_gdf['elev_mean'].mean()
    else:
        logger.warning(f"⚠️  HRU shapefile not found, using DEM mean elevation")
        mean_hru_elevation = mean_elevation
    
    # Calculate temperature adjustment
    elevation_difference = mean_hru_elevation - mean_elevation
    lapse_rate = 6.5 / 1000  # °C per meter
    temperature_adjustment = elevation_difference * lapse_rate
    
    logger.info(f"\n🌡️  Temperature Adjustment Calculations:")
    logger.info(f"   Mean DEM elevation: {mean_elevation:.1f} m")
    logger.info(f"   Mean HRU elevation: {mean_hru_elevation:.1f} m")
    logger.info(f"   Elevation difference: {elevation_difference:.1f} m")
    logger.info(f"   Temperature adjustment: {temperature_adjustment:.3f} K")
    
    # Find and process forcing files
    forcing_dir = project_dir / 'forcing' / 'forcing_noTadjust'
    
    if not forcing_dir.exists():
        logger.warning(f"⚠️  Forcing directory not found: {forcing_dir}")
        return
    
    file_list = sorted(forcing_dir.glob("*.nc"))
    logger.info(f"\n📦 Processing {len(file_list)} forcing files...")
    
    output_dir = project_dir / 'forcing' / 'SUMMA_input'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(file_list, 1):
        logger.info(f"   [{i}/{len(file_list)}] Processing {file.name}...")
        
        ds = xr.open_dataset(str(file))
        
        # Apply temperature adjustment
        if 'airtemp' in ds.data_vars:
            ds['airtemp'] = ds['airtemp'] + temperature_adjustment
        
        # Recalculate incoming longwave radiation
        if all(var in ds.data_vars for var in ['airtemp', 'spechum', 'airpres']):
            ds['LWRadAtm'] = empirical_lw_dilley_obrien(
                ds['airtemp'], 
                ds['spechum'], 
                ds['airpres']
            )
        
        # Save adjusted data
        output_path = output_dir / file.name
        ds.to_netcdf(output_path)
        ds.close()
    
    logger.info(f"✓ Forcing data adjustment complete. Output: {output_dir}")


def protect_and_preprocess(confluence, project_settings_dir):
    """Backup custom parameters, run preprocessing, restore parameters"""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL PREPROCESSING")
    logger.info("=" * 70)
    
    # Backup custom parameters
    logger.info("\n🔒 Backing up custom parameter files...")
    backup_dir = project_settings_dir / "backup_custom_config"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for fname in ["modelDecisions.txt", "localParamInfo.txt", "basinParamInfo.txt"]:
        src = project_settings_dir / fname
        dst = backup_dir / fname
        if src.exists():
            shutil.copy(src, dst)
    
    # Run preprocessing
    logger.info("\n⚙️  Running model-agnostic preprocessing...")
    confluence.managers['data'].run_model_agnostic_preprocessing()
    logger.info("✓ Model-agnostic preprocessing complete")
    
    logger.info("\n⚙️  Running model-specific preprocessing...")
    confluence.managers['model'].preprocess_models()
    logger.info("✓ Model-specific preprocessing complete")
    
    # Restore custom parameters
    logger.info("\n🔓 Restoring custom parameter files...")
    for fname in ["modelDecisions.txt", "localParamInfo.txt", "basinParamInfo.txt"]:
        src = backup_dir / fname
        dst = project_settings_dir / fname
        if src.exists():
            shutil.copy(src, dst)


def run_optimization(config_path, run_name=None, use_previous_params=False, algorithm=None, mpi_processes=None):
    """Main optimization execution"""
    start_time = datetime.now()
    
    # Load configuration
    logger.info(f"\n📂 Loading configuration: {config_path}")
    config_dict = load_config(config_path)
    
    # Override MPI processes if specified on command line
    if mpi_processes is not None:
        original_mpi = config_dict.get('MPI_PROCESSES', 1)
        logger.info(f"\n⚙️  Overriding MPI processes: {original_mpi} → {mpi_processes}")
        config_dict['MPI_PROCESSES'] = mpi_processes
    
    # Override algorithm if specified on command line
    original_algorithm = config_dict.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'DDS')
    if algorithm:
        logger.info(f"\n⚙️  Overriding optimization algorithm: {original_algorithm} → {algorithm}")
        config_dict['ITERATIVE_OPTIMIZATION_ALGORITHM'] = algorithm
    elif config_dict.get('MPI_PROCESSES', 1) == 1:
        # Recommend DDS for single processor
        current_alg = config_dict.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'DDS')
        if current_alg != 'DDS':
            logger.warning(f"\n💡 With MPI_PROCESSES=1, DDS is recommended (not {current_alg})")
            logger.warning(f"   DDS is sequential/intelligent - optimal for single processor")
            logger.warning(f"   Use: --algorithm DDS to switch")
    
    # Write modified config back to YAML so CONFLUENCE reads the overrides
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    if algorithm or mpi_processes is not None:
        logger.info(f"   ✓ Config updated")
    
    opt_alg = config_dict.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'DDS')
    mpi_procs = config_dict.get('MPI_PROCESSES', 1)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"CONFLUENCE {opt_alg} PARAMETER CALIBRATION OPTIMIZATION")
    logger.info("=" * 70)
    
    # Warn if DDS with parallel processing
    if opt_alg == 'DDS' and mpi_procs > 1:
        logger.warning("\n⚠️⚠️⚠️  CRITICAL WARNING ⚠️⚠️⚠️")
        logger.warning(f"  DDS is a SEQUENTIAL algorithm - each iteration needs the previous best!")
        logger.warning(f"  Config has MPI_PROCESSES = {mpi_procs}")
        logger.warning(f"  Running DDS in parallel defeats convergence.")
        logger.warning(f"  RECOMMENDATION: Set MPI_PROCESSES = 1 or use PSO/SCE\n")
    
    # Override experiment ID if custom run name provided
    if run_name:
        config_dict['EXPERIMENT_ID'] = f"{run_name}_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"   Custom run name: {config_dict['EXPERIMENT_ID']}")
        # Update config file with new experiment ID
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Initialize CONFLUENCE (now reads the updated config)
    logger.info(f"\n🔧 Initializing CONFLUENCE...")
    confluence = CONFLUENCE(
        config_path=config_path
    )
    
    # Setup parameters
    project_settings_dir = setup_parameters(confluence, config_dict, use_previous=use_previous_params)
    
    # Initialize project
    logger.info(f"\n📋 Setting up project structure...")
    project_dir = confluence.managers['project'].setup_project()
    confluence.managers['project'].create_pour_point()
    logger.info(f"   ✓ Project: {project_dir}")
    
    # Adjust forcing data (temperature, radiation, etc.)
    adjust_forcing_data(config_dict)
    
    # Protect custom parameters and run preprocessing
    protect_and_preprocess(confluence, project_settings_dir)
    
    # Run optimization
    logger.info(f"\n" + "=" * 70)
    logger.info(f"STARTING {opt_alg} OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Experiment ID: {config_dict['EXPERIMENT_ID']}")
    logger.info(f"Algorithm: {opt_alg}")
    logger.info(f"MPI Processes: {mpi_procs}")
    logger.info(f"Iterations: {config_dict.get('NUMBER_OF_ITERATIONS', 'default')}")
    logger.info(f"Parameters: {config_dict.get('PARAMS_TO_CALIBRATE', 'default')}")
    logger.info(f"Basin area: {config_dict.get('BASIN_AREA_M2', 'not specified')} m²")
    
    confluence.managers['optimization'].calibrate_model()
    
    elapsed = datetime.now() - start_time
    logger.info(f"\n✅ Optimization complete!")
    logger.info(f"   Elapsed time: {elapsed}")
    logger.info(f"   Results: {project_dir / 'optimisation'}")


def main():
    """Parse arguments and run optimization"""
    parser = argparse.ArgumentParser(
        description="Run optimization with preprocessing (supports multiple algorithms)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Algorithms:
  DDS  - Dynamical Dimensioned Search (sequential, intelligent)
  PSO  - Particle Swarm Optimization (parallel-friendly)
  SCE  - Shuffled Complex Evolution (parallel-friendly)
  GA   - Genetic Algorithm (parallel-friendly)
  DE   - Differential Evolution (parallel-friendly, robust)

Examples:
  # Standard run (uses algorithm from config)
  python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml
  
  # DDS with single processor (recommended combo)
  python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1
  
  # PSO with parallel processing
  python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml --algorithm PSO --mpi-processes 4
  
  # DDS with custom experiment name
  python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml --algorithm DDS --mpi-processes 1 --run-name "spongy"
  
  # Continue from previous best
  python optimize_confluence.py --config config_Tuolumne_lumped_v1.yaml --use-previous
        """
    )
    
    parser.add_argument('--config', required=True, help='Configuration YAML file')
    parser.add_argument('--algorithm', choices=['DDS', 'PSO', 'SCE', 'GA', 'DE'],
                       help='Optimization algorithm (overrides config file). DDS=Sequential (use with --mpi-processes 1), others are parallel-friendly')
    parser.add_argument('--mpi-processes', type=int, metavar='N',
                       help='Number of MPI processes (overrides config file). Use 1 with DDS, >1 with PSO/SCE/GA/DE')
    parser.add_argument('--run-name', help='Custom experiment name')
    parser.add_argument('--use-previous', action='store_true',
                       help='Initialize with previous best parameters')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Config files are at ess-project/0_config_files (up 2 parents)
        config_path = Path(__file__).parent.parent.parent / "0_config_files" / config_path
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        run_optimization(str(config_path), args.run_name, args.use_previous, args.algorithm, args.mpi_processes)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
