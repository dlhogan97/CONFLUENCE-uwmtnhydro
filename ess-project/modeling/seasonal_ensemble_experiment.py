#!/usr/bin/env python3
"""
Seasonal Forcing Ensemble Experiment

This module implements an experiment that:
1. Runs optimization to find best parameters
2. Does a long-term (20-year) baseline simulation
3. Builds seasonal forcing ensembles by swapping one season of the 21st year
   with the same season from each of the 20 prior years
4. Runs all ensemble members
5. Evaluates sensitivity via water balance (ET, storage change) and runoff ratio
6. Produces spaghetti plots of precipitation and streamflow signals

Usage:
    from seasonal_ensemble_experiment import SeasonalEnsembleExperiment
    
    exp = SeasonalEnsembleExperiment(config_path="path/to/config.yaml")
    exp.run_full_workflow()
"""

import sys
import os
import logging
import shutil
import subprocess
import time as _time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# Add CONFLUENCE root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from CONFLUENCE import CONFLUENCE
from utils.evaluation.calculate_sim_stats import get_KGE, get_NSE, get_RMSE
from utils.custom.forcing_processor import ForcingProcessor

logger = logging.getLogger('SeasonalEnsemble')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Season definitions (meteorological seasons)
SEASONS = {
    'DJF': {'name': 'Winter', 'months': [12, 1, 2], 'color': '#2166ac'},
    'MAM': {'name': 'Spring', 'months': [3, 4, 5], 'color': '#4dac26'},
    'JJA': {'name': 'Summer', 'months': [6, 7, 8], 'color': '#d01c8b'},
    'SON': {'name': 'Fall', 'months': [9, 10, 11], 'color': '#e66101'},
}

FORCING_VARS = ['pptrate', 'SWRadAtm', 'LWRadAtm', 'airpres', 'airtemp', 'windspd', 'spechum']

# State variables that must appear in SUMMA output for warm-start extraction
# (variables NOT already in the default outputControl.txt)
WARM_STATE_EXTRA_VARS = [
    'scalarCanopyIce',
    'scalarSfcMeltPond',
    'scalarAquiferStorage',
    'scalarCanairTemp',
    'scalarCanopyTemp',
    'mLayerVolFracIce',
    'mLayerVolFracLiq',
    'mLayerMatricHead',
]


# =============================================================================
# Configuration
# =============================================================================

class ExperimentConfig:
    """Container for experiment configuration derived from CONFLUENCE config."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.raw = yaml.safe_load(f)
        self.config_path = Path(config_path)
        
        self.data_dir = Path(self.raw['CONFLUENCE_DATA_DIR'])
        self.code_dir = Path(self.raw['CONFLUENCE_CODE_DIR'])
        self.domain_name = self.raw['DOMAIN_NAME']
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Time settings — will be overridden for each phase
        self.base_experiment_id = self.raw.get('EXPERIMENT_ID', 'seasonal_ensemble')
        
        # Forcing
        self.forcing_dataset = self.raw.get('FORCING_DATASET', 'ERA5')
        self.forcing_timestep = self.raw.get('FORCING_TIME_STEP_SIZE', 3600)
        
        # Optimization
        self.opt_algorithm = self.raw.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'DDS')
        self.opt_iterations = self.raw.get('NUMBER_OF_ITERATIONS', 200)
        self.opt_metric = self.raw.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Model
        self.hydro_model = self.raw.get('HYDROLOGICAL_MODEL', 'SUMMA')
        self.basin_area_m2 = self.raw.get('BASIN_AREA_M2', None)
        
        # Paths
        self.forcing_dir = self.project_dir / 'forcing'
        self.summa_input_dir = self.forcing_dir / 'SUMMA_input'
        self.settings_dir = self.project_dir / 'settings' / 'SUMMA'
        self.obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        self.opt_dir = self.project_dir / 'optimisation'
        self.ensemble_dir = self.project_dir / 'seasonal_ensemble'
        self.plots_dir = self.project_dir / 'plots' / 'seasonal_ensemble'
    
    def get_confluence_config(self, **overrides) -> dict:
        """Return a copy of the config with optional overrides."""
        cfg = dict(self.raw)
        cfg.update(overrides)
        return cfg


# =============================================================================
# Phase 1: Optimization — Find Best Parameters  
# =============================================================================

class ParameterOptimizer:
    """Runs CONFLUENCE optimization and extracts best parameters."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
    
    def run_optimization(self) -> pd.DataFrame:
        """Run optimization via the existing CONFLUENCE optimization pipeline."""
        logger.info("=" * 70)
        logger.info("PHASE 1: PARAMETER OPTIMIZATION")
        logger.info("=" * 70)
        
        confluence = CONFLUENCE(config_path=str(self.cfg.config_path))
        confluence.managers['optimization'].calibrate_model()
        
        logger.info("Optimization complete.")
        return self.get_best_parameters()
    
    def get_best_parameters(self) -> pd.DataFrame:
        """Extract best parameters from the most recent optimization run."""
        opt_dirs = sorted(
            [d for d in self.cfg.opt_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if not opt_dirs:
            raise FileNotFoundError(f"No optimization results in {self.cfg.opt_dir}")
        
        # Try best_parameters.csv first, then fall back to iteration results
        best_file = opt_dirs[0] / "best_parameters.csv"
        if best_file.exists():
            logger.info(f"Loading best parameters from {best_file}")
            return pd.read_csv(best_file)
        
        # Fall back to iteration results
        result_files = list(opt_dirs[0].glob("*iteration_results.csv"))
        if not result_files:
            raise FileNotFoundError(f"No results found in {opt_dirs[0]}")
        
        results = pd.read_csv(result_files[0])
        metric = self.cfg.opt_metric
        best_idx = results[metric].idxmax()
        best_row = results.iloc[best_idx]
        
        # Extract parameter columns (exclude metric columns)
        metric_cols = {'KGE', 'NSE', 'RMSE', 'MAE', 'iteration', 'objective'}
        param_cols = [c for c in results.columns if c not in metric_cols]
        
        best_df = pd.DataFrame({
            'parameter': param_cols,
            'value': [best_row[c] for c in param_cols]
        })
        
        logger.info(f"Best {metric}: {best_row[metric]:.4f} at iteration {best_idx}")
        return best_df
    
    def load_existing_parameters(self, params_csv: str) -> pd.DataFrame:
        """Load parameters from a previously saved CSV."""
        return pd.read_csv(params_csv)


# =============================================================================
# Phase 2: Model Runner — Execute SUMMA with Given Parameters
# =============================================================================

class ModelRunner:
    """Handles SUMMA execution with specific parameters and time periods."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
    
    def apply_parameters(self, params_df: pd.DataFrame):
        """Write optimized parameters to SUMMA config files."""
        sys.path.insert(0, str(self.cfg.code_dir))
        from utils.custom.adjust_settings import update_and_reformat_parameter_file
        
        param_dict = dict(zip(params_df['parameter'], params_df['value']))
        
        local_params = {k: v for k, v in param_dict.items()
                       if not k.startswith('basin__') and not k.startswith('routing')}
        basin_params = {k: v for k, v in param_dict.items()
                       if k.startswith('basin__') or k.startswith('routing')}
        
        local_file = self.cfg.settings_dir / 'localParamInfo.txt'
        basin_file = self.cfg.settings_dir / 'basinParamInfo.txt'
        
        if local_params and local_file.exists():
            update_and_reformat_parameter_file(local_file, local_params, reformat_all=True)
            logger.info(f"Updated {len(local_params)} local parameters")
        
        if basin_params and basin_file.exists():
            update_and_reformat_parameter_file(basin_file, basin_params, reformat_all=True)
            logger.info(f"Updated {len(basin_params)} basin parameters")
    
    def update_time_period(self, start: str, end: str):
        """Update the simulation time period in SUMMA fileManager."""
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        if not fm_path.exists():
            raise FileNotFoundError(f"File manager not found: {fm_path}")
        
        with open(fm_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('simStartTime'):
                new_lines.append(f"simStartTime    '{start}'\n")
            elif line.strip().startswith('simEndTime'):
                new_lines.append(f"simEndTime      '{end}'\n")
            else:
                new_lines.append(line)
        
        with open(fm_path, 'w') as f:
            f.writelines(new_lines)
        
        logger.info(f"Updated simulation period: {start} to {end}")
    
    def update_forcing_file_list(self, forcing_files: List[str]):
        """Update the SUMMA forcing file list."""
        ffl_path = self.cfg.settings_dir / 'forcingFileList.txt'
        with open(ffl_path, 'w') as f:
            for fname in forcing_files:
                f.write(f"{fname}\n")
        logger.info(f"Updated forcing file list: {len(forcing_files)} files")
    
    def update_output_path(self, output_dir: str):
        """Update the output path in the SUMMA fileManager."""
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        with open(fm_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('outputPath'):
                new_lines.append(f"outputPath      '{output_dir}/'\n")
            else:
                new_lines.append(line)
        
        with open(fm_path, 'w') as f:
            f.writelines(new_lines)
    
    def update_experiment_prefix(self, prefix: str):
        """Update the output prefix in the SUMMA fileManager."""
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        with open(fm_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('outFilePrefix'):
                new_lines.append(f"outFilePrefix   '{prefix}'\n")
            else:
                new_lines.append(line)
        
        with open(fm_path, 'w') as f:
            f.writelines(new_lines)
    
    def run_summa(self, experiment_id: str, output_dir: Path) -> Path:
        """Execute SUMMA and return path to output file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_output_path(str(output_dir))
        self.update_experiment_prefix(experiment_id)
        
        summa_exe = self.cfg.raw.get('SUMMA_EXE', 'summa')
        summa_install = self.cfg.raw.get('SUMMA_INSTALL_PATH', 'default')
        if summa_install == 'default':
            summa_path = self.cfg.data_dir / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_install)
        
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        cmd = f"{summa_path / summa_exe} -m {fm_path}"
        
        logger.info(f"Running SUMMA: {experiment_id}")
        log_file = log_dir / f'{experiment_id}.log'
        
        with open(log_file, 'w') as lf:
            result = subprocess.run(
                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                timeout=10800  # 3-hour timeout
            )
        
        if result.returncode != 0:
            raise RuntimeError(f"SUMMA failed for {experiment_id}. See {log_file}")
        
        # Find the output file
        output_files = sorted(output_dir.glob(f"{experiment_id}*_timestep.nc"))
        if not output_files:
            output_files = sorted(output_dir.glob(f"{experiment_id}*.nc"))
        
        if not output_files:
            raise FileNotFoundError(f"No output files found in {output_dir}")
        
        logger.info(f"SUMMA complete: {output_files[-1].name}")
        return output_files[-1]


# =============================================================================
# Phase 3: Forcing Ensemble Builder — Seasonal Swap Logic
# =============================================================================

class ForcingEnsembleBuilder:
    """Builds ensemble forcing files by swapping seasons across years."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
    
    def load_forcing_data(self, forcing_dir: Optional[Path] = None) -> xr.Dataset:
        """Load all forcing files into a single dataset."""
        fdir = forcing_dir or self.cfg.summa_input_dir
        files = sorted(fdir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No forcing files in {fdir}")
        
        ds = xr.open_mfdataset(files, combine='by_coords')
        logger.info(f"Loaded forcing: {ds.time.values[0]} to {ds.time.values[-1]}, "
                    f"{len(ds.time)} timesteps")
        return ds
    
    def get_season_mask(self, times: xr.DataArray, season: str) -> np.ndarray:
        """Create a boolean mask for timesteps belonging to a season."""
        months = SEASONS[season]['months']
        return np.isin(pd.DatetimeIndex(times.values).month, months)
    
    def get_year_range(self, ds: xr.Dataset) -> Tuple[int, int]:
        """Return (first_year, last_year) of the dataset."""
        years = pd.DatetimeIndex(ds.time.values).year
        return int(years.min()), int(years.max())
    
    def build_ensemble_for_season(
        self,
        full_forcing: xr.Dataset,
        target_year: int,
        donor_years: List[int],
        season: str,
        output_dir: Path
    ) -> List[Path]:
        """
        Build ensemble forcing files by replacing one season of the target year.
        
        For each donor year, takes the seasonal forcing from that year and
        splices it into the target year's forcing, keeping all other seasons
        from the target year intact.
        
        Parameters
        ----------
        full_forcing : xr.Dataset
            The complete multi-year forcing dataset.
        target_year : int
            The year whose simulation we want to perturb (year 21).
        donor_years : list of int
            Years from which to draw replacement seasonal forcing (years 1-20).
        season : str
            Season code: 'DJF', 'MAM', 'JJA', or 'SON'.
        output_dir : Path
            Directory to write ensemble forcing files.
            
        Returns
        -------
        list of Path
            Paths to the created ensemble forcing files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        months = SEASONS[season]['months']
        created_files = []
        
        # Extract target year forcing as baseline
        target_start = f"{target_year}-01-01"
        target_end = f"{target_year}-12-31 23:00"
        target_ds = full_forcing.sel(time=slice(target_start, target_end))
        
        for donor_year in donor_years:
            logger.info(f"  Building ensemble: {season} from {donor_year} → {target_year}")
            
            # Start with a copy of the target year
            ensemble_ds = target_ds.copy(deep=True)
            
            # Get donor season data
            for month in months:
                # Handle December crossing year boundary for DJF
                if season == 'DJF' and month == 12:
                    donor_month_year = donor_year - 1
                else:
                    donor_month_year = donor_year
                
                # Select the donor month
                donor_data = full_forcing.sel(
                    time=(pd.DatetimeIndex(full_forcing.time.values).month == month) &
                         (pd.DatetimeIndex(full_forcing.time.values).year == donor_month_year)
                )
                
                # Select matching month in target
                target_month_mask = (
                    (pd.DatetimeIndex(ensemble_ds.time.values).month == month)
                )
                
                if len(donor_data.time) == 0:
                    logger.warning(f"    No data for {donor_month_year}-{month:02d}, skipping")
                    continue
                
                # Match timestep count — handle leap year differences
                target_times = ensemble_ds.time.values[target_month_mask]
                n_target = len(target_times)
                n_donor = len(donor_data.time)
                
                if n_donor == 0 or n_target == 0:
                    continue
                
                # If donor has more timesteps, truncate; if fewer, repeat last
                for var in FORCING_VARS:
                    if var not in ensemble_ds.data_vars:
                        continue
                    
                    donor_vals = donor_data[var].values
                    
                    if n_donor >= n_target:
                        replacement = donor_vals[:n_target]
                    else:
                        # Pad by repeating the last timestep
                        pad = np.repeat(donor_vals[-1:], n_target - n_donor, axis=0)
                        replacement = np.concatenate([donor_vals, pad], axis=0)
                    
                    # Apply replacement
                    ensemble_ds[var].values[target_month_mask] = replacement
            
            # Save ensemble member
            domain = self.cfg.domain_name
            fname = f"{domain}_{season}_{target_year}_from_{donor_year}.nc"
            out_path = output_dir / fname
            
            # Set encoding for time
            encoding = {'time': {'units': 'hours since 1900-01-01', 'calendar': 'gregorian'}}
            ensemble_ds.to_netcdf(out_path, encoding=encoding)
            created_files.append(out_path)
            logger.info(f"    Saved: {fname}")
        
        return created_files
    
    def build_all_season_ensembles(
        self,
        full_forcing: xr.Dataset,
        target_year: int,
        donor_years: List[int]
    ) -> Dict[str, List[Path]]:
        """Build ensembles for all four seasons."""
        all_files = {}
        for season in SEASONS:
            logger.info(f"\nBuilding {season} ({SEASONS[season]['name']}) ensemble...")
            out_dir = self.cfg.ensemble_dir / 'forcing' / season
            files = self.build_ensemble_for_season(
                full_forcing, target_year, donor_years, season, out_dir
            )
            all_files[season] = files
        return all_files


# =============================================================================
# Phase 4: Ensemble Runner — Execute All Ensemble Members
# =============================================================================

class EnsembleRunner:
    """Runs SUMMA for each ensemble member and collects results."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
        self.runner = ModelRunner(exp_config)
    
    def run_baseline(
        self,
        params_df: pd.DataFrame,
        start: str,
        end: str,
        experiment_id: str = 'baseline'
    ) -> Path:
        """Run the baseline (unperturbed) simulation."""
        logger.info("=" * 70)
        logger.info(f"RUNNING BASELINE: {start} to {end}")
        logger.info("=" * 70)
        
        self.runner.apply_parameters(params_df)
        self.runner.update_time_period(start, end)
        
        output_dir = self.cfg.project_dir / 'simulations' / experiment_id / 'SUMMA'
        return self.runner.run_summa(experiment_id, output_dir)
    
    def run_target_year_baseline(
        self,
        params_df: pd.DataFrame,
        target_year: int
    ) -> Path:
        """Run the unperturbed target year simulation."""
        start = f"{target_year}-01-01 01:00"
        end = f"{target_year}-12-31 23:00"
        exp_id = f"target_year_{target_year}"
        
        output_dir = self.cfg.ensemble_dir / 'results' / 'baseline'
        self.runner.apply_parameters(params_df)
        self.runner.update_time_period(start, end)
        return self.runner.run_summa(exp_id, output_dir)
    
    def run_season_ensemble(
        self,
        params_df: pd.DataFrame,
        season: str,
        target_year: int,
        ensemble_forcing_files: List[Path]
    ) -> Dict[int, Path]:
        """Run all ensemble members for one season."""
        logger.info("=" * 70)
        logger.info(f"RUNNING {season} ENSEMBLE ({len(ensemble_forcing_files)} members)")
        logger.info("=" * 70)
        
        results = {}
        self.runner.apply_parameters(params_df)
        
        start = f"{target_year}-01-01 01:00"
        end = f"{target_year}-12-31 23:00"
        self.runner.update_time_period(start, end)
        
        for forcing_file in ensemble_forcing_files:
            # Extract donor year from filename: {domain}_{season}_{target}_from_{donor}.nc
            parts = forcing_file.stem.split('_from_')
            donor_year = int(parts[-1])
            
            exp_id = f"{self.cfg.domain_name}_{season}_from{donor_year}"
            output_dir = self.cfg.ensemble_dir / 'results' / season / f'from_{donor_year}'
            
            # Point SUMMA at this specific forcing file
            self.runner.update_forcing_file_list([forcing_file.name])
            
            # Update forcing path to the ensemble forcing directory
            fm_path = self.cfg.settings_dir / 'fileManager.txt'
            with open(fm_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if line.strip().startswith('forcingPath'):
                    new_lines.append(f"forcingPath     '{forcing_file.parent}/'\n")
                else:
                    new_lines.append(line)
            with open(fm_path, 'w') as f:
                f.writelines(new_lines)
            
            try:
                output_file = self.runner.run_summa(exp_id, output_dir)
                results[donor_year] = output_file
            except (RuntimeError, FileNotFoundError) as e:
                logger.error(f"  Failed for donor year {donor_year}: {e}")
                continue
        
        logger.info(f"  Completed {len(results)}/{len(ensemble_forcing_files)} members")
        return results
    
    def run_all_ensembles(
        self,
        params_df: pd.DataFrame,
        target_year: int,
        ensemble_files: Dict[str, List[Path]]
    ) -> Dict[str, Dict[int, Path]]:
        """Run ensembles for all seasons."""
        all_results = {}
        for season, files in ensemble_files.items():
            all_results[season] = self.run_season_ensemble(
                params_df, season, target_year, files
            )
        return all_results


# =============================================================================
# Phase 4b: Parallel Ensemble Runner — Background & Concurrent Execution
# =============================================================================

class ParallelEnsembleRunner:
    """
    Runs SUMMA ensemble members in parallel with isolated workspaces.
    
    Each member gets its own fileManager.txt and settings directory (symlinked)
    so that multiple SUMMA processes can run concurrently without interference.
    
    Supports two execution modes:
    1. Python-managed parallelism via subprocess.Popen (launch_all)
    2. Shell script generation for nohup/overnight runs (generate_run_script)
    """
    
    def __init__(self, exp_config: ExperimentConfig, max_workers: int = 4):
        self.cfg = exp_config
        self.max_workers = max_workers
        self.run_base = self.cfg.ensemble_dir / 'runs'
        self.log_dir = self.cfg.ensemble_dir / 'logs'
        self.results_base = self.cfg.ensemble_dir / 'results'
        
        # Resolve SUMMA executable
        summa_exe_name = self.cfg.raw.get('SUMMA_EXE', 'summa')
        summa_install = self.cfg.raw.get('SUMMA_INSTALL_PATH', 'default')
        if summa_install == 'default':
            self.summa_exe = str(
                self.cfg.data_dir / 'installs' / 'summa' / 'bin' / summa_exe_name
            )
        else:
            self.summa_exe = str(Path(summa_install) / summa_exe_name)
        
        # Track members and processes
        self.members = {}     # member_id -> config dict
        self.active = {}      # member_id -> {'proc': Popen, 'log_fh': file handle}
        self.completed = {}   # member_id -> {'returncode': int, 'output_file': Path}
        self.failed = {}      # member_id -> {'returncode': int, 'error': str}
    
    @staticmethod
    def _member_id(season: str, donor_year: int) -> str:
        return f"{season}_from{donor_year}"
    
    def _output_prefix(self, season: str, donor_year: int) -> str:
        return f"{self.cfg.domain_name}_{season}_from{donor_year}"
    
    # ------------------------------------------------------------------
    # Workspace preparation
    # ------------------------------------------------------------------
    
    def prepare_member(
        self, season: str, donor_year: int, target_year: int, forcing_file: Path
    ) -> str:
        """
        Create an isolated workspace for one ensemble member.
        
        Creates a per-member settings directory that symlinks shared files
        from the main SUMMA settings and overrides only forcingFileList.txt.
        A dedicated fileManager.txt is written that points to the member's
        own settings, forcing, and output directories.
        """
        mid = self._member_id(season, donor_year)
        
        run_dir = self.run_base / mid
        settings_dir = run_dir / 'settings'
        output_dir = self.results_base / season / f'from_{donor_year}'
        
        for d in [run_dir, settings_dir, output_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Symlink shared settings files into the member settings dir
        for item in self.cfg.settings_dir.iterdir():
            link = settings_dir / item.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(item.resolve())
        
        # Override forcingFileList.txt with member-specific content
        ffl = settings_dir / 'forcingFileList.txt'
        if ffl.is_symlink():
            ffl.unlink()
        ffl.write_text(forcing_file.name + '\n')
        
        # Override fileManager.txt symlink with a member-specific copy
        fm_link = settings_dir / 'fileManager.txt'
        if fm_link.is_symlink():
            fm_link.unlink()
        
        # Write the per-member fileManager.txt in the run directory
        prefix = self._output_prefix(season, donor_year)
        fm_path = run_dir / 'fileManager.txt'
        self._write_file_manager(
            fm_path,
            settings_path=str(settings_dir),
            forcing_path=str(forcing_file.parent),
            output_path=str(output_dir),
            prefix=prefix,
            start=f"{target_year}-01-01 01:00",
            end=f"{target_year}-12-31 23:00",
        )
        
        self.members[mid] = {
            'season': season,
            'donor_year': donor_year,
            'run_dir': run_dir,
            'output_dir': output_dir,
            'prefix': prefix,
            'fm_path': fm_path,
            'log_file': self.log_dir / f'{mid}.log',
            'forcing_file': forcing_file,
        }
        return mid
    
    def _write_file_manager(
        self, path, settings_path, forcing_path, output_path, prefix, start, end
    ):
        """Write a complete fileManager.txt from the template in the main settings."""
        template = self.cfg.settings_dir / 'fileManager.txt'
        if not template.exists():
            raise FileNotFoundError(f"Template fileManager.txt not found: {template}")
        
        with open(template, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith('simStartTime'):
                new_lines.append(f"simStartTime         '{start}'\n")
            elif s.startswith('simEndTime'):
                new_lines.append(f"simEndTime           '{end}'\n")
            elif s.startswith('outFilePrefix'):
                new_lines.append(f"outFilePrefix        '{prefix}'\n")
            elif s.startswith('settingsPath'):
                new_lines.append(f"settingsPath         '{settings_path}/'\n")
            elif s.startswith('forcingPath'):
                new_lines.append(f"forcingPath          '{forcing_path}/'\n")
            elif s.startswith('outputPath'):
                new_lines.append(f"outputPath           '{output_path}/'\n")
            else:
                new_lines.append(line)
        
        with open(path, 'w') as f:
            f.writelines(new_lines)
    
    def prepare_all(
        self,
        target_year: int,
        ensemble_forcing_files: Dict[str, List[Path]],
    ) -> Dict[str, dict]:
        """Prepare workspaces for every ensemble member across all seasons."""
        total = sum(len(ff) for ff in ensemble_forcing_files.values())
        logger.info(f"Preparing {total} ensemble member workspaces...")
        
        for season, forcing_files in ensemble_forcing_files.items():
            for ff in forcing_files:
                parts = ff.stem.split('_from_')
                donor_year = int(parts[-1])
                self.prepare_member(season, donor_year, target_year, ff)
        
        logger.info(f"  {len(self.members)} workspaces ready in {self.run_base}")
        return self.members
    
    # ------------------------------------------------------------------
    # Python-managed parallel launch
    # ------------------------------------------------------------------
    
    def launch_all(self, poll_interval: float = 30.0):
        """
        Launch all prepared members with bounded concurrency.
        
        Blocks until every member finishes. Use poll_interval (seconds)
        to control how often progress is logged.
        """
        if not self.members:
            raise ValueError("No members prepared — call prepare_all() first.")
        
        pending = list(self.members.keys())
        total = len(pending)
        logger.info(f"Launching {total} SUMMA runs (max {self.max_workers} parallel)...")
        
        while pending or self.active:
            # Fill available worker slots
            while pending and len(self.active) < self.max_workers:
                mid = pending.pop(0)
                self._launch_one(mid)
            
            _time.sleep(poll_interval)
            self._poll_active()
            
            n_done = len(self.completed) + len(self.failed)
            logger.info(
                f"  Progress: {n_done}/{total} done, "
                f"{len(self.active)} running, {len(pending)} queued"
            )
        
        logger.info("=" * 60)
        logger.info(
            f"ALL RUNS COMPLETE — {len(self.completed)} succeeded, "
            f"{len(self.failed)} failed"
        )
        for mid in self.failed:
            logger.error(f"  FAILED: {mid} (rc={self.failed[mid]['returncode']})")
        logger.info("=" * 60)
    
    def _launch_one(self, member_id: str):
        info = self.members[member_id]
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
        })
        cmd = f"{self.summa_exe} -m {info['fm_path']}"
        lf = open(info['log_file'], 'w')
        proc = subprocess.Popen(
            cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env
        )
        self.active[member_id] = {'proc': proc, 'log_fh': lf}
        logger.info(f"  LAUNCHED: {member_id}  (PID {proc.pid})")
    
    def _poll_active(self):
        finished = []
        for mid, ctx in self.active.items():
            rc = ctx['proc'].poll()
            if rc is None:
                continue
            finished.append(mid)
            ctx['log_fh'].close()
            
            if rc == 0:
                odir = self.members[mid]['output_dir']
                prefix = self.members[mid]['prefix']
                out_files = sorted(odir.glob(f"{prefix}*_timestep.nc"))
                if not out_files:
                    out_files = sorted(odir.glob(f"{prefix}*.nc"))
                self.completed[mid] = {
                    'returncode': rc,
                    'output_file': out_files[-1] if out_files else None,
                }
                logger.info(f"  DONE: {mid}")
            else:
                error_tail = ""
                log_file = self.members[mid]['log_file']
                if log_file.exists():
                    with open(log_file) as f:
                        error_tail = ''.join(f.readlines()[-5:])
                self.failed[mid] = {'returncode': rc, 'error': error_tail}
                logger.warning(f"  FAILED: {mid} (rc={rc})")
        
        for mid in finished:
            del self.active[mid]
    
    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------
    
    def get_results(self) -> Dict[str, Dict[int, Path]]:
        """Return results organised as {season: {donor_year: output_path}}."""
        results = {}
        for mid, info in self.completed.items():
            m = self.members[mid]
            season, donor_year = m['season'], m['donor_year']
            results.setdefault(season, {})
            if info['output_file'] is not None:
                results[season][donor_year] = info['output_file']
        return results
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'total': len(self.members),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'active': len(self.active),
            'pending': (
                len(self.members) - len(self.completed)
                - len(self.failed) - len(self.active)
            ),
            'failed_members': list(self.failed.keys()),
        }
    
    # ------------------------------------------------------------------
    # Shell script for nohup / overnight runs
    # ------------------------------------------------------------------
    
    def generate_run_script(self, script_path: Optional[Path] = None) -> Path:
        """
        Write a self-contained bash script that runs every prepared member.
        
        The script manages concurrency itself (MAX_PARALLEL), so it can be
        launched with:
            nohup bash run_ensemble.sh > logs/ensemble_main.log 2>&1 &
        """
        if not self.members:
            raise ValueError("No members prepared — call prepare_all() first.")
        
        script = script_path or (self.cfg.ensemble_dir / 'run_ensemble.sh')
        total = len(self.members)
        
        header = '\n'.join([
            '#!/bin/bash',
            '# Seasonal Forcing Ensemble — auto-generated run script',
            f'# Generated: {datetime.now().isoformat()}',
            f'# Members:   {total}',
            f'# Max parallel: {self.max_workers}',
            '',
            f'MAX_PARALLEL={self.max_workers}',
            f'SUMMA_EXE="{self.summa_exe}"',
            '',
            'export OMP_NUM_THREADS=1',
            'export MKL_NUM_THREADS=1',
            'export OPENBLAS_NUM_THREADS=1',
            '',
            'declare -a PIDS',
            'declare -a NAMES',
            'COMPLETED=0',
            'FAILED=0',
            f'TOTAL={total}',
            '',
            '# ---- concurrency helper ----',
            'wait_for_slot() {',
            '    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do',
            '        for i in "${!PIDS[@]}"; do',
            '            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then',
            '                wait "${PIDS[$i]}"',
            '                RC=$?',
            '                if [ $RC -eq 0 ]; then',
            '                    COMPLETED=$((COMPLETED + 1))',
            '                    echo "[$(date +%H:%M:%S)] DONE: ${NAMES[$i]}  ($COMPLETED/$TOTAL)"',
            '                else',
            '                    FAILED=$((FAILED + 1))',
            '                    echo "[$(date +%H:%M:%S)] FAIL: ${NAMES[$i]}  (rc=$RC)"',
            '                fi',
            '                unset "PIDS[$i]"',
            '                unset "NAMES[$i]"',
            '                PIDS=("${PIDS[@]}")',
            '                NAMES=("${NAMES[@]}")',
            '                return',
            '            fi',
            '        done',
            '        sleep 5',
            '    done',
            '}',
            '',
            'echo "========================================"',
            f'echo "Ensemble run — {total} members"',
            'echo "========================================"',
            'START_TIME=$(date +%s)',
            '',
        ])
        
        member_lines = []
        for mid, info in self.members.items():
            fm = info['fm_path']
            log = info['log_file']
            member_lines.append(f'wait_for_slot')
            member_lines.append(f'echo "[$(date +%H:%M:%S)] LAUNCH: {mid}"')
            member_lines.append(f'$SUMMA_EXE -m "{fm}" > "{log}" 2>&1 &')
            member_lines.append(f'PIDS+=($!)')
            member_lines.append(f'NAMES+=("{mid}")')
            member_lines.append('')
        
        footer = '\n'.join([
            '# ---- wait for stragglers ----',
            'for i in "${!PIDS[@]}"; do',
            '    wait "${PIDS[$i]}"',
            '    RC=$?',
            '    if [ $RC -eq 0 ]; then',
            '        COMPLETED=$((COMPLETED + 1))',
            '        echo "[$(date +%H:%M:%S)] DONE: ${NAMES[$i]}  ($COMPLETED/$TOTAL)"',
            '    else',
            '        FAILED=$((FAILED + 1))',
            '        echo "[$(date +%H:%M:%S)] FAIL: ${NAMES[$i]}"',
            '    fi',
            'done',
            '',
            'END_TIME=$(date +%s)',
            'ELAPSED=$((END_TIME - START_TIME))',
            'HOURS=$((ELAPSED / 3600))',
            'MINS=$(((ELAPSED % 3600) / 60))',
            '',
            'echo "========================================"',
            'echo "ENSEMBLE COMPLETE"',
            'echo "  Succeeded: $COMPLETED / $TOTAL"',
            'echo "  Failed:    $FAILED"',
            'echo "  Elapsed:   ${HOURS}h ${MINS}m"',
            'echo "========================================"',
        ])
        
        with open(script, 'w') as f:
            f.write(header + '\n')
            f.write('\n'.join(member_lines) + '\n')
            f.write(footer + '\n')
        
        script.chmod(0o755)
        logger.info(f"Run script: {script}")
        logger.info(
            f"  Overnight usage:  nohup bash {script} "
            f"> {self.log_dir}/ensemble_main.log 2>&1 &"
        )
        return script
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    
    def cleanup_run_dirs(self):
        """Remove temporary per-member run directories (keeps results & logs)."""
        if self.run_base.exists():
            shutil.rmtree(self.run_base)
            logger.info(f"Cleaned up: {self.run_base}")


# =============================================================================
# Phase 5: Evaluation — Water Balance & Sensitivity Metrics
# =============================================================================

class EnsembleEvaluator:
    """Evaluates ensemble results for water balance and sensitivity."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
    
    def extract_water_balance(self, output_file: Path, basin_area_m2: float = None) -> pd.DataFrame:
        """
        Extract water balance components from a SUMMA output file.
        
        Returns a DataFrame with daily values of:
        - precipitation (P)
        - total ET (evaporation + transpiration + sublimation)
        - runoff (surface + base)
        - storage change (delta SWE + delta soil water)
        - runoff ratio (Q/P)
        """
        ds = xr.open_dataset(output_file)
        
        area = basin_area_m2 or self.cfg.basin_area_m2 or 1.0
        
        # Precipitation (input forcing, already in the output as pptrate)
        ppt = ds['pptrate'].squeeze() if 'pptrate' in ds else None
        
        # ET components (m/s)
        et_total = ds.get('scalarTotalET', 0)
        
        # Runoff (m/s)
        runoff = ds.get('scalarTotalRunoff', ds.get('averageRoutedRunoff', None))

        # Baseflow (m/s)
        aquifer_baseflow = ds.get('scalarAquiferBaseflow', None)
        soil_baseflow = ds.get('scalarSoilBaseflow', None)
        total_baseflow = aquifer_baseflow + soil_baseflow if aquifer_baseflow is not None and soil_baseflow is not None else None 

        # Storage (m) — absolute values at each timestep
        # Calculate monthly change: SM_end - SM_start for each month
        soil_liq = ds.get('scalarTotalSoilLiq', 0)
        soil_ice = ds.get('scalarTotalSoilIce', 0)
        soil_moisture = soil_liq + soil_ice
        
        # Build DataFrame with daily resampling
        time = pd.DatetimeIndex(ds.time.values)
        
        df = pd.DataFrame(index=time)
        if ppt is not None:
            # Squeeze any hru dimension
            ppt_vals = ppt.values.squeeze() if hasattr(ppt, 'values') else ppt
            df['P'] = ppt_vals
        
        if hasattr(et_total, 'values'):
            df['ET'] = et_total.values.squeeze()
        
        if runoff is not None:
            df['Q'] = runoff.values.squeeze()
            # Convert to m³/s if area is provided
            if area > 1.0:
                df['Q_cms'] = df['Q'] * area
        
        if total_baseflow is not None:
            df['baseflow'] = total_baseflow.values.squeeze()
            if area > 1.0:
                df['baseflow_cms'] = df['baseflow'] * area

        if hasattr(soil_moisture, 'values'):
            df['storage'] = soil_moisture.values.squeeze()
            df['soil_water'] = soil_moisture.values.squeeze() if hasattr(soil_moisture, 'values') else 0
        
        
        ds.close()
        
        # Daily aggregation
        daily = df.resample('D').mean()
        
        # Storage change (daily delta)
        if 'storage' in daily.columns:
            daily['delta_storage'] = daily['storage'].diff()
        
        # Runoff ratio
        if 'Q' in daily.columns and 'P' in daily.columns:
            daily['runoff_ratio'] = np.where(
                daily['P'] > 1e-10, daily['Q'] / daily['P'], np.nan
            )
        
        return daily
    
    def evaluate_ensemble(
        self,
        baseline_file: Path,
        ensemble_files: Dict[int, Path],
        season: str
    ) -> pd.DataFrame:
        """
        Evaluate all ensemble members against the baseline.
        
        Returns a DataFrame with one row per ensemble member containing:
        - seasonal total P, ET, Q
        - seasonal mean storage change
        - seasonal runoff ratio
        - seasonal KGE of streamflow vs baseline
        """
        months = SEASONS[season]['months']
        
        baseline_wb = self.extract_water_balance(baseline_file)
        baseline_seasonal = baseline_wb[baseline_wb.index.month.isin(months)]
        
        records = []
        for donor_year, output_file in ensemble_files.items():
            try:
                member_wb = self.extract_water_balance(output_file)
                member_seasonal = member_wb[member_wb.index.month.isin(months)]
                
                record = {
                    'donor_year': donor_year,
                    'season': season,
                    'P_total': member_seasonal['P'].sum() if 'P' in member_seasonal else np.nan,
                    'ET_total': member_seasonal['ET'].sum() if 'ET' in member_seasonal else np.nan,
                    'Q_total': member_seasonal['Q'].sum() if 'Q' in member_seasonal else np.nan,
                    'delta_storage_mean': member_seasonal['delta_storage'].mean() if 'delta_storage' in member_seasonal else np.nan,
                    'runoff_ratio': (
                        member_seasonal['Q'].sum() / member_seasonal['P'].sum()
                        if 'Q' in member_seasonal and 'P' in member_seasonal 
                        and member_seasonal['P'].sum() > 1e-10
                        else np.nan
                    ),
                    'swe_mean': member_seasonal['swe'].mean() if 'swe' in member_seasonal else np.nan,
                }
                
                # KGE of daily streamflow vs baseline
                if 'Q' in member_seasonal and 'Q' in baseline_seasonal:
                    common_idx = member_seasonal.index.intersection(baseline_seasonal.index)
                    if len(common_idx) > 10:
                        obs = baseline_seasonal.loc[common_idx, 'Q'].values
                        sim = member_seasonal.loc[common_idx, 'Q'].values
                        record['KGE_vs_baseline'] = get_KGE(obs, sim)
                
                # Compute deviation from baseline
                bl = {
                    'P_total': baseline_seasonal['P'].sum() if 'P' in baseline_seasonal else np.nan,
                    'ET_total': baseline_seasonal['ET'].sum() if 'ET' in baseline_seasonal else np.nan,
                    'Q_total': baseline_seasonal['Q'].sum() if 'Q' in baseline_seasonal else np.nan,
                    'runoff_ratio': (
                        baseline_seasonal['Q'].sum() / baseline_seasonal['P'].sum()
                        if 'Q' in baseline_seasonal and 'P' in baseline_seasonal
                        and baseline_seasonal['P'].sum() > 1e-10
                        else np.nan
                    ),
                }
                record['P_anomaly_pct'] = _pct_change(bl['P_total'], record['P_total'])
                record['ET_anomaly_pct'] = _pct_change(bl['ET_total'], record['ET_total'])
                record['Q_anomaly_pct'] = _pct_change(bl['Q_total'], record['Q_total'])
                record['RR_anomaly_pct'] = _pct_change(bl['runoff_ratio'], record['runoff_ratio'])
                
                records.append(record)
                
            except Exception as e:
                logger.warning(f"  Failed to evaluate donor year {donor_year}: {e}")
                continue
        
        return pd.DataFrame(records)
    
    def evaluate_all_seasons(
        self,
        baseline_file: Path,
        all_ensemble_files: Dict[str, Dict[int, Path]]
    ) -> pd.DataFrame:
        """Evaluate all seasons and combine results."""
        dfs = []
        for season, files in all_ensemble_files.items():
            logger.info(f"Evaluating {season} ensemble ({len(files)} members)...")
            df = self.evaluate_ensemble(baseline_file, files, season)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Save results
        self.cfg.ensemble_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cfg.ensemble_dir / 'ensemble_evaluation_results.csv'
        combined.to_csv(out_path, index=False)
        logger.info(f"Evaluation results saved: {out_path}")
        
        return combined
    
    def compute_sensitivity_summary(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sensitivity metrics per season.
        
        For each season, reports the spread (std, range) of water balance 
        components and runoff ratio across ensemble members, as an indicator
        of how sensitive the system is to that season's forcing.
        """
        summary_records = []
        for season in SEASONS:
            season_df = eval_df[eval_df['season'] == season]
            if season_df.empty:
                continue
            
            record = {
                'season': season,
                'season_name': SEASONS[season]['name'],
                'n_members': len(season_df),
                
                # Spread in each variable
                'P_std': season_df['P_total'].std(),
                'P_range': season_df['P_total'].max() - season_df['P_total'].min(),
                
                'ET_std': season_df['ET_total'].std(),
                'ET_range': season_df['ET_total'].max() - season_df['ET_total'].min(),
                
                'Q_std': season_df['Q_total'].std(),
                'Q_range': season_df['Q_total'].max() - season_df['Q_total'].min(),
                
                'RR_std': season_df['runoff_ratio'].std(),
                'RR_range': season_df['runoff_ratio'].max() - season_df['runoff_ratio'].min(),
                
                'delta_S_std': season_df['delta_storage_mean'].std(),
                'delta_S_range': season_df['delta_storage_mean'].max() - season_df['delta_storage_mean'].min(),
                
                # Mean anomalies
                'mean_P_anomaly_pct': season_df['P_anomaly_pct'].mean(),
                'mean_ET_anomaly_pct': season_df['ET_anomaly_pct'].mean(),
                'mean_Q_anomaly_pct': season_df['Q_anomaly_pct'].mean(),
                'mean_RR_anomaly_pct': season_df['RR_anomaly_pct'].mean(),
                
                # Elasticity: %ΔQ / %ΔP (how sensitive is runoff to precip change)
                'elasticity_Q_to_P': _safe_elasticity(
                    season_df['Q_anomaly_pct'], season_df['P_anomaly_pct']
                ),
            }
            summary_records.append(record)
        
        summary = pd.DataFrame(summary_records)
        out_path = self.cfg.ensemble_dir / 'sensitivity_summary.csv'
        summary.to_csv(out_path, index=False)
        logger.info(f"Sensitivity summary saved: {out_path}")
        return summary


def _pct_change(baseline, member):
    """Percent change from baseline to member."""
    if baseline is None or np.isnan(baseline) or abs(baseline) < 1e-15:
        return np.nan
    return ((member - baseline) / abs(baseline)) * 100.0


def _safe_elasticity(q_pct: pd.Series, p_pct: pd.Series) -> float:
    """Linear regression slope of %ΔQ on %ΔP."""
    valid = ~(q_pct.isna() | p_pct.isna())
    if valid.sum() < 3:
        return np.nan
    x = p_pct[valid].values
    y = q_pct[valid].values
    if np.std(x) < 1e-10:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return slope


# =============================================================================
# Phase 6: Visualization — Spaghetti Plots & Sensitivity Panels
# =============================================================================

class EnsembleVisualizer:
    """Creates spaghetti plots and sensitivity visualizations."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.cfg = exp_config
        self.cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_spaghetti_precip_streamflow(
        self,
        baseline_file: Path,
        ensemble_files: Dict[int, Path],
        season: str,
        target_year: int,
        obs_df: Optional[pd.DataFrame] = None,
        figsize: Tuple = (16, 10)
    ) -> Path:
        """
        Create a two-panel spaghetti plot showing:
        - Top: Precipitation ensemble
        - Bottom: Streamflow ensemble
        
        Each ensemble member is a thin colored line, baseline is thick black.
        """
        fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        season_color = SEASONS[season]['color']
        season_name = SEASONS[season]['name']
        months = SEASONS[season]['months']
        
        # Load baseline
        bl_ds = xr.open_dataset(baseline_file)
        bl_time = pd.DatetimeIndex(bl_ds.time.values)
        bl_ppt = bl_ds['pptrate'].values.squeeze() if 'pptrate' in bl_ds else None
        bl_q = bl_ds['averageRoutedRunoff'].values.squeeze() if 'averageRoutedRunoff' in bl_ds else bl_ds['scalarTotalRunoff'].values.squeeze()
        
        # Convert streamflow units if basin area available
        area = self.cfg.basin_area_m2 or 1.0
        if area > 1.0:
            bl_q = bl_q * area
        
        # Daily resample for clarity
        bl_df = pd.DataFrame({'P': bl_ppt, 'Q': bl_q}, index=bl_time).resample('D').mean()
        bl_ds.close()
        
        # Season highlight region
        season_mask = bl_df.index.month.isin(months)
        
        # Plot ensemble members
        alpha_member = max(0.15, 0.8 / max(len(ensemble_files), 1))
        
        for donor_year, efile in sorted(ensemble_files.items()):
            try:
                m_ds = xr.open_dataset(efile)
                m_time = pd.DatetimeIndex(m_ds.time.values)
                m_ppt = m_ds['pptrate'].values.squeeze() if 'pptrate' in m_ds else None
                m_q = m_ds['averageRoutedRunoff'].values.squeeze() if 'averageRoutedRunoff' in m_ds else m_ds['scalarTotalRunoff'].values.squeeze()
                if area > 1.0:
                    m_q = m_q * area
                
                m_df = pd.DataFrame({'P': m_ppt, 'Q': m_q}, index=m_time).resample('D').mean()
                m_ds.close()
                
                # Plot only the perturbed season portion differently
                ax_p.plot(m_df.index, m_df['P'] * 3600, color=season_color,
                         alpha=alpha_member, linewidth=0.8, zorder=2)
                ax_q.plot(m_df.index, m_df['Q'], color=season_color,
                         alpha=alpha_member, linewidth=0.8, zorder=2)
            except Exception as e:
                logger.warning(f"  Could not plot donor year {donor_year}: {e}")
        
        # Plot baseline on top
        ax_p.plot(bl_df.index, bl_df['P'] * 3600, color='black',
                 linewidth=2.0, label=f'Baseline ({target_year})', zorder=5)
        ax_q.plot(bl_df.index, bl_df['Q'], color='black',
                 linewidth=2.0, label=f'Baseline ({target_year})', zorder=5)
        
        # Plot observations if available
        if obs_df is not None and 'discharge_cms' in obs_df.columns:
            obs_period = obs_df.loc[str(target_year)]
            if not obs_period.empty:
                ax_q.plot(obs_period.index, obs_period['discharge_cms'],
                         color='red', linewidth=1.5, linestyle='--',
                         label='Observed', zorder=6)
        
        # Shade the perturbed season
        for ax in [ax_p, ax_q]:
            for start_month in months:
                m_start = pd.Timestamp(f'{target_year}-{start_month:02d}-01')
                if start_month == 12:
                    m_end = pd.Timestamp(f'{target_year}-12-31')
                else:
                    next_month = start_month + 1
                    m_end = pd.Timestamp(f'{target_year}-{next_month:02d}-01') - timedelta(days=1)
                ax.axvspan(m_start, m_end, alpha=0.08, color=season_color, zorder=0)
        
        # Format axes
        ax_p.set_ylabel('Precipitation (mm/hr)')
        ax_p.set_title(
            f'Seasonal Forcing Ensemble: {season_name} ({season})\n'
            f'Target Year {target_year} with forcing replaced from 20 prior years',
            fontsize=14, fontweight='bold'
        )
        ax_p.legend(loc='upper right')
        ax_p.grid(True, alpha=0.3)
        ax_p.invert_yaxis()  # Precipitation convention: bars from top
        
        unit = 'm³/s' if area > 1.0 else 'm/s'
        ax_q.set_ylabel(f'Streamflow ({unit})')
        ax_q.set_xlabel('Date')
        ax_q.legend(loc='upper right')
        ax_q.grid(True, alpha=0.3)
        ax_q.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_q.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax_q.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add ensemble member legend entry
        ensemble_line = Line2D([0], [0], color=season_color, linewidth=1.5, alpha=0.6)
        baseline_line = Line2D([0], [0], color='black', linewidth=2.0)
        custom_legend = [baseline_line, ensemble_line]
        labels = [f'Baseline {target_year}', f'Ensemble ({len(ensemble_files)} members)']
        if obs_df is not None:
            obs_line = Line2D([0], [0], color='red', linewidth=1.5, linestyle='--')
            custom_legend.append(obs_line)
            labels.append('Observed')
        ax_q.legend(custom_legend, labels, loc='upper right')
        
        plt.tight_layout()
        
        out_path = self.cfg.plots_dir / f'spaghetti_{season}_{target_year}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved spaghetti plot: {out_path}")
        return out_path
    
    def plot_all_seasons_spaghetti(
        self,
        baseline_file: Path,
        all_ensemble_results: Dict[str, Dict[int, Path]],
        target_year: int,
        obs_df: Optional[pd.DataFrame] = None
    ) -> List[Path]:
        """Generate spaghetti plots for all seasons."""
        paths = []
        for season, files in all_ensemble_results.items():
            p = self.plot_spaghetti_precip_streamflow(
                baseline_file, files, season, target_year, obs_df
            )
            paths.append(p)
        return paths
    
    def plot_sensitivity_dashboard(
        self,
        eval_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        figsize: Tuple = (18, 14)
    ) -> Path:
        """
        Create a multi-panel sensitivity dashboard showing:
        1. Bar chart: spread in Q, ET, storage by season
        2. Scatter: P anomaly vs Q anomaly (elasticity) 
        3. Box plots: runoff ratio distribution by season
        4. Bar chart: mean anomaly percentages by season
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        season_colors = [SEASONS[s]['color'] for s in SEASONS]
        season_labels = [f"{s}\n({SEASONS[s]['name']})" for s in SEASONS]
        
        # --- Panel 1: Spread (std) in Q, ET, delta_S by season ---
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(SEASONS))
        width = 0.25
        
        if not summary_df.empty:
            ax1.bar(x - width, summary_df['Q_std'].values, width, label='Runoff (Q)',
                   color='#2166ac', alpha=0.8)
            ax1.bar(x, summary_df['ET_std'].values, width, label='ET',
                   color='#4dac26', alpha=0.8)
            ax1.bar(x + width, summary_df['delta_S_std'].values, width, label='ΔStorage',
                   color='#e66101', alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(season_labels)
        ax1.set_ylabel('Standard Deviation (m/s)')
        ax1.set_title('Ensemble Spread by Season', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # --- Panel 2: P anomaly vs Q anomaly (elasticity) ---
        ax2 = fig.add_subplot(gs[0, 1])
        for season in SEASONS:
            sdf = eval_df[eval_df['season'] == season]
            if not sdf.empty:
                ax2.scatter(sdf['P_anomaly_pct'], sdf['Q_anomaly_pct'],
                          color=SEASONS[season]['color'], label=season,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        # Add 1:1 line
        lims = ax2.get_xlim()
        ax2.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
        ax2.set_xlabel('Precipitation Anomaly (%)')
        ax2.set_ylabel('Runoff Anomaly (%)')
        ax2.set_title('Precipitation–Runoff Elasticity', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # --- Panel 3: Runoff ratio box plots ---
        ax3 = fig.add_subplot(gs[1, 0])
        season_data = []
        for season in SEASONS:
            sdf = eval_df[eval_df['season'] == season]['runoff_ratio'].dropna()
            season_data.append(sdf.values)
        
        bp = ax3.boxplot(season_data, labels=season_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], season_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax3.set_ylabel('Runoff Ratio (Q/P)')
        ax3.set_title('Runoff Ratio Distribution by Season', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # --- Panel 4: Mean anomaly bar chart ---
        ax4 = fig.add_subplot(gs[1, 1])
        if not summary_df.empty:
            x = np.arange(len(SEASONS))
            width = 0.2
            ax4.bar(x - 1.5*width, summary_df['mean_P_anomaly_pct'].values, width,
                   label='P', color='#2166ac', alpha=0.8)
            ax4.bar(x - 0.5*width, summary_df['mean_ET_anomaly_pct'].values, width,
                   label='ET', color='#4dac26', alpha=0.8)
            ax4.bar(x + 0.5*width, summary_df['mean_Q_anomaly_pct'].values, width,
                   label='Q', color='#d01c8b', alpha=0.8)
            ax4.bar(x + 1.5*width, summary_df['mean_RR_anomaly_pct'].values, width,
                   label='RR', color='#e66101', alpha=0.8)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(season_labels)
        
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.set_ylabel('Mean Anomaly (%)')
        ax4.set_title('Mean Water Balance Anomalies', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Seasonal Forcing Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.01)
        
        out_path = self.cfg.plots_dir / 'sensitivity_dashboard.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved sensitivity dashboard: {out_path}")
        return out_path
    
    def plot_combined_spaghetti(
        self,
        baseline_file: Path,
        all_ensemble_results: Dict[str, Dict[int, Path]],
        target_year: int,
        obs_df: Optional[pd.DataFrame] = None,
        figsize: Tuple = (18, 20)
    ) -> Path:
        """
        Generate a combined 4×2 figure with all seasons.
        Each row is one season with precip (left) and streamflow (right).
        """
        fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
        area = self.cfg.basin_area_m2 or 1.0
        
        # Load baseline once
        bl_ds = xr.open_dataset(baseline_file)
        bl_time = pd.DatetimeIndex(bl_ds.time.values)
        bl_ppt = bl_ds['pptrate'].values.squeeze() if 'pptrate' in bl_ds else None
        bl_q_var = 'averageRoutedRunoff' if 'averageRoutedRunoff' in bl_ds else 'scalarTotalRunoff'
        bl_q = bl_ds[bl_q_var].values.squeeze()
        if area > 1.0:
            bl_q = bl_q * area
        bl_df = pd.DataFrame({'P': bl_ppt, 'Q': bl_q}, index=bl_time).resample('D').mean()
        bl_ds.close()
        
        for row, season in enumerate(SEASONS):
            ax_p = axes[row, 0]
            ax_q = axes[row, 1]
            color = SEASONS[season]['color']
            months = SEASONS[season]['months']
            ensemble_files = all_ensemble_results.get(season, {})
            
            alpha_m = max(0.15, 0.8 / max(len(ensemble_files), 1))
            
            # Plot ensemble members
            for donor_year, efile in sorted(ensemble_files.items()):
                try:
                    m_ds = xr.open_dataset(efile)
                    m_time = pd.DatetimeIndex(m_ds.time.values)
                    m_ppt = m_ds['pptrate'].values.squeeze() if 'pptrate' in m_ds else None
                    m_q = m_ds[bl_q_var].values.squeeze() if bl_q_var in m_ds else m_ds['scalarTotalRunoff'].values.squeeze()
                    if area > 1.0:
                        m_q = m_q * area
                    m_df = pd.DataFrame({'P': m_ppt, 'Q': m_q}, index=m_time).resample('D').mean()
                    m_ds.close()
                    
                    ax_p.plot(m_df.index, m_df['P'] * 3600, color=color,
                             alpha=alpha_m, linewidth=0.6)
                    ax_q.plot(m_df.index, m_df['Q'], color=color,
                             alpha=alpha_m, linewidth=0.6)
                except Exception:
                    pass
            
            # Plot baseline
            ax_p.plot(bl_df.index, bl_df['P'] * 3600, color='black', linewidth=1.5)
            ax_q.plot(bl_df.index, bl_df['Q'], color='black', linewidth=1.5)
            
            # Observations
            if obs_df is not None and 'discharge_cms' in obs_df.columns:
                obs_period = obs_df.loc[str(target_year)]
                if not obs_period.empty:
                    ax_q.plot(obs_period.index, obs_period['discharge_cms'],
                             color='red', linewidth=1.0, linestyle='--', alpha=0.8)
            
            # Shade season months
            for m in months:
                m_start = pd.Timestamp(f'{target_year}-{m:02d}-01')
                if m == 12:
                    m_end = pd.Timestamp(f'{target_year}-12-31')
                else:
                    nm = m + 1
                    m_end = pd.Timestamp(f'{target_year}-{nm:02d}-01') - timedelta(days=1)
                ax_p.axvspan(m_start, m_end, alpha=0.08, color=color)
                ax_q.axvspan(m_start, m_end, alpha=0.08, color=color)
            
            ax_p.set_ylabel(f'{season}\nmm/hr')
            ax_p.invert_yaxis()
            ax_p.grid(True, alpha=0.2)
            
            unit = 'm³/s' if area > 1.0 else 'm/s'
            ax_q.set_ylabel(f'{season}\n{unit}')
            ax_q.grid(True, alpha=0.2)
            
            if row == 0:
                ax_p.set_title('Precipitation', fontweight='bold')
                ax_q.set_title('Streamflow', fontweight='bold')
        
        # Format x-axis
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[-1, 1].xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.setp(axes[-1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(axes[-1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Shared legend
        ensemble_line = Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.6)
        baseline_line = Line2D([0], [0], color='black', linewidth=2.0)
        handles = [baseline_line, ensemble_line]
        labels = [f'Baseline {target_year}', 'Ensemble members']
        if obs_df is not None:
            obs_line = Line2D([0], [0], color='red', linewidth=1.5, linestyle='--')
            handles.append(obs_line)
            labels.append('Observed')
        fig.legend(handles, labels, loc='upper center', ncol=3,
                  bbox_to_anchor=(0.5, 1.02), fontsize=12)
        
        fig.suptitle(
            f'Seasonal Forcing Ensemble — Target Year {target_year}',
            fontsize=16, fontweight='bold', y=1.05
        )
        
        plt.tight_layout()
        out_path = self.cfg.plots_dir / f'combined_spaghetti_{target_year}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved combined spaghetti: {out_path}")
        return out_path


# =============================================================================
# Main Orchestrator
# =============================================================================

class SeasonalEnsembleExperiment:
    """
    Full workflow orchestrator for the seasonal forcing ensemble experiment.
    
    Steps:
    1. Optimize parameters (or load existing)
    2. Run 20-year baseline simulation
    3. Run unperturbed target year (year 21)
    4. Build seasonal forcing ensembles
    5. Run all ensemble members (parallel or via shell script)
    6. Evaluate water balance sensitivity
    7. Generate visualizations
    
    Parallel execution modes:
    - parallel=True in step5 → Python-managed subprocess pool, blocks until done
    - generate_run_script() → bash script for nohup / overnight runs
    """
    
    def __init__(self, config_path: str, max_workers: int = 4):
        self.cfg = ExperimentConfig(config_path)
        self.max_workers = max_workers
        self.optimizer = ParameterOptimizer(self.cfg)
        self.ensemble_builder = ForcingEnsembleBuilder(self.cfg)
        self.ensemble_runner = EnsembleRunner(self.cfg)
        self.parallel_runner = None  # created lazily in step5
        self.evaluator = EnsembleEvaluator(self.cfg)
        self.visualizer = EnsembleVisualizer(self.cfg)
        
        # State
        self.best_params = None
        self.baseline_output = None
        self.target_baseline_output = None
        self.ensemble_forcing_files = None
        self.ensemble_results = None
        self.eval_results = None
        self.sensitivity_summary = None
    
    # ------------------------------------------------------------------
    # Step 0: Data preparation & coverage checks
    # ------------------------------------------------------------------
    
    def check_data_coverage(
        self,
        baseline_start_year: int,
        target_year: int,
    ) -> Dict[str, Any]:
        """
        Check that forcing and observation data cover the experiment period.
        
        Returns a dict with coverage info and any gaps found.
        """
        logger.info("\n" + "=" * 70)
        logger.info("DATA COVERAGE CHECK")
        logger.info("=" * 70)
        
        report: Dict[str, Any] = {
            'forcing_ok': False,
            'obs_ok': False,
            'forcing_years': [],
            'forcing_gaps': [],
            'obs_start': None,
            'obs_end': None,
            'obs_gap_years': [],
        }
        
        # --- Forcing coverage ---
        forcing_files = sorted(self.cfg.summa_input_dir.glob('*.nc'))
        if not forcing_files:
            logger.error(f"No forcing files in {self.cfg.summa_input_dir}")
            return report
        
        # Extract year-month from filenames
        forcing_ym = set()
        for f in forcing_files:
            parts = f.stem.split('_')
            yyyymm = parts[-1]
            forcing_ym.add(yyyymm)
        
        # Check each needed year has 12 months
        needed_years = list(range(baseline_start_year, target_year + 1))
        for yr in needed_years:
            year_months = [f"{yr}{m:02d}" for m in range(1, 13)]
            missing = [ym for ym in year_months if ym not in forcing_ym]
            if missing:
                report['forcing_gaps'].extend(missing)
                logger.warning(f"  Forcing gap: {yr} missing {len(missing)} months")
            else:
                report['forcing_years'].append(yr)
        
        report['forcing_ok'] = len(report['forcing_gaps']) == 0
        logger.info(
            f"Forcing: {len(report['forcing_years'])}/{len(needed_years)} years complete"
            + (" — ALL GOOD" if report['forcing_ok'] else f" — {len(report['forcing_gaps'])} months missing")
        )
        
        # --- Observation coverage ---
        obs_files = list(self.cfg.obs_dir.glob('*streamflow*.csv'))
        if obs_files:
            obs_df = pd.read_csv(obs_files[0], parse_dates=['datetime'])
            report['obs_start'] = obs_df['datetime'].min()
            report['obs_end'] = obs_df['datetime'].max()
            obs_start_year = report['obs_start'].year
            
            # Check which experiment years lack obs
            for yr in needed_years:
                if yr < obs_start_year:
                    report['obs_gap_years'].append(yr)
            
            report['obs_ok'] = len(report['obs_gap_years']) == 0
            logger.info(
                f"Observations: {report['obs_start'].date()} to {report['obs_end'].date()}"
            )
            if report['obs_gap_years']:
                logger.warning(
                    f"  Obs missing for years: {report['obs_gap_years']}"
                    f"\n  (Obs not strictly required — only used for evaluation plots)"
                )
            else:
                logger.info("  Obs covers full experiment period")
        else:
            logger.warning(f"No observation files in {self.cfg.obs_dir}")
        
        return report
    
    def step0_prepare_data(
        self,
        start: str = None,
        end: str = None,
        download_obs: bool = True,
        create_forcing: bool = True,
        recalc_longwave: bool = True,
    ):
        """
        Run the complete data preparation pipeline.

        This ensures forcing files (merged, basin-averaged, and SUMMA-ready)
        and observations exist for the requested period.  Uses the
        ForcingProcessor for ERA5 merge → basin-average → SUMMA input,
        then updates the SUMMA forcingFileList.txt.

        Parameters
        ----------
        start : str, optional
            Override EXPERIMENT_TIME_START in config (e.g., '1999-01-01 00:00').
        end : str, optional
            Override EXPERIMENT_TIME_END (e.g., '2019-12-31 23:00').
        download_obs : bool
            Download / refresh streamflow observations.
        create_forcing : bool
            Run forcing merge, basin-averaging, and SUMMA input creation.
        recalc_longwave : bool
            Recalculate longwave radiation via Dilley & O'Brien (1998).
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 0: DATA PREPARATION")
        logger.info("=" * 70)

        # Build config overrides for the extended period
        cfg_overrides = {}
        if start:
            cfg_overrides['EXPERIMENT_TIME_START'] = start
        if end:
            cfg_overrides['EXPERIMENT_TIME_END'] = end

        cfg_dict = self.cfg.get_confluence_config(**cfg_overrides)

        confluence = CONFLUENCE(config_path=str(self.cfg.config_path))
        # Apply overrides directly to the CONFLUENCE instance config
        for k, v in cfg_overrides.items():
            confluence.config[k] = v

        if download_obs:
            logger.info("Downloading / refreshing observations...")
            confluence.managers['data'].process_observed_data()
            logger.info("Observations ready.")

        if create_forcing:
            # --- Use ForcingProcessor for the 3-step forcing pipeline ---
            fp = ForcingProcessor(cfg_dict)
            status = fp.check_status(verbose=True)

            # 1) Merge raw ERA5 surface + pressure → merged files
            if status['missing_merged']:
                logger.info(f"Merging {len(status['missing_merged'])} raw ERA5 files...")
                fp.merge_era5()
            else:
                logger.info("All merged ERA5 files already exist.")

            # 2) Basin-average merged files via CONFLUENCE / EASYMORE
            if status['missing_basin_avg']:
                logger.info(
                    f"Basin-averaging {len(status['missing_basin_avg'])} merged files "
                    f"(existing files will be skipped)..."
                )
                confluence.managers['data'].run_model_agnostic_preprocessing()
            else:
                logger.info("All basin-averaged files already exist.")

            # 3) Create SUMMA input files from basin-averaged data
            # Re-check after basin averaging since new files may now exist
            fp.check_status(verbose=False)
            if fp.missing_summa:
                logger.info(
                    f"Creating {len(fp.missing_summa)} SUMMA input files..."
                )
                fp.create_summa_input(recalc_longwave=recalc_longwave)
            else:
                logger.info("All SUMMA input files already exist.")

            # 4) Update forcingFileList.txt so SUMMA sees all available files
            self._update_summa_forcing_file_list()

            logger.info("Forcing data ready.")

    def _update_summa_forcing_file_list(self):
        """Rewrite forcingFileList.txt to list all files in SUMMA_input/."""
        summa_input_dir = self.cfg.summa_input_dir
        ffl_path = self.cfg.settings_dir / 'forcingFileList.txt'

        files = sorted(f.name for f in summa_input_dir.glob('*.nc'))
        if not files:
            logger.warning(f"No .nc files found in {summa_input_dir}")
            return

        with open(ffl_path, 'w') as fh:
            for fname in files:
                fh.write(f"{fname}\n")

        logger.info(
            f"Updated {ffl_path.name}: {len(files)} files "
            f"({files[0]} … {files[-1]})"
        )
    
    def step1_optimize(self, skip_if_exists: bool = True) -> pd.DataFrame:
        """Run or load optimization."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: PARAMETER OPTIMIZATION")
        logger.info("=" * 70)
        
        if skip_if_exists and self.cfg.opt_dir.exists():
            existing = list(self.cfg.opt_dir.glob("*/best_parameters.csv"))
            if existing:
                logger.info(f"Found existing optimization results, loading...")
                self.best_params = self.optimizer.get_best_parameters()
                return self.best_params
        
        self.best_params = self.optimizer.run_optimization()
        return self.best_params
    
    def step2_long_term_run(
        self,
        start: str,
        end: str,
        experiment_id: str = 'longterm_baseline'
    ) -> Path:
        """Run the 20-year baseline simulation."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: LONG-TERM (20-YEAR) BASELINE RUN")
        logger.info("=" * 70)
        
        if self.best_params is None:
            raise ValueError("Run step1_optimize first")
        
        # Ensure state variables are in outputControl so we can extract warm state
        self._add_state_vars_to_output_control()
        
        self.baseline_output = self.ensemble_runner.run_baseline(
            self.best_params, start, end, experiment_id
        )
        return self.baseline_output
    
    def step2b_create_warm_state(self, summa_output: Path = None) -> Path:
        """
        Extract final state from the long-term run and create a warm-start file.
        
        Reads the last timestep of the SUMMA output, extracts all state
        variables, and writes a new warmState.nc that replaces the default
        cold state (which starts with zeros).
        
        Parameters
        ----------
        summa_output : Path, optional
            Path to SUMMA output NC file. Defaults to self.baseline_output.
        
        Returns
        -------
        Path to the newly created warm state file.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2b: CREATE WARM STATE FROM BASELINE")
        logger.info("=" * 70)
        
        output_nc = summa_output or self.baseline_output
        if output_nc is None:
            raise ValueError("No baseline output available. Run step2_long_term_run first.")
        
        warm_state_path = self.cfg.settings_dir / 'warmState.nc'
        cold_state_path = self.cfg.settings_dir / 'coldState.nc'
        
        ds_cold = xr.open_dataset(cold_state_path)
        ds_out = xr.open_dataset(output_nc)
        
        # Last timestep
        last_time = pd.Timestamp(ds_out.time.values[-1])
        ds_last = ds_out.isel(time=-1)
        n_soil = int(ds_cold['nSoil'].values.flat[0])
        
        logger.info(f"Extracting state from: {output_nc.name}")
        logger.info(f"Last timestep: {last_time}")
        logger.info(f"Soil layers: {n_soil}")
        
        def get_scalar(var_name, default=0.0):
            """Get scalar value from output, trying multiple suffixes."""
            for suffix in ['', '_inst', '_sum', '_mean']:
                name = var_name + suffix
                if name in ds_last:
                    val = float(ds_last[name].values.flat[0])
                    logger.info(f"  {var_name:30s} = {val:12.4f}  (from {name})")
                    return val
            logger.warning(f"  {var_name:30s} = {default:12.4f}  (DEFAULT — not in output)")
            return default
        
        def get_soil_layers(var_name, n_soil, default_val):
            """Extract soil-layer values (last n_soil of midToto dim)."""
            for suffix in ['', '_inst', '_sum', '_mean']:
                name = var_name + suffix
                if name in ds_last:
                    vals = ds_last[name].values.squeeze()
                    # Soil layers are the last n_soil entries of the midToto dim
                    soil_vals = vals[-n_soil:]
                    logger.info(f"  {var_name:30s} = {soil_vals}  (from {name})")
                    return soil_vals
            logger.warning(f"  {var_name:30s} = not found, keeping defaults")
            return None
        
        # Clone the cold state
        ds_warm = ds_cold.copy(deep=True)
        
        # --- Scalar state variables ---
        ds_warm['scalarCanopyIce'].values[:] = get_scalar('scalarCanopyIce')
        ds_warm['scalarCanopyLiq'].values[:] = get_scalar('scalarCanopyLiq')
        ds_warm['scalarSnowDepth'].values[:] = get_scalar('scalarSnowDepth')
        ds_warm['scalarSWE'].values[:] = get_scalar('scalarSWE')
        ds_warm['scalarSfcMeltPond'].values[:] = get_scalar('scalarSfcMeltPond')
        ds_warm['scalarAquiferStorage'].values[:] = get_scalar(
            'scalarAquiferStorage',
            default=float(ds_cold['scalarAquiferStorage'].values.flat[0]),
        )
        ds_warm['scalarSnowAlbedo'].values[:] = get_scalar('scalarSnowAlbedo')
        ds_warm['scalarCanairTemp'].values[:] = get_scalar('scalarCanairTemp', default=283.16)
        ds_warm['scalarCanopyTemp'].values[:] = get_scalar('scalarCanopyTemp', default=283.16)
        
        # --- Layer state variables (extract soil layers only) ---
        layer_vars = {
            'mLayerTemp': 283.16,
            'mLayerVolFracIce': 0.0,
            'mLayerVolFracLiq': 0.2,
            'mLayerMatricHead': -1.0,
        }
        for var, default in layer_vars.items():
            soil_vals = get_soil_layers(var, n_soil, default)
            if soil_vals is not None:
                ds_warm[var].values[:] = soil_vals.reshape(ds_warm[var].shape)
        
        # Keep nSnow=0 — SUMMA will create snow layers dynamically from SWE/depth
        ds_warm['nSnow'].values[:] = 0
        
        # Update metadata
        ds_warm.attrs['Author'] = 'Created by seasonal_ensemble_experiment.py'
        ds_warm.attrs['History'] = (
            f'Warm state extracted from {output_nc.name}, '
            f'last timestep {last_time}'
        )
        ds_warm.attrs['Purpose'] = (
            'Warm start initial conditions from long-term baseline run'
        )
        
        ds_warm.to_netcdf(warm_state_path)
        ds_cold.close()
        ds_out.close()
        
        # Update fileManager.txt to use the warm state
        self._update_init_condition_file('warmState.nc')
        
        logger.info(f"Warm state written to {warm_state_path}")
        logger.info("fileManager.txt updated to use warmState.nc")
        return warm_state_path
    
    def _add_state_vars_to_output_control(self):
        """Ensure outputControl.txt includes state variables needed for warm start."""
        oc_path = self.cfg.settings_dir / 'outputControl.txt'
        with open(oc_path, 'r') as f:
            content = f.read()
        
        to_add = [v for v in WARM_STATE_EXTRA_VARS if v not in content]
        if not to_add:
            logger.info("All state variables already in outputControl.txt")
            return
        
        with open(oc_path, 'a') as f:
            f.write('\n! State variables for warm start extraction\n')
            for var in to_add:
                f.write(f'{var:40s} | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n')
        
        logger.info(f"Added {len(to_add)} state variables to outputControl.txt")
    
    def _update_init_condition_file(self, filename: str):
        """Update initConditionFile in fileManager.txt."""
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        with open(fm_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip().startswith('initConditionFile'):
                new_lines.append(f"initConditionFile    '{filename}'\n")
            else:
                new_lines.append(line)
        
        with open(fm_path, 'w') as f:
            f.writelines(new_lines)
        logger.info(f"initConditionFile → {filename}")
    
    def step3_target_year_baseline(self, target_year: int) -> Path:
        """Run the unperturbed target year."""
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 3: TARGET YEAR BASELINE ({target_year})")
        logger.info("=" * 70)
        
        self.target_baseline_output = self.ensemble_runner.run_target_year_baseline(
            self.best_params, target_year
        )
        return self.target_baseline_output
    
    def step4_build_ensembles(
        self,
        target_year: int,
        donor_years: List[int],
        forcing_dir: Optional[Path] = None
    ) -> Dict[str, List[Path]]:
        """Build seasonal forcing ensemble files."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: BUILD SEASONAL FORCING ENSEMBLES")
        logger.info("=" * 70)
        
        full_forcing = self.ensemble_builder.load_forcing_data(forcing_dir)
        self.ensemble_forcing_files = self.ensemble_builder.build_all_season_ensembles(
            full_forcing, target_year, donor_years
        )
        full_forcing.close()
        return self.ensemble_forcing_files
    
    def step5_run_ensembles(
        self,
        target_year: int,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        poll_interval: float = 30.0,
    ) -> Dict[str, Dict[int, Path]]:
        """
        Run all seasonal ensemble members.
        
        Parameters
        ----------
        target_year : int
            The year being simulated.
        parallel : bool
            If True, use ParallelEnsembleRunner (concurrent subprocesses).
            If False, use the sequential EnsembleRunner.
        max_workers : int, optional
            Override the default max_workers for parallel mode.
        poll_interval : float
            Seconds between progress log messages (parallel mode only).
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: RUN ALL ENSEMBLE MEMBERS")
        logger.info("=" * 70)
        
        if self.ensemble_forcing_files is None:
            raise ValueError("Run step4_build_ensembles first")
        
        if parallel:
            workers = max_workers or self.max_workers
            self.parallel_runner = ParallelEnsembleRunner(self.cfg, max_workers=workers)
            self.parallel_runner.prepare_all(target_year, self.ensemble_forcing_files)
            self.parallel_runner.launch_all(poll_interval=poll_interval)
            self.ensemble_results = self.parallel_runner.get_results()
        else:
            self.ensemble_results = self.ensemble_runner.run_all_ensembles(
                self.best_params, target_year, self.ensemble_forcing_files
            )
        
        return self.ensemble_results
    
    def step5_generate_script(
        self,
        target_year: int,
        max_workers: Optional[int] = None,
        script_path: Optional[Path] = None,
    ) -> Path:
        """
        Prepare workspaces and generate a bash script for overnight execution.
        
        Does NOT run anything — produces a script that can be launched with:
            nohup bash run_ensemble.sh > logs/ensemble_main.log 2>&1 &
        
        Returns the path to the generated script.
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5 (SCRIPT MODE): PREPARE WORKSPACES & GENERATE RUN SCRIPT")
        logger.info("=" * 70)
        
        if self.ensemble_forcing_files is None:
            raise ValueError("Run step4_build_ensembles first")
        
        workers = max_workers or self.max_workers
        self.parallel_runner = ParallelEnsembleRunner(self.cfg, max_workers=workers)
        self.parallel_runner.prepare_all(target_year, self.ensemble_forcing_files)
        return self.parallel_runner.generate_run_script(script_path)
    
    def step5_collect_results(self) -> Dict[str, Dict[int, Path]]:
        """
        After an overnight script run, scan the results directory and collect
        output file paths for evaluation.
        """
        logger.info("Scanning for completed ensemble results...")
        results = {}
        results_base = self.cfg.ensemble_dir / 'results'
        
        for season in SEASONS:
            season_dir = results_base / season
            if not season_dir.exists():
                continue
            results[season] = {}
            for donor_dir in sorted(season_dir.iterdir()):
                if not donor_dir.is_dir() or not donor_dir.name.startswith('from_'):
                    continue
                donor_year = int(donor_dir.name.replace('from_', ''))
                nc_files = sorted(donor_dir.glob('*_timestep.nc'))
                if not nc_files:
                    nc_files = sorted(donor_dir.glob('*.nc'))
                if nc_files:
                    results[season][donor_year] = nc_files[-1]
            
            if results[season]:
                logger.info(f"  {season}: {len(results[season])} completed members")
        
        self.ensemble_results = results
        return results
    
    def step6_evaluate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate ensemble results."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: SENSITIVITY EVALUATION")
        logger.info("=" * 70)
        
        if self.target_baseline_output is None or self.ensemble_results is None:
            raise ValueError("Run steps 3 and 5 first")
        
        self.eval_results = self.evaluator.evaluate_all_seasons(
            self.target_baseline_output, self.ensemble_results
        )
        self.sensitivity_summary = self.evaluator.compute_sensitivity_summary(self.eval_results)
        
        return self.eval_results, self.sensitivity_summary
    
    def step7_visualize(
        self,
        target_year: int,
        obs_csv: Optional[str] = None
    ) -> List[Path]:
        """Generate all visualizations."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: VISUALIZATION")
        logger.info("=" * 70)
        
        # Load observations if available
        obs_df = None
        if obs_csv:
            obs_df = pd.read_csv(obs_csv, parse_dates=['datetime']).set_index('datetime')
        else:
            obs_files = list(self.cfg.obs_dir.glob(f"*streamflow*.csv"))
            if obs_files:
                obs_df = pd.read_csv(obs_files[0], parse_dates=['datetime']).set_index('datetime')
        
        plots = []
        
        # Individual season spaghetti plots
        plots.extend(self.visualizer.plot_all_seasons_spaghetti(
            self.target_baseline_output, self.ensemble_results,
            target_year, obs_df
        ))
        
        # Combined 4-season spaghetti
        plots.append(self.visualizer.plot_combined_spaghetti(
            self.target_baseline_output, self.ensemble_results,
            target_year, obs_df
        ))
        
        # Sensitivity dashboard
        if self.eval_results is not None and self.sensitivity_summary is not None:
            plots.append(self.visualizer.plot_sensitivity_dashboard(
                self.eval_results, self.sensitivity_summary
            ))
        
        return plots
    
    def run_full_workflow(
        self,
        baseline_start: str,
        baseline_end: str,
        target_year: int,
        donor_years: Optional[List[int]] = None,
        skip_optimization: bool = True,
        forcing_dir: Optional[Path] = None,
        obs_csv: Optional[str] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Execute the complete seasonal ensemble experiment.
        
        Parameters
        ----------
        baseline_start : str
            Start of the 20-year baseline period (e.g., '2003-01-01 01:00')
        baseline_end : str  
            End of the 20-year baseline period (e.g., '2022-12-31 23:00')
        target_year : int
            The 21st year to analyze (e.g., 2023)
        donor_years : list of int, optional
            Years to use as forcing donors. Defaults to all years in baseline.
        skip_optimization : bool
            If True, load existing optimization results.
        forcing_dir : Path, optional
            Directory containing forcing files.
        obs_csv : str, optional
            Path to observations CSV.
        parallel : bool
            Use parallel SUMMA execution (default True).
        max_workers : int, optional
            Number of concurrent SUMMA processes.
        """
        if donor_years is None:
            start_yr = int(baseline_start[:4])
            end_yr = int(baseline_end[:4])
            donor_years = list(range(start_yr, end_yr + 1))
        
        self.step1_optimize(skip_if_exists=skip_optimization)
        self.step2_long_term_run(baseline_start, baseline_end)
        self.step2b_create_warm_state()
        self.step3_target_year_baseline(target_year)
        self.step4_build_ensembles(target_year, donor_years, forcing_dir)
        self.step5_run_ensembles(target_year, parallel=parallel, max_workers=max_workers)
        self.step6_evaluate()
        self.step7_visualize(target_year, obs_csv)
        
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results: {self.cfg.ensemble_dir}")
        logger.info(f"Plots:   {self.cfg.plots_dir}")
        
        return {
            'best_params': self.best_params,
            'eval_results': self.eval_results,
            'sensitivity_summary': self.sensitivity_summary,
        }
