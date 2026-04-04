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

Each run creates a dated experiment folder (YYYYMMDD_experiment_name/) under simulations/
containing all outputs, settings, and configuration for reproducibility.

Usage:
    from seasonal_ensemble_experiment import SeasonalEnsembleExperiment
    
    exp = SeasonalEnsembleExperiment(
        config_path="path/to/config.yaml",
        experiment_name="test_run"  # optional; defaults to config EXPERIMENT_ID
    )
    exp.run_full_workflow(
        baseline_start="2003-01-01 01:00",
        baseline_end="2022-12-31 23:00",
        target_year=2023
    )
    
Utilities:
    backup_experiment_results(project_dir, domain_name, backup_name=None)
        - Moves non-dated result directories to a timestamped backup
"""

import sys
import os
import logging
import shutil
import subprocess
import time as _time
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import netCDF4 as nc
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


# =============================================================================
# Utility Functions
# =============================================================================

def backup_experiment_results(project_dir: Path, domain_name: str, backup_name: str = None) -> Path:
    """
    Move existing (non-dated) results to a timestamped backup.
    
    Useful before starting a new workflow to preserve old runs that don't yet
    follow the YYYYMMDD_name naming convention.
    
    Parameters
    ----------
    project_dir : Path
        Base project directory (data_dir/domain_xyz)
    domain_name : str
        Domain name (e.g., 'East_River_lumped')
    backup_name : str, optional
        Name for backup folder. If None, uses 'backup_pre_dated_<YYYYMMDD>'.
    
    Returns
    -------
    Path
        Path to backup folder, or None if no results to backup.
    """
    sim_dir = project_dir / 'simulations'
    if not sim_dir.exists():
        return None
    
    # Find folders that don't follow YYYYMMDD_name pattern
    old_results = []
    for item in sim_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        # Skip if already dated (starts with YYYYMMDD_)
        if len(name) >= 9 and name[:8].isdigit() and name[8] == '_':
            continue
        old_results.append(item)
    
    if not old_results:
        logger.info("No old (non-dated) results to backup.")
        return None
    
    # Create backup folder
    if backup_name is None:
        now = datetime.now()
        backup_name = f"backup_pre_dated_{now.strftime('%Y%m%d_%H%M%S')}"
    
    backup_dir = sim_dir / backup_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Backing up {len(old_results)} old result folders to: {backup_name}")
    for item in old_results:
        dst = backup_dir / item.name
        if dst.exists():
            logger.warning(f"  Destination exists, skipping: {item.name}")
            continue
        shutil.move(str(item), str(dst))
        logger.info(f"  Moved: {item.name}")
    
    return backup_dir


def split_optimized_parameter_groups(
    params_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split optimization output into local and basin/routing parameter groups."""
    # Linear-reservoir routing params are valid in post-processing workflows,
    # but are not SUMMA local/basin parameters and must not be mapped to
    # localParamInfo/basinParamInfo during model optimization seeding.
    unsupported_summa_seed_params = {'k_fast', 'k_slow', 'f_fast'}

    local_params: Dict[str, Any] = {}
    basin_params: Dict[str, Any] = {}

    for raw_name, value in zip(params_df['parameter'], params_df['value']):
        name = str(raw_name).strip()

        if name in unsupported_summa_seed_params:
            continue

        if name.startswith('basin__'):
            basin_params[name.split('basin__', 1)[1]] = value
        elif name.startswith('routing__'):
            basin_params[name.split('routing__', 1)[1]] = value
        elif name.startswith('routing'):
            basin_params[name] = value
        else:
            local_params[name] = value

    return local_params, basin_params


# Season definitions aligned to the water year (Oct–Sep).
# Each season spans exactly 3 months within a single calendar year,
# eliminating any cross-year month-donor complications.
#
#   OND  Fall:   Oct, Nov, Dec  (calendar year = water_year - 1)
#   JFM  Winter: Jan, Feb, Mar  (calendar year = water_year)
#   AMJ  Spring: Apr, May, Jun  (calendar year = water_year)
#   JAS  Summer: Jul, Aug, Sep  (calendar year = water_year)
SEASONS = {
    'OND': {'name': 'Fall',   'months': [10, 11, 12], 'color': '#e66101'},
    'JFM': {'name': 'Winter', 'months': [1,  2,  3],  'color': '#2166ac'},
    'AMJ': {'name': 'Spring', 'months': [4,  5,  6],  'color': '#4dac26'},
    'JAS': {'name': 'Summer', 'months': [7,  8,  9],  'color': '#d01c8b'},
}

FORCING_VARS = ['pptrate', 'SWRadAtm', 'LWRadAtm', 'airpres', 'airtemp', 'windspd', 'spechum']

# State variables that must appear in SUMMA output for warm-start extraction
# (variables NOT already in the default outputControl.txt)
WARM_STATE_EXTRA_VARS = [
    'scalarCanopyIce',
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
        
        # Shared paths (not experiment-specific)
        self.forcing_dir = self.project_dir / 'forcing'
        self.summa_input_dir = self.forcing_dir / 'SUMMA_input'
        self.obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        self.opt_dir = self.project_dir / 'optimisation'
        
        # Base settings dir (will be copied to experiment folder)
        self.base_settings_dir = self.project_dir / 'settings' / 'SUMMA'
        
        # Experiment-specific paths (initialized by initialize_experiment_workspace)
        self.settings_dir = None
        self.ensemble_dir = None
        self.plots_dir = None
        self.experiment_workspace = None
    
    def initialize_experiment_workspace(self, experiment_name: str = None) -> Path:
        """
        Initialize a dated experiment workspace.
        
        Creates a folder: YYYYMMDD_experiment_name/
        with subdirectories for settings, results, and plots.
        
        Copies config file and SUMMA settings to the workspace.
        
        Parameters
        ----------
        experiment_name : str, optional
            Name to include in folder name. Defaults to base_experiment_id.
        
        Returns
        -------
        Path
            Root path of the experiment workspace
        """
        if experiment_name is None:
            experiment_name = self.base_experiment_id
        
        # Create dated folder name: YYYYMMDD_experiment_name
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        exp_folder_name = f"{date_str}_{experiment_name}"
        
        self.experiment_workspace = self.project_dir / 'simulations' / exp_folder_name
        self.experiment_workspace.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized experiment workspace: {self.experiment_workspace.name}")
        
        # Create subdirectories
        self.settings_dir = self.experiment_workspace / 'settings'
        ensemble_base = self.experiment_workspace / 'ensemble'
        self.ensemble_dir = ensemble_base
        self.plots_dir = ensemble_base / 'plots'
        results_dir = ensemble_base / 'results'
        
        for d in [self.settings_dir, self.ensemble_dir, self.plots_dir, results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Copy config file to experiment folder for reproducibility
        config_backup = self.experiment_workspace / f"config_{now.strftime('%Y%m%d_%H%M%S')}.yaml"
        shutil.copy2(self.config_path, config_backup)
        logger.info(f"Backed up config: {config_backup.name}")
        
        # Copy base SUMMA settings to experiment folder
        if self.base_settings_dir.exists():
            for item in self.base_settings_dir.iterdir():
                dst = self.settings_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dst)
                elif item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
            logger.info(f"Copied SUMMA settings to: {self.settings_dir}")
        else:
            logger.warning(f"Base settings dir not found: {self.base_settings_dir}")
        
        # Update fileManager.txt paths to point to experiment workspace
        self._update_file_manager_paths()
        self._apply_summa_decisions_from_config()
        
        # Create README.md to document the experiment
        readme_path = self.experiment_workspace / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(f"# Experiment: {exp_folder_name}\n\n")
            f.write(f"**Date:** {now.isoformat()}\n\n")
            f.write(f"**Config:** {self.config_path.name}\n\n")
            f.write(f"**Domain:** {self.domain_name}\n\n")
            f.write("## Directory Structure\n\n")
            f.write("- `settings/` - SUMMA configuration and parameters\n")
            f.write("- `ensemble/results/` - Model simulation outputs\n")
            f.write("- `ensemble/plots/` - Visualization outputs\n")
            f.write("- `config_*.yaml` - Configuration snapshot\n")
        
        return self.experiment_workspace

    def use_existing_experiment_workspace(self, experiment_ref: str) -> Path:
        """
        Attach to an existing experiment workspace for analysis/restart runs.

        Parameters
        ----------
        experiment_ref : str
            Either an absolute path to an experiment folder, or a folder name
            under project_dir/simulations (e.g., '20260316_bigBuckt').

        Returns
        -------
        Path
            Resolved experiment workspace path.
        """
        candidate = Path(experiment_ref).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_dir / 'simulations' / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Existing experiment workspace not found: {candidate}")

        settings_dir = candidate / 'settings'
        ensemble_dir = candidate / 'ensemble'
        plots_dir = ensemble_dir / 'plots'
        results_dir = ensemble_dir / 'results'

        if not settings_dir.exists():
            raise FileNotFoundError(f"Missing settings directory in existing workspace: {settings_dir}")
        if not ensemble_dir.exists():
            raise FileNotFoundError(f"Missing ensemble directory in existing workspace: {ensemble_dir}")

        plots_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_workspace = candidate
        self.settings_dir = settings_dir
        self.ensemble_dir = ensemble_dir
        self.plots_dir = plots_dir

        # Ensure workspace decisions match the active YAML when reusing an experiment.
        self._apply_summa_decisions_from_config()

        logger.info(f"Using existing experiment workspace: {self.experiment_workspace}")
        return self.experiment_workspace

    def _apply_summa_decisions_from_config(self):
        """Apply SUMMA_DECISION_OPTIONS from config to workspace modelDecisions.txt."""
        decisions_cfg = self.raw.get('SUMMA_DECISION_OPTIONS', {})
        if not decisions_cfg:
            return

        model_decisions = self.settings_dir / 'modelDecisions.txt'
        if not model_decisions.exists():
            logger.warning(f"modelDecisions.txt not found in workspace settings: {model_decisions}")
            return

        updates = {}
        for key, value in decisions_cfg.items():
            if isinstance(value, list):
                if value:
                    updates[key] = str(value[0])
            elif value is not None:
                updates[key] = str(value)

        if not updates:
            return

        from utils.custom.adjust_settings import edit_modelDecisions
        edit_modelDecisions(model_decisions, updates)
        logger.info(f"Applied {len(updates)} SUMMA decision options to {model_decisions}")

    def _update_file_manager_paths(self):
        """Update fileManager.txt to use experiment-specific settings/results paths."""
        fm_path = self.settings_dir / 'fileManager.txt'
        if not fm_path.exists():
            logger.warning(f"fileManager.txt not found at {fm_path}")
            return

        with open(fm_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            s = line.strip()
            if s.startswith('settingsPath'):
                new_lines.append(f"settingsPath         '{self.settings_dir}/'\n")
            elif s.startswith('forcingPath'):
                new_lines.append(f"forcingPath          '{self.summa_input_dir}/'\n")
            elif s.startswith('outputPath'):
                new_lines.append(f"outputPath           '{self.ensemble_dir / 'results'}/'\n")
            else:
                new_lines.append(line)

        with open(fm_path, 'w') as f:
            f.writelines(new_lines)

        logger.info(
            f"Updated fileManager.txt paths:\n"
            f"  settingsPath -> {self.settings_dir}/\n"
            f"  forcingPath  -> {self.summa_input_dir}/\n"
            f"  outputPath   -> {self.ensemble_dir / 'results'}/"
        )
    
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
        self.last_run_label: Optional[str] = None

    def seed_default_settings_from_best(self, params_csv: str) -> pd.DataFrame:
        """
        Apply a prior best-parameter CSV to experiment-local SUMMA settings.

        This affects optimization initialization because iterative optimization
        copies from the source settings directory when creating a run workspace.
        """
        params_path = Path(params_csv).expanduser()
        if not params_path.is_absolute():
            params_path = (Path.cwd() / params_path).resolve()
        if not params_path.exists():
            raise FileNotFoundError(f"Seed parameters CSV not found: {params_path}")

        params_df = pd.read_csv(params_path)
        if 'parameter' not in params_df.columns or 'value' not in params_df.columns:
            raise ValueError("Seed CSV must contain 'parameter' and 'value' columns")

        # Ignore linear-reservoir-only parameters that cannot be mapped to
        # SUMMA local/basin parameter files.
        unsupported_summa_seed_params = {'k_fast', 'k_slow', 'f_fast'}
        skipped = params_df[params_df['parameter'].astype(str).str.strip().isin(unsupported_summa_seed_params)]
        if len(skipped) > 0:
            logger.info(
                "Ignoring non-SUMMA seed parameters: "
                + ', '.join(sorted(set(skipped['parameter'].astype(str).str.strip().tolist())))
            )
            params_df = params_df[
                ~params_df['parameter'].astype(str).str.strip().isin(unsupported_summa_seed_params)
            ].copy()

        from utils.custom.adjust_settings import update_and_reformat_parameter_file

        local_updates, basin_updates = split_optimized_parameter_groups(params_df)
        settings_dir = self.cfg.settings_dir
        if settings_dir is None:
            raise RuntimeError('Experiment settings directory is not initialized.')

        local_file = settings_dir / 'localParamInfo.txt'
        basin_file = settings_dir / 'basinParamInfo.txt'

        if not local_file.exists() or not basin_file.exists():
            raise FileNotFoundError(
                "Default settings files not found for optimization seeding: "
                f"{local_file}, {basin_file}"
            )

        local_result = update_and_reformat_parameter_file(
            local_file, local_updates, reformat_all=True, verbose=False
        )
        basin_result = update_and_reformat_parameter_file(
            basin_file, basin_updates, reformat_all=True, verbose=False
        )

        logger.info(
            "Seeded optimization defaults from prior best parameters: "
            f"{params_path}"
        )
        logger.info(
            f"  Updated default local params: {int(local_result.get('updated_count', 0))}"
        )
        logger.info(
            f"  Updated default basin params: {int(basin_result.get('updated_count', 0))}"
        )

        return params_df

    def _validate_workspace_settings(self, seed_params_df: Optional[pd.DataFrame] = None):
        """Fail fast if workspace settings drift from config expectations."""
        settings_dir = self.cfg.settings_dir
        if settings_dir is None:
            raise RuntimeError('Experiment settings directory is not initialized.')

        required_files = [
            settings_dir / 'modelDecisions.txt',
            settings_dir / 'localParamInfo.txt',
            settings_dir / 'basinParamInfo.txt',
            settings_dir / 'fileManager.txt',
        ]
        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                'Workspace settings validation failed. Missing required files:\n'
                + '\n'.join(missing_files)
            )

        # Validate model decisions against active YAML choices.
        decisions_cfg = self.cfg.raw.get('SUMMA_DECISION_OPTIONS', {})
        expected_decisions = {}
        for key, value in decisions_cfg.items():
            if isinstance(value, list):
                if value:
                    expected_decisions[key] = str(value[0])
            elif value is not None:
                expected_decisions[key] = str(value)

        actual_decisions = {}
        with open(settings_dir / 'modelDecisions.txt', 'r') as fin:
            for line in fin:
                stripped = line.lstrip()
                if stripped.startswith('!') or stripped.strip() == '':
                    continue
                code = line.split('!', 1)[0].strip()
                tokens = code.split()
                if len(tokens) >= 2:
                    actual_decisions[tokens[0]] = tokens[1]

        decision_mismatches = []
        for key, expected in expected_decisions.items():
            actual = actual_decisions.get(key)
            if actual != expected:
                decision_mismatches.append(f'{key}: expected={expected}, actual={actual}')

        if decision_mismatches:
            preview = '\n'.join(decision_mismatches[:20])
            raise RuntimeError(
                'Workspace modelDecisions.txt does not match SUMMA_DECISION_OPTIONS. '
                'Refusing to start optimization.\n'
                + preview
            )

        # Validate fileManager settingsPath points to this workspace.
        fm_settings_path = None
        with open(settings_dir / 'fileManager.txt', 'r') as fin:
            for line in fin:
                if line.strip().startswith('settingsPath'):
                    parts = line.split("'", 2)
                    if len(parts) >= 2:
                        fm_settings_path = parts[1]
                    break
        expected_settings_prefix = str(settings_dir) + '/'
        if fm_settings_path is None or fm_settings_path != expected_settings_prefix:
            raise RuntimeError(
                'Workspace fileManager.txt settingsPath mismatch. '
                f'expected={expected_settings_prefix}, actual={fm_settings_path}'
            )

        # If seeding was requested, verify parameters exist in workspace parameter files.
        if seed_params_df is not None and len(seed_params_df) > 0:
            unsupported_summa_seed_params = {'k_fast', 'k_slow', 'f_fast'}

            def _read_param_names(path: Path) -> set:
                names = set()
                with open(path, 'r') as fin:
                    for raw in fin:
                        s = raw.strip()
                        if not s or s.startswith('!'):
                            continue
                        if '|' in raw:
                            names.add(raw.split('|', 1)[0].strip())
                        else:
                            names.add(s.split()[0])
                return names

            local_names = _read_param_names(settings_dir / 'localParamInfo.txt')
            basin_names = _read_param_names(settings_dir / 'basinParamInfo.txt')

            missing_seed = []
            for raw_name in seed_params_df['parameter']:
                name = str(raw_name).strip()
                if name in unsupported_summa_seed_params:
                    continue
                if name.startswith('basin__'):
                    candidate = name.split('basin__', 1)[1]
                    present = candidate in basin_names
                elif name.startswith('routing__'):
                    candidate = name.split('routing__', 1)[1]
                    present = candidate in basin_names
                else:
                    candidate = name
                    present = (candidate in local_names) or (candidate in basin_names)
                if not present:
                    missing_seed.append(name)

            if missing_seed:
                preview = ', '.join(missing_seed[:25])
                raise RuntimeError(
                    'Seed parameters could not be mapped to workspace parameter files. '
                    'Refusing to start optimization. Missing (sample): '
                    + preview
                )
    
    def run_optimization(self, seed_params_csv: Optional[str] = None) -> pd.DataFrame:
        """Run optimization via the existing CONFLUENCE optimization pipeline."""
        logger.info("=" * 70)
        logger.info("PHASE 1: PARAMETER OPTIMIZATION")
        logger.info("=" * 70)

        seed_params_df: Optional[pd.DataFrame] = None
        if seed_params_csv:
            seed_params_df = self.seed_default_settings_from_best(seed_params_csv)

        # Guardrail: ensure workspace settings match active config before launching DDS.
        self._validate_workspace_settings(seed_params_df)
        
        run_label = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.last_run_label = run_label
        confluence = CONFLUENCE(
            config_path=str(self.cfg.config_path),
            config_overrides={
                'OPTIMIZATION_RUN_LABEL': run_label,
                'OPTIMIZATION_SOURCE_SETTINGS_DIR': str(self.cfg.settings_dir),
            }
        )
        logger.info(f"Optimization run label: {run_label}")
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
        """Write optimized parameters to SUMMA config files.

        Uses the same update+reformat pattern as the notebook examples via
        update_and_reformat_parameter_file(..., reformat_all=True).
        """
        sys.path.insert(0, str(self.cfg.code_dir))
        from utils.custom.adjust_settings import update_and_reformat_parameter_file, _format_parameter_value

        if 'parameter' not in params_df.columns or 'value' not in params_df.columns:
            raise ValueError("params_df must contain 'parameter' and 'value' columns")

        local_params, basin_params = split_optimized_parameter_groups(params_df)
        
        def _apply_with_fallback(file_path: Path, updates: Dict[str, Any], label: str) -> int:
            """Apply updates using adjust_settings, with a syntax-preserving fallback."""
            if not updates or not file_path.exists():
                return 0

            result = update_and_reformat_parameter_file(
                file_path, updates, reformat_all=True, verbose=False
            )
            updated_count = int(result.get('updated_count', 0))

            if updated_count > 0:
                return updated_count

            # Fallback for files where regex-based parser does not match current formatting.
            with open(file_path, 'r') as f_in:
                lines = f_in.readlines()

            new_lines = []
            fallback_updates = 0
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('!') or '|' not in line:
                    new_lines.append(line)
                    continue

                parts = line.rstrip('\n').split('|')
                if len(parts) < 4:
                    new_lines.append(line)
                    continue

                param_name = parts[0].strip()
                if param_name in updates:
                    parts[1] = f" {_format_parameter_value(updates[param_name])} "
                    line = '|'.join(parts) + '\n'
                    fallback_updates += 1

                new_lines.append(line)

            if fallback_updates > 0:
                with open(file_path, 'w') as f_out:
                    f_out.writelines(new_lines)
                logger.warning(
                    f"Primary parser found no matches in {file_path.name}; "
                    f"applied {fallback_updates} {label} updates via syntax-preserving fallback."
                )

            return fallback_updates

        local_file = self.cfg.settings_dir / 'localParamInfo.txt'
        basin_file = self.cfg.settings_dir / 'basinParamInfo.txt'

        local_count = _apply_with_fallback(local_file, local_params, 'local')
        basin_count = _apply_with_fallback(basin_file, basin_params, 'basin')

        logger.info(f"Updated {local_count} local parameters")
        logger.info(f"Updated {basin_count} basin parameters")
    
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

    def update_forcing_path(self, forcing_dir: Path):
        """Update forcingPath in SUMMA fileManager."""
        fm_path = self.cfg.settings_dir / 'fileManager.txt'
        with open(fm_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith('forcingPath'):
                new_lines.append(f"forcingPath     '{forcing_dir}/'\n")
            else:
                new_lines.append(line)

        with open(fm_path, 'w') as f:
            f.writelines(new_lines)
    
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
        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{experiment_id}_{run_stamp}.log'
        latest_log_file = log_dir / f'{experiment_id}.log'

        import os as _os
        run_env = _os.environ.copy()
        run_env.setdefault('OMP_NUM_THREADS', '1')

        with open(log_file, 'w') as lf:
            result = subprocess.run(
                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                timeout=10800, env=run_env, cwd=str(log_dir)  # 3-hour timeout
            )

        shutil.copy2(log_file, latest_log_file)
        
        if result.returncode != 0:
            mpi_logs = sorted(log_dir.glob('mpi*')) + sorted(log_dir.glob('*worker*'))
            mpi_hint = ''
            if mpi_logs:
                mpi_hint = f" MPI side logs in {log_dir}: {', '.join(p.name for p in mpi_logs[:10])}"
            raise RuntimeError(
                f"SUMMA failed for {experiment_id}. See {log_file} (latest: {latest_log_file}).{mpi_hint}"
            )
        
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
    
    def load_forcing_data(
        self,
        forcing_dir: Optional[Path] = None,
        years: Optional[List[int]] = None,
        water_year: bool = False,
    ) -> xr.Dataset:
        """Load forcing files into a single dataset.

        Parameters
        ----------
        forcing_dir : Path, optional
            Directory containing monthly NetCDF files.
        years : list of int, optional
            If given, only load files whose names contain these years
            (plus Oct-Dec of the preceding year for OND/Fall handling).
            Dramatically reduces memory for large archives.
        """
        fdir = forcing_dir or self.cfg.summa_input_dir
        all_files = sorted(fdir.glob("*.nc"))
        if not all_files:
            raise FileNotFoundError(f"No forcing files in {fdir}")

        if years is not None:
            # For calendar-year mode include all months for year and Dec of year-1.
            # For water-year mode include all months of year-1 and year.
            needed = set()
            for y in years:
                for m in range(1, 13):
                    needed.add(f"{y}{m:02d}")
                if water_year:
                    for m in range(1, 13):
                        needed.add(f"{y - 1}{m:02d}")
                else:
                    needed.add(f"{y - 1}12")
            files = [f for f in all_files if any(tag in f.stem for tag in needed)]
            files = sorted(files)
            if not files:
                logger.warning("Year filter matched no files — loading all")
                files = all_files
        else:
            files = all_files

        logger.info(f"Loading {len(files)} of {len(all_files)} forcing files …")

        # Avoid open_mfdataset hangs on some file/locking setups by loading
        # files sequentially, then concatenating in-memory.
        loaded = []
        for i, fp in enumerate(files, start=1):
            if i == 1 or i % 25 == 0 or i == len(files):
                logger.info(f"  Loading forcing file {i}/{len(files)}: {fp.name}")
            with xr.open_dataset(fp) as ds_i:
                loaded.append(ds_i.load())

        ds = xr.concat(loaded, dim='time').sortby('time')
        ds = ds.sel(time=~ds.indexes['time'].duplicated())

        logger.info(f"Loaded forcing: {ds.time.values[0]} to {ds.time.values[-1]}, "
                    f"{len(ds.time)} timesteps")
        return ds
    
    def build_ensemble_for_season(
        self,
        full_forcing: xr.Dataset,
        target_year: int,
        donor_years: List[int],
        season: str,
        output_dir: Path,
        water_year: bool = False,
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

        forcing_times = pd.DatetimeIndex(full_forcing.time.values)
        forcing_years = forcing_times.year
        forcing_months = forcing_times.month

        # SUMMA expects monthly forcing files listed in forcingFileList.txt.
        # Build monthly outputs per member instead of a single annual file.
        if water_year:
            target_months = [(target_year - 1, 10), (target_year - 1, 11), (target_year - 1, 12)]
            target_months.extend((target_year, m) for m in range(1, 10))
        else:
            target_months = [(target_year, m) for m in range(1, 13)]

        source_monthly = {}
        for fp in sorted(self.cfg.summa_input_dir.glob('*.nc')):
            tag = fp.stem.split('_')[-1]
            if len(tag) == 6 and tag.isdigit():
                source_monthly[tag] = fp
        
        # Extract target window forcing as baseline
        if water_year:
            target_start = f"{target_year - 1}-10-01"
            target_end = f"{target_year}-09-30 23:00"
        else:
            target_start = f"{target_year}-01-01"
            target_end = f"{target_year}-12-31 23:00"
        target_ds = full_forcing.sel(time=slice(target_start, target_end))
        target_times = pd.DatetimeIndex(target_ds.time.values)

        # Reuse masks because each ensemble member has the same target timeline.
        season_target_masks = {
            month: (target_times.month == month)
            for month in set(months)
        }
        output_month_masks = {
            (year_i, month_i): (target_times.year == year_i) & (target_times.month == month_i)
            for year_i, month_i in target_months
        }

        template_meta = {}
        for year_i, month_i in target_months:
            month_tag = f"{year_i}{month_i:02d}"
            template_file = source_monthly.get(month_tag)
            if template_file is None:
                raise FileNotFoundError(f"Missing template forcing file for {month_tag}")
            if month_tag in template_meta:
                continue

            with xr.open_dataset(template_file) as template_ds:
                var_attrs = {}
                var_encoding = {}
                for var_name in template_ds.variables:
                    var_attrs[var_name] = dict(template_ds[var_name].attrs)
                    filtered_encoding = {
                        key: value
                        for key, value in template_ds[var_name].encoding.items()
                        if key in {'dtype', '_FillValue', 'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'chunksizes'}
                        and value is not None
                    }
                    if filtered_encoding:
                        var_encoding[var_name] = filtered_encoding

                template_meta[month_tag] = {
                    'template_name': template_file.name,
                    'dataset_attrs': dict(template_ds.attrs),
                    'variable_attrs': var_attrs,
                    'variable_encoding': var_encoding,
                }
        
        for donor_year in donor_years:
            logger.info(f"  Building ensemble: {season} from {donor_year} → {target_year}")
            
            # Start with a copy of the target year
            ensemble_ds = target_ds.copy(deep=True)
            
            # Get donor season data
            for month in months:
                # Oct-Dec belong to the prior calendar year of the water year;
                # Jan-Sep belong to the target calendar year.
                # Seasons are now defined to be entirely within one of these
                # two groups, so no cross-year mixing is needed.
                if month >= 10:
                    donor_month_year = donor_year - 1
                else:
                    donor_month_year = donor_year
                
                # Select the donor month
                donor_mask = (forcing_months == month) & (forcing_years == donor_month_year)
                donor_indices = np.flatnonzero(donor_mask)
                
                # Select matching month in target
                target_month_mask = season_target_masks[month]
                
                if donor_indices.size == 0:
                    logger.warning(f"    No data for {donor_month_year}-{month:02d}, skipping")
                    continue

                donor_data = full_forcing.isel(time=donor_indices)
                
                # Match timestep count — handle leap year differences
                n_target = int(np.count_nonzero(target_month_mask))
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
            
            member_dir = output_dir / f"from_{donor_year}"
            member_dir.mkdir(parents=True, exist_ok=True)

            for year_i, month_i in target_months:
                month_mask = output_month_masks[(year_i, month_i)]
                month_ds = ensemble_ds.sel(time=month_mask)
                if month_ds.sizes.get('time', 0) == 0:
                    continue

                if 'data_step' in month_ds.variables:
                    month_ds = month_ds.drop_vars('data_step')

                for static_var in ('latitude', 'longitude', 'hruId'):
                    if static_var in month_ds.variables and 'time' in month_ds[static_var].dims:
                        month_ds[static_var] = month_ds[static_var].isel(time=0, drop=True)

                month_tag = f"{year_i}{month_i:02d}"
                metadata = template_meta[month_tag]
                month_ds.attrs = dict(metadata['dataset_attrs'])
                encoding = {}
                for var_name in month_ds.variables:
                    if var_name in metadata['variable_attrs']:
                        month_ds[var_name].attrs = dict(metadata['variable_attrs'][var_name])
                    if var_name in metadata['variable_encoding']:
                        encoding[var_name] = dict(metadata['variable_encoding'][var_name])

                # Keep SUMMA-native time convention; default xarray encoding
                # can switch to relative "hours since month-start".
                encoding['time'] = {
                    'dtype': 'int32',
                    'units': 'seconds since 1970-01-01',
                    'calendar': 'proleptic_gregorian',
                }

                # Preserve native monthly filenames so members can reuse the
                # standard forcingFileList.txt entries.
                out_name = metadata['template_name']
                out_path = member_dir / out_name
                month_ds.to_netcdf(out_path, encoding=encoding)

            created_files.append(member_dir)
            logger.info(f"    Saved member forcing directory: {member_dir.name}")
        
        return created_files
    
    def build_all_season_ensembles(
        self,
        full_forcing: xr.Dataset,
        target_year: int,
        donor_years: List[int],
        water_year: bool = False,
    ) -> Dict[str, List[Path]]:
        """Build ensembles for all four seasons."""
        all_files = {}
        for season in SEASONS:
            logger.info(f"\nBuilding {season} ({SEASONS[season]['name']}) ensemble...")
            out_dir = self.cfg.ensemble_dir / 'forcing' / season
            files = self.build_ensemble_for_season(
                full_forcing, target_year, donor_years, season, out_dir, water_year=water_year
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
        experiment_id: str = 'baseline_longterm'
    ) -> Path:
        """Run the baseline (unperturbed) simulation.
        
        Saves to experiment-specific folder to ensure reproducibility
        and avoid conflicts with other experiments using different configs.
        """
        logger.info("=" * 70)
        logger.info(f"RUNNING BASELINE: {start} to {end}")
        logger.info("=" * 70)
        
        self.runner.apply_parameters(params_df)
        self.runner.update_time_period(start, end)
        
        # Save to experiment-specific folder, not shared simulations folder
        output_dir = self.cfg.ensemble_dir / 'results' / 'baseline_longterm'
        return self.runner.run_summa(experiment_id, output_dir)
    
    def run_target_year_baseline(
        self,
        params_df: pd.DataFrame,
        target_year: int,
        water_year: bool = False,
    ) -> Path:
        """Run the unperturbed target year simulation.
        
        Saves to separate 'baseline_target' subfolder to keep distinct
        from the longterm baseline (baseline_longterm).
        """
        if water_year:
            start = f"{target_year - 1}-10-01 01:00"
            end = f"{target_year}-09-30 23:00"
            exp_id = f"target_wy_{target_year}"
        else:
            start = f"{target_year}-01-01 01:00"
            end = f"{target_year}-12-31 23:00"
            exp_id = f"target_year_{target_year}"
        # Save to separate folder to distinguish from longterm baseline
        output_dir = self.cfg.ensemble_dir / 'results' / 'baseline_target'
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
            self.runner.update_forcing_path(forcing_file.parent)
            
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
        self,
        season: str,
        donor_year: int,
        target_year: int,
        forcing_file: Path,
        water_year: bool = False,
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

        if forcing_file.is_dir():
            member_files = sorted(forcing_file.glob('*.nc'))
            if not member_files:
                raise FileNotFoundError(f"No forcing files found in member directory: {forcing_file}")

            # SUMMA expects the full forcing list indexing behavior; provide
            # a complete per-member forcing directory by symlinking native
            # files and replacing only perturbed months.
            forcing_overlay = run_dir / 'forcing_overlay'
            forcing_overlay.mkdir(parents=True, exist_ok=True)

            for src in sorted(self.cfg.summa_input_dir.glob('*.nc')):
                link = forcing_overlay / src.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(src.resolve())

            for perturbed in member_files:
                dst = forcing_overlay / perturbed.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                shutil.copy2(perturbed, dst)

            src_ffl = self.cfg.settings_dir / 'forcingFileList.txt'
            ffl.write_text(src_ffl.read_text())
            forcing_path = forcing_overlay
        else:
            ffl.write_text(forcing_file.name + '\n')
            forcing_path = forcing_file.parent
        
        # Override fileManager.txt symlink with a member-specific copy
        fm_link = settings_dir / 'fileManager.txt'
        if fm_link.is_symlink():
            fm_link.unlink()
        
        # Write the per-member fileManager.txt in the run directory
        prefix = self._output_prefix(season, donor_year)
        fm_path = run_dir / 'fileManager.txt'
        if water_year:
            start = f"{target_year - 1}-10-01 01:00"
            end = f"{target_year}-09-30 23:00"
        else:
            start = f"{target_year}-01-01 01:00"
            end = f"{target_year}-12-31 23:00"

        self._write_file_manager(
            fm_path,
            settings_path=str(settings_dir),
            forcing_path=str(forcing_path),
            output_path=str(output_dir),
            prefix=prefix,
            start=start,
            end=end,
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
        water_year: bool = False,
    ) -> Dict[str, dict]:
        """Prepare workspaces for every ensemble member across all seasons."""
        total = sum(len(ff) for ff in ensemble_forcing_files.values())
        logger.info(f"Preparing {total} ensemble member workspaces...")
        
        for season, forcing_files in ensemble_forcing_files.items():
            for ff in forcing_files:
                if ff.is_dir() and ff.name.startswith('from_'):
                    donor_year = int(ff.name.split('_', 1)[1])
                else:
                    parts = ff.stem.split('_from_')
                    donor_year = int(parts[-1])
                self.prepare_member(
                    season,
                    donor_year,
                    target_year,
                    ff,
                    water_year=water_year,
                )
        
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
        lf = open(info['log_file'], 'a')
        lf.write(f"\n===== launch {datetime.now().isoformat()} =====\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env,
            cwd=str(info['run_dir'])
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
                if out_files:
                    self.completed[mid] = {
                        'returncode': rc,
                        'output_file': out_files[-1],
                    }
                    logger.info(f"  DONE: {mid}")
                else:
                    self.failed[mid] = {
                        'returncode': 99,
                        'error': 'Process exited 0 but no output NetCDF was produced',
                    }
                    logger.warning(f"  FAILED: {mid} (no output file)")
            else:
                error_tail = ""
                log_file = self.members[mid]['log_file']
                run_dir = self.members[mid]['run_dir']
                if log_file.exists():
                    with open(log_file) as f:
                        error_tail = ''.join(deque(f, maxlen=5))
                mpi_logs = sorted(run_dir.glob('mpi*')) + sorted(run_dir.glob('*worker*'))
                mpi_hint = ''
                if mpi_logs:
                    mpi_hint = f"\nMPI logs: {', '.join(p.name for p in mpi_logs[:10])}"
                self.failed[mid] = {'returncode': rc, 'error': error_tail + mpi_hint}
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
            'declare -a OUTDIRS',
            'declare -a PREFIXES',
            'COMPLETED=0',
            'FAILED=0',
            f'TOTAL={total}',
            '',
            'member_output_exists() {',
            '    local idx="$1"',
            '    local odir="${OUTDIRS[$idx]}"',
            '    local prefix="${PREFIXES[$idx]}"',
            '    shopt -s nullglob',
            '    local files=("$odir"/"$prefix"*_timestep.nc "$odir"/"$prefix"*.nc)',
            '    shopt -u nullglob',
            '    [ ${#files[@]} -gt 0 ]',
            '}',
            '',
            '# ---- concurrency helper ----',
            'wait_for_slot() {',
            '    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do',
            '        for i in "${!PIDS[@]}"; do',
            '            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then',
            '                wait "${PIDS[$i]}"',
            '                RC=$?',
            '                if [ $RC -eq 0 ] && member_output_exists "$i"; then',
            '                    COMPLETED=$((COMPLETED + 1))',
            '                    echo "[$(date +%H:%M:%S)] DONE: ${NAMES[$i]}  ($COMPLETED/$TOTAL)"',
            '                else',
            '                    FAILED=$((FAILED + 1))',
            '                    echo "[$(date +%H:%M:%S)] FAIL: ${NAMES[$i]}  (rc=$RC or no output)"',
            '                fi',
            '                unset "PIDS[$i]"',
            '                unset "NAMES[$i]"',
            '                unset "OUTDIRS[$i]"',
            '                unset "PREFIXES[$i]"',
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
            member_lines.append(f'OUTDIRS+=("{info["output_dir"]}")')
            member_lines.append(f'PREFIXES+=("{info["prefix"]}")')
            member_lines.append('')
        
        footer = '\n'.join([
            '# ---- wait for stragglers ----',
            'for i in "${!PIDS[@]}"; do',
            '    wait "${PIDS[$i]}"',
            '    RC=$?',
            '    if [ $RC -eq 0 ] && member_output_exists "$i"; then',
            '        COMPLETED=$((COMPLETED + 1))',
            '        echo "[$(date +%H:%M:%S)] DONE: ${NAMES[$i]}  ($COMPLETED/$TOTAL)"',
            '    else',
            '        FAILED=$((FAILED + 1))',
            '        echo "[$(date +%H:%M:%S)] FAIL: ${NAMES[$i]}  (rc=$RC or no output)"',
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
                
                # NSE of daily streamflow vs baseline
                if 'Q' in member_seasonal and 'Q' in baseline_seasonal:
                    common_idx = member_seasonal.index.intersection(baseline_seasonal.index)
                    if len(common_idx) > 10:
                        obs = baseline_seasonal.loc[common_idx, 'Q'].values
                        sim = member_seasonal.loc[common_idx, 'Q'].values
                        record['NSE_vs_baseline'] = get_NSE(obs, sim)
                
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
        figsize: Tuple = (11, 7)
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
        if 'scalarTotalRunoff' not in bl_ds:
            raise KeyError("scalarTotalRunoff not found in baseline file for plotting")
        bl_q = bl_ds['scalarTotalRunoff'].values.squeeze()
        
        # Convert streamflow units if basin area available
        area = self.cfg.basin_area_m2 or 1.0
        if area > 1.0:
            bl_q = bl_q * area
        
        # Daily aggregation for plotting
        bl_df = pd.DataFrame({'P': bl_ppt, 'Q': bl_q}, index=bl_time).resample('D').mean()
        bl_ds.close()
        
        # Season highlight region
        season_mask = bl_df.index.month.isin(months)
        
        # Filter observations to water year if provided
        if obs_df is not None:
            water_year_start = pd.Timestamp(f'{target_year - 1}-10-01')
            water_year_end = pd.Timestamp(f'{target_year}-09-30')
            obs_df = obs_df.loc[water_year_start:water_year_end]
        
        # Plot ensemble members with higher opacity for visibility
        alpha_member = max(0.35, 1.0 / max(len(ensemble_files), 1))
        
        for donor_year, efile in sorted(ensemble_files.items()):
            try:
                m_ds = xr.open_dataset(efile)
                m_time = pd.DatetimeIndex(m_ds.time.values)
                m_ppt = m_ds['pptrate'].values.squeeze() if 'pptrate' in m_ds else None
                if 'scalarTotalRunoff' not in m_ds:
                    raise KeyError("scalarTotalRunoff not found in ensemble member file for plotting")
                m_q = m_ds['scalarTotalRunoff'].values.squeeze()
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
            obs_period = obs_df
            if not obs_period.empty:
                ax_q.plot(obs_period.index, obs_period['discharge_cms'],
                         color='red', linewidth=1.5, linestyle='--',
                         label='Observed', zorder=6)
        
        # Shade the perturbed season.
        # Oct-Dec (months >= 10) belong to calendar year target_year-1 in the water year.
        for ax in [ax_p, ax_q]:
            for start_month in months:
                shade_year = target_year - 1 if start_month >= 10 else target_year
                m_start = pd.Timestamp(f'{shade_year}-{start_month:02d}-01')
                if start_month == 12:
                    m_end = pd.Timestamp(f'{shade_year}-12-31')
                else:
                    next_month = start_month + 1
                    m_end = pd.Timestamp(f'{shade_year}-{next_month:02d}-01') - timedelta(days=1)
                ax.axvspan(m_start, m_end, alpha=0.08, color=season_color, zorder=0)
        
        # Format axes
        ax_p.set_ylabel('Precipitation (mm/hr)', fontsize=12, fontweight='bold')
        ax_p.set_title(
            f'Seasonal Forcing Ensemble: {season_name} ({season})\n'
            f'Target Year {target_year} with forcing replaced from 20 prior years',
            fontsize=13, fontweight='bold'
        )
        ax_p.legend(loc='upper right', fontsize=10)
        ax_p.grid(True, alpha=0.3)
        ax_p.invert_yaxis()  # Precipitation convention: bars from top
        ax_p.tick_params(labelsize=10)
        
        unit = 'm³/s' if area > 1.0 else 'm/s'
        ax_q.set_ylabel(f'Streamflow ({unit})', fontsize=12, fontweight='bold')
        ax_q.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax_q.legend(loc='upper right', fontsize=10)
        ax_q.grid(True, alpha=0.3)
        ax_q.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_q.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax_q.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
        ax_q.tick_params(labelsize=10)
        
        # Add ensemble member legend entry
        ensemble_line = Line2D([0], [0], color=season_color, linewidth=1.5, alpha=0.75)
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
        figsize: Tuple = (14, 14)
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
        if 'scalarTotalRunoff' not in bl_ds:
            raise KeyError("scalarTotalRunoff not found in baseline file for plotting")
        bl_q = bl_ds['scalarTotalRunoff'].values.squeeze()
        if area > 1.0:
            bl_q = bl_q * area
        bl_df = pd.DataFrame({'P': bl_ppt, 'Q': bl_q}, index=bl_time).resample('D').mean()
        bl_ds.close()
        
        # Filter observations to water year if provided
        if obs_df is not None:
            water_year_start = pd.Timestamp(f'{target_year - 1}-10-01')
            water_year_end = pd.Timestamp(f'{target_year}-09-30')
            obs_df = obs_df.loc[water_year_start:water_year_end]
        
        for row, season in enumerate(SEASONS):
            ax_p = axes[row, 0]
            ax_q = axes[row, 1]
            color = SEASONS[season]['color']
            months = SEASONS[season]['months']
            ensemble_files = all_ensemble_results.get(season, {})
            
            alpha_m = max(0.40, 1.0 / max(len(ensemble_files), 1))
            
            # Plot ensemble members
            for donor_year, efile in sorted(ensemble_files.items()):
                try:
                    m_ds = xr.open_dataset(efile)
                    m_time = pd.DatetimeIndex(m_ds.time.values)
                    m_ppt = m_ds['pptrate'].values.squeeze() if 'pptrate' in m_ds else None
                    if 'scalarTotalRunoff' not in m_ds:
                        raise KeyError("scalarTotalRunoff not found in ensemble member file for plotting")
                    m_q = m_ds['scalarTotalRunoff'].values.squeeze()
                    if area > 1.0:
                        m_q = m_q * area
                    m_df = pd.DataFrame({'P': m_ppt, 'Q': m_q}, index=m_time).resample('D').mean()
                    m_ds.close()
                    
                    ax_p.plot(m_df.index, m_df['P'] * 3600, color=color,
                             alpha=alpha_m, linewidth=0.8)
                    ax_q.plot(m_df.index, m_df['Q'], color=color,
                             alpha=alpha_m, linewidth=0.8)
                except Exception:
                    pass
            
            # Plot baseline
            ax_p.plot(bl_df.index, bl_df['P'] * 3600, color='black', linewidth=2.0)
            ax_q.plot(bl_df.index, bl_df['Q'], color='black', linewidth=2.0)
            
            # Observations (filtered to water year)
            if obs_df is not None and 'discharge_cms' in obs_df.columns:
                obs_period = obs_df
                if not obs_period.empty:
                    ax_q.plot(obs_period.index, obs_period['discharge_cms'],
                             color='red', linewidth=1.2, linestyle='--', alpha=0.8)
            
            # Shade season months.
            # Oct-Dec (months >= 10) belong to calendar year target_year-1 in the water year.
            for m in months:
                shade_year = target_year - 1 if m >= 10 else target_year
                m_start = pd.Timestamp(f'{shade_year}-{m:02d}-01')
                if m == 12:
                    m_end = pd.Timestamp(f'{shade_year}-12-31')
                else:
                    nm = m + 1
                    m_end = pd.Timestamp(f'{shade_year}-{nm:02d}-01') - timedelta(days=1)
                ax_p.axvspan(m_start, m_end, alpha=0.08, color=color)
                ax_q.axvspan(m_start, m_end, alpha=0.08, color=color)
            
            ax_p.set_ylabel(f'{season}\nmm/hr', fontsize=11, fontweight='bold')
            ax_p.invert_yaxis()
            ax_p.grid(True, alpha=0.2)
            ax_p.tick_params(labelsize=9)
            
            unit = 'm³/s' if area > 1.0 else 'm/s'
            ax_q.set_ylabel(f'{season}\n{unit}', fontsize=11, fontweight='bold')
            ax_q.grid(True, alpha=0.2)
            ax_q.tick_params(labelsize=9)
            
            if row == 0:
                ax_p.set_title('Precipitation', fontweight='bold', fontsize=12)
                ax_q.set_title('Streamflow', fontweight='bold', fontsize=12)
        
        # Format x-axis
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axes[-1, 1].xaxis.set_major_locator(mdates.MonthLocator())
        
        plt.setp(axes[-1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(axes[-1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        # Shared legend
        ensemble_line = Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.75)
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
    
    def __init__(
        self,
        config_path: str,
        max_workers: int = 4,
        experiment_name: str = None,
        existing_experiment: Optional[str] = None,
    ):
        self.cfg = ExperimentConfig(config_path)
        self.max_workers = max_workers
        
        # Either attach to an existing workspace or create a new dated one.
        if existing_experiment:
            self.cfg.use_existing_experiment_workspace(existing_experiment)
        else:
            self.cfg.initialize_experiment_workspace(experiment_name)
        
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
        self.ensemble_water_year = False
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
            Default is False to preserve native forcing LWRadAtm values.
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
            force_rebuild_summa_input = bool(cfg_dict.get('FORCE_REBUILD_SUMMA_INPUT', False))
            keep_summa_input_backup = bool(cfg_dict.get('KEEP_SUMMA_INPUT_BACKUP', True))
            forcing_product_tag = cfg_dict.get('FORCING_PRODUCT_TAG', None)

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
            if force_rebuild_summa_input or fp.missing_summa:
                if force_rebuild_summa_input:
                    logger.info("Force rebuilding SUMMA_input from selected basin-averaged files...")
                logger.info(
                    f"Creating {len(fp.missing_summa)} SUMMA input files..."
                )
                fp.create_summa_input(
                    recalc_longwave=recalc_longwave,
                    force_rebuild=force_rebuild_summa_input,
                    keep_backup=keep_summa_input_backup,
                    source_tag=forcing_product_tag,
                )
            else:
                logger.info("All SUMMA input files already exist.")

            # 3b) Guardrail: ensure every SUMMA forcing file has consistent data_step.
            ds_report = fp.validate_summa_input_data_step(
                fix_missing=True,
                fix_mismatch=True,
                verbose=False,
            )
            logger.info(
                "SUMMA forcing data_step check: "
                f"checked={ds_report['checked']}, "
                f"fixed_missing={ds_report['fixed_missing']}, "
                f"fixed_mismatch={ds_report['fixed_mismatch']}, "
                f"remaining_missing={len(ds_report['missing'])}, "
                f"remaining_mismatch={len(ds_report['mismatch'])}"
            )

            if ds_report['missing'] or ds_report['mismatch']:
                raise RuntimeError(
                    "SUMMA forcing files failed data_step consistency checks. "
                    "Inspect SUMMA_input files before running model."
                )

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
    
    def step1_optimize(
        self,
        skip_if_exists: bool = True,
        seed_params_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run or load optimization and apply best parameters to settings files."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: PARAMETER OPTIMIZATION")
        logger.info("=" * 70)

        if seed_params_csv and skip_if_exists:
            logger.info(
                "seed_params_csv provided; forcing a new optimization run "
                "(skip_if_exists=False)."
            )
            skip_if_exists = False
        
        self.best_params = None
        if skip_if_exists and self.cfg.opt_dir.exists():
            existing = list(self.cfg.opt_dir.glob("*/best_parameters.csv"))
            if existing:
                logger.info(f"Found existing optimization results, loading...")
                self.best_params = self.optimizer.get_best_parameters()

        if self.best_params is None:
            self.best_params = self.optimizer.run_optimization(seed_params_csv=seed_params_csv)
        
        # Apply best parameters to settings files with correct formatting
        logger.info("Applying best parameters to SUMMA settings files...")
        self.ensemble_runner.runner.apply_parameters(self.best_params)
        logger.info("✓ Best parameters applied to localParamInfo.txt and basinParamInfo.txt")

        # Save a copy of the best parameters alongside the settings files
        params_out = self.cfg.settings_dir / 'best_parameters.csv'
        self.best_params.to_csv(params_out, index=False)
        logger.info(f"✓ Best parameters saved to: {params_out.relative_to(self.cfg.project_dir)}")

        return self.best_params
    
    def step2_long_term_run(
        self,
        start: str,
        end: str,
        experiment_id: str = 'baseline_longterm'
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
    
    def step2b_create_warm_state(
        self,
        summa_output: Path = None,
        extract_time: Optional[str] = None,
    ) -> Path:
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
        configured_cold_state = self.cfg.raw.get('SETTINGS_SUMMA_COLDSTATE', 'coldState_updated.nc')
        cold_state_path = Path(configured_cold_state)
        if not cold_state_path.is_absolute():
            cold_state_path = self.cfg.settings_dir / cold_state_path
        
        ds_cold = xr.open_dataset(cold_state_path)
        ds_out = xr.open_dataset(output_nc)
        
        # Use explicit extract_time when provided, otherwise use last timestep.
        if extract_time is not None:
            target_ts = pd.Timestamp(extract_time)
            all_times = pd.DatetimeIndex(ds_out.time.values)
            # Use nearest-neighbor to handle nanosecond precision/rounding artifacts
            t_idx = int(all_times.get_indexer([target_ts], method='nearest')[0])
            actual_ts = pd.Timestamp(ds_out.time.values[t_idx])
            time_diff = abs((actual_ts - target_ts).total_seconds())
            if time_diff > 3600:  # More than 1 hour away
                raise ValueError(
                    f"extract_time {target_ts} not found within 1 hour in {output_nc.name}. "
                    f"Closest available: {actual_ts} ({time_diff:.1f}s away). "
                    f"Available range: {all_times.min()} to {all_times.max()}"
                )
            logger.info(f"extract_time {target_ts} -> nearest available: {actual_ts}")
        else:
            t_idx = -1

        last_time = pd.Timestamp(ds_out.time.values[t_idx])
        ds_last = ds_out.isel(time=t_idx)
        n_soil = int(ds_cold['nSoil'].values.flat[0])
        
        logger.info(f"Extracting state from: {output_nc.name}")
        logger.info(f"Last timestep: {last_time}")
        logger.info(f"Soil layers: {n_soil}")

        def _is_valid_scalar(val: float) -> bool:
            return np.isfinite(val) and abs(val) < 1e6 and val > -9000

        def _is_valid_array(arr: np.ndarray) -> bool:
            if arr.size == 0:
                return False
            if not np.all(np.isfinite(arr)):
                return False
            # SUMMA fill values frequently appear as -9999 in outputs.
            if np.any(arr <= -9000):
                return False
            if np.any(np.abs(arr) > 1e6):
                return False
            return True
        
        def get_scalar(var_name, default=0.0):
            """Get scalar value from output, trying multiple suffixes."""
            for suffix in ['', '_inst', '_mean', '_sum']:
                name = var_name + suffix
                if name in ds_last:
                    val = float(ds_last[name].values.flat[0])
                    if _is_valid_scalar(val):
                        logger.info(f"  {var_name:30s} = {val:12.4f}  (from {name})")
                        return val
                    logger.warning(
                        f"  {var_name:30s} = {val:12.4f}  (invalid from {name}, using default)"
                    )
                    break
            logger.warning(f"  {var_name:30s} = {default:12.4f}  (DEFAULT — not in output)")
            return default
        
        def get_layer_values(var_name, n_target):
            """Extract the first n_target valid layer values from output arrays.

            SUMMA outputs can include padded snow/soil layers with fill values like -9999.
            For restart files we want the physically valid soil/interface entries only.
            """
            for suffix in ['', '_inst', '_mean', '_sum']:
                name = var_name + suffix
                if name in ds_last:
                    vals = np.asarray(ds_last[name].values.squeeze(), dtype=float).reshape(-1)
                    valid_vals = vals[np.isfinite(vals) & (vals > -9000) & (np.abs(vals) < 1e6)]
                    if valid_vals.size >= n_target:
                        selected = valid_vals[:n_target]
                        logger.info(f"  {var_name:30s} = {selected}  (from {name})")
                        return selected
                    logger.warning(
                        f"  {var_name:30s} = only {valid_vals.size} valid values in {name}, keeping defaults"
                    )
                    break
            logger.warning(f"  {var_name:30s} = not found, keeping defaults")
            return None
        
        # Copy cold state file and update values in place to preserve exact
        # NetCDF structure SUMMA expects for restart files.
        shutil.copy2(cold_state_path, warm_state_path)

        scalar_updates = {
            'scalarCanopyIce': get_scalar('scalarCanopyIce'),
            'scalarCanopyLiq': get_scalar('scalarCanopyLiq'),
            'scalarSnowDepth': get_scalar('scalarSnowDepth'),
            'scalarSWE': get_scalar('scalarSWE'),
            'scalarSfcMeltPond': get_scalar('scalarSfcMeltPond'),
            'scalarAquiferStorage': get_scalar(
                'scalarAquiferStorage',
                default=float(ds_cold['scalarAquiferStorage'].values.flat[0]),
            ),
            'scalarSnowAlbedo': get_scalar('scalarSnowAlbedo'),
            'scalarCanairTemp': get_scalar('scalarCanairTemp', default=283.16),
            'scalarCanopyTemp': get_scalar('scalarCanopyTemp', default=283.16),
        }

        layer_vars = [
            'mLayerTemp',
            'mLayerVolFracIce',
            'mLayerVolFracLiq',
            'mLayerMatricHead',
        ]
        layer_updates = {
            var: get_layer_values(var, n_soil)
            for var in layer_vars
        }
        interface_updates = {
            'iLayerHeight': get_layer_values('iLayerHeight', n_soil + 1),
            'mLayerDepth': get_layer_values('mLayerDepth', n_soil),
        }

        with nc.Dataset(warm_state_path, 'r+') as ds_warm:
            for var, val in scalar_updates.items():
                if var in ds_warm.variables:
                    ds_warm.variables[var][:] = val

            for var, soil_vals in layer_updates.items():
                if soil_vals is None or var not in ds_warm.variables:
                    continue
                var_data = ds_warm.variables[var]
                var_data[:] = np.asarray(soil_vals, dtype=var_data.dtype).reshape(var_data.shape)

            for var, vals in interface_updates.items():
                if vals is None or var not in ds_warm.variables:
                    continue
                var_data = ds_warm.variables[var]
                var_data[:] = np.asarray(vals, dtype=var_data.dtype).reshape(var_data.shape)

            # Keep nSnow=0 — SUMMA will create snow layers dynamically.
            if 'nSnow' in ds_warm.variables:
                ds_warm.variables['nSnow'][:] = 0

            ds_warm.setncattr('Author', 'Created by seasonal_ensemble_experiment.py')
            ds_warm.setncattr(
                'History',
                f'Warm state extracted from {output_nc.name}, last timestep {last_time}'
            )
            ds_warm.setncattr(
                'Purpose',
                'Warm start initial conditions from long-term baseline run'
            )

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
                # Request time-mean output to avoid invalid integration flags
                # for prognostic/non-time variables in some SUMMA builds.
                f.write(f'{var:40s} | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n')
        
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
    
    def _slice_target_year_from_continuous(self, target_year: int) -> Optional[Path]:
        """Create target-year baseline by slicing from a continuous baseline run.

        Returns None when the continuous baseline does not cover the target water year.
        """
        if self.baseline_output is None or not Path(self.baseline_output).exists():
            return None

        wy_start = pd.Timestamp(f"{target_year - 1}-10-01 01:00")
        wy_end = pd.Timestamp(f"{target_year}-09-30 23:00")

        with xr.open_dataset(self.baseline_output) as ds:
            times = pd.DatetimeIndex(ds.time.values)
            if wy_start < times.min() or wy_end > times.max():
                logger.info(
                    "Continuous baseline does not cover target WY window "
                    f"({wy_start} to {wy_end}); falling back to standalone target-year run."
                )
                return None

            ds_wy = ds.sel(time=slice(wy_start, wy_end)).load()

        out_dir = self.cfg.ensemble_dir / 'results' / 'baseline'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'target_wy_{target_year}_timestep.nc'
        ds_wy.to_netcdf(out_path)
        ds_wy.close()

        logger.info(f"Created target WY baseline by slicing continuous run: {out_path}")
        return out_path

    def step3_target_year_baseline(
        self,
        target_year: int,
        prefer_continuous: bool = True,
    ) -> Path:
        """Create or run the unperturbed target year baseline.

        If prefer_continuous=True and the long-term baseline output covers the
        target water year, this method slices that window directly instead of
        running a second standalone SUMMA simulation.
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"STEP 3: TARGET YEAR BASELINE ({target_year})")
        logger.info("=" * 70)

        if prefer_continuous:
            sliced = self._slice_target_year_from_continuous(target_year)
            if sliced is not None:
                self.target_baseline_output = sliced
                return self.target_baseline_output
        
        self.target_baseline_output = self.ensemble_runner.run_target_year_baseline(
            self.best_params, target_year, water_year=True
        )
        return self.target_baseline_output
    
    def step4_build_ensembles(
        self,
        target_year: int,
        donor_years: List[int],
        forcing_dir: Optional[Path] = None,
        water_year: bool = True,
    ) -> Dict[str, List[Path]]:
        """Build seasonal forcing ensemble files."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: BUILD SEASONAL FORCING ENSEMBLES")
        logger.info("=" * 70)
        
        needed_years = sorted(set(donor_years) | {target_year})
        self.ensemble_water_year = water_year
        full_forcing = self.ensemble_builder.load_forcing_data(
            forcing_dir,
            years=needed_years,
            water_year=water_year,
        )
        self.ensemble_forcing_files = self.ensemble_builder.build_all_season_ensembles(
            full_forcing, target_year, donor_years, water_year=water_year
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
            self.parallel_runner.prepare_all(
                target_year,
                self.ensemble_forcing_files,
                water_year=self.ensemble_water_year,
            )
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
        self.parallel_runner.prepare_all(
            target_year,
            self.ensemble_forcing_files,
            water_year=self.ensemble_water_year,
        )
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
        
        # Filter observations to water year (Oct of previous year through Sep of target year)
        if obs_df is not None:
            water_year_start = pd.Timestamp(f'{target_year - 1}-10-01')
            water_year_end = pd.Timestamp(f'{target_year}-09-30')
            obs_df = obs_df.loc[water_year_start:water_year_end]
        
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
        self.step3_target_year_baseline(target_year, prefer_continuous=True)
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
