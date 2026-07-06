"""
Parallel Experiment Runner for Climate Perturbation Scenarios

This module orchestrates parallel execution of multiple SUMMA model runs with
perturbed forcing data, managing resources and aggregating results.

Author: dlhogan
Date: March 2, 2026
"""

import os
import sys
import shutil
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a single experimental scenario."""
    scenario_id: str
    forcing_dir: Path
    output_dir: Path
    coldstate_path: Path
    config_updates: Dict = field(default_factory=dict)
    season: str = ''
    temp_delta: float = 0.0
    precip_mult: float = 1.0
    

class ParallelExperimentRunner:
    """
    Execute multiple SUMMA scenarios in parallel with resource management.
    
    Features:
    - Parallel execution of independent scenarios
    - Automatic directory structure creation
    - Configuration file management
    - Progress tracking and logging
    - Error handling and retry logic
    - Result organization and metadata
    """
    
    def __init__(self,
                 base_config_path: Union[str, Path],
                 experiment_base_dir: Union[str, Path],
                 equilibrium_coldstate: Union[str, Path],
                 max_workers: int = 4,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize parallel experiment runner.
        
        Parameters
        ----------
        base_config_path : Path
            Path to base CONFLUENCE configuration file
        experiment_base_dir : Path
            Base directory for all experimental runs
        equilibrium_coldstate : Path
            Path to equilibrium coldState.nc from spinup run
        max_workers : int
            Maximum number of parallel workers (default 4)
        logger : logging.Logger, optional
            Logger instance
        """
        self.base_config_path = Path(base_config_path)
        self.experiment_base_dir = Path(experiment_base_dir)
        self.equilibrium_coldstate = Path(equilibrium_coldstate)
        self.max_workers = max_workers
        
        self.logger = logger or self._setup_logger()
        
        # Verify inputs
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config_path}")
        if not self.equilibrium_coldstate.exists():
            raise FileNotFoundError(f"Equilibrium coldState not found: {self.equilibrium_coldstate}")
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Create directory structure
        self.runs_dir = self.experiment_base_dir / 'runs'
        self.results_dir = self.experiment_base_dir / 'results'
        self.logs_dir = self.experiment_base_dir / 'logs'
        
        for dir_path in [self.runs_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ParallelExperimentRunner")
        self.logger.info(f"  Base config: {self.base_config_path}")
        self.logger.info(f"  Experiment dir: {self.experiment_base_dir}")
        self.logger.info(f"  Max workers: {self.max_workers}")
        
        # Track scenarios
        self.scenarios: List[ScenarioConfig] = []
        self.results: Dict[str, Dict] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Create default logger."""
        logger = logging.getLogger('ParallelExperimentRunner')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_scenario(self,
                    scenario_id: str,
                    forcing_dir: Union[str, Path],
                    season: str = '',
                    temp_delta: float = 0.0,
                    precip_mult: float = 1.0,
                    config_updates: Optional[Dict] = None) -> None:
        """
        Add a scenario to the execution queue.
        
        Parameters
        ----------
        scenario_id : str
            Unique scenario identifier
        forcing_dir : Path
            Directory containing perturbed forcing files for this scenario
        season : str
            Season that was perturbed
        temp_delta : float
            Temperature perturbation applied (°C)
        precip_mult : float
            Precipitation multiplier applied
        config_updates : dict, optional
            Additional configuration updates for this scenario
        """
        forcing_dir = Path(forcing_dir)
        if not forcing_dir.exists():
            raise FileNotFoundError(f"Forcing directory not found: {forcing_dir}")
        
        output_dir = self.runs_dir / scenario_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scenario = ScenarioConfig(
            scenario_id=scenario_id,
            forcing_dir=forcing_dir,
            output_dir=output_dir,
            coldstate_path=self.equilibrium_coldstate,
            config_updates=config_updates or {},
            season=season,
            temp_delta=temp_delta,
            precip_mult=precip_mult
        )
        
        self.scenarios.append(scenario)
        self.logger.debug(f"Added scenario: {scenario_id}")
    
    def add_scenarios_from_forcing_dirs(self,
                                       forcing_base_dir: Union[str, Path],
                                       scenario_pattern: str = 'WY*') -> int:
        """
        Automatically add scenarios from a directory of perturbed forcing files.
        
        Parameters
        ----------
        forcing_base_dir : Path
            Base directory containing subdirectories of perturbed forcing
        scenario_pattern : str
            Glob pattern to match scenario directories (default 'WY*')
            
        Returns
        -------
        count : int
            Number of scenarios added
        """
        forcing_base_dir = Path(forcing_base_dir)
        scenario_dirs = sorted(forcing_base_dir.glob(scenario_pattern))
        
        for scenario_dir in scenario_dirs:
            if not scenario_dir.is_dir():
                continue
            
            # Parse scenario ID to extract metadata (assumes naming convention)
            scenario_id = scenario_dir.name
            
            # Try to parse season and perturbations from name
            # Expected format: WY{year}_{season}_{temp}_{precip}
            parts = scenario_id.split('_')
            season = parts[1] if len(parts) > 1 else ''
            
            # Parse temperature
            temp_delta = 0.0
            if len(parts) > 2:
                temp_str = parts[2]
                if 'warm' in temp_str:
                    temp_delta = float(temp_str.replace('warmplus', '').replace('warm', ''))
                elif 'cold' in temp_str:
                    temp_delta = -float(temp_str.replace('coldminus', '').replace('cold', ''))
            
            # Parse precipitation
            precip_mult = 1.0
            if len(parts) > 3:
                precip_str = parts[3]
                if precip_str != 'baseline':
                    # Extract number from 'wet130' or 'dry70'
                    import re
                    match = re.search(r'\d+', precip_str)
                    if match:
                        precip_mult = float(match.group()) / 100.0
            
            self.add_scenario(
                scenario_id=scenario_id,
                forcing_dir=scenario_dir,
                season=season,
                temp_delta=temp_delta,
                precip_mult=precip_mult
            )
        
        self.logger.info(f"Added {len(scenario_dirs)} scenarios from {forcing_base_dir}")
        return len(scenario_dirs)
    
    def prepare_scenario(self, scenario: ScenarioConfig) -> bool:
        """
        Prepare a scenario for execution: copy files, update configs.
        
        Parameters
        ----------
        scenario : ScenarioConfig
            Scenario configuration
            
        Returns
        -------
        success : bool
            True if preparation successful
        """
        try:
            # Create settings directory
            settings_dir = scenario.output_dir / 'settings' / 'SUMMA'
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy equilibrium coldState
            coldstate_dest = settings_dir / 'coldState.nc'
            shutil.copy2(scenario.coldstate_path, coldstate_dest)
            
            # Copy base SUMMA settings files
            base_settings_dir = Path(self.base_config['CONFLUENCE_DATA_DIR']) / f"domain_{self.base_config['DOMAIN_NAME']}" / 'settings' / 'SUMMA'
            
            required_files = [
                'fileManager.txt',
                'modelDecisions.txt', 
                'outputControl.txt',
                'localParamInfo.txt',
                'basinParamInfo.txt',
                'attributes.nc',
                'trialParams.nc'
            ]
            
            for filename in required_files:
                src = base_settings_dir / filename
                if src.exists():
                    shutil.copy2(src, settings_dir / filename)
            
            # Create scenario-specific config
            scenario_config = self.base_config.copy()
            scenario_config['EXPERIMENT_ID'] = scenario.scenario_id
            scenario_config.update(scenario.config_updates)
            
            # Update forcing file paths to point to perturbed forcing
            # This requires modifying forcingFileList.txt
            self._update_forcing_file_list(settings_dir, scenario.forcing_dir)
            
            # Save scenario config
            config_path = scenario.output_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(scenario_config, f, default_flow_style=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare {scenario.scenario_id}: {e}")
            return False
    
    def _update_forcing_file_list(self, settings_dir: Path, forcing_dir: Path) -> None:
        """Update forcingFileList.txt to point to perturbed forcing files."""
        forcing_files = sorted(forcing_dir.glob('*.nc'))
        
        forcing_list_path = settings_dir / 'forcingFileList.txt'
        with open(forcing_list_path, 'w') as f:
            f.write("! Forcing file list for perturbed scenario\n")
            f.write("! Auto-generated by ParallelExperimentRunner\n")
            f.write("'---'  ! header\n")
            for forcing_file in forcing_files:
                f.write(f"'{forcing_file}'\n")
    
    def run_scenario(self, scenario: ScenarioConfig) -> Dict:
        """
        Execute a single scenario (called by parallel workers).
        
        Parameters
        ----------
        scenario : ScenarioConfig
            Scenario configuration
            
        Returns
        -------
        result : dict
            Execution result with status and metrics
        """
        result = {
            'scenario_id': scenario.scenario_id,
            'status': 'FAILED',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'error_message': None
        }
        
        start_time = datetime.now()
        
        try:
            # Prepare scenario
            if not self.prepare_scenario(scenario):
                result['error_message'] = 'Preparation failed'
                return result
            
            # Load scenario config
            config_path = scenario.output_dir / 'config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Import CONFLUENCE here (in worker process)
            sys.path.append(str(Path(config['CONFLUENCE_CODE_DIR'])))
            from CONFLUENCE import CONFLUENCE
            
            # Initialize CONFLUENCE for this scenario
            confluence = CONFLUENCE(config)
            
            # Run SUMMA
            self.logger.info(f"Running SUMMA for {scenario.scenario_id}...")
            confluence.managers['model'].run_summa()
            
            # Check if output exists
            output_file = scenario.output_dir / 'simulations' / f"{config['EXPERIMENT_ID']}_timestep.nc"
            if not output_file.exists():
                result['error_message'] = 'Output file not created'
                return result
            
            result['status'] = 'SUCCESS'
            result['output_file'] = str(output_file)
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"Failed to run {scenario.scenario_id}: {e}")
        
        finally:
            end_time = datetime.now()
            result['end_time'] = end_time.isoformat()
            result['duration_seconds'] = (end_time - start_time).total_seconds()
        
        return result
    
    def run_all_parallel(self) -> pd.DataFrame:
        """
        Execute all scenarios in parallel.
        
        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with execution results for all scenarios
        """
        if not self.scenarios:
            raise ValueError("No scenarios to run. Use add_scenario() first.")
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Starting parallel execution of {len(self.scenarios)} scenarios")
        self.logger.info(f"Max workers: {self.max_workers}")
        self.logger.info(f"{'='*70}\n")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self.run_scenario, scenario): scenario
                for scenario in self.scenarios
            }
            
            # Process completed scenarios
            completed = 0
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
                    self.logger.info(
                        f"{status_symbol} [{completed}/{len(self.scenarios)}] "
                        f"{scenario.scenario_id}: {result['status']} "
                        f"({result['duration_seconds']:.1f}s)"
                    )
                    
                except Exception as e:
                    self.logger.error(f"✗ {scenario.scenario_id} raised exception: {e}")
                    results.append({
                        'scenario_id': scenario.scenario_id,
                        'status': 'FAILED',
                        'error_message': str(e)
                    })
                    completed += 1
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Summary
        success_count = (results_df['status'] == 'SUCCESS').sum()
        fail_count = len(results_df) - success_count
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"EXECUTION COMPLETE")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total scenarios: {len(results_df)}")
        self.logger.info(f"  SUCCESS: {success_count}")
        self.logger.info(f"  FAILED:  {fail_count}")
        
        if success_count > 0:
            avg_duration = results_df[results_df['status'] == 'SUCCESS']['duration_seconds'].mean()
            self.logger.info(f"Average runtime: {avg_duration:.1f} seconds")
        
        # Save results
        results_path = self.results_dir / 'execution_results.csv'
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"\nResults saved to: {results_path}")
        
        return results_df
    
    def run_all_sequential(self) -> pd.DataFrame:
        """
        Execute all scenarios sequentially (for debugging).
        
        Returns
        -------
        results_df : pd.DataFrame
            DataFrame with execution results
        """
        self.logger.info(f"Starting sequential execution of {len(self.scenarios)} scenarios")
        
        results = []
        for i, scenario in enumerate(self.scenarios, 1):
            self.logger.info(f"\n[{i}/{len(self.scenarios)}] Running {scenario.scenario_id}...")
            result = self.run_scenario(scenario)
            results.append(result)
            
            status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
            self.logger.info(f"{status_symbol} {scenario.scenario_id}: {result['status']}")
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.results_dir / 'execution_results.csv'
        results_df.to_csv(results_path, index=False)
        
        return results_df


def quick_run_factorial_experiments(base_config_path: Union[str, Path],
                                   experiment_base_dir: Union[str, Path],
                                   forcing_base_dir: Union[str, Path],
                                   equilibrium_coldstate: Union[str, Path],
                                   max_workers: int = 4,
                                   scenario_pattern: str = 'WY*') -> pd.DataFrame:
    """
    Convenience function to run all factorial experiments from perturbed forcing.
    
    Parameters
    ----------
    base_config_path : Path
        Base CONFLUENCE configuration
    experiment_base_dir : Path
        Base directory for experiment outputs
    forcing_base_dir : Path
        Directory containing perturbed forcing subdirectories
    equilibrium_coldstate : Path
        Equilibrium initial conditions from spinup
    max_workers : int
        Number of parallel workers
    scenario_pattern : str
        Glob pattern for scenario directories
        
    Returns
    -------
    results_df : pd.DataFrame
        Execution results
        
    Examples
    --------
    >>> results = quick_run_factorial_experiments(
    ...     base_config_path='config_Tuolumne.yaml',
    ...     experiment_base_dir='/scratch/experiments',
    ...     forcing_base_dir='/scratch/experiments/forcing_perturbed',
    ...     equilibrium_coldstate='/scratch/data/spinup/coldState_2015.nc',
    ...     max_workers=8
    ... )
    """
    runner = ParallelExperimentRunner(
        base_config_path=base_config_path,
        experiment_base_dir=experiment_base_dir,
        equilibrium_coldstate=equilibrium_coldstate,
        max_workers=max_workers
    )
    
    # Auto-discover scenarios
    runner.add_scenarios_from_forcing_dirs(forcing_base_dir, scenario_pattern)
    
    # Run in parallel
    results_df = runner.run_all_parallel()
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run climate perturbation experiments in parallel')
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--experiment-dir', type=str, required=True, help='Experiment base directory')
    parser.add_argument('--forcing-dir', type=str, required=True, help='Perturbed forcing directory')
    parser.add_argument('--coldstate', type=str, required=True, help='Equilibrium coldState file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (for debugging)')
    
    args = parser.parse_args()
    
    runner = ParallelExperimentRunner(
        base_config_path=args.config,
        experiment_base_dir=args.experiment_dir,
        equilibrium_coldstate=args.coldstate,
        max_workers=args.workers
    )
    
    runner.add_scenarios_from_forcing_dirs(args.forcing_dir)
    
    if args.sequential:
        results = runner.run_all_sequential()
    else:
        results = runner.run_all_parallel()
    
    print("\n✓ Execution complete")
    print(f"Results: {args.experiment_dir}/results/execution_results.csv")
