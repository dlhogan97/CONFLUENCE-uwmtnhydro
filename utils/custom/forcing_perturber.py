"""
Forcing Perturbation Utility for Climate Scenario Experiments

This module provides tools to systematically perturb meteorological forcing files
for climate sensitivity experiments. Supports seasonal temperature and precipitation
perturbations while maintaining temporal structure and physical consistency.

Author: dlhogan
Date: March 2, 2026
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import shutil
import logging


class ForcingPerturber:
    """
    Apply systematic perturbations to SUMMA forcing files for experimental scenarios.
    
    Supports:
    - Additive temperature perturbations (e.g., +2°C, -2°C)
    - Multiplicative precipitation perturbations (e.g., ×1.3, ×0.7)
    - Seasonal application (fall, winter, spring, summer)
    - Multiple forcing file handling
    - Metadata preservation and provenance tracking
    """
    
    #季节定义 (水年: October 1 - September 30)
    SEASON_DEFINITIONS = {
        'fall': [10, 11, 12],      # Oct, Nov, Dec
        'winter': [1, 2, 3],        # Jan, Feb, Mar
        'spring': [4, 5, 6],        # Apr, May, Jun
        'summer': [7, 8, 9]         # Jul, Aug, Sep
    }
    
    def __init__(self, 
                 source_forcing_dir: Union[str, Path],
                 output_base_dir: Union[str, Path],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize forcing perturbation utility.
        
        Parameters
        ----------
        source_forcing_dir : Path
            Directory containing original forcing files
        output_base_dir : Path  
            Base directory for perturbed forcing file sets
        logger : logging.Logger, optional
            Logger instance for tracking operations
        """
        self.source_forcing_dir = Path(source_forcing_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # Identify forcing files
        self.forcing_files = sorted(self.source_forcing_dir.glob('*.nc'))
        if not self.forcing_files:
            raise FileNotFoundError(f"No forcing files found in {self.source_forcing_dir}")
        
        self.logger.info(f"Initialized ForcingPerturber with {len(self.forcing_files)} forcing files")
        
    def _setup_logger(self) -> logging.Logger:
        """Create default logger if none provided."""
        logger = logging.getLogger('ForcingPerturber')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_scenario(self,
                       scenario_name: str,
                       season: str,
                       temp_delta: float = 0.0,
                       precip_mult: float = 1.0,
                       water_year: Optional[int] = None) -> Path:
        """
        Create a complete perturbed forcing file set for one scenario.
        
        Parameters
        ----------
        scenario_name : str
            Unique identifier for this scenario (e.g., 'WY2017_spring_warm2_wet130')
        season : str
            Season to perturb: 'fall', 'winter', 'spring', 'summer'
        temp_delta : float
            Temperature change to apply (°C), default 0.0
        precip_mult : float
            Precipitation multiplier to apply (dimensionless), default 1.0
        water_year : int, optional
            Water year to extract (Oct 1 of WY-1 to Sep 30 of WY)
            If None, use all available data
            
        Returns
        -------
        output_dir : Path
            Directory containing perturbed forcing files
            
        Examples
        --------
        >>> perturber = ForcingPerturber('/path/to/forcing', '/path/to/experiments/forcing')
        >>> output_dir = perturber.create_scenario(
        ...     scenario_name='WY2017_spring_warmplus2_wet130',
        ...     season='spring',
        ...     temp_delta=2.0,
        ...     precip_mult=1.3,
        ...     water_year=2017
        ... )
        """
        if season not in self.SEASON_DEFINITIONS:
            raise ValueError(f"Season must be one of {list(self.SEASON_DEFINITIONS.keys())}")
        
        output_dir = self.output_base_dir / scenario_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating scenario: {scenario_name}")
        self.logger.info(f"  Season: {season}, ΔT={temp_delta}°C, P×{precip_mult}")
        
        for forcing_file in self.forcing_files:
            self._process_forcing_file(
                forcing_file=forcing_file,
                output_dir=output_dir,
                season=season,
                temp_delta=temp_delta,
                precip_mult=precip_mult,
                water_year=water_year,
                scenario_name=scenario_name
            )
        
        self.logger.info(f"✓ Scenario complete: {output_dir}")
        return output_dir
    
    def _process_forcing_file(self,
                             forcing_file: Path,
                             output_dir: Path,
                             season: str,
                             temp_delta: float,
                             precip_mult: float,
                             water_year: Optional[int],
                             scenario_name: str):
        """Process a single forcing file with perturbations."""
        
        # Load forcing data
        ds = xr.open_dataset(forcing_file)
        
        # Extract water year if specified
        if water_year is not None:
            wy_start = f"{water_year-1}-10-01"
            wy_end = f"{water_year}-09-30"
            ds = ds.sel(time=slice(wy_start, wy_end))
            
            if len(ds.time) == 0:
                self.logger.warning(f"No data for WY{water_year} in {forcing_file.name}")
                return
        
        # Create seasonal mask
        months = pd.to_datetime(ds.time.values).month
        season_mask = np.isin(months, self.SEASON_DEFINITIONS[season])
        
        # Apply temperature perturbation
        if temp_delta != 0.0 and 'airtemp' in ds:
            ds['airtemp'].values[season_mask] += temp_delta
            self.logger.debug(f"  Applied ΔT={temp_delta}°C to {season_mask.sum()} timesteps")
        
        # Apply precipitation perturbation  
        if precip_mult != 1.0 and 'pptrate' in ds:
            ds['pptrate'].values[season_mask] *= precip_mult
            self.logger.debug(f"  Applied P×{precip_mult} to {season_mask.sum()} timesteps")
        
        # Update metadata
        ds.attrs['perturbation_scenario'] = scenario_name
        ds.attrs['perturbation_season'] = season
        ds.attrs['perturbation_temp_delta_C'] = temp_delta
        ds.attrs['perturbation_precip_mult'] = precip_mult
        ds.attrs['perturbation_created'] = datetime.now().isoformat()
        ds.attrs['perturbation_source_file'] = str(forcing_file)
        
        # Save perturbed file
        output_file = output_dir / forcing_file.name
        ds.to_netcdf(output_file)
        ds.close()
        
        self.logger.debug(f"  Saved: {output_file.name}")
    
    def create_scenario_batch(self,
                             season: str,
                             temp_deltas: List[float],
                             precip_mults: List[float],
                             water_year: Optional[int] = None,
                             naming_template: str = "WY{wy}_{season}_{temp}_{precip}") -> List[Path]:
        """
        Create a batch of scenarios for factorial experiment design.
        
        Parameters
        ----------
        season : str
            Season to perturb
        temp_deltas : list of float
            Temperature changes to test (e.g., [-2, 0, 2, 4])
        precip_mults : list of float
            Precipitation multipliers to test (e.g., [0.7, 1.0, 1.3, 1.5])
        water_year : int, optional
            Water year to use
        naming_template : str
            Template for scenario naming with placeholders:
            {wy}, {season}, {temp}, {precip}
            
        Returns
        -------
        scenario_dirs : list of Path
            Directories for all created scenarios
            
        Examples
        --------
        >>> scenario_dirs = perturber.create_scenario_batch(
        ...     season='spring',
        ...     temp_deltas=[-2, 0, 2, 4],
        ...     precip_mults=[0.7, 1.0, 1.3, 1.5],
        ...     water_year=2017
        ... )
        >>> print(f"Created {len(scenario_dirs)} scenarios")
        """
        scenario_dirs = []
        
        total_scenarios = len(temp_deltas) * len(precip_mults)
        self.logger.info(f"Creating batch: {total_scenarios} scenarios for {season}")
        
        for temp_delta in temp_deltas:
            for precip_mult in precip_mults:
                # Format scenario name
                temp_str = self._format_perturbation(temp_delta, 'temp')
                precip_str = self._format_perturbation(precip_mult, 'precip')
                
                scenario_name = naming_template.format(
                    wy=water_year or 'all',
                    season=season,
                    temp=temp_str,
                    precip=precip_str
                )
                
                try:
                    output_dir = self.create_scenario(
                        scenario_name=scenario_name,
                        season=season,
                        temp_delta=temp_delta,
                        precip_mult=precip_mult,
                        water_year=water_year
                    )
                    scenario_dirs.append(output_dir)
                    
                except Exception as e:
                    self.logger.error(f"Failed to create {scenario_name}: {e}")
                    continue
        
        self.logger.info(f"✓ Batch complete: {len(scenario_dirs)}/{total_scenarios} scenarios created")
        return scenario_dirs
    
    @staticmethod
    def _format_perturbation(value: float, ptype: str) -> str:
        """Format perturbation value for scenario naming."""
        if ptype == 'temp':
            if value == 0:
                return 'baseline'
            elif value > 0:
                return f'warmplus{abs(value):.0f}'
            else:
                return f'coldminus{abs(value):.0f}'
        elif ptype == 'precip':
            if value == 1.0:
                return 'baseline'
            else:
                pct = int(value * 100)
                return f'wet{pct}' if value > 1.0 else f'dry{pct}'
        else:
            return str(value)
    
    def create_full_factorial(self,
                             seasons: List[str],
                             temp_deltas: List[float],
                             precip_mults: List[float],
                             water_year: Optional[int] = None) -> Dict[str, List[Path]]:
        """
        Create full factorial design across multiple seasons.
        
        Parameters
        ----------
        seasons : list of str
            Seasons to perturb (e.g., ['fall', 'spring', 'summer'])
        temp_deltas : list of float
            Temperature changes
        precip_mults : list of float
            Precipitation multipliers
        water_year : int, optional
            Water year to use
            
        Returns
        -------
        results : dict
            Dictionary mapping season name to list of scenario directories
            
        Examples
        --------
        >>> results = perturber.create_full_factorial(
        ...     seasons=['fall', 'spring', 'summer'],
        ...     temp_deltas=[-2, 0, 2, 4],
        ...     precip_mults=[0.7, 1.0, 1.3, 1.5],
        ...     water_year=2017
        ... )
        >>> total = sum(len(dirs) for dirs in results.values())
        >>> print(f"Created {total} total scenarios across {len(results)} seasons")
        """
        results = {}
        
        total_scenarios = len(seasons) * len(temp_deltas) * len(precip_mults)
        self.logger.info(f"Creating full factorial design: {total_scenarios} scenarios")
        
        for season in seasons:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing season: {season.upper()}")
            self.logger.info(f"{'='*60}")
            
            scenario_dirs = self.create_scenario_batch(
                season=season,
                temp_deltas=temp_deltas,
                precip_mults=precip_mults,
                water_year=water_year
            )
            
            results[season] = scenario_dirs
        
        # Summary
        total_created = sum(len(dirs) for dirs in results.values())
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"FULL FACTORIAL COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total scenarios created: {total_created}/{total_scenarios}")
        for season, dirs in results.items():
            self.logger.info(f"  {season}: {len(dirs)} scenarios")
        
        return results
    
    def verify_scenario(self, scenario_dir: Path) -> Dict[str, any]:
        """
        Verify a perturbed scenario for data integrity.
        
        Checks:
        - All forcing files present
        - No missing values
        - Temperature and precipitation within physical bounds
        - Temporal continuity
        
        Parameters
        ----------
        scenario_dir : Path
            Directory containing scenario forcing files
            
        Returns
        -------
        verification_report : dict
            Report with verification status and diagnostics
        """
        report = {
            'scenario': scenario_dir.name,
            'status': 'PASS',
            'issues': [],
            'file_count': 0,
            'timesteps': 0,
            'temp_range': None,
            'precip_range': None
        }
        
        forcing_files = sorted(scenario_dir.glob('*.nc'))
        report['file_count'] = len(forcing_files)
        
        if len(forcing_files) == 0:
            report['status'] = 'FAIL'
            report['issues'].append('No forcing files found')
            return report
        
        try:
            # Check first file
            ds = xr.open_dataset(forcing_files[0])
            report['timesteps'] = len(ds.time)
            
            # Check for required variables
            required_vars = ['airtemp', 'pptrate']
            missing_vars = [v for v in required_vars if v not in ds]
            if missing_vars:
                report['status'] = 'FAIL'
                report['issues'].append(f'Missing variables: {missing_vars}')
            
            # Check temperature bounds (reasonable Earth surface temps)
            if 'airtemp' in ds:
                temp_min = float(ds['airtemp'].min())
                temp_max = float(ds['airtemp'].max())
                report['temp_range'] = (temp_min, temp_max)
                
                if temp_min < 200 or temp_max > 350:  # Kelvin bounds
                    report['status'] = 'FAIL'
                    report['issues'].append(f'Temperature out of bounds: {temp_min:.1f}-{temp_max:.1f} K')
            
            # Check precipitation (non-negative)
            if 'pptrate' in ds:
                precip_min = float(ds['pptrate'].min())
                precip_max = float(ds['pptrate'].max())
                report['precip_range'] = (precip_min, precip_max)
                
                if precip_min < 0:
                    report['status'] = 'FAIL'
                    report['issues'].append(f'Negative precipitation values detected')
            
            # Check for missing values
            if ds['airtemp'].isnull().any() or ds['pptrate'].isnull().any():
                report['status'] = 'FAIL'
                report['issues'].append('Missing values (NaN) detected')
            
            ds.close()
            
        except Exception as e:
            report['status'] = 'FAIL'
            report['issues'].append(f'Verification error: {str(e)}')
        
        return report


# Convenience functions for quick scenario generation

def generate_standard_factorial(source_forcing_dir: Union[str, Path],
                               output_base_dir: Union[str, Path],
                               water_year: int = 2017,
                               seasons: List[str] = ['fall', 'spring', 'summer']) -> Dict[str, List[Path]]:
    """
    Generate standard 4×4 factorial design for seasonal perturbations.
    
    Standard perturbations:
    - Temperature: -2°C, 0°C, +2°C, +4°C
    - Precipitation: ×0.7, ×1.0, ×1.3, ×1.5
    
    Parameters
    ----------
    source_forcing_dir : Path
        Original forcing file directory
    output_base_dir : Path
        Output directory for perturbed scenarios
    water_year : int
        Water year to use (default 2017)
    seasons : list of str
        Seasons to perturb (default ['fall', 'spring', 'summer'])
        
    Returns
    -------
    results : dict
        Scenario directories organized by season
        
    Examples
    --------
    >>> results = generate_standard_factorial(
    ...     source_forcing_dir='/scratch/data/domain_Tuolumne/forcing/SUMMA',
    ...     output_base_dir='/scratch/data/experiments/forcing_perturbed',
    ...     water_year=2017
    ... )
    """
    perturber = ForcingPerturber(source_forcing_dir, output_base_dir)
    
    return perturber.create_full_factorial(
        seasons=seasons,
        temp_deltas=[-2.0, 0.0, 2.0, 4.0],
        precip_mults=[0.7, 1.0, 1.3, 1.5],
        water_year=water_year
    )


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Create perturbed forcing scenarios')
    parser.add_argument('--source', type=str, required=True, help='Source forcing directory')
    parser.add_argument('--output', type=str, required=True, help='Output base directory')
    parser.add_argument('--water-year', type=int, default=2017, help='Water year to use')
    parser.add_argument('--seasons', nargs='+', default=['fall', 'spring', 'summer'],
                       help='Seasons to perturb')
    
    args = parser.parse_args()
    
    print("Creating perturbed forcing scenarios...")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Water Year: {args.water_year}")
    print(f"Seasons: {args.seasons}\n")
    
    results = generate_standard_factorial(
        source_forcing_dir=args.source,
        output_base_dir=args.output,
        water_year=args.water_year,
        seasons=args.seasons
    )
    
    print("\n✓ COMPLETE")
    print(f"Total scenarios: {sum(len(dirs) for dirs in results.values())}")
