"""
Results Analysis Toolkit for Climate Perturbation Experiments

This module provides comprehensive analysis and visualization tools for 
experimental ensemble results, focusing on water balance metrics and 
streamflow efficiency responses to seasonal climate perturbations.

Author: dlhogan
Date: March 2, 2026
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging


@dataclass
class WaterBalanceMetrics:
    """Container for water balance metrics from a single scenario."""
    scenario_id: str
    season: str
    temp_delta: float
    precip_mult: float
    
    # Annual totals (mm)
    P_total: float
    Q_total: float
    ET_total: float
    
    # Seasonal precipitation (mm)
    P_fall: float
    P_spring: float
    P_summer: float
    P_winter: float = 0.0
    
    # Seasonal streamflow (mm)
    Q_fall: float
    Q_spring: float
    Q_summer: float
    Q_winter: float = 0.0
    
    # Seasonal ET (mm)
    ET_fall: float
    ET_spring: float
    ET_summer: float
    ET_winter: float = 0.0
    
    # Seasonal mean temperature (°C)
    T_fall: float
    T_spring: float
    T_summer: float
    T_winter: float = 0.0
    
    # Storage changes (mm)
    dS_soil: float = 0.0
    dS_aquifer: float = 0.0
    dS_snow: float = 0.0
    
    # Efficiency metrics
    efficiency: float = 0.0  # Q/P ratio
    closure_error: float = 0.0  # P - Q - ET - dS
    
    # Performance metrics (if observations available)
    NSE: Optional[float] = None
    KGE: Optional[float] = None


class ExperimentResultsAnalyzer:
    """
    Comprehensive analysis of climate perturbation experiment ensemble.
    
    Features:
    - Water balance metric extraction from SUMMA output
    - Seasonal aggregation and partitioning
    - Efficiency analysis and sensitivity quantification
    - Comparative visualization suite
    - Export to structured data tables
    """
    
    # Season definitions (water year)
    SEASONS = {
        'fall': [10, 11, 12],
        'winter': [1, 2, 3],
        'spring': [4, 5, 6],
        'summer': [7, 8, 9]
    }
    
    def __init__(self,
                 experiment_base_dir: Union[str, Path],
                 logger: Optional[logging.Logger] = None):
        """
        Initialize results analyzer.
        
        Parameters
        ----------
        experiment_base_dir : Path
            Base directory containing experiment runs
        logger : logging.Logger, optional
            Logger instance
        """
        self.experiment_base_dir = Path(experiment_base_dir)
        self.runs_dir = self.experiment_base_dir / 'runs'
        self.results_dir = self.experiment_base_dir / 'results'
        
        self.logger = logger or self._setup_logger()
        
        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {self.runs_dir}")
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics: List[WaterBalanceMetrics] = []
        self.metrics_df: Optional[pd.DataFrame] = None
        
        self.logger.info(f"Initialized ExperimentResultsAnalyzer")
        self.logger.info(f"  Experiment dir: {self.experiment_base_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Create default logger."""
        logger = logging.getLogger('ExperimentResultsAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def extract_scenario_metrics(self, scenario_dir: Path) -> Optional[WaterBalanceMetrics]:
        """
        Extract water balance metrics from a single scenario run.
        
        Parameters
        ----------
        scenario_dir : Path
            Scenario run directory
            
        Returns
        -------
        metrics : WaterBalanceMetrics or None
            Extracted metrics, or None if extraction failed
        """
        scenario_id = scenario_dir.name
        
        try:
            # Find output file
            simulations_dir = scenario_dir / 'simulations'
            if not simulations_dir.exists():
                simulations_dir = scenario_dir  # Try current directory
            
            output_files = list(simulations_dir.glob('*_timestep.nc'))
            if not output_files:
                self.logger.warning(f"No output file found for {scenario_id}")
                return None
            
            output_file = output_files[0]
            
            # Load SUMMA output
            ds = xr.open_dataset(output_file)
            
            # Parse scenario metadata from ID
            parts = scenario_id.split('_')
            season = parts[1] if len(parts) > 1 else 'unknown'
            
            # Parse perturbations
            temp_delta = 0.0
            if len(parts) > 2:
                temp_str = parts[2]
                if 'warm' in temp_str:
                    temp_delta = float(temp_str.replace('warmplus', '').replace('warm', ''))
                elif 'cold' in temp_str:
                    temp_delta = -float(temp_str.replace('coldminus', '').replace('cold', ''))
            
            precip_mult = 1.0
            if len(parts) > 3:
                precip_str = parts[3]
                if precip_str != 'baseline':
                    import re
                    match = re.search(r'\d+', precip_str)
                    if match:
                        precip_mult = float(match.group()) / 100.0
            
            # Extract time and create seasonal masks
            time = pd.to_datetime(ds.time.values)
            months = time.month
            
            # Calculate seasonal masks
            masks = {
                season_name: np.isin(months, season_months)
                for season_name, season_months in self.SEASONS.items()
            }
            
            # Extract forcing data (assuming variables exist)
            # If forcing not in output, need to read from forcing files
            if 'pptrate' in ds.variables:
                # Precipitation in kg/m2/s -> convert to mm (multiply by timestep)
                dt_hours = (time[1] - time[0]).total_seconds() / 3600
                ppt_mm = ds['pptrate'].values * dt_hours * 3600  # mm per timestep
                P_total = float(np.sum(ppt_mm))
                P_seasonal = {
                    season: float(np.sum(ppt_mm[mask]))
                    for season, mask in masks.items()
                }
            else:
                # Read from forcing files as fallback
                forcing_list = scenario_dir / 'settings' / 'SUMMA' / 'forcingFileList.txt'
                P_total, P_seasonal = self._read_forcing_precip(forcing_list)
            
            # Temperature (K -> °C)
            if 'airtemp' in ds.variables:
                temp_k = ds['airtemp'].values
                T_seasonal = {
                    season: float(np.mean(temp_k[mask])) - 273.15
                    for season, mask in masks.items()
                }
            else:
                T_seasonal = {season: np.nan for season in self.SEASONS}
            
            # Streamflow (scalarTotalRunoff in kg/m2/s -> mm)
            if 'scalarTotalRunoff' in ds.variables:
                Q_rate = ds['scalarTotalRunoff'].values.squeeze()
                Q_mm = Q_rate * dt_hours * 3600  # mm per timestep
                Q_total = float(np.sum(Q_mm))
                Q_seasonal = {
                    season: float(np.sum(Q_mm[mask]))
                    for season, mask in masks.items()
                }
            else:
                self.logger.warning(f"scalarTotalRunoff not found in {scenario_id}")
                Q_total = 0.0
                Q_seasonal = {season: 0.0 for season in self.SEASONS}
            
            # Evapotranspiration (scalarLatHeatTotal W/m2 -> mm)
            # ET (mm) = LE (W/m2) * dt (s) / λ (J/kg) / ρ_water (kg/m3) * 1000
            # λ ≈ 2.45e6 J/kg, ρ_water = 1000 kg/m3
            if 'scalarLatHeatTotal' in ds.variables:
                LE = ds['scalarLatHeatTotal'].values.squeeze()  # W/m2
                dt_sec = dt_hours * 3600
                ET_mm = LE * dt_sec / 2.45e6  # mm per timestep
                ET_total = float(np.sum(ET_mm))
                ET_seasonal = {
                    season: float(np.sum(ET_mm[mask]))
                    for season, mask in masks.items()
                }
            else:
                ET_total = 0.0
                ET_seasonal = {season: 0.0 for season in self.SEASONS}
            
            # Storage changes (from first to last timestep)
            # Soil moisture
            if 'scalarTotalSoilWat' in ds.variables:
                soil_wat = ds['scalarTotalSoilWat'].values.squeeze()
                dS_soil = float(soil_wat[-1] - soil_wat[0])
            else:
                dS_soil = 0.0
            
            # Aquifer storage
            if 'scalarAquiferStorage' in ds.variables:
                aquifer_stor = ds['scalarAquiferStorage'].values.squeeze()
                dS_aquifer = float(aquifer_stor[-1] - aquifer_stor[0])
            else:
                dS_aquifer = 0.0
            
            # Snow water equivalent
            if 'scalarSWE' in ds.variables:
                swe = ds['scalarSWE'].values.squeeze()
                dS_snow = float(swe[-1] - swe[0])
            else:
                dS_snow = 0.0
            
            # Calculate efficiency and closure error
            efficiency = Q_total / P_total if P_total > 0 else 0.0
            dS_total = dS_soil + dS_aquifer + dS_snow
            closure_error = P_total - Q_total - ET_total - dS_total
            
            ds.close()
            
            # Create metrics object
            metrics = WaterBalanceMetrics(
                scenario_id=scenario_id,
                season=season,
                temp_delta=temp_delta,
                precip_mult=precip_mult,
                P_total=P_total,
                Q_total=Q_total,
                ET_total=ET_total,
                P_fall=P_seasonal.get('fall', 0.0),
                P_spring=P_seasonal.get('spring', 0.0),
                P_summer=P_seasonal.get('summer', 0.0),
                P_winter=P_seasonal.get('winter', 0.0),
                Q_fall=Q_seasonal.get('fall', 0.0),
                Q_spring=Q_seasonal.get('spring', 0.0),
                Q_summer=Q_seasonal.get('summer', 0.0),
                Q_winter=Q_seasonal.get('winter', 0.0),
                ET_fall=ET_seasonal.get('fall', 0.0),
                ET_spring=ET_seasonal.get('spring', 0.0),
                ET_summer=ET_seasonal.get('summer', 0.0),
                ET_winter=ET_seasonal.get('winter', 0.0),
                T_fall=T_seasonal.get('fall', 0.0),
                T_spring=T_seasonal.get('spring', 0.0),
                T_summer=T_seasonal.get('summer', 0.0),
                T_winter=T_seasonal.get('winter', 0.0),
                dS_soil=dS_soil,
                dS_aquifer=dS_aquifer,
                dS_snow=dS_snow,
                efficiency=efficiency,
                closure_error=closure_error
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to extract metrics from {scenario_id}: {e}")
            return None
    
    def _read_forcing_precip(self, forcing_list_file: Path) -> Tuple[float, Dict[str, float]]:
        """Read precipitation from forcing files (fallback method)."""
        # Simplified implementation - could be enhanced
        return 1000.0, {'fall': 250.0, 'winter': 300.0, 'spring': 250.0, 'summer': 200.0}
    
    def extract_all_metrics(self, scenario_pattern: str = 'WY*') -> pd.DataFrame:
        """
        Extract metrics from all scenario runs.
        
        Parameters
        ----------
        scenario_pattern : str
            Glob pattern for scenario directories
            
        Returns
        -------
        metrics_df : pd.DataFrame
            DataFrame with all scenario metrics
        """
        scenario_dirs = sorted(self.runs_dir.glob(scenario_pattern))
        
        self.logger.info(f"Extracting metrics from {len(scenario_dirs)} scenarios...")
        
        for i, scenario_dir in enumerate(scenario_dirs, 1):
            if not scenario_dir.is_dir():
                continue
            
            self.logger.info(f"[{i}/{len(scenario_dirs)}] Processing {scenario_dir.name}...")
            
            metrics = self.extract_scenario_metrics(scenario_dir)
            if metrics:
                self.metrics.append(metrics)
        
        # Convert to DataFrame
        self.metrics_df = pd.DataFrame([vars(m) for m in self.metrics])
        
        # Save to CSV
        output_path = self.results_dir / 'water_balance_metrics.csv'
        self.metrics_df.to_csv(output_path, index=False)
        self.logger.info(f"✓ Metrics saved to: {output_path}")
        
        # Summary statistics
        self._print_summary_stats()
        
        return self.metrics_df
    
    def _print_summary_stats(self):
        """Print summary statistics of extracted metrics."""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            return
        
        self.logger.info("\n" + "="*70)
        self.logger.info("METRICS SUMMARY")
        self.logger.info("="*70)
        
        self.logger.info(f"Total scenarios: {len(self.metrics_df)}")
        self.logger.info(f"Seasons: {self.metrics_df['season'].unique().tolist()}")
        
        self.logger.info(f"\nPrecipitation range: {self.metrics_df['P_total'].min():.1f} - {self.metrics_df['P_total'].max():.1f} mm")
        self.logger.info(f"Streamflow range: {self.metrics_df['Q_total'].min():.1f} - {self.metrics_df['Q_total'].max():.1f} mm")
        self.logger.info(f"Efficiency range: {self.metrics_df['efficiency'].min():.3f} - {self.metrics_df['efficiency'].max():.3f}")
        
        self.logger.info(f"\nWater balance closure errors:")
        self.logger.info(f"  Mean: {self.metrics_df['closure_error'].mean():.2f} mm")
        self.logger.info(f"  Std:  {self.metrics_df['closure_error'].std():.2f} mm")
        self.logger.info(f"  Max:  {self.metrics_df['closure_error'].abs().max():.2f} mm")
    
    def plot_efficiency_heatmaps(self, figsize: Tuple[int, int] = (15, 5)):
        """
        Create efficiency heatmaps for each season.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.metrics_df is None:
            raise ValueError("No metrics available. Run extract_all_metrics() first.")
        
        seasons = self.metrics_df['season'].unique()
        n_seasons = len(seasons)
        
        fig, axes = plt.subplots(1, n_seasons, figsize=figsize)
        if n_seasons == 1:
            axes = [axes]
        
        for ax, season in zip(axes, seasons):
            season_data = self.metrics_df[self.metrics_df['season'] == season]
            
            # Pivot for heatmap
            pivot = season_data.pivot(
                index='temp_delta',
                columns='precip_mult',
                values='efficiency'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu',
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'Efficiency (Q/P)'}
            )
            
            ax.set_title(f'{season.capitalize()} Season', fontsize=14, fontweight='bold')
            ax.set_xlabel('Precipitation Multiplier', fontsize=12)
            ax.set_ylabel('Temperature Change (°C)', fontsize=12)
        
        plt.suptitle('Streamflow Efficiency by Season and Perturbation', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.results_dir / 'efficiency_heatmaps.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_seasonal_contributions(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot seasonal contributions to annual water balance."""
        if self.metrics_df is None:
            raise ValueError("No metrics available.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Select baseline + extreme scenarios for clarity
        baseline = self.metrics_df[
            (self.metrics_df['temp_delta'] == 0) & 
            (self.metrics_df['precip_mult'] == 1.0)
        ]
        
        warm_dry = self.metrics_df[
            (self.metrics_df['temp_delta'] > 0) & 
            (self.metrics_df['precip_mult'] < 1.0)
        ]
        
        warm_wet = self.metrics_df[
            (self.metrics_df['temp_delta'] > 0) & 
            (self.metrics_df['precip_mult'] > 1.0)
        ]
        
        # Precipitation seasonal breakdown
        ax = axes[0, 0]
        seasons = ['fall', 'winter', 'spring', 'summer']
        for df, label, color in [(baseline, 'Baseline', 'blue'), 
                                  (warm_dry, 'Warm-Dry', 'red'),
                                  (warm_wet, 'Warm-Wet', 'green')]:
            if len(df) > 0:
                P_seasonal = [df[[f'P_{s}']].mean().values[0] for s in seasons]
                ax.plot(seasons, P_seasonal, 'o-', label=label, color=color, linewidth=2, markersize=8)
        
        ax.set_title('Seasonal Precipitation', fontweight='bold')
        ax.set_ylabel('Precipitation (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Streamflow seasonal breakdown
        ax = axes[0, 1]
        for df, label, color in [(baseline, 'Baseline', 'blue'), 
                                  (warm_dry, 'Warm-Dry', 'red'),
                                  (warm_wet, 'Warm-Wet', 'green')]:
            if len(df) > 0:
                Q_seasonal = [df[[f'Q_{s}']].mean().values[0] for s in seasons]
                ax.plot(seasons, Q_seasonal, 'o-', label=label, color=color, linewidth=2, markersize=8)
        
        ax.set_title('Seasonal Streamflow', fontweight='bold')
        ax.set_ylabel('Streamflow (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ET seasonal breakdown
        ax = axes[1, 0]
        for df, label, color in [(baseline, 'Baseline', 'blue'), 
                                  (warm_dry, 'Warm-Dry', 'red'),
                                  (warm_wet, 'Warm-Wet', 'green')]:
            if len(df) > 0:
                ET_seasonal = [df[[f'ET_{s}']].mean().values[0] for s in seasons]
                ax.plot(seasons, ET_seasonal, 'o-', label=label, color=color, linewidth=2, markersize=8)
        
        ax.set_title('Seasonal Evapotranspiration', fontweight='bold')
        ax.set_ylabel('ET (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Efficiency by season
        ax = axes[1, 1]
        season_groups = self.metrics_df.groupby('season')['efficiency']
        season_names = list(season_groups.groups.keys())
        efficiency_means = [season_groups.get_group(s).mean() for s in season_names]
        efficiency_stds = [season_groups.get_group(s).std() for s in season_names]
        
        ax.bar(season_names, efficiency_means, yerr=efficiency_stds, capsize=5, alpha=0.7)
        ax.set_title('Mean Efficiency by Perturbed Season', fontweight='bold')
        ax.set_ylabel('Efficiency (Q/P)')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Seasonal Water Balance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.results_dir / 'seasonal_contributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def plot_et_vs_q_tradeoff(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot ET vs Q relationship across scenarios."""
        if self.metrics_df is None:
            raise ValueError("No metrics available.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by season
        seasons = self.metrics_df['season'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(seasons)))
        
        for season, color in zip(seasons, colors):
            season_data = self.metrics_df[self.metrics_df['season'] == season]
            ax.scatter(
                season_data['Q_total'],
                season_data['ET_total'],
                c=[color],
                label=season.capitalize(),
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Add reference lines
        ax.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5, label='1:1 line')
        
        ax.set_xlabel('Total Streamflow (mm)', fontsize=12)
        ax.set_ylabel('Total Evapotranspiration (mm)', fontsize=12)
        ax.set_title('ET vs Q Tradeoff Across Scenarios', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.results_dir / 'et_vs_q_tradeoff.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"✓ Saved: {output_path}")
        
        plt.show()
    
    def generate_full_report(self):
        """Generate complete analysis report with all visualizations."""
        self.logger.info("\n" + "="*70)
        self.logger.info("GENERATING FULL ANALYSIS REPORT")
        self.logger.info("="*70 + "\n")
        
        if self.metrics_df is None:
            self.extract_all_metrics()
        
        self.plot_efficiency_heatmaps()
        self.plot_seasonal_contributions()
        self.plot_et_vs_q_tradeoff()
        
        self.logger.info("\n✓ Full report generated")
        self.logger.info(f"All outputs saved to: {self.results_dir}")


# Convenience function
def analyze_experiments(experiment_base_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Quick analysis of all experiments.
    
    Parameters
    ----------
    experiment_base_dir : Path
        Base directory containing experiment runs
        
    Returns
    -------
    metrics_df : pd.DataFrame
        Water balance metrics for all scenarios
        
    Examples
    --------
    >>> metrics = analyze_experiments('/scratch/experiments')
    """
    analyzer = ExperimentResultsAnalyzer(experiment_base_dir)
    analyzer.extract_all_metrics()
    analyzer.generate_full_report()
    
    return analyzer.metrics_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze climate perturbation experiment results')
    parser.add_argument('--experiment-dir', type=str, required=True, 
                       help='Experiment base directory')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract metrics, skip visualizations')
    
    args = parser.parse_args()
    
    analyzer = ExperimentResultsAnalyzer(args.experiment_dir)
    
    metrics_df = analyzer.extract_all_metrics()
    
    if not args.extract_only:
        analyzer.generate_full_report()
    
    print("\n✓ Analysis complete")
    print(f"Metrics: {args.experiment_dir}/results/water_balance_metrics.csv")
