"""
Interactive SUMMA Optimizer
===========================
A lightweight, notebook-friendly optimizer for SUMMA hydrological models.
Supports live convergence plots, hydrograph comparisons, and easy parameter control.

Usage:
    from interactive_optimizer import InteractiveOptimizer
    opt = InteractiveOptimizer.from_config('path/to/config.yaml')
    opt.run(n_iterations=100)
    opt.plot_best()
"""

import subprocess
import time
import re
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from IPython.display import display, clear_output


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    """One calibration parameter with name, default, and bounds."""
    name: str
    default: float
    lower: float
    upper: float
    kind: str = 'local'  # 'local' or 'basin'

    @property
    def range(self):
        return self.upper - self.lower

    def normalize(self, value: float) -> float:
        if self.range == 0:
            return 0.5
        return (value - self.lower) / self.range

    def denormalize(self, norm: float) -> float:
        return self.lower + norm * self.range


@dataclass
class TrialResult:
    """Stores one evaluation: parameters tried and score obtained."""
    iteration: int
    params: Dict[str, float]
    kge: float = np.nan
    nse: float = np.nan
    pbias: float = np.nan
    rmse: float = np.nan
    r: float = np.nan
    alpha: float = np.nan
    beta: float = np.nan
    runtime_s: float = 0.0
    sim_discharge: Optional[pd.Series] = None
    success: bool = False


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class InteractiveOptimizer:
    """
    Interactive DDS optimizer for SUMMA.

    Reads parameter bounds from localParamInfo.txt / basinParamInfo.txt,
    writes trialParams.nc, runs SUMMA via subprocess, loads output,
    calculates KGE/NSE against observed streamflow, and provides
    live-updating matplotlib plots.

    Parameters
    ----------
    settings_dir : Path
        SUMMA settings directory (contains fileManager.txt, etc.)
    output_dir : Path
        Where SUMMA writes output NetCDF files.
    obs_path : Path
        Preprocessed observed streamflow CSV (datetime, discharge_cms).
    basin_area_m2 : float
        Basin area in m² for unit conversion (m/s -> m³/s).
    params_to_calibrate : list of str
        Parameter names to optimize (from localParamInfo.txt).
    basin_params_to_calibrate : list of str
        Basin parameter names to optimize (from basinParamInfo.txt).
    cal_start, cal_end : str
        Calibration period date strings ('YYYY-MM-DD').
    summa_exe : str
        SUMMA executable path.
    file_manager : str
        fileManager.txt filename.
    dds_r : float
        DDS perturbation parameter (fraction of range, default 0.2).
    metric : str
        Primary metric to optimize: 'KGE' or 'NSE'.
    output_prefix : str
        SUMMA output file prefix.
    """

    def __init__(
        self,
        settings_dir: Path,
        output_dir: Path,
        obs_path: Path,
        basin_area_m2: float,
        params_to_calibrate: List[str],
        basin_params_to_calibrate: List[str],
        cal_start: str,
        cal_end: str,
        summa_exe: str = '/usr/local/bin/summa',
        file_manager: str = 'fileManager.txt',
        dds_r: float = 0.2,
        metric: str = 'KGE',
        output_prefix: str = 'interactive_opt',
    ):
        self.settings_dir = Path(settings_dir)
        self.output_dir = Path(output_dir)
        self.obs_path = Path(obs_path)
        self.basin_area_m2 = basin_area_m2
        self.summa_exe = summa_exe
        self.file_manager = file_manager
        self.dds_r = dds_r
        self.metric = metric.upper()
        self.output_prefix = output_prefix
        self.cal_start = pd.Timestamp(cal_start)
        self.cal_end = pd.Timestamp(cal_end)

        # Parse parameter bounds from SUMMA setting files
        self.param_info: Dict[str, ParamInfo] = {}
        local_file = self.settings_dir / 'localParamInfo.txt'
        basin_file = self.settings_dir / 'basinParamInfo.txt'

        for name in params_to_calibrate:
            pinfo = self._parse_param(local_file, name, kind='local')
            if pinfo:
                self.param_info[name] = pinfo

        for name in basin_params_to_calibrate:
            pinfo = self._parse_param(basin_file, name, kind='basin')
            if pinfo:
                self.param_info[name] = pinfo

        self.param_names = list(self.param_info.keys())
        self.n_params = len(self.param_names)

        # Load observations once
        self.obs = self._load_obs()

        # History
        self.history: List[TrialResult] = []
        self.best_result: Optional[TrialResult] = None

        print(f"InteractiveOptimizer ready")
        print(f"  {self.n_params} parameters: {self.param_names}")
        print(f"  Metric: {self.metric}")
        print(f"  Calibration: {self.cal_start.date()} to {self.cal_end.date()}")
        print(f"  Basin area: {self.basin_area_m2/1e6:.1f} km²")
        print(f"  Obs records: {len(self.obs)} hours")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str, output_prefix: str = 'interactive_opt') -> 'InteractiveOptimizer':
        """
        Build optimizer from a CONFLUENCE YAML config file.

        Automatically resolves all directory paths, parameter lists,
        calibration period, basin area (from shapefile), etc.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_dir = Path(cfg['CONFLUENCE_DATA_DIR'])
        domain = cfg['DOMAIN_NAME']
        project_dir = data_dir / f'domain_{domain}'

        # Settings dir: use the run_dds working copy
        sim_name = cfg.get('EXPERIMENT_ID', 'interactive')
        settings_dir = project_dir / 'simulations' / 'run_dds' / 'settings' / 'SUMMA'
        output_dir = project_dir / 'simulations' / 'run_dds' / 'SUMMA'

        # Observations
        obs_path = (project_dir / 'observations' / 'streamflow' / 'preprocessed' /
                    f'{domain}_streamflow_processed.csv')

        # Basin area from shapefile (use max to skip artifact rows)
        basin_area = cls._get_basin_area(project_dir, cfg)

        # Parameters
        local_params = [p.strip() for p in cfg.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        basin_params = [p.strip() for p in cfg.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]

        # Calibration period
        cal_str = cfg.get('CALIBRATION_PERIOD', '')
        cal_parts = [s.strip() for s in cal_str.split(',')]
        cal_start = cal_parts[0] if len(cal_parts) >= 1 else '2018-10-01'
        cal_end = cal_parts[1] if len(cal_parts) >= 2 else '2020-09-30'

        # SUMMA exe
        summa_install = cfg.get('SUMMA_INSTALL_PATH', '/usr/local/bin')
        summa_exe_name = cfg.get('SUMMA_EXE', 'summa')
        summa_exe = f"{summa_install}/{summa_exe_name}"

        return cls(
            settings_dir=settings_dir,
            output_dir=output_dir,
            obs_path=obs_path,
            basin_area_m2=basin_area,
            params_to_calibrate=local_params,
            basin_params_to_calibrate=basin_params,
            cal_start=cal_start,
            cal_end=cal_end,
            summa_exe=summa_exe,
            file_manager=cfg.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt'),
            dds_r=cfg.get('DDS_R', 0.2),
            metric=cfg.get('OPTIMIZATION_METRIC', 'KGE'),
            output_prefix=output_prefix,
        )

    @staticmethod
    def _get_basin_area(project_dir: Path, cfg: dict) -> float:
        """Get basin area in m² from shapefile, skipping artifact rows."""
        try:
            import geopandas as gpd
            shp_dir = project_dir / 'shapefiles' / 'river_basins'
            shp_files = list(shp_dir.glob('*.shp'))
            if shp_files:
                gdf = gpd.read_file(str(shp_files[0]))
                area_col = cfg.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                if area_col in gdf.columns:
                    # Filter out artifact rows (< 1 km²)
                    real = gdf[gdf[area_col] > 1e6]
                    if not real.empty:
                        return float(real[area_col].sum())
                    return float(gdf[area_col].max())
        except Exception as e:
            print(f"Warning: Could not read basin area from shapefile: {e}")

        # Fallback
        return cfg.get('BASIN_AREA_M2', 774_508_623.0)

    # ------------------------------------------------------------------
    # Parameter I/O
    # ------------------------------------------------------------------

    def _parse_param(self, filepath: Path, name: str, kind: str = 'local') -> Optional[ParamInfo]:
        """Parse one parameter's default/bounds from a pipe-delimited SUMMA param file."""
        if not filepath.exists():
            print(f"  Warning: {filepath.name} not found")
            return None

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('!') or line.startswith("'") or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4 and parts[0] == name:
                    def to_float(s):
                        return float(s.replace('d', 'e').replace('D', 'E'))
                    try:
                        default = to_float(parts[1])
                        lower = to_float(parts[2])
                        upper = to_float(parts[3])
                        # Ensure lower < upper
                        if lower > upper:
                            lower, upper = upper, lower
                        if lower == upper:
                            upper = lower * 1.1 + 1e-10
                        return ParamInfo(name=name, default=default,
                                         lower=lower, upper=upper, kind=kind)
                    except ValueError:
                        print(f"  Warning: Could not parse bounds for {name}")
                        return None

        print(f"  Warning: Parameter '{name}' not found in {filepath.name}")
        return None

    def _write_trial_params(self, params: Dict[str, float]):
        """Write parameter values to trialParams.nc for SUMMA."""
        trial_path = self.settings_dir / 'trialParams.nc'
        attr_path = self.settings_dir / 'attributes.nc'

        # Read HRU/GRU IDs from attributes
        with netCDF4.Dataset(str(attr_path), 'r') as attr_ds:
            hru_ids = attr_ds.variables['hruId'][:].copy()
            gru_ids = attr_ds.variables['gruId'][:].copy()
            n_hru = len(hru_ids)
            n_gru = len(gru_ids)

        # Create trialParams.nc
        if trial_path.exists():
            trial_path.unlink()

        with netCDF4.Dataset(str(trial_path), 'w', format='NETCDF4') as ds:
            ds.createDimension('hru', n_hru)
            ds.createDimension('gru', n_gru)

            hru_var = ds.createVariable('hruId', 'f8', ('hru',))
            hru_var[:] = hru_ids
            gru_var = ds.createVariable('gruId', 'f8', ('gru',))
            gru_var[:] = gru_ids

            for pname, pval in params.items():
                info = self.param_info.get(pname)
                if info is None:
                    continue
                if info.kind == 'basin' or pname.startswith('basin__'):
                    var = ds.createVariable(pname, 'f8', ('gru',))
                    var[:] = np.full(n_gru, pval)
                else:
                    var = ds.createVariable(pname, 'f8', ('hru',))
                    var[:] = np.full(n_hru, pval)

    def _update_file_manager_prefix(self):
        """Set the output file prefix in fileManager.txt."""
        fm_path = self.settings_dir / self.file_manager
        if not fm_path.exists():
            return

        text = fm_path.read_text()
        new_text = re.sub(
            r"(outFilePrefix\s+')[^']*(')",
            rf"\g<1>{self.output_prefix}\2",
            text,
        )
        fm_path.write_text(new_text)

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    def _run_summa(self) -> bool:
        """Run SUMMA as subprocess. Returns True on success."""
        fm_path = self.settings_dir / self.file_manager
        cmd = f"{self.summa_exe} -m {fm_path}"

        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'interactive_opt.log'

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, shell=True,
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour timeout
                )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("  SUMMA timed out (>2h)")
            return False
        except Exception as e:
            print(f"  SUMMA error: {e}")
            return False

    def _load_sim_discharge(self) -> Optional[pd.Series]:
        """Load simulated discharge from SUMMA output, convert m/s -> m³/s."""
        # Find the output file
        pattern = f"{self.output_prefix}*_timestep.nc"
        output_files = sorted(self.output_dir.glob(pattern))
        if not output_files:
            print(f"  No output matching {pattern} in {self.output_dir}")
            return None

        out_file = output_files[-1]

        try:
            with netCDF4.Dataset(str(out_file), 'r') as ds:
                n_time = ds.dimensions['time'].size
                if n_time == 0:
                    print("  Output has 0 timesteps")
                    return None

                # Read time
                time_var = ds.variables['time']
                times = netCDF4.num2date(time_var[:], time_var.units,
                                         calendar=getattr(time_var, 'calendar', 'standard'))
                times = pd.DatetimeIndex([pd.Timestamp(t) for t in times])

                # Read runoff (m/s) and convert to m³/s
                runoff = ds.variables['scalarTotalRunoff'][:, 0]
                discharge = runoff * self.basin_area_m2

            return pd.Series(discharge, index=times, name='sim_discharge_cms')

        except Exception as e:
            print(f"  Error loading output: {e}")
            return None

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _load_obs(self) -> pd.Series:
        """Load observed streamflow."""
        df = pd.read_csv(self.obs_path, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        return df['discharge_cms']

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calc_metrics(obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        """Calculate KGE, NSE, RMSE, PBIAS, r, alpha, beta."""
        common = obs.index.intersection(sim.index)
        if len(common) < 48:  # need at least 2 days
            return {'KGE': -999, 'NSE': -999, 'RMSE': 999,
                    'PBIAS': 999, 'r': 0, 'alpha': 0, 'beta': 0}

        o = obs.loc[common].values.astype(float)
        s = sim.loc[common].values.astype(float)

        valid = np.isfinite(o) & np.isfinite(s)
        o, s = o[valid], s[valid]

        if len(o) < 48:
            return {'KGE': -999, 'NSE': -999, 'RMSE': 999,
                    'PBIAS': 999, 'r': 0, 'alpha': 0, 'beta': 0}

        # NSE
        ss_res = np.sum((o - s) ** 2)
        ss_tot = np.sum((o - np.mean(o)) ** 2)
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else -999

        # KGE components
        r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0
        alpha = np.std(s) / np.std(o) if np.std(o) > 0 else 0
        beta = np.mean(s) / np.mean(o) if np.mean(o) > 0 else 0
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        # RMSE
        rmse = np.sqrt(np.mean((o - s) ** 2))

        # PBIAS (%)
        pbias = 100 * np.sum(s - o) / np.sum(o) if np.sum(o) > 0 else 999

        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse,
                'PBIAS': pbias, 'r': r, 'alpha': alpha, 'beta': beta}

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate(self, params: Dict[str, float], iteration: int = 0,
                 quiet: bool = False) -> TrialResult:
        """
        Run SUMMA with given parameters, return metrics.

        Parameters
        ----------
        params : dict
            Maps parameter name -> value (physical units).
        iteration : int
            Iteration counter for bookkeeping.
        quiet : bool
            Suppress print output.

        Returns
        -------
        TrialResult
        """
        t0 = time.time()

        # Write parameters
        self._write_trial_params(params)

        # Run SUMMA
        success = self._run_summa()
        runtime = time.time() - t0

        if not success:
            result = TrialResult(iteration=iteration, params=dict(params),
                                 runtime_s=runtime, success=False)
            if not quiet:
                print(f"  [{iteration}] SUMMA FAILED ({runtime:.0f}s)")
            return result

        # Load output
        sim = self._load_sim_discharge()
        if sim is None:
            result = TrialResult(iteration=iteration, params=dict(params),
                                 runtime_s=runtime, success=False)
            if not quiet:
                print(f"  [{iteration}] No output ({runtime:.0f}s)")
            return result

        # Round to hourly for alignment
        sim.index = sim.index.round('h')

        # Slice to calibration period
        obs_cal = self.obs.loc[self.cal_start:self.cal_end]
        sim_cal = sim.loc[self.cal_start:self.cal_end]

        metrics = self.calc_metrics(obs_cal, sim_cal)

        result = TrialResult(
            iteration=iteration,
            params=dict(params),
            kge=metrics['KGE'],
            nse=metrics['NSE'],
            pbias=metrics['PBIAS'],
            rmse=metrics['RMSE'],
            r=metrics['r'],
            alpha=metrics['alpha'],
            beta=metrics['beta'],
            runtime_s=runtime,
            sim_discharge=sim,  # Store full time series
            success=True,
        )

        if not quiet:
            score = metrics[self.metric] if self.metric in metrics else metrics['KGE']
            print(f"  [{iteration:3d}] {self.metric}={score:+.4f}  "
                  f"NSE={metrics['NSE']:+.4f}  PBIAS={metrics['PBIAS']:+.1f}%  "
                  f"({runtime:.0f}s)")

        return result

    # ------------------------------------------------------------------
    # DDS Algorithm
    # ------------------------------------------------------------------

    def run(self, n_iterations: int = 100, live_plot: bool = True,
            initial_params: Optional[Dict[str, float]] = None,
            plot_interval: int = 1) -> pd.DataFrame:
        """
        Run DDS optimization with live plotting.

        Parameters
        ----------
        n_iterations : int
            Number of SUMMA evaluations.
        live_plot : bool
            Show live-updating convergence + hydrograph plots.
        initial_params : dict, optional
            Starting parameter values. Defaults to current localParamInfo defaults.
        plot_interval : int
            Update plots every N iterations (default 1 = every iteration).

        Returns
        -------
        pd.DataFrame
            Full optimization history.
        """
        print(f"\n{'='*60}")
        print(f"  DDS Optimization: {n_iterations} iterations")
        print(f"  Metric: {self.metric}  |  DDS r: {self.dds_r}")
        print(f"{'='*60}\n")

        self._update_file_manager_prefix()

        # Initialize with defaults or provided params
        if initial_params is None:
            current = {name: info.default for name, info in self.param_info.items()}
        else:
            current = {name: initial_params.get(name, info.default)
                       for name, info in self.param_info.items()}

        # Print starting parameters
        print("Starting parameters:")
        for name, val in current.items():
            info = self.param_info[name]
            print(f"  {name:30s}: {val:12.6g}  [{info.lower:.4g}, {info.upper:.4g}]")
        print()

        # Normalize current solution
        x_best = np.array([self.param_info[n].normalize(current[n])
                           for n in self.param_names])

        # Evaluate initial solution
        result = self.evaluate(current, iteration=0)
        self.history = [result]

        score_key = self.metric.lower()  # 'kge' or 'nse'
        best_score = getattr(result, score_key, -999)
        self.best_result = result

        # Setup live plot
        if live_plot:
            fig, axes = self._setup_live_plot()

        # --- DDS main loop ---
        for i in range(1, n_iterations):
            # Perturbation probability (decreases logarithmically)
            if n_iterations > 1:
                prob = 1.0 - np.log(i) / np.log(n_iterations)
            else:
                prob = 1.0
            prob = max(prob, 1.0 / self.n_params)

            # Select params to perturb
            perturb_mask = np.random.random(self.n_params) < prob
            if not perturb_mask.any():
                perturb_mask[np.random.randint(self.n_params)] = True

            # Generate trial solution
            x_trial = x_best.copy()
            for j in np.where(perturb_mask)[0]:
                perturbation = np.random.normal(0, self.dds_r)
                x_trial[j] += perturbation
                # Reflect at bounds [0, 1]
                if x_trial[j] < 0:
                    x_trial[j] = abs(x_trial[j])
                if x_trial[j] > 1:
                    x_trial[j] = 2.0 - x_trial[j]
                x_trial[j] = np.clip(x_trial[j], 0, 1)

            # Denormalize to physical values
            trial_params = {
                name: self.param_info[name].denormalize(x_trial[k])
                for k, name in enumerate(self.param_names)
            }

            # Evaluate
            result = self.evaluate(trial_params, iteration=i)
            self.history.append(result)

            # Greedy acceptance
            trial_score = getattr(result, score_key, -999)
            if result.success and trial_score > best_score:
                best_score = trial_score
                x_best = x_trial.copy()
                self.best_result = result
                current = trial_params

            # Live plot update
            if live_plot and (i % plot_interval == 0 or i == n_iterations - 1):
                self._update_live_plot(fig, axes)

        # Final summary
        print(f"\n{'='*60}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Best {self.metric}: {best_score:.4f}")
        if self.best_result:
            print(f"  Best NSE:  {self.best_result.nse:.4f}")
            print(f"  Best KGE:  {self.best_result.kge:.4f}")
            print(f"  PBIAS:     {self.best_result.pbias:+.1f}%")
        print(f"\n  Best parameters:")
        if self.best_result:
            for name, val in self.best_result.params.items():
                info = self.param_info[name]
                norm = info.normalize(val)
                bound_flag = ' <<<' if norm < 0.02 or norm > 0.98 else ''
                print(f"    {name:30s}: {val:12.6g}{bound_flag}")

        return self.get_history_df()

    # ------------------------------------------------------------------
    # Live plotting
    # ------------------------------------------------------------------

    def _setup_live_plot(self):
        """Create the live figure with 3 panels."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Interactive Optimization', fontsize=14, fontweight='bold')
        plt.ion()
        plt.show()
        return fig, axes

    def _update_live_plot(self, fig, axes):
        """Refresh all 4 panels of the live plot."""
        clear_output(wait=True)

        for ax in axes.flat:
            ax.clear()

        iters = list(range(len(self.history)))
        score_key = self.metric.lower()

        # --- Panel 1: Convergence ---
        ax1 = axes[0, 0]
        scores = [getattr(h, score_key, np.nan) for h in self.history]
        valid_scores = [s if h.success else np.nan for s, h in zip(scores, self.history)]

        # Running best
        running_best = []
        best_so_far = -999
        for s in valid_scores:
            if np.isfinite(s) and s > best_so_far:
                best_so_far = s
            running_best.append(best_so_far if best_so_far > -999 else np.nan)

        ax1.plot(iters, valid_scores, 'o', alpha=0.4, markersize=4, color='steelblue',
                 label='Each trial')
        ax1.plot(iters, running_best, '-', linewidth=2, color='red', label='Best so far')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel(self.metric)
        ax1.set_title(f'Convergence (best {self.metric} = {best_so_far:.4f})')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Hydrograph of best ---
        ax2 = axes[0, 1]
        if self.best_result and self.best_result.sim_discharge is not None:
            sim = self.best_result.sim_discharge
            obs_cal = self.obs.loc[self.cal_start:self.cal_end]
            sim_cal = sim.loc[self.cal_start:self.cal_end]

            # Daily means for cleaner plot
            obs_d = obs_cal.resample('D').mean()
            sim_d = sim_cal.resample('D').mean()

            ax2.plot(obs_d.index, obs_d.values, 'b-', linewidth=1.5, label='Observed')
            ax2.plot(sim_d.index, sim_d.values, 'r-', linewidth=1.5, alpha=0.8,
                     label='Simulated (best)')
            ax2.set_ylabel('Q (m³/s)')
            ax2.set_title(f'Best Hydrograph (KGE={self.best_result.kge:.3f}, '
                          f'NSE={self.best_result.nse:.3f})')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            # Rotate x labels
            for label in ax2.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')
        else:
            ax2.text(0.5, 0.5, 'No successful run yet', transform=ax2.transAxes,
                     ha='center', va='center', fontsize=14, color='gray')

        # --- Panel 3: Parameter values (normalized) ---
        ax3 = axes[1, 0]
        if self.best_result:
            names_short = [n.replace('basin__', 'b__')[:20] for n in self.param_names]
            norms = [self.param_info[n].normalize(self.best_result.params[n])
                     for n in self.param_names]
            colors = ['#e74c3c' if v < 0.05 or v > 0.95 else '#2ecc71' for v in norms]
            bars = ax3.barh(names_short, norms, color=colors, edgecolor='black', linewidth=0.5)
            ax3.axvline(0, color='gray', linewidth=0.5)
            ax3.axvline(1, color='gray', linewidth=0.5)
            ax3.set_xlim(-0.05, 1.05)
            ax3.set_xlabel('Normalized value (0=lower, 1=upper)')
            ax3.set_title('Best Parameters (red = near bound)')
        else:
            ax3.text(0.5, 0.5, 'Waiting...', transform=ax3.transAxes,
                     ha='center', va='center')

        # --- Panel 4: Scatter obs vs sim ---
        ax4 = axes[1, 1]
        if self.best_result and self.best_result.sim_discharge is not None:
            sim = self.best_result.sim_discharge
            obs_cal = self.obs.loc[self.cal_start:self.cal_end]
            sim_cal = sim.loc[self.cal_start:self.cal_end]
            common = obs_cal.index.intersection(sim_cal.index)
            o = obs_cal.loc[common]
            s = sim_cal.loc[common]

            ax4.scatter(o, s, alpha=0.3, s=8, color='steelblue')
            lim_max = max(o.max(), s.max()) * 1.05
            ax4.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, label='1:1')
            ax4.set_xlabel('Observed Q (m³/s)')
            ax4.set_ylabel('Simulated Q (m³/s)')
            ax4.set_title(f'Obs vs Sim (PBIAS={self.best_result.pbias:+.1f}%)')
            ax4.set_xlim(0, lim_max)
            ax4.set_ylim(0, lim_max)
            ax4.set_aspect('equal')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        display(fig)

    # ------------------------------------------------------------------
    # Results & export
    # ------------------------------------------------------------------

    def get_history_df(self) -> pd.DataFrame:
        """Return optimization history as a DataFrame."""
        rows = []
        for h in self.history:
            row = {'iteration': h.iteration, 'KGE': h.kge, 'NSE': h.nse,
                   'PBIAS': h.pbias, 'RMSE': h.rmse, 'r': h.r,
                   'alpha': h.alpha, 'beta': h.beta,
                   'runtime_s': h.runtime_s, 'success': h.success}
            row.update(h.params)
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_best(self, eval_start: Optional[str] = None, eval_end: Optional[str] = None):
        """
        Plot the best simulation vs observations.
        Optionally specify an evaluation period different from calibration.
        """
        if not self.best_result or self.best_result.sim_discharge is None:
            print("No successful simulation to plot.")
            return

        sim = self.best_result.sim_discharge
        sim.index = sim.index.round('h')

        # Full time range
        start = eval_start or str(sim.index.min().date())
        end = eval_end or str(sim.index.max().date())

        obs_period = self.obs.loc[start:end]
        sim_period = sim.loc[start:end]

        # Daily means
        obs_d = obs_period.resample('D').mean()
        sim_d = sim_period.resample('D').mean()

        metrics = self.calc_metrics(obs_period, sim_period)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Hydrograph
        ax1 = axes[0]
        ax1.plot(obs_d.index, obs_d.values, 'b-', linewidth=1.5, label='Observed')
        ax1.plot(sim_d.index, sim_d.values, 'r-', linewidth=1.5, alpha=0.8, label='Simulated')
        ax1.axvspan(self.cal_start, self.cal_end, alpha=0.08, color='green', label='Cal. period')
        ax1.set_ylabel('Q (m³/s)')
        ax1.set_title(f'KGE={metrics["KGE"]:.3f}  NSE={metrics["NSE"]:.3f}  '
                       f'PBIAS={metrics["PBIAS"]:+.1f}%  r={metrics["r"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals
        ax2 = axes[1]
        common = obs_d.index.intersection(sim_d.index)
        resid = sim_d.loc[common] - obs_d.loc[common]
        ax2.bar(common, resid.values, width=1, color=['#e74c3c' if v > 0 else '#3498db'
                for v in resid.values], alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('Residual (m³/s)')
        ax2.set_title('Sim - Obs (daily)')
        ax2.grid(True, alpha=0.3)

        # Flow duration curve
        ax3 = axes[2]
        obs_sorted = np.sort(obs_d.dropna().values)[::-1]
        sim_sorted = np.sort(sim_d.dropna().values)[::-1]
        exc_obs = np.arange(1, len(obs_sorted)+1) / len(obs_sorted) * 100
        exc_sim = np.arange(1, len(sim_sorted)+1) / len(sim_sorted) * 100
        ax3.semilogy(exc_obs, obs_sorted, 'b-', linewidth=1.5, label='Observed')
        ax3.semilogy(exc_sim, sim_sorted, 'r-', linewidth=1.5, alpha=0.8, label='Simulated')
        ax3.set_xlabel('Exceedance probability (%)')
        ax3.set_ylabel('Q (m³/s)')
        ax3.set_title('Flow Duration Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f'Best Model: {self.metric}={getattr(self.best_result, self.metric.lower()):.4f}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()

    def save_best(self, filepath: Optional[str] = None):
        """Save best parameters to CSV."""
        if not self.best_result:
            print("No results to save.")
            return

        if filepath is None:
            filepath = self.output_dir / 'interactive_best_parameters.csv'

        rows = []
        for name, val in self.best_result.params.items():
            info = self.param_info[name]
            rows.append({
                'parameter': name,
                'value': val,
                'type': info.kind,
                'lower': info.lower,
                'upper': info.upper,
                'normalized': info.normalize(val),
            })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Saved best parameters to: {filepath}")
        return df

    def get_best_params(self) -> Dict[str, float]:
        """Return best parameter values as a dict."""
        if self.best_result:
            return dict(self.best_result.params)
        return {name: info.default for name, info in self.param_info.items()}

    # ------------------------------------------------------------------
    # Convenience: single manual run
    # ------------------------------------------------------------------

    def test_params(self, **kwargs) -> TrialResult:
        """
        Run a single evaluation with specified parameters.
        Unspecified params use current defaults.

        Example:
            result = opt.test_params(k_soil=5e-6, frozenPrecipMultip=0.8)
        """
        params = {name: info.default for name, info in self.param_info.items()}
        params.update(kwargs)
        return self.evaluate(params, iteration=len(self.history))

    def resume(self, n_iterations: int = 50, live_plot: bool = True,
               plot_interval: int = 1) -> pd.DataFrame:
        """
        Continue optimization from wherever we left off.
        Uses the current best solution as starting point.
        """
        if not self.best_result:
            print("No previous run found. Use run() first.")
            return self.get_history_df()

        print(f"\nResuming from iteration {len(self.history)}, "
              f"best {self.metric}={getattr(self.best_result, self.metric.lower()):.4f}")

        # Reconstruct normalized best
        x_best = np.array([
            self.param_info[n].normalize(self.best_result.params[n])
            for n in self.param_names
        ])
        best_score = getattr(self.best_result, self.metric.lower())
        score_key = self.metric.lower()

        self._update_file_manager_prefix()

        if live_plot:
            fig, axes = self._setup_live_plot()

        start_iter = len(self.history)

        for i in range(start_iter, start_iter + n_iterations):
            total_iters = start_iter + n_iterations
            if total_iters > 1:
                prob = 1.0 - np.log(i - start_iter + 1) / np.log(n_iterations)
            else:
                prob = 1.0
            prob = max(prob, 1.0 / self.n_params)

            perturb_mask = np.random.random(self.n_params) < prob
            if not perturb_mask.any():
                perturb_mask[np.random.randint(self.n_params)] = True

            x_trial = x_best.copy()
            for j in np.where(perturb_mask)[0]:
                perturbation = np.random.normal(0, self.dds_r)
                x_trial[j] += perturbation
                if x_trial[j] < 0:
                    x_trial[j] = abs(x_trial[j])
                if x_trial[j] > 1:
                    x_trial[j] = 2.0 - x_trial[j]
                x_trial[j] = np.clip(x_trial[j], 0, 1)

            trial_params = {
                name: self.param_info[name].denormalize(x_trial[k])
                for k, name in enumerate(self.param_names)
            }

            result = self.evaluate(trial_params, iteration=i)
            self.history.append(result)

            trial_score = getattr(result, score_key, -999)
            if result.success and trial_score > best_score:
                best_score = trial_score
                x_best = x_trial.copy()
                self.best_result = result

            if live_plot and ((i - start_iter) % plot_interval == 0 or
                              i == start_iter + n_iterations - 1):
                self._update_live_plot(fig, axes)

        print(f"\nResume complete. Best {self.metric}: {best_score:.4f}")
        return self.get_history_df()
