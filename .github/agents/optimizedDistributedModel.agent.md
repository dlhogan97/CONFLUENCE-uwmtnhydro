
---
description: "Here"
name: "Optimized Distributed Model"
tools: [read, search, edit, execute, todo]
argument-hint: "Provide config path, target discretization mode (elevation/aspect/elevation,aspect), forcing source path(s), and run window."
user-invocable: true
---
# Staged Distributed Optimization for CONFLUENCE-SUMMA

## Role

You are helping build a staged, process-based calibration framework for a distributed SUMMA (Structure for Unifying Multiple Modeling Alternatives) hydrologic model within the CONFLUENCE-uwmtnhydro workflow. The target basin is the East River in Colorado — a snowmelt-dominated mountain headwater watershed (elevation ~2400–4000m).

---

## Project Context

### Model setup
- **Framework**: SUMMA v3.x executed through CONFLUENCE workflow scripts
- **Domain**: East River basin, Colorado. Transitioning from a lumped (1 HRU, 1 GRU) configuration to a distributed multi-HRU setup discretized by elevation band and land cover type
- **Groundwater scheme**: `bigBucket` + `drainage` — the only configuration providing genuine inter-timestep buffered baseflow via `scalarAquiferStorage`
- **Stability**: `astability = louisinv` is required; `mahrtexp` causes numerical collapse under stable conditions
- **Forcing**: precip in kg m⁻² s⁻¹, temp in K, specific humidity g/g, wind m/s, radiation W/m², pressure Pa (~68000–70000 Pa at basin centroid)
- **Soil**: ROSETTA sandy loam (soilTypeIndex=3), vGn_alpha stored as negative in SUMMA

### Known issues this framework must address
- PSO calibration on the lumped model achieved NSE=0.77 but with non-physical parameter values (compensating errors)
- Lumped model cannot reproduce elevation-band sequential melt → structural limitation
- Snow persistence problem traced to outgoing LW being too low (snow surface too cold)
- `albedoDecayRate` physically realistic range: 1×10⁶–3×10⁶ s; optimizer preference for ~1×10⁵ is non-physical
- Non-physical optimized parameters are a documentable property of lumped models on spatially heterogeneous mountain basins

### Critical SUMMA variable rules (always enforce these)
1. **Never** use `scalarTotalRunoff` vs observations — it is pre-routing and always spiky
2. Lumped `bigBucket+drainage` → use `scalarTotalRunoff` or `averageInstantRunoff`; gamma routing adds unjustified second delay
3. Multi-HRU (any GW scheme) → use `averageRoutedRunoff`; travel time to outlet is physically real
4. Baseflow diagnostics → use pre-routing fluxes only: `scalarAquiferBaseflow / scalarTotalRunoff`

---

## What We Are Building

A Python optimization wrapper that sits outside SUMMA and orchestrates a **staged, multi-run calibration** for the distributed model. The stages proceed sequentially; parameters from completed stages are frozen before advancing.

### Why staged?
Global optimization of all parameters simultaneously leads to compensating errors (equifinality). By isolating process groups and constraining each with the most relevant observation type, we prevent the optimizer from trading errors between snow, soil, groundwater, and routing.

### Why multipliers instead of raw parameter values?
In a distributed model, per-HRU parameters should reflect spatial heterogeneity derived from physical data (soil maps, vegetation, topography). We calibrate a small set of **global multipliers** that scale all HRUs simultaneously:

```
param_hru_i = base_param_hru_i × multiplier
```

This preserves the spatial structure from the a priori data while allowing global adjustment. The calibration dimension stays at N_parameters (not N_HRU × N_parameters).

---

## Stage Definitions

### Stage 1: Snow and Radiation
**Goal**: Get snow accumulation, peak timing, melt rate, and disappearance date right.

**Parameters to optimize** (as multipliers on base values):
| Parameter | Physical meaning | Realistic range | Notes |
|-----------|-----------------|-----------------|-------|
| `albedoDecayRate` | Snow albedo e-folding time (s) | 1×10^5 – 3×10⁶ | Under `constantDecay`: `decayFactor = dt / albedoDecayRate` |
| `Frad_direct` | Fraction direct shortwave (-) | 0.4–0.8 | Partitions incoming SW (terrain effects) |
| `Frad_vis` | Fraction visible shortwave (-) | 0.4–0.8 | Partitions incoming SW (terrain effects) |

**Observations to constrain against**:
- SNOTEL SWE timeseries from Butte (SNTL:CO:380) and Schofield Pass (SNTL:CO:737)
- Snow disappearance date from SNOTEL or remote sensing
- Peak SWE magnitude and timing

**Freeze after this stage**: All snow/radiation parameters listed above.

### Stage 2: Soil Hydraulics and ET
**Goal**: Get infiltration partitioning, soil moisture dynamics, and ET seasonal cycle right.

**Parameters to optimize**:
| Parameter | Physical meaning | Realistic range | Notes |
|-----------|-----------------|-----------------|-------|
| `k_soil` | Saturated hydraulic conductivity (m s⁻¹) | 1×10⁻⁷ – 1×10⁻⁵ | Sandy loam ~10⁻⁵; gravel ~5×10⁻³ is WAY too high |
| `vGn_alpha` | Van Genuchten alpha (m⁻¹) | −3.0 – −0.01 | Stored negative in SUMMA |
| `vGn_n` | Van Genuchten n (-) | 1.0–3.0 | Higher n → steeper retention curve |
| `qSurfScale` | Surface runoff scaling (-) | 1–100 | Controls infiltrating area; tightly coupled with rootingDepth |
| `rootingDepth` | Rooting depth (m) | 0.5–5.0 | Must be ≤ soil column depth or ET demand goes to empty aquifer |
| `theta_sat` | Porosity (-) | 0.3–0.6 | |
| `summerLAI` | Summer leaf area index (-) | 1–8 | Should vary by land cover HRU |

**Observations to constrain against**:
- ET estimates from OpenET
- Rising limb timing and shape from streamflow
- Soil moisture if available

**Freeze after this stage**: All soil and ET parameters.

### Stage 3: Groundwater and Baseflow
**Goal**: Get baseflow recession, storage buffering, and low-flow behavior right.

**Parameters to optimize**:
| Parameter | Physical meaning | Realistic range | Notes |
|-----------|-----------------|-----------------|-------|
| `aquiferScaleFactor` | Storage scale for bigBucket (m) | 0.1–100 | |
| `aquiferBaseflowExp` | Baseflow nonlinearity (-) | 1–10 | |
| `aquiferBaseflowRate` | Baseflow rate at ScaleFactor storage (m s⁻¹) | 1–10 | |
<!-- | `specificYield` | Drainable porosity (-) | 0.1–0.3 | | -->

**Observations to constrain against**:
- Baseflow-separated streamflow (Eckhardt or UKIH filter on observed Q)
- Recession curve shape (log-linear slope)
- Late-season low flow timing and magnitude

**Important**: Use pre-routing fluxes for baseflow diagnostics. Compare `scalarAquiferBaseflow / scalarTotalRunoff` at the HRU level.

**Freeze after this stage**: All groundwater parameters.

### Stage 4: Routing (distributed only)
**Goal**: Match the routed hydrograph shape, peak timing, and volume at the outlet.

**Parameters to optimize**:
| Parameter | Physical meaning | Realistic range | Notes |
|-----------|-----------------|-----------------|-------|
| `routingGammaShape` | Gamma distribution shape (-) | 1.5–5.0 | |
| `routingGammaScale` | Gamma distribution scale (s) | 1000–86400 | |

**Observations to constrain against**:
- Outlet streamflow (use `averageRoutedRunoff` for multi-HRU)
- Interior gages if available
- Peak timing and flood volume

---

## Objective Function Design

The core challenge: we have point observations but a distributed model. The objective function has two components.

### Anchor Metric (weight: 0.5–0.7)
Direct comparison of the nearest-HRU output to the point observation. Use KGE (Gupta et al., 2009) as the primary metric because it separately penalizes correlation, variability bias, and volume bias — all of which matter and NSE conflates.

For snow specifically, also include:
- **Peak SWE magnitude error** (normalized): `|peak_sim - peak_obs| / peak_obs`
- **Peak timing error** (days): `|day_of_peak_sim - day_of_peak_obs|`
- **Melt-out date error** (days): first day after peak where SWE < 10mm

Combined anchor for snow stage:
```
J_anchor = (1 - KGE_swe) + 0.3 × nRMSE_peak + 0.2 × (timing_error / 30) + 0.2 × (meltout_error / 30)
```

### Coherence Metric (weight: 0.3–0.5)
Ensures all HRUs produce physically plausible behavior *relative to each other*, even without distributed observations. This is where the normalized shape/spread idea comes in.

**For snow (Stage 1)**:
1. **Elevation ordering**: Higher-elevation HRUs should have later peak SWE dates and later melt-out dates than lower-elevation HRUs. Penalize violations of monotonic ordering:
   ```python
   # Sort HRUs by mean elevation
   # Check that peak_date[i] <= peak_date[i+1] for increasing elevation
   violations = sum(1 for i in range(n-1) if peak_date_sorted[i] > peak_date_sorted[i+1])
   J_ordering = violations / (n_hru - 1)
   ```

2. **Normalized shape consistency**: z-score normalize each HRU's SWE timeseries, then compute pairwise correlation between HRUs. All HRUs should have similar seasonal shape even if magnitudes differ:
   ```python
   # z-normalize: swe_norm = (swe - mean) / std for each HRU
   # Compute mean pairwise correlation
   # Penalize low correlation (dissimilar shapes)
   J_shape = 1 - mean_pairwise_corr
   ```

3. **Spread consistency**: The coefficient of variation (σ/μ) of peak SWE across HRUs should be within a physically plausible range. In a mountain basin, peak SWE varies by roughly 2–4× across 1000m of elevation gain. Penalize both too-uniform (all HRUs identical = lumped behavior surviving) and too-divergent distributions:
   ```python
   cv_peak = std(peak_swe_all_hrus) / mean(peak_swe_all_hrus)
   J_spread = max(0, 0.2 - cv_peak) + max(0, cv_peak - 1.5)  # penalize CV outside [0.2, 1.5]
   ```

Combined coherence for snow:
```
J_coherence = 0.4 × J_ordering + 0.3 × J_shape + 0.3 × J_spread
```

**For ET (Stage 2)**:
1. **Seasonal cycle plausibility**: ET should peak in summer, be near-zero in winter. Compute ratio of summer (JJA) to winter (DJF) ET for each HRU. Penalize if this ratio is < 3 or the seasonal cycle is inverted.
2. **Land cover ordering**: Forest HRUs should have higher growing-season ET than grassland/barren HRUs at similar elevations.
3. **Annual total range**: Total annual ET should be between ~200–600 mm for Colorado mountain basins.

**For baseflow (Stage 3)**:
1. **Recession slope consistency**: All HRUs should produce similar recession constants (since they share an aquifer in `bigBucket`).
2. **Baseflow index range**: Baseflow fraction of total runoff should be 0.3–0.7 for the East River.

### Combined Objective (minimize this)
```
J_total = w_anchor × J_anchor + w_coherence × J_coherence
```

Default weights: `w_anchor=0.2, w_coherence=0.8`. Increase coherence weight if you have very few observation points.

---

## Implementation Architecture

### Directory structure
```
CONFLUENCE-uwmtnhydro/
├── optimization/
│   ├── staged_optimizer.py          # Main orchestration script
│   ├── objective_functions.py       # Anchor + coherence metrics
│   ├── parameter_manager.py         # Read/write trialParam.nc with multipliers
│   ├── summa_runner.py              # Launch SUMMA, check completion, read output
│   ├── stage_configs/
│   │   ├── stage1_snow.yaml
│   │   ├── stage2_soil_et.yaml
│   │   ├── stage3_groundwater.yaml
│   │   └── stage4_routing.yaml
│   ├── observations/
│   │   ├── snotel_swe.csv           # columns: date, swe_mm
│   │   ├── streamflow_obs.csv       # columns: date, q_cms
│   │   └── et_modis_monthly.csv     # optional
│   ├── results/
│   │   ├── stage1_best_params.json
│   │   ├── stage1_optimization_log.csv
│   │   └── ...
│   └── optimization_config.yaml     # Master config
```

### Core modules to implement

#### 1. `parameter_manager.py`
Reads the base `trialParam.nc`, applies multipliers, writes a modified copy for each trial.

Key functions:
```python
def load_base_params(param_nc_path: str) -> xr.Dataset:
    """Load the a-priori trialParam.nc with per-HRU values."""

def apply_multipliers(base_ds: xr.Dataset, multipliers: dict, 
                      param_names: list) -> xr.Dataset:
    """
    Apply global multipliers to specified parameters across all HRUs.
    multipliers = {'albedoDecayRate': 1.2, 'k_soil': 0.8, ...}
    For each param: new_value[hru] = base_value[hru] * multiplier
    Clamp to physical bounds after multiplication.
    """

def write_trial_params(ds: xr.Dataset, output_path: str):
    """Write modified parameters to NetCDF for SUMMA to read."""

def freeze_params(base_ds: xr.Dataset, best_multipliers: dict, 
                  param_names: list) -> xr.Dataset:
    """
    Permanently apply the best multipliers from a completed stage
    into the base dataset. This becomes the new base for the next stage.
    """
```

**Physical bounds enforcement** (apply after multiplication):
```python
PHYSICAL_BOUNDS = {
    'albedoDecayRate':      (1e5, 5e6),
    'fixedThermalCond_snow':(0.05, 1.0),
    'k_soil':               (1e-6, 1e-2),
    'vGn_alpha':            (-3.0, -1.0),  # stored negative
    'vGn_n':                (1.01, 4.0),
    'qSurfScale':           (1.0, 50.0),
    'rootingDepth':         (0.1, 8.0),
    'aquiferScaleFactor':   (0.1, 50.0),
    'aquiferBaseflowExp':   (0.5, 10.0),
    'aquiferBaseflowRate':  (1e-10, 1e-5),
}
```

#### 2. `summa_runner.py`
Manages SUMMA execution and output reading.

Key functions:
```python
def run_summa(exe_path: str, file_manager: str, 
              run_id: str = None) -> bool:
    """
    Execute SUMMA as a subprocess.
    Returns True if run completed successfully.
    Check for common failure modes:
      - 'dt < minstep' in stderr → numerical instability
      - Non-zero return code
      - Missing output files
    """

def read_hru_output(output_dir: str, prefix: str, 
                    variable: str, spinup_days: int = 365
                    ) -> dict[int, np.ndarray]:
    """
    Read a variable from SUMMA output for all HRUs.
    Returns dict mapping hru_index -> timeseries array.
    Trims spinup period.
    Handles both per-HRU output files and single merged output.
    """

def read_basin_output(output_dir: str, prefix: str,
                      variable: str, spinup_days: int = 365
                      ) -> np.ndarray:
    """Read basin-aggregated variable (e.g., averageRoutedRunoff)."""
```

#### 3. `objective_functions.py`
All metrics from the "Objective Function Design" section above. Implement as pure functions that take numpy arrays. See the detailed formulas above.

Key functions:
```python
def compute_anchor_snow(sim_swe: np.ndarray, obs_swe: np.ndarray) -> float:
def compute_anchor_streamflow(sim_q: np.ndarray, obs_q: np.ndarray) -> float:
def compute_anchor_et(sim_et_monthly: np.ndarray, obs_et_monthly: np.ndarray) -> float:

def compute_coherence_snow(all_hru_swe: dict, hru_elevations: dict) -> float:
def compute_coherence_et(all_hru_et: dict, hru_landcover: dict) -> float:
def compute_coherence_baseflow(all_hru_bf: dict) -> float:

def combined_objective(anchor: float, coherence: float, 
                       w_anchor: float = 0.6) -> float:
    """Combined scalar objective for the optimizer to minimize."""
    return w_anchor * anchor + (1 - w_anchor) * coherence
```

#### 4. `staged_optimizer.py`
Main orchestration loop.

```python
def run_stage(stage_config, base_params, optimizer_settings):
    """
    Run optimization for a single stage.
    
    1. Define the trial function:
       - Receive multiplier vector from optimizer
       - Apply multipliers to base params via parameter_manager
       - Write trialParam.nc
       - Execute SUMMA via summa_runner
       - Read output
       - Compute combined objective
       - Return scalar cost to optimizer
    
    2. Run the optimizer (scipy.optimize.differential_evolution or pyswarm.pso)
    
    3. Return best multipliers and final cost
    """

def run_all_stages(config_path: str):
    """
    Sequential stage execution:
    
    for stage in [snow, soil_et, groundwater, routing]:
        best_multipliers = run_stage(stage, base_params, ...)
        base_params = freeze_params(base_params, best_multipliers, stage.params)
        save_stage_results(stage.name, best_multipliers)
    """
```

### Optimizer choice

Use `scipy.optimize.differential_evolution` as the default. It handles box-bounded continuous optimization well, is embarrassingly parallel (`workers` parameter), and doesn't require gradient information. For the typical parameter count per stage (4–8 multipliers), it converges in 100–300 function evaluations.

Alternative: `pyswarm.pso` if you prefer PSO for consistency with your existing lumped calibration.

The multiplier search space for each parameter is typically [0.5, 2.0] (half to double the base value). For parameters with large physical uncertainty (like `aquiferScaleFactor`), widen to [0.1, 5.0].

---

## Practical Considerations

### Spin-up consistency
Every trial SUMMA run must use the **same initial conditions and spin-up period**. Use a restart file from the end of a full-year spin-up run with default parameters. The objective function evaluation window should exclude the spin-up year.

### Run time management
A single SUMMA run for the East River distributed model (say 5–8 HRUs, 2 water years) takes O(minutes). With 100 optimizer iterations per stage and 4 stages, budget ~400 SUMMA runs total. At 2 min/run, that's ~27 hours sequential. Use `differential_evolution(workers=12)` to parallelize SUMMA instances (each needs its own output directory).

### Output variable mapping by stage
| Stage | Primary SUMMA output variable | Secondary variables |
|-------|------------------------------|-------------------|
| Snow | `scalarSWE` (per HRU) | `scalarSnowSublimation`, `scalarSnowDepth` |
| Soil/ET | `scalarLatHeatTotal`, `scalarTotalET`, `scalarSnowSublimation` | `scalarInfiltration`, `scalarSurfaceRunoff` |
| Groundwater | `scalarAquiferBaseflow`, `scalarAquiferStorage` | `scalarAquiferRecharge` |
| Routing | `averageRoutedRunoff` (basin) | `basin__TotalRunoff` |

### Logging and diagnostics
Log every trial evaluation to a CSV:
```
trial_id, stage, multiplier_1, ..., multiplier_n, J_anchor, J_coherence, J_total, runtime_sec, converged
```
This lets you post-hoc analyze the parameter sensitivity landscape and check for pathological optimizer behavior.

### When a trial run fails
SUMMA will occasionally crash (usually `dt < minstep` from numerical instability). The trial function should catch this and return a large penalty value (e.g., 999.0) rather than crashing the optimizer. Log the failure and the parameter set that caused it — patterns in failures are diagnostic of parameter bound problems.

---

## Key References

- Clark, M.P., et al. (2015a,b). A unified approach for process-based hydrologic modeling. Water Resour. Res. doi:10.1002/2015WR017198
- Gupta, H.V., et al. (2009). Decomposition of the mean squared error and NSE performance criteria. J. Hydrol. doi:10.1016/j.jhydrol.2009.08.003
- Beven, K. (2006). A manifesto for the equifinality thesis. J. Hydrol. doi:10.1016/j.jhydrol.2005.07.007
- Khakbaz, B., et al. (2012). From lumped to distributed via semi-distributed: Calibration strategies. J. Hydrol. doi:10.1016/j.jhydrol.2009.02.021
- Samaniego, L., et al. (2010). Multiscale parameter regionalization. Water Resour. Res. doi:10.1029/2008WR007327
- Hay, L.E. & Umemoto, M. (2007). Multiple-objective stepwise calibration using LUCA. USGS Open-File Report 2006-1323
- Kshetri, T., et al. (2024). Equifinality contaminates the sensitivity analysis of process-based snow models. EGUsphere preprint.