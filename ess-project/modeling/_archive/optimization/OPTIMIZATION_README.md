# DDS Parameter Calibration Optimization

Complete standalone workflow for running CONFLUENCE DDS parameter optimization with minimal setup.

**Location:** `ess-project/modeling/optimization/`

## Quick Start

```bash
# Navigate to optimization directory
cd ~/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/optimization

# Run optimization with default settings
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "my_experiment"

# Continue from previous best parameters
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "refinement" --use-previous

# Show help
./optimize.sh --help
```

## What It Does

The `optimize_confluence.py` script handles the complete optimization workflow:

1. **Configuration Setup** - Loads YAML config and initializes CONFLUENCE
2. **Parameter Initialization** - Sets up parameters from base settings or previous optimization
3. **Forcing Adjustments** - Applies elevation-based temperature lapsing and radiation corrections
4. **Model Preprocessing** - Runs CONFLUENCE domain delineation, discretization, and model setup
5. **DDS Optimization** - Executes Dynamically Dimensioned Search calibration
6. **Results Organization** - Saves outputs and best parameters to project directory

## Files

| File | Purpose |
|------|---------|
| `optimize_confluence.py` | Main optimization script (all preprocessing + calibration) |
| `optimize.sh` | Bash wrapper for easy command-line execution |
| `Analyze_Optimization_Results.ipynb` | Separate notebook for analyzing and visualizing results |
| `TuolumneBasinLumped.ipynb` | Manual model runs and validation (no optimization code) |

## Usage Examples

### Standard Optimization Run
```bash
./optimize.sh --config config_Tuolumne_lumped_v1.yaml
```
- Uses `EXPERIMENT_ID` from config as run name
- Initializes with default parameter values from `0_base_settings/SUMMA/`

### Custom Experiment Name
```bash
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "spongy_aquifer"
```
Creates experiment ID: `spongy_aquifer_20260225`

### Refinement Run (Continue from Best)
```bash
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "refinement_v2" --use-previous
```
- Finds the most recent optimization results
- Loads best parameters as starting point
- Runs another iteration of optimization

### Direct Python Usage
```bash
python3 optimize_confluence.py \
    --config config_Tuolumne_lumped_v1.yaml \
    --run-name "my_run" \
    --use-previous
```

## Configuration Requirements

Your YAML config file **must** include:

```yaml
# Critical for streamflow unit conversion (m/s → m³/s)
BASIN_AREA_M2: 774515407.93

# Optimization settings
ITERATIVE_OPTIMIZATION_ALGORITHM: DDS
NUMBER_OF_ITERATIONS: 200  # Change this to adjust run length
DDS_R: 0.4

# Parameters to calibrate
PARAMS_TO_CALIBRATE: k_soil, theta_sat, theta_res, aquiferScaleFactor, rootingDepth
BASIN_PARAMS_TO_CALIBRATE: basin__aquiferScaleFactor, basin__aquiferHydCond

# Performance metric
OPTIMIZATION_METRIC: NSE
```

## Preprocessing Steps (Automatic)

The script automatically handles these preprocessing steps:

### 1. Parameter Setup
- Copies base parameter files from `0_base_settings/SUMMA/`
- Updates model decisions (groundwater, boundary conditions, albedo method)
- Initializes with default or previous-best parameters

### 2. Temperature Lapsing & Radiation Correction
- Loads DEM and calculates elevation difference between HRUs and basin mean
- Applies temperature adjustment: `T_adjusted = T + (elevation_diff × lapse_rate)`
- Recalculates incoming longwave radiation using Dilley & O'Brien (2002) empirical formula
- Outputs adjusted forcing to `{domain}/forcing/SUMMA_input/`

### 3. Model Preprocessing
- Protects custom parameter files (backs them up)
- Runs CONFLUENCE model-agnostic preprocessing
- Runs CONFLUENCE model-specific preprocessing (SUMMA setup)
- Restores custom parameter files

### 4. DDS Optimization
- Executes parameter search algorithm
- Saves convergence history and best parameters
- Generates optimization results to `{domain}/optimisation/`

## Results Location

After running, outputs are saved to:

```
{CONFLUENCE_DATA_DIR}/domain_Tuolumne_River_lumped/
└── optimisation/
    ├── run_dds_opt_<timestamp>_<run_name>/
    │   ├── SUMMA/              # Simulated streamflow
    │   ├── best_parameters.csv # Optimal parameters found
    │   └── optimization_history.csv
    └── *_dds_history.csv       # Full iteration history
```

## Analyzing Results

Use the separate analysis notebook **after** optimization completes:

```bash
# Open the analysis notebook
jupyter notebook Analyze_Optimization_Results.ipynb
```

This notebook provides:
- **Convergence plots** - How NSE improved over iterations
- **Parameter sensitivity** - Which parameters matter most
- **Best parameters** - Values found by optimization
- **Model validation** - Simulated vs observed streamflow
- **Performance metrics** - NSE, KGE, correlation

## Common Workflows

### Workflow 1: Single Optimization Run
```bash
# Run 200 iterations with default parameters
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "baseline"

# Analyze results
jupyter notebook Analyze_Optimization_Results.ipynb
```

### Workflow 2: Iterative Refinement
```bash
# Initial run
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "round1"

# Analyze and refine (run 2)
./optimize.sh --config config_Tuolumne_lumped_v1.yaml --run-name "round2" --use-previous

# Analyze round 2 results
jupyter notebook Analyze_Optimization_Results.ipynb
```

### Workflow 3: Manual Testing
After optimization, use `TuolumneBasinLumped.ipynb` to:
- Run manual simulations with optimized parameters
- Compare against observations
- Validate model behavior
- Create publication-quality plots

## Troubleshooting

**Config file not found:**
```bash
ls 0_config_files/
# Use a file from this list
./optimize.sh --config config_Tuolumne_lumped_v1.yaml
```

**Missing BASIN_AREA_M2:**
Add to your config:
```yaml
BASIN_AREA_M2: 774515407.93
```
This is critical for correct streamflow unit conversion!

**Forcing data not found:**
Ensure forcing files exist at:
```
{domain}/forcing/forcing_noTadjust/
```

**Previous optimization not found (for --use-previous):**
Check that results exist:
```bash
ls {CONFLUENCE_DATA_DIR}/domain_Tuolumne_River_lumped/optimisation/
```

## Advanced Usage

### Modify Parameter Bounds

Edit `config_YAML_file.yaml` to change optimization bounds:

```yaml
# LOCAL PARAMETERS
tempCritRain:
  description: "Critical rain-snow temperature"
  dtype: float
  bounds: [270, 280]    # Change these bounds

k_soil:
  description: "Soil hydraulic conductivity"  
  dtype: float
  bounds: [1e-7, 1e-4]  # Logarithmic bounds recommended
```

### Monitor Progress During Run

```bash
# Watch convergence in real-time
watch -n 30 'tail -5 {CONFLUENCE_DATA_DIR}/domain_Tuolumne_River_lumped/optimisation/*_dds_history.csv'
```

### Compare Multiple Runs

```bash
# Run multiple experiments
./optimize.sh --config config.yaml --run-name "exp_a"
./optimize.sh --config config.yaml --run-name "exp_b"  
./optimize.sh --config config.yaml --run-name "exp_c"

# Analyze any run by checking optimization folder
jupyter notebook Analyze_Optimization_Results.ipynb
```

## Key Differences from Notebook

This standalone approach differs from the interactive notebook:

| Aspect | Standalone Script | Interactive Notebook |
|--------|------------------|----------------------|
| Forcing prep | Automatic | Manual cells |
| Parameter setup | Automatic | Manual cells |
| Optimization | Single command | Manual cell execution |
| Analysis | Separate notebook | Same notebook |
| Repeatability | Fully scripted | Requires manual steps |
| Configuration | Via YAML | Via notebook variables |

## Default Parameters

The base settings (`0_base_settings/SUMMA/`) contain these initial values:

```
tempCritRain:        274.1 K
k_soil:              9.4e-6 m/s
theta_sat:           0.5160 (dimensionless)
theta_res:           0.0270 (dimensionless)
rootingDepth:        6.876 m
basin__aquiferHydCond:    0.0010 m/s
basin__aquiferScaleFactor: 50.0 (dimensionless)
routingGammaShape:   2.5 (dimensionless)
routingGammaScale:   4.6e4 s
```

To change defaults, edit files in `0_base_settings/SUMMA/`:
- `localParamInfo.txt` - HRU-scale parameters
- `basinParamInfo.txt` - Watershed-scale parameters

## Performance Notes

- **Typical runtime:** 1-2 hours for 200 iterations (varies by PC count in MPI_PROCESSES)
- **Disk space:** ~5-10 GB per optimization run
- **Memory:** Scales with domain complexity and forcing resolution

## Citation

If using this optimization workflow, cite:

- CONFLUENCE: [appropriate citation]
- DDS Algorithm: [Tolson & Shoemaker, 2007]

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review detailed output in optimization logs
3. Consult `Analyze_Optimization_Results.ipynb` for result diagnostics
