# CONFLUENCE-uwmtnhydro — Project Context for Claude

## What is This Project?

**CONFLUENCE** (Community Optimization Nexus for Leveraging Understanding of Environmental Networks in Computational Exploration) is a Python-based hydrological modeling framework. This fork (`CONFLUENCE-uwmtnhydro`) is the University of Washington Mountain Hydrology group's active research branch, focused on:

- Seasonal ensemble streamflow forecasting
- Parameter calibration and sensitivity analysis
- Lumped, semi-distributed, and distributed watershed modeling
- Multi-model comparison (SUMMA, FUSE, GR, HYPE, FLASH, MESH)

The primary research domain is **East River Basin** (Colorado) with additional work on Tuolumne (CA) and other basins.

---

## Repository Structure

```
CONFLUENCE-uwmtnhydro/
├── CONFLUENCE.py                    # Main orchestrator class
├── confluence                       # Bash CLI wrapper
├── 0_config_files/                  # Config templates and active experiment configs
├── 0_base_settings/                 # Template files for external models (SUMMA, FUSE, mizuRoute, OSTRICH)
├── utils/                           # Core Python modules (60 files)
│   ├── project/                     # Project setup, workflow orchestration, logging
│   ├── models/                      # Model interfaces (summa_utils.py, fuse_utils.py, mizuroute_utils.py, etc.)
│   ├── data/                        # Forcing acquisition and preprocessing
│   ├── geospatial/                  # Domain definition, DEM analysis, HRU discretization
│   ├── optimization/                # DDS, PSO, SCE-UA, neural network calibration
│   ├── evaluation/                  # NSE/KGE/RMSE metrics, benchmarking, sensitivity analysis
│   └── custom/                      # Project-specific utilities (forcing, plotting, analysis)
└── ess-project/                     # UW-specific research project
    ├── 0_config_files/              # Domain configs (East River, Tuolumne variants)
    ├── 0_base_settings/SUMMA/       # Domain-specific SUMMA settings (overrides root templates)
    ├── 1_forcing/                   # ERA5, GRIDMET download scripts
    ├── modeling/                    # Experiments and workflows
    │   ├── seasonal_ensemble_experiment.py   # Main seasonal forecast orchestrator
    │   ├── run_ensemble.sh                   # Bash launch script
    │   ├── optimization/                     # Calibration scripts
    │   └── 02_lumped/, 03_semiDistributed/  # Domain-specific notebooks
    ├── notebooks/                   # Analysis notebooks
    └── observed_data/               # Streamflow observations
```

---

## SUMMA — The Primary Hydrological Model

### What is SUMMA?

SUMMA (Structure for Unifying Multiple Modeling Alternatives) is a physics-based, modular hydrological model written in Fortran. It simulates coupled water and energy balance processes at the HRU (Hydrological Response Unit) scale.

**Key papers:**
- Clark et al. (2015a) WRR, doi:10.1002/2015WR017200 — Part 1: SUMMA framework/philosophy
- Clark et al. (2015b) WRR, doi:10.1002/2015WR017198 — Part 2: Demonstration and evaluation
- NCAR Technical Note 526 (2024): [opensky.ucar.edu/system/files/2024-08/technotes_526.pdf](https://opensky.ucar.edu/system/files/2024-08/technotes_526.pdf)
- Documentation: [summa.readthedocs.io](https://summa.readthedocs.io/en/latest/)

### Spatial Hierarchy

- **HRU** (Hydrological Response Unit): Basic computational unit with homogeneous properties
- **GRU** (Grouped Response Unit): Basin-scale grouping of HRUs; used for routing and aggregation
- Lumped mode = 1 HRU/GRU; distributed = many HRUs per GRU

### SUMMA Physics Options (`modelDecisions.txt`)

The physics are controlled by `modelDecisions.txt`. Current project decisions:

| Decision | Option | Description |
|----------|--------|-------------|
| `soilCatTbl` | `ROSETTA` | Soil parameterization (pedotransfer functions) |
| `vegeParTbl` | `MODIFIED_IGBP_MODIS_NOAH` | Vegetation classification |
| `soilStress` | `NoahType` | Soil moisture control on stomatal resistance |
| `stomResist` | `Jarvis` or `BallBerry` | Stomatal resistance formulation |
| `num_method` | `itertive` | Iterative solver |
| `groundwatr` | `bigBuckt` or `noXplicit` | Explicit aquifer vs. no explicit groundwater |
| `snowIncept` | `lightSnow` | Snow interception |
| `snowLayers` | `CLM_2010` | Layer combination/subdivision |
| `compaction` | `anderson` | Snow compaction (Anderson 1976) |
| `f_Richards` | `mixdform` | Mixed-form Richards equation for soil |
| `spatial_gw` | `localColumn` | Column-by-column groundwater (no lateral flow) |
| `subRouting` | `timeDlay` | Time-delay hillslope routing |
| `astability` | `louisinv` | Atmospheric stability (Louis et al. 1979) |
| `LAI_method` | `monTable` | LAI from monthly lookup table |

### SUMMA Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `fileManager.txt` | settings dir | Master control: paths, sim times, all file pointers |
| `modelDecisions.txt` | settings dir | Physics option selection |
| `localParamInfo.txt` | settings dir | HRU-level parameters with bounds (186 params) |
| `basinParamInfo.txt` | settings dir | Basin/GRU-level parameters |
| `outputControl.txt` | settings dir | Which variables to output and how (instant/sum/mean) |
| `forcingFileList.txt` | settings dir | List of forcing netCDF input files |
| `attributes.nc` | settings dir | HRU static attributes (elevation, slope, soil/veg type, area) |
| `coldState.nc` | settings dir | Initial state (temperatures, moisture, snow, groundwater) |
| `trialParams.nc` | settings dir | Adjustable parameter values for calibration |
| `TBL_*.TBL` | settings dir | Lookup tables: VEGPARM, SOILPARM, GENPARM, MPTABLE |

**Base templates** are in `0_base_settings/SUMMA/` (and `ess-project/0_base_settings/SUMMA/`). The preprocessor copies and modifies these for each experiment.

### Forcing Data Format

SUMMA forcing files are **netCDF** with:

```
Dimensions: time × hru
Variables:
  hruId        — HRU identifiers
  data_step    — Timestep in seconds (e.g., 3600 for hourly)
  pptrate      — Precipitation rate (mm/s)
  airtemp      — Air temperature (K)
  airpres      — Air pressure (Pa)
  spechum      — Specific humidity (kg/kg)
  windspd      — Wind speed (m/s)
  LWRadAtm     — Longwave radiation (W/m²)
  SWRadAtm     — Shortwave radiation (W/m²)

Time coordinate: seconds since 1990-01-01 00:00:00 (SUMMA standard — critical!)
```

Forcing preprocessing converts ERA5/GRIDMET data to this format, handles NaN interpolation, and optionally applies lapse rate corrections for elevation.

### Key SUMMA Output Variables

| Variable | Type | Description |
|----------|------|-------------|
| `averageRoutedRunoff` | sum | Routed runoff → fed to mizuRoute |
| `scalarTotalRunoff` | sum | Total surface + subsurface runoff |
| `scalarSWE` | mean | Snow water equivalent |
| `scalarSnowDepth` | mean | Snow depth |
| `scalarTotalET` | sum | Total evapotranspiration |
| `scalarAquiferStorage` | mean | Groundwater storage (bigBuckt only) |
| `scalarAquiferBaseflow` | sum | Baseflow to stream |
| `mLayerVolFracLiq` | mean | Soil moisture per layer |
| `mLayerTemp` | mean | Soil temperature per layer |
| `scalarInfiltration` | sum | Infiltration rate |
| `scalarSurfaceRunoff` | sum | Overland flow |

`outputControl.txt` format: `varname | timestep | instant | sum | mean | var | min | max | mode`

### Key Parameters for Calibration

**HRU-level** (`localParamInfo.txt`, file: `trialParams.nc`):
- `k_soil` — Saturated hydraulic conductivity
- `theta_sat` — Saturated soil moisture
- `theta_res` — Residual soil moisture
- `rootingDepth` — Maximum rooting depth
- `albedoMax`, `albedoMinWinter` — Snow albedo bounds

**Basin-level** (`basinParamInfo.txt`):
- `basin__aquiferHydCond` — Aquifer hydraulic conductivity
- `basin__aquiferScaleFactor` — Storage scaling
- `basin__aquiferBaseflowExp` — Baseflow recession exponent
- `routingGammaShape`, `routingGammaScale` — Hillslope routing shape

### Running SUMMA

```bash
summa -m fileManager.txt
```

Python interface via `utils/models/summa_utils.py`:
- `SummaPreProcessor` — generates all config/forcing files
- `SummaRunner` — executes SUMMA (serial or parallel)
- `SUMMAPostprocessor` — extracts results, computes water balance

---

## Companion Tools

All tools are installed at `/home/dlhogan/tools/src/`. The workflow dependency order:

```
gistool → TauDEM → datatool → SUMMA → mizuRoute
                  ↕
               ostrich (calibration loop)
```

### TauDEM (Terrain Analysis Using Digital Elevation Models)

**Location:** `/home/dlhogan/tools/src/TauDEM`
**Purpose:** DEM analysis → flow direction, accumulation, watershed delineation
**Key executables:** `pitremove`, `d8flowdir`, `aread8`, `streamnet`, `gagewatershed`
**Outputs:** GeoTIFF rasters (flow direction, accumulation, catchment masks)
**Used by:** `utils/geospatial/geofabric_utils.py` for domain definition

### gistool (Geospatial Information System Tool)

**Location:** `/home/dlhogan/tools/src/gistool`
**Purpose:** Downloads and processes geospatial datasets (DEM, soil, land cover, vegetation)
**Data sources:** MERIT Hydro, Landsat, MODIS, SoilGrids, GSDE, depth-to-bedrock
**Outputs:** GeoTIFF or NetCDF
**Used by:** `utils/data/attribute_processing.py` for spatial attribute extraction
**HPC support:** SLURM/PBS/LFS schedulers configured for multiple clusters

### datatool (Data Management Tool)

**Location:** `/home/dlhogan/tools/src/datatool`
**Purpose:** Downloads and standardizes meteorological/climate forcing data
**Data sources:** ERA5, DAYMET, ECCC-RDRS, NASA NEX-GDDP-CMIP6, NCAR CONUS, others
**Outputs:** NetCDF with standardized meteorological variables
**Used by:** `utils/data/data_manager.py` for forcing acquisition
**HPC support:** Same cluster/scheduler configurations as gistool

### mizuRoute (River Routing Model)

**Location:** `/home/dlhogan/tools/src/mizuRoute`
**Executable:** `/home/dlhogan/tools/src/mizuRoute/route/bin/route_runoff`
**Purpose:** Routes lateral flows from SUMMA through the stream network to outlet
**Inputs:** SUMMA's `averageRoutedRunoff` (netCDF), river network topology, spatial weights
**Outputs:** Streamflow timeseries at reaches/outlet (netCDF)
**Used by:** `utils/models/mizuroute_utils.py`
**Config files:** Control file + `param.nml.default`

### ostrich (Optimization Toolkit)

**Location:** `/home/dlhogan/tools/src/ostrich`
**Executable:** `/home/dlhogan/tools/bin/OstrichMPI` (MPI-enabled)
**Purpose:** Parameter calibration and uncertainty estimation for SUMMA/mizuRoute
**Algorithms:** DDS, SCE-UA, Genetic Algorithm, Particle Swarm, NSGAII, GLUE, Latin Hypercube
**Used by:** `utils/optimization/ostrich.py`; wraps SUMMA in an optimization loop

---

## CONFLUENCE Workflow

### Full Automated Workflow

```python
from CONFLUENCE import CONFLUENCE
confluence = CONFLUENCE(config_path=Path("config.yaml"))
confluence.run_workflow()
```

Or step-by-step:
```bash
confluence --config ess-project/0_config_files/config_East_River_lumped.yaml
```

**Workflow steps (in order):**
1. `setup_project` — directory structure
2. `acquire_attributes` — DEM, soil, land cover (via gistool/TauDEM)
3. `define_domain` — watershed boundary delineation (via TauDEM)
4. `discretize_domain` — HRU/GRU creation (elevation bands, soil class, etc.)
5. `acquire_forcings` — download ERA5/GRIDMET (via datatool)
6. `run_model_agnostic_preprocessing` — interpolation, lapse rates
7. `preprocess_models` — generate SUMMA/mizuRoute config files
8. `run_models` — execute SUMMA + mizuRoute
9. `calibrate_parameters` — DDS/PSO/SCE-UA optimization
10. `run_benchmarking` — NSE/KGE/RMSE evaluation
11. `analyze_results` — statistical analysis

### Seasonal Ensemble Experiment (ESS Project Focus)

**Entry points:**
- `ess-project/modeling/run_ensemble.sh` — bash launcher
- `ess-project/modeling/seasonal_ensemble_experiment.py` — main orchestrator

**Strategy:** Replace target year's forcing for one season (OND/JFM/AMJ/JAS) with that same season from historical donor years (e.g., 2003–2022). Creates ~80 ensemble members per target year.

**Seasons:**
- OND: Oct–Nov–Dec (Fall)
- JFM: Jan–Feb–Mar (Winter)
- AMJ: Apr–May–Jun (Spring)
- JAS: Jul–Aug–Sep (Summer)

**Workflow steps:**
1. `step1_optimize` — calibrate parameters with DDS → `best_parameters.csv`
2. `step2_long_term_run` — 20-year baseline spin-up run
3. `step2b_create_warm_state` — extract model state at initialization time
4. `step3_target_year_baseline` — unperturbed reference run
5. `step4_build_ensembles` — generate perturbed forcing files
6. `step5_generate_script` — parallel ensemble runner script

**Output structure:**
```
YYYYMMDD_experiment_name/
├── settings/              # SUMMA configuration snapshot
├── ensemble/results/
│   ├── baseline_longterm/ # 20-year spin-up output
│   ├── baseline_target/   # Target year reference
│   └── OND/, JFM/, AMJ/, JAS/  # Seasonal ensemble outputs
└── config_*.yaml          # Configuration snapshot
```

---

## Configuration YAML Structure

**Key config files:**
- Root template: `0_config_files/config_template.yaml` (full documentation of all params)
- East River lumped: `ess-project/0_config_files/config_East_River_lumped.yaml`
- Tuolumne variants: `ess-project/0_config_files/config_Tuolumne_lumped_*.yaml`

**Critical parameters:**
```yaml
# Domain
DOMAIN_NAME: East_River_lumped
EXPERIMENT_ID: 20260318_experiment_name
DOMAIN_DEFINITION_METHOD: lumped    # lumped | point | subset | delineate
POUR_POINT_COORDS: "38.7/-106.9"

# Time
EXPERIMENT_TIME_START: "2007-10-01 00:00"
EXPERIMENT_TIME_END: "2016-09-30 23:00"
CALIBRATION_PERIOD: ["2007-10-01", "2012-09-30"]
EVALUATION_PERIOD: ["2012-10-01", "2016-09-30"]
FORCING_TIME_STEP_SIZE: 3600        # seconds (must match forcing data)

# Model
HYDROLOGICAL_MODEL: SUMMA
ROUTING_MODEL: mizuRoute
SUMMA_EXE: summa
SETTINGS_SUMMA_CONNECT_HRUS: true   # true for distributed, false for lumped

# Forcing
FORCING_DATASET: ERA5
APPLY_LAPSE_RATE: true
LAPSE_RATE: -0.0065                 # K/m

# Calibration
PARAMS_TO_CALIBRATE: "k_soil, theta_sat, theta_res, rootingDepth"
BASIN_PARAMS_TO_CALIBRATE: "basin__aquiferScaleFactor, basin__aquiferHydCond"
OPTIMISATION_METHODS: DDS
OPTIMISATION_OBJECTIVES: KGE
MAX_ITERATIONS: 1000
```

---

## Key Python Modules Reference

| Module | Path | Role |
|--------|------|------|
| `summa_utils.py` | `utils/models/` | SUMMA pre/run/post-processor (114KB, main model interface) |
| `mizuroute_utils.py` | `utils/models/` | mizuRoute pre/run/post-processor |
| `agnosticPreProcessor.py` | `utils/data/` | Model-agnostic forcing prep (98KB) |
| `attribute_processing.py` | `utils/data/` | Geospatial attribute extraction (255KB) |
| `iterative_optimizer.py` | `utils/optimization/` | DDS, PSO, SCE-UA (306KB) |
| `calibration_targets.py` | `utils/optimization/` | NSE, KGE, RMSE objective functions |
| `geofabric_utils.py` | `utils/geospatial/` | DEM/geofabric processing (86KB) |
| `discretization_utils.py` | `utils/geospatial/` | HRU discretization (76KB) |
| `workflow_orchestrator.py` | `utils/project/` | Workflow step coordination |
| `seasonal_ensemble_experiment.py` | `ess-project/modeling/` | Ensemble forecast orchestrator (136KB) |
| `forcing_perturber.py` | `utils/custom/` | Ensemble forcing generation |

---

## SUMMA Physics — Key Concepts

### Energy Balance
SUMMA solves coupled water and energy balance equations at each HRU:
- Net radiation = incoming SW + incoming LW − reflected SW − emitted LW
- Sensible heat: atmospheric stability function (Louis et al. 1979)
- Latent heat: transpiration (Ball-Berry/Jarvis stomatal resistance) + soil evaporation + sublimation
- Canopy energy balance: iterative solution for leaf temperature

### Water Balance
- **Infiltration:** Green-Ampt or equivalent, constrained by saturated hydraulic conductivity
- **Soil moisture:** Richards equation (mixed form) solved iteratively
- **Root water uptake:** Controlled by wilting point and soil moisture stress (NoahType)
- **Baseflow:** Power-law function of aquifer storage (`bigBuckt` mode)
- **Surface runoff:** Saturation excess or infiltration excess

### Snow Pack
- **Layering:** CLM 2010 method — automatic subdivision and combination of snow layers
- **Density:** New snow density from Hedstrom & Pomeroy (1998), evolves via Anderson (1976) compaction
- **Albedo:** Exponential decay (conDecay method)
- **Melt:** Energy-balance driven; refreezing possible in surface layer
- **Interception:** Light snow option in canopy

### Numerical Approach
- Iterative solver with analytical Jacobians
- Adaptive timestepping: minimum 1s, maximum 3600s (one forcing step)
- Convergence tolerances in `localParamInfo.txt` (e.g., `relConvTol_liquid = 0.001`)

---

## Scientific Foundation (Clark et al. 2015)

### Philosophy: Method of Multiple Working Hypotheses

Clark et al. (2015a, doi:10.1002/2015WR017198) articulate SUMMA's core design philosophy: rather than selecting a single "best" model structure a priori, SUMMA enables **simultaneous, systematic evaluation of multiple competing hypotheses** about hydrological processes. This mirrors Chamberlin's (1890) scientific method applied to hydrology.

**Two fundamental propositions:**
1. Most hydrological models share the same underlying conservation equations — differences between models arise primarily from choices in (a) spatial variability representations and (b) flux parameterizations.
2. The key scientific questions are *not* which model is "best" in absolute terms, but rather which process representations are most important under which conditions.

**Four design requirements for systematic model analysis:**
1. A common set of conservation equations applicable across many model configurations
2. A flexible spatial representation (GRU/HRU hierarchy)
3. A library of alternative process parameterizations for each flux
4. A robust numerical solver capable of handling many physics combinations

### Spatial Hierarchy — Formal Definitions

**GRU (Grouped Response Unit):**
- Spatially contiguous units at the basin scale
- Each GRU maps to one or more stream reaches in the routing network
- GRUs aggregate HRU outputs (area-weighted) to produce lateral inflows to mizuRoute
- Designed to be coarse enough for computational efficiency, fine enough to capture spatial patterns

**HRU (Hydrological Response Unit):**
- Basic computational unit within a GRU; assumed to have homogeneous properties
- Each HRU is a 1-D vertical column: canopy top → active groundwater layer
- Multiple HRUs within one GRU can exchange lateral fluxes (if `SETTINGS_SUMMA_CONNECT_HRUS: true`)
- HRUs characterized by: elevation, slope, aspect, soil type, vegetation type, area fraction

**Model domain:** Canopy top through the vadose zone to the active groundwater layer — corresponds to the Earth's **Critical Zone** (the thin near-surface layer where rock, soil, water, air, and living organisms interact).

**"Model mimicry":** SUMMA can reproduce the behavior of many existing models (VIC, CLM, Noah, SAC-SMA, etc.) by selecting appropriate combinations of physics options — enabling direct, controlled model inter-comparison.

### Conservation Equations (Clark et al. 2015b, doi:10.1002/2015WR017200)

SUMMA solves the following coupled conservation equations at each HRU timestep:

**Canopy thermodynamics** (Eq. 4/25):
```
dU_can/dt = R_net,can + H_below + LE_can − H_above − LE_above − M_can
```
Where U_can = canopy internal energy, R_net = net radiation, H = sensible heat, LE = latent heat, M = melt.

**Canopy air space** (Eq. 6/26):
- Instantaneous equilibrium assumed (no storage in canopy air)
- Aerodynamic resistances couple canopy, below-canopy air, and atmosphere

**Snow/soil thermodynamics** (Eq. 10/29):
```
ρ·c·∂T/∂t = −∂q_H/∂z + Q_source
```
Where q_H = conductive heat flux, Q_source = phase change energy (latent heat of fusion).

**Canopy hydrology** (Eq. 12/27–28):
```
dS_can/dt = P_through + P_drip − ET_can − throughfall
```
Controls canopy interception, throughfall, stem flow, and drip.

**Snow hydrology** (Eq. 13/30–31):
```
dSWE/dt = snowfall + refreeze − melt − sublimation
```
Multi-layer; mass and energy balanced per layer; liquid water percolates through snow.

**Soil hydrology — Richards equation** (Eq. 15/32):
```
∂θ/∂t = −∂q/∂z + S_root
```
Where θ = volumetric water content, q = Darcy flux (van Genuchten hydraulic functions), S_root = root water uptake sink term. The **mixed-form** (`mixdform`) discretizes as:
```
(∂θ/∂ψ)·∂ψ/∂t = −∂q/∂z + S_root
```
This is more numerically stable than the ψ-only or θ-only forms near saturation.

**Aquifer** (Eq. 20):
```
dS_aq/dt = drainage_from_soil − baseflow
baseflow = K_aq · S_aq^exp
```
Only active when `groundwatr = bigBuckt`; when `noXplicit`, aquifer storage is bypassed.

**Snow albedo** (Eq. 24):
```
α = α_min + (α_max − α_min) · exp(−decay_rate · t_since_snowfall)
```
Exponential decay (`conDecay`) between fresh-snow maximum and old-snow minimum.

### Process Parameterization Options (Table 1, Part 2)

Key decision points and available options:

| Process | Option Name | Description |
|---------|-------------|-------------|
| Stomatal resistance | `Jarvis` | Empirical multiplicative stress functions |
| Stomatal resistance | `BallBerry` | Photosynthesis-coupled (medlyn/Ball-Berry-Collatz) |
| Below-canopy wind | `logBelowCanopy` | Log wind profile below canopy |
| Below-canopy wind | `exponential` | Exponential attenuation |
| Snowmelt | `temperature_index` | Simple degree-day |
| Snowmelt | `energy_balance` | Full energy balance (SUMMA default) |
| Infiltration | `constantInfRt` | Constant infiltration rate |
| Infiltration | `SoilProfile` | Variable rate from soil profile state |
| Lateral flow | `localColumn` | No lateral exchange between HRUs (1-D) |
| Lateral flow | `distributed` | Lateral redistribution of soil moisture |
| Groundwater | `bigBuckt` | Explicit linear reservoir aquifer |
| Groundwater | `noXplicit` | No explicit aquifer; drainage exits domain |
| Groundwater | `qTopmodl` | TOPMODEL-based groundwater parameterization |
| Soil parameterization | `ROSETTA` | Pedotransfer functions from texture/OC |
| Albedo decay | `conDecay` | Continuous exponential decay |
| Albedo decay | `dmSnfall` | Decay with snowfall events |

### Numerical Scheme

**Method:** Implicit Euler time integration with Newton-Raphson iteration.

**Operator splitting:** Rather than solving all conservation equations simultaneously (which would be computationally prohibitive), SUMMA splits into:
1. **Phase 1:** Solve energy balance for each layer (canopy, snow layers, soil layers) — compute temperatures
2. **Phase 2:** Use temperatures to drive water balance (melt, ET, infiltration, drainage)
3. **Phase 3:** Update snow/soil layer thicknesses and merge/split snow layers

**Adaptive timestepping:** If Newton-Raphson fails to converge within the tolerance at the forcing timestep (e.g., 3600s), the solver halves the substep and retries. Minimum substep is 1 second.

**Analytical Jacobians:** SUMMA derives Jacobians analytically (not numerically) for the Newton-Raphson iteration — faster and more accurate convergence than finite-difference approximation.

### Key Findings from Case Studies (Part 2)

The Clark et al. (2015b) paper evaluated SUMMA across three sites (Reynolds Creek, Senator Beck, Reynolds Mountain East). Key insights applicable to this project:

1. **Stomatal resistance:** Ball-Berry (`BallBerry`) consistently outperformed Jarvis at all sites — it captures vegetation feedback on transpiration more physically correctly. **Prefer `BallBerry` for East River.**

2. **Below-canopy wind profile:** The log wind profile (`logBelowCanopy`) was critical for correctly partitioning sensible and latent heat below forest canopies. Exponential attenuation underestimated below-canopy turbulence.

3. **Dust-on-snow albedo:** At Senator Beck (Colorado, similar to East River), dust loading significantly accelerated snowmelt through albedo reduction. The albedo decay rate parameter (`albedoDecayRate`) needs calibration against SWE observations.

4. **Lateral flow vs. 1-D Richards:** Enabling lateral redistribution of soil moisture between HRUs (`distributed` lateral flow) improved runoff simulation compared to purely vertical 1-D Richards equation (`localColumn`) in areas with terrain-driven subsurface flow. However, lateral flow adds computational cost and requires careful HRU delineation.

5. **GRU spatial resolution:** Coarser GRU resolution smooths spatial variability but speeds computation. For lumped East River runs (1 HRU/GRU), spatial averaging is already implicit.

6. **Groundwater representation:** The explicit bucket aquifer (`bigBuckt`) improved baseflow recession curves but required calibration of aquifer parameters. For sites with strong shallow groundwater connection to streams, `bigBuckt` is preferable to `noXplicit`.

---

## Data Sources Used

| Dataset | Type | Used For |
|---------|------|---------|
| ERA5 | Atmospheric reanalysis | Primary forcing (hourly, global) |
| GRIDMET | Gridded surface meteo | Alternative/supplemental forcing (western US) |
| MERIT Hydro | DEM-derived hydrography | Stream network, watershed delineation |
| MODIS | Satellite | Land cover, LAI |
| SoilGrids / ROSETTA | Soil data | Hydraulic parameters, soil classification |
| USGS streamflow | Observations | Calibration and validation target |

---

## Common File Patterns

- `*.nc` — All model inputs and outputs are netCDF4
- `fileManager.txt` — SUMMA's master control file (points to everything else)
- `config_*.yaml` — CONFLUENCE experiment configuration
- `best_parameters.csv` — Output of calibration, input to ensemble runs
- `forcingFileList.txt` — List of forcing netCDF files for SUMMA
- `YYYYMMDD_<experiment>/` — Timestamped experiment output directories

---

## Gotchas and Important Notes

1. **SUMMA time coordinate:** Must be seconds since `1990-01-01 00:00:00`. Forcing files from ERA5/GRIDMET must be converted — this is handled in `summa_utils.py` but easy to break.

2. **HRU ID consistency:** `hruId` must match across `attributes.nc`, `coldState.nc`, `trialParams.nc`, and all forcing files.

3. **`outputControl.txt` flags:** `instant`, `sum`, and `mean` are mutually exclusive per variable. Use `sum` for fluxes (rates accumulate over timestep), `mean` for state variables averaged over sub-steps.

4. **`bigBuckt` vs `noXplicit`:** Lumped East River runs use `noXplicit` (no explicit groundwater); distributed runs typically use `bigBuckt`. Basin params differ between modes.

5. **`SETTINGS_SUMMA_CONNECT_HRUS`:** Set to `true` for distributed (enables lateral flow between HRUs), `false` for lumped.

6. **Ensemble experiments create many runs:** Each ensemble member modifies forcing files in-place. The `forcing_perturber.py` tool handles this; don't modify forcing files manually during runs.

7. **Calibration period vs evaluation period:** CONFLUENCE enforces these separately. The config `CALIBRATION_PERIOD` and `EVALUATION_PERIOD` control which data are used for each.
