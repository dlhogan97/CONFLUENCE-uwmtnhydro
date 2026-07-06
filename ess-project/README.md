# ESS Project — UW Mountain Hydrology

Research project using the CONFLUENCE framework to study seasonal streamflow
predictability and distributed hydrological modeling in snow-dominated basins.

---

## Active Domains

| Domain | Config | Groundwater | Status |
|--------|--------|-------------|--------|
| East River, CO (distributed elevAspect) | `0_config_files/config_East_River_distributed_elevAspect.yaml` | `bigBuckt` (explicit aquifer) | active |
| Tuolumne River, CA (distributed elevAspect) | `0_config_files/config_Tuolumne_River_distributed_elevAspect.yaml` | `bigBuckt` | active |

Model output lives under `/scratch/dlhogan/ess-project-data/domain_<name>/`.

---

## Directory Structure

```
ess-project/
├── 0_config_files/          # Active YAML configs (2 domains)
│   └── archived/            # Superseded configs (lumped, noXplicit, elevTPI variants)
│
├── 0_base_settings/         # SUMMA / mizuRoute / FUSE template files
│
├── 1_forcing/               # Forcing download scripts (ERA5, GRIDMET)
│
├── figures/                 # Publication figures
│   ├── draft/               # Working figures
│   ├── final/               # Submission-ready
│   ├── src/                 # Figure generation scripts
│   └── figure_components/   # Panels and sub-figures
│
├── modeling/
│   ├── baseline/            # Distributed run scripts and HRU setup
│   │   ├── distributed_settings_builder.py
│   │   ├── run_distributed_workflow.py
│   │   ├── run_longterm_baseline.py
│   │   ├── update_settings_for_hrus.py
│   │   └── aspect/ elevation/ snow_heterogeneity/  (HRU band configs)
│   │
│   ├── calibration/         # Staged multi-objective calibration
│   │   ├── staged_optimizer.py          # Main optimizer
│   │   ├── pre_calibrate_routing.py     # Linear-reservoir routing pre-cal
│   │   ├── optimization_config_bigBuckt.yaml
│   │   ├── optimization_config_bigBuckt_stages2to4.yaml
│   │   ├── optimization_config_bigBuckt_restart_gw.yaml
│   │   ├── optimization_config_qTopmodel*.yaml
│   │   ├── stage_configs/               # Per-stage YAML (snow→ET→soil→GW→routing)
│   │   └── observations/                # Streamflow obs for calibration
│   │
│   ├── synthetic/           # Seasonal ensemble + synthetic experiments
│   │   ├── seasonal_ensemble_experiment.py   # Main orchestrator
│   │   ├── run_ensemble.sh                   # tmux launcher
│   │   ├── run_emulation.sh                  # NN surrogate calibration launcher
│   │   ├── synthetic_extended_runs.py        # Synthetic climate scenario runs
│   │   ├── build_ensemble_summary.py         # Aggregate ensemble output
│   │   ├── build_synthetic_summary.py        # Aggregate synthetic output
│   │   ├── east_linRes_params.json           # Routing params (East River)
│   │   └── tuolumne_linRes_params.json       # Routing params (Tuolumne)
│   │
│   └── _archive/            # Superseded scripts and notebooks
│
├── notebooks/
│   ├── analysis/            # Result synthesis (ensemble, runoff ratios, SWE-Q)
│   ├── evaluation/          # Model vs. observations (ASO, streamflow, fSCA)
│   ├── exploration/         # Data EDA (SNOTEL, soil moisture)
│   └── _archive/sandbox/    # Scratch notebooks
│
├── observed_data/           # USGS streamflow and other obs
│   ├── East_River/
│   └── Tuolumne_River/
│
└── results/                 # Processed output / summary tables
    ├── data/
    ├── east/
    └── tuolumne/
```

---

## Key Workflows

### 1. Staged Calibration

Multi-stage optimization (snow → ET → soil → groundwater → routing) using DDS.

```bash
cd ess-project/modeling/calibration

# Stage 1 (snow) through all stages for bigBuckt groundwater:
python staged_optimizer.py --config optimization_config_bigBuckt.yaml

# Resume from groundwater stage:
python staged_optimizer.py --config optimization_config_bigBuckt_restart_gw.yaml

# Pre-calibrate linear-reservoir routing parameters:
python pre_calibrate_routing.py --config optimization_config_bigBuckt.yaml
```

### 2. Long-Term Baseline Run

Spins up a 20-year baseline for use as ensemble initial conditions.

```bash
cd ess-project/modeling/baseline
python run_longterm_baseline.py --config ../../0_config_files/config_East_River_distributed_elevAspect.yaml
```

### 3. Seasonal Ensemble Experiment

Generates ~80-member streamflow forecast ensemble by swapping one season's
forcing with historical donor years (2001–2020), then running SUMMA in parallel.

```bash
tmux new -s ensemble
cd ess-project/modeling/synthetic

# Full run (optimize → baseline → ensemble setup):
bash run_ensemble.sh \
    --config ../../0_config_files/config_East_River_distributed_elevAspect.yaml \
    --target-year 2021

# Skip optimization, load existing parameters:
bash run_ensemble.sh \
    --config ../../0_config_files/config_East_River_distributed_elevAspect.yaml \
    --skip-optimization \
    --best-params-csv /scratch/.../best_parameters.csv \
    --target-year 2021
```

### 4. Neural Network Parameter Emulation

Trains a surrogate model on SUMMA parameter–KGE pairs for sensitivity analysis.

```bash
tmux new -s emulation
cd ess-project/modeling/synthetic
bash run_emulation.sh \
    --config ../../0_config_files/config_East_River_distributed_elevAspect.yaml \
    --n-train 800 --epochs 500
```

---

## Environment

```bash
conda activate ess-project-env
```

Dependencies: `environment.forcing.yml` (root) / `CONFLUENCE-base.yml` (root).

---

## CONFLUENCE Framework

This project is built on top of the CONFLUENCE hydrological modeling framework
(see [root README](../README.md) and [CLAUDE.md](../CLAUDE.md)).

Key framework modules used here:

| Module | Path | Role |
|--------|------|------|
| `summa_utils.py` | `utils/models/` | SUMMA pre/run/post-processing |
| `agnosticPreProcessor.py` | `utils/data/` | Forcing preparation |
| `iterative_optimizer.py` | `utils/optimization/` | DDS / PSO / SCE-UA |
| `linear_reservoir.py` | `utils/custom/` | Two-reservoir routing calibration |
| `adjust_settings.py` | `utils/custom/` | SUMMA settings patching |
| `calc.py` | `utils/custom/` | Empirical LW radiation correction |
