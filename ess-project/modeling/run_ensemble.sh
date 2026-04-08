#!/bin/bash
# Run inside tmux:  tmux new -s ensemble
# Then:             bash run_ensemble.sh --config path/to/config.yaml [options]
# Detach:           Ctrl+B, D
# Reconnect:        tmux attach -t ensemble

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_ensemble.sh --config <config_file> [options]

Required:
  -c, --config PATH              CONFLUENCE config YAML

Optional:
  -e, --existing-experiment REF  Reuse an existing experiment workspace
                                 REF can be a folder name under simulations/
                                 or an absolute path
            --experiment-id NAME       Override EXPERIMENT_ID for new workspace name
                                                                 (used only when not reusing existing workspace)
      --allow-existing-overwrite Allow in-place updates when reusing an
                                 existing experiment workspace
      --seed-params-csv PATH     Seed optimization from an existing
                                 best_parameters.csv file
            --best-params-csv PATH     Load this best_parameters.csv directly when
                                                                 --skip-optimization is used
      --skip-optimization        Do not run a fresh optimization; load
                                 existing best parameters instead
      --baseline-start TEXT      Long-term run start timestamp
                                 default: 2000-10-01
      --baseline-end TEXT        Long-term run end timestamp
                                 default: 2021-09-30
      --warm-state-time TEXT     Timestamp to extract warm state
                                 default: 2020-09-30 23:00
      --target-year YEAR         Target water year
                                 default: 2021
      --donor-start YEAR         First donor year
                                 default: 2001
      --donor-end YEAR           Last donor year
                                 default: 2020
      --max-workers N            Worker count for generated script
                                 default: 12
  -h, --help                     Show this help text

Examples:
  bash run_ensemble.sh \
      --config ../0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml

  bash run_ensemble.sh \
      --config ../0_config_files/config_East_River_lumped_seasonal_bigBuckt.yaml \
      --seed-params-csv /scratch/.../best_parameters.csv

  bash run_ensemble.sh \
      --config ../0_config_files/config_East_River_lumped_seasonal_bigBuckt.yaml \
      --experiment-id bigBuckt_rerun \
      --skip-optimization \
      --best-params-csv /scratch/.../best_parameters.csv

  bash run_ensemble.sh \
      --config ../0_config_files/config_East_River_lumped_seasonal_bigBuckt.yaml \
      --existing-experiment 20260316_bigBuckt \
      --allow-existing-overwrite
EOF
}

CONFIG_FILE=""
EXISTING_EXPERIMENT=""
EXPERIMENT_ID_OVERRIDE=""
ALLOW_EXISTING_OVERWRITE=0
SEED_PARAMS_CSV=""
BEST_PARAMS_CSV=""
SKIP_OPTIMIZATION=0
BASELINE_START="2000-10-01"
BASELINE_END="2021-09-30"
WARM_STATE_TIME="2020-09-30 23:00"
TARGET_YEAR=2021
DONOR_START=2001
DONOR_END=2020
MAX_WORKERS=12

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -e|--existing-experiment)
            EXISTING_EXPERIMENT="$2"
            shift 2
            ;;
        --experiment-id)
            EXPERIMENT_ID_OVERRIDE="$2"
            shift 2
            ;;
        --allow-existing-overwrite)
            ALLOW_EXISTING_OVERWRITE=1
            shift
            ;;
        --seed-params-csv)
            SEED_PARAMS_CSV="$2"
            shift 2
            ;;
        --best-params-csv)
            BEST_PARAMS_CSV="$2"
            shift 2
            ;;
        --skip-optimization)
            SKIP_OPTIMIZATION=1
            shift
            ;;
        --baseline-start)
            BASELINE_START="$2"
            shift 2
            ;;
        --baseline-end)
            BASELINE_END="$2"
            shift 2
            ;;
        --warm-state-time)
            WARM_STATE_TIME="$2"
            shift 2
            ;;
        --target-year)
            TARGET_YEAR="$2"
            shift 2
            ;;
        --donor-start)
            DONOR_START="$2"
            shift 2
            ;;
        --donor-end)
            DONOR_END="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$CONFIG_FILE" ]]; then
                CONFIG_FILE="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$CONFIG_FILE" ]]; then
    usage
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

if [[ -n "$SEED_PARAMS_CSV" && ! -f "$SEED_PARAMS_CSV" ]]; then
    echo "ERROR: seed params CSV not found: $SEED_PARAMS_CSV" >&2
    exit 1
fi

if [[ -n "$BEST_PARAMS_CSV" && ! -f "$BEST_PARAMS_CSV" ]]; then
    echo "ERROR: best params CSV not found: $BEST_PARAMS_CSV" >&2
    exit 1
fi

if [[ "$DONOR_END" -lt "$DONOR_START" ]]; then
    echo "ERROR: donor-end must be >= donor-start" >&2
    exit 1
fi

if [[ -n "$EXISTING_EXPERIMENT" && "$ALLOW_EXISTING_OVERWRITE" -ne 1 ]]; then
    echo "ERROR: --existing-experiment reuses that workspace in place." >&2
    echo "It will overwrite copied settings, baseline files, forcing members, and ensemble outputs in that experiment folder." >&2
    echo "Re-run with --allow-existing-overwrite if that is what you want." >&2
    exit 1
fi

CONFIG_FILE=$(realpath "$CONFIG_FILE")
if [[ -n "$SEED_PARAMS_CSV" ]]; then
    SEED_PARAMS_CSV=$(realpath "$SEED_PARAMS_CSV")
fi
if [[ -n "$BEST_PARAMS_CSV" ]]; then
    BEST_PARAMS_CSV=$(realpath "$BEST_PARAMS_CSV")
fi

# Read EXPERIMENT_ID from the YAML config file for new dated workspaces.
experiment_name=$(grep -m1 '^EXPERIMENT_ID:' "$CONFIG_FILE" | awk '{print $2}')
if [[ -z "$experiment_name" ]]; then
    echo "ERROR: EXPERIMENT_ID not found in $CONFIG_FILE" >&2
    exit 1
fi

if [[ -n "$EXPERIMENT_ID_OVERRIDE" ]]; then
    experiment_name="$EXPERIMENT_ID_OVERRIDE"
fi

if [[ "$SKIP_OPTIMIZATION" -eq 1 && -n "$SEED_PARAMS_CSV" ]]; then
    echo "ERROR: --seed-params-csv cannot be combined with --skip-optimization." >&2
    echo "Use --best-params-csv to load an explicit best parameter file when skipping optimization." >&2
    exit 1
fi

if [[ "$SKIP_OPTIMIZATION" -eq 0 && -n "$BEST_PARAMS_CSV" ]]; then
    echo "ERROR: --best-params-csv is only valid with --skip-optimization." >&2
    exit 1
fi

if [[ -n "$EXISTING_EXPERIMENT" ]]; then
    echo "Experiment: reusing existing workspace -> $EXISTING_EXPERIMENT"
else
    echo "Experiment: $experiment_name  ->  folder will be: $(date +%Y%m%d)_${experiment_name}"
fi

echo "Config: $CONFIG_FILE"
echo "Seed params CSV: ${SEED_PARAMS_CSV:-<none>}"
echo "Best params CSV (skip mode): ${BEST_PARAMS_CSV:-<none>}"
echo "Run fresh optimization: $([[ "$SKIP_OPTIMIZATION" -eq 1 ]] && echo no || echo yes)"
echo "Target year: $TARGET_YEAR"
echo "Donor years: $DONOR_START to $DONOR_END"
echo "Max workers: $MAX_WORKERS"

# Lumped model = 1 HRU; no benefit from multiple OpenMP threads.
# Increase to match HRU count if switching to distributed.
export OMP_NUM_THREADS=1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGS="$SCRIPT_DIR/logs"
mkdir -p "$LOGS"

# 1. Activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ess-project-env
cd "$SCRIPT_DIR"

export RUN_CONFIG_FILE="$CONFIG_FILE"
export RUN_EXISTING_EXPERIMENT="$EXISTING_EXPERIMENT"
export RUN_SEED_PARAMS_CSV="$SEED_PARAMS_CSV"
export RUN_BEST_PARAMS_CSV="$BEST_PARAMS_CSV"
export RUN_SKIP_OPTIMIZATION="$SKIP_OPTIMIZATION"
export RUN_BASELINE_START="$BASELINE_START"
export RUN_BASELINE_END="$BASELINE_END"
export RUN_WARM_STATE_TIME="$WARM_STATE_TIME"
export RUN_TARGET_YEAR="$TARGET_YEAR"
export RUN_DONOR_START="$DONOR_START"
export RUN_DONOR_END="$DONOR_END"
export RUN_MAX_WORKERS="$MAX_WORKERS"
export RUN_EXPERIMENT_NAME="$experiment_name"

python - <<'PY'
import os
import shutil
from pathlib import Path

import pandas as pd

from seasonal_ensemble_experiment import SeasonalEnsembleExperiment


def load_existing_best_params(exp: SeasonalEnsembleExperiment):
    best_params = exp.optimizer.get_best_parameters()
    exp.best_params = best_params
    exp.ensemble_runner.runner.apply_parameters(best_params)
    params_out = exp.cfg.settings_dir / 'best_parameters.csv'
    best_params.to_csv(params_out, index=False)
    print(f'Loaded existing optimization parameters -> {params_out}')
    return best_params


def load_best_params_from_csv(exp: SeasonalEnsembleExperiment, params_csv: str):
    params_path = Path(params_csv)
    best_params = pd.read_csv(params_path)
    required = {'parameter', 'value'}
    if not required.issubset(set(best_params.columns)):
        raise ValueError(
            f'Best-params CSV must contain columns {sorted(required)}; '
            f'got {list(best_params.columns)}'
        )

    exp.best_params = best_params
    exp.ensemble_runner.runner.apply_parameters(best_params)
    params_out = exp.cfg.settings_dir / 'best_parameters.csv'
    best_params.to_csv(params_out, index=False)
    print(f'Loaded best parameters from explicit source {params_path} -> {params_out}')
    return best_params


config_path = os.environ['RUN_CONFIG_FILE']
existing_experiment = os.environ.get('RUN_EXISTING_EXPERIMENT') or None
seed_params_csv = os.environ.get('RUN_SEED_PARAMS_CSV') or None
best_params_csv = os.environ.get('RUN_BEST_PARAMS_CSV') or None
skip_optimization = os.environ.get('RUN_SKIP_OPTIMIZATION', '0') == '1'
baseline_start = os.environ['RUN_BASELINE_START']
baseline_end = os.environ['RUN_BASELINE_END']
warm_state_time = os.environ['RUN_WARM_STATE_TIME']
target_year = int(os.environ['RUN_TARGET_YEAR'])
donor_start = int(os.environ['RUN_DONOR_START'])
donor_end = int(os.environ['RUN_DONOR_END'])
max_workers = int(os.environ['RUN_MAX_WORKERS'])
experiment_name = os.environ['RUN_EXPERIMENT_NAME']
donor_years = list(range(donor_start, donor_end + 1))

exp = None
created_new_workspace = existing_experiment is None
longterm_completed = False

try:
    exp = SeasonalEnsembleExperiment(
        config_path,
        max_workers=max_workers,
        experiment_name=(None if existing_experiment else experiment_name),
        existing_experiment=existing_experiment,
    )
    print(f'Experiment workspace: {exp.cfg.experiment_workspace}')

    # grab the observation data 
    exp.step0_prepare_data()
    
    if skip_optimization:
        if best_params_csv:
            best_params = load_best_params_from_csv(exp, best_params_csv)
        else:
            best_params = load_existing_best_params(exp)
    else:
        best_params = exp.step1_optimize(
            skip_if_exists=False,
            seed_params_csv=seed_params_csv,
        )

    print('Optimization parameters ready:')
    print(best_params)

    exp.step2_long_term_run(start=baseline_start, end=baseline_end)
    longterm_completed = True
    exp.step2b_create_warm_state(extract_time=warm_state_time)
    exp.step3_target_year_baseline(target_year=target_year, prefer_continuous=True)
    exp.step4_build_ensembles(target_year=target_year, donor_years=donor_years)
    script = exp.step5_generate_script(target_year=target_year, max_workers=max_workers)
    print(f'Run script ready: {script}')

except Exception as e:
    print(f'ERROR: setup failed: {e}')
    if created_new_workspace and exp is not None:
        workspace = Path(exp.cfg.experiment_workspace)
        if workspace.exists():
            print(
                f'Preserving experiment workspace for recovery/debugging: {workspace}'
            )

        run_label = getattr(exp.optimizer, 'last_run_label', None)
        if run_label:
            sim_dir = exp.cfg.project_dir / 'simulations'
            matches = [p for p in sim_dir.glob(f'{run_label}_run_*') if p.is_dir()]
            if matches:
                print(
                    'Optimization run dir(s) also preserved: '
                    + ', '.join(str(p) for p in matches)
                )
    raise
PY

echo "Setup complete. Generated ensemble runner script under the experiment workspace."
echo "To prepare in-place inside an existing workspace, re-run this command with --existing-experiment and --allow-existing-overwrite."
echo "Logs directory: $LOGS"