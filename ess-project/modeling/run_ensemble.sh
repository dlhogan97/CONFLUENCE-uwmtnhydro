#!/bin/bash
# Run inside tmux:  tmux new -s ensemble
# Then:             bash run_ensemble.sh path/to/config.yaml
# Detach:           Ctrl+B, D
# Reconnect:        tmux attach -t ensemble
#
# Usage:
#   bash run_ensemble.sh ../0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml

set -e

# --- Input ---
if [[ -z "$1" ]]; then
    echo "Usage: bash run_ensemble.sh <config_file>"
    echo "  e.g. bash run_ensemble.sh ../0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml"
    exit 1
fi

config_file="$1"

# Read EXPERIMENT_ID from the YAML config file
experiment_name=$(grep -m1 '^EXPERIMENT_ID:' "${config_file}" | awk '{print $2}')
if [[ -z "${experiment_name}" ]]; then
    echo "ERROR: EXPERIMENT_ID not found in ${config_file}"
    exit 1
fi
echo "Experiment: ${experiment_name}  →  folder will be: $(date +%Y%m%d)_${experiment_name}"

# Lumped model = 1 HRU; no benefit from multiple OpenMP threads.
# Increase to match HRU count if switching to distributed.
export OMP_NUM_THREADS=1
LOGS="./logs"
mkdir -p $LOGS

# 1. Activate env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate CONFLUENCE-base
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling

# 2. Prepare ensembles and generate run script
python -c "
from seasonal_ensemble_experiment import SeasonalEnsembleExperiment

# experiment_name is used to create dated folder: YYYYMMDD_noxPlicit/
# All results, settings, and config are saved in that folder.
exp = SeasonalEnsembleExperiment('${config_file}', max_workers=12, experiment_name='${experiment_name}')
print(f'Experiment workspace: {exp.cfg.experiment_workspace}')

# Run step 0 to prepare settings files (only needs to be done once, but is fast so we do it every time to be safe)
# exp.step0_prepare_data() 

# Run optimization first (loads latest if present, otherwise runs a new one)
best_params = exp.step1_optimize(skip_if_exists=False)
print('Optimization parameters ready:')
print(best_params)

# Output goes to: {experiment_workspace}/ensemble/results/baseline_longterm/
exp.step2_long_term_run(start='2000-10-01', end='2021-09-30')
exp.step2b_create_warm_state(extract_time='2020-09-30 23:00')
exp.step3_target_year_baseline(target_year=2021, prefer_continuous=True)
exp.step4_build_ensembles(target_year=2021, donor_years=list(range(2001, 2021)))
script = exp.step5_generate_script(target_year=2021, max_workers=12)
# print(f'Script ready: {script}')
"

echo "Setup complete. Run ensemble with:"
echo "  nohup bash run_ensemble.sh > $LOGS/ensemble_main.log 2>&1 &"