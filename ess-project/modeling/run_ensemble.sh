#!/bin/bash
# Run inside tmux:  tmux new -s ensemble
# Then:             bash ensemble_runner.sh
# Detach:           Ctrl+B, D
# Reconnect:        tmux attach -t ensemble

set -e

# Lumped model = 1 HRU; no benefit from multiple OpenMP threads.
# Increase to match HRU count if switching to distributed.
export OMP_NUM_THREADS=1

config_file="../0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml"
LOGS="./logs"
mkdir -p $LOGS

# 1. Activate env
source ~/miniforge3/bin/activate
conda activate CONFLUENCE-base
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling

# 2. Prepare ensembles and generate run script
python -c "
from seasonal_ensemble_experiment import SeasonalEnsembleExperiment
import pandas as pd

exp = SeasonalEnsembleExperiment('${config_file}', max_workers=12)

# Manual parameter set (skip optimization updates)
manual_params = {
	'tempCritRain': 274.1000,
	'k_soil': 9.4e-6,
	'theta_sat': 0.5160,
	'theta_res': 0.0270,
	'rootingDepth': 6.876,
	'basin__aquiferHydCond': 0.0010,
	'basin__aquiferScaleFactor': 50.0000,
	'routingGammaShape': 2.5000,
	'routingGammaScale': 4.6e4,
}

params_df = pd.DataFrame(
	{'parameter': list(manual_params.keys()), 'value': list(manual_params.values())}
)

exp.best_params = params_df
exp.ensemble_runner.runner.apply_parameters(params_df)

exp.step2_long_term_run(start='2000-10-01', end='2021-09-30', experiment_id='longterm_baseline')
exp.step2b_create_warm_state(extract_time='2020-09-30 23:00')
exp.step3_target_year_baseline(target_year=2021, prefer_continuous=True)
exp.step4_build_ensembles(target_year=2021, donor_years=list(range(2001, 2021)))
script = exp.step5_generate_script(target_year=2021, max_workers=12)
print(f'Script ready: {script}')
"

echo "Setup complete. Run ensemble with:"
echo "  nohup bash run_ensemble.sh > $LOGS/ensemble_main.log 2>&1 &"