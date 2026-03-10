#!/bin/bash
# Run inside tmux:  tmux new -s ensemble
# Then:             bash ensemble_runner.sh
# Detach:           Ctrl+B, D
# Reconnect:        tmux attach -t ensemble

set -e

config_file="../0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml"

# 1. Activate env
source ~/miniforge3/bin/activate
conda activate CONFLUENCE-base
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling

# 2. Prepare ensembles and generate run script
python -c "
from seasonal_ensemble_experiment import SeasonalEnsembleExperiment
exp = SeasonalEnsembleExperiment('${config_file}', max_workers=12)
exp.step1_optimize(skip_if_exists=False)
exp.step4_build_ensembles(target_year=2021, donor_years=list(range(1999, 2021)))
script = exp.step5_generate_script(target_year=2021, max_workers=12)
print(f'Script ready: {script}')
"

echo "Setup complete. Run ensemble with:"
echo "  nohup bash run_ensemble.sh > $LOGS/ensemble_main.log 2>&1 &"