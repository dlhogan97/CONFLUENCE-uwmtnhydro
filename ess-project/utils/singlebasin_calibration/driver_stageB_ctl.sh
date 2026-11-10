#!/bin/bash
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/singlebasin_calibration
LOG=/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/stageB/driver_ctl.log
for m in M0 M1; do pop=12; [ $m = M1 ] && pop=10
  echo "$(date +%T) start ctl $m" >> $LOG; python3 run_stageB.py ctl $m $pop 30 8 >> $LOG 2>&1
done; echo "$(date +%T) STAGE-B ctl FITS DONE" >> $LOG
