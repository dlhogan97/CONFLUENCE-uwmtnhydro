#!/bin/bash
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/singlebasin_calibration
LOG=/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/stageB/driver.log
for s in old r1 r2; do for m in M0 M1; do
  pop=12; [ $m = M1 ] && pop=10
  echo "$(date +%T) start $s $m" >> $LOG
  python3 run_stageB.py $s $m $pop 30 19 >> $LOG 2>&1
done; done
echo "$(date +%T) STAGE-B FITS DONE" >> $LOG
