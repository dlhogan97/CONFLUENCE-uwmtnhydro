#!/bin/bash
# East memory tests on the no-storage control state (2026-10-21):
#  (1) M0 / M1 refit on the longer WY2009-2025 window;
#  (2) M1 with the deep-store size FIXED (Sf_d 0.5-8 m), both windows, to map the fit-vs-memory trade-off.
cd /home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/singlebasin_calibration
LOG=/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/stageB/driver_memory.log
NW=6
echo "$(date +%T) start" >> $LOG
STAGEB_END=2025-09-30 python3 run_stageB.py ctl M0 12 30 $NW >> $LOG 2>&1
STAGEB_END=2025-09-30 python3 run_stageB.py ctl M1 10 30 $NW >> $LOG 2>&1
for end in 2025-09-30 2020-09-30; do for s in 0.5 1 2 4 8; do
  STAGEB_END=$end python3 run_stageB.py ctl M1 10 30 $NW sfd=$s >> $LOG 2>&1
done; done
echo "$(date +%T) MEMORY TESTS DONE" >> $LOG
