#!/bin/bash
# East aquiferBaseflowExp sweep, valley-bottom store. 8 members, run all in parallel.
OUT=/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/sweep_exp_20260812
echo "[$(date +%T)] launching $(wc -l < $OUT/jobs.txt) East exp members 8-wide"
xargs -a "$OUT/jobs.txt" -L1 -P 8 bash -c '
  r=$(dirname "$0")
  summa -m "$0" > "$r/summa.log" 2>&1
  if grep -q "FORTRAN STOP: finished simulation successfully" "$r/summa.log"; then
    echo "[$(date +%T)] OK   $(basename $r)"
  else
    echo "[$(date +%T)] FAIL $(basename $r): $(grep -m1 -E "FATAL|Error|error" "$r/summa.log" | head -c 140)"
  fi'
echo "[$(date +%T)] ALL DONE"
