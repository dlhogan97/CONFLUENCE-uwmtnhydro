#!/bin/bash
# Overnight sweep: build 9 Tuolumne members (parallel), then run all 20 jobs 10-wide.
cd "$(dirname "$0")"
SWEEP=/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_sweep_20260812
EASTS=/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/sweep_sf_20260812

echo "[$(date +%T)] building Tuolumne members ..."
xargs -a tuo_members.txt -L1 -P 9 bash -c '
  d='"$SWEEP"'/$1
  if [ -f "$d/fileManager.txt" ]; then echo "  $1 already built"; exit 0; fi
  python build_member.py $0 $1 > build_$1.log 2>&1 && echo "  built $1" || echo "  FAILED $1"'
echo "[$(date +%T)] build stage done"

# job list: every fileManager, Tuolumne first (longer) so the tail packs well
: > jobs_all.txt
for d in "$SWEEP"/*/;   do [ -f "$d/fileManager.txt" ] && echo "$d/fileManager.txt" >> jobs_all.txt; done
for d in "$EASTS"/*/;   do [ -f "$d/fileManager.txt" ] && echo "$d/fileManager.txt" >> jobs_all.txt; done
echo "[$(date +%T)] $(wc -l < jobs_all.txt) jobs queued, running 10-wide"

xargs -a jobs_all.txt -L1 -P 10 bash -c '
  r=$(dirname "$0")
  if ls "$r"/output/*.nc >/dev/null 2>&1; then echo "  skip $(basename $r) (has output)"; exit 0; fi
  summa -m "$0" > "$r/summa.log" 2>&1
  if grep -q "FORTRAN STOP: finished simulation successfully" "$r/summa.log"; then
    echo "[$(date +%T)] OK   $(basename $r)"
  else
    echo "[$(date +%T)] FAIL $(basename $r): $(grep -m1 -E "FATAL|Error" "$r/summa.log" | head -c 120)"
  fi'
echo "[$(date +%T)] ALL DONE"
