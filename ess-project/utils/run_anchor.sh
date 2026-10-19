#!/bin/bash
# Anchor-bin snowpatch members: build 3 designs in parallel, then run them.
cd "$(dirname "$0")"
D=/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/designs_anchor_20260812
EXP=/scratch/dlhogan/ess-project-data/domain_Tuolumne_River_distributed_elevAspect/simulations/snowpatch_anchor_20260812

echo "[$(date +%T)] building 3 anchor members ..."
xargs -a "$D/members.txt" -L1 -P 3 bash -c '
  python build_member_anchor.py "$0" "$1" > build_$1.log 2>&1 \
    && echo "  built $1" || echo "  FAILED $1 (see build_$1.log)"'
echo "[$(date +%T)] build stage done"

: > jobs_anchor.txt
for d in "$EXP"/*/; do [ -f "$d/fileManager.txt" ] && echo "$d/fileManager.txt" >> jobs_anchor.txt; done
echo "[$(date +%T)] $(wc -l < jobs_anchor.txt) jobs queued"
xargs -a jobs_anchor.txt -L1 -P 3 bash -c '
  r=$(dirname "$0")
  summa -m "$0" > "$r/summa.log" 2>&1
  if grep -q "FORTRAN STOP: finished simulation successfully" "$r/summa.log"; then
    echo "[$(date +%T)] OK   $(basename $r)"
  else
    echo "[$(date +%T)] FAIL $(basename $r): $(grep -m1 -E "FATAL|Error|error" "$r/summa.log" | head -c 160)"
  fi'
echo "[$(date +%T)] ALL DONE"
