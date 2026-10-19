#!/usr/bin/env python3
"""Re-run the best-so-far trial of a v2 calibration with FULL per-HRU output (SWE, precip, ET, ...) into <rundir>/best_full/.
usage: run_best_full.py <domain> <rundir> [rank=0]"""
import sys, json, glob, subprocess, os
from pathlib import Path
import numpy as np
from run_calibration_v2 import RunnerV2, CFG, SPECS
from run_calibration import GEN_OUTPUT_CONTROL, SUMMA_EXE
dom, rd = sys.argv[1], Path(sys.argv[2]); rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rec = [json.loads(l) for f in glob.glob(str(rd/"work/logs/trials_*.jsonl")) for l in open(f)]
ok = sorted([r for r in rec if r["reason"] == "ok"], key=lambda r: r["objective"]); b = ok[rank]
print("trial objective", b["objective"], b["parts"])
cfg = dict(CFG[dom]); tag = ""
if len(sys.argv) > 5:                       # optional window override: <sim_start> <sim_end> <analysis_start>, e.g. 2016-10-01 2025-09-30 2019-10-01
    cfg.update(sim_start=sys.argv[3+1], sim_end=sys.argv[3+2], analysis_start=sys.argv[3+3]) if len(sys.argv) > 6 else None; tag = "_ext"
run = RunnerV2(dom, work_root=str(rd/("work_full"+tag)), **cfg)
kinds = {s.name: s.kind for s in SPECS[dom]}
params = {k: (kinds[k], v) for k, v in b["params"].items()}
out = rd/"best_full"/f"rank{rank}{tag}"; out.mkdir(parents=True, exist_ok=True)
run._stage(out); run._apply_params(out, params); run._wet_soil(out); fdir = run._apply_forcing(out, params)
run._write_filemanager(out, "best", forcing=fdir); (out/"outputControl.txt").write_text(GEN_OUTPUT_CONTROL + "scalarSnowAlbedo         | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")
(out/"params.json").write_text(json.dumps(b["params"], indent=1)); (out/"out").mkdir(exist_ok=True)
env = {**os.environ, "OMP_NUM_THREADS": "1"}
subprocess.run([SUMMA_EXE, "-m", str(out/"fileManager.txt")], env=env, stdout=open(out/"summa.log", "w"), stderr=subprocess.STDOUT)
print("done", out)
