"""stageA_best_full.py -- rerun the best East stage-A trial over an extended window and KEEP the output (calibration deletes it). usage: python stageA_best_full.py <stageA run dir> [sim_end=2025-09-30]"""
import sys, json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import run_calibration_stageA as S
from run_calibration import GEN_OUTPUT_CONTROL
rd = Path(sys.argv[1]); end = sys.argv[2] if len(sys.argv) > 2 else "2025-09-30"
x = json.load(open(rd/"result.json"))["x"]; cfg = dict(S.CFG[S.DOM]); cfg["sim_end"] = end
r = S.RunnerA(work_root=str(rd/"best_full_work"), **cfg)
work = r.work_root/"keep"
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
r._stage(work); params = r._vector_to_params(np.asarray(x)); r._apply_params(work, params); r._wet_soil(work)
fdir = r._apply_forcing(work, params); r._write_filemanager(work, "best", forcing=fdir)
(work/"outputControl.txt").write_text(GEN_OUTPUT_CONTROL + "scalarSnowAlbedo         | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")  # full output: stage B needs baseflow / surface runoff per HRU
p = subprocess.run([S.SUMMA_EXE, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
print(p.returncode, [str(f) for f in (work/"out").glob("*")]); print(p.stderr[-500:])
