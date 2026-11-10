"""routing_best_full.py -- rerun the best trial of a run_calibration_east_routing run over an extended window and KEEP the output.
usage: python routing_best_full.py <routing run dir> [sim_end=2025-09-30] [sim_start=1981-10-01] [out subdir=best_full_1981]
Uses result.json if present, otherwise the best accepted trial in work/logs. Writes <run>/<subdir>/{best.json, keep/out/best_timestep.nc}."""
import sys, json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import run_calibration_east_routing as R
import run_calibration_control as C
from run_calibration import GEN_OUTPUT_CONTROL
from run_calibration_v2 import best_trial
rd = Path(sys.argv[1]); end = sys.argv[2] if len(sys.argv) > 2 else "2025-09-30"
start = sys.argv[3] if len(sys.argv) > 3 else "1981-10-01"; sub = sys.argv[4] if len(sys.argv) > 4 else "best_full_1981"
if (rd/"result.json").exists(): x = json.load(open(rd/"result.json"))["x"]; src = "result.json"
else: b = best_trial(rd)[0]; x = b["x"]; src = f"best trial (objective {b['objective']:.4f})"
cfg = dict(C.CFG["East_River"], sim_start=start, sim_end=end)
r = R.RunnerRouting(work_root=str(rd/sub), **cfg)
work = r.work_root/"keep"
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
r._stage(work); params = r._vector_to_params(np.asarray(x)); r._apply_params(work, params); r._wet_soil(work)
fdir = r._apply_forcing(work, params); r._write_filemanager(work, "best", forcing=fdir)
(work/"outputControl.txt").write_text(GEN_OUTPUT_CONTROL + "scalarSnowAlbedo         | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n")
(r.work_root/"best.json").write_text(json.dumps(dict(source=src, x=list(map(float, x)), sim_start=start, sim_end=end, params={k: v[1] for k, v in params.items()}), indent=2))
p = subprocess.run([C.DB.SUMMA_DS, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
print(src, p.returncode, [str(f) for f in (work/"out").glob("*")]); print(p.stderr[-500:])
