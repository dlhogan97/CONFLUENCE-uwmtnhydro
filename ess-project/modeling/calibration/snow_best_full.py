"""snow_best_full.py -- rerun the best trial of a run_calibration_snow run and KEEP daily per-HRU output (calibration deletes it).
Uses result.json if present, otherwise the best 'ok' trial. Writes <run>/best_full/{best.json, keep/out/best_day.nc}.
usage: python snow_best_full.py <snow run dir> [out subdir=best_full]   (env SNOW_LW_INCREMENT=0 turns the longwave increment off)"""
import sys, json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import run_calibration_snow as S
from run_calibration_v2 import best_trial
rd = Path(sys.argv[1]); sub = sys.argv[2] if len(sys.argv) > 2 else "best_full"
if (rd/"result.json").exists(): x = json.load(open(rd/"result.json"))["x"]; src = "result.json"
else: b = best_trial(rd)[0]; x = b["x"]; src = f"best trial (objective {b['objective']:.4f})"
r = S.RunnerSnow(work_root=str(rd/sub), **S.CFG)
work = r.work_root/"keep"
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
r._stage(work); params = r._vector_to_params(np.asarray(x)); r._apply_params(work, params); r._wet_soil(work)
fdir = r._apply_forcing(work, params); r._write_filemanager(work, "best", forcing=fdir)
V = ["averageRoutedRunoff", "scalarSWE", "scalarSnowfall", "scalarRainfall", "scalarSnowSublimation", "scalarCanopySublimation", "scalarSnowAlbedo",
     "scalarSurfaceRunoff", "scalarSoilBaseflow", "scalarRainPlusMelt", "scalarGroundAbsorbedSolar", "scalarLWNetGround", "scalarSenHeatGround",
     "scalarLatHeatGround", "scalarGroundNetNrgFlux", "scalarSnowDrainage", "scalarTotalET"]
(work/"outputControl.txt").write_text("hruId | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0\n" + "".join(f"{v} | 24 | 0 | 0 | 1 | 0 | 0 | 0 | 0\n" for v in V))
(r.work_root/"best.json").write_text(json.dumps(dict(source=src, x=list(map(float, x)), params={k: v[1] for k, v in params.items()}), indent=2))
p = subprocess.run([S.DB.SUMMA_DS, "-m", str(work/"fileManager.txt")], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
print(src, p.returncode, [str(f) for f in (work/"out").glob("*")]); print(p.stderr[-500:])
