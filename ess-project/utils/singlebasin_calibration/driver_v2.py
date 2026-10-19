import subprocess, sys, time
from pathlib import Path
import fit_v2 as V
HERE = Path(__file__).parent; LOG = V.OUT/"driver_v2.log"; POP = {"M0": 12, "M1": 10}
def log(*a):
    with open(LOG, "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ")+" ".join(map(str, a))+"\n")
log("v2 pipeline start")
for win in ("full", "A", "B"):
    for m in ("M0", "M1"):
        if (V.OUT/f"fit_{m}_{win}.json").exists(): continue
        log("fit start", m, win)
        r = subprocess.run([sys.executable, str(HERE/"fit_v2.py"), m, win, str(POP[m]), "30", "19"], cwd=HERE, capture_output=True, text=True)
        log("fit done", m, win, (r.stdout.strip().splitlines() or [""])[-1], "rc", r.returncode, r.stderr[-200:] if r.returncode else "")
log("V2 PIPELINE DONE")
