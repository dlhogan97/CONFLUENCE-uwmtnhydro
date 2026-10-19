"""Overnight pipeline: fit M0-M3 on full/A/B windows (emulator), verify the full-window fits in SUMMA (new singleBasin build), score."""
import subprocess, time, json, sys, os
from pathlib import Path
import fit_models as F
import summa_runs as SR
HERE = Path(__file__).parent
POP = {"M0": 12, "M1": 10, "M2": 10, "M3": 10}; ITER = 30
LOG = F.OUT/"driver.log"
def log(*a):
    with open(LOG, "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ")+" ".join(map(str, a))+"\n")
def run_fit(model, win, nw):
    if (F.OUT/f"fit_{model}_{win}.json").exists(): return
    log("fit start", model, win, "workers", nw)
    r = subprocess.run([sys.executable, str(HERE/"fit_models.py"), model, win, str(POP[model]), str(ITER), str(nw)], cwd=HERE, capture_output=True, text=True)
    log("fit done", model, win, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", "rc", r.returncode, (r.stderr[-300:] if r.returncode else ""))
procs = {}
log("pipeline start")
for m in ("M0", "M1", "M2", "M3"): run_fit(m, "full", 19)
for m in ("M0", "M1", "M2", "M3"):
    name = f"{m}_full"; fit = json.load(open(F.OUT/f"fit_{m}_full.json")); T = SR.build(name, fit, singlebasin=(m != "M0"))
    if not SR.finished(T): procs[name] = (SR.launch(T), T); log("summa launched", name)
for win in ("A", "B"):
    for m in ("M0", "M1", "M2", "M3"): run_fit(m, win, 15)
log("fits complete; waiting for SUMMA")
while any(not SR.finished(T) for _, T in procs.values()):
    time.sleep(60)
scores = {}
for m in ("M0", "M1", "M2", "M3"):
    name = f"{m}_full"; T = SR.ROOT/name
    out, q = SR.score(T, name, singlebasin=(m != "M0")); scores[name] = out; q.to_csv(F.OUT/f"summa_daily_{name}.csv", header=["q_cms"])
json.dump(scores, open(F.OUT/"summa_scores.json", "w"), default=float, indent=1)
log("PIPELINE DONE")
