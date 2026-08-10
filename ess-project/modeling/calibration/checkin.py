#!/usr/bin/env python3
"""checkin.py — one-shot status of the running signature calibrations.

    python3 checkin.py                 # both basins, newest run each
    python3 checkin.py East_River      # one basin
    python3 checkin.py --params        # also dump best params + convergence spread

Auto-detects the newest run directory per basin, so it never goes stale.
Reads the per-trial jsonl logs directly (best_so_far.py lags behind a live run).
"""
import glob
import json
import os
import subprocess
import sys
from collections import Counter

RUNS = "/scratch/dlhogan/ess-project-data/calibration/runs"
LOGS = "/scratch/dlhogan/ess-project-data/calibration/logs"
BASINS = ["East_River", "Tuolumne_River"]
PARTS = ["com_timing", "monthly_volume", "concentration", "recession", "lowflow_sepnov"]


def newest_run(basin):
    ds = sorted(glob.glob(f"{RUNS}/{basin}_*"), key=os.path.getmtime)
    return ds[-1] if ds else None


def health():
    def n(pat):
        r = subprocess.run(["pgrep", "-fc", pat], capture_output=True, text=True)
        return r.stdout.strip() or "0"
    load = open("/proc/loadavg").read().split()[:3]
    print(f"HEALTH  East drivers {n('run_calibration.py East_River')}   "
          f"Tuolumne drivers {n('run_calibration.py Tuolumne_River')}   "
          f"summa {n('summa.exe')}   load {' '.join(load)}")


def report(basin, show_params=False):
    run = newest_run(basin)
    if not run:
        print(f"\n=== {basin}: no runs found ==="); return
    rows = [json.loads(l) for p in glob.glob(f"{run}/work/logs/trials_*.jsonl") for l in open(p)]
    rows.sort(key=lambda r: r["t"])
    ok = [r for r in rows if r["reason"] == "ok"]
    nspec = len(ok[0]["params"]) if ok else (len(rows[0].get("params", {})) if rows else 0)
    gensz = 8 * nspec if nspec else 64
    print(f"\n=== {basin}  ({os.path.basename(run)}) ===")
    print(f"  trials {len(rows)}  (~{len(rows)/gensz:.1f} gens of {gensz})   "
          f"feasible {len(ok)} ({100*len(ok)/max(len(rows),1):.0f}%)")
    if len(rows) != len(ok):
        print(f"  rejections: {dict(Counter(r['reason'] for r in rows if r['reason'] != 'ok'))}")
    snaps = sorted(os.path.basename(d) for d in glob.glob(f"{run}/generations/gen_*"))
    if snaps:
        print(f"  snapshots: {snaps[0]}..{snaps[-1]} ({len(snaps)})")
    if not ok:
        print("  (no feasible trial yet)"); return

    keys = [k for k in PARTS if k in ok[0]["parts"]]
    hdr = " ".join(f"{k[:6]:>6}" for k in keys)
    print(f"  {'gen':>4} {'genbest':>9} {'running':>9}  {hdr}  {'KGE':>5} {'NSE':>6} {'tau':>5}")
    best, br = None, None
    for g in range(len(rows) // gensz + 1):
        chunk = [r for r in rows[g*gensz:(g+1)*gensz] if r["reason"] == "ok"]
        if not chunk:
            continue
        gb = min(chunk, key=lambda r: r["objective"])
        if best is None or gb["objective"] < best:
            best, br = gb["objective"], gb
        p, d = br["parts"], br["diagnostics"]
        print(f"  {g:>4} {gb['objective']:>9.4f} {best:>9.4f}  "
              + " ".join(f"{p.get(k, float('nan')):>6.3f}" for k in keys)
              + f"  {d.get('KGE', 0):>5.2f} {d.get('NSE', 0):>6.2f} {d.get('tau_sim_days', 0):>5.1f}")
    print(f"  BEST params: " + "  ".join(f"{k}={v:.4g}" for k, v in br["params"].items()))

    if show_params:
        import numpy as np
        top = sorted(ok, key=lambda r: r["objective"])[:max(8, len(ok)//5)]
        print(f"  convergence (top {len(top)}):")
        for k in br["params"]:
            v = np.array([r["params"][k] for r in top])
            spread = (v.max()-v.min())/max(abs(np.median(v)), 1e-12)
            print(f"     {k:26s} med {np.median(v):>11.4g}   rel-spread {spread:>6.1%}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    health()
    for b in (args or BASINS):
        report(b, show_params="--params" in sys.argv)
