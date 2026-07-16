#!/usr/bin/env python3
"""
best_so_far.py — Scan a calibration's trial logs and report the best parameter set found.

Also the manual checkpoint tool. The box reboots uncleanly (lightning, no UPS), so before
relaunching a lost run, save the best point so the restart warm-starts from it:

    python best_so_far.py East_River                  # report only (newest run with results)
    python best_so_far.py East_River --write-seed     # ... and checkpoint it for the relaunch
    python best_so_far.py runs/East_River_20260715_   # a specific run dir

Then relaunch:

    cd <this dir> && OMP_NUM_THREADS=1 setsid python3 run_calibration.py East_River \
        --workers 20 --popsize 8 --maxiter 20 > $CALIB_OUT/logs/cal_East_River.log 2>&1 &

OMP_NUM_THREADS=1 is not optional: each SUMMA is a 1-GRU domain that gains nothing from
threads, but will grab all 24 cores and fight the other 19 workers if left unset.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

OUT = os.environ.get("CALIB_OUT", "/scratch/dlhogan/ess-project-data/calibration")


def trials(run):
    recs = []
    for f in glob.glob(f"{run}/work/logs/*.jsonl"):
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="East_River",
                    help="domain name, or a specific run directory")
    ap.add_argument("--write-seed", action="store_true",
                    help="checkpoint the best params to seeds/<domain>.json for a warm restart")
    args = ap.parse_args()

    if "/" in args.target:
        runs, domain = [args.target], Path(args.target).name.rsplit("_", 2)[0]
    else:
        domain = args.target
        runs = sorted(glob.glob(f"{OUT}/runs/{domain}_*"), reverse=True)
    if not runs:
        sys.exit(f"no runs found for {domain} under {OUT}/runs")

    # Walk back to the newest run that produced a feasible trial: a run can die with zero of
    # them (e.g. every SUMMA timed out), and stopping at the newest dir would report nothing.
    # Only compare within one objective definition -- values are not comparable across changes.
    best = src = None
    for r in runs:
        recs = trials(r)
        ok = [x for x in recs if x.get("reason") == "ok" and x.get("objective") is not None]
        print(f"{Path(r).name:34s} trials={len(recs):4d} feasible={len(ok):4d}")
        if ok and best is None:
            best, src = min(ok, key=lambda x: x["objective"]), r
        if best is not None and "/" in args.target:
            break

    if best is None:
        print("\nno feasible trial in any run — nothing to checkpoint")
        return

    print(f"\nBEST from {Path(src).name}: objective = {best['objective']:.4f}")
    print(f"  parts:       { {k: round(v, 3) for k, v in best['parts'].items()} }")
    print(f"  diagnostics: { {k: round(v, 3) for k, v in best['diagnostics'].items()} }")
    print("  params:")
    for k, v in best["params"].items():
        print(f'      "{k}": {v:g},')

    if args.write_seed:
        p = Path(OUT) / "seeds" / f"{domain}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(best["params"], indent=2))
        print(f"\n  checkpointed -> {p}")
        print("  the next run_calibration.py launch will warm-start from this point")


if __name__ == "__main__":
    main()
