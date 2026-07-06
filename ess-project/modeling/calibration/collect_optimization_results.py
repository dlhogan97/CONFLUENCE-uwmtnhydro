#!/usr/bin/env python3
"""
collect_optimization_results.py

Compiles all staged optimization trial CSVs from a given optimization root directory
into a single DataFrame and produces a best-per-stage summary.

Usage (standalone):
    python collect_optimization_results.py [--opt_dir PATH] [--out_csv PATH]

Usage (in a notebook):
    from collect_optimization_results import load_all_trials, best_per_stage_summary
    trials = load_all_trials(opt_dir)
    summary = best_per_stage_summary(trials)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OPT_DIR_DEFAULT = Path(
    "/scratch/dlhogan/ess-project-data"
    "/domain_Tuolumne_River_distributed_elev/optimization"
)


def load_all_trials(opt_dir: Path | str) -> pd.DataFrame:
    """
    Walk all staged_results_* subdirectories and concatenate every
    stage*_trials.csv into one DataFrame with added columns:
        experiment  — name of the staged_results_* dir
        KGE_calib   — 1 - J_total (calibration KGE equivalent)
    """
    opt_dir = Path(opt_dir)
    frames = []

    for exp_dir in sorted(opt_dir.glob("staged_results_*")):
        if not exp_dir.is_dir():
            continue
        experiment = exp_dir.name

        for csv_path in sorted(exp_dir.glob("stage*_trials.csv")):
            df = pd.read_csv(csv_path)
            df.insert(0, "experiment", experiment)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No trial CSVs found under {opt_dir}")

    trials = pd.concat(frames, ignore_index=True, sort=False)

    # KGE equivalent (cost → skill): 0 cost = KGE 1.0; 1 cost = KGE 0.0
    trials["KGE_calib"] = 1.0 - trials["J_total"]

    return trials


def best_per_stage_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """
    For each (experiment, stage) group, return the trial with minimum J_total
    (among converged trials only).  Columns: experiment, stage, best J_total,
    KGE_calib, J_anchor, J_coherence, runtime_sec, and the parameter columns.
    """
    converged = trials[trials["converged"] == True].copy()

    idx = converged.groupby(["experiment", "stage"])["J_total"].idxmin()
    best = converged.loc[idx].reset_index(drop=True)

    # reorder so key metrics come first
    front_cols = [
        "experiment", "stage", "trial_id",
        "J_anchor", "J_coherence", "J_total", "KGE_calib",
        "runtime_sec", "converged",
    ]
    param_cols = [c for c in best.columns if c not in front_cols]
    best = best[front_cols + param_cols]

    return best.sort_values(["experiment", "stage"]).reset_index(drop=True)


def final_params_summary(opt_dir: Path | str) -> pd.DataFrame:
    """
    Load final_best_params.json from each experiment directory into a
    wide DataFrame (one row per experiment).
    """
    opt_dir = Path(opt_dir)
    rows = []

    for exp_dir in sorted(opt_dir.glob("staged_results_*")):
        params_file = exp_dir / "final_best_params.json"
        if not params_file.exists():
            continue
        params = json.loads(params_file.read_text())
        params["experiment"] = exp_dir.name
        rows.append(params)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = ["experiment"] + [c for c in df.columns if c != "experiment"]
    return df[cols].reset_index(drop=True)


def trial_count_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Count total and converged trials per (experiment, stage)."""
    g = trials.groupby(["experiment", "stage"])
    return (
        g["trial_id"]
        .count()
        .rename("n_total")
        .to_frame()
        .join(g["converged"].sum().rename("n_converged"))
        .join(g["J_total"].min().rename("best_J_total"))
        .assign(best_KGE_calib=lambda d: 1 - d["best_J_total"])
        .reset_index()
        .sort_values(["experiment", "stage"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile optimization trial results")
    parser.add_argument(
        "--opt_dir",
        default=str(OPT_DIR_DEFAULT),
        help="Path to the optimization directory (contains staged_results_* subdirs)",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional: path to write the full trials DataFrame as CSV",
    )
    args = parser.parse_args()

    opt_dir = Path(args.opt_dir)
    print(f"Loading trials from: {opt_dir}\n")

    trials = load_all_trials(opt_dir)
    print(f"Total trials loaded: {len(trials):,}")
    print(f"Experiments found:   {sorted(trials['experiment'].unique())}\n")

    counts = trial_count_summary(trials)
    print("=== Trial counts and best calibration KGE per stage ===")
    print(counts.to_string(index=False))
    print()

    best = best_per_stage_summary(trials)
    print("=== Best trial per (experiment, stage) ===")
    display_cols = ["experiment", "stage", "J_anchor", "J_coherence", "J_total", "KGE_calib"]
    print(best[display_cols].to_string(index=False))
    print()

    final_params = final_params_summary(opt_dir)
    if not final_params.empty:
        print("=== Final best parameters per experiment ===")
        print(final_params.to_string(index=False))
        print()

    if args.out_csv:
        out = Path(args.out_csv)
        trials.to_csv(out, index=False)
        print(f"Full trials DataFrame written to: {out}")

        summary_path = out.with_stem(out.stem + "_best_summary")
        best.to_csv(summary_path, index=False)
        print(f"Best-per-stage summary written to: {summary_path}")


if __name__ == "__main__":
    main()
