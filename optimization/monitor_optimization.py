#!/usr/bin/env python3
"""
monitor_optimization.py — Live monitoring of staged SUMMA optimization progress.

Reads the per-stage trial CSV logs written by staged_optimizer.py and prints
a rolling summary of:
  - Best cost (J_total, J_anchor, J_coherence) found so far
  - KGE equivalent for anchor objectives (1 - J_anchor)
  - Generation-by-generation best and mean cost
  - Best multiplier values found so far

Usage
-----
    # One-shot snapshot
    python optimization/monitor_optimization.py \
        --results-dir /scratch/dlhogan/ess-project-data/domain_East_River_distributed/optimization/staged_results

    # Auto-refresh every 30 seconds (like watch)
    python optimization/monitor_optimization.py \
        --results-dir /scratch/dlhogan/.../staged_results \
        --watch --interval 30

    # Shorter output (just best-per-stage)
    python optimization/monitor_optimization.py --results-dir ... --brief
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage metadata (for labelling objective types)
# ---------------------------------------------------------------------------

STAGE_META = {
    "stage1_snow": {
        "params": ["albedoDecayRate", "Frad_direct", "Frad_vis"],
        "anchor_label": "KGE (SWE vs SNOTEL)",
        "pop_hint": 36,  # popsize × n_params
    },
    "stage2_soil_et": {
        "params": ["k_soil", "vGn_alpha", "vGn_n", "qSurfScale", "rootingDepth", "theta_sat", "summerLAI"],
        "anchor_label": "KGE (ET vs OpenET)",
        "pop_hint": 84,
    },
    "stage3_groundwater": {
        "params": ["aquiferScaleFactor", "aquiferBaseflowExp", "aquiferBaseflowRate"],
        "anchor_label": "KGE (baseflow vs Eckhardt)",
        "pop_hint": 36,
    },
    "stage4_routing": {
        "params": ["routingGammaShape", "routingGammaScale"],
        "anchor_label": "KGE (routed Q vs USGS)",
        "pop_hint": 20,
    },
}

STAGE_ORDER = ["stage1_snow", "stage2_soil_et", "stage3_groundwater", "stage4_routing"]


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def load_trials(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a stage trial CSV; return None if missing or empty."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "J_total" not in df.columns:
            return None
        df = df[df["converged"] == True].copy()  # noqa: E712  — only converged trials
        return df if not df.empty else None
    except Exception as e:
        print(f"  [warn] Could not read {csv_path}: {e}", file=sys.stderr)
        return None


def infer_population_size(df: pd.DataFrame, stage_name: str) -> int:
    """Estimate population size from stage metadata or df length."""
    meta = STAGE_META.get(stage_name, {})
    return meta.get("pop_hint", 20)


# ---------------------------------------------------------------------------
# Per-stage summary
# ---------------------------------------------------------------------------

def summarise_stage(stage_name: str, df: pd.DataFrame) -> dict:
    """Compute summary statistics for one stage's trial log."""
    meta = STAGE_META.get(stage_name, {})
    pop_size = infer_population_size(df, stage_name)
    n_trials = len(df)

    best_row = df.loc[df["J_total"].idxmin()]
    best_cost = float(best_row["J_total"])
    best_anchor = float(best_row["J_anchor"])
    best_coh = float(best_row["J_coherence"])

    # KGE for anchor metrics is 1 - J_anchor (since J_anchor = 1 - KGE)
    best_kge = 1.0 - best_anchor

    # Estimate generation number
    n_complete_gens = n_trials // pop_size

    # Running best per generation
    gen_stats = []
    for g in range(n_complete_gens):
        gen_df = df.iloc[g * pop_size:(g + 1) * pop_size]
        gen_stats.append({
            "gen": g + 1,
            "best_J": gen_df["J_total"].min(),
            "mean_J": gen_df["J_total"].mean(),
            "best_KGE": 1.0 - gen_df["J_anchor"].min(),
        })

    # Running best across all trials (cumulative minimum)
    running_best = df["J_total"].cummin().values

    # Best multiplier values
    param_cols = [c for c in df.columns if c.startswith("mult_")]
    best_params = {c.replace("mult_", ""): float(best_row[c]) for c in param_cols}

    return {
        "stage": stage_name,
        "n_trials": n_trials,
        "n_complete_gens": n_complete_gens,
        "best_cost": best_cost,
        "best_kge": best_kge,
        "best_anchor": best_anchor,
        "best_coherence": best_coh,
        "anchor_label": meta.get("anchor_label", "KGE (anchor)"),
        "gen_stats": gen_stats,
        "running_best": running_best,
        "best_params": best_params,
        "avg_runtime_sec": df["runtime_sec"].mean() if "runtime_sec" in df.columns else None,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"

def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"

def _bold(t): return _color(t, _BOLD)
def _green(t): return _color(t, _GREEN)
def _cyan(t): return _color(t, _CYAN)
def _yellow(t): return _color(t, _YELLOW)
def _dim(t): return _color(t, _DIM)


def kge_bar(kge_val: float, width: int = 20) -> str:
    """ASCII progress bar for KGE in [-1, 1] scaled to [0, width]."""
    frac = max(0.0, min(1.0, (kge_val + 1.0) / 2.0))  # map [-1,1] → [0,1]
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    if kge_val >= 0.6:
        bar = _green(bar)
    elif kge_val >= 0.3:
        bar = _yellow(bar)
    else:
        bar = _dim(bar)
    return f"[{bar}] {kge_val:+.3f}"


def print_stage_summary(s: dict, brief: bool = False) -> None:
    stage = s["stage"]
    print()
    print(_bold(f"  {'─'*60}"))
    print(_bold(f"  {stage.upper()}") + f"  —  {s['anchor_label']}")
    print(f"  {'─'*60}")
    print(f"  Trials evaluated : {s['n_trials']:>5}  "
          f"({s['n_complete_gens']} complete generations)")
    if s["avg_runtime_sec"]:
        print(f"  Avg SUMMA runtime: {s['avg_runtime_sec']:.1f}s / trial")
    print()
    print(f"  {'BEST SO FAR':25s}  J_total={s['best_cost']:.4f}  "
          f"J_anchor={s['best_anchor']:.4f}  J_coh={s['best_coherence']:.4f}")
    print(f"  {'':25s}  {s['anchor_label']}: {kge_bar(s['best_kge'])}")
    print()

    if not brief and s["best_params"]:
        print(f"  Best multipliers:")
        for param, val in s["best_params"].items():
            print(f"    {param:<25s} × {val:.4f}")
        print()

    if not brief and s["gen_stats"]:
        print(f"  Generation-by-generation progress (converged trials only):")
        print(f"  {'Gen':>4}  {'Best J':>8}  {'Mean J':>8}  {'Best KGE':>9}  Trend")
        prev_best = None
        for g in s["gen_stats"]:
            trend = ""
            if prev_best is not None:
                if g["best_J"] < prev_best - 0.002:
                    trend = _green("▼ improving")
                elif g["best_J"] > prev_best + 0.002:
                    trend = _yellow("▲ worse")
                else:
                    trend = _dim("  plateau")
            prev_best = g["best_J"]
            print(f"  {g['gen']:>4}  {g['best_J']:>8.4f}  {g['mean_J']:>8.4f}  "
                  f"{g['best_KGE']:>+9.3f}  {trend}")
        print()


# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

def display(results_dir: Path, brief: bool = False) -> None:
    """Print full optimization monitoring output."""
    if sys.stdout.isatty():
        print("\033[H\033[2J", end="")  # clear screen

    print(_bold(f"{'='*64}"))
    print(_bold("  SUMMA Staged Optimization Monitor"))
    print(_bold(f"  {results_dir}"))
    print(_bold(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(_bold(f"{'='*64}"))

    any_data = False
    for stage_name in STAGE_ORDER:
        csv_path = results_dir / f"{stage_name}_trials.csv"
        df = load_trials(csv_path)
        if df is None:
            status = "pending"
            # Check if the best_params JSON exists → stage completed
            json_path = results_dir / f"{stage_name}_best_params.json"
            if json_path.exists():
                status = "completed (frozen)"
            print(f"\n  {stage_name.upper():30s}  [{status}]")
            continue

        any_data = True
        summary = summarise_stage(stage_name, df)
        print_stage_summary(summary, brief=brief)

    if not any_data:
        print("\n  No trial data yet. Is the optimizer running?")
        print(f"  Watching: {results_dir}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor SUMMA staged optimization progress")
    parser.add_argument(
        "--results-dir", "-r",
        required=True,
        help="Path to staged_results directory (contains *_trials.csv files)",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously refresh (like watch)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="Brief output: skip generation table and parameter values",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        print(f"Watching {results_dir} — refreshing every {args.interval}s  (Ctrl+C to stop)")
        try:
            while True:
                display(results_dir, brief=args.brief)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        display(results_dir, brief=args.brief)


if __name__ == "__main__":
    main()
