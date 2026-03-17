#!/usr/bin/env python3
"""
Example usage of the restructured seasonal ensemble experiment workflow.

This demonstrates:
1. Single experiment run with dated folder
2. Multiple comparative runs with different configs
3. Accessing results from specific experiment runs
"""

import sys
from pathlib import Path

# Add CONFLUENCE root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seasonal_ensemble_experiment import SeasonalEnsembleExperiment


def example_1_single_experiment():
    """Example 1: Run a single experiment with automatic dated folder."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Experiment Run")
    print("="*70 + "\n")
    
    config_path = Path(__file__).parent.parent / "0_config_files" / "config_East_River_lumped_seasonal_noxPlicit.yaml"
    
    # Create experiment - will auto-create YYYYMMDD_seasonal_ensemble/ folder
    exp = SeasonalEnsembleExperiment(
        config_path=str(config_path),
        experiment_name="example_run"  # Optional; defaults to config EXPERIMENT_ID
    )
    
    print(f"\n✓ Experiment initialized")
    print(f"  Workspace: {exp.cfg.experiment_workspace}")
    print(f"  Settings: {exp.cfg.settings_dir}")
    print(f"  Results: {exp.cfg.ensemble_dir}")
    print(f"  Plots: {exp.cfg.plots_dir}")
    
    # Run full workflow
    print(f"\nStarting workflow...\n")
    
    # exp.run_full_workflow(
    #     baseline_start="2003-01-01 01:00",
    #     baseline_end="2022-12-31 23:00",
    #     target_year=2023,
    #     skip_optimization=True
    # )
    
    print(f"\n✓ Workflow complete!")
    print(f"  Results location: {exp.cfg.experiment_workspace}")
    
    return exp


def example_2_comparative_runs():
    """Example 2: Run multiple experiments with different configs."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparative Runs with Different Configs")
    print("="*70 + "\n")
    
    config_dir = Path(__file__).parent.parent / "0_config_files"
    
    configs = [
        ("config_East_River_lumped_seasonal_noxPlicit.yaml", "noxplicit_snowmodel"),
        ("config_East_River_lumped_seasonal_bigBuckt.yaml", "big_bucket"),
    ]
    
    experiments = []
    
    for config_name, exp_name in configs:
        config_path = config_dir / config_name
        
        if not config_path.exists():
            print(f"⚠ Config not found: {config_path}")
            continue
        
        print(f"\nRunning: {exp_name}")
        print(f"  Config: {config_name}")
        
        exp = SeasonalEnsembleExperiment(
            config_path=str(config_path),
            experiment_name=exp_name
        )
        
        print(f"  ✓ Workspace: {exp.cfg.experiment_workspace.name}")
        experiments.append(exp)
        
        # Uncomment to run:
        # exp.run_full_workflow(
        #     baseline_start="2003-01-01 01:00",
        #     baseline_end="2022-12-31 23:00",
        #     target_year=2023,
        #     skip_optimization=True
        # )
    
    print(f"\n✓ Initialized {len(experiments)} experiments")
    print(f"  Each has independent results folder and settings copy")
    print(f"  No risk of overwriting between runs!")
    
    return experiments


def example_3_access_results():
    """Example 3: Access results from a specific experiment."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Accessing Results from Previous Runs")
    print("="*70 + "\n")
    
    import pandas as pd
    
    data_dir = Path("/scratch/dlhogan/ess-project-data")
    domain_name = "East_River_lumped"
    
    sim_dir = data_dir / "domain_" + domain_name / "simulations"
    
    # Find all dated experiments
    experiments = sorted([d for d in sim_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(experiments)} experiment runs:")
    for exp_folder in experiments[-5:]:  # Show last 5
        readme_path = exp_folder / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                first_line = f.readline().strip()
            print(f"  • {exp_folder.name}")
            print(f"    {first_line}")
    
    # Access specific experiment results
    print(f"\n" + "-"*70)
    print("Accessing a specific experiment's results:")
    print("-"*70)
    
    recent_exp = None
    for folder in reversed(experiments):
        if folder.name.startswith("2026") and folder.is_dir():
            recent_exp = folder
            break
    
    if recent_exp:
        print(f"\nRecent experiment: {recent_exp.name}")
        
        # List contents
        results_dir = recent_exp / "ensemble" / "results"
        plots_dir = recent_exp / "ensemble" / "plots"
        settings_dir = recent_exp / "settings"
        
        print(f"  Results dir: {results_dir}")
        print(f"  Plots dir: {plots_dir}")
        print(f"  Settings dir: {settings_dir}")
        
        # Show structure
        if results_dir.exists():
            subdirs = [d.name for d in results_dir.iterdir() if d.is_dir()]
            print(f"\n  Result folders: {subdirs}")
        
        if settings_dir.exists():
            param_files = [f.name for f in settings_dir.glob("*.txt")]
            print(f"\n  Settings files: {param_files}")
        
        # Show config snapshot
        config_files = list(recent_exp.glob("config_*.yaml"))
        if config_files:
            print(f"\n  Config snapshot: {config_files[0].name}")


def main():
    """Run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples of restructured workflow")
    parser.add_argument(
        'example',
        nargs='?',
        default='1',
        choices=['1', '2', '3', 'all'],
        help='Which example to run (default: 1)'
    )
    
    args = parser.parse_args()
    
    if args.example in ('1', 'all'):
        example_1_single_experiment()
    
    if args.example in ('2', 'all'):
        example_2_comparative_runs()
    
    if args.example in ('3', 'all'):
        example_3_access_results()


if __name__ == '__main__':
    main()
