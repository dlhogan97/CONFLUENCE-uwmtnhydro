#!/usr/bin/env python3
"""
Run the seasonal ensemble workflow using existing optimization parameters.
This script will:
1. Load best parameters from existing optimization (skip new optimization)
2. Apply parameters to settings files
3. Run 20-year baseline
4. Create warm state
5. Build seasonal forcing ensembles
6. Run all ensemble members (parallel)
7. Evaluate and visualize results

Usage:
    python run_seasonal_ensemble_workflow.py
    
Or in background:
    nohup python run_seasonal_ensemble_workflow.py > logs/seasonal_ensemble_full.log 2>&1 &
"""

import sys
import logging
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ess_project.modeling.seasonal_ensemble_experiment import SeasonalEnsembleExperiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the full seasonal ensemble workflow."""
    
    # Use East River lumped config
    config_path = Path(project_root) / "ess-project/0_config_files/config_East_River_lumped_seasonal_noxPlicit.yaml"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("SEASONAL ENSEMBLE WORKFLOW - USING EXISTING OPTIMIZATION PARAMETERS")
    print("=" * 80)
    print(f"Config: {config_path.name}")
    print(f"Workspace: {config_path.parent.parent}")
    print()
    
    # Initialize experiment
    exp = SeasonalEnsembleExperiment(str(config_path))
    
    # Parameters
    target_year = 2021
    start_year = 2000
    end_year = 2020
    donor_years = list(range(2001, 2021))
    
    print(f"Target year: {target_year}")
    print(f"Baseline period: {start_year} - {end_year}")
    print(f"Donor years: {donor_years[0]} - {donor_years[-1]} ({len(donor_years)} years)")
    print()
    
    try:
        # Step 1: Load best parameters (skip_if_exists=True skips new optimization)
        print("\n" + "=" * 80)
        print("STEP 1: Loading best parameters from existing optimization")
        print("=" * 80)
        best_params = exp.step1_optimize(skip_if_exists=True)
        print(f"✓ Loaded {len(best_params)} optimized parameters")
        print(best_params.to_string())
        
        # Step 2: 20-year baseline run
        print("\n" + "=" * 80)
        print("STEP 2: Running 20-year baseline simulation")
        print("=" * 80)
        baseline_path = exp.step2_long_term_run(
            start=f"{start_year}-10-01",
            end=f"{end_year}-09-30",
            experiment_id='longterm_baseline'
        )
        print(f"✓ Baseline output: {baseline_path}")
        
        # Step 2b: Create warm state
        print("\n" + "=" * 80)
        print("STEP 2b: Creating warm state from baseline")
        print("=" * 80)
        warm_state_path = exp.step2b_create_warm_state(baseline_path)
        print(f"✓ Warm state created: {warm_state_path}")
        
        # Step 3: Target year baseline
        print("\n" + "=" * 80)
        print(f"STEP 3: Running target year ({target_year}) baseline")
        print("=" * 80)
        target_baseline = exp.step3_target_year_baseline(target_year)
        print(f"✓ Target year baseline: {target_baseline}")
        
        # Step 4: Build ensembles
        print("\n" + "=" * 80)
        print("STEP 4: Building seasonal forcing ensembles")
        print("=" * 80)
        ensemble_forcing = exp.step4_build_ensembles(
            target_year=target_year,
            donor_years=donor_years,
            water_year=True
        )
        total_members = sum(len(v) for v in ensemble_forcing.values())
        print(f"✓ Built {len(ensemble_forcing)} seasonal ensembles with {total_members} total members")
        
        # Step 5: Run all ensembles (parallel with 80 workers)
        print("\n" + "=" * 80)
        print("STEP 5: Running all ensemble members (parallel)")
        print("=" * 80)
        ensemble_results = exp.step5_run_ensembles(
            target_year=target_year,
            parallel=True,
            max_workers=80,
            poll_interval=30.0
        )
        print(f"✓ Completed all {total_members} ensemble runs")
        
        # Step 6: Evaluate sensitivity
        print("\n" + "=" * 80)
        print("STEP 6: Evaluating sensitivity metrics")
        print("=" * 80)
        exp.step6_evaluate_sensitivity(target_year=target_year)
        print(f"✓ Sensitivity evaluation complete")
        
        # Step 7: Visualize results
        print("\n" + "=" * 80)
        print("STEP 7: Creating visualizations")
        print("=" * 80)
        obs_csv = Path(exp.cfg.obs_dir) / "USGS_09112500_streamflow.csv"
        plots = exp.step7_visualize(target_year=target_year, obs_csv=str(obs_csv))
        print(f"✓ Created {len(plots)} visualizations")
        
        print("\n" + "=" * 80)
        print("✓✓✓ SEASONAL ENSEMBLE WORKFLOW COMPLETE ✓✓✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
