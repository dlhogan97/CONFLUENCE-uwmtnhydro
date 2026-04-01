#!/usr/bin/env python
"""
Download OpenET data for the East River Basin

This script demonstrates how to use the OpenET utility to download
evapotranspiration timeseries data for the East River lumped basin.

Usage:
    Interactive mode (prompts for options):
        python download_East_et.py
    
    Command-line mode:
        python download_East_et.py --option 1
        python download_East_et.py --option 2 --variable eto
        python download_East_et.py --option 3 --start-date 2019-01-01 --end-date 2022-12-31

Options:
    1. Ensemble ET (monthly) - Recommended for most use cases
    2. All Models (monthly) - For model comparison/uncertainty analysis
    3. Ensemble ET (daily) - Higher temporal resolution
    4. Custom - Specify all parameters
    5. Per-Polygon - Get data for each polygon/HRU separately

Available variables: et, eto, etof, ndvi, pr, etr
Available models: disalexi, eemetric, geesebal, ptjpl, sims, ssebop, ensemble
"""

import sys
import argparse
from pathlib import Path

# Add the repository root to Python path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from utils.data.openet_utils import OpenETClient, download_openet_data


# Configuration
SHAPEFILE_PATH = "/scratch/dlhogan/ess-project-data/domain_East_River_lumped/shapefiles/catchment/East_River_lumped_HRUs_GRUs.shp"
OUTPUT_DIR = "/scratch/dlhogan/ess-project-data/domain_East_River_lumped/observations/et"
ENV_FILE = repo_root / ".env"

# Default date range
DEFAULT_START_DATE = "2012-10-01"
DEFAULT_END_DATE = "2022-09-30"

# Available options
AVAILABLE_VARIABLES = ['et', 'eto', 'etof', 'ndvi', 'pr', 'etr', 'et_mad_min', 'et_mad_max']
AVAILABLE_MODELS = ['disalexi', 'eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop', 'ensemble']

VARIABLE_ALIASES = {
    'ET': 'et',
    'ETr': 'etr',
    'ET_MAD_MIN': 'et_mad_min',
    'ET_MAD_MAX': 'et_mad_max',
}

def normalize_variable_name(var: str) -> str:
    v = (var or 'et').strip()
    v = VARIABLE_ALIASES.get(v, v.lower())
    if v not in AVAILABLE_VARIABLES:
        raise ValueError(f"Unsupported variable '{var}'. Allowed: {AVAILABLE_VARIABLES} or aliases {list(VARIABLE_ALIASES.keys())}")
    return v

def _normalize_variable_list(variables) -> list[str]:
    if isinstance(variables, str):
        variables = [variables]
    return [normalize_variable_name(v) for v in variables]

def download_ensemble_monthly_multi(start_date: str, end_date: str, variables: list[str]):
    """Backward-compatible wrapper for multi-variable monthly ensemble download."""
    return download_ensemble_monthly(start_date, end_date, variables)

def download_ensemble_monthly(start_date: str, end_date: str, variable: str | list[str] = 'et'):
    """Download ensemble ET data at monthly resolution for one or more variables."""
    variables = _normalize_variable_list(variable)
    results = {}

    for v in variables:
        print(f"\n--- Downloading Ensemble {v.upper()} (monthly) ---")
        df = download_openet_data(
            shapefile_path=SHAPEFILE_PATH,
            output_dir=OUTPUT_DIR,
            start_date=start_date,
            end_date=end_date,
            env_file=str(ENV_FILE),
            variable=v,
            models=['ensemble'],
            interval='monthly',
            units='mm',
            output_filename=f'openet_{v}_ensemble_East_monthly.csv'
        )
        print(f"Downloaded {len(df)} monthly records for {v}")
        results[v] = df

    # Return a DataFrame for single-variable calls, dict for multi-variable calls
    if len(variables) == 1:
        only = variables[0]
        print("\nFirst few records:")
        print(results[only].head())
        return results[only]

    return results


def check_env_file():
    """Check if the environment file exists."""
    if not ENV_FILE.exists():
        print(f"\nError: Environment file not found at {ENV_FILE}")
        print("Please copy .env.template to .env and add your OpenET API key.")
        print("\nExample:")
        print(f"  cp {repo_root}/.env.template {repo_root}/.env")
        print("  # Edit .env and add your API key")
        return False
    return True

def download_all_models_monthly(start_date: str, end_date: str, variable: str = 'et'):
    """Download data from all models at monthly resolution."""
    print(f"\n--- Downloading All Models {variable.upper()} (monthly) ---")
    client = OpenETClient(env_file=str(ENV_FILE))
    df = client.get_multiple_models_from_shapefile(
        shapefile_path=SHAPEFILE_PATH,
        start_date=start_date,
        end_date=end_date,
        models=None,  # None = all available models
        variable=variable,
        interval='monthly',
        units='mm',
        output_path=Path(OUTPUT_DIR) / f'openet_{variable}_all_models_East_monthly.csv'
    )
    print(f"Downloaded data for {len(df.columns)} models")
    print("\nModel columns:")
    print(df.columns.tolist())
    return df


def download_ensemble_daily(start_date: str, end_date: str, variable: str = 'et'):
    """Download ensemble ET data at daily resolution."""
    print(f"\n--- Downloading Ensemble {variable.upper()} (daily) ---")
    df = download_openet_data(
        shapefile_path=SHAPEFILE_PATH,
        output_dir=OUTPUT_DIR,
        start_date=start_date,
        end_date=end_date,
        env_file=str(ENV_FILE),
        variable=variable,
        models=['ensemble'],
        interval='daily',
        units='mm',
        output_filename=f'openet_{variable}_ensemble_East_daily.csv'
    )
    print(f"Downloaded {len(df)} daily records")
    return df


def download_custom(start_date: str, end_date: str, variable: str, 
                    models: list, interval: str):
    """Download data with custom parameters."""
    print(f"\n--- Custom Download: {variable.upper()}, {models}, {interval} ---")
    
    if len(models) == 1:
        df = download_openet_data(
            shapefile_path=SHAPEFILE_PATH,
            output_dir=OUTPUT_DIR,
            start_date=start_date,
            end_date=end_date,
            env_file=str(ENV_FILE),
            variable=variable,
            models=models,
            interval=interval,
            units='mm',
            output_filename=f'openet_{variable}_{models[0]}_East_{interval}.csv'
        )
    else:
        client = OpenETClient(env_file=str(ENV_FILE))
        df = client.get_multiple_models_from_shapefile(
            shapefile_path=SHAPEFILE_PATH,
            start_date=start_date,
            end_date=end_date,
            models=models,
            variable=variable,
            interval=interval,
            units='mm',
            output_path=Path(OUTPUT_DIR) / f'openet_{variable}_{"_".join(models)}_East_{interval}.csv'
        )
    
    print(f"Downloaded {len(df)} records")
    print("\nFirst few records:")
    print(df.head())
    return df


def download_per_polygon(start_date: str, end_date: str, variable: str = 'et',
                         model: str = 'ensemble', interval: str = 'monthly',
                         id_column: str = None):
    """Download data for each polygon/HRU separately."""
    print(f"\n--- Per-Polygon Download: {variable.upper()}, {model}, {interval} ---")
    client = OpenETClient(env_file=str(ENV_FILE))
    
    # Create subdirectory for per-polygon data
    per_polygon_dir = Path(OUTPUT_DIR) / 'per_polygon'
    
    df = client.get_timeseries_per_polygon(
        shapefile_path=SHAPEFILE_PATH,
        start_date=start_date,
        end_date=end_date,
        variable=variable,
        model=model,
        interval=interval,
        units='mm',
        id_column=id_column,
        output_dir=per_polygon_dir,
        combine=True
    )
    
    print(f"\nDownloaded data for {df['polygon_id'].nunique()} polygons")
    print(f"Total records: {len(df)}")
    print(f"\nPolygon IDs: {df['polygon_id'].unique().tolist()}")
    print("\nFirst few records:")
    print(df.head(10))
    return df


def interactive_mode():
    """Run in interactive mode with prompts."""
    print("=" * 60)
    print("OpenET Data Download for East River Basin")
    print("=" * 60)
    
    if not check_env_file():
        return 1
    
    print(f"\nShapefile: {SHAPEFILE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Choose option
    print("\nDownload Options:")
    print("  1. Ensemble ET (monthly) - Recommended")
    print("  2. All Models (monthly) - For comparison")
    print("  3. Ensemble ET (daily) - Higher temporal resolution")
    print("  4. Custom - Specify all parameters")
    print("  5. Per-Polygon - Get data for each polygon/HRU separately")
    
    option = input("\nEnter option (1-5): ").strip()
    
    # Get date range
    print(f"\nDefault date range: {DEFAULT_START_DATE} to {DEFAULT_END_DATE}")
    use_default = input("Use default date range? (y/n): ").strip().lower()
    
    if use_default == 'y':
        start_date = DEFAULT_START_DATE
        end_date = DEFAULT_END_DATE
    else:
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        end_date = input("End date (YYYY-MM-DD): ").strip()
    
    # Get variable for options 1-3, 5
    variable = 'et'
    if option in ['1', '2', '3', '5']:
        print(f"\nAvailable variables: {', '.join(AVAILABLE_VARIABLES)}")
        var_input = input("Variable (default: et): ").strip().lower()
        if var_input and var_input in AVAILABLE_VARIABLES:
            variable = var_input
    
    try:
        if option == '1':
            download_ensemble_monthly(start_date, end_date, variable)
        elif option == '2':
            download_all_models_monthly(start_date, end_date, variable)
        elif option == '3':
            download_ensemble_daily(start_date, end_date, variable)
        elif option == '4':
            # Custom options
            print(f"\nAvailable variables: {', '.join(AVAILABLE_VARIABLES)}")
            variable = input("Variable: ").strip().lower()
            
            print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
            print("Enter 'all' for all models, or comma-separated list (e.g., ensemble,ssebop)")
            models_input = input("Models: ").strip().lower()
            
            if models_input == 'all':
                models = None
            else:
                models = [m.strip() for m in models_input.split(',')]
            
            print("\nAvailable intervals: daily, monthly, annual")
            interval = input("Interval (default: monthly): ").strip().lower() or 'monthly'
            
            if models is None:
                # Use all models
                client = OpenETClient(env_file=str(ENV_FILE))
                client.get_multiple_models_from_shapefile(
                    shapefile_path=SHAPEFILE_PATH,
                    start_date=start_date,
                    end_date=end_date,
                    models=None,
                    variable=variable,
                    interval=interval,
                    units='mm',
                    output_path=Path(OUTPUT_DIR) / f'openet_{variable}_all_models_East_{interval}.csv'
                )
            else:
                download_custom(start_date, end_date, variable, models, interval)
        elif option == '5':
            # Per-polygon options
            print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
            model = input("Model (default: ensemble): ").strip().lower() or 'ensemble'
            
            print("\nAvailable intervals: daily, monthly, annual")
            interval = input("Interval (default: monthly): ").strip().lower() or 'monthly'
            
            print("\nID column for polygons (leave blank for auto-detect: HRU_ID, GRU_ID, ID)")
            id_column = input("ID column: ").strip() or None
            
            download_per_polygon(start_date, end_date, variable, model, interval, id_column)
        else:
            print("Invalid option. Please choose 1-5.")
            return 1
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Files saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1
    except Exception as e:
        print(f"\nError downloading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def cli_mode(args):
    """Run with command-line arguments."""
    print("=" * 60)
    print("OpenET Data Download for East River Basin")
    print("=" * 60)
    
    if not check_env_file():
        return 1
    
    start_date = args.start_date or DEFAULT_START_DATE
    end_date = args.end_date or DEFAULT_END_DATE
    variable = normalize_variable_name(args.variable or 'et')
    variables = [normalize_variable_name(v) for v in (args.variables.split(',') if args.variables else [])]
    
    print(f"\nShapefile: {SHAPEFILE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Variables: {', '.join(variables)}")
    try:
        if args.option == 1:
            if variables:
                download_ensemble_monthly_multi(start_date, end_date, variables)
            else:
                download_ensemble_monthly(start_date, end_date, variable)
        elif args.option == 2:
            download_all_models_monthly(start_date, end_date, variable)
        elif args.option == 3:
            download_ensemble_daily(start_date, end_date, variable)
        elif args.option == 4:
            models = args.models.split(',') if args.models else ['ensemble']
            interval = args.interval or 'monthly'
            download_custom(start_date, end_date, variable, models, interval)
        elif args.option == 5:
            model = args.models.split(',')[0] if args.models else 'ensemble'
            interval = args.interval or 'monthly'
            download_per_polygon(start_date, end_date, variable, model, interval, args.id_column)
        else:
            print("Invalid option. Please choose 1-5.")
            return 1
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Files saved to: {OUTPUT_DIR}")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1
    except Exception as e:
        print(f"\nError downloading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download OpenET data for East River Basin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Options:
  1: Ensemble ET (monthly) - Recommended for most use cases
  2: All Models (monthly) - For model comparison/uncertainty analysis  
  3: Ensemble ET (daily) - Higher temporal resolution
  4: Custom - Specify all parameters
  5: Per-Polygon - Get data for each polygon/HRU separately

Examples:
  python download_East_et.py --option 1 --variables ET,ET_MAD_MIN,ET_MAD_MAX
  python download_East_et.py --option 1 --variable et
        """
    )
    
    parser.add_argument('--option', '-o', type=int, choices=[1, 2, 3, 4, 5],
                        help='Download option (1-5)')
    parser.add_argument('--start-date', '-s', type=str,
                        help=f'Start date (YYYY-MM-DD), default: {DEFAULT_START_DATE}')
    parser.add_argument('--end-date', '-e', type=str,
                        help=f'End date (YYYY-MM-DD), default: {DEFAULT_END_DATE}')
    parser.add_argument('--variable', '-v', type=str,
                        help='Single variable to download (default: et)')
    parser.add_argument('--variables', type=str,
                        help='Comma-separated variables (option 1 only), e.g. ET,ET_MAD_MIN,ET_MAD_MAX')
    parser.add_argument('--interval', '-i', type=str, choices=['daily', 'monthly', 'annual'],
                        help='Temporal interval (for options 4, 5; default: monthly)')
    parser.add_argument('--id-column', type=str,
                        help='Column name for polygon IDs (for option 5; auto-detects HRU_ID, GRU_ID, ID)')
    
    args = parser.parse_args()
    
    # If no option provided, run interactive mode
    if args.option is None:
        return interactive_mode()
    else:
        return cli_mode(args)


if __name__ == '__main__':
    sys.exit(main())