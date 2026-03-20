import re
from pathlib import Path
from typing import Dict


def _format_parameter_value(value) -> str:
    """
    Format parameter value with 4 decimal places.
    Uses 'd' notation for exponentials (e.g., 1.0d-9) and fixed notation otherwise.
    
    Always converts to float first to ensure consistent formatting regardless of input type.
    """
    # Convert to float, handling string inputs
    try:
        val = float(value)
    except (ValueError, TypeError):
        # If conversion fails, return as-is with padding
        return f"{str(value):>12}"
    
    # Handle scientific notation for very large/small numbers
    if abs(val) < 0.001 or abs(val) > 1e5:
        # Format with 4 decimal places in scientific notation
        formatted = f"{val:.4e}"
        # Replace 'e' with 'd' for Fortran-style double precision notation
        formatted = formatted.replace('e', 'd')
        return formatted.rjust(12)
    else:
        # Regular notation with exactly 4 decimal places
        return f"{val:12.4f}"


def update_and_reformat_parameter_file(
    file_path: Path, 
    parameter_updates: Dict[str, float],
    reformat_all: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Update a parameter file with new values and optionally reformat all values to 4-decimal precision.
    
    This is the primary function for updating parameter files with clean, controlled behavior.
    It handles both updating specific parameter values and reformatting all values to ensure 
    consistent 4-decimal precision formatting.
    
    Args:
        file_path: Path to localParamInfo.txt or basinParamInfo.txt
        parameter_updates: Dictionary of parameter names to new values to update
                          Can be empty dict {} to skip updates and only reformat
        reformat_all: If True, reformats ALL values in file to 4-decimal precision (default: True)
        verbose: If True, prints what was updated
        
    Returns:
        Dictionary with:
            - 'updated_count': number of parameters updated with new values
            - 'reformatted_count': number of parameter lines reformatted
            
    Example:
        >>> # Option 1: Use manual edits only
        >>> manual_updates = {'k_soil': 9.4e-6, 'theta_sat': 0.516}
        >>> update_and_reformat_parameter_file(localParamInfo_file, manual_updates)
        
        >>> # Option 2: Use best parameters from optimization
        >>> best_params = pd.read_csv('best_parameters.csv')
        >>> best_dict = dict(zip(best_params['parameter'], best_params['value']))
        >>> update_and_reformat_parameter_file(localParamInfo_file, best_dict)
        
        >>> # Option 3: Only reformat, no updates
        >>> update_and_reformat_parameter_file(localParamInfo_file, {})
    """
    pattern = re.compile(r'^(\s*[^!][A-Za-z0-9_]+)(\s*\|)([^|]+)(\|)([^|]+)(\|)([^|]+)(\|.*)$')

    with open(file_path, 'r') as fin:
        lines = fin.readlines()

    new_lines = []
    updated_count = 0
    reformatted_count = 0

    for line in lines:
        m = pattern.match(line)
        if m:
            param_name = m.group(1)  # Parameter name with whitespace
            pipe1 = m.group(2)
            value_str = m.group(3).strip()
            pipe2 = m.group(4)
            lower_str = m.group(5).strip()
            pipe3 = m.group(6)
            upper_str = m.group(7).strip()
            rest = m.group(8)

            # Step 1: Update with new value if provided
            var = param_name.strip()
            if var in parameter_updates:
                value_str = str(parameter_updates[var])
                updated_count += 1

            # Step 2: Reformat all values to 4-decimal precision
            if reformat_all:
                value_formatted = _format_parameter_value(value_str)
                lower_formatted = _format_parameter_value(lower_str)
                upper_formatted = _format_parameter_value(upper_str)
                line = f"{param_name}{pipe1}{value_formatted}{pipe2}{lower_formatted}{pipe3}{upper_formatted}{rest}\n"
                reformatted_count += 1
        
        new_lines.append(line)
    
    with open(file_path, 'w') as fout:
        fout.writelines(new_lines)
    
    if verbose:
        result_msg = f"Updated {file_path.name}:"
        if updated_count > 0:
            result_msg += f"\n  ✓ Updated {updated_count} parameters with new values"
        if reformat_all:
            result_msg += f"\n  ✓ Reformatted {reformatted_count} lines to 4-decimal precision"
        print(result_msg)
    
    return {'updated_count': updated_count, 'reformatted_count': reformatted_count}


def update_file_manager(filepath, **kwargs):
    """
    Update key-value pairs in a SUMMA file manager text file.
    
    Usage:
        update_file_manager("fileManager.txt", 
                            outFilePrefix="new_prefix",
                            simStartTime="2014-10-01 01:00")
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    updated = []
    for line in lines:
        # skip empty lines and comments
        if line.strip() == '' or line.strip().startswith('!'):
            updated.append(line)
            continue
        
        key = line.split()[0]
        if key in kwargs:
            value = kwargs[key]
            # wrap in quotes if not already
            if not value.startswith("'"):
                value = f"'{value}'"
            updated.append(f"{key:<25}{value}\n")
        else:
            updated.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(updated)
    
    print(f"Updated {filepath}:")
    for k, v in kwargs.items():
        print(f"  {k} → {v}")


def edit_modelDecisions(file_path, updates):
    """
    Update model decision selections in modelDecisions.txt file.
    
    Replaces the method (2nd token) for parameters listed in updates dictionary.
    
    Args:
        file_path: Path to modelDecisions.txt
        updates: Dictionary where keys are parameter names and values are new method choices
        
    Example:
        >>> edit_modelDecisions('modelDecisions.txt', {
        ...     'groundwatr': 'bigBuckt',
        ...     'bcLowrSoiH': 'drainage'
        ... })
    """
    with open(file_path, 'r') as fin:
        lines = fin.readlines()

    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('!') or stripped.strip() == '':
            new_lines.append(line)
            continue

        parts = line.split('!', 1)
        code = parts[0].strip()
        comment = '!' + parts[1] if len(parts) > 1 else ''

        tokens = code.split()
        if len(tokens) >= 2:
            param, method = tokens[0], tokens[1]
            if param in updates:
                method = updates[param]
            new_code = f"{param:<15}{method:<25}"
            new_line = f"{new_code}{(' ' + comment) if comment else ''}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(file_path, 'w') as fout:
        fout.writelines(new_lines)