import re

def edit_localParamInfo(file_path, updates):
    """
    Overwrites the file, replacing the first value for variables listed in updates.
    """
    pattern = re.compile(r'^(\s*[^!][A-Za-z0-9_]+)(\s*\|)([^|]+)(\|.*)$')
    with open(file_path, 'r') as fin:
        lines = fin.readlines()

    new_lines = []
    for line in lines:
        m = pattern.match(line)
        if m:
            var = m.group(1).strip()
            if var in updates:
                new_val = f" {updates[var]:>12} "
                line = f"{m.group(1)}{m.group(2)}{new_val}{m.group(4)} \n"
        new_lines.append(line)

    with open(file_path, 'w') as fout:
        fout.writelines(new_lines)

def edit_modelDecisions(file_path, updates):
    """
    Overwrites the file, replacing the method (2nd token) for parameters listed in updates.
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