#!/usr/bin/env python3
"""Seed elev+aspect SUMMA settings from an elevation-only best run.

This utility is designed for the East River workflow where:
- source trialParams has 5 HRUs (one per elevation band)
- target trialParams has 25 HRUs (5 elevation bands x 5 aspects)

For HRU-dimensioned variables, each elevation-band value is repeated across
all aspects for that band. Other dimensions are copied when shapes match.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


COPYABLE_SETTING_FILES = [
    "modelDecisions.txt",
    "localParamInfo.txt",
    "basinParamInfo.txt",
    "outputControl.txt",
    "TBL_GENPARM.TBL",
    "TBL_MPTABLE.TBL",
    "TBL_SOILPARM.TBL",
    "TBL_VEGPARM.TBL",
]


def _resolve_settings_dir(path: Path) -> Path:
    """Accept either a run root, settings root, or SUMMA settings dir."""
    p = path.expanduser().resolve()
    candidates = [
        p,
        p / "settings",
        p / "settings" / "SUMMA",
        p / "SUMMA",
    ]
    for c in candidates:
        if (c / "trialParams.nc").exists() and (c / "modelDecisions.txt").exists():
            return c
    raise FileNotFoundError(
        f"Could not find SUMMA settings folder under: {path}. "
        "Expected files include trialParams.nc and modelDecisions.txt."
    )


def _copy_settings_files(
    source_settings: Path,
    target_settings: Path,
    copy_file_manager: bool,
    dry_run: bool,
) -> None:
    files = list(COPYABLE_SETTING_FILES)
    if copy_file_manager:
        files.append("fileManager.txt")

    missing = []
    copied = []
    for name in files:
        src = source_settings / name
        dst = target_settings / name
        if not src.exists():
            missing.append(name)
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(name)

    verb = "Would copy" if dry_run else "Copied"
    print(f"{verb} {len(copied)} settings files to {target_settings}")
    if copied:
        print("  " + ", ".join(copied))
    if missing:
        print("Missing in source (skipped):")
        print("  " + ", ".join(missing))


def _repeat_hru_axis(values: np.ndarray, hru_axis: int, repeats: int) -> np.ndarray:
    return np.repeat(values, repeats=repeats, axis=hru_axis)


def _target_hru_from_attributes(target_trial: Path) -> tuple[int | None, np.ndarray | None]:
    """Return target HRU count (and optional hruId values) from attributes.nc."""
    attrs_path = target_trial.parent / "attributes.nc"
    if not attrs_path.exists():
        return None, None

    with xr.open_dataset(attrs_path) as attrs_in:
        attrs = attrs_in.load()

    hru_count = int(attrs.sizes.get("hru", 0)) if "hru" in attrs.sizes else 0
    hru_ids = None
    if "hruId" in attrs and "hru" in attrs["hruId"].dims:
        hru_ids = np.asarray(attrs["hruId"].values).reshape(-1)
        if hru_ids.size > 0:
            hru_count = int(hru_ids.size)

    if hru_count <= 0:
        return None, None
    return hru_count, hru_ids


def _expand_trial_params(
    source_trial: Path,
    target_trial: Path,
    n_bands: int,
    aspects_per_band: int,
    create_missing_vars: bool,
    dry_run: bool,
) -> None:
    with xr.open_dataset(source_trial) as src_in:
        src = src_in.load()
    with xr.open_dataset(target_trial) as tgt_in:
        tgt = tgt_in.load()

    src_hru = int(src.sizes.get("hru", 0))
    tgt_hru = int(tgt.sizes.get("hru", 0))
    expected_tgt = n_bands * aspects_per_band
    attr_hru, attr_hru_ids = _target_hru_from_attributes(target_trial)
    desired_tgt_hru = int(attr_hru or (tgt_hru if tgt_hru > 0 else expected_tgt))

    if src_hru not in (0, n_bands):
        raise ValueError(
            f"Source HRU size is {src_hru}, expected {n_bands} for elevation bands."
        )
    if desired_tgt_hru <= 0:
        raise ValueError(
            "Could not determine target HRU size from attributes.nc or target trialParams.nc."
        )

    print(
        "Target HRU sizing: "
        f"attributes={attr_hru if attr_hru is not None else 'n/a'}, "
        f"trialParams={tgt_hru}, using={desired_tgt_hru}"
    )
    if desired_tgt_hru != expected_tgt:
        print(
            "Note: target HRU size differs from n_bands*aspects_per_band "
            f"({expected_tgt}); proceeding with attributes-based size {desired_tgt_hru}."
        )

    if "hru" in tgt.dims and tgt_hru > 0 and tgt_hru != desired_tgt_hru:
        # Rebuild all HRU-dimensioned variables to the authoritative attributes.nc size.
        tgt = tgt.drop_dims("hru")
        print(
            f"Dropped existing HRU-dimensioned target variables (hru={tgt_hru}) "
            f"to rebuild at hru={desired_tgt_hru}."
        )

    if "hru" not in tgt.dims:
        if attr_hru_ids is not None and attr_hru_ids.size == desired_tgt_hru:
            tgt = tgt.assign_coords(hru=("hru", attr_hru_ids.astype(np.int64, copy=False)))
        else:
            tgt = tgt.assign_coords(hru=("hru", np.arange(1, desired_tgt_hru + 1, dtype=np.int64)))

    updated = []
    skipped = []

    def _adapt_values(src_da: xr.DataArray, target_dims: tuple[str, ...], target_shape: tuple[int, ...]) -> np.ndarray:
        if tuple(src_da.dims) != tuple(target_dims):
            raise ValueError(f"dim mismatch src={src_da.dims} target={target_dims}")

        src_vals = src_da.values
        if src_vals.shape == target_shape:
            return src_vals

        if "hru" not in src_da.dims:
            raise ValueError(f"shape mismatch src={src_vals.shape} target={target_shape}")

        hru_axis = src_da.dims.index("hru")
        src_hru_len = src_vals.shape[hru_axis]
        tgt_hru_len = target_shape[hru_axis]

        if src_hru_len == tgt_hru_len:
            return src_vals

        if src_hru_len == n_bands and tgt_hru_len >= n_bands and tgt_hru_len % n_bands == 0:
            repeats = tgt_hru_len // n_bands
            expanded = _repeat_hru_axis(src_vals, hru_axis=hru_axis, repeats=repeats)
            if expanded.shape != target_shape:
                raise ValueError(f"expanded shape {expanded.shape} != target {target_shape}")
            return expanded

        if src_hru_len == 1:
            return np.broadcast_to(src_vals, target_shape)

        raise ValueError(f"unsupported HRU sizes src={src_hru_len} target={tgt_hru_len}")

    for name, src_da in src.data_vars.items():
        if name == "hruId" and attr_hru_ids is not None:
            tgt[name] = xr.DataArray(
                attr_hru_ids.astype(src_da.dtype, copy=False),
                dims=("hru",),
            )
            updated.append(name)
            continue

        if name in tgt.data_vars:
            tgt_da = tgt[name]
            try:
                tgt[name].values[...] = _adapt_values(src_da, tuple(tgt_da.dims), tuple(tgt_da.shape))
                updated.append(name)
            except ValueError as exc:
                skipped.append((name, str(exc)))
            continue

        if not create_missing_vars:
            skipped.append((name, "missing in target"))
            continue

        target_dims = tuple(src_da.dims)
        target_shape = []
        for d in target_dims:
            if d == "hru":
                target_shape.append(desired_tgt_hru)
            else:
                target_shape.append(int(tgt.sizes.get(d, src.sizes[d])))
        target_shape_t = tuple(target_shape)

        try:
            new_vals = _adapt_values(src_da, target_dims, target_shape_t)
            tgt[name] = xr.DataArray(new_vals, dims=target_dims)
            updated.append(name)
        except ValueError as exc:
            skipped.append((name, f"cannot create: {exc}"))

    print(f"trialParams update summary: {len(updated)} updated, {len(skipped)} skipped")
    if skipped:
        print("Skipped variables:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if dry_run:
        print("Dry run enabled: no file written.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target_trial.with_suffix(target_trial.suffix + f".bak_{stamp}")
    shutil.copy2(target_trial, backup)
    tgt.to_netcdf(target_trial)
    print(f"Backed up target trialParams -> {backup}")
    print(f"Wrote expanded trialParams -> {target_trial}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Seed elev+aspect SUMMA settings from elevation-only run and expand "
            "trialParams HRU variables from 5 bands to 25 HRUs."
        )
    )
    p.add_argument(
        "--source",
        required=True,
        help=(
            "Source path: run directory, settings directory, or SUMMA settings directory "
            "from elevation-only best run."
        ),
    )
    p.add_argument(
        "--target",
        required=True,
        help="Target path: settings directory or SUMMA settings directory for elev+aspect domain.",
    )
    p.add_argument(
        "--source-trial",
        default="trialParams.nc",
        help="Source trial params filename inside source settings (default: trialParams.nc).",
    )
    p.add_argument(
        "--target-trial",
        default="trialParams.nc",
        help="Target trial params filename inside target settings (default: trialParams.nc).",
    )
    p.add_argument(
        "--n-bands",
        type=int,
        default=5,
        help="Number of elevation bands in source trial params (default: 5).",
    )
    p.add_argument(
        "--aspects-per-band",
        type=int,
        default=5,
        help="Number of aspect classes per elevation band in target (default: 5).",
    )
    p.add_argument(
        "--skip-copy-settings",
        action="store_true",
        help="Do not copy text/table settings files from source to target.",
    )
    p.add_argument(
        "--copy-file-manager",
        action="store_true",
        help="Also copy fileManager.txt from source to target (off by default).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report changes without writing target trialParams.nc.",
    )
    p.add_argument(
        "--no-create-missing-vars",
        action="store_true",
        help="Do not create variables missing in target trialParams file.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    source_settings = _resolve_settings_dir(Path(args.source))
    target_settings = _resolve_settings_dir(Path(args.target))

    source_trial = source_settings / args.source_trial
    target_trial = target_settings / args.target_trial
    if not source_trial.exists():
        raise FileNotFoundError(f"Source trialParams file not found: {source_trial}")
    if not target_trial.exists():
        raise FileNotFoundError(f"Target trialParams file not found: {target_trial}")

    print(f"Source settings: {source_settings}")
    print(f"Target settings: {target_settings}")
    print(f"Source trialParams: {source_trial.name}")
    print(f"Target trialParams: {target_trial.name}")

    if not args.skip_copy_settings:
        _copy_settings_files(
            source_settings=source_settings,
            target_settings=target_settings,
            copy_file_manager=args.copy_file_manager,
            dry_run=bool(args.dry_run),
        )
    else:
        print("Skipping settings-file copy (requested).")

    _expand_trial_params(
        source_trial=source_trial,
        target_trial=target_trial,
        n_bands=int(args.n_bands),
        aspects_per_band=int(args.aspects_per_band),
        create_missing_vars=not bool(args.no_create_missing_vars),
        dry_run=bool(args.dry_run),
    )

    print("Done.")


if __name__ == "__main__":
    main()
