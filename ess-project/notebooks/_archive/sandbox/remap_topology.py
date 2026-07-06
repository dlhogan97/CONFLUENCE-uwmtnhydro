import argparse
import xarray as xr
import numpy as np
from pathlib import Path

def build_ordered_unique(arr):
    uniq, idx = np.unique(arr, return_index=True)
    order = np.argsort(idx)
    return uniq[order]

def remap_topology(inpath, outpath, remap_all=False, sentinel_vals=(0,-1)):
    ds = xr.open_dataset(inpath)
    if "segId" not in ds or "downSegId" not in ds:
        raise RuntimeError("Expected segId and downSegId variables in topology file")

    seg_orig = ds["segId"].values.flatten()
    down_orig = ds["downSegId"].values.flatten()

    # preserve original order of seg entries
    seg_unique = build_ordered_unique(seg_orig)
    mapping = {int(old): int(i+1) for i, old in enumerate(seg_unique)}  # 1..N

    # remap segId
    seg_new = np.array([mapping[int(v)] for v in seg_orig], dtype=int)
    # remap downSegId: keep sentinel values (0 or -1) as-is; warn on unknown positive ids
    def remap_down(v):
        iv = int(v)
        if iv in mapping:
            return mapping[iv]
        if iv in sentinel_vals:
            return iv
        raise ValueError(f"downSegId value {iv} not in segId mapping and not a sentinel")

    down_new = np.array([remap_down(v) for v in down_orig], dtype=int)

    # Create copy and assign new arrays (preserve original dims/shape)
    ds_out = ds.copy()
    ds_out["segId"].values = seg_new.reshape(ds["segId"].values.shape)
    ds_out["downSegId"].values = down_new.reshape(ds["downSegId"].values.shape)

    if remap_all:
        # Replace any integer-valued variables that exactly match seg_unique values (use with care)
        segset = set(int(x) for x in seg_unique)
        for name, var in list(ds_out.data_vars.items()):
            if np.issubdtype(var.dtype, np.integer):
                vals = var.values
                flat = vals.flatten()
                if flat.size and set(np.unique(flat)).issubset(segset.union(set(sentinel_vals))):
                    # map values
                    mapped = np.vectorize(lambda x: mapping[int(x)] if int(x) in mapping else int(x))(flat)
                    ds_out[name].values = mapped.reshape(vals.shape)

    # Validation
    final_seg = np.unique(ds_out["segId"].values)
    if not np.array_equal(final_seg, np.arange(1, final_seg.size+1)):
        raise RuntimeError("Remapped segId not contiguous 1..N")

    # Check downstream references in range or sentinel
    ds_out.to_netcdf(outpath)
    ds.close()
    ds_out.close()
    print(f"Wrote remapped topology to: {outpath}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--remap-all", action="store_true", help="Also remap any integer vars referencing segId")
    args = p.parse_args()
    remap_topology(Path(args.infile), Path(args.outfile), remap_all=args.remap_all)