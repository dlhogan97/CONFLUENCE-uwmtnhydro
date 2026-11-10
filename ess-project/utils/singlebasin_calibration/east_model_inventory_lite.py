"""Daily series of the main East models (subset of east_model_inventory.py), for side analyses."""
import os, json
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd, netCDF4 as nc
import emu_core as E, stageB_emulator as B, east_ctl_treat_cal as C, east_threshold_cal as T

D = "/scratch/dlhogan/ess-project-data/domain_East_River_distributed_elevAspect/simulations/store_memory_sandbox/East/deepstore_dev/"


def _best(path):
    tr = [json.loads(l) for l in open(path)]; return min([r for r in tr if r["objective"] < 10], key=lambda r: r["objective"])["params"]


def _summa(run, fn):
    d = nc.Dataset(f"{run}/out/{fn}"); t = pd.DatetimeIndex(nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    A = float(np.array(nc.Dataset(f"{run}/attributes.nc")["HRUarea"][:]).sum()); return (pd.Series(np.array(d["averageRoutedRunoff"][:]).ravel(), index=t)*A).resample("D").mean()


def load_main():
    ef = pd.read_csv(C.X.OUT/"east_final_models_daily.csv", index_col=0, parse_dates=True)
    rd = lambda m: pd.read_csv(D+f"overnight/summa_daily_{m}_full.csv", index_col=0, parse_dates=True)["q_cms"]
    return {"observed": E.obs, "current control": ef.control, "current deep store": ef.treatment,
            "threshold + deep, DE": T.run_model(_best(T.OUTD/"trials_treatment.jsonl"), "treatment")[0],
            "SUMMA overnight M0": rd("M0"), "SUMMA overnight M1": rd("M1"),
            "SUMMA baseline": _summa("/scratch/dlhogan/ess-project-data/calibration/bestruns/East_River_baseline_1981_2025/run", "baseline_timestep.nc"),
            "SUMMA control, fast aquifer": _summa(B.SRC["ctlfull"][0], "best_timestep.nc")}
