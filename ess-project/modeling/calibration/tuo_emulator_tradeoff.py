"""
ET / precipitation trade-off with ASO-consistent snow (2026-11-03). Snow frozen at the emulator's aso fit; below-snow parameters refitted to observed flow
(tuo_emulator_fit.stage_soil terms) with ET held at >= ET_MIN, and water input scaled by P (a stand-in for precipitation scale: rain + melt both scaled).
Question: can the observed timing coexist with ET ~340 mm (user's estimate) at SUMMA's precipitation, or does it need more water?
-> OUT/emulator/tradeoff.csv
usage: python tuo_emulator_tradeoff.py
"""
import json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import tuo_emulator_fit as F, tuo_below_emulator as BE

CASES = [(et, p) for p in (1.0, 1.1) for et in (300.0, 340.0, 380.0)]


def one(case):
    et_min, P = case; ps = json.loads((F.OUT/"aso.json").read_text())["params"]; obs = F.obs_flow()
    swe, swi, _ = F.snow_diag(ps); swi = swi*P; snowy = swe > 1.0
    def run(pb):
        q, et = BE.simulate(pb, swi, snowy, F.X["SW"], F.X["T"], F.H, F.SPLIT, F.TOPO); d = F.flow_diag(q, obs)
        d["ET"] = float((et[F.A0:F.A1]*F.H.w.values).sum(1).resample("YS-OCT").sum().mean()); return d
    def f(u):
        d = run(BE.params_of(u))
        return ((1 - d["KGE_AprNov"])/0.1 + (abs(d["AprMay"]) + abs(d["JunJul"]) + abs(d["AugSep"]))/3/15 + abs(d["com_bias_d"])/10
                + max(et_min - d["ET"], 0, d["ET"] - 420)/10)
    r = F.fit(f, len(BE.PARAMS), 1, maxiter=120); pb = BE.params_of(r.x)
    return dict(ET_min=et_min, P=P, objective=float(r.fun), **run(pb), **{f"p_{k}": v for k, v in pb.items()})


if __name__ == "__main__":
    with ProcessPoolExecutor(len(CASES)) as ex: rows = list(ex.map(one, CASES))
    T = pd.DataFrame(rows); T.to_csv(F.OUT/"tradeoff.csv", index=False); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
