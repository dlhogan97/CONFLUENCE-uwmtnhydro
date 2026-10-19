"""Assemble result tables (markdown + csv) from the fit/SUMMA json files."""
import json, numpy as np, pandas as pd
from pathlib import Path
import fit_models as F, emu_core as E
O = F.OUT
NAMES = {"M0": "M0 baseline (independent buckets)", "M1": "M1 single-basin, full", "M2": "M2 single-basin, reduced (linear)", "M3": "M3 single-basin, memory-informed"}
def load(m, w):
    f = O/f"fit_{m}_{w}.json"; return json.load(open(f)) if f.exists() else None
def r3(x): return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
def mdtable(df): 
    cols = list(df.columns); s = "| "+" | ".join(cols)+" |\n|"+"|".join("---" for _ in cols)+"|\n"
    for _, r in df.iterrows(): s += "| "+" | ".join(str(v) for v in r.values)+" |\n"
    return s
out = []
# observed
obs = {w: E.memory(E.obs, *E.WINDOWS[w]["mem"]) for w in E.WINDOWS}
out.append("## Observed memory by window\n\n"+mdtable(pd.DataFrame([{"window": {"full": "1986-2024", "A": "1986-2005", "B": "2006-2024"}[w], "n": obs[w]["n"], "raw lag1": r3(obs[w]["raw1"]), "resid lag1": r3(obs[w]["res1"]), "raw lag5": r3(obs[w]["raw5"]), "resid lag5": r3(obs[w]["res5"])} for w in obs])))
# full-window fits
rows = []; bounds = {m: {s[0]: (s[1], s[2]) for s in F.SPEC[m]} for m in F.SPEC}
for m in ("M0", "M1", "M2", "M3"):
    d = load(m, "full")
    if not d: continue
    sk = d["eval"]["full"]["skill"]; me = d["eval"]["full"]["memory"]; p = d["params"]; peg = []
    for (n, lo, hi), xv in zip(F.SPEC[m], d["x"]):
        if xv < lo+0.01*(hi-lo) or xv > hi-0.01*(hi-lo): peg.append(n)
    rows.append({"model": NAMES[m], "free params": len(F.SPEC[m]), "objective J": f"{d['J']:.3f}", "KGE": r3(sk["KGE"]), "logNSE": r3(sk["logNSE"]), "Aug-Nov bias %": f"{sk['PB_AN']:+.1f}", "total bias %": f"{sk['PB']:+.1f}",
                 "raw lag1": r3(me["raw1"]), "resid lag1": r3(me["res1"]), "raw lag5": r3(me["raw5"]), "resid lag5": r3(me["res5"]), "prior-yr deep storage link": (r3(me.get("Sprev_pk")) if m != "M0" else "-"), "leak share": r3(d["eval"]["deep"]["leak_share"]) if m != "M0" else "-", "pegged at bound": ", ".join(peg) or "none"})
out.append("## Full-window fits (calibrated and scored on WY1987-2024; memory 1986-2024)\n\n"+mdtable(pd.DataFrame(rows)))
rows = []
for m in ("M0", "M1", "M2", "M3"):
    d = load(m, "full")
    if d: p = d["params"]; rows.append({"model": NAMES[m], "aquiferScaleFactor": f"{p['Sf_h']:.3g}", "aquiferBaseflowExp": f"{p['exp_h']:.3g}", "routingGammaScale": f"{p['scale']:.0f}", "leak coef (x rate)": f"{p['lam']:.3g}" if m != "M0" else "-", "leak exp": f"{p['m_leak']:.3g}" if m != "M0" else "-", "deep scale factor": f"{p['Sf_d']:.3g}" if m != "M0" else "-", "deep exp": f"{p['exp_d']:.3g}" if m != "M0" else "-"})
out.append("### Fitted parameters (full window)\n\n"+mdtable(pd.DataFrame(rows)))
# time split
rows = []
for m in ("M0", "M1", "M2", "M3"):
    for cal, val in (("A", "B"), ("B", "A")):
        d = load(m, cal)
        if not d: continue
        c = d["eval"][cal]; v = d["eval"][val]
        rows.append({"model": NAMES[m], "calibrated on": {"A": "1986-2005", "B": "2006-2024"}[cal], "cal KGE": r3(c["skill"]["KGE"]), "valid KGE": r3(v["skill"]["KGE"]), "cal logNSE": r3(c["skill"]["logNSE"]), "valid logNSE": r3(v["skill"]["logNSE"]),
                     "cal AugNov bias": f"{c['skill']['PB_AN']:+.0f}", "valid AugNov bias": f"{v['skill']['PB_AN']:+.0f}", "cal resid lag1": r3(c["memory"]["res1"]), "valid resid lag1": r3(v["memory"]["res1"]), "valid OBS resid lag1": r3(obs[val]["res1"]), "valid raw lag1": r3(v["memory"]["raw1"]), "valid OBS raw lag1": r3(obs[val]["raw1"])})
out.append("## Time-split validation (fit on one half, evaluated on the other)\n\n"+mdtable(pd.DataFrame(rows)))
# SUMMA vs emulator
f = O/"summa_scores.json"
if f.exists():
    sc = json.load(open(f)); rows = []
    for m in ("M0", "M1", "M2", "M3"):
        s = sc.get(f"{m}_full"); d = load(m, "full")
        if not s: continue
        e = d["eval"]["full"]; a = s["full"]
        rows.append({"model": NAMES[m], "KGE SUMMA/emulator": f"{a['skill']['KGE']:.3f} / {e['skill']['KGE']:.3f}", "logNSE": f"{a['skill']['logNSE']:.3f} / {e['skill']['logNSE']:.3f}", "AugNov bias": f"{a['skill']['PB_AN']:+.1f} / {e['skill']['PB_AN']:+.1f}", "resid lag1": f"{a['memory']['res1']:.3f} / {e['memory']['res1']:.3f}", "raw lag1": f"{a['memory']['raw1']:.3f} / {e['memory']['raw1']:.3f}"})
    out.append("## Real SUMMA runs (new singleBasin build) vs emulator, full window\n\n"+mdtable(pd.DataFrame(rows)))
open(O/"results_tables.md", "w").write("\n".join(out)); print("\n".join(out))
