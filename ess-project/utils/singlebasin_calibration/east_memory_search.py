"""
East River: can a simple added component carry low-flow memory it was not fitted to? (screen-only search, 2026-10-30)
  baseline    SUMMA paper run (runs/East_River_20260720_094546), every parameter frozen; recharge, surface runoff and aquifer parameters as in east_paper_booster.py.
  emulator    daily, one-way: SUMMA recharge/runoff -> added component -> HRU buckets + stream. --stage validate checks it against the SUMMA baseline.
  structures  fixed list below, at most 4 added parameters each, water conserved. All divert water from the baseline's own recharge or surface runoff, so
              precipitation is never added. Bucket and routing parameters stay frozen.
  stages      screen  Latin hypercube across each structure's bounds: what can it do at all? Reported only, never used to pick parameters.
              fit     differential evolution on the PAPER objective, WY1993-2002, with several seeds. Memory is not in the objective.
              report  best fit of each structure against the pre-set gates (GATES). Near-equal fits are shown as a range; a structure is judged on its best fit.
  significance rho1 is compared to the largest rho1 across ALL structures under one shared permutation of years, so testing many structures is paid for.
  mechanism   recharge of each wet year is raised 50% and the response of the FOLLOWING fall minimum is measured against the unperturbed run.
Resumes: finished stage/structure/seed outputs are skipped. Outputs -> stageB/memory_search/ (report.md is the summary).
usage: python east_memory_search.py --stage validate|screen|fit|report|all [--workers 20]
"""
import os, sys, json, time, argparse, logging, warnings; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from pathlib import Path
os.environ.setdefault("STAGEB_T0", "1981-10-01"); os.environ.setdefault("STAGEB_END", "2025-09-30")
import numpy as np, pandas as pd, numba as nb
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve
from scipy.stats import gamma, qmc
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"modeling"/"calibration"))
from signature_objective import signature_objective
import emu_core as E, east_paper_booster as PB, east_booster as EB

st, AT, P0, obs = PB.st, PB.AT, PB.P0, EB.obs
OUTD = PB.B.OUT/"memory_search"; OUTD.mkdir(exist_ok=True)
CAL0, CAL1, TAU_TARGET, BASE_OBJ = "1992-10-01", "2002-09-30", 27.0, 0.3370
D = pd.DatetimeIndex(pd.Series(0, index=st["tt"]).resample("1D").mean().index)
_day = lambda a: pd.DataFrame(a, index=st["tt"]).resample("1D").mean().values
RCH, SR = _day(st["Rch"]), _day(st["SR"]); A = st["A"].astype(float); DT = 86400.0

F, TAU, RTH, N, SMAX, BETA, STH, TAU2 = ("f", 0.001, 0.8, False), ("tau_d", 30.0, 3650.0, True), ("rth_mm", 0.1, 10.0, True), ("n", 1.0, 4.0, False), ("smax_mm", 1.0, 500.0, True), ("beta", 0.1, 5.0, False), ("sth_mm", 1.0, 1000.0, True), ("tau2_d", 365.0, 7300.0, True)
#            id: (code, parameters). linear = the structure that already failed, kept as the negative control.
STRUCT = {"linear": (0, [F, TAU]), "cascade": (1, [F, TAU, N]), "thresh": (2, [F, RTH, TAU]), "thresh_cascade": (3, [F, RTH, TAU, N]),
          "fillspill": (4, [SMAX, BETA, TAU]), "connect": (5, [F, STH, TAU, TAU2]), "sr_capture": (6, [F, TAU, N])}
GATES = dict(kge_eval=0.8324, nse_eval=0.7123, paper_obj=BASE_OBJ, recess_within_d=15.0, fallmin_within=10.0, rho1_09_above=0.0, p_family=0.05)


@nb.njit
def _core(code, Rch, SRr, A, rate, Sf, ex, f, rth, tau, n, smax, beta, sth, tau2, dt, passes, Qout, Sout):
    NT, NH = Rch.shape; At = A.sum(); k = rate/Sf**ex; S = np.ones(NH)*0.5; C = np.zeros(4); C[0] = sth; Fs = np.zeros(NH)
    for p in range(passes):
        for t in range(NT):
            inflow = 0.0; q = 0.0; drain = 0.0
            for i in range(NH):
                R = Rch[t, i]; sr = SRr[t, i]; div = 0.0; w = A[i]/At
                if code <= 1:
                    if R > 0: div = f*R
                elif code == 2 or code == 3:
                    if R > rth: div = f*(R-rth)
                elif code == 4:
                    if R > 0:
                        pas = min(1.0, (Fs[i]/smax)**beta); div = (1.0-pas)*R
                    fn = (Fs[i]+dt*div)/(1.0+dt/tau); drain += fn/tau*w; Fs[i] = fn
                elif code == 5:
                    if R > 0: div = f*R
                elif code == 6:
                    sc = f*sr; sr = sr-sc; inflow += sc*w
                if code != 4: inflow += div*w
                U = R-div; s0 = S[i]; sn = s0
                for it in range(30):
                    fx = sn-s0-dt*U+dt*k*sn**ex; fp = 1.0+dt*k*ex*sn**(ex-1.0); step = fx/fp; sn = sn-step
                    if sn < 1e-12: sn = 1e-12
                    if abs(step) < 1e-12*max(1.0, sn): break
                S[i] = sn; q += (sr+k*sn**ex)*w
            if code == 4: Qd = drain
            elif code == 5:
                s = (C[0]+dt*inflow+dt/tau*sth)/(1.0+dt/tau+dt/tau2)         # Q = S/tau2 always, plus (S-Sth)/tau once the store connects above Sth
                if s < sth: s = (C[0]+dt*inflow)/(1.0+dt/tau2)
                Qd = s/tau2+max(s-sth, 0.0)/tau; C[0] = s
            else:
                x = inflow
                for j in range(n):
                    C[j] = (C[j]+dt*x)/(1.0+dt/tau); x = C[j]/tau
                Qd = x
            Qout[t] = q+Qd; Sout[t] = C.sum()+(Fs*A).sum()/At
    return S


_uh = None
def uh():
    global _uh
    if _uh is None:
        c = gamma.cdf(np.arange(0, 121)*3600.0*2.0, a=E.SHAPE, scale=P0["scale"]); h = np.diff(c)/np.diff(c).sum()      # 2-h kernel averaged to days, inflow uniform within the day
        hh = np.convolve(h, np.ones(12)/12.0); _uh = np.add.reduceat(hh, np.arange(0, len(hh), 12)); _uh = _uh/_uh.sum()
    return _uh


def simulate(name, p, rch=RCH, passes=3):
    code = STRUCT[name][0]; g = lambda k, d: float(p.get(k, d)); Qout, Sout = np.zeros(len(rch)), np.zeros(len(rch))
    _core(code, rch, SR, A, P0["rate"], g("Sf_h", P0["Sf"]), g("exp_h", P0["exp"]), g("f", 0.0), g("rth_mm", 0.0)/1000.0/86400.0, g("tau_d", 318.0)*86400.0, int(round(g("n", 1.0))), g("smax_mm", 1.0)/1000.0, g("beta", 1.0), g("sth_mm", 0.0)/1000.0, g("tau2_d", 3650.0)*86400.0, DT, passes, Qout, Sout)
    return pd.Series(fftconvolve(Qout, uh())[:len(Qout)]*AT, index=D), pd.Series(Sout, index=D)


def paper_objective(q):
    r = signature_objective(q[CAL0:CAL1], obs[CAL0:CAL1], TAU_TARGET, area_m2=AT); return float(r.objective)


def rho1(q, y0=1987): return float(E.lag_r(E.resid(*E.series(q, y0, 2024)), 1))


def params_of(name, u):
    """u in [0, 1]^k -> physical parameters (log bounds where flagged)."""
    out = {}
    for (k, lo, hi, lg), v in zip(STRUCT[name][1], u): out[k] = float(10**(np.log10(lo)+v*(np.log10(hi)-np.log10(lo))) if lg else lo+v*(hi-lo))
    return out


def trial(args):
    name, u, fh = args; p = params_of(name, u); q, _ = simulate(name, p)
    rec = dict(params=p, objective=paper_objective(q), rho1_87=rho1(q), rho1_09=rho1(q, 2009))
    if fh:
        with open(fh, "a") as g: g.write(json.dumps(rec)+"\n")
    return rec


class Objective:
    def __init__(self, name, fh): self.name, self.fh = name, fh
    def __call__(self, u): return trial((self.name, u, self.fh))["objective"]


def validate():
    d = PB.nc.Dataset(PB.B.SRC["old"][0]+"/out/"+PB.B.SRC["old"][1]); t = pd.DatetimeIndex(PB.nc.num2date(d["time"][:], d["time"].units, only_use_cftime_datetimes=False))
    s = (pd.Series(np.array(d["averageRoutedRunoff"][:]).ravel(), index=t)*AT).resample("1D").mean(); d.close(); t0 = time.time()
    q, _ = simulate("linear", dict(f=0.0)); dt = time.time()-t0
    a, b = EB.score(s), EB.score(q); j = pd.concat([s.rename("s"), q.rename("q")], axis=1).dropna()["1986-10-01":"2024-09-30"]
    print(f"daily emulator vs SUMMA baseline: r = {j.corr().iloc[0, 1]:.5f}, mean {j.q.mean():.3f} vs {j.s.mean():.3f} m3/s, {dt:.2f} s per run; paper objective {paper_objective(q):.4f} vs {BASE_OBJ}")
    print(pd.DataFrame(dict(SUMMA=a, daily=b)).round(3).to_string())
    for name in STRUCT:                                                       # water balance: every structure must return what it took, apart from final storage
        p = params_of(name, [0.5]*len(STRUCT[name][1])); q1, sd = simulate(name, p, passes=1); y = pd.concat([q1.rename("a"), s.rename("b")], axis=1).dropna()
        print(f"  {name:15s} {p}  mean flow {q1['1986-10-01':'2024-09-30'].mean():.3f} m3/s  end storage {sd.iloc[-1]*1000:.0f} mm  rho1 {rho1(simulate(name, p)[0]):+.2f}")


def screen(workers, n=5000):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for name in STRUCT:
            fh = OUTD/f"screen_{name}.jsonl"
            if fh.exists() and sum(1 for _ in open(fh)) >= n: continue
            fh.unlink(missing_ok=True); U = qmc.LatinHypercube(d=len(STRUCT[name][1]), seed=1).random(n); t0 = time.time()
            for _ in pool.imap_unordered(trial, [(name, u, str(fh)) for u in U], chunksize=20): pass
            print(f"screen {name}: {n} trials, {time.time()-t0:.0f} s", flush=True)


def fit(workers, seeds=(1, 2, 3, 4, 5)):
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for name in STRUCT:
            k = len(STRUCT[name][1])
            for seed in seeds:
                res = OUTD/f"fit_{name}_s{seed}.json"
                if res.exists(): continue
                fh = OUTD/f"fit_{name}_s{seed}.jsonl"; fh.unlink(missing_ok=True); t0 = time.time()
                r = differential_evolution(Objective(name, str(fh)), bounds=[(0.0, 1.0)]*k, popsize=15, maxiter=60, tol=1e-4, seed=seed, workers=pool.map, updating="deferred", polish=False, init="latinhypercube")
                res.write_text(json.dumps(dict(params=params_of(name, r.x), objective=float(r.fun), nfev=int(r.nfev))))
                print(f"fit {name} seed {seed}: obj {r.fun:.4f} rho1 {rho1(simulate(name, params_of(name, r.x))[0]):+.2f} ({time.time()-t0:.0f} s)", flush=True)


def resid_series(q): return E.resid(*E.series(q, 1987, 2024))


def pulse_response(name, p, q0):
    """Raise April-July recharge 50% in each wet year (top third of recharge) and return the median % change of the next year's fall minimum."""
    wy = D.year+(D.month >= 10); tot = pd.Series(RCH.sum(1), index=D).groupby(wy).sum().loc[1987:2023]; lf0 = np.exp(E.series(q0, 1987, 2024)[0]); out = []
    for y in tot[tot >= tot.quantile(2/3)].index:
        r = RCH.copy(); m = (D >= f"{y}-04-01") & (D <= f"{y}-07-31"); r[m] *= 1.5; q1, _ = simulate(name, p, rch=r)
        v = np.exp(E.series(q1, y+1, y+1)[0])
        if y+1 in lf0.index and len(v): out.append(100.0*(v.iloc[0]/lf0[y+1]-1.0))
    return float(np.median(out))


def report():
    rows, near, res = [], [], {}
    q0, _ = simulate("linear", dict(f=0.0)); base = dict(model="baseline (daily emulator)", paper_obj=paper_objective(q0), **EB.score(q0)); rows.append(base)
    pb = pulse_response("linear", dict(f=0.0), q0); rows[0]["next_fall_pct_per_wet_pulse"] = pb
    for name in STRUCT:
        fits = [json.loads(open(f).read()) for f in sorted(OUTD.glob(f"fit_{name}_s*.json"))]
        if not fits: continue
        b = min(fits, key=lambda x: x["objective"]); q, sd = simulate(name, b["params"]); res[name] = resid_series(q)
        rows.append(dict(model=name, paper_obj=b["objective"], params=json.dumps({k: round(v, 3) for k, v in b["params"].items()}), seed_spread=max(x["objective"] for x in fits)-b["objective"],
                         next_fall_pct_per_wet_pulse=pulse_response(name, b["params"], q), store_mm=float(sd["1986-10-01":"2024-09-30"].mean()*1000), **EB.score(q)))
        tr = [json.loads(l) for f in OUTD.glob(f"fit_{name}_s*.jsonl") for l in open(f)]; ok = [t for t in tr if t["objective"] <= 1.02*b["objective"]]
        near.append(dict(model=name, n_near=len(ok), rho1_87_lo=min(t["rho1_87"] for t in ok), rho1_87_hi=max(t["rho1_87"] for t in ok), rho1_09_lo=min(t["rho1_09"] for t in ok), rho1_09_hi=max(t["rho1_09"] for t in ok)))
    R = pd.DataFrame(rows).set_index("model"); N = pd.DataFrame(near).set_index("model") if near else pd.DataFrame()
    # family-wise null: one shared permutation of years across every structure, keep the largest rho1
    M = pd.concat(res, axis=1).dropna(); rng = np.random.default_rng(7); act = {k: E.lag_r(M[k], 1) for k in M}; null = []
    for _ in range(5000):
        i = rng.permutation(len(M)); null.append(max(E.lag_r(pd.Series(M[k].values[i], index=M.index), 1) for k in M))
    null = np.array(null); R["p_family"] = pd.Series({k: (np.sum(null >= v)+1)/(len(null)+1) for k, v in act.items()})
    ow = obs; O = EB.score(ow); G = GATES
    R["pass_skill"] = (R.KGE_eval >= G["kge_eval"]) & (R.NSE_eval >= G["nse_eval"]) & (R.paper_obj <= G["paper_obj"])
    R["pass_flow"] = ((R.recess_d-O["recess_d"]).abs() <= G["recess_within_d"]) & (R.fallmin_bias.abs() <= G["fallmin_within"]) & ((R.winter_tau_d-O["winter_tau_d"]).abs() < abs(base["winter_tau_d"]-O["winter_tau_d"]))
    R["pass_memory"] = (R.p_family < G["p_family"]) & (R.rho1_09 > G["rho1_09_above"]) & (R.next_fall_pct_per_wet_pulse > 0)
    R["PASS"] = R.pass_skill & R.pass_flow & R.pass_memory; R.loc[R.index[0], "PASS"] = False
    S = []
    for name in STRUCT:
        f = OUTD/f"screen_{name}.jsonl"
        if f.exists():
            t = pd.read_json(f, lines=True); g = t[t.objective <= 1.10*BASE_OBJ]
            S.append(dict(model=name, n=len(t), n_within_10pct_obj=len(g), rho1_max_all=t.rho1_87.max(), rho1_max_within_10pct=g.rho1_87.max() if len(g) else np.nan))
    S = pd.DataFrame(S).set_index("model") if S else pd.DataFrame()
    R.to_csv(OUTD/"report_best.csv"); N.to_csv(OUTD/"report_near.csv"); S.to_csv(OUTD/"report_screen.csv")
    cols = ["paper_obj", "KGE_eval", "NSE_eval", "recess_d", "winter_tau_d", "fallmin_bias", "rho1_87", "p_family", "rho1_09", "next_fall_pct_per_wet_pulse", "store_mm", "pass_skill", "pass_flow", "pass_memory", "PASS"]
    md = f"# East memory search ({pd.Timestamp.now():%Y-%m-%d %H:%M})\n\nObserved: {json.dumps({k: round(O[k], 2) for k in ('recess_d', 'winter_tau_d', 'rho1_87', 'rho1_09')})}\n\n## Best fit per structure (parameters fitted to the paper objective only)\n\n{R[cols].round(3).to_markdown()}\n\n## Parameters\n\n{R[['params', 'seed_spread']].round(4).to_markdown()}\n\n## Near-equal fits (within 2% of best; range only, not used to pass)\n\n{N.round(2).to_markdown() if len(N) else ''}\n\n## Screen: what each structure can do at all (not used to pick parameters)\n\n{S.round(2).to_markdown() if len(S) else ''}\n"
    (OUTD/"report.md").write_text(md); print(md)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--workers", type=int, default=20); a = ap.parse_args()
    if a.stage == "validate": validate()
    if a.stage in ("screen", "all"): screen(a.workers)
    if a.stage in ("fit", "all"): fit(a.workers)
    if a.stage in ("report", "all"): report()


if __name__ == "__main__":
    main()
