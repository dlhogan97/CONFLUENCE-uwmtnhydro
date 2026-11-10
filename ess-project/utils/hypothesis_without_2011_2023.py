"""Re-run the patch-hypothesis evidence with the 'mega snow years' removed. MEGA = 2011 and 2023 (chosen by the user; both are large-snow years: end-of-season area 3.5 and 4.0 km2).
(A) low flow vs snow area (rank, raw and partial|peak); (B) mediation of the HH low-flow trend; (C) precip-adjusted baseflow trend; (D) USGS diurnal amplitude / transition vs snow-covered area; (E) patch-melt budget share.
-> figures/draft/hypothesis_without_2011_2023.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, io, contextlib
from scipy.stats import spearmanr
U = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/"; D = U+"figures/draft/"; FS = "/scratch/dlhogan/ess-project-data/geospatial-data/snowpatch_fsca/"; rng = np.random.default_rng(5)
eosA = pd.read_csv(FS+"eos_final.csv").set_index("year"); MEGA = [2011, 2023]; print("years removed:", MEGA, "| their end-of-season areas (km2):", eosA.A_shadow[MEGA].round(1).to_dict())
def load(name, stop):
    src = open(U+"utils/"+name).read(); ns = {}
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:src.index(stop)], name, "exec"), ns)
    return ns
out = []
# ---------- (A) cross-sectional correlations
ns = load("tuo_patch_signal_integration.py", "rows = []"); TARG, PRED, pcor = ns["TARG"], ns["PRED"], ns["pcor"]; NEST = ns["NEST"]
hh_src = load("tuo_baseflow_trend_precip.py", "rows = []; keep = {}"); G = hh_src["G"]
def sepvol(q):
    q = q.asfreq("D"); r = {}
    for y in sorted(set(q.index.year)):
        z = q[(q.index.year == y) & (q.index.month == 9)].dropna()
        if len(z) >= 25: r[y] = z.mean()*30*86400/1e6
    return pd.Series(r)
for g in ["USGS11274790", "HH unimpaired"]:
    TARG[(g, "Sep volume")] = sepvol(G[g]); TARG[(g, "Aug-Nov min 7-d")] = ns["low7"](G[g])
snow = PRED["snow area (EOS min)"]; peak = hh_src["metrics"](G["HH unimpaired"]); 
pk = pd.Series({y: v for y, v in ns["ann"].groupby("year").q_mean.first().items()}) if False else None
src2 = load("tuo_lowflow_vs_snowpatch_area.py", "rows = []; DATA"); PEAK = src2["PEAK"]
print("\n(A) rank correlation of low flow with end-of-season snow area: all years | mega years removed  [raw rho (n) ; partial|peak rho]")
for (g, t), s in TARG.items():
    if t not in ("Sep volume", "Aug-Nov min 7-d") or g not in ["LBM", "LABvTB", "T120", "DFkBC", "USGS11274790", "HH unimpaired"]: continue
    d = pd.concat([s.rename("q"), snow.rename("a"), PEAK.rename("pk")], axis=1).dropna(); dd = d.drop(index=MEGA, errors="ignore")
    row = dict(part="A", gauge=g, metric=t)
    for lab, z in (("all", d), ("no_mega", dd)):
        if len(z) >= 8:
            r, p = spearmanr(z.a, z.q); rp, pp = pcor(z.a.values, z.q.values, [z.pk.values]); row.update({f"n_{lab}": len(z), f"rho_{lab}": r, f"p_{lab}": p, f"partial_{lab}": rp, f"p_partial_{lab}": pp})
    out.append(row); print(f"  {g:14s} {t:16s} all: {row.get('rho_all', np.nan):+.2f} (n={row.get('n_all', 0)}, p={row.get('p_all', np.nan):.2f}; partial {row.get('partial_all', np.nan):+.2f}) | no mega: {row.get('rho_no_mega', np.nan):+.2f} (n={row.get('n_no_mega', 0)}, p={row.get('p_no_mega', np.nan):.2f}; partial {row.get('partial_no_mega', np.nan):+.2f}, p={row.get('p_partial_no_mega', np.nan):.2f})")
# ---------- (B) mediation + (C) trend
med_ns = load("lowflow_trend_mediation.py", "rows = []; pd.set_option"); med, M, P, tair, eos = med_ns["med"], med_ns["M"], med_ns["P"], med_ns["tair"], med_ns["eos"]
fit = hh_src["fit"]
print("\n(B) mediation, HH 1985-2024, precip control; snow area = shadow-inclusive: all years vs mega removed")
for metric in ["WY min 7-d", "WY Q10", "Aug-Nov min 7-d", "Sep mean"]:
    for tier in ["A_shadow", "A_strict"]:
        df = pd.concat([M[metric].rename("Q"), eos[tier].clip(lower=0.02).rename("A"), P.rename("P"), tair.rename("Tm")], axis=1).dropna(); df = df[(df.index >= 1985) & (df.index <= 2024)]; res = {}
        for lab, z in (("all", df), ("no_mega", df.drop(index=MEGA, errors="ignore"))):
            base = med(z, "P"); boots = []
            for _ in range(2000):
                s = z.iloc[rng.integers(0, len(z), len(z))]
                if s.index.nunique() < 18: continue
                try: boots.append(med(s, "P"))
                except Exception: pass
            Bt = pd.DataFrame(boots); res[lab] = dict(n=len(z), total=100*(np.exp(base["total"])-1), tlo=100*(np.exp(np.percentile(Bt.total, 5))-1), thi=100*(np.exp(np.percentile(Bt.total, 95))-1), snow=100*(np.exp(base["a"])-1), b=base["b"], blo=np.percentile(Bt.b, 5), bhi=np.percentile(Bt.b, 95), prop=base["prop"], plo=np.percentile(Bt.prop.clip(-2, 3), 5), phi=np.percentile(Bt.prop.clip(-2, 3), 95))
        a_, b_ = res["all"], res["no_mega"]; out.append(dict(part="B", gauge="HH unimpaired", metric=f"{metric} [{tier}]", n_all=a_["n"], n_no_mega=b_["n"], total_all=a_["total"], total_no_mega=b_["total"], snow_trend_all=a_["snow"], snow_trend_no_mega=b_["snow"], b_all=a_["b"], b_no_mega=b_["b"], prop_all=a_["prop"], prop_no_mega=b_["prop"]))
        print(f"  {metric:16s} {tier:9s}  trend {a_['total']:+.1f} -> {b_['total']:+.1f} %/dec (90% CI no-mega {b_['tlo']:+.0f}..{b_['thi']:+.0f}) | snow-area trend {a_['snow']:+.0f} -> {b_['snow']:+.0f} %/dec | elasticity {a_['b']:+.2f} -> {b_['b']:+.2f} (CI {b_['blo']:+.2f}..{b_['bhi']:+.2f}) | share mediated {a_['prop']:+.2f} -> {b_['prop']:+.2f} (CI {b_['plo']:+.1f}..{b_['phi']:+.1f})")
print("\n(C) precip-adjusted baseflow trend (PRISM WY precip control), %/decade [p]: all years -> mega removed")
P_, metrics = hh_src["P"], hh_src["metrics"]
for g, rng_yrs in (("HH unimpaired", range(1982, 2026)), ("USGS11274790", range(1982, 2026))):
    Mg = metrics(G[g])
    for m in ["WY min 7-d", "WY Q10", "Aug-Nov min 7-d", "Sep mean", "Aug mean"]:
        a = fit(Mg[m], [(P_, True)], rng_yrs); b = fit(Mg[m], [(P_, True)], [y for y in rng_yrs if y not in MEGA])
        if a and b: out.append(dict(part="C", gauge=g, metric=m, n_all=a["n"], n_no_mega=b["n"], trend_all=a["pct_per_decade"], p_all=a["p"], trend_no_mega=b["pct_per_decade"], p_no_mega=b["p"])); print(f"  {g:14s} {m:16s} {a['pct_per_decade']:+6.1f} [{a['p']:.3f}] (n={a['n']}) -> {b['pct_per_decade']:+6.1f} [{b['p']:.3f}] (n={b['n']})")
# ---------- (D) diurnal vs SCA, (E) budget
dn = load("usgs_diurnal_precip_sca.py", "# transition per year"); Sy = dn["Sy"]; ATOT = dn["ATOT"]; 
def summar(Z, lab):
    y = np.log10(Z.amp_abs); ls = np.log10(Z.sca_km2.clip(lower=1)); lq = np.log10(Z.q_mean); r2 = lambda c: 1-(y-np.column_stack([np.ones(len(Z)), c])@np.linalg.lstsq(np.column_stack([np.ones(len(Z)), c]), y, rcond=None)[0]).var()/y.var()
    return dict(lab=lab, n=len(Z), years=Z.year.nunique(), rho_amp_sca=spearmanr(Z.amp_abs, Z.sca)[0], rho_hour_sca=spearmanr(Z.u, Z.sca)[0], slope=np.polyfit(ls, y, 1)[0], r2_sca=r2(ls), r2_q=r2(lq), r2_both=r2(np.column_stack([ls, lq])))
MEGA_D = [y for y in MEGA if y >= 2013]; print("\n(D) USGS diurnal amplitude vs snow-covered area (2013-2025 days): all vs mega years", MEGA_D, "removed")
for lab, Z in (("all", Sy), ("no_mega", Sy[~Sy.year.isin(MEGA_D)])):
    r = summar(Z, lab); out.append(dict(part="D", gauge="USGS11274790", metric=lab, **{k: v for k, v in r.items() if k != "lab"})); print(f"  {lab:8s} n={r['n']} days, {r['years']} yr: rho(amp,SCA)={r['rho_amp_sca']:+.2f}, rho(hour,SCA)={r['rho_hour_sca']:+.2f}, log-log slope {r['slope']:.2f}, R2 SCA {r['r2_sca']:.2f} / flow {r['r2_q']:.2f} / both {r['r2_both']:.2f}")
T = pd.read_csv(D+"usgs_diurnal_sca_transition.csv"); T = T[(T.q_med > 5) & T.sca.notna()]
for lab, z in (("all", T), ("no_mega", T[~T.year.isin(MEGA_D)])): print(f"  transition [{lab}] n={len(z)}: flow median {z.q_med.median():.1f} CV {z.q_med.std()/z.q_med.mean():.2f} | SCA median {100*z.sca.median():.1f}% CV {z.sca.std()/z.sca.mean():.2f} | doy median {z.doy.median():.0f} CV {z.doy.std()/z.doy.mean():.2f}")
B = pd.read_csv(D+"tuo_patch_melt_budget.csv"); B = B.dropna(subset=["V_flow_Mm3"]); print("\n(E) patch-melt capacity / Aug-Sep USGS flow volume (median, range): all vs mega removed", [y for y in MEGA if y >= 2007])
for lab, z in (("all", B), ("no_mega", B[~B.year.isin(MEGA)])): print(f"  {lab:8s} n={len(z)}: degree-day 4 {z.ratio_DDF4.median():.2f} ({z.ratio_DDF4.min():.2f}-{z.ratio_DDF4.max():.2f}); degree-day 8 {z.ratio_DDF8.median():.2f} ({z.ratio_DDF8.min():.2f}-{z.ratio_DDF8.max():.2f}); energy {z.ratio_SWenergy.median():.2f}; rho(V_melt, V_flow)={spearmanr(z.V_DDF8_Mm3, z.V_flow_Mm3)[0]:+.2f}")
pd.DataFrame(out).to_csv(D+"hypothesis_without_2011_2023.csv", index=False)
