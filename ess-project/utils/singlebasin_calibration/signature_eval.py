import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration")
sys.path.insert(0,"/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/utils/singlebasin_calibration")
import signature_objective as S
import emu_core as E, fit_models as F
O=F.OUT; obs=E.obs
runs={"original baseline (calibrated 2026-07)":pd.read_csv("/scratch/dlhogan/ess-project-data/streamflow-data/EastRiver/EastRiver_SUMMA_baseline_dailyDischarge_1981_2025.csv",index_col=0,parse_dates=True)["discharge"]}
for m,nm in (("M0","M0 baseline recalibrated"),("M1","M1 single-basin"),("M2","M2 single-basin linear"),("M3","M3 single-basin memory-fit")): runs[nm]=pd.read_csv(O/f"summa_daily_{m}_full.csv",index_col=0,parse_dates=True)["q_cms"]
P=("1986-10-01","2024-09-30"); AREA=E.Atot
rows=[]; monthly={}
for nm,q in runs.items():
    j=pd.concat([q.rename("s"),obs.rename("o")],axis=1).dropna().loc[P[0]:P[1]]; s=j.s.clip(lower=S.Q_FLOOR); o=j.o.clip(lower=S.Q_FLOOR)
    r=S.signature_objective(s,o,27.0,area_m2=AREA)
    # extra, easily read diagnostics
    tq_s=S.directional_stats_by_wy(s)["tQ_days"]; tq_o=S.directional_stats_by_wy(o)["tQ_days"]; c=tq_s.index.intersection(tq_o.index)
    pk=lambda x:x.rolling(7,center=True).mean().groupby(x.index.year+(x.index.month>=10)).max()
    mj=lambda x:x[x.index.month.isin([5,6,7])].groupby((x.index.year+(x.index.month>=10))[x.index.month.isin([5,6,7])]).sum()
    mm=lambda x:x.groupby(x.index.month).mean()
    monthly[nm]=100*(mm(s)/mm(o)-1)
    lf=lambda x:x[x.index.month.isin([9,10,11])].rolling(7,center=True,min_periods=7).mean().groupby(x[x.index.month.isin([9,10,11])].index.year).min()
    rows.append({"run":nm,"objective":r.objective,**{k:v for k,v in r.parts.items()},
      "total volume bias %":100*(s.sum()/o.sum()-1),"COM timing bias (d)":float((tq_s.loc[c]-tq_o.loc[c]).mean()),"COM timing |err| (d)":float((tq_s.loc[c]-tq_o.loc[c]).abs().mean()),
      "annual peak (7d) bias %":100*(pk(s).mean()/pk(o).mean()-1),"May-Jul volume bias %":100*(mj(s).mean()/mj(o).mean()-1),
      "Sep-Nov 7d-min bias % (median yr)":100*(lf(s).median()/lf(o).median()-1),"tau sim (d)":r.diagnostics["tau_sim_days"],"KGE (cost 1-KGE)":1-r.diagnostics["KGE"]})
df=pd.DataFrame(rows).set_index("run"); pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print("signature errors (0 = perfect; objective is their equal-weight mean) and readable diagnostics, WY1987-2024"); print(df.round(3).T.to_string())
print("\nmean monthly flow bias % (model vs obs), WY1987-2024:"); print(pd.DataFrame(monthly).reindex([10,11,12,1,2,3,4,5,6,7,8,9]).round(1).to_string())
df.to_csv("sig_eval.csv"); pd.DataFrame(monthly).to_csv("sig_monthly_bias.csv")
