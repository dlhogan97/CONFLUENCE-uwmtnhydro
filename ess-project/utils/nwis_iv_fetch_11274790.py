"""Fetch 15-min instantaneous discharge (00060, cfs) for USGS 11274790, Aug 1 - Oct 31 each year 2007-2025 -> modeling/calibration/observations/usgs11274790_iv_augoct.csv (local time as reported, tz_cd column; discharge converted to m3/s)."""
import io, time, urllib.request, pandas as pd
OUT = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/usgs11274790_iv_augoct.csv"; parts = []
for y in range(2007, 2026):
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=11274790&parameterCd=00060&startDT={y}-08-01&endDT={y}-10-31&siteStatus=all"
    for k in range(3):
        try: txt = urllib.request.urlopen(url, timeout=120).read().decode(); break
        except Exception as e: print("retry", y, e); time.sleep(5)
    d = pd.read_csv(io.StringIO(txt), sep="\t", comment="#", skiprows=[1]); 
    if len(d) == 0: print(y, "no rows"); continue
    d.columns = ["agency", "site", "datetime", "tz", "q_cfs", "qual"]; d["q_cfs"] = pd.to_numeric(d.q_cfs, errors="coerce"); d["q_cms"] = d.q_cfs*0.0283168; parts.append(d[["datetime", "tz", "q_cms", "qual"]]); print(y, len(d), d.qual.value_counts().to_dict(), d.tz.unique())
pd.concat(parts).to_csv(OUT, index=False); print("saved", OUT)
