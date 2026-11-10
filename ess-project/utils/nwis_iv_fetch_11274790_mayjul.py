"""Fetch 15-min instantaneous discharge (00060, cfs) for USGS 11274790, May 1 - Jul 31 each year 2007-2025 -> modeling/calibration/observations/usgs11274790_iv_mayjul.csv (local time as reported; m3/s)."""
import io, time, urllib.request, pandas as pd
OUT = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/usgs11274790_iv_mayjul.csv"; parts = []
for y in range(2007, 2026):
    url = f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=11274790&parameterCd=00060&startDT={y}-05-01&endDT={y}-07-31&siteStatus=all"
    for k in range(3):
        try: txt = urllib.request.urlopen(url, timeout=120).read().decode(); break
        except Exception as e: print("retry", y, e); time.sleep(5)
    d = pd.read_csv(io.StringIO(txt), sep="\t", comment="#", dtype=str); d = d[d.iloc[:, 3].isin(["PDT", "PST"])]
    d.columns = ["agency", "site", "datetime", "tz", "q_cfs", "qual"]; d["q_cms"] = pd.to_numeric(d.q_cfs, errors="coerce")*0.0283168; parts.append(d[["datetime", "tz", "q_cms", "qual"]]); print(y, len(d), d.tz.unique())
pd.concat(parts).to_csv(OUT, index=False); print("saved", OUT)
