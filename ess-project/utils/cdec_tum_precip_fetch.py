"""Fetch CDEC TUM (Tuolumne Meadows, 2620 m) daily accumulated precipitation (sensor 2, inches; resets 1 Oct) Apr 1 - Oct 31, 2007-2025 -> modeling/calibration/observations/tum_daily_precip.csv (accum_in, increment_in, increment_mm)."""
import io, urllib.request, pandas as pd, numpy as np
OUT = "/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/tum_daily_precip.csv"; parts = []
for y in range(2007, 2026):
    u = f"https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=TUM&SensorNums=2&dur_code=D&Start={y}-04-01&End={y}-11-01"
    d = pd.read_csv(io.StringIO(urllib.request.urlopen(u, timeout=60).read().decode())); d["date"] = pd.to_datetime(d["OBS DATE"].str[:8]); d["accum_in"] = pd.to_numeric(d.VALUE, errors="coerce"); parts.append(d[["date", "accum_in"]]); print(y, len(d), int(d.accum_in.notna().sum()))
x = pd.concat(parts).drop_duplicates("date").set_index("date").sort_index().asfreq("D"); x["increment_in"] = x.accum_in.diff(); x.loc[x.increment_in < -0.5, "increment_in"] = np.nan   # resets / glitches
x["increment_mm"] = x.increment_in*25.4; x.to_csv(OUT); print(x[(x.index.month >= 5) & (x.index.month <= 9)].increment_in.describe())
