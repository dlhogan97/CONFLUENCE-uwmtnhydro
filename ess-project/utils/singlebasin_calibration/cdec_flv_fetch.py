"""Vogelsang (CDEC FLV) incoming/reflected solar + albedo, event data 2021-07..2025-12 -> modeling/calibration/observations/cdec_FLV_radiation.csv (PST timestamps in CDEC)"""
import io, time, urllib.request, pandas as pd
OUT = '/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/cdec_FLV_radiation.csv'
parts = []
for y in range(2021, 2026):
    for a, b in ((f'{y}-01-01', f'{y}-04-01'), (f'{y}-04-01', f'{y}-07-01'), (f'{y}-07-01', f'{y}-10-01'), (f'{y}-10-01', f'{y+1}-01-01')):
        if y == 2021 and a < '2021-07-01': continue
        u = f'https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=FLV&SensorNums=245,246,247&dur_code=E&Start={a}&End={b}'
        try: t = urllib.request.urlopen(u, timeout=180).read().decode()
        except Exception as e: print(a, 'ERR', e); continue
        d = pd.read_csv(io.StringIO(t)); parts.append(d); print(a, len(d), flush=True); time.sleep(0.5)
d = pd.concat(parts); d['time'] = pd.to_datetime(d['DATE TIME']); d['VALUE'] = pd.to_numeric(d['VALUE'], errors='coerce')
w = d.pivot_table(index='time', columns='SENSOR_NUMBER', values='VALUE', aggfunc='mean').rename(columns={245: 'albedo_pct', 246: 'sw_up', 247: 'sw_down'})
w.to_csv(OUT); print(w.describe().round(1).to_string()); print(w.index.min(), w.index.max())
