"""Fetch hourly-ish solar_radiation (+QC flags) for the two Yosemite-area stations from Synoptic, cache to CSV (no token in the output)."""
import json, time, urllib.request, urllib.parse, pandas as pd
TOK = open('/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/.synoptic_token').read().strip()
OUT = '/home/dlhogan/projects/forked-repos/CONFLUENCE-uwmtnhydro/ess-project/modeling/calibration/observations/'
def pull(stid, y0, y1):
    rows = []
    for y in range(y0, y1+1):
        for (a, b) in ((f'{y}01010000', f'{y}07010000'), (f'{y}07010000', f'{y+1}01010000')):
            q = urllib.parse.urlencode(dict(stid=stid, token=TOK, start=a, end=b, vars='solar_radiation', obtimezone='utc', qc='off', qc_flags='on', units='metric'))
            try: r = json.load(urllib.request.urlopen('https://api.synopticdata.com/v2/stations/timeseries?'+q, timeout=120))
            except Exception as e: print(stid, a, 'ERR', type(e).__name__); continue
            if r['SUMMARY'].get('RESPONSE_CODE') != 1 or not r.get('STATION'): continue
            o = r['STATION'][0]['OBSERVATIONS']; k = [k for k in o if k.startswith('solar_radiation_set')]
            if not k: continue
            v = o[k[0]]; fl = o.get('qc_flags', {}).get(k[0]+'_qc') or o.get('qc', {}).get(k[0])
            df = pd.DataFrame({'time': pd.to_datetime(o['date_time']), 'sw': v})
            if fl is not None and len(fl) == len(v): df['qc'] = fl
            rows.append(df); time.sleep(0.3)
        print(stid, y, sum(len(r) for r in rows), flush=True)
    d = pd.concat(rows).drop_duplicates('time').sort_values('time'); d['time'] = d.time.dt.tz_localize(None) if d.time.dt.tz is not None else d.time
    d.to_csv(OUT+f'synoptic_{stid}_solar.csv', index=False); return d
for stid, y0 in (('VGNC1', 2022), ('WWRC1', 1999)): 
    d = pull(stid, y0, 2025); print(stid, len(d), d.time.min(), d.time.max(), d.sw.describe().round(1).to_dict())
