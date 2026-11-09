"""NEWS2 trajectories from the open MIMIC-IV and eICU-CRD demo databases.

Hourly NEWS2 (Royal College of Physicians, 2017, SpO2 scale 1) with last-observation-carried-forward
windows; each patient-hour maps to the guideline monitoring tier and to a scheduling weight.
"""
import numpy as np, pandas as pd, json, os

MIMIC = os.environ.get('MIMIC_DIR', './mimic')
EICU = os.environ.get('EICU_DIR', './eicu')

# guideline minimum observation interval (h) per clinical-response tier; weight = 1 / interval
TIER_INTERVAL = {0: 12.0, 1: 5.0, 2: 1.0, 3: 1.0, 4: 0.25}
TIER_NAME = {0: 'score 0', 1: 'low (1-4)', 2: 'single red 3', 3: 'medium (5-6)', 4: 'high (>=7)'}


def sc_rr(x):
    return np.select([x <= 8, x <= 11, x <= 20, x <= 24], [3, 1, 0, 2], 3)


def sc_spo2(x):
    return np.select([x <= 91, x <= 93, x <= 95], [3, 2, 1], 0)


def sc_sbp(x):
    return np.select([x <= 90, x <= 100, x <= 110, x <= 219], [3, 2, 1, 0], 3)


def sc_hr(x):
    return np.select([x <= 40, x <= 50, x <= 90, x <= 110, x <= 130], [3, 1, 0, 1, 2], 3)


def sc_temp(x):
    return np.select([x <= 35.0, x <= 36.0, x <= 38.0, x <= 39.0], [3, 1, 0, 1], 2)


def score(df):
    """df columns: rr, spo2, sbp, hr, temp, o2 (0/1), alert (0/1). Missing temp/alert contribute 0."""
    parts = np.stack([sc_rr(df.rr.values), sc_spo2(df.spo2.values), sc_sbp(df.sbp.values), sc_hr(df.hr.values),
                      np.where(df.temp.isna(), 0, sc_temp(df.temp.fillna(37).values)),
                      2 * df.o2.fillna(0).values.astype(int), 3 * (1 - df.alert.fillna(1).values.astype(int))], 1)
    tot = parts.sum(1)
    red = (parts[:, [0, 1, 2, 3, 4, 6]] == 3).any(1)
    tier = np.where(tot >= 7, 4, np.where(tot >= 5, 3, np.where(red, 2, np.where(tot >= 1, 1, 0))))
    return tot, tier


def _hourly(ev, key, limits):
    """ev: rows (stay, hour float, var, value). Hourly grid with LOCF limited per variable."""
    out = []
    for sid, g in ev.groupby(key):
        h0, h1 = int(np.floor(g.hour.min())), int(np.ceil(g.hour.max()))
        grid = pd.DataFrame({'hour': np.arange(h0, h1 + 1, dtype=float)})
        for var, lim in limits.items():
            s = g[g['var'] == var].sort_values('hour')[['hour', 'value']].rename(columns={'hour': 'h_' + var, 'value': var})
            if len(s) == 0:
                grid[var] = np.nan
                continue
            m = pd.merge_asof(grid, s, left_on='hour', right_on='h_' + var, direction='backward')
            m.loc[(m.hour - m['h_' + var]) > lim, var] = np.nan
            grid[var] = m[var].values
        grid['stay'] = sid
        out.append(grid)
    d = pd.concat(out, ignore_index=True)
    return d.dropna(subset=['rr', 'spo2', 'sbp', 'hr'])


LIM = dict(rr=4, spo2=4, sbp=4, hr=4, temp=8, o2=12, alert=8)


def mimic():
    c = pd.read_csv(f'{MIMIC}/icu/chartevents.csv.gz', usecols=['stay_id', 'charttime', 'itemid', 'valuenum', 'value'])
    st = pd.read_csv(f'{MIMIC}/icu/icustays.csv.gz', usecols=['stay_id', 'intime'])
    c = c.merge(st, on='stay_id')
    c['hour'] = (pd.to_datetime(c.charttime) - pd.to_datetime(c.intime)).dt.total_seconds() / 3600
    m = {220210: 'rr', 224690: 'rr', 220277: 'spo2', 220179: 'sbp', 220050: 'sbp', 220045: 'hr'}
    ev = c[c.itemid.isin(m)].assign(var=lambda d: d.itemid.map(m), value=lambda d: d.valuenum)
    tf = c[c.itemid == 223761].assign(var='temp', value=lambda d: (d.valuenum - 32) / 1.8)
    tc = c[c.itemid == 223762].assign(var='temp', value=lambda d: d.valuenum)
    o2 = c[c.itemid == 226732].assign(var='o2', value=lambda d: (~d.value.fillna('None').isin(['None'])).astype(float))
    ey = c[c.itemid == 220739].assign(var='alert', value=lambda d: (d.valuenum >= 4).astype(float))
    ev = pd.concat([ev, tf, tc, o2, ey])[['stay_id', 'hour', 'var', 'value']].dropna(subset=['value'])
    ev = ev[(ev['var'] != 'temp') | ((ev.value > 30) & (ev.value < 43))]
    ev = ev[(ev['var'] != 'hr') | ((ev.value > 10) & (ev.value < 250))]
    ev = ev[(ev['var'] != 'sbp') | ((ev.value > 30) & (ev.value < 300))]
    ev = ev[(ev['var'] != 'spo2') | ((ev.value > 50) & (ev.value <= 100))]
    ev = ev[(ev['var'] != 'rr') | ((ev.value > 0) & (ev.value < 70))]
    return _hourly(ev, 'stay_id', LIM)


def eicu():
    vp = pd.read_csv(f'{EICU}/vitalPeriodic.csv.gz', usecols=['patientunitstayid', 'observationoffset', 'temperature', 'sao2', 'heartrate', 'respiration', 'systemicsystolic'])
    va = pd.read_csv(f'{EICU}/vitalAperiodic.csv.gz', usecols=['patientunitstayid', 'observationoffset', 'noninvasivesystolic'])
    nc = pd.read_csv(f'{EICU}/nurseCharting.csv.gz', usecols=['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel', 'nursingchartcelltypevalname', 'nursingchartvalue'])
    rows = []
    for col, var in (('temperature', 'temp'), ('sao2', 'spo2'), ('heartrate', 'hr'), ('respiration', 'rr'), ('systemicsystolic', 'sbp')):
        d = vp[['patientunitstayid', 'observationoffset', col]].dropna().rename(columns={col: 'value', 'observationoffset': 'off'})
        rows.append(d.assign(var=var))
    rows.append(va.dropna().rename(columns={'noninvasivesystolic': 'value', 'observationoffset': 'off'}).assign(var='sbp'))
    lab = nc.nursingchartcelltypevallabel
    for lb, var in (('Heart Rate', 'hr'), ('Respiratory Rate', 'rr'), ('O2 Saturation', 'spo2'), ('Temperature', 'temp')):
        d = nc[lab == lb]
        if var == 'temp':
            d = d[d.nursingchartcelltypevalname.str.contains('(C)', regex=False)]
        d = d.assign(value=pd.to_numeric(d.nursingchartvalue, errors='coerce'))
        rows.append(d[['patientunitstayid', 'nursingchartoffset', 'value']].rename(columns={'nursingchartoffset': 'off'}).assign(var=var))
    d = nc[(lab == 'Non-Invasive BP') & (nc.nursingchartcelltypevalname == 'Non-Invasive BP Systolic')]
    rows.append(d.assign(value=pd.to_numeric(d.nursingchartvalue, errors='coerce'))[['patientunitstayid', 'nursingchartoffset', 'value']].rename(columns={'nursingchartoffset': 'off'}).assign(var='sbp'))
    d = nc[lab == 'O2 Admin Device']
    rows.append(d.assign(value=(~d.nursingchartvalue.str.lower().str.contains('room air|ra$', na=False)).astype(float))[['patientunitstayid', 'nursingchartoffset', 'value']].rename(columns={'nursingchartoffset': 'off'}).assign(var='o2'))
    d = nc[lab == 'Glasgow coma score']
    d = d[d.nursingchartcelltypevalname == 'Eyes']
    rows.append(d.assign(value=(pd.to_numeric(d.nursingchartvalue, errors='coerce') >= 4).astype(float))[['patientunitstayid', 'nursingchartoffset', 'value']].rename(columns={'nursingchartoffset': 'off'}).assign(var='alert'))
    ev = pd.concat(rows).dropna(subset=['value'])
    ev['hour'] = ev.off / 60.0
    ev = ev.rename(columns={'patientunitstayid': 'stay_id'})
    rng = dict(temp=(30, 43), hr=(10, 250), sbp=(30, 300), spo2=(50, 100.01), rr=(0.1, 70))
    for v, (a, b) in rng.items():
        ev = ev[(ev['var'] != v) | ((ev.value > a) & (ev.value < b))]
    return _hourly(ev[['stay_id', 'hour', 'var', 'value']], 'stay_id', LIM)


def build():
    out = {}
    for name, fn in (('mimic', mimic), ('eicu', eicu)):
        d = fn()
        tot, tier = score(d)
        d = d.assign(news2=tot, tier=tier, w=[1 / TIER_INTERVAL[t] for t in tier])
        d.to_csv(f'news2_{name}.csv', index=False)
        out[name] = dict(stays=int(d.stay.nunique()), hours=int(len(d)),
                         tier_frac={TIER_NAME[k]: float((tier == k).mean()) for k in range(5)},
                         news2_mean=float(tot.mean()), news2_median=float(np.median(tot)))
    json.dump(out, open('news2_stats.json', 'w'), indent=1)
    return out


def load(name):
    return pd.read_csv(f'news2_{name}.csv')


if __name__ == '__main__':
    print(json.dumps(build(), indent=1))
