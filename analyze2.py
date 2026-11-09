"""Summaries, paired tests and derived metrics from evaluation outputs."""
import pickle, numpy as np
from scipy.stats import wilcoxon

SLOT_MS = 10.0


def per_ep(o):
    n_al = np.maximum(o['n_al'], 1)
    d = dict(swaoi=o['swaoi'] * SLOT_MS, aoi=o['aoi'] * SLOT_MS, aoi_hi=o['aoi_hi'] * SLOT_MS,
             viol=100 * o['viol'] / n_al, fail=100 * o['fail_tx'] / np.maximum(o['tx'], 1),
             rfshare=100 * o['ok_rf'] / np.maximum(o['ok_opt'] + o['ok_rf'], 1),
             att=(o['tx'] + o['rf']) / np.maximum(o['ok_opt'] + o['ok_rf'], 1), ho=o['ho'], n_al=o['n_al'], vcount=o['viol'])
    c = np.cumsum(o['ahist'], 1); c = c / c[:, -1:]
    d['p99'] = (np.argmax(c >= 0.99, 1)) * SLOT_MS
    lc = o['lhist'].sum(0)
    d['lat_cdf'] = np.cumsum(lc) / max(lc.sum(), 1)
    return d


def table(R, names, seeds=(0, 1, 2)):
    out = {}
    for n in names:
        eps = [per_ep(R[(n, s)]) for s in seeds if (n, s) in R]
        if not eps:
            continue
        row = {}
        for k in ('swaoi', 'aoi', 'aoi_hi', 'fail', 'rfshare', 'att', 'p99'):
            seed_means = [e[k].mean() for e in eps]
            row[k] = (float(np.mean(seed_means)), float(np.std(seed_means)))
        tot_v = sum(e['vcount'].sum() for e in eps); tot_a = sum(e['n_al'].sum() for e in eps)
        row['viol'] = (100.0 * tot_v / max(tot_a, 1), 0.0)
        row['alarms'] = int(tot_a)
        row['lat_cdf'] = np.mean([e['lat_cdf'] for e in eps], 0)
        row['ep_swaoi'] = np.mean([e['swaoi'] for e in eps], 0)
        row['ep_fail'] = np.mean([e['fail'] for e in eps], 0)
        row['ep_aoi'] = np.mean([e['aoi'] for e in eps], 0)
        out[n] = row
    return out


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def paired(T, ref, others, key='ep_swaoi'):
    rows = []
    for n in others:
        a, b = T[ref][key], T[n][key]
        d = a - b
        if np.allclose(d, 0):
            rows.append((n, 0.0, 1.0, 0.0)); continue
        res = wilcoxon(a, b, method='approx')
        r = abs(res.zstatistic) / np.sqrt(len(a))
        rows.append((n, float(d.mean()), float(res.pvalue), float(r)))
    adj = holm([r[2] for r in rows])
    return [(r[0], r[1], r[2], a, r[3]) for r, a in zip(rows, adj)]
