"""Corpus statistics (Table IV) and NEWS2 summaries."""
import glob, pickle, json, numpy as np
import env

out = {}
for split in ('train', 'val', 'test', 'big'):
    fs = sorted(glob.glob(f'corpus/{split}_*.pkl'))
    S = dict(eps=len(fs), beds=[], nodes=[], lum=[], rx=[], staff=[], walk=[], curtain=[], blk=[], best=[], best_sub=[], rf=[], area=[])
    for f in fs:
        e = pickle.load(open(f, 'rb'))
        x = e['G'].astype(np.float32); g0 = e['G0'].astype(np.float32)
        vis = g0 > -119
        X, RF, nm, km = env.load_episode(e)
        K = len(e['lum'])
        S['beds'].append(len(e['walk'])); S['nodes'].append(x.shape[1]); S['lum'].append(K); S['rx'].append(int(km.sum()))
        S['staff'].append(e['n_staff']); S['walk'].append(float(np.mean(e['walk']))); S['curtain'].append(float(e['curtain'].any(0).mean()))
        S['blk'].append(float((vis & (x < g0 - 3)).sum() / vis.sum()))
        S['best'].append(float(np.median(x.max(2))))
        S['best_sub'].append(float(np.median(x[:, :, km[:K]].max(2))))
        S['rf'].append(float(np.median(e['snr_rf'])))
        S['area'].append(float(e['L'] * e['W']))
    out[split] = {k: (v if k == 'eps' else [float(np.mean(v)), float(np.min(v)), float(np.max(v))]) for k, v in S.items()}
out['news2'] = json.load(open('news2_stats.json'))
json.dump(out, open('results/corpus_stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'news2'}, indent=0)[:1500])
