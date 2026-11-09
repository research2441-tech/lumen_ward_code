"""Episode pools (ray-traced, twin-generated, physics-only) and NEWS2 severity sampling."""
import glob, pickle, numpy as np, pandas as pd
import env

_cache = {}


def _load(files, key='G'):
    out = []
    for f in files:
        e = pickle.load(open(f, 'rb'))
        if key != 'G':
            e['G'] = e[key]
        out.append(env.load_episode(e))
    return out


def pool(src, n=None):
    if (src, n) in _cache:
        return _cache[(src, n)]
    if src in ('rt', 'train'):
        p = _load(sorted(glob.glob('corpus/train_*.pkl')))
    elif src == 'val':
        p = _load(sorted(glob.glob('corpus/val_*.pkl')))
    elif src == 'test':
        p = _load(sorted(glob.glob('corpus/test_*.pkl')))
    elif src == 'big':
        p = _load(sorted(glob.glob('corpus/big_*.pkl')))
    elif src in ('twin', 'phys', 'memoryless', 'uncoupled'):
        p = _load(sorted(glob.glob(f'synth_gen/{src}_*.pkl'))[:n], key='G')
    _cache[(src, n)] = p
    return p


def severity(db, part):
    d = pd.read_csv(f'news2_{db}.csv')
    stays = np.sort(d.stay.unique())
    rr = np.random.default_rng(11)
    tr = set(rr.choice(stays, len(stays) // 2, replace=False))
    if db == 'mimic':
        d = d[d.stay.isin(tr)] if part == 'train' else d[~d.stay.isin(tr)]
    return d.w.values.astype(np.float32), d.tier.values


def draw_w(sev, nm, rng):
    w, _ = sev
    P = len(nm) // 3
    wp = w[rng.integers(len(w), size=P)]
    return np.where(nm, np.repeat(wp, 3), 1.0).astype(np.float32)
