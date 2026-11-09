"""Fidelity of the channel-twin variants against held-out ray-traced test wards."""
import glob, pickle, numpy as np, jax, jax.numpy as jnp
from scipy.stats import wasserstein_distance, ks_2samp
import twin, env

if __name__ == '__main__':
    G = pickle.load(open('twin_test_gen.pkl', 'rb'))
P = pickle.load(open('twin_params.pkl', 'rb'))  # AR-MDN baseline (one-step NLL only)
BURN = 200
VARS = ('twin', 'uncoupled', 'memoryless', 'phys')


def stats(x, g0):
    x = x[BURN:].astype(np.float32); g0 = g0[BURN:].astype(np.float32)
    vis = g0 > -119
    blk = vis & (x < g0 - 3)
    on = blk[1:] & ~blk[:-1]
    off = ~blk[1:] & blk[:-1]
    T = x.shape[0]
    nvis = vis.sum()
    dur = blk.sum() / max(on.sum(), 1) * 10                 # ms
    multi = blk.sum(2)[blk.any(2)]
    best = x.max(2)
    return dict(blk_frac=blk.sum() / nvis, onset_per_min=on.sum() / (vis.mean(0).sum() * T / 6000), dur_ms=dur,
                co_blk=float(multi.mean()) if multi.size else 0.0, best=best.ravel(), xvis=x[vis][::7], xall=x.ravel()[::11])


def nll_test():
    files = sorted(glob.glob('corpus/test_*.pkl'))
    X, Y = twin.dataset(files, 0.02, 3)
    Xn = (X - P['mu']) / P['sd']
    out = {}
    for k in ('mdn', 'gauss'):
        p = jax.tree.map(jnp.asarray, P[k])
        out[k] = float(np.mean([np.asarray(twin.nll(p, jnp.asarray(Xn[i:i + 200000]), jnp.asarray(Y[i:i + 200000]))).mean() for i in range(0, len(X), 200000)]))
    return out


if __name__ == '__main__':
    R = {v: [] for v in ('true',) + VARS}
    for f, gen in G.items():
        e = pickle.load(open(f, 'rb'))
        R['true'].append(stats(e['G'], e['G0']))
        for v in VARS:
            R[v].append(stats(gen[v], e['G0']))
    S = {}
    for v in R:
        S[v] = {k: float(np.mean([r[k] for r in R[v]])) for k in ('blk_frac', 'onset_per_min', 'dur_ms', 'co_blk')}
        if v != 'true':
            tb = np.concatenate([r['best'] for r in R['true']]); vb = np.concatenate([r['best'] for r in R[v]])
            tv = np.concatenate([r['xvis'] for r in R['true']]); vv = np.concatenate([r['xvis'] for r in R[v]])
            S[v]['W1_best_db'] = float(wasserstein_distance(tb[::5], vb[::5]))
            S[v]['KS_best'] = float(ks_2samp(tb[::50], vb[::50]).statistic)
            S[v]['W1_vis_db'] = float(wasserstein_distance(tv, vv))
    S['nll'] = nll_test()
    S['best_samples'] = {v: np.concatenate([r['best'] for r in R[v]])[::40] for v in R}
    pickle.dump(S, open('results/twin_fidelity.pkl', 'wb'))
    for v in R:
        print(v, {k: round(val, 4) for k, val in S[v].items()})
    print(S['nll'])
