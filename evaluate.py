"""Evaluate scheduling policies on held-out ray-traced wards (test, big) with MIMIC-IV (held-out stays)
and eICU-CRD severity; configuration sweeps. Output: results/eval_<name>.pkl"""
import sys, os, pickle, glob, numpy as np, jax, jax.numpy as jnp
import env, data

os.makedirs('results', exist_ok=True)
pat = jnp.asarray(np.repeat(np.arange(env.N_MAX // 3), 3))
TH0 = env.init_theta(jax.random.PRNGKey(0))


def learned(tag):
    return jax.tree.map(jnp.asarray, pickle.load(open(f'es_{tag}.pkl', 'rb'))['theta'])


def policies(avail):
    P = [('RR', 'rr', TH0, False, 0.0), ('MW-AoI', 'mw', TH0, False, 0.0), ('TO', 'to', TH0, False, 0.0),
         ('DGET-obj', 'dget', TH0, False, 0.0), ('SW-Whittle', 'wsw', TH0, False, 0.0),
         ('SW-Whittle+C+AP', 'wsw_ap', TH0, True, 0.0)]
    if 'twin' in avail:
        th = learned('twin')
        P += [('LUMEN-Ward', 'lumen', th, True, 1.0), ('LUMEN w/o certificates', 'lumen', th, False, 1.0),
              ('LUMEN w/o alarm term', 'lumen', th, True, 0.0)]
    for tag, name in (('rt', 'LUMEN (ray-traced training)'), ('phys', 'LUMEN (no-occlusion twin)')):
        if tag in avail:
            P.append((name, 'lumen', learned(tag), True, 1.0))
    return P


def evaluate(pols, split, sev_db, cfg_over, seeds=(0, 1, 2), nosev=(), n_eps=None):
    eps = data.pool(split, n_eps)[:n_eps]
    sev = data.severity(sev_db, 'test')
    cfg = dict(env.CFG, **cfg_over)
    runner = env.make_run(tuple(sorted(cfg.items())))
    one = lambda pol, th, cert, lam, X, RF, nm, km, w, key: runner(pol, th, cert, lam, X, RF, nm, km, w, pat, key)
    vm = jax.jit(jax.vmap(one, in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0)))
    X = jnp.asarray(np.stack([e[0] for e in eps]).astype(np.float32)); RF = jnp.asarray(np.stack([e[1] for e in eps]))
    nm = jnp.asarray(np.stack([e[2] for e in eps])); km = jnp.asarray(np.stack([e[3] for e in eps]))
    out = {}
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        W = np.stack([data.draw_w(sev, np.asarray(e[2]), rng) for e in eps])
        keys = jax.random.split(jax.random.PRNGKey(500 + s), len(eps))
        for name, pol, th, cert, lam in pols:
            w = jnp.ones_like(jnp.asarray(W)) if name in nosev else jnp.asarray(W)
            o = vm(env.POL[pol], th, cert, lam, X, RF, nm, km, w, keys)
            if name in nosev:            # score with the true severity weights
                o2 = vm(env.POL[pol], th, cert, lam, X, RF, nm, km, jnp.asarray(W), keys)
            out[(name, s)] = jax.tree.map(np.asarray, o)
    return out


if __name__ == '__main__':
    what = sys.argv[1]
    avail = [t for t in ('twin', 'rt', 'phys') if os.path.exists(f'es_{t}.pkl')]
    pols = policies(avail)
    if what == 'main':
        R = {}
        for split, db in (('test', 'mimic'), ('test', 'eicu'), ('big', 'mimic')):
            R[(split, db)] = evaluate(pols, split, db, {})
            print(split, db, flush=True)
        pickle.dump(R, open('results/eval_main.pkl', 'wb'))
    elif what == 'tier':
        sub = [p for p in pols if p[0] in ('MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward', 'LUMEN w/o alarm term')]
        R = {db: evaluate(sub, 'test', db, {}) for db in ('mimic', 'eicu')}
        pickle.dump(R, open('results/eval_tier.pkl', 'wb'))
    elif what == 'alpha':
        pol = [p for p in pols if p[0] == 'SW-Whittle+C+AP']
        R = {}
        for src in ('test', 'twin', 'phys', 'memoryless', 'uncoupled'):
            for a in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
                R[(src, a)] = evaluate(pol, src, 'mimic', {'alpha': a}, seeds=(0,), n_eps=32)
            print(src, flush=True)
        pickle.dump(R, open('results/eval_alpha.pkl', 'wb'))
    else:
        sub = [p for p in pols if p[0] in ('MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward')]
        grid = dict(D=[1, 5, 10, 20], lam_alarm=[0.02, 0.1, 0.5], p_tx=[0.0075, 0.010, 0.015], k_rf=[0, 1, 2], L_max=[10, 25, 50])
        R = {}
        default = None
        for k, vals in grid.items():
            for v in vals:
                if v == env.CFG[k]:
                    if default is None:
                        default = evaluate(sub, 'test', 'mimic', {}, seeds=(0,))
                    R[(k, v)] = default
                else:
                    R[(k, v)] = evaluate(sub, 'test', 'mimic', {k: v}, seeds=(0,))
                print(k, v, flush=True)
        pickle.dump(R, open('results/eval_sweep.pkl', 'wb'))
