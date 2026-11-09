"""Lagrangian evolution-strategy training of the shared LUMEN-Ward scorer.

Fitness of a candidate = mean over training windows of SW-AoI(candidate)/SW-AoI(reference index policy)
+ lambda * (alarm-violation rate - delta). The multiplier lambda follows dual ascent on the centre policy."""
import sys, glob, pickle, time, numpy as np, jax, jax.numpy as jnp, optax, jax.flatten_util
import env, data

src, tag = sys.argv[1], sys.argv[2]
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 80
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
use_alarm = 0 if 'noalarm' in tag else 1
use_sev = 'nosev' not in tag
use_cert = 'nocert' not in tag
M, E, TW, SIG, DELTA = 24, 8, 1500, 0.10, 0.01
pool = data.pool(src)
val = data.pool('val')
sev_tr = data.severity('mimic', 'train')
r = np.random.default_rng(seed)
theta = env.init_theta(jax.random.PRNGKey(seed))
flat, unravel = jax.flatten_util.ravel_pytree(theta)
opt = optax.adam(0.03)
st = opt.init(flat)
run = env.run


def batch(n, T, rng, source):
    Xs, RFs, nms, kms, ws = [], [], [], [], []
    for _ in range(n):
        ep = source[rng.integers(len(source))]
        X, RF, nm, km = ep
        t0 = rng.integers(0, X.shape[0] - T)
        Xs.append(X[t0:t0 + T]); RFs.append(RF[t0:t0 + T]); nms.append(nm); kms.append(km)
        w = data.draw_w(sev_tr, nm, rng) if use_sev else np.where(nm, 1.0, 1.0).astype(np.float32)
        ws.append(w)
    return [jnp.asarray(np.stack(a).astype(np.float32)) for a in (Xs, RFs)] + [jnp.asarray(np.stack(a)) for a in (nms, kms, ws)]


pat = jnp.asarray(np.repeat(np.arange(env.N_MAX // 3), 3))


def one(th, pol, X, RF, nm, km, w, key, lam):
    return run(pol, th, use_cert, lam, X, RF, nm, km, w, pat, key)


ev = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None)), in_axes=(0, None, None, None, None, None, None, None, None)))
ref = jax.jit(jax.vmap(one, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None)))
lam_dual = 0.0
hist = []
best = (np.inf, flat)
t0 = time.time()
for it in range(iters):
    X, RF, nm, km, w = batch(E, TW, r, pool)
    keys = jax.random.split(jax.random.PRNGKey(seed * 1000 + it), E)
    o_ref = ref(theta, env.POL['wsw_ap'], X, RF, nm, km, w, keys, 1.0)
    eps = r.standard_normal((M // 2, flat.size)).astype(np.float32)
    eps = np.concatenate([eps, -eps])
    cand = jnp.asarray(flat[None] + SIG * eps)
    ths = jax.vmap(unravel)(cand)
    o = ev(ths, env.POL['lumen'], X, RF, nm, km, w, keys, float(use_alarm))
    rel = np.asarray(o['swaoi'] / o_ref['swaoi'][None]).mean(1)
    vr = np.asarray(o['viol']).sum(1) / np.maximum(np.asarray(o['n_al']).sum(1), 1)
    fit = rel + lam_dual * (vr - DELTA)
    rank = np.argsort(np.argsort(fit)) / (M - 1) - 0.5                 # centred ranks (lower fitness = better)
    g = (rank[:, None] * eps).mean(0) / SIG
    u, st = opt.update(jnp.asarray(g), st, flat)
    flat = optax.apply_updates(flat, u)
    theta = unravel(flat)
    lam_dual = max(0.0, lam_dual + 5.0 * (vr.mean() - DELTA)) if use_alarm else 0.0
    rec = dict(it=it, rel=float(rel.mean()), viol=float(vr.mean()), lam=lam_dual, t=time.time() - t0)
    if it % 10 == 9 or it == iters - 1:
        vX, vRF, vnm, vkm, vw = batch(16, 3000, np.random.default_rng(777), val)
        vk = jax.random.split(jax.random.PRNGKey(4242), 16)
        ov = ref(theta, env.POL['lumen'], vX, vRF, vnm, vkm, vw, vk, float(use_alarm))
        orf = ref(theta, env.POL['wsw_ap'], vX, vRF, vnm, vkm, vw, vk, 1.0)
        vrel = float(np.mean(np.asarray(ov['swaoi'] / orf['swaoi'])))
        vv = float(np.asarray(ov['viol']).sum() / max(np.asarray(ov['n_al']).sum(), 1))
        score = vrel + 10.0 * max(0.0, vv - DELTA)
        rec.update(val_rel=vrel, val_viol=vv)
        if score < best[0]:
            best = (score, np.asarray(flat))
    hist.append(rec)
    print(rec, flush=True)
pickle.dump(dict(theta=jax.tree.map(np.asarray, unravel(jnp.asarray(best[1]))), hist=hist, src=src, tag=tag, best=best[0]),
            open(f'es_{tag}.pkl', 'wb'))
