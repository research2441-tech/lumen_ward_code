"""Physics-anchored generative channel twin.

Observable part: closed-form unobstructed LOS gain g0 of every node-receiver link (node pose from the
body-worn IMU, bed and luminaire geometry). Learned part: an autoregressive conditional mixture-density
network (MDN) generating the dB gain innovation of every link, which absorbs diffuse multipath and the
unobservable shadowing by staff, curtains and other patients. Links of one node share the uniform
variate used to pick the mixture component, which reproduces simultaneous shadowing of several
receivers by one occluder.
"""
import numpy as np, jax, jax.numpy as jnp, optax, pickle, glob

FLOOR = -90.0
NF = 15
KMIX = 4
EMA = np.exp(-1 / 100.0)                       # 1 s memory of the LOS deficit


def static_feats(e):
    """Per (t, n, l) observable features that do not depend on generated history."""
    g0 = np.maximum(e['G0'].astype(np.float32), FLOOR)
    vis = (e['G0'].astype(np.float32) > -119).astype(np.float32)
    dg0 = np.diff(g0, axis=0, prepend=g0[:1])
    d = e['lum'][None, None].astype(np.float32) - e['pos'][:, :, None]
    hd = np.linalg.norm(d[..., :2], axis=-1)
    cps = d[..., 2] / np.linalg.norm(d, axis=-1)
    T, N, K = g0.shape
    nt = np.zeros((T, N, K, 3), np.float32)
    nt[..., 0] = (e['ntype'] == 0)[None, :, None]; nt[..., 1] = (e['ntype'] == 1)[None, :, None]; nt[..., 2] = (e['ntype'] == 2)[None, :, None]
    walk = np.repeat(e['walk'], 3).astype(np.float32)[None, :, None] * np.ones((T, 1, K), np.float32)
    H = np.full((T, N, K), e['H'], np.float32)
    s = np.stack([(g0 + 60) / 15, vis, np.clip(dg0, -5, 5), hd / 4, cps, walk, H - 3], -1)
    return np.concatenate([s, nt], -1), g0          # 10 static features


def dyn_feats(x1, x2, ema, g0p, couple):
    """History features: last two gains, LOS deficit, its 1 s average and the node-level mean innovation."""
    return jnp.stack([(x1 + 60) / 15, jnp.clip(x1 - x2, -5, 5), jnp.clip(x1 - g0p, -40, 10) / 10, ema / 10, couple], -1)


def init_params(key, nin=NF, h=128, kmix=KMIX):
    k = jax.random.split(key, 3)
    f = lambda kk, a, b: jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=f(k[0], nin, h), b1=jnp.zeros(h), W2=f(k[1], h, h), b2=jnp.zeros(h),
                W3=f(k[2], h, 3 * kmix) * 0.1, b3=jnp.zeros(3 * kmix))


def mdn(p, x):
    h = jax.nn.gelu(x @ p['W1'] + p['b1'])
    h = jax.nn.gelu(h @ p['W2'] + p['b2'])
    o = h @ p['W3'] + p['b3']
    k = p['W3'].shape[1] // 3
    logit, mu, ls = o[..., :k], o[..., k:2 * k], o[..., 2 * k:]
    return jax.nn.log_softmax(logit), mu, jnp.clip(ls, -5, 3)


def nll(p, x, y):
    lw, mu, ls = mdn(p, x)
    lp = lw - 0.5 * ((y[..., None] - mu) / jnp.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)
    return -jax.nn.logsumexp(lp, -1)


def dataset(files, frac, seed, gauss=False):
    """Teacher-forced samples (features, innovation) from ray-traced episodes."""
    r = np.random.default_rng(seed)
    X, Y = [], []
    for f in files:
        e = pickle.load(open(f, 'rb'))
        sf, g0 = static_feats(e)
        x = e['G'].astype(np.float32)
        T = x.shape[0]
        def_ = np.clip(x - g0, -40, 10)
        ema = np.zeros_like(def_)
        for t in range(1, T):
            ema[t] = EMA * ema[t - 1] + (1 - EMA) * def_[t - 1]
        inn = np.diff(x, axis=0, prepend=x[:1])
        couple = np.repeat(np.clip(inn, -5, 5).mean(2, keepdims=True), x.shape[2], 2)
        ts = r.choice(np.arange(2, T), int(frac * T), replace=False)
        dyn = np.stack([(x[ts - 1] + 60) / 15, np.clip(x[ts - 1] - x[ts - 2], -5, 5), np.clip(x[ts - 1] - g0[ts - 1], -40, 10) / 10,
                        ema[ts] / 10, couple[ts - 1]], -1)
        X.append(np.concatenate([sf[ts], dyn], -1).reshape(-1, NF))
        Y.append((x[ts] - x[ts - 1]).reshape(-1))
    return np.concatenate(X), np.concatenate(Y)


def train(X, Y, Xv, Yv, seed=0, epochs=12, bs=4096, lr=2e-3, kmix=KMIX):
    p = init_params(jax.random.PRNGKey(seed), kmix=kmix)
    steps = epochs * (len(X) // bs)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-5)
    st = opt.init(p)

    @jax.jit
    def step(p, st, xb, yb):
        l, g = jax.value_and_grad(lambda q: nll(q, xb, yb).mean())(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    vl = jax.jit(lambda q, a, b: nll(q, a, b).mean())
    r = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = r.permutation(len(X))
        for i in range(len(X) // bs):
            b = idx[i * bs:(i + 1) * bs]
            p, st, l = step(p, st, jnp.asarray(X[b]), jnp.asarray(Y[b]))
        v = float(np.mean([vl(p, jnp.asarray(Xv[i:i + 200000]), jnp.asarray(Yv[i:i + 200000])) for i in range(0, len(Xv), 200000)]))
        hist.append(v)
        print('epoch', ep, 'val nll', v, flush=True)
    return p, hist


def rollout(p, sf, g0, x_init, key, mu_, sd_, coupled=True, gauss=False):
    """Free-running generation of a whole episode. sf: T x N x K x 10 static features; returns T x N x K dB gains."""
    T, N, K, _ = sf.shape
    sf = jnp.asarray(sf); g0 = jnp.asarray(g0)

    def body(c, inp):
        x1, x2, ema, couple, k = c
        s, g0p, g0t = inp
        k, k1, k2 = jax.random.split(k, 3)
        ema = EMA * ema + (1 - EMA) * jnp.clip(x1 - g0p, -40, 10)
        feat = jnp.concatenate([s, dyn_feats(x1, x2, ema, g0p, couple)], -1)
        lw, mu, ls = mdn(p, (feat - mu_) / sd_)
        if gauss:
            m = (jnp.exp(lw) * mu).sum(-1)
            v = (jnp.exp(lw) * (jnp.exp(2 * ls) + mu ** 2)).sum(-1) - m ** 2
            y = m + jnp.sqrt(v) * jax.random.normal(k2, m.shape)
        else:
            u = jax.random.uniform(k1, (N, 1)) if coupled else jax.random.uniform(k1, (N, K))
            cdf = jnp.cumsum(jnp.exp(lw), -1)
            comp = (u[..., None] > cdf).sum(-1)
            comp = jnp.minimum(comp, lw.shape[-1] - 1)
            mu_c = jnp.take_along_axis(mu, comp[..., None], -1)[..., 0]
            s_c = jnp.exp(jnp.take_along_axis(ls, comp[..., None], -1)[..., 0])
            y = mu_c + s_c * jax.random.normal(k2, mu_c.shape)
        x = x1 + y
        couple = jnp.repeat(jnp.clip(y, -5, 5).mean(1, keepdims=True), K, 1)
        return (x, x1, ema, couple, k), x

    g0prev = jnp.concatenate([g0[:1], g0[:-1]])
    x0 = jnp.asarray(x_init)
    c0 = (x0, x0, jnp.zeros_like(x0), jnp.zeros_like(x0), key)
    _, xs = jax.lax.scan(body, c0, (sf, g0prev, g0))
    return xs


rollout_jit = jax.jit(rollout, static_argnames=('coupled', 'gauss'))


def physics_only(g0, diffuse_db):
    """Deterministic baseline: unobstructed LOS plus a constant diffuse floor."""
    return 10 * np.log10(10 ** (g0 / 10) + 10 ** (diffuse_db / 10))
