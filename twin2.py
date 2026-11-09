"""Physics-structured generative channel twin (used by LUMEN-Ward).

x_{nl,t} = 10 log10( B_{nl,t} * 10^{g0_{nl,t}/10} + 10^{d_{nl,t}/10} )
  g0 : closed-form unobstructed LOS gain from the observable geometry (node pose, bed, luminaire)
  B  : LOS occlusion state (1 clear, 0 shadowed by staff, curtains or bodies), a learned
       state-dependent Markov process whose transition probability is an MLP of geometry, current state,
       run length and the occluded fraction of the node's other links; links of one node share the
       uniform variate of each slot, so one occluder can shadow several receivers at once.
  d  : diffuse level, MLP mean and spread from geometry, with a 1 s AR(1) fluctuation.
"""
import numpy as np, jax, jax.numpy as jnp, optax, pickle

RHO = np.exp(-1 / 100.0)


def geo(e):
    T, N, K = e['G0'].shape
    g0 = e['G0'].astype(np.float32)
    vis = g0 > -119
    d = e['lum'][None, None].astype(np.float32) - e['pos'][:, :, None]
    hd = np.linalg.norm(d[..., :2], axis=-1)
    cps = d[..., 2] / np.linalg.norm(d, axis=-1)
    nz = np.broadcast_to(e['nrm'][..., 2][..., None], (T, N, K))
    z = np.broadcast_to(e['pos'][..., 2][..., None], (T, N, K))
    nt = np.zeros((T, N, K, 3), np.float32)
    for i in range(3):
        nt[..., i] = (e['ntype'] == i)[None, :, None]
    walk = np.broadcast_to(np.repeat(e['walk'], 3).astype(np.float32)[None, :, None], (T, N, K))
    room = np.broadcast_to(np.array([e['H'] - 3, e['L'] / 10, e['W'] / 10], np.float32), (T, N, K, 3))
    dg0 = np.clip(np.diff(np.maximum(g0, -90), axis=0, prepend=np.maximum(g0[:1], -90)), -5, 5)
    fd = np.concatenate([np.stack([hd / 4, cps, nz, z, walk], -1), room, nt], -1)            # 11 diffuse feats
    fo = np.concatenate([np.stack([(np.maximum(g0, -90) + 60) / 15, hd / 4, cps, nz, z, walk, dg0], -1), room[..., :1], nt], -1)  # 11
    return g0, vis, fd.astype(np.float32), fo.astype(np.float32)


def occl_state(x, g0, vis):
    blk = vis & (x < g0 - 3)
    T = blk.shape[0]
    run = np.zeros(blk.shape, np.float32)
    for t in range(1, T):
        run[t] = np.where(blk[t] == blk[t - 1], run[t - 1] + 1, 0)
    frac = (blk.sum(2, keepdims=True) - blk) / np.maximum(vis.sum(2, keepdims=True) - 1, 1)
    return blk, run, frac


def datasets(files, frac, seed):
    r = np.random.default_rng(seed)
    Fd, Yd, Fo, Yo = [], [], [], []
    for f in files:
        e = pickle.load(open(f, 'rb'))
        x = e['G'].astype(np.float32)
        g0, vis, fd, fo = geo(e)
        blk, run, fr = occl_state(x, g0, vis)
        T = x.shape[0]
        ts = r.choice(np.arange(1, T), int(frac * T), replace=False)
        m = (~vis | blk)[ts]
        Fd.append(fd[ts][m]); Yd.append(x[ts][m])
        mv = (vis[ts] & vis[ts - 1])
        st = np.stack([blk[ts - 1].astype(np.float32), np.log1p(run[ts - 1]) / 5, fr[ts - 1]], -1)
        Fo.append(np.concatenate([fo[ts], st], -1)[mv]); Yo.append(blk[ts][mv].astype(np.float32))
    return [np.concatenate(a) for a in (Fd, Yd, Fo, Yo)]


def init(key, nin, nout, h):
    k = jax.random.split(key, 3)
    f = lambda kk, a, b: jax.random.normal(kk, (a, b)) * np.sqrt(2.0 / a)
    return dict(W1=f(k[0], nin, h), b1=jnp.zeros(h), W2=f(k[1], h, h), b2=jnp.zeros(h), W3=f(k[2], h, nout) * 0.1, b3=jnp.zeros(nout))


def net(p, x):
    h = jax.nn.gelu(x @ p['W1'] + p['b1'])
    h = jax.nn.gelu(h @ p['W2'] + p['b2'])
    return h @ p['W3'] + p['b3']


def loss_d(p, x, y):
    o = net(p, x)
    m, ls = o[..., 0] - 58.0, jnp.clip(o[..., 1], -4, 3)
    return 0.5 * ((y - m) / jnp.exp(ls)) ** 2 + ls


def loss_o(p, x, y):
    z = net(p, x)[..., 0]
    return optax.sigmoid_binary_cross_entropy(z, y)


def fit(lossf, nin, nout, X, Y, Xv, Yv, epochs=8, bs=4096, lr=2e-3, h=64, seed=0):
    p = init(jax.random.PRNGKey(seed), nin, nout, h)
    steps = epochs * max(1, len(X) // bs)
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-5)
    st = opt.init(p)

    @jax.jit
    def step(p, st, xb, yb):
        l, g = jax.value_and_grad(lambda q: lossf(q, xb, yb).mean())(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    vl = jax.jit(lambda q, a, b: lossf(q, a, b).mean())
    r = np.random.default_rng(seed)
    hist = []
    best = (np.inf, p)
    for ep in range(epochs):
        idx = r.permutation(len(X))
        for i in range(len(X) // bs):
            b = idx[i * bs:(i + 1) * bs]
            p, st, _ = step(p, st, jnp.asarray(X[b]), jnp.asarray(Y[b]))
        v = float(np.mean([vl(p, jnp.asarray(Xv[i:i + 200000]), jnp.asarray(Yv[i:i + 200000])) for i in range(0, len(Xv), 200000)]))
        hist.append(v)
        if v < best[0]:
            best = (v, p)
        print('epoch', ep, v, flush=True)
    return jax.tree.map(np.asarray, best[1]), hist


def _roll(P, fd, fo, g0, vis, key, coupled, memoryless, occlusion):
    T, N, K = g0.shape
    od = net(P['d'], fd)
    dm, ds = od[..., 0] - 58.0, jnp.exp(jnp.clip(od[..., 1], -4, 3))

    def body(c, inp):
        B, R, eps, k = c
        fo_t, g0_t, vis_t, vprev, dm_t, ds_t = inp
        k, k1, k2 = jax.random.split(k, 3)
        fr = ((B * vprev).sum(1, keepdims=True) - B * vprev) / jnp.maximum(vprev.sum(1, keepdims=True) - 1, 1)
        st = jnp.stack([B, jnp.log1p(R) / 5, fr], -1)
        if memoryless:
            st = jnp.stack([jnp.zeros_like(B), jnp.zeros_like(R), jnp.zeros_like(fr)], -1)
        pb = jax.nn.sigmoid(net(P['o'], jnp.concatenate([fo_t, st], -1))[..., 0])
        u = jax.random.uniform(k1, (N, 1)) if coupled else jax.random.uniform(k1, (N, K))
        Bn = jnp.where(vis_t & vprev.astype(bool), (u < pb).astype(jnp.float32), 0.0) * occlusion
        R = jnp.where(Bn == B, R + 1, 0.0)
        eps = RHO * eps + np.sqrt(1 - RHO ** 2) * jax.random.normal(k2, (N, K))
        d = dm_t + ds_t * eps
        x = 10 * jnp.log10(jnp.where(vis_t, (1 - Bn) * 10 ** (g0_t / 10), 0.0) + 10 ** (d / 10))
        return (Bn, R, eps, k), x

    vprev = jnp.concatenate([vis[:1], vis[:-1]]).astype(jnp.float32)
    c0 = (jnp.zeros((N, K)), jnp.zeros((N, K)), jax.random.normal(key, (N, K)), key)
    _, xs = jax.lax.scan(body, c0, (fo, g0, vis, vprev, dm, ds))
    return xs


roll = jax.jit(_roll, static_argnames=('coupled', 'memoryless', 'occlusion'))


def generate(P, e, key, variant='twin'):
    g0, vis, fd, fo = geo(e)
    Pj = jax.tree.map(jnp.asarray, P)
    xs = roll(Pj, jnp.asarray(fd), jnp.asarray(fo), jnp.asarray(g0), jnp.asarray(vis), key,
              coupled=(variant != 'uncoupled'), memoryless=(variant == 'memoryless'), occlusion=(0.0 if variant == 'phys' else 1.0))
    return np.asarray(xs)
