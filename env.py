"""Ward uplink scheduler simulator (JAX): hybrid optical-RF polling, certificates, severity-weighted AoI,
alarm latency. One slot = 10 ms; every ceiling receiver polls at most one node per slot; K_RF shared RF slots."""
import numpy as np, jax, jax.numpy as jnp
import phy

SLOT = 0.01
N_MAX, K_MAX = 30, 18
CFG = dict(p_tx=0.010, bw=5e6, gamma_db=10.35, gamma_rf_db=10.0, k_rf=1, D=5, alpha=0.02, eta=0.05,
           L_max=25, lam_alarm=0.1, amax=400)


def phy_consts(bw):
    lk = phy.Link('ul')
    lk.bw = bw
    A = (lk.R * lk.K * lk.sig) ** 2
    B = (lk.R * lk.sig) ** 2 * lk.clip_var
    N0 = float(lk.noise_var(np.array(1e-6)))
    return A, B, N0


A_SIG, B_CLIP, N0 = phy_consts(CFG['bw'])
NPOL = 7          # rr, mw, to, dget, swmw, swmw_ap, learned
POL = dict(rr=0, mw=1, to=2, dget=3, wsw=4, wsw_ap=5, lumen=6)

# ------------------------------------------------------------------------------------- learned scorer
NFE, NFR = 11, 7


def init_theta(key, h=16):
    k = jax.random.split(key, 4)
    f = lambda kk, a, b: jax.random.normal(kk, (a, b)) * 0.1 / np.sqrt(a)
    return dict(W1=f(k[0], NFE, h), b1=jnp.zeros(h), w2=f(k[1], h, 1)[:, 0] * 0, b2=jnp.zeros(()),
                R1=f(k[2], NFR, h), c1=jnp.zeros(h), r2=f(k[3], h, 1)[:, 0] * 0, c2=jnp.zeros(()),
                la=jnp.zeros(()))


def mlp(W, b, w, c, x):
    return jnp.tanh(x @ W + b) @ w + c


def lin(db):
    return 10 ** (db / 10)


def step_fn(pol, theta, use_cert, lam, cfg):
    gam = lin(cfg['gamma_db']); gam_rf = lin(cfg['gamma_rf_db'])
    L = cfg['L_max']

    def step(s, inp):
        X, Xd, Xd10, RF, u_rf, u_al, u_aln, t = inp
        a, aw, q, last, lastsch, nm, km, w, pat, Np, al_rate = s['a'], s['aw'], s['q'], s['last'], s['lastsch'], s['nm'], s['km'], s['w'], s['pat'], s['Np'], s['al_rate']
        N, K = X.shape
        # ---------------- alarms: Bernoulli per patient, assigned to one of its nodes
        newal = (u_al < al_rate * SLOT)                        # per patient (P,)
        tgt = (u_aln * 3).astype(jnp.int32)                    # node within patient
        node_al = jnp.zeros(N, bool).at[jnp.arange(newal.shape[0]) * 3 + tgt].set(newal, mode='drop') & nm
        aw = jnp.where(node_al & (aw < 0), 0, aw)
        alarm = aw >= 0
        # ---------------- channel knowledge
        xh = Xd                                                 # delayed measurement (D slots)
        cert = jnp.where(use_cert, xh - q, xh)
        Sc = A_SIG * lin(2 * cert) ; Sh = A_SIG * lin(2 * xh)
        Sg = A_SIG * lin(2 * X)                                 # genie (current)
        Sd = Sc
        Ih_src = Sh
        alone = Sd / (N0 + B_CLIP * Sd) >= gam
        alone = alone & nm[:, None] & km[None, :]
        n_alt = alone.sum(1)
        marg = 10 * jnp.log10(Sd / (N0 + B_CLIP * Sd) / gam + 1e-12)
        p_rf = jnp.exp(-gam_rf / lin(RF)) * nm
        wa = w * a
        # ---------------- priorities
        trend = jnp.clip(Xd - Xd10, -10, 10)
        fe = jnp.stack([jnp.log(jnp.broadcast_to(w[:, None], (N, K))), jnp.log(jnp.broadcast_to(a[:, None], (N, K))) / 3,
                        jnp.broadcast_to(alarm[:, None], (N, K)).astype(jnp.float32),
                        jnp.broadcast_to((jnp.maximum(aw, 0) / L)[:, None], (N, K)),
                        jnp.clip(marg, -10, 30) / 10, trend / 5, jnp.broadcast_to((n_alt / 4.0)[:, None], (N, K)),
                        jnp.broadcast_to((10 * jnp.log10(p_rf + 1e-6) / 10)[:, None], (N, K)),
                        jnp.broadcast_to((alone.sum(0) / 6.0)[None, :], (N, K)),
                        jnp.clip(marg - jnp.where(alone, marg, -10.0).max(1, keepdims=True), -20, 0) / 10,
                        jnp.clip(10 * jnp.log10(((Ih_src * km[None, :]).sum(1, keepdims=True) - Ih_src) / N0 + 1e-3), -10, 50) / 20], -1)
        af = a.astype(jnp.float32)
        pr = jnp.maximum(p_rf, 1e-3)
        whit = jnp.log(w) + jnp.log(af) + jnp.log(af + 1.0)                    # Whittle index, p = 1 (certified link)
        base = jnp.select([pol == POL['rr'], pol == POL['mw'], pol == POL['to'], pol == POL['dget']],
                          [jnp.log(lastsch.astype(jnp.float32) + 1), jnp.log(af) + jnp.log(af + 1.0), jnp.zeros(N), jnp.log(1.0 + af)], whit)
        base = base + jnp.where(pol == POL['wsw_ap'], 30.0 * alarm, 0.0)
        whit_rf = jnp.log(w) + jnp.log(pr) + jnp.log(af) + jnp.log(af + 2.0 / pr - 1.0)
        base_rf = jnp.select([pol == POL['rr'], pol == POL['mw'], pol == POL['to'], pol == POL['dget']],
                             [jnp.log(lastsch.astype(jnp.float32) + 1) + jnp.log(pr), jnp.log(pr) + jnp.log(af) + jnp.log(af + 2.0 / pr - 1.0),
                              jnp.log(pr), jnp.log(1.0 + af) + jnp.log(pr)], whit_rf)
        base_rf = base_rf + jnp.where(pol == POL['wsw_ap'], 30.0 * alarm, 0.0)
        res = mlp(theta['W1'], theta['b1'], theta['w2'], theta['b2'], fe)
        sc = base[:, None] + 0.05 * jnp.clip(marg, -10, 30) + jnp.where(pol == POL['to'], marg, 0.0)
        sc = sc + jnp.where(pol == POL['lumen'], res + lam * (alarm * (1.0 + jnp.maximum(aw, 0) / L))[:, None], 0.0)
        sc = jnp.where(alone, sc, -jnp.inf)

        # ---------------- greedy matching with interference feasibility
        def rnd(c, _):
            asg_n, asg_k, I, slack = c                          # asg_n: (N,) receiver or -1 ; asg_k: (K,) node or -1
            free = (asg_n[:, None] < 0) & (asg_k[None, :] < 0)
            Sown = Sd
            ok_new = Sown / (N0 + B_CLIP * Sown + I[None, :]) >= gam
            occ = asg_k >= 0
            add = Ih_src                                           # interference the candidate adds at each receiver
            ok_old = jnp.all(jnp.where(occ[None, :], add <= slack[None, :], True), axis=1)
            m = free & ok_new & ok_old[:, None]
            v = jnp.where(m, sc, -jnp.inf)
            idx = jnp.argmax(v.reshape(-1))
            n, k = idx // K, idx % K
            go = jnp.isfinite(v.reshape(-1)[idx])
            asg_n = jnp.where(go, asg_n.at[n].set(k), asg_n)
            asg_k = jnp.where(go, asg_k.at[k].set(n), asg_k)
            I = jnp.where(go, I + Ih_src[n] * (jnp.arange(K) != k), I)
            sig_k = Sd[asg_k.clip(0), jnp.arange(K)]
            slack = jnp.where(asg_k >= 0, sig_k / gam - N0 - B_CLIP * sig_k - I, jnp.inf)
            return (asg_n, asg_k, I, slack), None

        c0 = (-jnp.ones(N, jnp.int32), -jnp.ones(K, jnp.int32), jnp.zeros(K), jnp.full(K, jnp.inf))
        (asg_n, asg_k, I, _), _ = jax.lax.scan(rnd, c0, None, length=K)
        # ---------------- RF fallback
        fr = jnp.stack([jnp.log(w), jnp.log(a.astype(jnp.float32)) / 3, alarm.astype(jnp.float32), jnp.maximum(aw, 0) / L,
                        10 * jnp.log10(p_rf + 1e-6) / 10, (n_alt / 4.0), jnp.log(w * a) / 3], -1)
        rres = mlp(theta['R1'], theta['c1'], theta['r2'], theta['c2'], fr)
        rsc = base_rf + jnp.where(pol == POL['lumen'], rres + lam * alarm * (1.0 + jnp.maximum(aw, 0) / L), 0.0)
        rsc = jnp.where((asg_n < 0) & nm & (p_rf > 0.3), rsc, -jnp.inf)
        top = jnp.argsort(-rsc)[:cfg['k_rf']]
        rf_sel = jnp.zeros(N, bool).at[top].set(jnp.isfinite(rsc[top]))
        # ---------------- outcomes (true channel, actual interference)
        tx = asg_n >= 0
        Str = A_SIG * lin(2 * X)
        Itot = (Str * tx[:, None]).sum(0)
        k_of = asg_n.clip(0)
        sig = Str[jnp.arange(N), k_of]
        sinr = sig / (N0 + B_CLIP * sig + Itot[k_of] - sig)
        ok_opt = tx & (sinr >= gam)
        ok_rf = rf_sel & (u_rf < p_rf)
        ok = ok_opt | ok_rf
        ho = ok_opt & (last >= 0) & (last != k_of)
        last = jnp.where(ok_opt, k_of, last)
        lat = jnp.where(ok & alarm, aw + 1, -1)
        aw = jnp.where(ok, -1, jnp.where(alarm, aw + 1, aw))
        a_new = jnp.where(ok, 1, jnp.minimum(a + 1, cfg['amax']))
        lastsch = jnp.where(tx | rf_sel, 0, lastsch + 1)
        # ---------------- certificate tracker update (beacon measurement of every link)
        v = (X < xh - q).astype(jnp.float32)
        q = jnp.clip(q + cfg['eta'] * (v - cfg['alpha']), 0.0, 40.0)
        s2 = dict(s, a=a_new, aw=aw, q=q, last=last, lastsch=lastsch)
        lv = jnp.array([1 / 12, 1 / 5, 1.0, 4.0])
        lvm = (jnp.abs(w[None, :] - lv[:, None]) < 1e-4) & nm[None, :]
        out = dict(a_lvl=(a[None, :] * lvm).sum(1), n_lvl=lvm.sum(1), wa=(w * a * nm).sum(), a=(a * nm).sum(), a_hi=(a * nm * (w >= 1.0)).sum(),
                   ahist=jnp.zeros(cfg['amax'] + 1).at[a].add(w * nm),
                   lat=lat, n_ok_opt=ok_opt.sum(), n_tx=tx.sum(), n_rf=rf_sel.sum(), n_ok_rf=ok_rf.sum(), ho=ho.sum(),
                   viol_tx=(tx & ~ok_opt).sum())
        return s2, out
    return step


def run(pol, theta, use_cert, lam, X, RF, nm, km, w, pat, key, cfg=CFG, al_rate=None):
    """X: T x N x K (dB), RF: T x N (dB). Returns aggregate metrics for one episode."""
    T, N, K = X.shape
    D = cfg['D']
    X = X + 10 * np.log10(cfg['p_tx'] / 0.030)          # gains were traced for a 30 mW reference emitter
    Xd = jnp.concatenate([jnp.repeat(X[:1], D, 0), X[:-D]])
    Xd10 = jnp.concatenate([jnp.repeat(X[:1], D + 10, 0), X[:-(D + 10)]])
    k1, k2, k3 = jax.random.split(key, 3)
    P = N // 3
    u_rf = jax.random.uniform(k1, (T, N)); u_al = jax.random.uniform(k2, (T, P)); u_aln = jax.random.uniform(k3, (T, P))
    s0 = dict(a=jnp.ones(N, jnp.int32), aw=-jnp.ones(N, jnp.int32), q=jnp.full((N, K), 0.5), last=-jnp.ones(N, jnp.int32),
              lastsch=jnp.zeros(N, jnp.int32), nm=nm, km=km, w=w, pat=pat, Np=nm.sum(),
              al_rate=jnp.where(jnp.arange(P) * 3 < nm.sum(), cfg['lam_alarm'] if al_rate is None else al_rate, 0.0))
    st = step_fn(pol, theta, use_cert, lam, cfg)
    _, o = jax.lax.scan(st, s0, (X, Xd, Xd10, RF, u_rf, u_al, u_aln, jnp.arange(T)))
    lat = o['lat']
    n_al = (lat > 0).sum()
    lh = jnp.zeros(cfg['amax'] + 1).at[jnp.clip(lat, 0, cfg['amax'])].add((lat > 0).astype(jnp.float32))
    wsum = (w * nm).sum()
    return dict(swaoi=o['wa'].sum() / T / wsum, aoi=o['a'].sum() / T / nm.sum(),
                aoi_hi=o['a_hi'].sum() / T / jnp.maximum((nm * (w >= 1.0)).sum(), 1),
                ahist=o['ahist'].sum(0), lhist=lh, n_al=n_al, viol=(lat > cfg['L_max']).sum(),
                tx=o['n_tx'].sum(), ok_opt=o['n_ok_opt'].sum(), rf=o['n_rf'].sum(), ok_rf=o['n_ok_rf'].sum(),
                ho=o['ho'].sum(), fail_tx=o['viol_tx'].sum(), a_lvl=o['a_lvl'].sum(0), n_lvl=o['n_lvl'].sum(0),
                lat_lvl=jnp.stack([((lat > cfg['L_max']) & (jnp.abs(w - v) < 1e-4)[None, :]).sum() for v in (1 / 12, 1 / 5, 1.0, 4.0)]),
                al_lvl=jnp.stack([((lat > 0) & (jnp.abs(w - v) < 1e-4)[None, :]).sum() for v in (1 / 12, 1 / 5, 1.0, 4.0)]))


from functools import partial, lru_cache


@lru_cache(None)
def make_run(items=None):
    cfg = dict(CFG) if items is None else dict(items)
    return jax.jit(partial(run, cfg=cfg))


def load_episode(e, n_max=N_MAX, k_max=K_MAX, rx_sub=True):
    T, N, K = e['G'].shape
    X = np.full((T, n_max, k_max), -120.0, np.float16)
    X[:, :N, :K] = e['G']
    RF = np.full((T, n_max), -50.0, np.float32)
    RF[:, :N] = e['snr_rf']
    nm = np.zeros(n_max, bool); nm[:N] = True
    km = np.zeros(k_max, bool)
    lum = e['lum']
    ix = np.searchsorted(np.unique(np.round(lum[:, 0], 3)), np.round(lum[:, 0], 3))
    iy = np.searchsorted(np.unique(np.round(lum[:, 1], 3)), np.round(lum[:, 1], 3))
    km[:K] = (ix + iy) % 2 == 0 if rx_sub else True        # IR receivers in a checkerboard subset of luminaires
    return X, RF, nm, km
