"""Multi-patient hospital ward scenes and ray-traced uplink channels.

A ward has two rows of beds along the long walls, a central aisle, a ceiling grid of luminaires that
carry infrared uplink photodiodes, privacy curtains drawn during care, mobile staff and one ceiling
RF access point. Each patient wears three nodes (chest ECG patch, wrist pulse oximeter, upper-arm
cuff/thermometer). In-bed patients follow a procedural posture/arm-motion model; ambulating patients
follow CMU motion-capture walking clips. Channels: LOS with body/staff/curtain shadowing, first-order
diffuse reflections and an integrating-sphere tail (Paper-1 ray tracer), sampled every 10 ms slot.
"""
import numpy as np
import bvh, rooms, raytrace as rt, simulate as S

FS = 100.0                        # slot rate (10 ms slots)
NODE_TYPES = ('chest', 'wrist', 'arm')
P_TX_UL = 0.030                   # mean optical power of the body IR emitter (W)
RF = dict(ptx_dbm=-20.0, pl0=40.0, n=2.0, noise_dbm=-104.0, body_db=15.0, staff_db=6.0, sigma_db=3.0)


def layout(seed, big=False):
    r = np.random.default_rng(seed)
    nb = 5 if big else int(r.choice([3, 4]))
    pitch = r.uniform(2.4, 3.0)
    L = nb * pitch + 1.5
    W = r.uniform(6.5, 8.0) if not big else r.uniform(7.5, 8.5)
    H = r.uniform(2.7, 3.2)
    ny = 3 if W > 7.2 else 2
    nx = int(np.clip(round(L / 2.6), 3, 7))
    xs = L * (np.arange(nx) + 0.5) / nx
    ys = W * (np.arange(ny) + 0.5) / ny
    lum = np.array([[x, y, H - 0.02] for x in xs for y in ys])
    beds = []
    for row in (0, 1):
        for j in range(nb):
            xc = 0.75 + (j + 0.5) * pitch
            beds.append(dict(xc=xc, row=row, y_head=0.15 if row == 0 else W - 0.15, sgn=1 if row == 0 else -1))
    room = dict(name='ward', L=L, W=W, H=H, lum=lum, boxes=[], tall=[], beds=beds, pitch=pitch,
                walk=(0.4, 2.7, L - 0.4, W - 2.7), keepout=[],
                rho=dict(ceiling=r.uniform(0.70, 0.85), wall=r.uniform(0.50, 0.75), floor=r.uniform(0.20, 0.40)),
                rf_ap=np.array([L / 2, W / 2, H - 0.1]))
    return room


def curtain_boxes(room, b):
    """Three curtain panels (z 0.4-2.2 m) around bed b: two sides and the aisle-facing front."""
    bd = room['beds'][b]
    xc, s = bd['xc'], bd['sgn']
    y0 = 0.0 if s > 0 else room['W']
    yf = y0 + s * 2.95
    ya, yb = min(y0, yf), max(y0, yf)
    half = min(1.05, room['pitch'] / 2 - 0.05)
    t = 0.01
    return [(xc - half - t, ya, 0.4, xc - half + t, yb, 2.2),
            (xc + half - t, ya, 0.4, xc + half + t, yb, 2.2),
            (xc - half, yf - t, 0.4, xc + half, yf + t, 2.2)]


# ----------------------------------------------------------------------------------------------- patients
def _ou(r, T, tau, sd):
    a = np.exp(-1 / (tau * FS))
    x = np.zeros(T)
    x[0] = r.normal(0, sd)
    e = r.normal(0, sd * np.sqrt(1 - a * a), T)
    for t in range(1, T):
        x[t] = a * x[t - 1] + e[t]
    return x


def _rotv(v, k, ang):
    """Rodrigues rotation of vectors v (T x 3) about unit axes k (T x 3) by ang (T)."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    return v * c + np.cross(k, v) * s + k * (k * v).sum(1, keepdims=True) * (1 - c)


def in_bed(room, b, T, r):
    """Procedural in-bed patient: posture switching (supine, left/right lateral, semi-Fowler) and arm motion."""
    bd = room['beds'][b]
    s, xc = bd['sgn'], bd['xc']
    zm = 0.70 + r.uniform(-0.05, 0.05)                    # mattress top
    post = np.zeros(T, int)
    t, p = 0, int(r.choice(4, p=[0.45, 0.15, 0.15, 0.25]))
    while t < T:
        d = int(r.exponential(45.0) * FS) + 1
        post[t:t + d] = p
        t += d
        p = int(r.choice(4, p=[0.45, 0.15, 0.15, 0.25]))
    ax = np.array([0.0, s, 0.0])                          # head-to-foot axis
    hip = np.array([xc, bd['y_head'] + s * 1.05, zm + 0.12])
    up = np.array([0.0, 0.0, 1.0])
    side = np.array([1.0, 0.0, 0.0])
    sm = 0.2 * FS                                         # posture transitions take ~2 s (smoothing)
    ang_roll = np.where(post == 1, np.pi / 2, np.where(post == 2, -np.pi / 2, 0.0))
    ang_pitch = np.where(post == 3, np.deg2rad(45.0), 0.0)
    k = np.ones(int(sm)) / sm
    ang_roll = np.convolve(np.pad(ang_roll, (len(k), len(k)), mode='edge'), k, 'same')[len(k):-len(k)]
    ang_pitch = np.convolve(np.pad(ang_pitch, (len(k), len(k)), mode='edge'), k, 'same')[len(k):-len(k)]
    ang_roll = ang_roll + _ou(r, T, 8.0, np.deg2rad(6))
    # trunk direction (hip -> neck): towards the head, raised by the backrest angle
    back = -ax[None] * np.cos(ang_pitch)[:, None] + up[None] * np.sin(ang_pitch)[:, None]
    neck = hip[None] + 0.55 * back
    front0 = np.cross(np.broadcast_to(side, (T, 3)), back) * (-s)         # chest normal before rolling
    front0 /= np.linalg.norm(front0, axis=1, keepdims=True)
    front0 = np.where((front0[:, 2] < 0)[:, None], -front0, front0)
    front = _rotv(front0, back, ang_roll)
    lat = np.cross(back, front)
    lat /= np.linalg.norm(lat, axis=1, keepdims=True)
    chest_p = hip[None] + 0.38 * back + 0.17 * front
    # right arm: upper-arm cuff on the lateral surface; wrist moves (OU angles + occasional reaching)
    arm_ang = _ou(r, T, 6.0, np.deg2rad(25))
    reach = np.zeros(T)
    for _ in range(r.poisson(T / FS / 20.0)):
        t0 = r.integers(0, T)
        reach[t0:t0 + int(r.uniform(1, 4) * FS)] = np.deg2rad(r.uniform(40, 90))
    sh = neck - 0.05 * back + 0.20 * lat
    arm_n = _rotv(lat, back, arm_ang * 0.5)
    arm_p = sh - 0.15 * back + 0.06 * arm_n
    wr_n = _rotv(_rotv(front, back, arm_ang), lat, reach)
    wr_p = sh - 0.55 * back * np.cos(reach)[:, None] + 0.55 * front * np.sin(reach)[:, None] + 0.03 * wr_n + 0.05 * lat
    head = neck + 0.25 * back
    nodes = dict(chest=(chest_p, front), wrist=(wr_p, wr_n), arm=(arm_p, arm_n))
    occ = [(hip[None].repeat(T, 0), neck, 0.15), (neck, head, 0.11)]
    return nodes, occ, post


def walker(room, clip, T, r):
    c = bvh.load_clip(clip, FS)
    wr = dict(room); wr['keepout'] = []
    c = S.place(c, wr, r, min_len=2.0)
    if c is None:
        return None
    n = len(c['t'])
    t0 = int(r.integers(0, max(1, T - n)))
    idx = np.clip(np.arange(T) - t0, 0, n - 1)
    ps = S.node_poses(c)
    nodes = {}
    for k, src in (('chest', 'chest'), ('wrist', 'wrist'), ('arm', 'shoulder')):
        nodes[k] = (ps[src][0][idx], ps[src][1][idx])
    occ = [(c['Hips_p'][idx], c['Neck_p'][idx], 0.15), (c['Neck_p'][idx], c['Head_p'][idx] + np.array([0, 0, 0.12]), 0.11)]
    return nodes, occ, np.full(T, 4)


def staff_schedule(room, n, T, r, p_curtain=0.4):
    """Staff walk along the aisle to a bedside, stay for a care episode (curtain drawn with prob p), move on."""
    pos = np.zeros((n, T, 2))
    curt = np.zeros((len(room['beds']), T), bool)
    ymid = room['W'] / 2
    for s in range(n):
        p = np.array([r.uniform(1.0, room['L'] - 1.0), ymid + r.uniform(-0.3, 0.3)])
        t = 0
        while t < T:
            b = int(r.integers(len(room['beds'])))
            bd = room['beds'][b]
            stand = np.array([bd['xc'] + r.choice([-1, 1]) * 0.85, bd['y_head'] + bd['sgn'] * r.uniform(0.6, 1.6)])
            pts = [np.array([p[0], ymid]), np.array([stand[0], ymid]), stand]
            v = r.uniform(0.8, 1.3)
            for q in pts:
                while np.linalg.norm(q - p) > 1e-3 and t < T:
                    d = q - p
                    p = p + d / np.linalg.norm(d) * min(v / FS, np.linalg.norm(d))
                    pos[s, t] = p; t += 1
            dur = int(r.uniform(8, 40) * FS)
            if r.random() < p_curtain:
                curt[b, t:t + dur] = True
            pos[s, t:t + dur] = p
            t += dur
            back = np.array([p[0], ymid])
            while np.linalg.norm(back - p) > 1e-3 and t < T:
                d = back - p
                p = p + d / np.linalg.norm(d) * min(v / FS, np.linalg.norm(d))
                pos[s, t] = p; t += 1
            wait = int(r.uniform(0, 10) * FS)
            pos[s, t:t + wait] = p
            t += wait
    return pos[:, :T], curt[:, :T]


# ----------------------------------------------------------------------------------------------- channels
def los_free(room, r, nr):
    """Unobstructed LOS gain (node -> every ceiling receiver) and geometry; r, nr: M x 3."""
    ap = room['lum']
    g = rt.conc_gain(rt.Optics.n_ap, rt.Optics.fov_ap)
    m = rt.Optics.m_tx
    d = ap[None] - r[:, None]
    dist = np.linalg.norm(d, axis=2)
    cphi = np.clip((d * nr[:, None]).sum(2) / dist, 0, None)
    cpsi = np.clip(d[..., 2] / dist, 0, None)
    vis = cpsi >= np.cos(rt.Optics.fov_ap)
    G = (m + 1) * rt.Optics.A_ap / (2 * np.pi * dist ** 2) * cphi ** m * cpsi * g * vis
    return G, dist, cphi, cpsi


def episode(room, seed, clips, n_staff, p_walk=0.15, T=6000, chunk=100, geometry_only=False):
    r = np.random.default_rng(seed)
    pc = rooms.patches(room)
    pre = rt.precompute(room, pc)
    B = len(room['beds'])
    nodes, occ, post, walk = [], [], [], []
    for b in range(B):
        res = None
        if r.random() < p_walk:
            res = walker(room, clips[int(r.integers(len(clips)))], T, r)
        w = res is not None
        if res is None:
            res = in_bed(room, b, T, r)
        nd, oc, ps = res
        for k in NODE_TYPES:
            nodes.append(nd[k])
        occ += oc
        post.append(ps); walk.append(w)
    staff, curt = staff_schedule(room, n_staff, T, r)
    for s in range(n_staff):
        z0 = np.zeros((T, 3)); z0[:, :2] = staff[s]
        z1 = z0.copy(); z1[:, 2] = 1.75
        occ.append((z0, z1, 0.22))
    P = np.stack([n[0] for n in nodes], 1)                # T x N x 3
    Nr = np.stack([n[1] for n in nodes], 1)
    Nr = Nr / np.linalg.norm(Nr, axis=2, keepdims=True)
    ca = np.stack([o[0] for o in occ], 1); cb = np.stack([o[1] for o in occ], 1)
    cr = np.array([o[2] for o in occ])
    N, K = P.shape[1], len(room['lum'])
    G = np.zeros((T, N, K), np.float32); G0 = np.zeros((T, N, K), np.float32)
    eta = pre['rbar'] ** 2 / (1 - pre['rbar']) / pre['A_room'] * rt.Optics.A_ap * rt.Optics.n_ap ** 2
    m = rt.Optics.m_tx
    for t0 in range(0, T, chunk):
        sl = slice(t0, min(T, t0 + chunk))
        n_t = sl.stop - sl.start
        r_ = P[sl].reshape(-1, 3); n_ = Nr[sl].reshape(-1, 3)
        Gl, _, _, _ = los_free(room, r_, n_)
        if geometry_only:
            G0[sl] = Gl.reshape(n_t, N, K)
            continue
        # occluders: each node is not blocked by its own patient's torso it sits on (checked via offset)
        ca_ = np.repeat(ca[sl], N, 0); cb_ = np.repeat(cb[sl], N, 0)
        act = np.unique(np.nonzero(curt[:, sl])[0])
        boxes = sum([curtain_boxes(room, b) for b in act], [])
        blk = rt.blocked(r_, room['lum'], ca_, cb_, cr, boxes)
        dp = pre['c'][None] - r_[:, None]
        dpn = np.linalg.norm(dp, axis=2)
        cph = np.clip((dp * n_[:, None]).sum(2) / dpn, 0, None)
        cth = np.clip(-(dp * pre['n'][None]).sum(2) / dpn, 0, None)
        Pj = (m + 1) / (2 * np.pi * dpn ** 2) * cph ** m * cth * pre['area'][None]
        diff = Pj @ pre['S'] + eta
        G[sl] = (Gl * ~blk + diff).reshape(n_t, N, K)
        G0[sl] = Gl.reshape(n_t, N, K)
    # RF: log-distance path loss, own-body shadowing when the node faces away, staff shadowing, slow log-normal
    ap = room['rf_ap']
    d = ap[None, None] - P
    dist = np.linalg.norm(d, axis=2)
    facing = (d * Nr).sum(2) / dist
    sh = RF['body_db'] * np.clip(-facing, 0, 1)
    pa = P[:, :, None, :2]; st = staff.transpose(1, 0, 2)[:, None]
    seg = ap[None, None, None, :2] - pa
    tt = np.clip(((st - pa) * seg).sum(-1) / (seg ** 2).sum(-1), 0, 1)
    dd = np.linalg.norm(pa + tt[..., None] * seg - st, axis=-1)
    sh = sh + RF['staff_db'] * (dd < 0.3).sum(-1)
    ln = np.stack([_ou(r, T, 2.0, RF['sigma_db']) for _ in range(N)], 1)
    snr_rf = RF['ptx_dbm'] - RF['pl0'] - 10 * RF['n'] * np.log10(np.maximum(dist, 0.3)) - sh - ln - RF['noise_dbm']
    patient = np.repeat(np.arange(B), len(NODE_TYPES))
    ntype = np.tile(np.arange(len(NODE_TYPES)), B)
    return dict(G=G, G0=G0, pos=P.astype(np.float32), nrm=Nr.astype(np.float32), snr_rf=snr_rf.astype(np.float32),
                patient=patient, ntype=ntype, walk=np.array(walk), post=np.array(post, np.int8),
                staff=staff.astype(np.float32), curtain=curt, lum=room['lum'], L=room['L'], W=room['W'], H=room['H'])
