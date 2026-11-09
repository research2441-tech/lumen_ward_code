"""Twin data efficiency: fit on 4, 8, 16, 32 training wards (2 episodes each); evaluate occlusion statistics on test."""
import glob, pickle, json, numpy as np, jax
import twin2, eval_twin as ET

tr = sorted(glob.glob('corpus/train_*.pkl')); va = sorted(glob.glob('corpus/val_*.pkl'))
tests = sorted(glob.glob('corpus/test_*.pkl'))
Fdv, Ydv, Fov, Yov = twin2.datasets(va, 0.03, 1)
out = {}
for nw in (4, 8, 16, 32):
    files = [f for f in tr if int(f.split('_')[1]) < nw]
    Fd, Yd, Fo, Yo = twin2.datasets(files, 0.04, 0)
    pd_, _ = twin2.fit(twin2.loss_d, Fd.shape[1], 2, Fd, Yd, Fdv, Ydv, epochs=6, h=32, lr=1e-3)
    po, ho = twin2.fit(twin2.loss_o, Fo.shape[1], 1, Fo, Yo, Fov, Yov, epochs=6)
    P = dict(d=pd_, o=po)
    S = []
    for j, f in enumerate(tests):
        e = pickle.load(open(f, 'rb'))
        S.append(ET.stats(twin2.generate(P, e, jax.random.PRNGKey(900 + j)), e['G0']))
    out[nw] = dict(val_bce=float(min(ho)), **{k: float(np.mean([s[k] for s in S])) for k in ('blk_frac', 'onset_per_min', 'dur_ms', 'co_blk')})
    print(nw, out[nw], flush=True)
json.dump(out, open('results/twin_scaling.json', 'w'), indent=1)
