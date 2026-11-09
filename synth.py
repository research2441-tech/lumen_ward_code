"""Geometry-only synthetic ward episodes (no ray tracing) for twin-based scheduler training."""
import os, sys, numpy as np, pickle
from multiprocessing import Pool
import ward
from gen_corpus import BVH, TRAIN_CLIPS

OUT = 'synth'


def job(i):
    fn = f'{OUT}/syn_{i:03d}.pkl'
    if os.path.exists(fn):
        return fn
    room = ward.layout(5000 + i)
    r = np.random.default_rng(50000 + i)
    n_staff = int(r.integers(1, 5))
    e = ward.episode(room, 50000 + i, [f'{BVH}/{c}.bvh' for c in TRAIN_CLIPS], n_staff, geometry_only=True)
    e['G0'] = (10 * np.log10(np.maximum(e['G0'], 1e-12))).astype(np.float16)
    e.pop('G')
    e.update(split='synth', layout=i, ep=0, n_staff=n_staff, rho=room['rho'])
    pickle.dump(e, open(fn, 'wb'))
    return fn


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    with Pool(1) as p:
        for f in p.imap_unordered(job, range(n)):
            print(f, flush=True)
