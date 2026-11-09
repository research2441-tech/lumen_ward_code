"""Ray-traced ward corpus: train / validation / test / big-ward layouts, two 60 s episodes each."""
import os, sys, numpy as np, pickle
from multiprocessing import Pool
import ward

BVH = os.environ.get('BVH_DIR', './bvh')
TRAIN_CLIPS = ['16_%02d' % i for i in list(range(11, 35)) + [47, 58]] + ['07_%02d' % i for i in range(1, 13)]
VAL_CLIPS = ['08_%02d' % i for i in range(1, 12)]
TEST_CLIPS = ['02_01', '02_02', '05_01', '06_01', '12_01', '12_02', '12_03', '26_01', '27_01', '10_04'] + \
             ['17_%02d' % i for i in range(1, 10)] + ['15_01', '15_03', '15_09', '15_14']
SPLITS = dict(train=(1000, 32, TRAIN_CLIPS, False, (1, 4)), val=(2000, 8, VAL_CLIPS, False, (1, 4)),
              test=(3000, 16, TEST_CLIPS, False, (1, 4)), big=(4000, 8, TEST_CLIPS, True, (3, 5)))
OUT = 'corpus'


def job(a):
    split, li, ep = a
    s0, n, clips, big, (s_lo, s_hi) = SPLITS[split]
    fn = f'{OUT}/{split}_{li:02d}_{ep}.pkl'
    if os.path.exists(fn):
        return fn
    room = ward.layout(s0 + li, big=big)
    r = np.random.default_rng(10 * (s0 + li) + ep)
    n_staff = int(r.integers(s_lo, s_hi + 1))
    e = ward.episode(room, 10 * (s0 + li) + ep, [f'{BVH}/{c}.bvh' for c in clips], n_staff)
    e['G'] = (10 * np.log10(e['G'])).astype(np.float16)
    e['G0'] = (10 * np.log10(np.maximum(e['G0'], 1e-12))).astype(np.float16)
    e.update(split=split, layout=li, ep=ep, n_staff=n_staff, rho=room['rho'])
    pickle.dump(e, open(fn + '.tmp', 'wb'))
    os.replace(fn + '.tmp', fn)
    return fn


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, li, ep) for s in ('train', 'val', 'test', 'big') for li in range(SPLITS[s][1]) for ep in range(2)]
    with Pool(2) as p:
        for i, f in enumerate(p.imap_unordered(job, jobs)):
            print(i, f, flush=True)
