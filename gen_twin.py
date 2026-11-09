"""Generate channel episodes with the twin variants: twin (coupled Markov occlusion), uncoupled (independent
per-link uniforms), memoryless (occlusion without state/run-length memory), phys (no occlusion process)."""
import sys, glob, pickle, time, os, numpy as np, jax
import twin2

P = pickle.load(open('twin2_params.pkl', 'rb'))
VARS = ('twin', 'uncoupled', 'memoryless', 'phys')

if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'synth':
        os.makedirs('synth_gen', exist_ok=True)
        times = []
        for i, f in enumerate(sorted(glob.glob('synth/syn_*.pkl'))):
            e = pickle.load(open(f, 'rb'))
            for v in VARS:
                fn = f'synth_gen/{v}_{i:03d}.pkl'
                if os.path.exists(fn):
                    continue
                t = time.time()
                G = twin2.generate(P, e, jax.random.PRNGKey(i), v)
                times.append((v, time.time() - t, G.shape))
                pickle.dump(dict(e, G=G.astype(np.float16)), open(fn, 'wb'))
            print(i, flush=True)
        pickle.dump(dict(times=times), open('results/twin_gen_times.pkl', 'wb'))
    else:
        out = {}
        for j, f in enumerate(sorted(glob.glob('corpus/test_*.pkl'))):
            e = pickle.load(open(f, 'rb'))
            out[f] = {v: twin2.generate(P, e, jax.random.PRNGKey(900 + j), v).astype(np.float16) for v in VARS}
            print(f, flush=True)
        pickle.dump(out, open('twin_test_gen.pkl', 'wb'))
