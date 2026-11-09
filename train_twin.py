"""Train the generative channel twin (MDN, 4 components) and a single-Gaussian AR twin; save parameters."""
import glob, pickle, time, numpy as np, jax
import twin

tr = sorted(glob.glob('corpus/train_*.pkl')); va = sorted(glob.glob('corpus/val_*.pkl'))
t0 = time.time()
X, Y = twin.dataset(tr, 0.03, 0)
Xv, Yv = twin.dataset(va, 0.02, 1)
mu, sd = X.mean(0), X.std(0) + 1e-6
mu[[1, 5, 7, 8, 9]] = 0; sd[[1, 5, 7, 8, 9]] = 1             # keep binary features as they are
Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
print('samples', X.shape, Xv.shape, 'prep', time.time() - t0, flush=True)
out = {'mu': mu, 'sd': sd}
for name, k in (('mdn', 4), ('gauss', 1)):
    t1 = time.time()
    p, hist = twin.train(Xn, Y, Xvn, Yv, seed=0, epochs=10, kmix=k)
    out[name] = jax.tree.map(np.asarray, p)
    out[name + '_hist'] = hist
    out[name + '_time'] = time.time() - t1
    print(name, 'done', time.time() - t1, flush=True)
pickle.dump(out, open('twin_params.pkl', 'wb'))
