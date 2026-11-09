import glob, pickle, time, numpy as np
import twin2
tr = sorted(glob.glob('corpus/train_*.pkl')); va = sorted(glob.glob('corpus/val_*.pkl'))
t0 = time.time()
Fd, Yd, Fo, Yo = twin2.datasets(tr, 0.04, 0)
Fdv, Ydv, Fov, Yov = twin2.datasets(va, 0.03, 1)
print('data', Fd.shape, Fo.shape, 'blocked frac', Yo.mean(), time.time() - t0, flush=True)
t1 = time.time()
pd_, hd = twin2.fit(twin2.loss_d, Fd.shape[1], 2, Fd, Yd, Fdv, Ydv, epochs=6, h=32, lr=1e-3)
po, ho = twin2.fit(twin2.loss_o, Fo.shape[1], 1, Fo, Yo, Fov, Yov, epochs=6)
pickle.dump(dict(d=pd_, o=po, hist_d=hd, hist_o=ho, train_time=time.time() - t1, n_d=len(Fd), n_o=len(Fo)), open('twin2_params.pkl', 'wb'))
print('done', time.time() - t1)
