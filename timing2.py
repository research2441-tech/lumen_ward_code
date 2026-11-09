"""Cost of ray tracing vs twin generation, and of the online scheduler."""
import time, json, pickle, numpy as np, jax, jax.numpy as jnp
import ward, twin2, env, data
from gen_corpus import BVH, TEST_CLIPS

room = ward.layout(3002)
clips = [f'{BVH}/{c}.bvh' for c in TEST_CLIPS]
t = time.time(); e = ward.episode(room, 42, clips, 3, T=1000); rt = (time.time() - t) * 6
t = time.time(); g = ward.episode(room, 42, clips, 3, T=6000, geometry_only=True); geo = time.time() - t
P = pickle.load(open('twin2_params.pkl', 'rb'))
twin2.generate(P, g, jax.random.PRNGKey(0))
t = time.time(); twin2.generate(P, g, jax.random.PRNGKey(1)); tw = time.time() - t
ep = data.pool('test')[0]
X, RF, nm, km = ep
pat = jnp.asarray(np.repeat(np.arange(env.N_MAX // 3), 3))
w = jnp.ones(env.N_MAX)
th = pickle.load(open('es_twin.pkl', 'rb'))
run = env.make_run()
args = (jnp.asarray(X, jnp.float32), jnp.asarray(RF), jnp.asarray(nm), jnp.asarray(km), w, pat, jax.random.PRNGKey(0))
o = run(env.POL['lumen'], jax.tree.map(jnp.asarray, th['theta']), True, 1.0, *args); jax.block_until_ready(o)
t = time.time(); o = run(env.POL['lumen'], jax.tree.map(jnp.asarray, th['theta']), True, 1.0, *args); jax.block_until_ready(o); sch = (time.time() - t) / X.shape[0]
nparam = int(sum(np.size(v) for v in th['theta'].values()))
P1 = pickle.load(open('twin2_params.pkl', 'rb'))
ntw = int(sum(np.size(v) for h in ('d', 'o') for v in P1[h].values()))
es_t = th['hist'][-1]['t']
out = dict(raytrace_s_per_episode=rt, geometry_s_per_episode=geo, twin_s_per_episode=tw, scheduler_ms_per_slot=1e3 * sch,
           scorer_params=nparam, twin_params=ntw, twin_train_s=P1['train_time'], es_train_s=es_t, es_iters=len(th['hist']))
json.dump(out, open('results/timing.json', 'w'), indent=1)
print(out)
