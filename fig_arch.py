"""Fig. 1: LUMEN-Ward architecture (black and white)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'mathtext.fontset': 'dejavuserif'})
fig, ax = plt.subplots(figsize=(7.16, 4.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
FS = 6.4


def frame(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1.0', fc='#f3f3f3', ec='black', lw=0.6, ls='--'))
    ax.text(x + 1.2, y + h - 2.2, title, ha='left', va='center', fontsize=6.8, style='italic')


def box(x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.8', fc='white', ec='black', lw=0.85))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=FS, linespacing=1.3)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arrow(p, q, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.8, color='black', ls=ls, shrinkA=0, shrinkB=0))


def poly(pts, ls='-'):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=0.8, ls=ls)
    arrow(pts[-2], pts[-1], ls)


frame(1, 45, 56, 24, 'Offline: channel twin and scorer training')
frame(1, 18, 98, 24.5, 'Online: one decision per 10 ms slot')
frame(1, 0.5, 98, 15, 'Ward')
rt = box(3, 57, 15, 7, 'Ray-traced\ntraining wards')
fit = box(22, 57, 15, 7, 'Fit twin heads\n$\\mu_d,\\sigma_d,h_o$ (16)')
syn = box(3, 47.5, 15, 7, 'Synthetic ward\ngeometry')
tw = box(22, 47.5, 15, 7, 'Generative twin\n(12)-(15)')
es = box(41, 47.5, 14, 7, 'Lagrangian ES\n(18)-(20)')
arrow((rt['r'], rt['cy']), (fit['l'], fit['cy']))
arrow((fit['cx'], fit['b']), (tw['cx'], tw['t']))
arrow((syn['r'], syn['cy']), (tw['l'], tw['cy']))
arrow((tw['r'], tw['cy']), (es['l'], es['cy']))

bc = box(3, 29, 15, 7, 'Delayed beacons\n$\\hat{x}_{t}=x_{t-D}$')
ce = box(22, 29, 14, 7, 'Certificates\n$c=\\hat{x}-q$ (10)')
sc = box(40, 29, 18, 7, 'Scores $s_{n\\ell}$ (17):\nWhittle index + $f_\\theta$')
ma = box(62, 29, 16, 7, 'Greedy matching\nunder (11)')
rf = box(82, 29, 15, 7, 'Radio slots\n($K_{\\mathrm{RF}}$ best)')
ag = box(36, 19.5, 14, 7, 'Ages $a_n$,\nalarm waits')
nw = box(53, 19.5, 14, 7, 'NEWS2\nweights $w_n$ (5)')
arrow((bc['r'], bc['cy']), (ce['l'], ce['cy']))
arrow((ce['r'], ce['cy']), (sc['l'], sc['cy']))
arrow((sc['r'], sc['cy']), (ma['l'], ma['cy']))
arrow((ma['r'], ma['cy']), (rf['l'], rf['cy']))
arrow((43, ag['t']), (43, sc['b']))
arrow((56, nw['t']), (56, sc['b']))
arrow((48, es['b']), (48, sc['t']), ls='--')
ax.text(49, 40.3, '$\\theta$', fontsize=7)

nd = box(21, 3, 21, 9, 'Body nodes: IR emitters\non chest, wrist and arm')
rx = box(3, 3, 15, 9, 'Ceiling IR\nreceivers')
ap = box(82, 3, 15, 9, 'RF access\npoint')
db = box(50, 7, 20, 6, 'MIMIC-IV, eICU-CRD\nhourly NEWS2')
arrow((nd['l'], nd['cy']), (rx['r'], rx['cy']))
arrow((rx['cx'], rx['t']), (bc['cx'], bc['b']))
arrow((60, db['t']), (60, nw['b']))
arrow((rf['cx'], rf['b']), (ap['cx'], ap['t']))
arrow((nd['r'], 4.6), (ap['l'], 4.6))
ax.text(10.5 + 1.2, 23.0, 'gain\nreports', fontsize=5.8, ha='left', va='center')
ax.text(rf['cx'] + 1.0, 23.0, 'grants', fontsize=5.8, ha='left', va='center')
fig.savefig('figs/fig1_architecture.png', dpi=300, bbox_inches='tight')
