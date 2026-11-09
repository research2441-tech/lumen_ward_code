"""Figures 2-7 (black and white)."""
import pickle, glob, json, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import ward, analyze2 as A

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
                     'legend.frameon': False, 'mathtext.fontset': 'dejavuserif'})
LS = ['-', '--', ':', '-.', (0, (5, 1, 1, 1)), (0, (1, 2))]
GR = ['black', '0.35', '0.55', '0.7', '0.2', '0.45']


def fig_ward():
    f = 'corpus/test_02_0.pkl'
    e = pickle.load(open(f, 'rb'))
    room = ward.layout(3002)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a.add_patch(Rectangle((0, 0), room['L'], room['W'], fc='none', ec='black', lw=0.8))
    for bd in room['beds']:
        y0 = bd['y_head'] if bd['sgn'] > 0 else bd['y_head'] - 2.1
        a.add_patch(Rectangle((bd['xc'] - 0.5, y0), 1.0, 2.1, fc='0.88', ec='black', lw=0.5))
    t_c = int(np.argmax(e['curtain'].sum(0)))
    for bi in np.nonzero(e['curtain'][:, t_c])[0]:
        for bx in ward.curtain_boxes(room, bi):
            a.plot([bx[0], bx[3]] if bx[3] - bx[0] > 0.05 else [bx[0], bx[0]], [bx[1], bx[4]] if bx[4] - bx[1] > 0.05 else [bx[1], bx[1]], color='black', lw=1.4)
    lum = e['lum']
    import env
    _, _, _, km = env.load_episode(e)
    km = km[:len(lum)]
    a.scatter(lum[~km, 0], lum[~km, 1], marker='s', s=22, facecolors='none', edgecolors='black', lw=0.6, label='luminaire')
    a.scatter(lum[km, 0], lum[km, 1], marker='s', s=22, color='black', label='luminaire + IR receiver')
    a.scatter([room['rf_ap'][0]], [room['rf_ap'][1]], marker='^', s=40, color='white', edgecolors='black', lw=0.8, label='RF access point', zorder=5)
    for s in range(e['staff'].shape[0]):
        a.plot(e['staff'][s, :, 0], e['staff'][s, :, 1], color='0.4', ls='--', lw=0.7, label='staff path' if s == 0 else None)
    pos = e['pos']
    for p in range(len(e['walk'])):
        if e['walk'][p]:
            a.plot(pos[:, 3 * p, 0], pos[:, 3 * p, 1], color='black', lw=0.9, label='ambulating patient')
    a.set_xlim(-0.3, room['L'] + 0.3); a.set_ylim(-0.3, room['W'] + 0.3); a.set_aspect('equal')
    a.set_xlabel('x (m)'); a.set_ylabel('y (m)')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, handlelength=1.6, columnspacing=1.0)
    a.set_title('(a)', fontsize=8)
    G = pickle.load(open('twin_test_gen.pkl', 'rb'))[f]['twin'].astype(np.float32)
    x = e['G'].astype(np.float32)
    n = int(np.argmax(((x < e['G0'].astype(np.float32) - 3) & (e['G0'].astype(np.float32) > -119)).sum((0, 2))))
    ks = np.argsort(-np.median(x[:, n, :], 0))[:2]
    t = np.arange(x.shape[0]) * 0.01
    for i, k in enumerate(ks):
        b.plot(t, x[:, n, k], color='black', ls=LS[i], lw=0.8, label=f'ray-traced, receiver {i + 1}')
        b.plot(t, G[:, n, k] - 12, color='0.55', ls=LS[i], lw=0.8, label=f'twin, receiver {i + 1} (offset $-$12 dB)')
    b.set_xlabel('time (s)'); b.set_ylabel('uplink gain (dB)'); b.set_xlim(0, 60)
    lo = min(x[:, n, ks].min(), G[:, n, ks].min() - 12)
    b.set_ylim(lo - 2, x[:, n, ks].max() + 20)
    b.legend(loc='upper center', ncol=2, fontsize=5.6, handlelength=2.2, columnspacing=0.8, borderaxespad=0.1)
    b.set_title('(b)', fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig('figs/fig2_ward.png', dpi=300, bbox_inches='tight')


def fig_twin():
    S = pickle.load(open('results/twin_fidelity.pkl', 'rb'))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    names = [('true', 'ray-traced'), ('twin', 'twin'), ('uncoupled', 'twin, independent links'), ('memoryless', 'twin, memoryless'), ('phys', 'no occlusion process')]
    for i, (k, lab) in enumerate(names):
        v = np.sort(S['best_samples'][k]); a.plot(v, np.linspace(0, 1, len(v)), color=GR[i], ls=LS[i], label=lab)
    a.set_xlabel('best-receiver gain of a node (dB)'); a.set_ylabel('CDF'); a.legend(fontsize=6, loc='lower right'); a.set_title('(a)', fontsize=8)
    a.set_xlim(np.percentile(S['best_samples']['true'], 0.5) - 2, np.percentile(S['best_samples']['true'], 99.5) + 1)
    ks = ['true', 'twin', 'uncoupled', 'memoryless']
    labs = ['ray-traced', 'twin', 'indep. links', 'memoryless']
    xs = np.arange(3); wdt = 0.2
    hat = ['', '////', '....', 'xxxx']
    for i, k in enumerate(ks):
        vals = [100 * S[k]['blk_frac'], min(S[k]['onset_per_min'], 12), S[k]['dur_ms'] / 100]
        bars = b.bar(xs + (i - 1.5) * wdt, vals, wdt, color='white' if i else '0.3', edgecolor='black', hatch=hat[i], lw=0.6, label=labs[i])
        if k == 'memoryless':
            b.text(xs[1] + (i - 1.5) * wdt, 12.3, f"{S[k]['onset_per_min']:.0f}", ha='center', fontsize=5.8, rotation=90, va='bottom')
    b.set_xticks(xs); b.set_xticklabels(['shadowed\nlinks (%)', 'onsets per\nlink-minute', 'mean duration\n(0.1 s)'])
    b.set_ylim(0, 22); b.legend(fontsize=5.8, ncol=4, loc='upper center', columnspacing=0.8, handlelength=1.5); b.set_title('(b)', fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig('figs/fig3_twin.png', dpi=300, bbox_inches='tight')



KEY = ['RR', 'MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward']


def fig_main():
    R = pickle.load(open('results/eval_main.pkl', 'rb'))[('test', 'mimic')]
    T = A.table(R, KEY + ['LUMEN w/o certificates', 'LUMEN w/o alarm term'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.5))
    names = KEY
    xs = np.arange(len(names)); wd = 0.38
    sw = [T[n]['swaoi'][0] for n in names]; sws = [T[n]['swaoi'][1] for n in names]
    un = [T[n]['aoi'][0] for n in names]; uns = [T[n]['aoi'][1] for n in names]
    a.bar(xs - wd / 2, sw, wd, yerr=sws, color='0.3', edgecolor='black', lw=0.5, capsize=2, label='severity-weighted AoI')
    a.bar(xs + wd / 2, un, wd, yerr=uns, color='white', edgecolor='black', hatch='////', lw=0.5, capsize=2, label='unweighted AoI')
    a.set_xticks(xs); a.set_xticklabels(['RR', 'MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-W+C+AP', 'LUMEN'], rotation=25, ha='right')
    a.set_ylabel('average AoI (ms)'); a.set_ylim(0, 115); a.legend(fontsize=6, loc='upper left', ncol=2); a.set_title('(a)', fontsize=8)
    show = [('MW-AoI', 'MW-AoI'), ('DGET-obj', 'DGET-obj'), ('SW-Whittle', 'SW-Whittle'), ('LUMEN w/o alarm term', 'LUMEN w/o alarm term'), ('LUMEN-Ward', 'LUMEN-Ward')]
    lat = np.arange(401) * 10
    for i, (n, lab) in enumerate(show):
        c = T[n]['lat_cdf']; b.semilogy(lat[1:], np.maximum(1 - c[1:], 1e-5), color=GR[i], ls=LS[i], label=lab)
    b.axvline(250, color='black', lw=0.6, ls=':'); b.axhline(0.01, color='0.6', lw=0.6, ls=':')
    b.text(255, 0.4, '$L_{\\max}$', fontsize=7); b.text(620, 0.0125, '$\\delta=1\\%$', fontsize=7)
    b.set_xlim(0, 1000); b.set_ylim(1e-4, 1.2); b.set_xlabel('alarm latency (ms)'); b.set_ylabel('$\\Pr\\{\\ell > x\\}$')
    b.legend(fontsize=5.8, loc='upper right'); b.set_title('(b)', fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig('figs/fig4_main.png', dpi=300, bbox_inches='tight')


def fig_sweep():
    R = pickle.load(open('results/eval_sweep.pkl', 'rb'))
    pols = ['MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward']
    labs = ['MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-W+C+AP', 'LUMEN-Ward']
    panels = [('D', [1, 5, 10, 20], 'reporting delay $D$ (slots)', 1), ('p_tx', [0.0075, 0.010, 0.015], 'emitter power $P_{\\mathrm{o}}$ (mW)', 1000),
              ('k_rf', [0, 1, 2], 'radio slots $K_{\\mathrm{RF}}$', 1), ('lam_alarm', [0.02, 0.1, 0.5], 'alarm rate $\\lambda_a$ (s$^{-1}$)', 1)]
    fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.6), sharex='col')
    for j, (k, vals, xl, sc) in enumerate(panels):
        for i, p in enumerate(pols):
            sw = [A.table(R[(k, v)], [p], seeds=(0,))[p]['swaoi'][0] for v in vals]
            vi = [A.table(R[(k, v)], [p], seeds=(0,))[p]['viol'][0] for v in vals]
            x = np.array(vals) * sc
            axs[0, j].plot(x, sw, color=GR[i], ls=LS[i], marker='os^dv'[i], ms=3, label=labs[i])
            axs[1, j].plot(x, np.maximum(vi, 0.1), color=GR[i], ls=LS[i], marker='os^dv'[i], ms=3)
        axs[1, j].set_yscale('log'); axs[1, j].axhline(1.0, color='0.6', ls=':', lw=0.6)
        axs[1, j].set_yticks([0.1, 1, 10]); axs[1, j].set_yticklabels(['0.1', '1', '10']); axs[1, j].set_ylim(0.08, 60)
        axs[1, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if k == 'k_rf':
            axs[0, j].set_yscale('log'); axs[0, j].set_yticks([30, 100, 300, 1000, 3000]); axs[0, j].set_yticklabels(['30', '100', '300', '1000', '3000'])
            axs[0, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[1, j].set_xlabel(xl); axs[0, j].set_title(f'({"abcd"[j]})', fontsize=8)
        if k == 'lam_alarm':
            for r_ in (0, 1):
                axs[r_, j].set_xscale('log'); axs[r_, j].set_xticks([0.02, 0.1, 0.5]); axs[r_, j].set_xticklabels(['0.02', '0.1', '0.5'])
                axs[r_, j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0, 0].set_ylabel('SW-AoI (ms)'); axs[1, 0].set_ylabel('alarm violations (%)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.8)
    fig.savefig('figs/fig5_sweep.png', dpi=300, bbox_inches='tight')


def fig_alpha():
    R = pickle.load(open('results/eval_alpha.pkl', 'rb'))
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    src = [('test', 'ray-traced test wards'), ('twin', 'twin'), ('uncoupled', 'twin, independent links'), ('memoryless', 'twin, memoryless'), ('phys', 'no occlusion process')]
    for i, (s, lab) in enumerate(src):
        al = sorted(a for (ss, a) in R if ss == s)
        T = [A.table(R[(s, a)], ['SW-Whittle+C+AP'], seeds=(0,))['SW-Whittle+C+AP'] for a in al]
        ax.plot([t['fail'][0] for t in T], [t['swaoi'][0] for t in T], color=GR[i], ls=LS[i], marker='os^dv'[i], ms=3, label=lab)
    ax.set_xscale('log'); ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '5'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('failed optical transmissions (%)'); ax.set_ylabel('SW-AoI (ms)')
    ax.legend(fontsize=5.8, loc='upper right')
    fig.tight_layout(); fig.savefig('figs/fig6_alpha.png', dpi=300, bbox_inches='tight')



def fig_paired():
    R = pickle.load(open('results/eval_main.pkl', 'rb'))[('test', 'mimic')]
    T = A.table(R, ['MW-AoI', 'DGET-obj', 'LUMEN-Ward'])
    H = pickle.load(open('es_twin.pkl', 'rb'))['hist']
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for i, (n, m) in enumerate((('MW-AoI', 'o'), ('DGET-obj', 's'))):
        a.scatter(T[n]['ep_swaoi'], T['LUMEN-Ward']['ep_swaoi'], s=9, marker=m, facecolors='white' if i else 'black', edgecolors='black', lw=0.6, label=n)
        b.scatter(T[n]['ep_fail'], T['LUMEN-Ward']['ep_fail'], s=9, marker=m, facecolors='white' if i else 'black', edgecolors='black', lw=0.6, label=n)
    lo, hi = 30, 105
    a.plot([lo, hi], [lo, hi], color='0.5', ls=':', lw=0.7); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('comparator SW-AoI (ms)'); a.set_ylabel('LUMEN-Ward SW-AoI (ms)'); a.legend(fontsize=6, loc='upper left'); a.set_title('(a)', fontsize=8)
    b.plot([0, 14], [0, 14], color='0.5', ls=':', lw=0.7); b.set_xlim(0, 14); b.set_ylim(0, 14)
    b.set_xlabel('comparator failed tx (%)'); b.set_ylabel('LUMEN-Ward failed tx (%)'); b.legend(fontsize=6, loc='upper left'); b.set_title('(b)', fontsize=8)
    it = [h['it'] for h in H if 'val_rel' in h]
    c.plot([h['it'] for h in H], [h['rel'] for h in H], color='0.6', lw=0.6, label='population mean (train)')
    c.plot(it, [h['val_rel'] for h in H if 'val_rel' in h], color='black', marker='o', ms=3, lw=0.9, label='centre (validation)')
    c.axhline(1.0, color='black', ls=':', lw=0.6)
    c.set_xlabel('ES iteration'); c.set_ylabel('SW-AoI / index policy'); c.set_ylim(0.99, 1.01); c.legend(fontsize=6, loc='upper right'); c.set_title('(c)', fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig('figs/fig7_paired.png', dpi=300, bbox_inches='tight')



def tier_stats(RR, n, seeds=(0, 1, 2)):
    a = sum(RR[(n, s)]['a_lvl'].sum(0) for s in seeds); c = sum(RR[(n, s)]['n_lvl'].sum(0) for s in seeds)
    v = sum(RR[(n, s)]['lat_lvl'].sum(0) for s in seeds); al = sum(RR[(n, s)]['al_lvl'].sum(0) for s in seeds)
    return 10 * a / np.maximum(c, 1), 100 * v / np.maximum(al, 1), al


def fig_tier():
    R = pickle.load(open('results/eval_tier.pkl', 'rb'))['mimic']
    pols = [('MW-AoI', 'MW-AoI'), ('SW-Whittle', 'SW-Whittle'), ('LUMEN w/o alarm term', 'LUMEN w/o alarm term'), ('LUMEN-Ward', 'LUMEN-Ward')]
    hat = ['', '////', '....', '']
    fc = ['white', 'white', 'white', '0.3']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    xs = np.arange(4); wd = 0.2
    for i, (n, lab) in enumerate(pols):
        ag, vi, _ = tier_stats(R, n)
        a.bar(xs + (i - 1.5) * wd, ag, wd, color=fc[i], hatch=hat[i], edgecolor='black', lw=0.5, label=lab)
        b.bar(xs + (i - 1.5) * wd, np.maximum(vi, 0.02), wd, color=fc[i], hatch=hat[i], edgecolor='black', lw=0.5, label=lab)
    tl = ['score 0\n($w$=1/12)', '1–4\n($w$=1/5)', 'red 3 or 5–6\n($w$=1)', '$\\geq$7\n($w$=4)']
    for ax in (a, b):
        ax.set_xticks(xs); ax.set_xticklabels(tl, fontsize=6.5)
    a.set_ylabel('average AoI (ms)'); a.set_ylim(0, 260); a.legend(fontsize=6, ncol=2, loc='upper right'); a.set_title('(a)', fontsize=8)
    b.set_yscale('log'); b.set_ylim(0.01, 100); b.axhline(1.0, color='0.5', ls=':', lw=0.6)
    b.set_yticks([0.1, 1, 10]); b.set_yticklabels(['0.1', '1', '10']); b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_ylabel('alarm violations (%)'); b.set_title('(b)', fontsize=8)
    fig.tight_layout(w_pad=1.5)
    fig.savefig('figs/fig8_tier.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        globals()[fn]()
