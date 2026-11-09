"""Assemble the LUMEN-Ward manuscript (IEEE template) from text, results and figures."""
import pickle, json, numpy as np
import docbuild as D
import cite as C
import paper2_text as P
import analyze2 as A

TEMPLATE = './IEEE_template.docx'
OUT = './out/LUMEN_Ward_Q1_Paper.docx'
RM = pickle.load(open('results/eval_main.pkl', 'rb'))
RS = pickle.load(open('results/eval_sweep.pkl', 'rb'))
RA = pickle.load(open('results/eval_alpha.pkl', 'rb'))
CS = json.load(open('results/corpus_stats.json'))
TF = pickle.load(open('results/twin_fidelity.pkl', 'rb'))
FR = json.load(open('results/mdn_freerun.json'))
TM = json.load(open('results/timing.json'))
ALL = ['RR', 'MW-AoI', 'TO', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward', 'LUMEN w/o certificates', 'LUMEN w/o alarm term']
T = {k: A.table(v, ALL) for k, v in RM.items()}
TT = T[('test', 'mimic')]


def f1(x): return f'{x:.1f}'
def f2(x): return f'{x:.2f}'
def red(a, b): return 100 * (1 - a / b)
def sw(n, k=('test', 'mimic')): return T[k][n]['swaoi'][0]


def fmt_p(p):
    return '$<10^{-4}$' if p < 1e-4 else (f'{p:.4f}' if p < 0.01 else f'{p:.3f}')


def add_items(d, items):
    for kind, c in items:
        if kind == 'p':
            d.para(C.resolve(c))
        else:
            d.eq(c)


# ------------------------------------------------------------------------------------------------ tables
PARAMS = [
['Parameter', 'Value'],
['Slot length; episode length', '10 ms; 60 s (6000 slots)'],
['Ward size; beds; nodes per patient', '$61.8$–$130.6$ m$^2$; 6–10; 3'],
['Luminaires; IR receivers (checkerboard)', '6–18; 3–9'],
['Emitter: Lambertian order $m$; mean power $P_{\\mathrm{o}}$', '2; 10 mW (reference 30 mW)'],
['Receiver: area, FOV, concentrator index', '2 cm$^2$, 60°, 1.5'],
['Uplink DCO-OFDM: bandwidth, clipping $\\kappa$', '5 MHz, 3.2'],
['Decoding threshold $\\Gamma$ (4-QAM, SER $10^{-3}$)', '10.35 dB'],
['RF: $P_{\\mathrm{tx}}$, path-loss exponent, body / staff shadowing', '$-20$ dBm, 2.0, 15 / 6 dB'],
['RF: noise floor, $\\Gamma_{\\mathrm{RF}}$, radio slots $K_{\\mathrm{RF}}$', '$-104$ dBm, 10 dB, 1'],
['Reporting delay $D$; certificate level $\\alpha$; step $\\eta$', '5 slots; 0.02; 0.05'],
['Alarm rate $\\lambda_a$; bound $L_{\\max}$; level $\\delta$', '0.1 s$^{-1}$ per patient; 250 ms; 1%'],
['ES: population $M$, $\\sigma$, windows, iterations', '24, 0.10, $8\\times15$ s, 80'],
]


def CORPUS():
    r = [['Split', 'Wards / episodes', 'Beds', 'Nodes', 'IR receivers', 'Staff', 'Curtain time', 'LOS shadowed', 'Best gain (dB)']]
    lab = dict(train='Train', val='Validation', test='Test', big='Large-ward test')
    for s in ('train', 'val', 'test', 'big'):
        c = CS[s]
        r.append([lab[s], f"{c['eps'] // 2} / {c['eps']}", f"{c['beds'][1]:.0f}–{c['beds'][2]:.0f}", f"{c['nodes'][0]:.1f}",
                  f"{c['rx'][0]:.1f}", f"{c['staff'][1]:.0f}–{c['staff'][2]:.0f}", f"{100 * c['curtain'][0]:.0f}%",
                  f"{100 * c['blk'][0]:.1f}%", f"{c['best_sub'][0]:.1f}"])
    return r


def TWIN_TABLE():
    r = [['Generator', 'Shadowed (%)', 'Onsets / link-min', 'Duration (ms)', 'Co-shadowed links', 'W$_1$ best (dB)', 'KS best']]
    for k, lab in (('true', 'Ray-traced (reference)'), ('twin', 'Twin (proposed)'), ('uncoupled', 'Twin, independent links'),
                   ('memoryless', 'Twin, memoryless occlusion'), ('phys', 'No occlusion process')):
        s = TF[k]
        r.append([lab, f"{100 * s['blk_frac']:.2f}", f"{s['onset_per_min']:.2f}", f"{s['dur_ms']:.0f}", f"{s['co_blk']:.2f}",
                  '—' if k == 'true' else f"{s['W1_best_db']:.2f}", '—' if k == 'true' else f"{s['KS_best']:.3f}"])
    r.append(['AR mixture-density twin', f"diverged: {100 * FR['twin']['implausible']:.1f}% of link-slots outside $[-120,-20]$ dB", '', '', '', '', ''])
    return r


NAMES = {'RR': 'Round robin', 'MW-AoI': 'MW-AoI [@kadota18]', 'TO': 'Throughput-optimal', 'DGET-obj': 'DGET objective [@hamrouni25]',
         'SW-Whittle': 'Severity-weighted Whittle', 'SW-Whittle+C+AP': 'SW-Whittle + certificates + alarm priority',
         'LUMEN-Ward': 'LUMEN-Ward (proposed)'}


def MAIN_TABLE(key=('test', 'mimic'), names=('RR', 'MW-AoI', 'TO', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward')):
    r = [['Scheduler', 'SW-AoI (ms)', 'AoI (ms)', 'AoI, $w\\ge1$ (ms)', 'p99 age (ms)', 'Alarm viol. (%)', 'Failed tx (%)', 'Attempts / delivery']]
    TK = T[key]
    for n in names:
        t = TK[n]
        r.append([C.resolve(NAMES.get(n, n)), f"{t['swaoi'][0]:.1f} ± {t['swaoi'][1]:.1f}", f"{t['aoi'][0]:.1f}", f"{t['aoi_hi'][0]:.1f}",
                  f"{t['p99'][0]:.0f}", f"{t['viol'][0]:.2f}", f"{t['fail'][0]:.2f}", f"{t['att'][0]:.3f}"])
    return r


def ABL_TABLE():
    r = [['Variant', 'SW-AoI (ms)', 'ΔSW-AoI', 'Alarm viol. (%)', 'Failed tx (%)', 'Radio share (%)']]
    base = sw('LUMEN-Ward')
    for n, lab in (('LUMEN-Ward', 'Full LUMEN-Ward'), ('SW-Whittle+C+AP', 'Without learned residual ($f_\\theta=0$)'),
                   ('LUMEN w/o certificates', 'Without certificates ($q=0$)'), ('LUMEN w/o alarm term', 'Without alarm term ($\\beta=0$)'),
                   ('SW-Whittle', 'Index only (no certificates, no alarm term)'), ('MW-AoI', 'Without severity weights (MW-AoI)')):
        t = TT[n]
        r.append([lab, f"{t['swaoi'][0]:.1f}", '—' if n == 'LUMEN-Ward' else f"{100 * (t['swaoi'][0] / base - 1):+.1f}%",
                  f"{t['viol'][0]:.2f}", f"{t['fail'][0]:.2f}", f"{t['rfshare'][0]:.1f}"])
    return r


def GEN_TABLE():
    r = [['Scheduler', 'Test, MIMIC-IV', 'Test, eICU-CRD', 'Large wards, MIMIC-IV']]
    for n in ('MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN-Ward'):
        row = [C.resolve(NAMES.get(n, n))]
        for k in (('test', 'mimic'), ('test', 'eicu'), ('big', 'mimic')):
            t = T[k][n]
            row.append(f"{t['swaoi'][0]:.1f} / {t['viol'][0]:.2f} / {t['fail'][0]:.2f}")
        r.append(row)
    return r


def STAT_TABLE():
    others = ['RR', 'MW-AoI', 'DGET-obj', 'SW-Whittle', 'SW-Whittle+C+AP', 'LUMEN w/o certificates']
    r = [['Comparison (LUMEN-Ward vs.)', 'Mean Δ SW-AoI (ms)', '$p_{\\mathrm{Holm}}$', '$r$', 'Mean Δ failed tx (pp)', '$p_{\\mathrm{Holm}}$', '$r$']]
    a = A.paired(TT, 'LUMEN-Ward', others, 'ep_swaoi')
    b = A.paired(TT, 'LUMEN-Ward', others, 'ep_fail')
    for x, y in zip(a, b):
        r.append([C.resolve(NAMES.get(x[0], x[0])), f'{x[1]:+.2f}', fmt_p(x[3]), f'{x[4]:.2f}', f'{y[1]:+.2f}', fmt_p(y[3]), f'{y[4]:.2f}'])
    return r, a, b


def SWEEP_TABLE():
    pols = ['MW-AoI', 'DGET-obj', 'SW-Whittle', 'LUMEN-Ward']
    r = [['Setting', 'MW-AoI', 'DGET objective', 'SW-Whittle', 'LUMEN-Ward']]
    lab = {'D': 'Report delay $D$ = {} slots', 'p_tx': 'Emitter power {} mW', 'k_rf': 'Radio slots $K_{{\\mathrm{{RF}}}}$ = {}',
           'lam_alarm': 'Alarm rate {} s$^{{-1}}$', 'L_max': '$L_{{\\max}}$ = {} ms'}
    for (k, v) in [('D', 1), ('D', 10), ('D', 20), ('p_tx', 0.0075), ('p_tx', 0.015), ('k_rf', 0), ('k_rf', 2), ('lam_alarm', 0.02), ('lam_alarm', 0.5), ('L_max', 10), ('L_max', 50)]:
        vv = v * 1000 if k == 'p_tx' else (v * 10 if k == 'L_max' else v)
        row = [lab[k].format(f'{vv:g}')]
        for p in pols:
            t = A.table(RS[(k, v)], [p], seeds=(0,))[p]
            row.append(f"{t['swaoi'][0]:.1f} / {t['viol'][0]:.2f} / {t['fail'][0]:.2f}")
        r.append(row)
    return r


def EFF_TABLE():
    return [['Item', 'Cost'],
            ['Ray tracing one 60 s ward episode (CPU, NumPy)', f"{TM['raytrace_s_per_episode']:.0f} s"],
            ['Twin: geometry + generation of one episode', f"{TM['geometry_s_per_episode'] + TM['twin_s_per_episode']:.2f} s"],
            ['Twin parameters; fitting time', f"{TM['twin_params']:,}; {TM['twin_train_s']:.0f} s"],
            ['Scorer parameters; ES training (80 iterations)', f"{TM['scorer_params']}; {TM['es_train_s'] / 60:.0f} min"],
            ['Scheduler decision per slot (whole ward, JAX CPU)', f"{TM['scheduler_ms_per_slot']:.3f} ms"]]


# ------------------------------------------------------------------------------------------------ text
def ABSTRACT():
    mw, dg, lu = sw('MW-AoI'), sw('DGET-obj'), sw('LUMEN-Ward')
    return (f"Optical wireless access suits multi-patient hospital wards because ceiling luminaires sit above every bed and infrared "
            f"links emit no radio energy. A ward, however, is a shared and constantly shadowed medium: patients turn in bed, staff stand between "
            f"beds and ceilings, and curtains are drawn during care. Existing hybrid optical–RF schedulers treat all packets as equally urgent "
            f"and assume known link states. This paper presents LUMEN-Ward, a ward-scale uplink scheduler that minimises the age of information "
            f"weighted by each patient's National Early Warning Score 2 (NEWS2), keeps alarm latency inside a chance constraint, and admits an "
            f"optical link only when an online calibrated certificate guarantees its signal-to-interference-plus-noise ratio. Receivers share one "
            f"scorer that starts from a severity-weighted Whittle index and adds a residual trained by Lagrangian evolution strategies on a "
            f"physics-structured generative channel twin, which keeps the closed-form line of sight and learns the diffuse field and a coupled "
            f"Markov occlusion process. Evaluation uses 128 ray-traced one-minute episodes from 64 procedurally generated wards with CMU motion "
            f"capture and NEWS2 trajectories from the open MIMIC-IV and eICU-CRD demonstration databases. On held-out wards, LUMEN-Ward lowers "
            f"severity-weighted AoI to {f1(lu)} ms, against {f1(mw)} ms for max-weight AoI scheduling and {f1(dg)} ms for the DGET objective "
            f"({f1(red(lu, mw))}% lower). Failed optical transmissions fall from {f1(TT['MW-AoI']['fail'][0])}% to {f2(TT['LUMEN-Ward']['fail'][0])}%, and alarm-latency violations "
            f"stay at {f2(TT['LUMEN-Ward']['viol'][0])}% against {f1(TT['LUMEN w/o alarm term']['viol'][0])}% without the alarm term. The twin reproduces the shadowing "
            f"fraction, onset rate and duration of unseen wards and the certificate trade-off curve, while an autoregressive mixture-density twin "
            f"diverges. The learned residual adds less than 1% over the analytic index, which locates the gains in severity weighting, "
            f"certification and alarm control.")


def RES_TWIN():
    t, u, m, ph = TF['true'], TF['twin'], TF['memoryless'], TF['phys']
    return [
f"Table VI compares the generators on the 32 held-out test episodes. Every generator receives the observable geometry of the test ward and nothing else. The ray-traced reference shadows {100 * t['blk_frac']:.1f}% of visible link-slots, with {t['onset_per_min']:.1f} shadowing onsets per link-minute and a mean shadowing duration of {t['dur_ms']:.0f} ms. The twin generates {100 * u['blk_frac']:.1f}%, {u['onset_per_min']:.1f} onsets per link-minute and {u['dur_ms']:.0f} ms, all within 16% of the reference, although none of these wards, patients or staff schedules was seen in training. Removing the state and run-length inputs (memoryless occlusion) preserves the shadowed fraction but collapses durations to {m['dur_ms']:.0f} ms and inflates onsets to {m['onset_per_min']:.0f} per minute, so it gets the temporal structure entirely wrong. Independent per-link variates bring the co-shadowing count closer to the reference ({TF['uncoupled']['co_blk']:.2f} against {t['co_blk']:.2f}) but overestimate the shadowed fraction; the shared variate over-couples links ({u['co_blk']:.2f}). Fig. 3(a) shows that the best-receiver gain distributions of all variants lie within 0.8 dB of the reference in Wasserstein distance. The distribution of the best link therefore cannot reveal the occlusion dynamics, and these dynamics are what the scheduler reacts to.",
f"An unconstrained alternative was also tested. An autoregressive mixture-density network over dB innovations reaches a lower teacher-forced negative log-likelihood on the test wards than its single-Gaussian counterpart ({TF['nll']['mdn']:.2f} against {TF['nll']['gauss']:.2f} nats per sample). When it generates freely, however, {100 * FR['twin']['implausible']:.1f}% of link-slots leave the physically possible range of $-120$ to $-20$ dB within one minute, and {100 * FR['gauss']['implausible']:.0f}% do so for the Gaussian version. Good one-step likelihood therefore does not imply usable trajectories. Bounding every generated gain between the diffuse floor and the unobstructed value in (12) removes the drift by construction.",
    ]


SC = json.load(open('results/twin_scaling.json'))


def SCALE_TABLE():
    r = [['Training wards', 'Val. BCE', 'Shadowed (%)', 'Onsets / link-min', 'Duration (ms)', 'Co-shadowed']]
    for k in ('4', '8', '16', '32'):
        v = SC[k]
        r.append([k, f"{v['val_bce']:.4f}", f"{100 * v['blk_frac']:.2f}", f"{v['onset_per_min']:.2f}", f"{v['dur_ms']:.0f}", f"{v['co_blk']:.2f}"])
    t = TF['true']
    r.append(['Ray-traced reference', '—', f"{100 * t['blk_frac']:.2f}", f"{t['onset_per_min']:.2f}", f"{t['dur_ms']:.0f}", f"{t['co_blk']:.2f}"])
    return r


def RES_SCALE():
    return [f"Table VII shows how much ray tracing the twin needs. With four training wards (eight episodes) the occlusion head has seen too few transitions: it shadows {100 * SC['4']['blk_frac']:.0f}% of links and switches state {SC['4']['onset_per_min']:.0f} times per link-minute. Eight wards bring the onset rate to {SC['8']['onset_per_min']:.1f} per minute, and sixteen wards already match the reference within 10% on shadowed fraction, onset rate and duration ({100 * SC['16']['blk_frac']:.1f}%, {SC['16']['onset_per_min']:.1f} per minute, {SC['16']['dur_ms']:.0f} ms). Doubling to 32 wards lowers the validation cross-entropy further but does not improve these statistics, so roughly sixteen ray-traced wards suffice for this ward family. Beyond that point the twin generates new wards at almost no cost."]


def RES_MAIN():
    t = TT
    return [
f"Table VIII reports the main comparison on the 32 held-out test wards with severity drawn from held-out MIMIC-IV stays. Each entry is the mean over three seeds, which vary the severity draw, alarm arrivals and radio fading. Round robin, max-weight AoI and the DGET objective serve all patients alike and reach nearly identical unweighted and weighted AoI (about 60–64 ms). The throughput-optimal scheduler keeps polling the strongest links and starves the rest: its weighted AoI exceeds 2 s and {t['TO']['viol'][0]:.0f}% of its alarms miss the 250 ms bound. It is excluded from Fig. 4 for that reason.",
f"Severity weighting alone (SW-Whittle) lowers the weighted AoI to {f1(sw('SW-Whittle'))} ms. The price is visible in two places. The unweighted AoI rises to {f1(t['SW-Whittle']['aoi'][0])} ms because stable patients wait longer, and {f2(t['SW-Whittle']['viol'][0])}% of alarms miss $L_{{\\max}}$, six times the allowed level, because an alarm raised by a deprioritised patient waits behind high-severity traffic. LUMEN-Ward keeps most of the freshness gain, at {f1(sw('LUMEN-Ward'))} ms ({f1(red(sw('LUMEN-Ward'), sw('MW-AoI')))}% below MW-AoI and {f1(red(sw('LUMEN-Ward'), sw('DGET-obj')))}% below the DGET objective). It also brings alarm violations to {f2(t['LUMEN-Ward']['viol'][0])}% and failed optical transmissions to {f2(t['LUMEN-Ward']['fail'][0])}%, against {f2(t['MW-AoI']['fail'][0])}% for MW-AoI. Patients in the two highest NEWS2 tiers see an average age of {f1(t['LUMEN-Ward']['aoi_hi'][0])} ms, compared with {f1(t['MW-AoI']['aoi_hi'][0])} ms under MW-AoI.",
f"Fig. 4(b) shows where the alarm constraint binds. Without the alarm term the latency distribution follows the severity-weighted index and crosses $\\delta=1\\%$ near 500 ms. With it, the tail at 250 ms falls below $2\\times10^{{-3}}$. The residual floor near $10^{{-3}}$ comes from nodes that are shadowed on every receiver while the radio slot is taken, and no scheduling order can remove it. Fewer failed transmissions also save energy: LUMEN-Ward needs {t['LUMEN-Ward']['att'][0]:.3f} transmission attempts per delivered update, against {t['MW-AoI']['att'][0]:.3f} for MW-AoI.",
    ]


def TIER():
    R = pickle.load(open('results/eval_tier.pkl', 'rb'))
    import make_figs as MF
    out = {}
    for db in ('mimic', 'eicu'):
        for n in ('MW-AoI', 'DGET-obj', 'SW-Whittle', 'LUMEN w/o alarm term', 'LUMEN-Ward'):
            out[(db, n)] = MF.tier_stats(R[db], n)
    return out


TS = TIER()


def TIER_TABLE():
    r = [['Database / scheduler', 'Score 0', 'Score 1–4', 'Red 3 or 5–6', 'Score $\\ge7$']]
    for db, lab in (('mimic', 'MIMIC-IV'), ('eicu', 'eICU-CRD')):
        for n in ('MW-AoI', 'DGET-obj', 'SW-Whittle', 'LUMEN w/o alarm term', 'LUMEN-Ward'):
            ag, vi, al = TS[(db, n)]
            r.append([f"{lab}: {C.resolve(NAMES.get(n, n)).replace(' [', ' [')}"] + [f"{ag[i]:.0f} / {vi[i]:.2f}" for i in range(4)])
    return r


def RES_TIER():
    m, l, s_, na = TS[('mimic', 'MW-AoI')], TS[('mimic', 'LUMEN-Ward')], TS[('mimic', 'SW-Whittle')], TS[('mimic', 'LUMEN w/o alarm term')]
    e_m, e_l = TS[('eicu', 'MW-AoI')], TS[('eicu', 'LUMEN-Ward')]
    return [
f"Table IX and Fig. 5 break the outcome down by NEWS2 tier. Under MW-AoI every tier sees an average age of 56–62 ms. LUMEN-Ward lowers the age of the highest tier to {l[0][3]:.0f} ms ({100 * (1 - l[0][3] / m[0][3]):.0f}% below MW-AoI) and raises that of patients scoring zero to {l[0][0]:.0f} ms, still more than five orders of magnitude below the twelve-hour observation interval the guideline asks for. The alarm data show why the alarm term is needed. Without it, {s_[1][0]:.0f}% of alarms from score-0 patients and {s_[1][1]:.0f}% of alarms from score 1–4 patients miss the 250 ms bound under SW-Whittle ({na[1][0]:.0f}% and {na[1][1]:.0f}% for LUMEN without the alarm term), whereas high-tier alarms are almost never late. With the alarm term, the violation rate of every tier is at most {max(l[1]):.2f}%. The same pattern appears with eICU-CRD severity: the highest tier improves from {e_m[0][3]:.0f} to {e_l[0][3]:.0f} ms, and no tier exceeds {max(e_l[1]):.2f}% violations. A severity-weighted scheduler without an alarm constraint therefore puts its delays exactly on the patients whose deterioration is least expected.",
    ]


def RES_ABL():
    base = sw('LUMEN-Ward')
    return [
f"Table X removes one component at a time. Setting the learned residual to zero changes the weighted AoI by {100 * (sw('SW-Whittle+C+AP') / base - 1):+.1f}% and leaves the alarm and failure rates unchanged. Evolution strategies with a shared scorer did not find a better allocation than the severity-weighted Whittle index on the twin, nor on probes that varied single features by hand (Section VII). Removing the certificates lowers the weighted AoI by {100 * (1 - sw('LUMEN w/o certificates') / base):.1f}%, because uncertified links allow more concurrent transmissions, but raises failed transmissions about twentyfold to {f2(TT['LUMEN w/o certificates']['fail'][0])}%. Removing the alarm term leaves freshness unchanged and pushes violations to {f2(TT['LUMEN w/o alarm term']['viol'][0])}%. Without severity weights the weighted AoI rises by {100 * (sw('MW-AoI') / base - 1):.0f}%. The three components therefore act on different metrics: severity weighting improves freshness, certificates improve reliability and energy, and the alarm term enforces the latency constraint.",
    ]


def RES_SENS():
    def g(k, v, p, m='swaoi'):
        return A.table(RS[(k, v)], [p], seeds=(0,))[p][m][0]
    a = {(s, al): A.table(RA[(s, al)], ['SW-Whittle+C+AP'], seeds=(0,))['SW-Whittle+C+AP'] for (s, al) in RA}
    return [
f"Fig. 6 and Table XI vary one operating parameter at a time with the others at their defaults. As the reporting delay grows from 1 to 20 slots, the weighted AoI of LUMEN-Ward rises from {f1(g('D', 1, 'LUMEN-Ward'))} to {f1(g('D', 20, 'LUMEN-Ward'))} ms. Its advantage over MW-AoI stays between {min(red(g('D', v, 'LUMEN-Ward'), g('D', v, 'MW-AoI')) for v in (1, 5, 10, 20)):.0f}% and {max(red(g('D', v, 'LUMEN-Ward'), g('D', v, 'MW-AoI')) for v in (1, 5, 10, 20)):.0f}%, because the certificate margins widen with staler reports. The failure rate of LUMEN-Ward rises only from 0.09% to 0.67% over the same range, whereas uncertified MW-AoI rises from 2.4% to 12.3%, so the certificates matter more as reports become staler. Lowering the emitter power to 7.5 mW makes every scheduler slower ({f1(g('p_tx', 0.0075, 'LUMEN-Ward'))} ms for LUMEN-Ward and {f1(g('p_tx', 0.0075, 'MW-AoI'))} ms for MW-AoI). Without a radio slot every scheduler fails: the weighted AoI exceeds {f1(min(g('k_rf', 0, p) for p in ('MW-AoI', 'LUMEN-Ward')) / 1000)} s and alarm violations reach {f1(g('k_rf', 0, 'LUMEN-Ward', 'viol'))}% for LUMEN-Ward, because nodes shadowed on every receiver, typically a chest patch against the mattress, have no other path. A single radio slot is therefore essential and cannot be replaced by optical scheduling. A second radio slot brings LUMEN-Ward to {f1(g('k_rf', 2, 'LUMEN-Ward'))} ms. Across alarm rates from 0.02 to 0.5 s$^{{-1}}$ per patient, LUMEN-Ward stays below $\\delta$ ({f2(max(g('lam_alarm', v, 'LUMEN-Ward', 'viol') for v in (0.02, 0.1, 0.5)))}% at most), while SW-Whittle ranges from {f1(min(g('lam_alarm', v, 'SW-Whittle', 'viol') for v in (0.02, 0.1, 0.5)))}% to {f1(max(g('lam_alarm', v, 'SW-Whittle', 'viol') for v in (0.02, 0.1, 0.5)))}%. Tightening $L_{{\\max}}$ to 100 ms gives {f2(g('L_max', 10, 'LUMEN-Ward', 'viol'))}% violations for LUMEN-Ward, against {f2(g('L_max', 10, 'SW-Whittle', 'viol'))}% for the index without the alarm term.",
f"The certificate level $\\alpha$ sets the trade-off between freshness and reliability (Fig. 7). On the ray-traced test wards, raising $\\alpha$ from 0.005 to 0.2 lowers the weighted AoI from {f1(a[('test', 0.005)]['swaoi'][0])} to {f1(a[('test', 0.2)]['swaoi'][0])} ms and raises failed transmissions from {f2(a[('test', 0.005)]['fail'][0])}% to {f2(a[('test', 0.2)]['fail'][0])}%. The twin reproduces this curve within {max(abs(a[('twin', al)]['swaoi'][0] - a[('test', al)]['swaoi'][0]) for al in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)):.1f} ms and {max(abs(a[('twin', al)]['fail'][0] - a[('test', al)]['fail'][0]) for al in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)):.2f} percentage points at every level, so $\\alpha$ can be chosen on the twin before deployment. The memoryless twin predicts failure rates near 4% at every $\\alpha$ and a weighted AoI up to {f1(a[('memoryless', 0.005)]['swaoi'][0])} ms, so it would mislead such a choice. The twin without an occlusion process is optimistic by about 1–2 ms.",
    ]


def RES_GEN():
    e, b = T[('test', 'eicu')], T[('big', 'mimic')]
    return [
f"Table XII changes the severity source and the ward size. With eICU-CRD severity, where low NEWS2 tiers are more common (Table IV), LUMEN-Ward reaches {f1(e['LUMEN-Ward']['swaoi'][0])} ms against {f1(e['MW-AoI']['swaoi'][0])} ms for MW-AoI ({f1(red(e['LUMEN-Ward']['swaoi'][0], e['MW-AoI']['swaoi'][0]))}% lower), and alarm violations stay at {f2(e['LUMEN-Ward']['viol'][0])}%. The large wards have ten beds, 30 nodes, three to five staff and more curtain time, and none was seen in training. There the weighted AoI is {f1(b['LUMEN-Ward']['swaoi'][0])} ms against {f1(b['MW-AoI']['swaoi'][0])} ms ({f1(red(b['LUMEN-Ward']['swaoi'][0], b['MW-AoI']['swaoi'][0]))}% lower) and {f1(b['DGET-obj']['swaoi'][0])} ms for the DGET objective. Alarm violations rise to {f2(b['LUMEN-Ward']['viol'][0])}%, still below $\\delta$ but with little margin. Without the alarm term they reach {f1(b['LUMEN w/o alarm term']['viol'][0])}%. The shared scorer runs unchanged on the larger ward because it does not depend on the number of beds or receivers.",
    ]


def RES_STAT(a, b):
    d = {x[0]: x for x in a}
    return [
f"Table XIII gives paired Wilcoxon signed-rank tests over the 32 test wards, with seed-averaged weighted AoI and failure rate per ward and Holm correction over the six comparisons. LUMEN-Ward has a lower weighted AoI than round robin, MW-AoI and the DGET objective in every comparison ($p_{{\\mathrm{{Holm}}}}$ {fmt_p(d['MW-AoI'][3])} against MW-AoI, effect size $r={d['MW-AoI'][4]:.2f}$). Its weighted AoI is slightly higher than that of the uncertified variants, the price of certification, which is matched by large and significant reductions in failed transmissions. The difference from the analytic index without the residual is {d['SW-Whittle+C+AP'][1]:+.2f} ms ($p_{{\\mathrm{{Holm}}}}$ {fmt_p(d['SW-Whittle+C+AP'][3])}), too small to matter in practice. Fig. 8(a) shows that the freshness gain holds ward by ward: LUMEN-Ward is below the diagonal in every one of the 32 test wards against both equal-urgency schedulers, and the gap widens in the slower wards, where capacity is scarcest. Fig. 8(b) shows the reliability gain: per-ward failure rates of up to 10% for the comparators fall below 1% in every ward. Fig. 8(c) documents the training of the residual. The validation ratio of the centre policy never leaves the band $1\\pm0.001$, which is the negative result discussed in Section VII.",
    ]


def RES_EFF():
    return [
f"Table XIV lists the costs. Ray tracing one ward episode takes {TM['raytrace_s_per_episode']:.0f} s on one CPU core. The twin produces the same 60 s of gains for every link in {TM['geometry_s_per_episode'] + TM['twin_s_per_episode']:.1f} s including the geometry, about {TM['raytrace_s_per_episode'] / (TM['geometry_s_per_episode'] + TM['twin_s_per_episode']):.0f} times faster, so the 128 synthetic training wards take about three minutes instead of more than half an hour. The scheduler decides a whole ward slot in {TM['scheduler_ms_per_slot']:.3f} ms on a CPU, well inside the 10 ms slot. The scorer has {TM['scorer_params']} parameters, so it fits on the controller of a luminaire.",
    ]


DISC = {
"Principal Findings": [
"The results separate three effects that are usually reported together. Severity weighting changes who is served, and it explains most of the freshness gain for high-risk patients. Certification changes which links are used, and it reduces failed transmissions by more than an order of magnitude at a small cost in concurrency. The alarm term changes when stable patients are served after an alarm, and it restores the latency constraint that severity weighting breaks. None of the three replaces the others, and Table X shows that removing any one degrades exactly one metric.",
"The learned residual did not improve on the analytic index. This negative result is informative. Within the policy class of a shared scorer and greedy certificate-gated matching, the Whittle index with $p\\approx1$ already ranks nodes well, and the remaining freedom (receiver choice, interference footprint, contention) moves the weighted AoI by less than one percent. Larger gains from learning would require a richer action space, for example choosing the certificate level per node or reserving radio slots ahead of predicted shadowing.",
],
"Role of the Generative Twin": [
"The twin turned out to be most useful as a calibrated what-if model rather than only as a training source. It reproduced the freshness–reliability curve of unseen ray-traced wards across a 40-fold range of $\\alpha$, so a deployment can pick $\\alpha$ and check the alarm budget before any measurement campaign. Its physical structure matters. An unconstrained autoregressive generator with good one-step likelihood drifted out of the physical range within a minute, and a twin without occlusion memory produced the wrong failure rates. The useful prior is the split between observable geometry and unobservable occluders.",
],
"Comparison with Prior Work": [
"Single-link optical MBSN studies optimise the modulation of one wearer [@antaki25] and leave the ward-level questions of who transmits and through which receiver open; LUMEN-Ward treats every link as a certified resource and schedules across them. Hybrid LiFi–WiFi load balancing assigns users to access points on a time scale of seconds to minutes [@wu20], while the ward requires a per-slot decision because shadowing onsets last a few hundred milliseconds. The DGET formulation jointly rewards throughput and freshness [@hamrouni25]; on the test wards its objective performs within 0.2% of equal-urgency max-weight scheduling, which shows that its throughput term adds little when updates are generated at will and that the missing ingredient is urgency. Urgency of information weights the age by context [@zheng20], and LUMEN-Ward instantiates that weight with a clinical score whose response tiers are fixed by guideline rather than tuned, which keeps the scheduler auditable for clinical staff.",
],
"Deployment Considerations": [
"The scheduler needs per-link gain reports, which an infrared receiver obtains from the pilots of polled packets and from short periodic beacons; at $D=5$ the reports are 50 ms old and still yield failure rates below 0.3%. The NEWS2 weight of each patient changes at most hourly and can be pushed from the electronic health record, so its signalling cost is negligible. The scorer and the certificates are local to the ward controller and need no cloud connection. A practical system would also log certificate violations per receiver, because a persistent excess over $\\alpha$ indicates a blocked or failing photodiode rather than patient motion. Finally, the same luminaires carry the visible-light downlink, so polling and acknowledgements need no additional radio spectrum.",
],
"Threats to Validity": [
"Internal validity rests on two points. Every scheduler shares the same matching, reports and radio fallback, so the comparisons isolate priorities and admission. Severity draws, alarm arrivals and fading use common random numbers across schedulers within a seed. Construct validity is limited by the AoI abstraction. Each poll delivers a fresh sample, so the age measures timeliness, not information content, and patients whose signals change slowly gain less from fresh updates than the weights suggest. External validity is limited by the ICU origin of both clinical databases and by the simulated channels. The large-ward test and the second database probe the two directions in which a deployment is most likely to differ, and both preserved the ranking of the schedulers.",
],
"Clinical and Engineering Implications": [
"The NEWS2 tiers set the relative weights, and the absolute ages remain well below any clinical observation interval. The weighting therefore matters most when capacity is scarce: during care rounds with curtains drawn, at low emitter power or in larger wards, where Table XII shows the largest absolute gaps. Severity-weighted freshness should always be paired with an explicit alarm constraint, because a scheduler that optimises the weighted objective alone delays alarms from patients classed as stable. Such alarms are often the clinically important ones.",
],
}

LIMITS = [
"The channels are ray-traced rather than measured. The ray tracer uses first-order diffuse reflections and an integrating-sphere tail, and it does not shadow diffuse paths. In-bed patients follow a procedural posture model because public motion capture of patients in bed is not available. The clinical data come from the 100-patient MIMIC-IV demonstration and the 2,500-stay eICU-CRD demonstration, both ICU populations. Their NEWS2 scores are therefore higher than those of a general ward, and consciousness was approximated from the eye-opening score. Alarms are modelled as a Bernoulli process at a stress rate, not taken from monitor logs. The scheduler assumes one transmission per receiver per 10 ms slot and a shared radio channel with a fixed number of slots. The certificate guarantee holds for the long-run frequency on each link, not for each transmission. Finally, the learned residual was trained with one seed and one policy class, so other architectures might find gains that this study did not.",
]


def CONCL():
    return [
f"LUMEN-Ward schedules the uplink of a multi-patient hybrid optical–RF ward by NEWS2 severity, under per-link reliability certificates and an alarm-latency chance constraint, using a physics-structured generative twin to tune and train the scheduler. On held-out ray-traced wards it lowered severity-weighted AoI by {f1(red(sw('LUMEN-Ward'), sw('MW-AoI')))}% relative to equal-urgency max-weight scheduling and by {f1(red(sw('LUMEN-Ward'), sw('DGET-obj')))}% relative to the DGET objective. It also cut failed optical transmissions from about 6% to {f2(TT['LUMEN-Ward']['fail'][0])}% and kept alarm violations below 1% at stress alarm rates, in larger unseen wards and with a second clinical database. The twin reproduced the occlusion statistics and the certificate trade-off of unseen wards about {TM['raytrace_s_per_episode'] / (TM['geometry_s_per_episode'] + TM['twin_s_per_episode']):.0f} times faster than ray tracing. Three directions follow. The first is measured validation in a ward mock-up with instrumented luminaires. The second is a richer learned action space that sets certificate levels per node and reserves radio slots ahead of predicted shadowing. The third, pursued in the companion work, is integrated sensing that uses the same luminaires to observe the patients whose data they carry.",
    ]


def build():
    d = D.Doc(TEMPLATE)
    d.title(P.TITLE); d.authors(P.AUTHORS); d.affil(P.AFFIL)
    d.abstract(ABSTRACT())
    d.keywords('Age of information, hybrid optical–RF networks, optical wireless communication, medical body sensor networks, NEWS2, '
               'conformal calibration, generative channel twin, multi-agent scheduling')
    d.end_front()
    d.section('Introduction')
    for p in P.INTRO:
        d.para(C.resolve(p))
    d.subsection('Research Objectives')
    for p in P.OBJECTIVES:
        d.para(C.resolve(p))
    d.subsection('Contributions')
    for p in P.CONTRIB:
        d.para(C.resolve(p))
    d.para('Section II reviews related work. Section III states the system model and the problem, Section IV derives LUMEN-Ward, Section V describes the data and protocol, Section VI reports the results, Section VII discusses them, and Sections VIII and IX give the limitations and conclusions.')
    d.section('Related Work')
    for sub, paras in P.RELATED.items():
        d.subsection(sub[3:])
        for p in paras:
            d.para(C.resolve(p))
    d.table([[C.resolve(c) for c in r] for r in P.RW_TABLE], 'Comparison of LUMEN-Ward with the closest optical, hybrid and freshness-scheduling studies',
            widths=[0.8, 1.5, 1.0, 1.4, 1.3, 1.6], wide=True, fs=7.5)
    d.section('System Model and Problem Formulation')
    for key in 'AB':
        name, items = P.METHOD[key]
        d.subsection(name); add_items(d, items)
        if key == 'A':
            d.table(P.NOTATION, 'Principal notation', widths=[1.0, 2.6], fs=7.5)
    d.section('LUMEN-Ward')
    for key in 'CDEF':
        name, items = P.METHOD[key]
        d.subsection(name); add_items(d, items)
        if key == 'C':
            d.figure('./figs/fig1_architecture.png',
                     'Architecture of LUMEN-Ward. Offline, a generative twin fitted on ray-traced wards produces training episodes for synthetic wards, and Lagrangian evolution strategies train the shared scorer parameters $\\theta$ (dashed). Online, delayed beacon reports become certified gains, the scorer combines the severity-weighted Whittle index with the learned residual, a greedy matching commits receiver–node pairs under the certified SINR condition, and the remaining radio slots are granted.', wide=True)
        if key == 'D':
            d.algorithm(C.resolve(P.ALGO1[0]), P.ALGO1[1])
        if key == 'E':
            d.algorithm(C.resolve(P.ALGO2[0]), P.ALGO2[1])
    d.section('Data and Experimental Protocol')
    d.subsection('Ward Corpus')
    d.para(C.resolve("Wards are generated procedurally with two rows of three to five beds along the long walls, bed pitch 2.4–3.0 m, width 6.5–8.5 m and ceiling height 2.7–3.2 m. Reflectances are drawn per ward (ceiling 0.70–0.85, walls 0.50–0.75, floor 0.20–0.40). Each in-bed patient switches among supine, left-lateral, right-lateral and semi-Fowler postures with exponential dwell times of mean 45 s, and moves the arm with Ornstein–Uhlenbeck angles and occasional reaching. With probability 0.15 a patient walks in the aisle instead, following a CMU walking clip [@cmu] from subject-disjoint pools (subjects 07 and 16 for training, 08 for validation, twelve others for testing). Staff walk to a bedside, stay 8–40 s for care and draw the three curtain panels around the bed with probability 0.4. Uplink gains for every node–receiver pair are ray-traced every 10 ms with the Paper-1 tracer (line of sight with body, staff and curtain shadowing, first-order diffuse reflections from 0.4 m patches and the integrating-sphere tail). Table III summarises the four splits, and Fig. 2 shows a test ward."))
    d.table(CORPUS(), 'Ray-traced ward corpus (means over episodes unless a range is given)', widths=[1.0, 0.9, 0.5, 0.55, 0.7, 0.5, 0.65, 0.7, 0.75], wide=True, fs=7.2,
            notes=['Best gain is the median over slots of the strongest IR-receiver gain of each node. Curtain time is the fraction of the episode with at least one curtain drawn.'])
    d.figure('./figs/fig2_ward.png', 'Test ward (8 beds). (a) Top view with beds, curtains drawn at the busiest instant (thick lines), luminaires, the checkerboard subset carrying IR receivers, the RF access point and staff paths. (b) Ray-traced uplink gain of the most shadowed node towards its two strongest receivers, and one twin realisation for the same geometry (offset by $-12$ dB for legibility); the twin draws its own occlusion events, so only the statistics, not the event times, should match.', wide=True)
    d.subsection('Clinical Severity Data')
    n2 = CS['news2']
    d.para(C.resolve(f"Hourly NEWS2 trajectories are computed from the open MIMIC-IV clinical database demonstration (v2.2, {n2['mimic']['stays']} ICU stays, {n2['mimic']['hours']:,} scored hours) [@johnson23] and the eICU-CRD demonstration (v2.0.1, {n2['eicu']['stays']:,} stays, {n2['eicu']['hours']:,} hours) [@pollard18]. Integrity was checked against the published SHA-256 manifests. Respiratory rate, SpO$_2$, systolic pressure and heart rate are carried forward for up to 4 h, temperature and consciousness for up to 8 h and the oxygen device for up to 12 h. Consciousness is approximated by the Glasgow eye-opening score. MIMIC-IV stays are split in half: one half provides severity for training and the other for testing, while eICU-CRD is used only for testing. The mean NEWS2 is {n2['mimic']['news2_mean']:.1f} in MIMIC-IV and {n2['eicu']['news2_mean']:.1f} in eICU-CRD, and the high tier (7 or more) covers {100 * n2['mimic']['tier_frac']['high (>=7)']:.0f}% and {100 * n2['eicu']['tier_frac']['high (>=7)']:.0f}% of hours, so the second database is a genuine shift in severity mix."))
    tf = n2
    tiers = [('score 0', '0', '12 h', '1/12'), ('low (1-4)', '1–4', '4–6 h (5 h used)', '1/5'), ('single red 3', 'one parameter scores 3', '1 h', '1'),
             ('medium (5-6)', '5–6', '1 h', '1'), ('high (>=7)', '$\\ge7$', 'continuous (0.25 h used)', '4')]
    rows = [['NEWS2 tier', 'Total / trigger', 'Observation interval', 'Weight $w$ (h$^{-1}$)', 'MIMIC-IV hours', 'eICU-CRD hours']]
    for k, trig, iv, wv in tiers:
        rows.append([k.replace('(>=7)', '').replace('(1-4)', '').replace('(5-6)', '').strip().capitalize(), trig, iv, wv,
                     f"{100 * tf['mimic']['tier_frac'][k]:.1f}%", f"{100 * tf['eicu']['tier_frac'][k]:.1f}%"])
    d.table(rows, 'NEWS2 clinical-response tiers, severity weights and their frequency in the two databases', widths=[0.9, 1.2, 1.2, 0.8, 0.8, 0.8], wide=True, fs=7.2)
    d.subsection('Baselines and Variants')
    d.para(C.resolve("All schedulers use the same delayed reports, the same greedy interference-aware matching and the same radio fallback, so they differ only in their priorities and admission rule. Round robin serves the node scheduled least recently. MW-AoI uses the max-weight age index $a(a+1)$ with equal weights [@kadota18]. The throughput-optimal scheduler maximises the SINR margin. The DGET objective jointly rewards throughput and delivery age, $\\log(1+a)$ with equal urgency. It reproduces the target that DGET learns to imitate [@hamrouni25] and uses the same channel reports as every other scheduler. SW-Whittle applies the severity-weighted index without certificates or an alarm term, and SW-Whittle + C + AP adds certificates and a fixed alarm priority. LUMEN-Ward adds the learned residual and the alarm term of (17). Its ablations remove the certificates or the alarm term at deployment."))
    d.table(PARAMS, 'Simulation and algorithm parameters', widths=[2.2, 1.5], fs=7.2)
    d.subsection('Training Protocol')
    d.para(C.resolve("The twin is fitted on teacher-forced samples drawn from 4% of the slots of the 64 training episodes. The fit uses 3.1 million diffuse-dominated link-slots for the diffuse head and 0.82 million visible link transitions for the occlusion head, of which 4.0% are shadowed. Both heads are two-layer GELU perceptrons (32 and 64 hidden units) trained with AdamW, a cosine learning-rate schedule and batches of 4,096 for six epochs, keeping the epoch with the lowest validation loss on the 16 validation episodes. The twin then generates 128 synthetic one-minute wards from new layouts, seeds and the training motion clips. Evolution strategies use 24 antithetic candidates per iteration, eight random 15 s windows from the synthetic pool and common random numbers. The step size is 0.03 with Adam, and the multiplier step is $\\eta_\\lambda=5$. Every ten iterations the centre policy is scored on sixteen 30 s validation windows from ray-traced validation wards, and the best centre is kept. No test ward, test subject or held-out clinical stay is used in any fitting or selection step."))
    d.subsection('Metrics and Statistical Analysis')
    d.para(C.resolve("Each test configuration runs the full 60 s of all 32 test episodes (16 for the large wards) with three seeds. Reported metrics are severity-weighted AoI $\\bar{A}_w$, unweighted AoI, AoI of patients in the two highest NEWS2 tiers, the 99th percentile of the weighted age distribution, the alarm-latency violation rate $\\Pr\\{\\ell>L_{\\max}\\}$ pooled over all alarms, the share of failed optical transmissions, and transmission attempts per delivered update as an energy proxy. Paired Wilcoxon signed-rank tests on per-ward seed means compare LUMEN-Ward with each scheduler, with Holm correction and effect size $r=|Z|/\\sqrt{n}$ [@wilcoxon45, holm79]. The twin is assessed on the same test wards by shadowing fraction, onset rate, duration, co-shadowing and the Wasserstein-1 and Kolmogorov–Smirnov distances of the best-receiver gain."))
    d.section('Results')
    d.subsection('Fidelity of the Generative Twin')
    for p in RES_TWIN():
        d.para(C.resolve(p))
    d.table(TWIN_TABLE(), 'Fidelity of channel generators on the 32 held-out ray-traced test episodes', widths=[1.6, 0.8, 0.9, 0.8, 0.9, 0.8, 0.7], wide=True, fs=7.2,
            notes=['Shadowed: visible link-slots more than 3 dB below the unobstructed gain. Co-shadowed: mean number of shadowed links of a node given at least one. W$_1$ and KS refer to the best-receiver gain of each node.'])
    d.figure('./figs/fig3_twin.png', 'Twin fidelity on held-out wards. (a) CDF of the best-receiver gain per node and slot. (b) Shadowing statistics; the onset rate of the memoryless variant (287 per link-minute) is clipped.', wide=True)
    for p in RES_SCALE():
        d.para(C.resolve(p))
    d.table(SCALE_TABLE(), 'Data efficiency of the twin: occlusion statistics on the 32 test episodes when the twin is fitted on the first 4–32 training wards', widths=[1.0, 0.7, 0.75, 0.8, 0.75, 0.75], fs=7.2)
    d.subsection('Main Benchmark Comparison')
    for p in RES_MAIN():
        d.para(C.resolve(p))
    d.table([[C.resolve(c) for c in r] for r in MAIN_TABLE()], 'Scheduling on the held-out test wards with MIMIC-IV severity (mean over three seeds; ± is the seed standard deviation)',
            widths=[2.0, 0.85, 0.65, 0.75, 0.65, 0.7, 0.7, 0.75], wide=True, fs=7.2,
            notes=['Alarm violations pool about 4,200 alarms per scheduler at $\\lambda_a=0.1$ s$^{-1}$; the throughput-optimal scheduler delivers fewer alarms because starved nodes never transmit.'])
    d.figure('./figs/fig4_main.png', 'Held-out test wards, MIMIC-IV severity. (a) Severity-weighted and unweighted AoI (mean ± seed SD). (b) Complementary CDF of alarm latency; the dotted lines mark $L_{\\max}=250$ ms and $\\delta=1\\%$.', wide=True)
    for p in RES_TIER():
        d.para(C.resolve(p))
    d.table(TIER_TABLE(), 'Average AoI (ms) / alarm violations (%) by NEWS2 tier on the test wards (three seeds)', widths=[2.2, 1.0, 1.0, 1.0, 1.0], wide=True, fs=7.2,
            notes=['Tier-level violation rates pool about 160 (score 0) to 1,500 alarms per scheduler with MIMIC-IV severity and 560 to 2,000 with eICU-CRD severity.'])
    d.figure('./figs/fig8_tier.png', 'Outcome by NEWS2 tier on the test wards (MIMIC-IV severity, three seeds). (a) Average AoI. (b) Alarm-latency violations; the dotted line is $\\delta=1\\%$ and zero values are drawn at 0.02%.', wide=True)
    d.subsection('Ablation Study')
    for p in RES_ABL():
        d.para(C.resolve(p))
    d.table(ABL_TABLE(), 'Ablation of LUMEN-Ward on the held-out test wards (MIMIC-IV severity, three seeds)', widths=[2.2, 0.8, 0.7, 0.8, 0.8, 0.8], wide=True, fs=7.2)
    d.subsection('Sensitivity and Robustness Analysis')
    for p in RES_SENS():
        d.para(C.resolve(p))
    d.figure('./figs/fig5_sweep.png', 'Sensitivity on the test wards (seed 0): severity-weighted AoI (top) and alarm-latency violations (bottom, dotted line at $\\delta=1\\%$; values below 0.1% are plotted at 0.1%) against (a) the reporting delay, (b) the emitter power, (c) the number of radio slots and (d) the alarm rate.', wide=True)
    d.table(SWEEP_TABLE(), 'Robustness to operating conditions on the test wards (seed 0): SW-AoI (ms) / alarm violations (%) / failed transmissions (%); all other parameters at their defaults', widths=[1.5, 1.25, 1.25, 1.25, 1.25], wide=True, fs=7.2)
    d.figure('./figs/fig6_alpha.png', 'Freshness–reliability trade-off of the certified index policy as the certificate level $\\alpha$ varies from 0.005 (left) to 0.2 (right), evaluated on the ray-traced test wards and on 32 synthetic wards from each generator.')
    d.subsection('Generalization/External Validation')
    for p in RES_GEN():
        d.para(C.resolve(p))
    d.table([[C.resolve(c) for c in r] for r in GEN_TABLE()], 'Generalisation across severity source and ward size: SW-AoI (ms) / alarm violations (%) / failed transmissions (%)',
            widths=[2.0, 1.4, 1.4, 1.5], wide=True, fs=7.2)
    st, a, b = STAT_TABLE()
    for p in RES_STAT(a, b):
        d.para(C.resolve(p))
    d.table(st, 'Paired Wilcoxon signed-rank tests of LUMEN-Ward against each scheduler on the 32 test wards', widths=[2.2, 0.9, 0.7, 0.5, 0.9, 0.7, 0.5], wide=True, fs=7.2,
            notes=['Δ is LUMEN-Ward minus the comparator (negative favours LUMEN-Ward); pp: percentage points.'])
    d.figure('./figs/fig7_paired.png', 'Per-ward comparison on the 32 test wards (seed means). (a) Severity-weighted AoI and (b) failed optical transmissions of LUMEN-Ward against MW-AoI and the DGET objective; points below the diagonal favour LUMEN-Ward. (c) Evolution-strategy training of the residual scorer on twin episodes, as SW-AoI relative to the analytic index policy.', wide=True)
    d.subsection('Efficiency Analysis')
    for p in RES_EFF():
        d.para(C.resolve(p))
    d.table(EFF_TABLE(), 'Computational cost (two-core CPU)', widths=[2.6, 1.0], fs=7.2)
    d.section('Discussion')
    for sub, paras in DISC.items():
        d.subsection(sub)
        for p in paras:
            d.para(C.resolve(p))
    d.section('Limitations')
    for p in LIMITS:
        d.para(C.resolve(p))
    d.section('Conclusion')
    for p in CONCL():
        d.para(C.resolve(p))
    d.unnumbered('Appendix: Proof of Proposition 1')
    d.para(C.resolve("Fix a link and drop its indices. The update (10) can be written as $q_{t+1}=q_t+\\eta(v_t-\\alpha)+\\zeta_t$, where $\\zeta_t\\ge0$ is the amount added by the clip at zero, which is non-zero only in slots with $v_t=0$ and $q_t<\\eta\\alpha$. The clip at $q_{\\max}$ is inactive whenever $q_{\\max}$ exceeds the dynamic range of the gain, because then $c_t=\\hat{x}_t-q_t$ lies below every possible gain once $q_t$ approaches $q_{\\max}$ and no further violation can occur. Summing over $T$ slots gives"))
    d.eq('q_{T+1}-q_{1}=\\eta\\sum_{t=1}^{T}\\left(v_t-\\alpha\\right)+\\sum_{t=1}^{T}\\zeta_t')
    d.para(C.resolve("Because $\\zeta_t\\ge0$ and $q_{T+1}\\le q_{\\max}$, rearranging gives $\\sum_t v_t\\le\\alpha T+(q_{\\max}-q_1)/\\eta$, which is (21) after division by $T$. The argument uses only the update rule and the bounds on $q$. It does not depend on how $x_t$ or $\\hat{x}_t$ are generated, so it holds under shadowing, reporting delay and any feedback between scheduling and measurement."))
    d.unnumbered('Data Availability')
    d.para('The MIMIC-IV and eICU-CRD demonstration databases are openly available from PhysioNet under the Open Data Commons Open Database License. The CMU motion-capture clips are public. The ray-traced ward corpus, twin episodes and all evaluation outputs are regenerated deterministically by the released code from the listed seeds.')
    d.unnumbered('Code Availability')
    d.para('The complete source code (ward generation, ray tracing, NEWS2 extraction, generative twin, JAX scheduler simulator, evolution-strategy training, evaluation, statistics and figures) is provided as a supplementary archive with a run script. [Repository URL and archived release DOI to be added upon acceptance.]')
    d.unnumbered('Ethics Statement')
    d.para('Only de-identified, openly licensed demonstration subsets of clinical databases and public motion-capture recordings were used. No new human-subject data were collected.')
    d.unnumbered('Author Contributions')
    d.para('[To be completed by the authors using the CRediT taxonomy.]')
    d.unnumbered('Conflict of Interest')
    d.para('The authors declare no competing interests.')
    d.unnumbered('Acknowledgment')
    d.para('Motion-capture data were obtained from mocap.cs.cmu.edu; the database was created with funding from NSF EIA-0196217. [Funding information to be added by the authors.]')
    d.unnumbered('References')
    d.references(C.bibliography())
    d.render(OUT)
    return d


if __name__ == '__main__':
    import os
    os.makedirs('./out', exist_ok=True)
    build()
    print('refs', len(C.ORDER), 'of', len(C.REFS))
