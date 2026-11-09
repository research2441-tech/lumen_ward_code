# LUMEN-Ward

Severity-weighted information freshness for multi-patient hybrid optical–RF hospital wards, with a
physics-structured generative channel twin and online link certificates.

All files sit in one folder; `run_all.sh` regenerates data, models, results and figures on a CPU.

| Stage | Files |
|---|---|
| Clinical severity (NEWS2 from MIMIC-IV / eICU-CRD demos) | `news2.py` |
| Ward scenes and ray tracing (beds, curtains, staff, CMU walkers) | `ward.py`, `rooms.py`, `raytrace.py`, `simulate.py`, `bvh.py`, `gen_corpus.py`, `synth.py` |
| DCO-OFDM link constants | `phy.py` |
| Generative channel twin (proposed) and AR-MDN baseline | `twin2.py`, `train_twin2.py`, `gen_twin.py`, `twin.py`, `train_twin.py`, `eval_twin.py`, `twin_scaling.py` |
| Scheduler simulator (JAX): certificates, matching, RF fallback, AoI, alarms; all baselines | `env.py`, `data.py` |
| Lagrangian evolution strategies for the shared scorer | `train_es.py` |
| Evaluation, statistics, figures | `evaluate.py`, `analyze2.py`, `corpus_stats.py`, `timing2.py`, `fig_arch.py`, `make_figs.py` |
| Manuscript build | `paper2_text.py`, `build_paper2.py`, `docbuild.py`, `cite.py`, `refs.py` |

Schedulers (`env.POL`): `rr` round robin, `mw` max-weight AoI, `to` throughput-optimal, `dget` DGET objective,
`wsw` severity-weighted Whittle, `wsw_ap` + certificates + alarm priority, `lumen` LUMEN-Ward.

Defaults: 10 ms slots, 10 mW IR emitters, 5 MHz DCO-OFDM, 4-QAM threshold 10.35 dB, one RF slot, reporting delay 5 slots,
certificate level 0.02, alarm rate 0.1/s per patient, L_max = 250 ms, delta = 1 %.

Data licences: MIMIC-IV and eICU-CRD demonstration subsets are released on PhysioNet under the ODbL; CMU motion capture is free for research use.
