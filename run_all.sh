#!/usr/bin/env bash
# Reproduces every table and figure of the LUMEN-Ward study (CPU only, two cores: about six hours).
set -e
python3 download_data.py           # CMU clips, MIMIC-IV demo, eICU-CRD demo
python3 news2.py                   # hourly NEWS2 trajectories and tier statistics
python3 gen_corpus.py              # 128 ray-traced ward episodes (train/val/test/big) -> ./corpus
python3 corpus_stats.py
python3 synth.py 128               # geometry-only synthetic wards -> ./synth
python3 train_twin.py              # autoregressive MDN baseline twin (one-step NLL, free-run divergence)
python3 train_twin2.py             # physics-structured twin (diffuse + occlusion heads)
python3 gen_twin.py test && python3 gen_twin.py synth
python3 eval_twin.py               # twin fidelity (Table VI, Fig. 3)
python3 twin_scaling.py            # data efficiency of the twin (Table VII)
python3 train_es.py twin twin 80 0 # Lagrangian ES of the shared scorer on twin episodes
python3 evaluate.py main && python3 evaluate.py tier && python3 evaluate.py alpha && python3 evaluate.py sweep
python3 timing2.py
python3 fig_arch.py && python3 make_figs.py fig_ward fig_twin fig_main fig_tier fig_sweep fig_alpha fig_paired
python3 build_paper2.py            # optional: manuscript (needs IEEE_template.docx and pandoc)
