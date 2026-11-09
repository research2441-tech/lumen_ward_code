"""Download the open data used in the study.
* CMU motion capture (BVH conversion): https://github.com/una-dinosauria/cmu-mocap
* MIMIC-IV Clinical Database Demo v2.2 and eICU-CRD Demo v2.0.1 (ODbL) from PhysioNet.
Checksums are verified against the SHA256SUMS.txt manifests published with each database."""
import os, hashlib, urllib.request
import gen_corpus as G

BVH = 'https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data'
os.makedirs('bvh', exist_ok=True)
for c in sorted(set(G.TRAIN_CLIPS + G.VAL_CLIPS + G.TEST_CLIPS)):
    fn = f'bvh/{c}.bvh'
    if not os.path.exists(fn):
        urllib.request.urlretrieve(f'{BVH}/0{c.split("_")[0]}/{c}.bvh', fn)

DB = {'mimic': ('https://physionet.org/files/mimic-iv-demo/2.2', ['SHA256SUMS.txt', 'icu/chartevents.csv.gz', 'icu/d_items.csv.gz', 'icu/icustays.csv.gz']),
      'eicu': ('https://physionet.org/files/eicu-crd-demo/2.0.1', ['SHA256SUMS.txt', 'vitalPeriodic.csv.gz', 'vitalAperiodic.csv.gz', 'patient.csv.gz', 'nurseCharting.csv.gz'])}
for d, (base, files) in DB.items():
    for f in files:
        fn = f'{d}/{f}'
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        if not os.path.exists(fn):
            urllib.request.urlretrieve(f'{base}/{f}', fn)
    sums = open(f'{d}/SHA256SUMS.txt').read()
    for f in files[1:]:
        h = hashlib.sha256(open(f'{d}/{f}', 'rb').read()).hexdigest()
        print(d, f, 'OK' if h in sums else 'CHECKSUM MISMATCH')
