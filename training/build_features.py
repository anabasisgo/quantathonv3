"""Step 1: run the submission featurize() on every training circuit -> training/features.csv.

Uses exactly the inference code path, so training and prediction features cannot drift apart.
    python training/build_features.py [--circuits training_circuits] [--workers 2]
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import zstandard

from common import FEATURES_CSV, ROOT, config

import nnrt


def featurize_file(path):
    with open(path, 'rb') as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
        text = reader.read().decode('utf-8', 'replace')
    feats = nnrt.featurize(text)
    timings = feats.pop('_timings')
    return {'filename': Path(path).name.removesuffix('.zst'), **feats, **timings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuits', default=str(ROOT / 'training_circuits'))
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    files = sorted(Path(args.circuits).glob('*.qasm.zst'), key=os.path.getsize)
    with Pool(args.workers, maxtasksperchild=10) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(featurize_file, files), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(files):
                print(json.dumps({'done': i, 'of': len(files)}), flush=True)
    df = pd.DataFrame(rows).sort_values('filename')
    df[['filename'] + config.FEATURES + [c for c in df.columns if c.startswith('t_')]].to_csv(FEATURES_CSV, index=False)
    print(f'wrote {len(df)} rows to {FEATURES_CSV}')


if __name__ == '__main__':
    main()
