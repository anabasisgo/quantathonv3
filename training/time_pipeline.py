"""Step 4: time the submission exactly as the harness calls it (featurize once, predict once per threshold),
per stage, over every circuit. Single process, like run.py. Writes training/timing.csv and prints a summary.
    python training/time_pipeline.py [--circuits training_circuits]
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import zstandard

from common import HARNESS, ROOT, config

import sys
sys.path.insert(0, str(HARNESS))
from model import RuntimeModel  # noqa: E402


def main():
    """Measure read, feature, and prediction stages in harness call order."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--circuits', default=str(ROOT / 'training_circuits'))
    args = ap.parse_args()
    t = time.perf_counter()
    model = RuntimeModel()
    load_s = time.perf_counter() - t
    rows = []
    for path in sorted(Path(args.circuits).glob('*.qasm.zst'), key=os.path.getsize):
        t0 = time.perf_counter()
        with open(path, 'rb') as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
            text = reader.read().decode('utf-8', 'replace')
        t1 = time.perf_counter()
        feats = model.featurize(text)
        t2 = time.perf_counter()
        pred_s = []
        for thr in config.THRESHOLDS:
            s = time.perf_counter()
            model.predict(feats, thr)
            pred_s.append(time.perf_counter() - s)
        rows.append({'filename': path.name, 'chars': len(text), 'read_s': t1 - t0, 'featurize_s': t2 - t1,
                     **{k[2:] + '_s': v for k, v in feats['_timings'].items()},
                     'predict_max_s': max(pred_s), 'predict_all_s': sum(pred_s),
                     'total_s': (t2 - t1) + sum(pred_s)})
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / 'timing.csv', index=False)
    cols = ['read_s', 'compact_s', 'probe_parse_s', 'bond_bound_s', 'probe_sim_s', 'featurize_s', 'predict_max_s', 'predict_all_s', 'total_s']
    summary = df[cols].describe(percentiles=[.5, .9, .99]).T[['min', 'mean', '50%', '90%', '99%', 'max']]
    pd.set_option('display.width', 200)
    print(f'model load: {load_s:.2f} s   circuits: {len(df)}')
    print(summary.round(3).to_string())
    print(f"over 10 s: {(df.total_s > 10).sum()}   over 15 s: {(df.total_s > 15).sum()}")


if __name__ == '__main__':
    main()
