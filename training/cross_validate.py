"""Step 2: grouped 5-fold cross-validation of the NN A recipe (all thresholds of a circuit, and structural
duplicates, are held out together). Prints the official score and error statistics; writes training/cv_predictions.csv.
    python training/cross_validate.py
"""
import json

import numpy as np

from common import Preprocessor, config, design, device, load_table, predict_networks, score, train_network, SEEDS


def main():
    df = load_table()
    dev = device()
    oof = np.zeros(len(df))
    for fold in sorted(df.fold.unique()):
        te = (df.fold == fold).values
        pre = Preprocessor().fit(df.loc[~te, config.FEATURES].astype(float).values)
        Xtr, Xte = design(df[~te], pre), design(df[te], pre)
        nets = [train_network(Xtr, df.y.values[~te], s, dev) for s in SEEDS]
        oof[te] = predict_networks(nets, Xte, dev)
        print(json.dumps({'fold': int(fold), 'score': round(100 * score(oof[te], df[te]).mean(), 2)}), flush=True)
    sc = score(oof, df)
    ok = ~df.is_timeout.values
    err = oof[ok] - np.log10(df.actual_s.values[ok])
    ya = np.log10(df.actual_s.values[ok])
    report = {
        'score': round(100 * sc.mean(), 2),
        **{f'score_t{t}': round(100 * sc[df.threshold.values == t].mean(), 2) for t in config.THRESHOLDS},
        'r2_log10': round(1 - (err ** 2).sum() / ((ya - ya.mean()) ** 2).sum(), 4),
        'within_2x_pct': round(100 * (np.abs(err) <= np.log10(2)).mean(), 2),
        'within_10x_pct': round(100 * (np.abs(err) <= 1).mean(), 2),
        'timeouts_at_cap': int((oof[~ok] >= np.log10(config.CAP_SECONDS)).sum()),
        'false_timeouts': int((oof[ok] >= np.log10(config.CAP_SECONDS)).sum()),
    }
    print(json.dumps(report))
    df.assign(pred_log10=oof, row_score=sc)[['filename', 'threshold', 'fold', 'actual_s', 'is_timeout', 'pred_log10', 'row_score']] \
        .to_csv('training/cv_predictions.csv', index=False)


if __name__ == '__main__':
    main()
