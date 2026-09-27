"""Step 3: train the final NN A on all 532 circuits and export numpy weights for the harness.
Writes quantathon-harness/artifacts/nn_a.npz (preprocessing + 5 networks) and nn_a.json (metadata).
    python training/train.py
"""
import json

import numpy as np

from common import (ARTIFACTS, EPOCHS, LAYERS, LR, SEEDS, WEIGHT_DECAY, Preprocessor, config, design, device,
                    load_table, predict_networks, train_network)


def main():
    """Fit the final ensemble, export artifacts, and check NumPy parity."""
    df = load_table()
    dev = device()
    pre = Preprocessor().fit(df[config.FEATURES].astype(float).values)
    X = design(df, pre)
    nets = [train_network(X, df.y.values, s, dev) for s in SEEDS]
    arrays = {'median': pre.median, 'keep': pre.keep.astype(np.uint8), 'mu': pre.mu, 'sd': pre.sd}
    for s, net in enumerate(nets):
        linears = [m for m in net if m.__class__.__name__ == 'Linear']
        for k, lin in enumerate(linears):
            arrays[f's{s}_W{k}'] = lin.weight.detach().cpu().numpy().T.astype(np.float32)
            arrays[f's{s}_b{k}'] = lin.bias.detach().cpu().numpy().astype(np.float32)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARTIFACTS / 'nn_a.npz', **arrays)
    meta = {'model': 'NN A', 'layers': list(LAYERS), 'n_layers': len(LAYERS) + 1, 'n_seeds': len(SEEDS),
            'epochs': EPOCHS, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'target': 'log10 seconds (timeouts at 14400 s)',
            'features': config.FEATURES, 'train_rows': int(len(df)), 'train_circuits': int(df.filename.nunique())}
    (ARTIFACTS / 'nn_a.json').write_text(json.dumps(meta, indent=1))
    # parity: the numpy inference used by the harness must reproduce the torch ensemble
    from nnrt import RuntimeNetwork
    ref = predict_networks(nets, X, dev)
    net = RuntimeNetwork(ARTIFACTS)
    rows = df.sample(200, random_state=0).index
    got = np.array([net.predict_log10(df.loc[i, config.FEATURES].to_dict(), int(df.loc[i, 'threshold'])) for i in rows])
    gap = float(np.abs(got - ref[rows]).max())
    print(f'saved {len(nets)} networks to {ARTIFACTS}; numpy vs torch max |diff| = {gap:.2e} log10 units')
    assert gap < 1e-4, 'numpy inference does not match the trained networks'


if __name__ == '__main__':
    main()
