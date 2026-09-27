"""NN A inference in numpy: preprocessing + 5-seed MLP ensemble (1024-512-256, GELU), predicting log10 seconds."""
import json
import math
from pathlib import Path

import numpy as np

from . import config

_erf = np.vectorize(math.erf, otypes=[np.float64])


def _gelu(x):  # exact GELU, matching torch.nn.GELU()
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


class RuntimeNetwork:
    def __init__(self, artifacts_dir):
        path = Path(artifacts_dir)
        meta = json.loads((path / 'nn_a.json').read_text())
        if meta['features'] != config.FEATURES:
            raise ValueError('artifact feature list does not match nnrt.config.FEATURES')
        w = np.load(path / 'nn_a.npz')
        self.median, self.keep = w['median'], w['keep'].astype(bool)
        self.mu, self.sd = w['mu'], w['sd']
        self.nets = [[(w[f's{s}_W{k}'], w[f's{s}_b{k}']) for k in range(meta['n_layers'])] for s in range(meta['n_seeds'])]

    def _inputs(self, feats, threshold):
        x = np.array([feats.get(k, np.nan) for k in config.FEATURES], dtype=np.float64)
        x = np.sign(x) * np.log1p(np.abs(x))
        x = np.where(np.isnan(x), self.median, x)
        x = (x[self.keep] - self.mu) / self.sd
        t = config.THRESHOLDS.index(threshold) if threshold in config.THRESHOLDS else int(np.argmin([abs(math.log2(threshold / c)) for c in config.THRESHOLDS]))
        onehot = np.zeros(4)
        onehot[t], onehot[3] = 1.0, t - 1
        return np.concatenate([x, onehot])

    def predict_log10(self, feats, threshold):
        x0 = self._inputs(feats, threshold)
        outs = []
        for layers in self.nets:
            x = x0
            for k, (W, b) in enumerate(layers):
                x = x @ W + b
                if k < len(layers) - 1:
                    x = _gelu(x)
            outs.append(float(x[0]))
        return float(np.mean(outs))
