"""Shared helpers for the training scripts: paths, labels, the official score, the network recipe."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'quantathon-harness'
ARTIFACTS = HARNESS / 'artifacts'
sys.path.insert(0, str(HARNESS))

from nnrt import config  # noqa: E402

LABELS = ROOT / 'runtime-data.csv'
FEATURES_CSV = Path(__file__).resolve().parent / 'features.csv'
FOLDS_CSV = Path(__file__).resolve().parent / 'folds.csv'
CAP = config.CAP_SECONDS

# NN A recipe (chosen by grouped CV; see README)
LAYERS = (1024, 512, 256)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS, LR, WEIGHT_DECAY, DROPOUT, HUBER_BETA = 600, 3e-3, 0.01, 0.1, 0.2


def load_table():
    """One row per labelled run, joined with circuit features. Target = log10 seconds; timeouts and the one
    over-cap 'success' enter training at the cap (a lower bound). Scoring always uses the original labels."""
    runs = pd.read_csv(LABELS)
    feats = pd.read_csv(FEATURES_CSV)
    df = runs.merge(feats, on='filename', how='left', validate='many_to_one')
    ok = df.status.eq('success')
    df['is_timeout'] = ~ok
    df['actual_s'] = np.where(ok, df.duration_s, CAP)
    df['y'] = np.log10(np.minimum(df.actual_s, CAP))
    folds = pd.read_csv(FOLDS_CSV)
    return df.merge(folds, on='filename', how='left')


def score(pred_log10, df):
    """Official metric: max(0, 1 - |log10(pred/actual)|/2); on timeout rows the prediction is capped at 14400 s."""
    p = 10.0 ** np.asarray(pred_log10)
    p = np.where(df.is_timeout, np.minimum(p, CAP), p)
    return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p, 1e-9) / df.actual_s.values)) / 2.0)


class Preprocessor:
    """Signed log1p -> training-median imputation -> drop constant columns -> standardize (fit on training rows)."""

    def fit(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        med = np.nanmedian(Z, axis=0)
        self.median = np.where(np.isnan(med), 0.0, med)
        Z = np.where(np.isnan(Z), self.median, Z)
        self.keep = Z.std(axis=0) > 1e-9
        self.mu, self.sd = Z[:, self.keep].mean(axis=0), Z[:, self.keep].std(axis=0)
        return self

    def transform(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        Z = np.where(np.isnan(Z), self.median, Z)
        return (Z[:, self.keep] - self.mu) / self.sd


def threshold_inputs(thresholds):
    idx = np.array([config.THRESHOLDS.index(int(t)) for t in thresholds])
    out = np.zeros((len(idx), 4))
    out[np.arange(len(idx)), idx] = 1.0
    out[:, 3] = idx - 1
    return out


def design(df, pre):
    return np.hstack([pre.transform(df[config.FEATURES].astype(float).values), threshold_inputs(df.threshold)])


def train_network(X, y, seed, device):
    import torch
    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for i, width in enumerate(LAYERS):
        layers += [torch.nn.Linear(d, width), torch.nn.GELU()] + ([torch.nn.Dropout(DROPOUT)] if i == 0 else [])
        d = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y[:, None], dtype=torch.float32, device=device)
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(net(Xt), yt, beta=HUBER_BETA)
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def predict_networks(nets, X, device):
    import torch
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return np.mean([n(Xt).cpu().numpy()[:, 0] for n in nets], axis=0)


def device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
