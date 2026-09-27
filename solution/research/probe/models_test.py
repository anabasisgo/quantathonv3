"""Compare ET, SVR, PCA/SVD features, runtime-matrix latent targets, small MLPs and blends
on the team's frozen folds (structural_v2_5fold) and stress splits. Official challenge score."""
import json, sys, time, warnings
import numpy as np, pandas as pd, torch
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

lab = pd.read_csv('data_first/checkpoint_v3/modeling_package/outcomes_and_splits.csv')
feat = pd.read_csv('data_first/feature_v4/inference_features_v4.csv')
sets = json.load(open('modeling_v5/FEATURE_SETS.json'))
df = lab.merge(feat, on='filename', how='left', suffixes=('', '_f')).reset_index(drop=True)
y = df.train_log10_time_or_lower_bound_primary.values
TIDX = df.threshold.map({16: 0, 64: 1, 512: 2}).values
circ = df.drop_duplicates('filename').set_index('filename')

def score(p, d):
    p = 10 ** p; eff = np.where(d.score_is_timeout == 1, np.minimum(p, 14400.), p)
    return np.maximum(0, 1 - np.abs(np.log10(np.maximum(1e-9, eff) / d.score_actual_s.values)) / 2)

class Prep:  # fold-internal signed log1p -> median impute -> drop constant -> standardize
    def fit(self, X):
        X = np.sign(X) * np.log1p(np.abs(X)); med = np.nanmedian(X, 0); self.med = np.where(np.isnan(med), 0, med)
        X = np.where(np.isnan(X), self.med, X); self.keep = X.std(0) > 1e-9
        self.mu, self.sd = X[:, self.keep].mean(0), X[:, self.keep].std(0); return self
    def __call__(self, X):
        X = np.sign(X) * np.log1p(np.abs(X)); X = np.where(np.isnan(X), self.med, X)
        return (X[:, self.keep] - self.mu) / self.sd

def th_onehot(idx):
    o = np.zeros((len(idx), 4)); o[np.arange(len(idx)), idx] = 1; o[:, 3] = idx - 1; return o  # 3 one-hots + centered ordinal

# ------------------------------------------------------------------ row-level learners
def et(Xtr, ytr, Xte, seed=0):
    m = ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.5, n_jobs=2, random_state=seed)
    return m.fit(Xtr, ytr).predict(Xte)

def svr_tuned(Xtr, ytr, Xte, groups, grid=((30, .05), (100, .05), (300, .05))):
    best = None
    for C, eps in grid:
        s = []
        for a, b in GroupKFold(3).split(Xtr, ytr, groups):
            m = SVR(C=C, epsilon=eps, gamma='scale').fit(Xtr[a], ytr[a])
            s.append(np.mean(np.minimum(1, np.abs(m.predict(Xtr[b]) - ytr[b]) / 2)))
        if best is None or np.mean(s) < best[0]: best = (np.mean(s), C, eps)
    return SVR(C=best[1], epsilon=best[2], gamma='scale').fit(Xtr, ytr).predict(Xte)

def mlp_train(Xtr, Ytr, Mtr, hidden, wd, bottleneck=None, seed=0, epochs=600, lr=3e-3, drop=0.1):
    """Ytr: (n, k) targets with mask Mtr. bottleneck -> circuit latent model with linear threshold heads."""
    torch.manual_seed(seed)
    d = Xtr.shape[1]; k = Ytr.shape[1]
    layers = [torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, hidden), torch.nn.GELU()]
    if bottleneck:
        layers += [torch.nn.Linear(hidden, bottleneck), torch.nn.Linear(bottleneck, k)]  # latent z -> 3 thresholds, linear
    else:
        layers += [torch.nn.Linear(hidden, k)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xtr, dtype=torch.float32); Y = torch.tensor(np.nan_to_num(Ytr), dtype=torch.float32)
    M = torch.tensor(Mtr, dtype=torch.float32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        err = (net(X) - Y) * M
        loss = torch.nn.functional.smooth_l1_loss(err, torch.zeros_like(err), beta=0.2, reduction='sum') / M.sum()
        loss.backward(); opt.step(); sched.step()
    net.eval(); return net

def mlp_predict(nets, X):
    with torch.no_grad():
        return np.mean([n(torch.tensor(X, dtype=torch.float32)).numpy() for n in nets], 0)

MLP_GRID = [(32, 1e-3), (64, 1e-2), (128, 1e-2), (64, 1e-1)]
def mlp_tune(Xtr, Ytr, Mtr, groups, bottleneck):
    best = None
    for h, wd in MLP_GRID:
        s = []
        for a, b in GroupKFold(3).split(Xtr, Ytr, groups):
            n = mlp_train(Xtr[a], Ytr[a], Mtr[a], h, wd, bottleneck, seed=0, epochs=400)
            p = mlp_predict([n], Xtr[b]); e = np.abs(p - np.nan_to_num(Ytr[b])) * Mtr[b]
            s.append((np.minimum(1, e / 2) * Mtr[b]).sum() / Mtr[b].sum())
        if best is None or np.mean(s) < best[0]: best = (np.mean(s), h, wd)
    return best[1], best[2]

# ------------------------------------------------------------------ circuit-level helpers
def circuit_table(rows_idx):
    """Return circuit list, Y (n_c x 3) log targets with NaN, mask."""
    sub = df.iloc[rows_idx]; files = sub.filename.unique()
    pos = {f: i for i, f in enumerate(files)}
    Y = np.full((len(files), 3), np.nan)
    for i, t, v in zip(sub.filename.map(pos).values, TIDX[rows_idx], y[rows_idx]): Y[i, t] = v
    return files, Y, (~np.isnan(Y)).astype(float)

def runtime_svd(Y, k):
    """Low-rank structure of the circuit x threshold log-runtime matrix (training circuits only)."""
    mu = np.nanmean(Y, 0); Yf = np.where(np.isnan(Y), mu, Y)
    for _ in range(30):  # iterative rank-k imputation of missing cells, used only to estimate loadings
        U, S, Vt = np.linalg.svd(Yf - mu, full_matrices=False)
        R = mu + (U[:, :k] * S[:k]) @ Vt[:k]; Yf = np.where(np.isnan(Y), R, Y)
    V = Vt[:k].T  # 3 x k loadings
    Z = np.zeros((len(Y), k))
    for i in range(len(Y)):  # latent scores from observed cells only
        o = ~np.isnan(Y[i]); Z[i] = np.linalg.lstsq(V[o], Y[i, o] - mu[o], rcond=None)[0]
    return mu, V, Z

# ------------------------------------------------------------------ one model on one split
def run(model, cols, tr, te):
    trI, teI = np.where(tr)[0], np.where(te)[0]
    Xall = df[cols].astype(float).values
    prep = Prep().fit(Xall[trI]); Xtr, Xte = prep(Xall[trI]), prep(Xall[teI])
    Rtr = np.hstack([Xtr, th_onehot(TIDX[trI])]); Rte = np.hstack([Xte, th_onehot(TIDX[teI])])
    gtr = df.filename.values[trI]
    if model == 'ET':
        return et(np.nan_to_num(np.hstack([Xall[trI], th_onehot(TIDX[trI])]), nan=-999), y[trI],
                  np.nan_to_num(np.hstack([Xall[teI], th_onehot(TIDX[teI])]), nan=-999))
    if model == 'SVR':
        return svr_tuned(Rtr, y[trI], Rte, gtr)
    if model.startswith('SVD'):  # PCA/SVD of the (standardized) feature matrix, fit on training rows' circuits
        k = int(model.split('-')[1].split('+')[0]); files, _, _ = circuit_table(trI)
        C = prep(circ.loc[files, cols].astype(float).values)
        _, _, Vt = np.linalg.svd(C - C.mean(0), full_matrices=False); P = Vt[:k].T; c0 = C.mean(0)
        Ptr = np.hstack([(Xtr - c0) @ P, th_onehot(TIDX[trI])]); Pte = np.hstack([(Xte - c0) @ P, th_onehot(TIDX[teI])])
        return svr_tuned(Ptr, y[trI], Pte, gtr) if model.endswith('SVR') else et(Ptr, y[trI], Pte)
    if model.startswith('LatentMLP'):  # circuit-level MLP: features -> latent (bottleneck) -> 3 threshold heads, masked loss
        k = int(model.split('-')[1]); files, Y, Mk = circuit_table(trI)
        Ctr = prep(circ.loc[files, cols].astype(float).values)
        te_files = df.filename.values[teI]; Cte = prep(circ.loc[te_files, cols].astype(float).values)
        h, wd = mlp_tune(Ctr, Y, Mk, files, k)
        nets = [mlp_train(Ctr, Y, Mk, h, wd, k, seed=s) for s in range(5)]
        full = mlp_predict(nets, Cte); return full[np.arange(len(teI)), TIDX[teI]]
    if model.startswith('Latent'):  # runtime-matrix SVD target: predict k circuit scores, reconstruct 3 thresholds
        k = int(model.split('-')[1][0]); learner = model.split('+')[1]
        files, Y, Mk = circuit_table(trI); mu, V, Z = runtime_svd(Y, k)
        Ctr = prep(circ.loc[files, cols].astype(float).values)
        te_files = df.filename.values[teI]; Cte = prep(circ.loc[te_files, cols].astype(float).values)
        Zhat = np.zeros((len(teI), k))
        for j in range(k):
            if learner == 'SVR': Zhat[:, j] = svr_tuned(Ctr, Z[:, j], Cte, files)
            else: Zhat[:, j] = et(Ctr, Z[:, j], Cte)
        full = mu + Zhat @ V.T
        return full[np.arange(len(teI)), TIDX[teI]]
    if model == 'MLP':  # row-level small MLP, 5-seed ensemble
        Ytr = y[trI][:, None]; Mtr = np.ones_like(Ytr)
        h, wd = mlp_tune(Rtr, Ytr, Mtr, gtr, None)
        nets = [mlp_train(Rtr, Ytr, Mtr, h, wd, None, seed=s) for s in range(5)]
        return mlp_predict(nets, Rte)[:, 0]
    raise ValueError(model)

SPLITS = ['larger_distinct_sizes', 'leave_qasm3_out', 'leave_other_qasm2_out', 'leave_sycamore_like_qasm2_out']
def evaluate(model, cols):
    oof = np.zeros(len(df)); t0 = time.time()
    for f in sorted(df.structural_v2_5fold.unique()):
        te = (df.structural_v2_5fold == f).values; oof[te] = run(model, cols, ~te, te)
    stress = {}
    for sp in SPLITS:
        tr = (df[sp] == 'train').values; te = (df[sp] == 'test').values
        p = np.zeros(len(df)); p[te] = run(model, cols, tr, te); stress[sp] = p
    return oof, stress, time.time() - t0

if __name__ == '__main__':
    fs = sys.argv[1]; models = sys.argv[2].split(',')
    base = fs.split('+')[0]
    cols = [c for c in sets[base] if c in df.columns]
    if fs.endswith('+probe'):  # add truncated-MPS probe + structural bond-bound features
        pr = pd.DataFrame([json.loads(l) for l in open('probe16.jsonl')]).set_index('filename')
        PCOLS = ['probe_frac_bonds_saturated', 'probe_mean_log2_bond', 'probe_max_entropy', 'probe_log_discarded',
                 'probe_trunc_frac', 'probe_first_sat_frac', 'probe_log_flops', 'probe_frac_ops_done',
                 'span_mean', 'span_max', 'bound_max_log2', 'bound_mean_log2', 'struct_logcost_16',
                 'struct_logcost_64', 'struct_logcost_512', 'struct_frac_bonds_ge_16', 'struct_frac_bonds_ge_64',
                 'struct_frac_bonds_ge_512']
        for c in PCOLS:
            df['pb_' + c] = df.filename.map(pr[c]).astype(float).values
            circ['pb_' + c] = circ.index.map(pr[c]).astype(float).values
        cols = cols + ['pb_' + c for c in PCOLS]
    for m in models:
        oof, stress, sec = evaluate(m, cols)
        np.save(f'preds/{fs}__{m}.npy', {'oof': oof, 'stress': stress}, allow_pickle=True)
        sc = score(oof, df)
        line = {'features': fs, 'model': m, 'cv5': round(100 * sc.mean(), 2),
                **{f't{t}': round(100 * sc[df.threshold.values == t].mean(), 2) for t in (16, 64, 512)},
                **{sp: round(100 * score(stress[sp][(df[sp] == 'test').values], df[(df[sp] == 'test').values]).mean(), 2) for sp in SPLITS},
                'sec': round(sec)}
        print(json.dumps(line), flush=True)
