"""Neural network size sweep (width x depth), 5-seed ensembles, compact + original probe features."""
import json, os, sys, time
import numpy as np, pandas as pd, torch
import models_test as M
from eval_variants import variant_cols, PRED
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEV, flush=True)

def train(X, y, width, depth, wd, seed, epochs=600, lr=3e-3, drop=0.1):
    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for i in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.GELU()] + ([torch.nn.Dropout(drop)] if i == 0 else [])
        d = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); yt = torch.tensor(y[:, None], dtype=torch.float32, device=DEV)
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(net(Xt), yt, beta=0.2); loss.backward(); opt.step(); sch.step()
    net.eval(); return net

def run(cols, tr, te, width, depth, wd):
    trI, teI = np.where(tr)[0], np.where(te)[0]
    X = M.df[cols].astype(float).values; prep = M.Prep().fit(X[trI])
    Rtr = np.hstack([prep(X[trI]), M.th_onehot(M.TIDX[trI])]); Rte = np.hstack([prep(X[teI]), M.th_onehot(M.TIDX[teI])])
    nets = [train(Rtr, M.y[trI], width, depth, wd, s) for s in range(5)]
    with torch.no_grad():
        return np.mean([n(torch.tensor(Rte, dtype=torch.float32, device=DEV)).cpu().numpy()[:, 0] for n in nets], 0), sum(p.numel() for p in nets[0].parameters())

if __name__ == '__main__':
    tag = sys.argv[2] if len(sys.argv) > 2 else 'c16b3.0'; cols = variant_cols(tag)
    for spec in sys.argv[1].split(','):
        w, d, wd = spec.split('x'); w, d, wd = int(w), int(d), float(wd)
        t0 = time.time(); oof = np.zeros(len(M.df))
        for f in sorted(M.df.structural_v2_5fold.unique()):
            te = (M.df.structural_v2_5fold == f).values; oof[te], nparam = run(cols, ~te, te, w, d, wd)
        stress = {}
        for sp in M.SPLITS:
            tr = (M.df[sp] == 'train').values; te = (M.df[sp] == 'test').values
            p = np.zeros(len(M.df)); p[te], _ = run(cols, tr, te, w, d, wd); stress[sp] = p
        os.makedirs(PRED, exist_ok=True); np.save(f'{PRED}/nn_{w}x{d}_wd{wd}_{tag}.npy', {'oof': oof, 'stress': stress}, allow_pickle=True)
        sc = M.score(oof, M.df)
        print(json.dumps({'probe': tag, 'width': w, 'depth': d, 'wd': wd, 'params': nparam, 'cv5': round(100 * sc.mean(), 2),
              **{sp.replace('leave_', 'lo_').replace('_out', ''): round(100 * M.score(stress[sp][(M.df[sp] == 'test').values], M.df[(M.df[sp] == 'test').values]).mean(), 2) for sp in M.SPLITS},
              'sec': round(time.time() - t0)}), flush=True)
