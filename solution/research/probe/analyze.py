import glob, itertools, json
import numpy as np, pandas as pd
lab = pd.read_csv('data_first/checkpoint_v3/modeling_package/outcomes_and_splits.csv')
df = lab.reset_index(drop=True)
SPLITS = ['larger_distinct_sizes', 'leave_qasm3_out', 'leave_other_qasm2_out', 'leave_sycamore_like_qasm2_out']
def score(p, d):
    p = 10 ** p; eff = np.where(d.score_is_timeout == 1, np.minimum(p, 14400.), p)
    return np.maximum(0, 1 - np.abs(np.log10(np.maximum(1e-9, eff) / d.score_actual_s.values)) / 2)
P = {f.split('/')[-1][:-4]: np.load(f, allow_pickle=True).item() for f in glob.glob('preds/*.npy')}
def blend(keys):
    return {'oof': np.mean([P[k]['oof'] for k in keys], 0),
            'stress': {s: np.mean([P[k]['stress'][s] for k in keys], 0) for s in SPLITS}}
for fs in sorted({k.split('__')[0] for k in P}):
    have = {k.split('__')[1] for k in P if k.startswith(fs + '__')}
    for combo in [('SVR', 'ET'), ('MLP', 'ET'), ('MLP', 'SVR', 'ET'), ('MLP', 'LatentMLP-2'), ('MLP', 'LatentMLP-2', 'ET'), ('MLP', 'SVR')]:
        if set(combo) <= have:
            P[f'{fs}__blend[{"+".join(combo)}]'] = blend([f'{fs}__{m}' for m in combo])
rows = []
for k, v in P.items():
    sc = score(v['oof'], df)
    r = {'features': k.split('__')[0], 'model': k.split('__')[1], 'cv5': sc.mean()}
    for t in (16, 64, 512): r[f't{t}'] = sc[df.threshold.values == t].mean()
    for s in SPLITS:
        te = (df[s] == 'test').values; r[s.replace('leave_', 'lo_').replace('_out', '').replace('_distinct_sizes', '')] = score(v['stress'][s][te], df[te]).mean()
    rows.append(r)
R = pd.DataFrame(rows).sort_values(['features', 'cv5'], ascending=[True, False])
num = R.select_dtypes('number').columns; R2 = R.copy(); R2[num] = (R2[num] * 100).round(2)
pd.set_option('display.width', 250); print(R2.to_string(index=False)); R2.to_csv('model_comparison.csv', index=False)

def boot(a, b, n=2000):
    d = pd.DataFrame({'f': df.filename, 'a': score(P[a]['oof'], df), 'b': score(P[b]['oof'], df)})
    g = d.groupby('f').agg(A=('a', 'sum'), B=('b', 'sum'), N=('a', 'count'))
    rng = np.random.default_rng(0); A, B, N = g.A.values, g.B.values, g.N.values
    bs = [(A[i].sum() - B[i].sum()) / N[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return 100 * (A.sum() - B.sum()) / N.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
print()
for fs in sorted({k.split('__')[0] for k in P}):
    for m in ['SVR', 'MLP', 'LatentMLP-2', 'blend[MLP+ET]', 'blend[MLP+SVR+ET]', 'blend[MLP+LatentMLP-2+ET]']:
        a = f'{fs}__{m}'
        if a in P and f'{fs}__ET' in P:
            print('%-12s %-26s vs ET: %+.2f pts  95%% CI [%+.2f, %+.2f]' % ((fs, m) + boot(a, f'{fs}__ET')))
print()
for fs in ('compact', 'generic'):
    for m in ('ET', 'SVR', 'MLP'):
        a, b = f'{fs}+probe__{m}', f'{fs}__{m}'
        if a in P and b in P:
            print('%-8s %-4s +probe vs without: %+.2f pts  95%% CI [%+.2f, %+.2f]' % ((fs, m) + boot(a, b)))
for c in (('MLP','SVR','ET'), ('MLP','ET'), ('MLP','SVR')):
    pass
