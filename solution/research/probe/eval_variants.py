"""Score probe variants (bond cap / time budget) and a structural-only variant with ET and SVR (+MLP optional)."""
import json, sys
import numpy as np, pandas as pd
import os
import models_test as M
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = os.path.join(HERE, 'preds')
def _p(name): return os.path.join(HERE, name)

PROBE_KEYS = ['frac_bonds_saturated', 'mean_log2_bond', 'max_entropy', 'log_discarded', 'trunc_frac',
              'first_sat_frac', 'log_flops', 'frac_ops_done']
STRUCT = ['span_mean', 'span_max', 'bound_max_log2', 'bound_mean_log2', 'struct_logcost_16', 'struct_logcost_64',
          'struct_logcost_512', 'struct_frac_bonds_ge_16', 'struct_frac_bonds_ge_64', 'struct_frac_bonds_ge_512']
base16 = pd.DataFrame([json.loads(l) for l in open(_p('probe16.jsonl'))]).set_index('filename')
var = pd.DataFrame([json.loads(l) for l in open(_p('probe_variants.jsonl'))]).set_index('filename')

def add(name, series):
    M.df[name] = M.df.filename.map(series).astype(float).values
    M.circ[name] = M.circ.index.map(series).astype(float).values
    return name

def variant_cols(tag):
    cols = [c for c in M.sets['compact'] if c in M.df.columns]
    if tag == 'none':
        return cols
    cols += [add('st_' + c, base16[c]) for c in STRUCT]
    if tag == 'struct_only':
        return cols
    if tag == 'c16b3.0':  # original run
        return cols + [add('pv_c16b3_' + k, base16['probe_' + k]) for k in PROBE_KEYS]
    if '+' in tag:  # combine two variants
        out = cols
        for t in tag.split('+'):
            out = out + [add(f'pv_{t}_{k}', var[f'{t}_probe_{k}']) for k in PROBE_KEYS]
        return out
    return cols + [add(f'pv_{tag}_{k}', var[f'{tag}_probe_{k}']) for k in PROBE_KEYS]

if __name__ == '__main__':
    tags = sys.argv[1].split(','); models = sys.argv[2].split(',')
    for tag in tags:
        cols = variant_cols(tag)
        for m in models:
            oof, stress, sec = M.evaluate(m, cols)
            os.makedirs(PRED, exist_ok=True); np.save(f'{PRED}/var_{tag}__{m}.npy', {'oof': oof, 'stress': stress}, allow_pickle=True)
            sc = M.score(oof, M.df)
            line = {'variant': tag, 'model': m, 'cv5': round(100 * sc.mean(), 2),
                    **{f't{t}': round(100 * sc[M.df.threshold.values == t].mean(), 2) for t in (16, 64, 512)},
                    **{sp.replace('leave_', 'lo_').replace('_out', ''): round(100 * M.score(stress[sp][(M.df[sp] == 'test').values], M.df[(M.df[sp] == 'test').values]).mean(), 2) for sp in M.SPLITS},
                    'sec': round(sec)}
            print(json.dumps(line), flush=True)
