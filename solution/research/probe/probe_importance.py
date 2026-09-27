"""(1) Can probe features replace handcrafted ones? (2) Grouped permutation importance of each probe feature."""
import json, sys, numpy as np, pandas as pd
import models_test as M
from eval_variants import add, base16, var, PROBE_KEYS, STRUCT
mode=sys.argv[1]
TAG='c16b1.0'
P=[add(f'pv_{TAG}_{k}', var[f'{TAG}_probe_{k}']) for k in PROBE_KEYS]
S=[add('st_'+c, base16[c]) for c in STRUCT]
comp=[c for c in M.sets['compact'] if c in M.df.columns]; full=[c for c in M.sets['full'] if c in M.df.columns]
def report(name,m,cols):
    oof,stress,sec=M.evaluate(m,cols); sc=M.score(oof,M.df)
    np.save(f'preds/imp_{name}__{m}.npy',{'oof':oof,'stress':stress},allow_pickle=True)
    print(json.dumps({'features':name,'n':len(cols),'model':m,'cv5':round(100*sc.mean(),2),
        **{s.replace('leave_','lo_').replace('_out',''):round(100*M.score(stress[s][(M.df[s]=='test').values],M.df[(M.df[s]=='test').values]).mean(),2) for s in M.SPLITS}}),flush=True)
if mode=='replace':
    for m in sys.argv[2].split(','):
        for name,cols in [('qubits+probe',['n_qubits']+P),('qubits+2q+probe',['n_qubits','twoq_count']+P),
                          ('qubits+2q+depth+probe',['n_qubits','twoq_count','gate_depth']+P),
                          ('compact+probe(c16b1)',comp+S+P),('full325',full),('full325+probe(c16b1)',full+S+P)]:
            report(name,m,cols)
if mode=='perm':
    from sklearn.ensemble import ExtraTreesRegressor
    cols=comp+S+P; X=np.nan_to_num(np.hstack([M.df[cols].astype(float).values,M.th_onehot(M.TIDX)]),nan=-999)
    files=M.df.filename.values; rng=np.random.default_rng(0)
    groups={**{c:[c] for c in P+S},'ALL probe (simulation)':P,'ALL structural bound':S}
    loss={g:[] for g in groups}; base=[]
    for f in sorted(M.df.structural_v2_5fold.unique()):
        te=(M.df.structural_v2_5fold==f).values
        mdl=ExtraTreesRegressor(400,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=0).fit(X[~te],M.y[~te])
        b=M.score(mdl.predict(X[te]),M.df[te]).mean(); base.append(b)
        uf=np.unique(files[te])
        for g,gc in groups.items():
            idx=[cols.index(c) for c in gc]; ls=[]
            for r in range(5):  # permute whole circuits (same donor for all thresholds and all group columns)
                donor=dict(zip(uf,rng.permutation(uf))); Xp=X[te].copy()
                src={fn:i for i,fn in zip(np.where(te)[0],files[te])}
                rows=np.array([src[donor[fn]] for fn in files[te]])
                Xp[:,idx]=X[rows][:,idx]
                ls.append(b-M.score(mdl.predict(Xp),M.df[te]).mean())
            loss[g].append(np.mean(ls))
    out=pd.Series({g:100*np.mean(v) for g,v in loss.items()}).sort_values(ascending=False)
    print('base cv',round(100*np.mean(base),2)); print(out.round(3).to_string())
