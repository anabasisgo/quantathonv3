"""Add a v5-style timeout gate (fold-trained classifier, fixed cutoff) on top of probe-feature predictions."""
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import models_test as M
from eval_variants import variant_cols
cols=variant_cols('c16b3.0')
X=np.nan_to_num(np.hstack([M.df[cols].astype(float).values, M.th_onehot(M.TIDX)]),nan=-999)
yc=(M.df.train_right_censored_primary==1).values  # 34 censored incl. anomaly
prob=np.zeros(len(M.df))
for f in sorted(M.df.structural_v2_5fold.unique()):
    te=(M.df.structural_v2_5fold==f).values
    prob[te]=ExtraTreesClassifier(500,min_samples_leaf=2,max_features=0.5,class_weight='balanced_subsample',n_jobs=2,random_state=0).fit(X[~te],yc[~te]).predict_proba(X[te])[:,1]
L=lambda k: np.load(f'preds/{k}.npy',allow_pickle=True).item()['oof']
base={'MLP':L('compact+probe__MLP'),'SVR':L('compact+probe__SVR'),'ET':L('compact+probe__ET')}
base['blend .25/.5/.25']=.25*base['SVR']+.5*base['MLP']+.25*base['ET']
cap=np.log10(14400); succ=(M.df.score_is_timeout==0).values
rows=[]
for name,p in base.items():
    for cut in (None,.25,.35,.5):
        q=p.copy() if cut is None else np.where(prob>=cut,np.maximum(p,cap),p)
        sc=M.score(q,M.df)
        rows.append({'model':name,'gate_cutoff':cut or '-', 'score':round(100*sc.mean(),2),
            'timeout_rows_score':round(100*sc[~succ].mean(),2),'timeouts_caught':int((q[~succ]>=cap).sum()),'false_timeouts':int((q[succ]>=cap).sum())})
print(pd.DataFrame(rows).to_string(index=False))
