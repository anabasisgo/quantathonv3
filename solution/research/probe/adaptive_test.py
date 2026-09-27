"""Adaptive probing: predict blow-up from cheap features first, run the probe only on circuits predicted to blow up."""
import json, sys, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import models_test as M
from eval_variants import variant_cols, base16
tag=sys.argv[1] if len(sys.argv)>1 else 'c16b3.0'; models=(sys.argv[2] if len(sys.argv)>2 else 'ET,SVR').split(',')
comp=[c for c in M.sets['compact'] if c in M.df.columns]
cols=variant_cols(tag); pcols=[c for c in cols if c.startswith('pv_')]
orig=M.df[pcols].copy()
# circuit-level blow-up label (>=10x from 16 to 512), where defined
w=M.df.pivot_table(index='filename',columns='threshold',values='train_log10_time_or_lower_bound_primary')
lab=((w[512]-w[16])>=1).where(w[512].notna())
C=M.df.drop_duplicates('filename').set_index('filename')
Xc=C[comp].astype(float).fillna(-999)
M.df['probed']=0.; cols2=cols+['probed']
secs=base16['probe_seconds'] if tag=='c16b3.0' else None
res={}
for cut in (None, .5, .2):
    for m in models:
        oof=np.zeros(len(M.df)); probed_frac=[]
        for f in sorted(M.df.structural_v2_5fold.unique()):
            te=(M.df.structural_v2_5fold==f).values
            trf=M.df.filename[~te].unique(); tef=M.df.filename[te].unique()
            if cut is None: probe_set=set(M.df.filename)
            else:
                known=[x for x in trf if not np.isnan(lab.get(x,np.nan))]
                clf=ExtraTreesClassifier(500,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=0).fit(Xc.loc[known].values,lab[known].values.astype(int))
                ptr=pd.Series(clf.predict_proba(Xc.loc[trf].values)[:,1],index=trf); pte=pd.Series(clf.predict_proba(Xc.loc[tef].values)[:,1],index=tef)
                probe_set=set(ptr.index[ptr>=cut])|set(pte.index[pte>=cut])
            mask=M.df.filename.isin(probe_set).values
            M.df[pcols]=orig.where(np.repeat(mask[:,None],len(pcols),axis=1)); M.df['probed']=mask*1.
            probed_frac.append(np.isin(tef,list(probe_set)).mean())
            oof[te]=M.run(m,cols2,~te,te)
        M.df[pcols]=orig
        sc=M.score(oof,M.df)
        print(json.dumps({'variant':tag,'model':m,'probe_only_if_p>=':cut,'frac_circuits_probed':round(float(np.mean(probed_frac)),3),'cv5':round(100*sc.mean(),2),
            **{f't{t}':round(100*sc[M.df.threshold.values==t].mean(),2) for t in (16,64,512)}}),flush=True)
