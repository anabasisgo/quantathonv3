import json, numpy as np, pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
lab=pd.read_csv('data_first/checkpoint_v3/modeling_package/outcomes_and_splits.csv')
feat=pd.read_csv('data_first/feature_v4/inference_features_v4.csv')
sets=json.load(open('modeling_v5/FEATURE_SETS.json'))
df=lab.merge(feat,on='filename',how='left',suffixes=('','_f'))
df['t_log2']=np.log2(df.threshold)
for t in (16,64,512): df[f't{t}']=(df.threshold==t)*1.
TH=['t_log2','t16','t64','t512']
y=df.train_log10_time_or_lower_bound_primary.values
def score(p,d):
    p=10**p; eff=np.where(d.score_is_timeout==1,np.minimum(p,14400.),p)
    return np.maximum(0,1-np.abs(np.log10(np.maximum(1e-9,eff)/d.score_actual_s.values))/2)
def prep(Xtr,Xte):  # fold-internal: signed log1p, median impute, drop constant, standardize
    f=lambda X: np.sign(X)*np.log1p(np.abs(X))
    Xtr,Xte=f(Xtr),f(Xte); med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xte=np.where(np.isnan(Xte),med,Xte)
    keep=Xtr.std(0)>1e-9; mu,sd=Xtr[:,keep].mean(0),Xtr[:,keep].std(0)
    return (Xtr[:,keep]-mu)/sd,(Xte[:,keep]-mu)/sd
GRID=[(C,eps,g) for C in (1,10,100) for eps in (0.05,0.15) for g in ('scale',0.3)]
def svr_fit_pred(Xtr,ytr,Xte,gtr,tune=True):
    best=(None,-1)
    if tune:
        for C,eps,g in GRID:
            s=[]
            for a,b in GroupKFold(3).split(Xtr,ytr,gtr):
                A,B=prep(Xtr[a],Xtr[b]); m=SVR(C=C,epsilon=eps,gamma=g).fit(A,ytr[a])
                s.append(-np.mean(np.minimum(1,np.abs(m.predict(B)-ytr[b])/2)))
            if np.mean(s)>best[1]: best=((C,eps,g),np.mean(s))
    C,eps,g=best[0]
    A,B=prep(Xtr,Xte); m=SVR(C=C,epsilon=eps,gamma=g).fit(A,ytr); return m.predict(B),best[0],m.n_support_.sum()
def et_fit_pred(Xtr,ytr,Xte):
    m=ExtraTreesRegressor(400,min_samples_leaf=2,max_features=0.5,n_jobs=1,random_state=0)
    return m.fit(np.nan_to_num(Xtr,nan=-999),ytr).predict(np.nan_to_num(Xte,nan=-999))
rows=[]
for fs in ('compact','generic','full'):
    cols=[c for c in sets[fs] if c in df.columns]+TH
    X=df[cols].astype(float).values; groups=df.filename.values
    for model in ('ExtraTrees','SVR-RBF'):
        r={'features':f'{fs} ({len(cols)-4})','model':model}
        pred=np.zeros(len(df)); params=[]
        for f in sorted(df.structural_v2_5fold.unique()):
            te=(df.structural_v2_5fold==f).values; tr=~te
            if model=='ExtraTrees': pred[te]=et_fit_pred(X[tr],y[tr],X[te])
            else:
                p,bp,nsv=svr_fit_pred(X[tr],y[tr],X[te],groups[tr]); pred[te]=p; params.append(bp)
        sc=score(pred,df); r['cv5']=sc.mean()
        for t in (16,64,512): r[f't{t}']=sc[df.threshold.values==t].mean()
        for sp in ('larger_distinct_sizes','leave_qasm3_out','leave_other_qasm2_out','leave_sycamore_like_qasm2_out'):
            tr=(df[sp]=='train').values; te=(df[sp]=='test').values
            p=et_fit_pred(X[tr],y[tr],X[te]) if model=='ExtraTrees' else svr_fit_pred(X[tr],y[tr],X[te],groups[tr])[0]
            r[sp.replace('leave_','lo_').replace('_out','').replace('_distinct_sizes','')]=score(p,df[te]).mean()
        if params: r['svr_params']=str(pd.Series([str(x) for x in params]).mode()[0])
        rows.append(r); print(r,flush=True)
out=pd.DataFrame(rows); out.to_csv('svm_vs_trees.csv',index=False)
num=out.select_dtypes('number').columns; out[num]=(out[num]*100).round(2)
pd.set_option('display.width',250); print(out.to_string(index=False))
