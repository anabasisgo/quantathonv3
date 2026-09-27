"""Can we tell, before (or with only a cheap) probe, which circuits' runtime explodes with threshold?"""
import json, sys, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
lab=pd.read_csv('data_first/checkpoint_v3/modeling_package/outcomes_and_splits.csv')
feat=pd.read_csv('data_first/feature_v4/inference_features_v4.csv')
sets=json.load(open('modeling_v5/FEATURE_SETS.json'))
p16=pd.DataFrame([json.loads(l) for l in open('probe16.jsonl')]).set_index('filename')
var=pd.DataFrame([json.loads(l) for l in open('probe_variants.jsonl')]).set_index('filename') if len(sys.argv)>1 else None
w=lab.pivot_table(index='filename',columns='threshold',values='train_log10_time_or_lower_bound_primary'); w.columns=[f'y{t}' for t in w.columns]
cz=lab.pivot_table(index='filename',columns='threshold',values='train_right_censored_primary'); cz.columns=[f'cz{t}' for t in cz.columns]
c=lab.drop_duplicates('filename').set_index('filename')[['structural_v2_5fold','content_group']]
d=w.join(cz).join(c).join(feat.set_index('filename'))
d=d[d.y16.notna()&d.y512.notna()]
d=d[~((d.cz16==1)&(d.cz512==0))]  # ratio undefined if only 16 censored
d['ratio']=d.y512-d.y16; d['boom']=(d.ratio>=1).astype(int)   # >=10x slower at 512 than at 16
print('circuits',len(d),'explode (>=10x)',d.boom.sum(), d.groupby('content_group').boom.mean().round(2).to_dict())
comp=[x for x in sets['compact'] if x in d.columns]
STRUCT=['span_mean','span_max','bound_max_log2','bound_mean_log2','struct_logcost_16','struct_logcost_64','struct_logcost_512','struct_frac_bonds_ge_16','struct_frac_bonds_ge_64','struct_frac_bonds_ge_512']
PK=['frac_bonds_saturated','mean_log2_bond','max_entropy','log_discarded','trunc_frac','first_sat_frac','log_flops','frac_ops_done']
for s in STRUCT: d['st_'+s]=p16[s].reindex(d.index)
for k in PK: d['p16_'+k]=p16['probe_'+k].reindex(d.index)
FS={'compact (no probe)':comp,'compact + structural bound (no simulation)':comp+['st_'+s for s in STRUCT],
    'compact + struct + probe cap16/3s':comp+['st_'+s for s in STRUCT]+['p16_'+k for k in PK]}
if var is not None:
    for tag in ['c4b0.5','c8b0.5','c16b0.3']:
        for k in PK: d[f'{tag}_{k}']=var[f'{tag}_probe_{k}'].reindex(d.index)
        FS[f'compact + struct + probe {tag}']=comp+['st_'+s for s in STRUCT]+[f'{tag}_{k}' for k in PK]
    tt=var[[f'c4b0.5_probe_seconds',f'c8b0.5_probe_seconds',f'c16b0.3_probe_seconds']].reindex(d.index)
for name,cols in FS.items():
    X=d[cols].astype(float).fillna(-999).values; pr=np.zeros(len(d)); rr=np.zeros(len(d))
    for f in sorted(d.structural_v2_5fold.unique()):
        te=(d.structural_v2_5fold==f).values
        pr[te]=ExtraTreesClassifier(500,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=0).fit(X[~te],d.boom.values[~te]).predict_proba(X[te])[:,1]
        rr[te]=ExtraTreesRegressor(500,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=0).fit(X[~te],d.ratio.values[~te]).predict(X[te])
    e=rr-d.ratio.values
    print(f'{name:45s} AUC {roc_auc_score(d.boom,pr):.3f}  AP {average_precision_score(d.boom,pr):.3f}  acc@0.5 {((pr>=.5)==d.boom).mean():.3f}  ratio R2 {1-(e**2).sum()/((d.ratio-d.ratio.mean())**2).sum():.3f}  ratio MAE {np.abs(e).mean():.3f} log10')
