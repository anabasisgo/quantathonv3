import glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
lab=pd.read_csv('data_first/checkpoint_v3/modeling_package/outcomes_and_splits.csv')
succ=(lab.score_is_timeout==0).values; ya=np.log10(lab.score_actual_s.values)
def score(p):
    q=10**p; eff=np.where(~succ,np.minimum(q,14400.),q)
    return np.maximum(0,1-np.abs(np.log10(np.maximum(1e-9,eff)/lab.score_actual_s.values))/2)
def metrics(p):
    e=p[succ]-ya[succ]; ss=((ya[succ]-ya[succ].mean())**2).sum()
    raw_t=lab.score_actual_s.values[succ]; raw_p=10**p[succ]
    return {'score':100*score(p).mean(),
        'R2_log':1-(e**2).sum()/ss,
        'R2_seconds':1-((raw_p-raw_t)**2).sum()/((raw_t-raw_t.mean())**2).sum(),
        'spearman':spearmanr(p[succ],ya[succ])[0],
        'MAE_log10':np.abs(e).mean(),
        'median_x_err':10**np.median(np.abs(e)),
        'within_2x_%':100*(np.abs(e)<=np.log10(2)).mean(),
        'within_10x_%':100*(np.abs(e)<=1).mean(),
        'worse_100x_%':100*(np.abs(e)>2).mean(),
        'timeouts_caught':int((p[~succ]>=np.log10(14400)).sum()),
        'false_timeout':int((p[succ]>=np.log10(14400)).sum())}
KEYS=['compact__ET','compact__SVR','compact__MLP','compact+probe__ET','compact+probe__SVR','compact+probe__MLP','generic+probe__MLP']
P={k:np.load(f'preds/{k}.npy',allow_pickle=True).item()['oof'] for k in KEYS}
P['compact+probe blend .25SVR/.5MLP/.25ET']=.25*P['compact+probe__SVR']+.5*P['compact+probe__MLP']+.25*P['compact+probe__ET']
P['compact+probe blend .5SVR/.5MLP']=.5*P['compact+probe__SVR']+.5*P['compact+probe__MLP']
R=pd.DataFrame({k:metrics(v) for k,v in P.items()}).T
pd.set_option('display.width',250); print(R.round(3).to_string())
print('\nsuccess rows',succ.sum(),'timeout rows',(~succ).sum())
# per threshold R2_log for best
b=P['compact+probe blend .25SVR/.5MLP/.25ET']
for t in (16,64,512):
    m=succ&(lab.threshold.values==t); e=b[m]-ya[m]; print('blend R2_log @',t, round(1-(e**2).sum()/((ya[m]-ya[m].mean())**2).sum(),3))
R.to_csv('metrics_table.csv')
