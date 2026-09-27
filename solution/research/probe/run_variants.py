import json, os, glob, time
from multiprocessing import Pool
import probe
CONFIGS=[(4,0.5),(8,0.5),(8,1.5),(16,0.3),(16,1.0),(32,3.0)]
def work(f):
    out={'filename':os.path.basename(f)}
    try:
        t0=time.perf_counter()
        with open(f,'rb') as fh: raw=fh.read()
        text=raw[:20_000_000].decode('utf-8','replace'); del raw
        P=probe.parse(text,max_ops=400_000,deadline=t0+4.0); out['parse_seconds']=time.perf_counter()-t0
        for chi,b in CONFIGS:
            r=probe.probe(P,chi=chi,budget_s=b)
            for k,v in r.items(): out[f'c{chi}b{b}_{k}']=v
    except Exception as e: out['error']=repr(e)[:200]
    return out
if __name__=='__main__':
    done={json.loads(l)['filename'] for l in open('probe_variants.jsonl')} if os.path.exists('probe_variants.jsonl') else set()
    files=[f for f in sorted(glob.glob('qasm/*.qasm'),key=os.path.getsize) if os.path.basename(f) not in done]
    with Pool(2,maxtasksperchild=10) as p, open('probe_variants.jsonl','a') as fh:
        for r in p.imap_unordered(work,files): fh.write(json.dumps(r)+'\n'); fh.flush()
