import sys, json, glob, os, time, traceback
from multiprocessing import Pool
import probe
chi=int(sys.argv[1]); out=sys.argv[2]
def work(f):
    try:
        r=probe.featurize_file(f, chi=chi, parse_budget=4.0, probe_budget=3.0)
    except Exception as e:
        r={'error':repr(e)[:200]}
    r['filename']=os.path.basename(f); return r
if __name__=='__main__':
    files=sorted(glob.glob('qasm/*.qasm'), key=os.path.getsize)
    with Pool(2, maxtasksperchild=20) as p, open(out,'w') as fh:
        for r in p.imap_unordered(work, files):
            fh.write(json.dumps(r)+'\n'); fh.flush()
