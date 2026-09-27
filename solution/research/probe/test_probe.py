import numpy as np, math, probe
rng=np.random.default_rng(0)
def sv_apply(psi,U,qs,n):
    psi=psi.reshape([2]*n); k=len(qs)
    psi=np.moveaxis(psi,qs,list(range(k))); sh=psi.shape
    psi=(U@psi.reshape(2**k,-1)).reshape(sh); psi=np.moveaxis(psi,list(range(k)),qs)
    return psi.reshape(-1)
def mps_state(m):
    v=m.A[0]
    for a in m.A[1:]: v=np.einsum('...x,xby->...by',v,a)
    return v.reshape(-1)
names1=['h','x','s','t','sx','rx','ry','rz','u3','u2']; names2=['cx','cz','cp','rzz','rxx','swap','iswap','crx','cu3','ecr','xx_plus_yy','dcx','ch']
worst=0
for trial in range(30):
    n=rng.integers(3,8); qasm=[f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];']
    for _ in range(40):
        if rng.random()<0.5:
            g=rng.choice(names1); a=int(rng.integers(n))
            np_={'rx':1,'ry':1,'rz':1,'u3':3,'u2':2}.get(g,0)
            ps='('+','.join(f'{rng.uniform(-3,3):.4f}' for _ in range(np_))+')' if np_ else ''
            qasm.append(f'{g}{ps} q[{a}];')
        else:
            g=rng.choice(names2); a,b=rng.choice(n,2,replace=False)
            np_={'cp':1,'rzz':1,'rxx':1,'crx':1,'cu3':3,'xx_plus_yy':2}.get(g,0)
            ps='('+','.join(f'{rng.uniform(-3,3):.4f}' for _ in range(np_))+')' if np_ else ''
            qasm.append(f'{g}{ps} q[{a}],q[{b}];')
    if trial%3==0: qasm.append(f'ccx q[0],q[1],q[2];')
    P=probe.parse('\n'.join(qasm))
    m=probe.MPS(n,chi=64)
    psi=np.zeros(2**n,complex); psi[0]=1
    for name,qs,params in P.ops:
        f=probe.ONEQ.get(name) if len(qs)==1 else probe.TWOQ.get(name)
        U=f(*params)
        psi=sv_apply(psi,U,list(qs),n)
        if len(qs)==1: m.one(U,qs[0])
        else: m.two(U,qs[0],qs[1])
    v=mps_state(m)
    fid=abs(np.vdot(psi,v))**2
    worst=max(worst,1-fid)
print('worst infidelity vs statevector over 30 random circuits:',worst)
# ccx truth table check via decomposition
P=probe.parse('OPENQASM 2.0;\nqreg q[3];\nx q[0];\nx q[1];\nccx q[0],q[1],q[2];')
psi=np.zeros(8,complex);psi[0]=1
for name,qs,params in P.ops:
    f=probe.ONEQ.get(name) if len(qs)==1 else probe.TWOQ.get(name); psi=sv_apply(psi,f(*params),list(qs),3)
print('ccx |110> ->', np.argmax(abs(psi)), 'expect index 7, prob',round(abs(psi[7])**2,6))
