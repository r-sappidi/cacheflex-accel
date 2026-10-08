"""Independent rounded FP16-GEMM reference and stable partition merge."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fa_independent_reference import inputs,gemm

def merge(ms,ls,os):
    mx=np.max(ms,axis=0)
    a=np.exp((ms-mx).astype(np.float64)).astype(np.float32)
    den=np.zeros_like(mx);out=np.zeros_like(os[0])
    for p in range(len(ms)):
        den=(den+a[p]*ls[p]).astype(np.float32)
        out=(out+a[p,:,None]*os[p]).astype(np.float32)
    return (out*(np.float32(1)/den)[:,None]).astype(np.float16)

def reference(t,bc,parts,chunk=64):
    q,k,v=inputs(t,True);q=q[0];out=np.empty_like(q)
    local=t//parts
    assert t%parts==0 and local%128==0
    for i in range(0,t,chunk):
        qq=q[i:i+chunk];n=len(qq);ms=[];ls=[];os=[]
        for p in range(parts):
            m=np.full(n,-1e30,dtype=np.float32);l=np.zeros(n,dtype=np.float32)
            o=np.zeros((n,64),dtype=np.float32)
            for j in range(p*local,(p+1)*local,bc):
                stop=min(j+bc,(p+1)*local)
                s=gemm(qq,k[j:stop].T).astype(np.float32)*np.float32(.125)
                mn=np.maximum(m,s.max(axis=1))
                with np.errstate(under='ignore'):
                    a=np.exp((m-mn).astype(np.float64)).astype(np.float32)
                z=np.exp((s-mn[:,None]).astype(np.float64)).astype(np.float32)
                # Match VL16 lane sum followed by horizontal reduction,
                # while allowing the existing explicit numerical tolerance.
                l=(a*l+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
                o=(o*a[:,None]).astype(np.float32)
                o=(o+gemm(z.astype(np.float16),v[j:stop]).astype(np.float32)).astype(np.float32)
                m=mn
            ms.append(m);ls.append(l);os.append(o)
        out[i:i+n]=merge(np.array(ms),np.array(ls),np.array(os))
    return out

def validate(path,expected,t,dense=False):
    a=np.fromfile(path,dtype=np.float16).reshape(t,64)
    err=np.abs(a.astype(np.float64)-expected.astype(np.float64))
    row=dict(elements=a.size,nonfinite=int((~np.isfinite(a)).sum()),
        errors=int((err>2e-6+.01*np.abs(expected.astype(np.float64))).sum()),
        max_abs_error=float(err.max()),tolerance='2e-6 + 1% abs(rounded reference)')
    assert row['nonfinite']==row['errors']==0,row
    if dense:
        q,k,v=inputs(t,True);q=q[0].astype(np.float64)
        s=q@k.astype(np.float64).T*.125;s-=s.max(axis=1)[:,None]
        w=np.exp(s);ref=(w@v.astype(np.float64))/w.sum(axis=1)[:,None]
        rel=float(np.linalg.norm(a.astype(np.float64)-ref)/np.linalg.norm(ref))
        row['dense_float64_relative_l2']=rel
        assert rel<.01,row
    return row
