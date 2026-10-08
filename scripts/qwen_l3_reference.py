"""Independent causal GQA oracle, with original sequential FP16 FMA rounding."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/cacheflex_flash_attention'))
from fa_independent_reference import gemm


def inputs(t,hq,hkv):
    def pattern(n,salt):
        i=np.arange(n,dtype=np.uint64)
        with np.errstate(over='ignore'):
            x=i+np.uint64((salt*0x9e3779b97f4a7c15)&((1<<64)-1))
            x=(x^(x>>30))*np.uint64(0xbf58476d1ce4e5b9)
            x=(x^(x>>27))*np.uint64(0x94d049bb133111eb)
            index=(((x^(x>>31))>>29)*5+salt*3)&7
        bits=np.array([0x2c00,0xac00,0x3000,0xb000,0x2800,0xa800,0x3400,0xb400],dtype=np.uint16)
        return bits[index.astype(np.intp)].view(np.float16)
    return (pattern(hq*t*64,1).reshape(hq,t,64),pattern(hkv*t*64,2).reshape(hkv,t,64),
            pattern(hkv*t*64,3).reshape(hkv,t,64))


def reference(t,bc,hq=14,hkv=2,causal=True,chunk=64,progress=None):
    q,k,v=inputs(t,hq,hkv);out=np.empty_like(q)
    for h in range(hq):
        kh=h//(hq//hkv)
        for i in range(0,t,chunk):
            qq=q[h,i:i+chunk];n=len(qq)
            m=np.full(n,-1e30,dtype=np.float32);l=np.zeros(n,dtype=np.float32)
            o=np.zeros((n,64),dtype=np.float32)
            end=min(t,((i+n+bc-1)//bc)*bc) if causal else t
            for j in range(0,end,bc):
                stop=min(j+bc,t)
                s=gemm(qq,k[kh,j:stop].T).astype(np.float32)*np.float32(.125)
                visible=np.arange(j,stop)[None,:]<=np.arange(i,i+n)[:,None]
                if causal:s=np.where(visible,s,-np.inf)
                mn=np.maximum(m,s.max(axis=1))
                with np.errstate(under='ignore'):
                    a=np.exp((m-mn).astype(np.float64)).astype(np.float32)
                    z=np.exp((s-mn[:,None]).astype(np.float64)).astype(np.float32)
                l=(a*l+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
                o=(o*a[:,None]).astype(np.float32)
                o=(o+gemm(z.astype(np.float16),v[kh,j:stop]).astype(np.float32)).astype(np.float32)
                m=mn
            out[h,i:i+n]=(o*(np.float32(1)/l)[:,None]).astype(np.float16)
        if progress:progress(h+1,hq)
    return out


def validate(path,expected,t,hq,hkv,causal,dense=False):
    a=np.fromfile(path,dtype=np.float16).reshape(hq,t,64)
    error=np.abs(a.astype(np.float64)-expected.astype(np.float64))
    row=dict(elements=a.size,nonfinite=int((~np.isfinite(a)).sum()),
             errors=int((error>2e-6+.01*np.abs(expected.astype(np.float64))).sum()),
             max_abs_error=float(error.max()),tolerance='2e-6 + 1% abs(rounded reference)')
    assert row['nonfinite']==row['errors']==0,row
    if causal:
        _,_,v=inputs(t,hq,hkv)
        # Row zero has exactly one visible key and must equal its value exactly.
        assert np.array_equal(a[:,0,:],v[np.arange(hq)//(hq//hkv),0,:]),'causal first-row identity'
        row['causal_first_row_exact']=True
    if dense:
        q,k,v=inputs(t,hq,hkv);relative=[]
        for h in range(hq):
            kh=h//(hq//hkv)
            s=q[h].astype(np.float64)@k[kh].astype(np.float64).T*.125
            if causal:s[np.triu_indices(t,1)]=-np.inf
            w=np.exp(s-s.max(axis=1)[:,None]);ref=(w@v[kh].astype(np.float64))/w.sum(axis=1)[:,None]
            relative.append(float(np.linalg.norm(a[h].astype(np.float64)-ref)/np.linalg.norm(ref)))
        row['dense_float64_relative_l2_by_head']=relative
        assert max(relative)<.01,row
    return row
