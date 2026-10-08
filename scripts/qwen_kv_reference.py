"""Independent software-merge oracle for FA-35's complementary KV ownership."""
import numpy as np
from qwen_l3_reference import inputs
from fa_independent_reference import gemm


def reference(t,bc,hq=14,hkv=2,causal=True,chunk=64):
    assert t%2048==0 and bc<=t//16 and (t//16)%bc==0
    q,k,v=inputs(t,hq,hkv);out=np.empty_like(q);half=t//16
    for h in range(hq):
        kh=h//(hq//hkv)
        for qi in range(0,t,chunk):
            qq=q[h,qi:qi+chunk];n=len(qq);partials=[]
            for core in range(8):
                m=np.full(n,-1e30,dtype=np.float32);den=np.zeros(n,dtype=np.float32);o=np.zeros((n,64),dtype=np.float32)
                for start in (core*half,(15-core)*half):
                    for j in range(start,start+half,bc):
                        if causal and j>=qi+n:continue
                        scores=gemm(qq,k[kh,j:j+bc].T).astype(np.float32)*np.float32(.125)
                        mask=np.arange(j,j+bc)[None,:]<=np.arange(qi,qi+n)[:,None]
                        if causal:scores=np.where(mask,scores,-np.inf)
                        new=np.maximum(m,scores.max(axis=1))
                        with np.errstate(under='ignore'):
                            alpha=np.exp((m-new).astype(np.float64)).astype(np.float32)
                            z=np.exp((scores-new[:,None]).astype(np.float64)).astype(np.float32)
                        den=(alpha*den+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
                        o=(o*alpha[:,None]).astype(np.float32)
                        o=(o+gemm(z.astype(np.float16),v[kh,j:j+bc]).astype(np.float32)).astype(np.float32)
                        m=new
                partials.append((m,den,o))
            maximum=np.maximum.reduce([np.where(d>0,m,np.float32(-1e30)) for m,d,o in partials])
            den=np.zeros(n,dtype=np.float32);value=np.zeros((n,64),dtype=np.float32)
            for m,d,o in partials:
                with np.errstate(under='ignore'):scale=np.exp((m-maximum).astype(np.float64)).astype(np.float32)
                scale=np.where(d>0,scale,np.float32(0))
                den=(den+scale*d).astype(np.float32)
                value=(value+(o*scale[:,None]).astype(np.float32)).astype(np.float32)
            out[h,qi:qi+n]=(value*(np.float32(1)/den)[:,None]).astype(np.float16)
    return out
