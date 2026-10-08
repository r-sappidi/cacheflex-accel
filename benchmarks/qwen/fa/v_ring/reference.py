"""Independent rounded-FMA oracle with the actual cyclic panel visitation order."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
from qwen_l3_reference import inputs,validate
from fa_independent_reference import gemm

def panels(t,bc,core):
    assert t%(8*bc)==0
    return [wave+((core-step)%8)*bc for wave in range(0,t,8*bc) for step in range(8)]

def reference(t,bc,hq=14,hkv=2,causal=True):
    q,k,v=inputs(t,hq,hkv);out=np.empty_like(q);rows=128 if t==1024 else 256
    groups=t//rows
    for h in range(hq):
        kh=h//(hq//hkv)
        for group in range(groups):
            core=(group-h)%groups%8
            base=group*rows
            for i in range(base,base+rows,16):
                    m=np.full(16,-1e30,np.float32);l=np.zeros(16,np.float32);o=np.zeros((16,64),np.float32)
                    for j in panels(t,bc,core):
                        if causal and j>=i+16:continue
                        score=gemm(q[h,i:i+16],k[kh,j:j+bc].T).astype(np.float32)*np.float32(.125)
                        if causal:score=np.where(np.arange(j,j+bc)[None,:]<=np.arange(i,i+16)[:,None],score,-np.inf)
                        mn=np.maximum(m,score.max(axis=1))
                        a=np.exp((m-mn).astype(np.float64)).astype(np.float32)
                        z=np.exp((score-mn[:,None]).astype(np.float64)).astype(np.float32)
                        l=(a*l+z.sum(axis=1,dtype=np.float32)).astype(np.float32)
                        o=(o*a[:,None]).astype(np.float32)
                        o=(o+gemm(z.astype(np.float16),v[kh,j:j+bc]).astype(np.float32)).astype(np.float32)
                        m=mn
                    out[h,i:i+16]=(o*(np.float32(1)/l)[:,None]).astype(np.float16)
    return out
