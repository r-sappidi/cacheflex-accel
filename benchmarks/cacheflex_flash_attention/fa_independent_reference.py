"""Independent NumPy online-attention reference with sequential FP16 GEMMs.

Uses mathematical exp (not the guest polynomial) and explicit FP32 online
state. Intended for d64 and small nonperiodic validation cases.
"""
import numpy as np

def inputs(t,rich=True):
    def pattern(n,salt):
        i=np.arange(n,dtype=np.uint64)
        if rich:
            with np.errstate(over='ignore'):
                x=i+np.uint64((salt*0x9e3779b97f4a7c15)&((1<<64)-1))
                x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
                x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
                i=(x^(x>>np.uint64(31)))>>np.uint64(29)
        bits=np.array([0x2c00,0xac00,0x3000,0xb000,0x2800,0xa800,0x3400,0xb400],dtype=np.uint16)
        return bits[((i*5+salt*3)&7).astype(np.intp)].view(np.float16)
    return pattern(8*t*64,1).reshape(8,t,64),pattern(t*64,2).reshape(t,64),pattern(t*64,3).reshape(t,64)

def gemm(a,b):
    # Float64 makes each product+sum exact for these FP16 operands before the
    # single FP16 rounding step; avoids float32 double rounding near ties.
    acc=np.zeros((a.shape[0],b.shape[1]),dtype=np.float16)
    for k in range(a.shape[1]):
        acc=(acc.astype(np.float64)+a[:,k:k+1].astype(np.float64)*b[k:k+1,:].astype(np.float64)).astype(np.float16)
    return acc

def reference(t,rich=True,br=64,bc=128):
    q,k,v=inputs(t,rich);out=np.empty_like(q)
    for h in range(8):
        for i in range(0,t,br):
            qq=q[h,i:i+br];n=qq.shape[0]
            m=np.full(n,-1e30,dtype=np.float32);l=np.zeros(n,dtype=np.float32);o=np.zeros((n,64),dtype=np.float32)
            for j in range(0,t,bc):
                s=gemm(qq,k[j:j+bc].T).astype(np.float32)*np.float32(.125)
                mn=np.maximum(m,np.max(s,axis=1))
                with np.errstate(under='ignore'):alpha=np.exp((m-mn).astype(np.float64)).astype(np.float32)
                z=np.exp((s-mn[:,None]).astype(np.float64)).astype(np.float32)
                l=(alpha*l+np.sum(z,axis=1,dtype=np.float32)).astype(np.float32)
                o=(o*alpha[:,None]).astype(np.float32)
                o=(o+gemm(z.astype(np.float16),v[j:j+bc]).astype(np.float32)).astype(np.float32)
                m=mn
            out[h,i:i+br]=(o*(np.float32(1)/l)[:,None]).astype(np.float16)
    return out

def check(file,t,rich=True,expected=None):
    actual=np.fromfile(file,dtype=np.float16).reshape(8,t,64);ref=reference(t,rich) if expected is None else expected
    error=np.abs(actual.astype(np.float64)-ref.astype(np.float64))
    failed=error>2e-6+.01*np.abs(ref.astype(np.float64))
    result=dict(elements=actual.size,nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
        errors=int(np.count_nonzero(failed)),max_abs_error=float(error.max()),
        relative_l2=float(np.linalg.norm(error.ravel())/max(np.linalg.norm(ref.astype(np.float64).ravel()),1e-30)),
        exact_fraction=float(np.mean(actual==ref)),tolerance='2e-6 + 1% abs(reference)',
        reference='Independent online FP32 state, sequential FP16 FMA rounding, NumPy mathematical exp')
    assert result['nonfinite']==result['errors']==0,result
    return result
