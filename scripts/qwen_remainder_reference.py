"""Independent full-output NumPy oracle for AI-12 (no guest code reused)."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
from build_access_timing import save,sha

OPS=('rmsnorm','residual','rope','rope_tables','embedding','final_norm','logits')
D=896
V=151936

def tables(t):
    freq=(1/np.power(np.float32(1000000),np.arange(32,dtype=np.float32)/32)).astype(np.float32)
    angle=(np.arange(t,dtype=np.float32)[:,None]*freq).astype(np.float32)
    return np.concatenate((np.cos(angle.astype(np.float64)),np.sin(angle.astype(np.float64))),axis=1),freq

def generate(path,op,t,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(912+OPS.index(op));empty=np.empty(0,dtype=np.float16)
    x=w=z=empty;ids=np.empty(0,dtype=np.uint32)
    if op in ('rmsnorm','final_norm'):
        x=rng.normal(0,.5,(t,D)).astype(np.float16);w=rng.uniform(.8,1.2,D).astype(np.float16)
        # Directed zero/low-magnitude rows test epsilon; real target retains them.
        x[0]=0
        if t>1:x[1]=np.float16(1e-4)
        norm=(x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2,axis=1,keepdims=True)+1e-6)).astype(np.float16)
        dense=norm.astype(np.float64)*w.astype(np.float64)
    elif op=='residual':
        x=rng.normal(0,.5,(t,D)).astype(np.float16);w=rng.normal(0,.5,(t,D)).astype(np.float16)
        dense=x.astype(np.float64)+w.astype(np.float64)
    elif op=='rope':
        x=rng.normal(0,.5,(16,t,64)).astype(np.float16);z=tables(t)[0].astype(np.float16)
        a=x[:,:,:32].astype(np.float64);b=x[:,:,32:].astype(np.float64)
        c=z[:,:32].astype(np.float64);s=z[:,32:].astype(np.float64)
        # Eager FP16 products round before addition, as in the pinned model.
        ac=(a*c).astype(np.float16).astype(float);bs=(b*s).astype(np.float16).astype(float)
        bc=(b*c).astype(np.float16).astype(float);as_=(a*s).astype(np.float16).astype(float)
        dense=np.concatenate((ac-bs,bc+as_),axis=2)
    elif op=='rope_tables':
        dense,freq=tables(t);ids=freq.view(np.uint32)
    elif op=='embedding':
        # Native tied embedding layout [vocabulary, hidden]. No free transpose.
        x=np.empty((v,D),dtype=np.float16)
        for lo in range(0,v,1024):x[lo:lo+1024]=rng.normal(0,.02,(min(1024,v-lo),D)).astype(np.float16)
        ids=rng.integers(0,v,t,dtype=np.uint32);ids[0]=0
        if t>1:ids[-1]=v-1
        if t>2:ids[1]=ids[0]
        dense=x[ids].astype(np.float64)
    else:
        x=rng.normal(0,.5,D).astype(np.float16);w=np.empty((v,D),dtype=np.float16);dense=np.empty(v,dtype=np.float64)
        for lo in range(0,v,1024):
            part=rng.normal(0,.02,(min(1024,v-lo),D)).astype(np.float16);w[lo:lo+1024]=part
            dense[lo:lo+len(part)]=part.astype(np.float64)@x.astype(np.float64)
    expected=dense.astype(np.float16).reshape(-1)
    with path.open('xb') as f:
        f.write(np.array([x.size,w.size,z.size,ids.size,expected.size],dtype='<u8').tobytes())
        for a in (x,w,z):f.write(a.astype('<f2',copy=False).tobytes())
        f.write(ids.astype('<u4',copy=False).tobytes())
    np.savez(path.with_suffix('.npz'),expected=expected,dense=dense.reshape(-1))
    save(path.with_suffix('.json'),dict(op=op,t=t,v=v,elements=expected.size,sha256=sha(path),reference_sha256=sha(path.with_suffix('.npz')),seed=912+OPS.index(op),synthetic=True))

def validate(path,inp,op):
    a=np.fromfile(path,dtype='<f2')
    with np.load(Path(inp).with_suffix('.npz')) as data:expected=data['expected'];dense=data['dense']
    assert a.shape==expected.shape and np.isfinite(a).all(),'size/nonfinite'
    error=np.abs(a.astype(float)-dense)
    # FP32 reductions may move a result across one FP16 rounding midpoint.
    # RMSNorm has two FP16 roundings; allow two ULPs plus the FP32 floor.
    floor=2e-5 if op=='logits' else 2e-6
    bound=floor+(0.002 if op in ('rmsnorm','final_norm') else 0.001)*np.abs(dense)
    if op in ('residual','rope','embedding'):
        assert np.array_equal(a.view(np.uint16),expected.view(np.uint16)),'exact FP16 arithmetic/layout mismatch'
    assert np.all(error<=bound),dict(errors=int((error>bound).sum()),max_abs=float(error.max()))
    return dict(elements=a.size,full_oracle=True,exact_oracle_bits=bool(np.array_equal(a.view(np.uint16),expected.view(np.uint16))),max_abs=float(error.max()),relative_l2=float(np.linalg.norm(a.astype(float)-dense)/max(1e-30,np.linalg.norm(dense))),bound=f'{floor} + {0.002 if op in ("rmsnorm","final_norm") else 0.001} * abs(reference)')
