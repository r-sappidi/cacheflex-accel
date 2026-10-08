"""Model-shaped synthetic operands; ordered FP32 and float64 math references."""
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np

def gemm(a,b,library):
    lib=ctypes.CDLL(str(library));fn=lib.ordered_gemm
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    fn.argtypes=[array,array,array,ctypes.c_int,ctypes.c_int,ctypes.c_int];fn.restype=None
    a=np.ascontiguousarray(a,dtype=np.float32);b=np.ascontiguousarray(b,dtype=np.float32)
    out=np.empty((len(a),b.shape[1]),dtype=np.float32)
    fn(a,b,out,len(a),a.shape[1],b.shape[1]);return out.astype(np.float16)

def activation(gate,up,library):
    lib=ctypes.CDLL(str(library));fn=lib.custom_activation
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    fn.argtypes=[array,array,array,ctypes.c_int];fn.restype=None
    gate=np.ascontiguousarray(gate,dtype=np.float32);up=np.ascontiguousarray(up,dtype=np.float32)
    out=np.empty_like(gate);fn(gate,up,out,gate.size)
    return out.astype(np.float16)

def acl_activation(gate,library):
    """Pinned ACL SVE SWISH arithmetic, not ideal exp/reciprocal arithmetic."""
    lib=ctypes.CDLL(str(library));fn=lib.acl_activation
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    fn.argtypes=[array,array,ctypes.c_int];fn.restype=None
    gate=np.ascontiguousarray(gate,dtype=np.float32)
    # The scalar reciprocal oracle handles normal positive denominators;
    # reject out-of-scope synthetic inputs rather than approximate silently.
    assert np.isfinite(gate).all() and np.max(np.abs(gate))<8
    out=np.empty_like(gate);fn(gate,out,gate.size)
    return out.astype(np.float16)

def generate(path,t,d,f,library,seed=137):
    # Separate RNGs keep weights constant across token counts and X prefix-aligned.
    x=(np.random.default_rng(seed).normal(0,.5,(t,d))).astype(np.float16)
    rng=np.random.default_rng(seed+1)
    wg=rng.normal(0,1/np.sqrt(d),(d,f)).astype(np.float16)
    wu=rng.normal(0,1/np.sqrt(d),(d,f)).astype(np.float16)
    wd=rng.normal(0,1/np.sqrt(f),(f,d)).astype(np.float16)
    gate=gemm(x,wg,library);up=gemm(x,wu,library)
    gd=gate.astype(np.float64)
    ideal_hidden=(gd/(1+np.exp(-gd))*up.astype(np.float64)).astype(np.float16)
    hidden=activation(gate,up,library)
    assert np.all(np.abs(hidden.astype(float)-ideal_hidden.astype(float))<=2*np.abs(np.spacing(np.abs(ideal_hidden))).astype(float)+2e-7)
    y=gemm(hidden,wd,library)
    assert np.isfinite(y).all()
    with path.open('xb') as stream:
        for a in (x,wg,wu,wd,y):stream.write(a.astype('<f2').tobytes())
    np.savez(path.with_suffix('.stages.npz'),gate=gate,up=up,hidden=hidden,y=y)
    path.with_suffix('.json').write_text(json.dumps(dict(t=t,d=d,f=f,seed=seed,synthetic=True,precision='FP16 storage, sequential FP32 accumulation, scalar SVE polynomial/reciprocal emulation for fused SiLU-product; independent float64 math audit retained',sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))

def validate_acl(path,inp,t,d,f,library):
    data=np.fromfile(inp,dtype='<f2');off=0
    def take(shape):
        nonlocal off
        n=int(np.prod(shape));a=data[off:off+n].reshape(shape);off+=n;return a
    x,wg,wu,wd=[take(s) for s in ((t,d),(d,f),(d,f),(f,d))]
    def dump(name,shape):
        a=np.fromfile(path/f'{name}.bin',dtype='<f2').reshape(shape)
        assert np.isfinite(a).all();return a
    gate,up,act,hidden=[dump(n,(t,f)) for n in ('gate','up','silu','hidden')]
    y=dump('output',(t,d));checks={}
    for name,actual,expected in [('gate',gate,gemm(x,wg,library)),('up',up,gemm(x,wu,library)),('down',y,gemm(hidden,wd,library))]:
        # Different FP32 reduction orders may round at different half boundaries.
        error=np.abs(actual.astype(float)-expected.astype(float))
        bound=2*np.abs(np.spacing(np.abs(expected))).astype(float)+2e-6
        checks[name]=dict(max_abs=float(error.max()),errors=int(np.sum(error>bound)))
        assert not checks[name]['errors'],(name,checks[name])
    expected=acl_activation(gate,library)
    assert np.array_equal(act,expected),'ACL SVE SiLU arithmetic mismatch'
    ideal=(gate.astype(float)/(1+np.exp(-gate.astype(float)))).astype(np.float16)
    error=np.abs(act.astype(float)-ideal.astype(float))
    checks['silu']=dict(exact_arithmetic_match=True,checked=int(act.size),
        max_abs_vs_ideal=float(error.max()),
        max_ulp_vs_ideal=float(np.max(error/np.abs(np.spacing(np.abs(ideal))).astype(float))))
    assert np.array_equal(hidden,(act.astype(float)*up.astype(float)).astype(np.float16))
    return checks
