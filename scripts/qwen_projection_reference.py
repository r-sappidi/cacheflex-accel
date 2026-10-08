"""Independent ordered FP32 and float64 references, with explicit FA layouts."""
import ctypes,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pathlib import Path
from build_access_timing import save,sha

def ordered(a,b,library):
 lib=ctypes.CDLL(str(library));fn=lib.ordered_gemm
 arr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');fn.argtypes=[arr,arr,arr,ctypes.c_int,ctypes.c_int,ctypes.c_int];fn.restype=None
 a=np.ascontiguousarray(a,dtype=np.float32);b=np.ascontiguousarray(b,dtype=np.float32);out=np.empty((len(a),b.shape[1]),dtype=np.float32)
 fn(a,b,out,len(a),a.shape[1],b.shape[1]);return out

def head_layout(row):return row.reshape(len(row),-1,64).transpose(1,0,2).copy().reshape(-1)
def row_layout(head,t):return head.reshape(-1,t,64).transpose(1,0,2).copy().reshape(t,-1)

def generate(path,t,op,library):
 n=1152 if op=='qkv' else 896;seed=213 if op=='qkv' else 217
 x=np.random.default_rng(seed).normal(0,.5,(t,896)).astype(np.float16)
 rng=np.random.default_rng(seed+1);w=rng.normal(0,1/np.sqrt(896),(896,n)).astype(np.float16)
 bias=rng.normal(0,.01,n).astype(np.float16) if op=='qkv' else np.zeros(n,dtype=np.float16)
 acc=ordered(x,w,library);y=(acc+bias.astype(np.float32)).astype(np.float16) if op=='qkv' else acc.astype(np.float16)
 dense=(x.astype(np.float64)@w.astype(np.float64)+bias.astype(np.float64))
 inp=x.reshape(-1) if op=='qkv' else head_layout(x)
 ref=head_layout(y) if op=='qkv' else y.reshape(-1);dense=head_layout(dense) if op=='qkv' else dense.reshape(-1)
 with path.open('xb') as f:
  for a in (inp,w,bias,ref):f.write(a.astype('<f2').tobytes())
 np.savez(path.with_suffix('.npz'),expected=ref,dense=dense)
 save(path.with_suffix('.json'),dict(t=t,op=op,sha256=sha(path),reference_sha256=sha(path.with_suffix('.npz')),synthetic=True,seed=seed,precision='FP16 operands; ordered FP32 accumulation; QKV FP32 bias before FP16 rounding.'))

def validate(path,inp,t,op):
 n=1152 if op=='qkv' else 896;a=np.fromfile(path,dtype='<f2');assert a.size==t*n and np.isfinite(a).all()
 with np.load(inp.with_suffix('.npz')) as f:ref=f['expected'];dense=f['dense']
 # Exact ordered reference for the unchanged accumulation order, including bias.
 assert np.array_equal(a.view(np.uint16),ref.view(np.uint16)),'ordered FP32/bias/layout output mismatch'
 error=np.abs(a.astype(float)-dense);bound=2e-6+.001*np.abs(dense)
 assert np.all(error<=bound),dict(errors=int((error>bound).sum()),max_abs=float(error.max()))
 return dict(elements=a.size,ordered_bits_exact=True,dense_errors=0,dense_max_abs=float(error.max()),dense_relative_l2=float(np.linalg.norm(a.astype(float)-dense)/np.linalg.norm(dense)),dense_bound='2e-6 + 0.1% abs(float64 reference)')
