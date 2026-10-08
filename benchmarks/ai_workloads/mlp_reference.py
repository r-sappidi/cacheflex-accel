"""Independent host SwiGLU chain reference with FP16 GEMM accumulation."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if sys.argv[1]=='helpers':
    p=(ROOT/'benchmarks/cacheflex_spm_gemm/vl16_2m4n_driver.cpp').read_text()
    pack=p[p.index('void packA24Neon('):p.index('\nvoid *worker(')].replace('group < 3','group < TileM / 8')
    s=(ROOT.parent/'cacheflex_micro/kernels/llama_bench/kernels_sve.hpp').read_text()
    def function(name):
        start=s.index(name); start=s.rfind('static ',0,start)
        return s[start:s.index('\n}',start)+3]
    exp=function('svfloat32_t sve_exp_f32(').replace('sve_exp_f32','mlp_exp')
    act=function('void swiglu_fp16(').replace('swiglu_fp16','mlp_swiglu').replace('sve_exp_f32','mlp_exp')
    # SVE FCVT h->s consumes the low half of each 32-bit lane. Duplicate
    # consecutive halves before widening, and compact even halves on narrowing.
    helpers='''
static inline svfloat16_t mlp_load_half(svbool_t pg,const __fp16* p) {
 auto h=svld1_f16(pg,p); return svzip1_f16(h,h);
}
static inline svfloat16_t mlp_narrow(svbool_t pg,svfloat32_t v) {
 auto h=svcvt_f16_f32_z(pg,v); return svuzp1_f16(h,h);
}
'''
    act=act.replace('svld1_f16(', 'mlp_load_half(').replace('svcvt_f16_f32_z(', 'mlp_narrow(')
    Path(sys.argv[2]).write_text(pack+'\n'+exp+'\n'+helpers+'\n'+act)
    sys.exit(0)
t,d,f=map(int,sys.argv[1:4]);out=Path(sys.argv[4]);rng=np.random.default_rng(20260909)
kc=int(sys.argv[5]) if len(sys.argv)>5 else 256
if kc<=0:raise ValueError('KC must be positive')
acc32=bool(int(sys.argv[6])) if len(sys.argv)>6 else False
seed=int(sys.argv[7]) if len(sys.argv)>7 else 20260909
xdiv=float(sys.argv[8]) if len(sys.argv)>8 else 128
rng=np.random.default_rng(seed)
def matrix(shape,div=128):return (rng.integers(-8,9,size=shape).astype(np.float32)/div).astype(np.float16)
x=matrix((t,d),xdiv);wg=matrix((d,f));wu=matrix((d,f));wd=matrix((f,d))
def gemm(a,b):
    if acc32:
        accum=np.zeros((a.shape[0],b.shape[1]),np.float32)
        for k in range(a.shape[1]):
            accum=(accum.astype(np.float64)+a[:,k,None].astype(np.float64)*b[k,None,:].astype(np.float64)).astype(np.float32)
        return accum.astype(np.float16)
    c=np.zeros((a.shape[0],b.shape[1]),np.float16)
    for start in range(0,a.shape[1],kc):
        accum=np.zeros_like(c)
        for k in range(start,min(start+kc,a.shape[1])):
            # Half products are exact in float32 at these input magnitudes.
            accum=(accum.astype(np.float32)+a[:,k,None].astype(np.float32)*b[k,None,:].astype(np.float32)).astype(np.float16)
        c=(c.astype(np.float32)+accum.astype(np.float32)).astype(np.float16)
    return c
gate=gemm(x,wg);up=gemm(x,wu)
hidden=(gate.astype(np.float64)/(1+np.exp(-gate.astype(np.float64)))*up.astype(np.float64)).astype(np.float16)
y=gemm(hidden,wd)
with out.open('xb') as stream:
    for a in (x,wg,wu,wd,y):stream.write(a.astype('<f2').tobytes())
print('REFERENCE',t,d,f,'nonzero',np.count_nonzero(y),'max_abs',float(np.max(np.abs(y))))
