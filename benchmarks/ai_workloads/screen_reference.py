"""Independent projection reference and minimal adapter of existing GEMM code."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent


def core_header():
    source = (HERE/'mlp_study.cpp').read_text().split('size_t check(Shared&s,int iteration) {')[0]
    assert source.count('__fp16*ap,__fp16*bp,int lda=0) {') == 1
    assert source.count('int np=Threads/s.mp') == 1
    source = source.replace('__fp16*ap,__fp16*bp,int lda=0) {', '__fp16*ap,__fp16*bp,int lda=0,int team=8) {')
    source = source.replace('int np=Threads/s.mp', 'int np=team/s.mp')
    return source + '\n}\n'


def mlp_source():
    source = (HERE/'mlp_study.cpp').read_text()
    call = 'm5_write_file(s.y,(size_t)s.t*s.d*2,0,"output.bin")'
    assert source.count(call) == 1
    helper = '''
static uint64_t screen_export(const void* data,uint64_t bytes,uint64_t offset,const char* name) {
 alignas(64) uint16_t scratch[32];
 auto* src=reinterpret_cast<const volatile uint16_t*>(data);
 for(uint64_t i=0;i<bytes;i+=64) {
  for(int j=0;j<32;++j)scratch[j]=src[i/2+j];
  asm volatile("dsb sy":::"memory");
  if(m5_write_file(scratch,64,offset+i,name)!=64)return i;
 }
 return bytes;
}
'''
    return source.replace('int main(int argc,char**argv) {', helper+'\nint main(int argc,char**argv) {').replace(call, call.replace('m5_write_file', 'screen_export'))


def gemm(a, b, kc):
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
    for k0 in range(0, a.shape[1], kc):
        acc = np.zeros_like(out)
        for k in range(k0, min(k0+kc, a.shape[1])):
            acc = (acc.astype(np.float64) + a[:, k, None].astype(np.float64)*b[k, None, :].astype(np.float64)).astype(np.float16)
        out = (out.astype(np.float32)+acc.astype(np.float32)).astype(np.float16)
    return out


def generate(path, m, k, n, groups, shared, kc, seed=137):
    rng = np.random.default_rng(seed)
    x = (rng.integers(-8, 9, (groups, m, k))/128).astype(np.float16)
    w = (rng.integers(-8, 9, (1 if shared else groups, k, n))/128).astype(np.float16)
    y = np.stack([gemm(x[g], w[0 if shared else g], kc) for g in range(groups)])
    with path.open('xb') as stream:
        for a in (x, w, y):
            stream.write(a.astype('<f2').tobytes())
    return y


def validate(path, inp, m, k, n, groups, shared):
    data = np.fromfile(inp, dtype='<f2')
    nx, nw = groups*m*k, (1 if shared else groups)*k*n
    expected = data[nx+nw:].reshape(groups, m, n)
    actual = np.fromfile(path/'output.bin', dtype='<f2').reshape(expected.shape)
    error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    assert np.isfinite(actual).all()
    assert (error <= 2e-6+.01*np.abs(expected.astype(np.float64))).all()
    x = data[:nx].reshape(groups, m, k).astype(np.float64)
    w = data[nx:nx+nw].reshape(-1, k, n).astype(np.float64)
    math = np.stack([x[g]@w[0 if shared else g] for g in range(groups)])
    relative_l2 = float(np.linalg.norm(actual-math)/max(np.linalg.norm(math), 1e-30))
    peak_error = float(np.max(np.abs(actual-math))/max(np.max(np.abs(math)), 1e-30))
    assert relative_l2 <= .01 and peak_error <= .02, (relative_l2, peak_error)
    h = 1469598103934665603
    for word in actual.view(np.uint16).flat:
        h = ((h ^ int(word))*1099511628211)&((1<<64)-1)
    return dict(elements=actual.size, errors=0, nonfinite=0, max_abs_error=float(error.max()),
                relative_l2=relative_l2, peak_normalized_max_abs=peak_error, hash=f'{h:016x}')


if __name__ == '__main__':
    if sys.argv[1] == 'header':
        Path(sys.argv[2]).write_text(core_header())
    elif sys.argv[1] == 'mlp-source':
        Path(sys.argv[2]).write_text(mlp_source())
    else:
        generate(Path(sys.argv[1]), *map(int, sys.argv[2:]))
