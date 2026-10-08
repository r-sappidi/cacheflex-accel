"""Independent per-operator numerical audit of the ACL FP16 fast-math chain.

ACL hybrid FP16 carries the accumulator across K; the custom kernel instead
sums separately rounded KC256 partials. Check actual arithmetic, not agreement
with the custom kernel's reduction schedule. All dumps are outside the ROI.
"""
from pathlib import Path
import json
import sys
import numpy as np

def sequential_half_gemm(a, b):
    # float64 evaluates each half-input multiply-add exactly before FP16 rounding.
    acc = np.zeros((a.shape[0], b.shape[1]), np.float16)
    af, bf = a.astype(np.float64), b.astype(np.float64)
    for k in range(a.shape[1]):
        acc = (acc.astype(np.float64) + af[:,k,None]*bf[k,None,:]).astype(np.float16)
    return acc

def validate(path, readfile, t, d, f):
    raw = np.fromfile(readfile, dtype='<f2')
    off = 0
    def take(shape):
        nonlocal off
        n = int(np.prod(shape)); a = raw[off:off+n].reshape(shape); off += n
        return a
    x,wg,wu,wd,legacy = (take(s) for s in ((t,d),(d,f),(d,f),(f,d),(t,d)))
    assert off == raw.size
    def tensor(name,shape):
        a = np.fromfile(path/(name+'.bin'),dtype='<f2').reshape(shape)
        assert np.isfinite(a).all(), name
        return a
    gate,up,act,hidden = (tensor(n,(t,f)) for n in ('gate','up','silu','hidden'))
    y = tensor('output',(t,d))
    checks = {}
    def check(name, actual, expected, ulps=0):
        assert np.isfinite(expected).all(), (name, 'nonfinite oracle')
        error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        # At powers of two, use the upward ULP magnitude, including subnormals.
        spacing = np.spacing(np.abs(expected)).astype(np.float64)
        bad = error > ulps*spacing
        checks[name] = dict(max_abs=float(error.max()), errors=int(bad.sum()), allowed_ulps=ulps)
    check('gate_gemm',gate,sequential_half_gemm(x,wg))
    check('up_gemm',up,sequential_half_gemm(x,wu))
    gf = gate.astype(np.float64)
    ideal_silu = (gf/(1+np.exp(-gf))).astype(np.float16)
    # FP16 vector SiLU is approximate; allow at most two output-half ULPs.
    check('silu',act,ideal_silu,2)
    check('multiply',hidden,(act.astype(np.float64)*up.astype(np.float64)).astype(np.float16))
    check('down_gemm',y,sequential_half_gemm(hidden,wd))
    # Also report distance to an independent high-precision mathematical chain.
    gd = x.astype(np.float64) @ wg.astype(np.float64)
    ud = x.astype(np.float64) @ wu.astype(np.float64)
    gold = ((gd/(1+np.exp(-gd)))*ud) @ wd.astype(np.float64)
    assert np.isfinite(gold).all()
    def metrics(a,b):
        a,b = a.astype(np.float64),b.astype(np.float64)
        return dict(max_abs=float(np.max(np.abs(a-b))),
                    relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)))
    result = dict(stages=checks, acl_vs_float64=metrics(y,gold),
                  legacy_reference_vs_float64=metrics(legacy,gold),
                  acl_vs_legacy_reference=metrics(y,legacy),
                  passed=all(c['errors']==0 for c in checks.values()))
    (path/'numerical_audit.json').write_text(json.dumps(result,indent=2))
    assert result['passed'], result
    return result

if __name__ == '__main__':
    print(json.dumps(validate(Path(sys.argv[1]),Path(sys.argv[2]),*map(int,sys.argv[3:6])),indent=2))
