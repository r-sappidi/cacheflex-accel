"""Common mathematical accuracy metrics; independent of KC/accumulation order."""
from pathlib import Path
import json
import sys
import numpy as np

def audit(path,inp,t,d,f):
    data=np.fromfile(inp,dtype='<f2');off=0
    def take(shape):
        nonlocal off
        n=int(np.prod(shape));a=data[off:off+n].reshape(shape).astype(np.float64);off+=n
        return a
    x,wg,wu,wd=(take(s) for s in ((t,d),(d,f),(d,f),(f,d)))
    gate=x@wg;up=x@wu
    reference=(gate/(1+np.exp(-gate))*up)@wd
    y=np.fromfile(path/'output.bin',dtype='<f2').reshape(t,d).astype(np.float64)
    assert np.isfinite(y).all() and np.isfinite(reference).all()
    err=y-reference
    l2=float(np.linalg.norm(err)/max(np.linalg.norm(reference),1e-30))
    linf=float(np.max(np.abs(err))/max(np.max(np.abs(reference)),1e-30))
    result=dict(relative_l2=l2,peak_normalized_max_abs=linf,max_abs=float(np.max(np.abs(err))),
                relative_l2_limit=0.01,peak_normalized_max_abs_limit=0.02,
                passed=l2<=0.01 and linf<=0.02)
    (path/'math_accuracy.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    result=audit(Path(sys.argv[1]),Path(sys.argv[2]),*map(int,sys.argv[3:6]))
    print(json.dumps(result,indent=2))
    sys.exit(0 if result['passed'] else 1)
