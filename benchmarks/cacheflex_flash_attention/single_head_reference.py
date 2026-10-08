"""Single-head numerical reference; same explicit rounding as the FA audit."""
import numpy as np
from fa_independent_reference import inputs, gemm


def reference(t, bc, br=32):
    q, k, v = inputs(t, True)
    q = q[0]
    out = np.empty_like(q)
    for i in range(0, t, br):
        qq = q[i:i+br]; n = len(qq)
        m = np.full(n, -1e30, dtype=np.float32)
        l = np.zeros(n, dtype=np.float32)
        o = np.zeros((n, 64), dtype=np.float32)
        for j in range(0, t, bc):
            s = gemm(qq, k[j:j+bc].T).astype(np.float32)*np.float32(.125)
            mn = np.maximum(m, s.max(axis=1))
            with np.errstate(under='ignore'):
                alpha = np.exp((m-mn).astype(np.float64)).astype(np.float32)
            z = np.exp((s-mn[:, None]).astype(np.float64)).astype(np.float32)
            l = (alpha*l+z.sum(axis=1, dtype=np.float32)).astype(np.float32)
            o = (o*alpha[:, None]).astype(np.float32)
            o = (o+gemm(z.astype(np.float16), v[j:j+bc]).astype(np.float32)).astype(np.float32)
            m = mn
        out[i:i+br] = (o*(np.float32(1)/l)[:, None]).astype(np.float16)
    return out


def check(path, expected):
    actual = np.fromfile(path, dtype=np.float16).reshape(expected.shape)
    error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    row = dict(elements=actual.size,
        nonfinite=int((~np.isfinite(actual)).sum()),
        errors=int((error > 2e-6+.01*np.abs(expected.astype(np.float64))).sum()),
        max_abs_error=float(error.max()), tolerance='2e-6 + 1% abs(reference)')
    assert row['errors']==row['nonfinite']==0, row
    return row
