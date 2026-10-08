"""Small host-only positive and corruption tests for the numerical gate."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from validate_mlp_acl import sequential_half_gemm, validate

class NumericalGateTest(unittest.TestCase):
    def test_valid_and_corrupt_chain(self):
        rng = np.random.default_rng(1)
        def values(shape):
            return (rng.integers(-8,9,size=shape)/128).astype(np.float16)
        t,d,f=2,4,8
        x,wg,wu,wd=(values(s) for s in ((t,d),(d,f),(d,f),(f,d)))
        gate=sequential_half_gemm(x,wg);up=sequential_half_gemm(x,wu)
        gf=gate.astype(np.float64)
        silu=(gf/(1+np.exp(-gf))).astype(np.float16)
        hidden=(silu.astype(np.float64)*up.astype(np.float64)).astype(np.float16)
        output=sequential_half_gemm(hidden,wd)
        with tempfile.TemporaryDirectory(prefix='mlp-acl-test-') as tmp:
            path=Path(tmp)
            np.concatenate([a.ravel() for a in (x,wg,wu,wd,output)]).astype('<f2').tofile(path/'input')
            for name,a in dict(gate=gate,up=up,silu=silu,hidden=hidden,output=output).items():
                a.astype('<f2').tofile(path/(name+'.bin'))
            self.assertTrue(validate(path,path/'input',t,d,f)['passed'])
            output[0,0]=1
            output.astype('<f2').tofile(path/'output.bin')
            with self.assertRaises(AssertionError):
                validate(path,path/'input',t,d,f)

if __name__ == '__main__':
    unittest.main()
