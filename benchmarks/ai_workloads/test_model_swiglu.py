"""Host regression checks for model-shaped pilot planning and references."""
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
import numpy as np
from model_swiglu_reference import gemm,acl_activation
from run_model_swiglu import OUT,command,expected_copies

class ModelPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.tmp.name)/'reference.so'
        subprocess.run(['g++','-O3','-march=native','-ffp-contract=off','-fopenmp','-shared','-fPIC',
            str(Path(__file__).with_name('model_swiglu_reference.cpp')),'-o',str(cls.library)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ordered_fp32_reference(self):
        rng=np.random.default_rng(917)
        a=rng.normal(size=(5,31)).astype(np.float16)
        b=rng.normal(size=(31,257)).astype(np.float16)
        ref=np.zeros((5,257),np.float32)
        for k in range(31):
            ref=(ref.astype(np.float64)+a[:,k,None].astype(np.float64)*b[k,None,:].astype(np.float64)).astype(np.float32)
        result=gemm(a,b,self.library)
        np.testing.assert_array_equal(result,ref.astype(np.float16))

    def test_acl_half_activation_rounding(self):
        # Source-defined exp -> half, add -> half, FRECPE/FRECPS and half mul.
        # This value differs from ideally rounded SiLU by three half ULPs.
        value=np.array([0,-0.1292724609375],dtype=np.float16)
        expected=np.array([0,-0.060546875],dtype=np.float16)
        np.testing.assert_array_equal(acl_activation(value,self.library),expected)

    def test_copy_count_rows_and_mapping(self):
        self.assertEqual(expected_copies(128,2048,8192,2)*64,2*96*1024*1024)
        self.assertEqual(expected_copies(1,256,512,8),expected_copies(1,256,512,1))

    def test_command_keeps_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            source=['sim','--outdir=old','config','--num-cpus=8','--l1d_size=512kB','--cmd','oldguest','--options','oldargs',"system.readfile='oldinput'",'system.cpu[0].workload[0].cmd=["old"]']
            (p/'command.json').write_text(json.dumps(source))
            for arch in ('cache','spm','acl'):
                cmd=command(p,Path('/sim'),Path('/out'),Path('/guest'),Path('/input'),arch,(128,2048,8192,2,128,0),5)
                self.assertIn('--num-cpus=8',cmd);self.assertIn('--l1d_size=512kB',cmd)
                self.assertIn("system.readfile='/input'",cmd)
                self.assertIn('128 2048 8192 first 5' if arch=='acl' else '128 2048 8192 2 128 0 5 1 8',cmd)

if __name__=='__main__':unittest.main()
