from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from audit_mlp_math import audit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
class StudyTests(unittest.TestCase):
 def test_original_reference_unchanged(self):
  with tempfile.TemporaryDirectory(prefix='mlp-reference-') as tmp:
   p=Path(tmp)/'input.bin'
   subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),'24','128','256',str(p)],check=True,stdout=subprocess.DEVNULL)
   old=ROOT/'results/ai_workloads/mlp_clean_20260909/runs/input_24_128_256.bin'
   self.assertEqual(p.read_bytes(),old.read_bytes())
 def test_row_generator(self):
  for rows in (8,16):
   with tempfile.TemporaryDirectory(prefix='mlp-rows-') as tmp:
    subprocess.run([sys.executable,str(HERE/'make_mlp_row_kernels.py'),tmp,str(rows)],check=True)
    for name in ('v1_24x1vl.cpp','kernels_tile_spm.hpp'):
     s=(Path(tmp)/name).read_text()
     if name.endswith('.hpp'):s=s[s.index('#elif defined(VL_16)'):]
     self.assertNotIn(f'fmla z{8+rows}.h',s)
     self.assertIn(f'K * {rows*2}',s)
     self.assertIn(f'st1h {{z{7+rows}.h}},p0,[x5]',s)
 def test_math_gate_rejects_corruption(self):
  with tempfile.TemporaryDirectory(prefix='mlp-math-') as tmp:
   p=Path(tmp);inp=p/'input.bin'
   subprocess.run([sys.executable,str(HERE/'mlp_reference.py'),'24','128','256',str(inp),'256','1'],check=True,stdout=subprocess.DEVNULL)
   data=np.fromfile(inp,dtype='<f2');data[-3072:].tofile(p/'output.bin')
   self.assertTrue(audit(p,inp,24,128,256)['passed'])
   np.ones(3072,dtype='<f2').tofile(p/'output.bin')
   self.assertFalse(audit(p,inp,24,128,256)['passed'])

if __name__=='__main__':unittest.main()
