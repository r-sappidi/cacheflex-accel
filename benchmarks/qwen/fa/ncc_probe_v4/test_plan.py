import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plan import service,work
class Test(unittest.TestCase):
 def test_packing(self):
  for rows,cols,kind in [(128,64,1),(16,256,2)]:
   indices=[k*rows+r if kind==1 else (r//8)*cols*8+k*8+r%8 for r in range(rows) for k in range(cols)]
   self.assertEqual(sorted(indices),list(range(rows*cols)))
 def test_work(self):
  for t in (256,1024,4096):
   tasks,updates,copies=work(t);self.assertEqual(sum(tasks),14*t//256)
   self.assertEqual(sum(updates)//16,14*(t//256)*(t//256+1)//2*16)
   self.assertEqual(copies,14*(t//256)*(t//256+1)//2*512)
 def test_service(self):
  self.assertEqual(service(5),5472);self.assertEqual(service(6),3424);self.assertEqual(service(7),2400);self.assertEqual(service(8),6152)
 def test_hook_order(self):
  h=(Path(__file__).parent/'hook.hh').read_text();self.assertIn('softfloat_round_near_even',h);self.assertIn('f16_mulAdd',h)
  self.assertIn('quiesceCycles(tc,cycles)',h)
if __name__=='__main__':unittest.main()
