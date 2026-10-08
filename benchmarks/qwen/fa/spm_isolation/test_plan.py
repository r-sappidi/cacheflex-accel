import unittest
from pathlib import Path
from plan import cases,parameters,work,overrides
class Tests(unittest.TestCase):
 def test_scope(self):
  c=cases();self.assertEqual(len(c),38);self.assertEqual(len({x['name'] for x in c}),38)
  for v in ('bridge96','bridge4096','copy_oracle'):
   self.assertEqual(parameters(v,True),parameters('control',True))
  for k in parameters('combined',True):self.assertTrue(k.endswith(('num_spm_ways','spm_access_response_latency','spmLoadPorts')))
 def test_shadow_accounting(self):
  def free(stores,other,copies,budget):
   used=stores+other+(0 if budget else copies)
   assert used<=48 and (not budget or copies<=budget)
   return min(48-used,budget-copies) if budget else 48-used
  for n in range(49):self.assertEqual(free(0,0,n,0),48-n)
  self.assertEqual(free(0,0,48,96),48)
  self.assertEqual(free(0,0,96,96),0)
  self.assertEqual(free(48,0,100,4096),0)
  self.assertEqual(free(0,48,100,4096),0)
 def test_work(self):
  self.assertEqual(work(256)[0],[2,2,2,2,2,2,1,1]);self.assertEqual(work(1024)[0],[7]*8)
  self.assertEqual(work(4096)[2],974848)
 def test_preserved_ordering_code(self):
  b=Path(__file__).resolve().parents[4]/'results/cacheflex_flash_attention/spm_isolation_20261004/build_v1'
  old=(b/'lsq_unit.original.cc').read_text();new=(b/'lsq_unit.cc').read_text()
  for start,end in [('LSQUnit::reserveSpmStore','LSQUnit::cleanupRetiredShadow'),('LSQUnit::executeLoad','LSQUnit::executeStore')]:
   a=old.index(start);z=old.index(end,a+len(start));aa=new.index(start);zz=new.index(end,aa+len(start))
   self.assertEqual(old[a:z],new[aa:zz])
if __name__=='__main__':unittest.main()
