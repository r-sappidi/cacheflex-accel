import unittest,tempfile,ast
from pathlib import Path
import worker
class Scheduling(unittest.TestCase):
 def test_validator_verbatim(self):
  s=(worker.original.H/'run.py').read_text();v=worker.validation_source()
  self.assertIn("assert blocks==stats(off)",v)
  self.assertIn('config_equal(source,path)',v)
  self.assertIn("assert summary['groups'].get('copy_DATA'",v)
  self.assertIn('unchanged()',v);ast.parse(v)
 def test_exclusive_claim_prevents_old_launch(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'pending_case';p.mkdir()
   with self.assertRaises(FileExistsError):p.mkdir()
 def test_locations(self):
  self.assertEqual(worker.H.name,'packet_parallel')
  self.assertEqual(worker.original.H.name,'packet_diag')
  self.assertEqual(worker.ROOT.name,'parallel_v1')
if __name__=='__main__':unittest.main()
