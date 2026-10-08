import importlib.util,json,multiprocessing as mp,tempfile,time,unittest
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');EXT=ROOT/'results/parallel_overlap_20260930'
spec=importlib.util.spec_from_file_location('broker_test',EXT/'broker.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def claim(name,q):q.put((name,b.acquire(name)))
class BrokerTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();b.EXT=Path(self.t.name)
  b.save(b.EXT/'broker.json',dict(cap=32*b.GIB,old_reserved=16*b.GIB,leases={},hold_new=False,heartbeat=time.time()))
 def tearDown(self):self.t.cleanup()
 def test_atomic_parallel_cap(self):
  ctx=mp.get_context('fork');q=ctx.Queue();ps=[ctx.Process(target=claim,args=(str(i),q)) for i in range(8)]
  for p in ps:p.start()
  for p in ps:p.join();self.assertEqual(p.exitcode,0)
  results=[q.get() for _ in ps];self.assertEqual(sum(ok for _,ok in results),2)
  s=b.read(b.EXT/'broker.json');self.assertEqual(b.total(s),s['cap'])
 def test_release_reuses_and_stale_failclosed(self):
  self.assertTrue(b.acquire('a'));self.assertTrue(b.acquire('b'));self.assertFalse(b.acquire('c'))
  b.release('a');self.assertTrue(b.acquire('c'))
  b.release('b')
  with b.locked() as s:s['heartbeat']=time.time()-25
  self.assertFalse(b.acquire('d'))
 def test_pressure_and_growth_reservations(self):
  self.assertTrue(b.acquire('a'))
  with b.locked() as s:s['leases']['a']['bytes']+=8*b.GIB;s['hold_new']=True
  self.assertFalse(b.acquire('b'));self.assertEqual(b.capacity()[0],1)
 def test_initial_admission_arithmetic(self):
  #3x16K at16;2FFN at8;2x32K at12;2paused32K at4.5;9worker caps.
  old=(20.5+3*16+2*8+2*12+2*4.5+9*1.5)*b.GIB
  cap=162121272524
  self.assertLessEqual(old+2*8*b.GIB,cap)
  self.assertGreater(old+3*8*b.GIB,cap)
if __name__=='__main__':unittest.main()
