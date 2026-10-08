import unittest,os
from pathlib import Path
from plan import ARMS,cases,argv,service,work
class Plan(unittest.TestCase):
 def test_observer_pairs(self):
  for arm in ('spm','async'):
   os.environ['FA48_ARM']=arm;a=cases()
   os.environ['FA48_ARM']=arm+'_off';b=cases()
   self.assertEqual([argv(c) for c in a],[argv(c) for c in b])
   self.assertEqual([c['observer'] for c in a],[1,1])
   self.assertEqual([c['observer'] for c in b],[0,0])
 def test_factorial(self):
  self.assertEqual({ARMS[a][2] for a in ('async','no_probe','zero_service','no_probe_zero')},{0,1,2,3})
  self.assertEqual([service(7,f) for f in range(6)],[2400,2400,0,0,2400,2400])
 def test_work(self):
  t,u,c=work(1024);self.assertEqual((sum(t),sum(u)//16,c),(56,2240,71680))
 def test_phase_map(self):
  p=Path(__file__).resolve().parents[4]/'results/cacheflex_flash_attention/pv_diag_20261005/guest_v1/phase_map.txt'
  rows=[x.split() for x in p.read_text().splitlines()];self.assertEqual(len(rows),len({a for a,p in rows}))
  self.assertTrue({5,6,7,8,9,11,13,17,19,21,24} <= {int(p) for a,p in rows})
 def test_partition_and_negative(self):
  ticks=[0]*26;last=0;phase=0
  for now,next_phase in [(10,5),(30,6),(55,7),(60,8),(70,9),(80,11),(90,24),(100,25)]:
   ticks[phase]+=now-last;last=now;phase=next_phase
  ticks[phase]+=110-last
  self.assertEqual(sum(ticks),110)
  self.assertEqual(ticks[5],20);self.assertEqual(ticks[6],25)
  ticks[8]+=1;self.assertNotEqual(sum(ticks),110)
if __name__=='__main__':unittest.main()
