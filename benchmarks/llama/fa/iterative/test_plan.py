import unittest
from plan import *
class Tests(unittest.TestCase):
 def test_exact_work_and_partition(self):
  for spm in (False,True):
   c=case(spm);w=describe(c)
   self.assertEqual(w['tasks'],[64]*8);self.assertEqual(len(set(w['updates'])),1)
   self.assertEqual(sum(w['copies']),2228224 if spm else 0)
   self.assertEqual(w['spm_bytes'],32768 if spm else 0)
   # Independent enumeration of actual task traversal.
   tasks=[0]*8;updates=[0]*8
   for core in range(8):
    for task in range(core,32*16,8):
     head=task//16;qi=(task%16+head)%16*256
     tasks[core]+=1;updates[core]+=256*((qi+256)//256)
   self.assertEqual(w['updates'],updates);self.assertEqual(w['tasks'],tasks)
 def test_reject_invalid(self):
  for delta in [dict(hq=14),dict(hkv=2),dict(br=24),dict(r=1024),dict(t=300)]:
   with self.assertRaises(AssertionError):describe(dict(case(True),**delta))
 def test_adaptive_and_budget(self):
  c,d=choose([],False);self.assertEqual(c,case(False))
  counter=dict(spm_sq_full=5,l1_credit_full=1,l1_misses=3,instructions=100,iq_full=0,icache_stall=0)
  row=dict(name=name(c),case=c,roi_us=100,counters=counter)
  nxt,why=choose([row],False);self.assertEqual(nxt['bc'],128);self.assertIn('observed',why)
  self.assertEqual(sum(nxt[k]!=c[k] for k in ('br','bc','r')),1)
  rows=[row]
  while True:
   nxt,why=choose(rows,False)
   if nxt is None:break
   self.assertNotIn(name(nxt),[x['name'] for x in rows])
   rows.append(dict(name=name(nxt),case=nxt,roi_us=110,counters=counter))
  self.assertEqual(why['status'],'local_neighbors_measured');self.assertLess(len(rows),10)
 def test_bc_oracle_key_and_gate(self):
  for axis,levels in LEVELS.items():
   for v in levels:
    c=dict(case(True),**{axis:v});describe(c);describe(gate(c));self.assertEqual(argv(c)[10:12],[32,8])
if __name__=='__main__':unittest.main()
