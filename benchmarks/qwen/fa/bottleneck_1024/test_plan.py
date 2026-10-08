import unittest
from plan import cases,argv,work
class Tests(unittest.TestCase):
 def test_twelve_paired_cases(self):
  rows=cases();self.assertEqual(len(rows),12);self.assertEqual(len({x['name'] for x in rows}),12)
  self.assertTrue(all(x['t']==1024 and x['kind']=='fa' for x in rows))
  for v in ('control','stage','l1','l2','both','packed'):
   self.assertEqual({x['spm'] for x in rows if x['variant']==v},{False,True})
 def test_geometry_and_work(self):
  for c in cases():self.assertEqual(argv(c)[:6],[1024,8,16,256,16,1024])
  tasks=[0]*8;updates=[0]*8;copies=0
  for task in range(56):
   head=task//4;group=(task%4+head)%4
   tasks[task%8]+=1;updates[task%8]+=256*(group+1);copies+=512*(group+1)
  self.assertEqual(work(1024),(tasks,updates,copies));self.assertEqual(tasks,[7]*8)
if __name__=='__main__':unittest.main()
