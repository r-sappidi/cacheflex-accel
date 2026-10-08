"""Independent recurrence/lifetime tests for delayed PV retirement."""
import random,unittest
from plan import work

def execute(seed,async_pv,save_alpha=True,drain_v=True):
 rng=random.Random(seed);out=[0.]*16;expected=out.copy();pending=None;events=[]
 for tile in range(4):
  if pending is not None:raise AssertionError('V replaced while pending')
  events.append(('stage',tile))
  for row in range(16):
   alpha=rng.random();value=rng.random();expected[row]=expected[row]*alpha+value
   events.append(('cpu',tile,row,pending is not None))
   if pending is not None:
    r,a,v=pending;out[r]=out[r]*(a if save_alpha else alpha)+v;pending=None
   pending=(row,alpha,value)
   if not async_pv:
    r,a,v=pending;out[r]=out[r]*a+v;pending=None
  if drain_v:
   r,a,v=pending if pending is not None else (0,1.,0.)
   if pending is not None:out[r]=out[r]*a+v;pending=None
 assert pending is None
 return out,expected,events
class Schedule(unittest.TestCase):
 def test_recurrence_and_overlap(self):
  for seed in range(100):
   a,e,events=execute(seed,True);b,f,bevents=execute(seed,False)
   self.assertEqual(a,e);self.assertEqual(a,b);self.assertEqual(e,f)
   self.assertEqual(sum(x[3] for x in events if x[0]=='cpu'),60)
   self.assertFalse(any(x[3] for x in bevents if x[0]=='cpu'))
 def test_alpha_lifetime_negative(self):
  a,e,_=execute(1,True,False);self.assertNotEqual(a,e)
 def test_v_lifetime_negative(self):
  with self.assertRaises(AssertionError):execute(1,True,drain_v=False)
 def test_target_work(self):
  tasks,updates,copies=work(1024)
  self.assertEqual(sum(tasks),56);self.assertEqual(sum(updates)//16,2240)
  self.assertEqual(copies,71680)
if __name__=='__main__':unittest.main()
