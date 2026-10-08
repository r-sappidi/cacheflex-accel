"""Independent interval arithmetic / plan checks for passive ROI attribution."""
import os,unittest
from plan import cases,argv,work

def intervals(spans,begin,end):
 points=sorted({begin,end,*[max(begin,min(end,x)) for a,b in spans for x in (a,b)]})
 busy=area=0
 for a,b in zip(points,points[1:]):
  n=sum(x<=a and b<=y for x,y in spans);busy+=(b-a)*(n>0);area+=(b-a)*n
 return busy,area
class Accounting(unittest.TestCase):
 def test_overlap_not_additive(self):
  self.assertEqual(intervals([(10,70),(20,50)],0,100),(60,90))
 def test_phase_split(self):
  spans=[(10,70),(20,50)]
  self.assertEqual([intervals(spans,a,b) for a,b in [(0,30),(30,60),(60,100)]],[(20,30),(30,50),(10,10)])
 def test_raw_squash_different_from_completed(self):
  self.assertEqual(intervals([(40,65),(55,80)],0,100),(40,50))
  completed_wait=65-40;self.assertEqual(completed_wait,25)
 def test_censored_packet(self):
  self.assertEqual(intervals([(90,120)],0,100),(10,10))
 def test_off_on_same_guest(self):
  for arm in ('cache','spm'):
   os.environ['FA49_ARM']=arm;a=cases();os.environ['FA49_ARM']=arm+'_off';b=cases()
   self.assertEqual([argv(c) for c in a],[argv(c) for c in b])
   self.assertEqual([c['observer'] for c in a],[1,1]);self.assertEqual([c['observer'] for c in b],[0,0])
 def test_counts(self):
  t,u,c=work(1024);self.assertEqual((sum(t),sum(u)//16,c),(56,2240,71680));self.assertEqual(sum(x*2 for x in u),c)
if __name__=='__main__':unittest.main()
