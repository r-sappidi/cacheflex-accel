import unittest,os,json
from pathlib import Path
import plan,build
class Schedule(unittest.TestCase):
 def test_staging_once_and_before_consumption(self):
  # Independent event model of all query-group causal extents, including tile
  # wraparound and reuse across tasks. Verify exact data at every PV use.
  for mode in range(4):
   for t in (256,512,1024,4096):
    slots={};copied=0;expected=0
    for kend in range(256,t+1,256):
     expected+=kend
     for j in range(0,kend,256):
      offset=(j//256%2)*256 if mode>=2 else 0
      def stage(first,count,off,source):
       nonlocal copied
       for r in range(first,first+count):slots[off+r]=(kend,source+r);copied+=1
      if mode in (0,3):stage(0,256,offset,j)
      for bi in range(0,256,16):
       if bi==0 and (mode==1 or mode==2 and j==0):
        for r in range(16):stage(r*16,16,offset,j)
       self.assertEqual([slots[offset+r] for r in range(256)],[(kend,j+r) for r in range(256)])
       if mode==2 and j+256<kend:stage(bi,16,((j//256+1)%2)*256,j+256)
    self.assertEqual(copied,expected)
    self.assertLessEqual(len(slots),512 if mode>=2 else 256)
 def test_encodings_nonoverlap(self):
  slots=[((r%1024)<<6)|(w<<16) for r in range(512) for w in range(2)]
  self.assertEqual(len(slots),len(set(slots)));self.assertTrue(all(x>>16<6 for x in slots))
 def test_fixed_geometry(self):
  for arm in ('cache','burst','paced','overlap','dbburst'):
   os.environ['FA44_ARM']=arm
   for c in plan.cases():
    self.assertEqual(plan.argv(c)[2:6],[16,256,16,1024]);self.assertEqual(len(plan.argv(c)),17)
    self.assertEqual(len(plan.parameters(c['variant'],c['spm'])),8)
 def test_source_hooks(self):
  source=build.transform((build.OLD/'driver.cpp').read_text())
  self.assertIn('r*16,16,slot_offset',source);self.assertIn('bi,16,((j/s.bc+1)%2)*s.bc',source)
  self.assertIn('j+s.bc<kend',source);self.assertIn('vbase(slot_offset,s.sets)',source)
if __name__=='__main__':unittest.main()
