import unittest
from pathlib import Path
class Plan(unittest.TestCase):
 def test_ownership(self):
  for t,r,hq in [(1024,128,2),(1024,128,14),(4096,256,14)]:
   groups=t//r;seen=set()
   for core in range(8):
    for task in range(core,hq*groups,8):
     h=task//groups;g=(task%groups+h)%groups
     self.assertEqual((g-h)%groups%8,core);self.assertNotIn((h,g),seen);seen.add((h,g))
   self.assertEqual(len(seen),hq*groups)
 def test_rotation_and_copy_counts(self):
  for t,bc,r,hq in [(1024,128,128,2),(4096,256,256,14)]:
   counts=[0,0];moves=0
   for head in range(hq):
    for taskround in range(t//r//8):
     for wave in range(0,t,8*bc):
      payload=list(range(8))
      for step in range(8):
       if step:payload=payload[-1:]+payload[:-1];moves+=1
       for core in range(8):
        self.assertEqual(payload[core],(core-step)%8)
        counts[0]+=2*bc
        if step==0:counts[1]+=2*bc
   self.assertEqual(counts,[hq*(t//r)*2*t,hq*(t//r)*(t//4)])
   self.assertEqual(moves,hq*(t//r//8)*(t//(8*bc))*7)
 def test_generated_kernel(self):
  root=Path(__file__).resolve().parents[4]
  s=(root/'results/cacheflex_flash_attention/v_only_ring_20261003/guest_v1/driver.cpp').read_text();s=s[s.index('static void* vring_worker'):s.index('int main(')]
  self.assertIn('cf_wide_qk_cache(',s);self.assertNotIn('ring_stage_k',s)
  self.assertIn('s.spm?768UL:',s);self.assertIn('if(j>=kend)continue;',s)
  self.assertIn('for(int task=c.id;task<s.hq*groups;task+=s.cores)',s)
  self.assertIn('m5_sum(0x52494e47,1,0,768,bc,2)',s)
if __name__=='__main__':unittest.main()
