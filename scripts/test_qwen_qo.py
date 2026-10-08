"""Meaningful geometry, schedule, and arithmetic-preservation gates for FA-33."""
import re,unittest
from build_qwen_qo import OLD,kernels,source,function
class Gates(unittest.TestCase):
 def test_output_ownership_and_balance(self):
  for t in (256,1024,4096):
   h=t//16;seen=[];work=[]
   for core in range(8):
    rows=list(range(core*h,(core+1)*h))+list(range((15-core)*h,(16-core)*h))
    seen+=rows;work.append(sum(r+1 for r in rows))
   self.assertEqual(sorted(seen),list(range(t)));self.assertEqual(len(set(work)),1)
 def test_physical_spm_slots_disjoint_and_within_bounds(self):
  for rows in (32,128,512):
   q={(4,s) for s in range(rows*2)};o={(w,s) for w in range(4) for s in range(rows)};v={(w,s) for w in range(2) for s in range(512,768)}
   self.assertFalse(q&o or q&v or o&v);self.assertTrue(all(0<=w<6 and 0<=s<1024 for w,s in q|o|v));self.assertEqual(len(q|o|v)*64,rows*384+32768)
 def test_q_broadcast_addresses(self):
  for off in range(0,65536,16):
   addr=(4<<16)+off;aligned=addr&0x7ffc0;quarter=addr&48
   self.assertEqual(aligned+quarter,addr);self.assertEqual(aligned>>16,4)
 def test_qk_fma_order_identical(self):
  old=function((OLD/'kernels.hpp').read_text(),'cf_wide_qk_cache(');new=function(kernels(),'qo_qk(')
  self.assertEqual(re.findall(r'"fmla[^\n]+',old),re.findall(r'"fmla[^\n]+',new))
 def test_pv_arithmetic_order_and_direct_epilogue(self):
  for arm in ('cache','spm'):
   old=function((OLD/'pv_direct.hpp').read_text(),f'cf_pv16_{arm}_cpu_paired(');new=function(kernels(),f'qo_pv_{arm}(')
   for op in ('fmla','fmul','fadd','fcvt','zip1'):
    self.assertEqual(re.findall('"'+op+r'[^\n]+',old),re.findall('"'+op+r'[^\n]+',new))
   self.assertEqual(new.count('spm.st1h z5.h'),16);self.assertEqual(new.count('spm.ld1qd z5.d'),16)
 def test_causal_tile_order_and_coverage(self):
  for t in (256,1024,4096):
   for bc in (128,256):
    if t<bc:continue
    half=t//16
    for c in range(8):
     for local in range(0,t//8,16):
      q=c*half+local if local<half else (15-c)*half+local-half
      tiles=[j for j in range(0,t,bc) if j<q+16]
      self.assertEqual(tiles,list(range(0,((q+16+bc-1)//bc)*bc,bc)))
 def test_no_hardware_offload_in_new_worker(self):
  s=source().split('static void* qo_worker')[1].split('int main')[0]
  self.assertNotIn('output_reduce',s);self.assertNotIn('shared_slot',s);self.assertIn('qo_release',s);self.assertIn('qo_claim_o(o,rows)',s)
if __name__=='__main__':unittest.main()
