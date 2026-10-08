"""Projection indexing, work, independent arithmetic and hardware-isolation checks."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from qwen_projection_model import BASE,BASELINE,describe
from qwen_projection_reference import ordered,generate,validate,head_layout,row_layout
from run_qwen_projection_v2 import command,config_audit,CONTROL

class Tests(unittest.TestCase):
 def test_counts(self):
  for t in (1,24,25,1024,4096):
   for op in ('qkv','out'):
    for mp in (1,2,4,8):
     for kc in (128,256,512,1024):
      x=describe(t,op,dict(mp=mp,kc=kc));counts=[]
      for core in x['cores']:
       n=0
       if core['rows']:
        for k0 in range(0,896,kc):n+=4*min(kc,896-k0)*(core['cols']//128)
       counts.append(n)
      self.assertEqual(counts,[r['copies'] for r in x['cores']])
      self.assertEqual(x['totals']['flops'],2*((t+7)//8*8)*896*(1152 if op=='qkv' else 896))
 def test_layout(self):
  a=np.arange(25*1152).reshape(25,1152)
  h=head_layout(a).reshape(18,25,64)
  for head in (0,13,14,15,16,17):self.assertTrue(np.array_equal(h[head],a[:,head*64:(head+1)*64]))
  self.assertTrue(np.array_equal(row_layout(h,25),a))
 def test_reference_and_corruption(self):
  lib=BASE/'reference/ordered_gemm_v2.so'
  a=np.array([[.5,-.75,.125],[.25,.5,-.5]],dtype=np.float16);b=np.array([[.5,-.5],[.25,.125],[-.75,.25]],dtype=np.float16)
  ref=np.zeros((2,2),dtype=np.float32)
  for k in range(3):ref=(ref+a[:,k,None].astype(np.float32)*b[k].astype(np.float32)).astype(np.float32)
  self.assertTrue(np.array_equal(ordered(a,b,lib),ref))
  with tempfile.TemporaryDirectory() as d:
   for op in ('qkv','out'):
    inp=Path(d)/(op+'.bin');generate(inp,25,op,lib)
    y=np.load(inp.with_suffix('.npz'))['expected'];p=Path(d)/'output.bin';y.tofile(p)
    self.assertTrue(validate(p,inp,25,op)['ordered_bits_exact'])
    y[0]+=np.float16(1);y.tofile(p)
    with self.assertRaises(AssertionError):validate(p,inp,25,op)
 def test_config(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);guest=Path('/guest');inp=Path('/input');c=dict(BASELINE,arm='spm',t=25,op='qkv')
   cmd,args=command(p,guest,inp,c,2);self.assertEqual(args,['qwen-projection','0','25','2','128','2'])
   data=json.loads((CONTROL/'config.json').read_text().replace(str(CONTROL),str(p)));s=data['system'];s['readfile']=str(inp)
   s['cpu'][0]['workload'][0]['cmd']=args;s['cpu'][0]['workload'][0]['executable']=str(guest)
   for i in range(8):s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']=6
   (p/'config.json').write_text(json.dumps(data));config_audit(p,guest,inp,args,'spm')
   s['cpu'][0]['SQEntries']+=1;(p/'config.json').write_text(json.dumps(data))
   with self.assertRaises(AssertionError):config_audit(p,guest,inp,args,'spm')
 def test_unchanged_primitive(self):
  old=Path('results/ai_workloads/qwen_swiglu_20260928/guests/custom_g0/mlp_study.cpp').read_text()
  new=(BASE/'guest_v2/projection_core.hpp').read_text().replace('int lda=0,bool convert=true) {','int lda=0) {').replace(' if (!convert) return;\n','')
  self.assertEqual(new,old[:old.index('size_t check(')]+'\n}\n')
if __name__=='__main__':unittest.main()
