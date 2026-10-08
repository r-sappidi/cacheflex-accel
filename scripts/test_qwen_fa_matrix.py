import re,unittest,json
from pathlib import Path
from qwen_fa_matrix_plan import *
from build_qwen_fa_matrix import source,KV_WORKER,OLD,function,kernels

class MatrixTests(unittest.TestCase):
 def test_original_factorial_counts_preserved(self):
  p=Path('results/cacheflex_flash_attention/qwen_qo_ring_20260930/experiment_v3/status.json')
  for row in json.loads(p.read_text())['completed']:
   c=dict(row['case'],br=16,hq=14,hkv=2);m=describe(c)
   self.assertEqual({k:m[k] for k in ('tasks','updates','copies')},row['work'])
   self.assertEqual(sum(m['copies']),row['copies'])
 def test_all_targets_and_gates_valid_unique(self):
  grid=anchors()+axial()[0]
  self.assertEqual(len(grid),len({name(c) for c in grid}))
  self.assertEqual(set(c['mode'] for c in grid),set(range(12)))
  for c in grid:describe(c);describe(gate(c))
 def test_head_ownership_no_duplicates_for_each_geometry(self):
  for r in (128,256,512):
   seen=[]
   for core in range(8):
    for sl in range(4096//8//r):
     for start in (core*256+sl*(r//2),(15-core)*256+sl*(r//2)):seen+=range(start,start+r//2)
   self.assertEqual(sorted(seen),list(range(4096)))
 def test_qv_ov_counts_are_separable(self):
  v=describe(case(3));qv=describe(case(6));ov=describe(case(7));qov=describe(case(5))
  self.assertEqual(sum(qv['copies'])-sum(v['copies']),14*4096*2)
  self.assertEqual(sum(ov['copies'])-sum(v['copies']),14*4096*4)
  self.assertEqual(sum(qov['copies']),sum(qv['copies'])+sum(ov['copies'])-sum(v['copies']))
 def test_kv_payload_once_and_spm_footprint(self):
  a=describe(case(10));b=describe(case(11))
  self.assertEqual(a['copy_payload_bytes'],2*4096*64*2*2)
  self.assertEqual(a['copy_payload_bytes'],b['copy_payload_bytes'])
  self.assertEqual(a['spm_bytes'],128*1024);self.assertEqual(b['spm_bytes'],256*1024)
 def test_causal_kv_work_independent_of_merge_batch(self):
  a=describe(case(9,r=16));b=describe(case(9,r=256))
  self.assertEqual(a['updates'],b['updates']);self.assertEqual(a['computed_pairs'],b['computed_pairs'])
  self.assertEqual(sum(a['tasks']),16*sum(b['tasks']))
 def test_static_slot_layouts_disjoint(self):
  for t in (2048,4096):
   for heads in (1,2):
    k={(w,s) for w in range(4) for s in range(heads*(t//8)//128*64)}
    v={(w,s) for w in (4,5) for s in range(heads*t//8)}
    self.assertFalse(k&v);self.assertTrue(all(w<6 and s<1024 for w,s in k|v))
 def test_qk_arithmetic_identical_for_resident_k(self):
  text=(OLD/'kernels.hpp').read_text()
  a=function(text,'cf_wide_qk_cache(');b=function(text,'cf_wide_qk_spm(')
  self.assertEqual(re.findall(r'"fmla[^\n]+',a),re.findall(r'"fmla[^\n]+',b))
 def test_new_source_policies(self):
  s=source()
  self.assertIn('qresident=layout_mode==4||layout_mode==5||layout_mode==6',s)
  self.assertIn('oresident=layout_mode==4||layout_mode==5||layout_mode==7',s)
  self.assertIn('if(lp==0)continue',s)
  self.assertIn('expf(s.maximum.p[p*rows+r]-mx)',s)
  self.assertNotIn('output_reduce',KV_WORKER)
 def test_interactions_and_neighbors_are_bounded_and_new(self):
  rows=[dict(name=name(c),case=c,roi_us=10000+i) for i,c in enumerate(anchors()+axial()[0])]
  selected,audit=interactions(rows)
  self.assertLessEqual(len(selected),48)
  self.assertFalse({name(c) for c in selected}&{r['name'] for r in rows})
  for c in selected:describe(c)
  n,a=neighbors(rows);self.assertLessEqual(len(n),24)
 def test_no_invalid_kv_boundary_tile(self):
  with self.assertRaises(AssertionError):describe(case(10,bc=512))

if __name__=='__main__':unittest.main()
