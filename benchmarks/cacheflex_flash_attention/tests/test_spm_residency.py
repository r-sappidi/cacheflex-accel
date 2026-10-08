from pathlib import Path
import csv,json,tempfile,unittest
from analyze_spm_residency import qualify,spm_phase
class Tests(unittest.TestCase):
 def test_protocol_and_store_disambiguation(self):
  for store in (False,True):
   with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);row=dict(tick='100',cache='system.ruby.l1_cntrl0.cache',addr='0x1000',buffer='core0.S',next_read_tick='120')
    (p/'residency.json').write_text(json.dumps({'bounds':{'BEGIN':50,'END':200}}))
    with (p/'evictions.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    lines=['1: page: CFRES MAP va=0x4000 pa=0x1000 bytes=4096',
     '100: system.ruby.l1_cntrl0.cache: CFRES REMOVE addr=0x1000',
     '100 0 L1Cache L1_Replacement M>I [0x1000, line 0x1000]']
    if store:lines.append('110: system.cpu0.commit: CFRES RETIRED va=0x4000 pc=0x8000 bytes=64 store=1 spm=0')
    lines.append('140: system.cpu0.commit: CFRES RETIRED va=0x4000 pc=0x8000 bytes=64 store=0 spm=0')
    (p/'debug.trace').write_text('\n'.join(lines))
    r=qualify(p);self.assertEqual(r['unresolved_protocol'],0)
    key=next(iter(r['counts']))
    self.assertIn('write_before_read_ambiguous' if store else 'post_removal_request_and_retired_read_without_store',key)
 def test_phase(self):
  self.assertEqual(spm_phase([('spm_kernel_1vl_vl16','file:1')]),'QK')
if __name__=='__main__':unittest.main()
