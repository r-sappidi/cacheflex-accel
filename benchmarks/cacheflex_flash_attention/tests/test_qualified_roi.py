"""Post-ROI correctness reads must not qualify measured-region evictions."""
import csv,json,tempfile,unittest
from pathlib import Path
from analyze_spm_residency import qualify
class Bounds(unittest.TestCase):
    def case(self,read_tick):
        with tempfile.TemporaryDirectory(prefix='cf-reuse-boundary-') as d:
            p=Path(d);(p/'residency.json').write_text(json.dumps({'bounds':{'BEGIN':50,'END':200}}))
            row=dict(tick=100,cache='system.ruby.l1_cntrl0.cache',addr='0x2000',buffer='O',next_read_tick=read_tick)
            with (p/'evictions.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerow(row)
            (p/'debug.trace').write_text('1: system: CFRES MAP va=0x1000 pa=0x2000 bytes=4096\n100: system.ruby.l1_cntrl0.cache: CFRES REMOVE addr=0x2000\n'+f'{read_tick}: system.cpu0.commit: CFRES RETIRED va=0x1000 bytes=64 spm=0 store=0\n')
            return qualify(p)
    def test_after_end_excluded(self):
        r=self.case(250);self.assertTrue(next(iter(r['counts'])).endswith('no_observed_retired_read'))
    def test_inside_roi_included(self):
        r=self.case(150);self.assertTrue(next(iter(r['counts'])).endswith('post_removal_request_and_retired_read_without_store'))
if __name__=='__main__':unittest.main()
