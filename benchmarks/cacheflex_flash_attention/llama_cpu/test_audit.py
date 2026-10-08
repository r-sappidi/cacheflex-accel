"""Positive/negative config checks and independent numerical checker tests."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from run import CONTROL,audit,numerical
class AuditTests(unittest.TestCase):
    def config(self,mutate):
        raw=(CONTROL/'config.json').read_text()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cfg=json.loads(raw.replace(str(CONTROL),str(p)))
            proc=cfg['system']['cpu'][0]['workload'][0]
            proc['executable']='/test/guest';proc['cmd']=['llama-fa','128','8']
            mutate(cfg);(p/'config.json').write_text(json.dumps(cfg))
            return audit(p,Path('/test/guest'),['128','8'])
    def test_guest_only_allowed(self):
        self.assertTrue(self.config(lambda c:None)['matched_all_fields_except_guest'])
    def test_cpu_change_rejected(self):
        with self.assertRaises(AssertionError):self.config(lambda c:c['system']['cpu'][0].update(numROBEntries=256))
    def test_l2_change_rejected(self):
        with self.assertRaises(AssertionError):self.config(lambda c:c['system']['ruby']['l1_cntrl0']['cache'].update(size=1048576))
    def test_numerical_rejects_corruption(self):
        ref=np.ones((2,64),dtype=np.float64)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);ref.astype(np.float16).tofile(p/'output.fp16');ref.astype(np.float32).tofile(p/'output.fp32')
            numerical(p,ref)
            bad=ref.copy();bad[0,0]=2;bad.astype(np.float16).tofile(p/'output.fp16')
            with self.assertRaises(AssertionError):numerical(p,ref)
if __name__=='__main__':unittest.main()
