"""Audit real one-/four-core config serialization, including shared Process refs."""
import json
from pathlib import Path
import tempfile
import unittest
from run import BASE,case,audit

class ConfigTests(unittest.TestCase):
    def test_shared_process_serialization(self):
        for n,t,br,g in ((1,384,32,5),(4,1024,32,3)):
            c=case(t,n,False,False,br,g,l2=64,stage='gate')
            p=BASE/'experiment_v2/runs'/f'gate_t{t}_n{n}_q_cache_br{br}_g{g}_l264'
            audit(c,p)

    def test_cache_size_change_rejected(self):
        c=case(384,1,False,False,32,5,l2=64,stage='gate')
        p=BASE/'experiment_v2/runs/gate_t384_n1_q_cache_br32_g5_l264'
        data=json.loads((p/'config.json').read_text())
        data['system']['ruby']['l1_cntrl0']['cache']['size']=131072
        with tempfile.TemporaryDirectory() as tmp:
            q=Path(tmp);(q/'config.json').write_text(json.dumps(data))
            with self.assertRaises(AssertionError):audit(c,q)

if __name__=='__main__':unittest.main()
