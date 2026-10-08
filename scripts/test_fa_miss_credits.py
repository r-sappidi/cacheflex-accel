import copy
import json
from pathlib import Path
import unittest
from run_fa_miss_credits import prior,command,audit_objects

class MissCreditTests(unittest.TestCase):
    def config(self):return json.loads((prior('spm',512)/'config.json').read_text())
    def changed(self,old,capacity):
        new=copy.deepcopy(old)
        for i in range(8):new['system']['ruby'][f'l0_cntrl{i}']['dataMissCredits']['capacity']=capacity
        return new
    def test_only_data_miss_credit_change_allowed(self):
        old=self.config()
        for cap in (8,16,32,64):audit_objects(old,self.changed(old,cap),cap)
    def test_lower_level_change_rejected(self):
        old=self.config();new=self.changed(old,64)
        new['system']['ruby']['l1_cntrl0']['missCredits']['capacity']=64
        with self.assertRaises(AssertionError):audit_objects(old,new,64)
    def test_cache_size_change_rejected(self):
        old=self.config();new=self.changed(old,64)
        new['system']['ruby']['l1_cntrl0']['cache']['size']=131072
        with self.assertRaises(AssertionError):audit_objects(old,new,64)
    def test_partial_core_change_rejected(self):
        old=self.config();new=self.changed(old,64)
        new['system']['ruby']['l0_cntrl7']['dataMissCredits']['capacity']=8
        with self.assertRaises(AssertionError):audit_objects(old,new,64)
    def test_no_inherited_debug_or_wrong_config_position(self):
        for b in ('cache','spm','spm_fast'):
            cmd=command(b,512,64,Path('/tmp/not-executed'))
            self.assertTrue(cmd[2].endswith('/micro_timing_se.py'))
            self.assertFalse(any(x.startswith('--debug') for x in cmd))
            self.assertEqual(sum('.dataMissCredits.capacity=64' in x for x in cmd),8)

if __name__=='__main__':unittest.main()
