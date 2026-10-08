import json
import unittest
from pathlib import Path
from fa_headroom_profiles import parameters,transformed,audit_objects
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'results/cacheflex_flash_attention/ifetch_edge_20260923/experiment/runs/spm_t512_edge1/config.json'

class HeadroomTests(unittest.TestCase):
    def setUp(self):self.c=json.loads(CONFIG.read_text())
    def test_disjoint_and_exact_composition(self):
        e=parameters(self.c,'execution');m=parameters(self.c,'memory')
        self.assertFalse(set(e)&set(m));self.assertEqual(parameters(self.c,'combined'),e|m)
    def test_all_profiles_audited(self):
        for v in ('control','execution','memory','combined'):
            audit_objects(self.c,transformed(self.c,v),v)
    def test_no_capacity_or_dram_change(self):
        for v in ('execution','memory','combined'):
            n=transformed(self.c,v)
            self.assertEqual(n['system']['mem_ctrls'],self.c['system']['mem_ctrls'])
            for i in range(8):
                for level,cache in (('l0','Dcache'),('l1','cache'),('l2','L2cache')):
                    self.assertEqual(n['system']['ruby'][f'{level}_cntrl{i}'][cache],
                                     self.c['system']['ruby'][f'{level}_cntrl{i}'][cache])
    def test_core_widths_within_frozen_limit(self):
        n=transformed(self.c,'execution')
        for cpu in n['system']['cpu']:
            self.assertEqual(cpu['issueWidth'],12);self.assertEqual(cpu['LQEntries'],32)
            self.assertEqual(cpu['numPhysCCRegs'],5*cpu['numPhysIntRegs'])
            self.assertEqual(cpu['fuPool']['FUList'][9]['count'],4)
    def test_missing_change_rejected(self):
        with self.assertRaises(AssertionError):audit_objects(self.c,self.c,'combined')
    def test_unrelated_change_rejected(self):
        n=transformed(self.c,'memory');n['system']['ruby']['l1_cntrl0']['cache']['size']*=2
        with self.assertRaises(AssertionError):audit_objects(self.c,n,'memory')

if __name__=='__main__':unittest.main()
