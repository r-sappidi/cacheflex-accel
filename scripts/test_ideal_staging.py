import copy
import json
from pathlib import Path
import unittest
import run_ideal_staging as r

class IdealStagingTests(unittest.TestCase):
    def fixture(self, variant):
        a = json.loads((r.prior('spm',512)/'config.json').read_text())
        b = copy.deepcopy(a)
        for i in range(8):
            seq = b['system']['ruby'][f'l0_cntrl{i}']['sequencer']
            seq['cacheflex_ideal_staging_cache'] = None if variant=='control' else f'system.ruby.l1_cntrl{i}.cache'
            seq['cacheflex_ideal_staging_controller'] = None if variant=='control' else f'system.ruby.l1_cntrl{i}'
            if variant=='ideal_local': b['system']['ruby'][f'l1_cntrl{i}']['spm_access_response_latency']=1
        return a,b

    def test_config_contract(self):
        for v in ('control','ideal','ideal_local'): r.audit_configs(*self.fixture(v),v)

    def test_rejects_added_capacity(self):
        a,b=self.fixture('ideal'); b['system']['ruby']['l1_cntrl0']['cache']['size']*=2
        with self.assertRaises(AssertionError): r.audit_configs(a,b,'ideal')

    def test_rejects_wrong_private_target(self):
        a,b=self.fixture('ideal')
        b['system']['ruby']['l0_cntrl0']['sequencer']['cacheflex_ideal_staging_cache']='system.ruby.l1_cntrl1.cache'
        with self.assertRaises(AssertionError): r.audit_configs(a,b,'ideal')

    def test_same_workload_and_sram(self):
        for v in ('control','ideal','ideal_local'):
            cmd=r.command('spm',4096,v,Path('/tmp/cf-oracle-test'))
            for arg in ('--num-cpus=8','--l1d_size=64kB','--l1d_num_spm_ways=2','--cacheflex-micro-timing'):
                self.assertIn(arg,cmd)
            self.assertEqual('--cacheflex-ideal-staging' in cmd,v!='control')
            self.assertNotIn('spm_separate_backing=True',' '.join(cmd))

    def test_new_counters_cannot_hide_control_difference(self):
        r.compare_control({'a':4},{'a':4,'seq.m_idealStagingCopies':0})
        with self.assertRaises(AssertionError): r.compare_control({'a':4},{'a':5})
        with self.assertRaises(AssertionError): r.compare_control({'a':4},{'a':4,'seq.m_idealStagingCopies':1})

if __name__=='__main__': unittest.main()
