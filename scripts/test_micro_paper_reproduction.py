import copy
import json
from pathlib import Path
import unittest
import numpy as np
import run_micro_paper_reproduction as runner


class ReproductionTests(unittest.TestCase):
    def config(self):
        return json.loads((runner.OLD/'runs/classic_cache_vl16_gate/config.json').read_text())

    def test_exact_script_fu_overrides(self):
        opts=runner.upstream_options()
        values={opts[i+1].split('=')[0]:opts[i+1].split('=')[1] for i,x in enumerate(opts[:-1]) if x=='-P'}
        for index,count in runner.FUS.items():
            self.assertEqual(values[f'system.cpu[0].fuPool.FUList[{index}].count'], str(count))
        self.assertEqual(values['system.cpu[0].spmLoadPorts'], '1')
        self.assertEqual(len(runner.TILES), 6)
        self.assertEqual(set(runner.TILES), {(r,c) for r in (32,64) for c in (128,256,384)})

    def test_paper_config_only_five_counts(self):
        old=self.config(); actual=copy.deepcopy(old)
        cpu=actual['system']['cpu'][0]
        for index,count in runner.FUS.items(): cpu['fuPool']['FUList'][index]['count']=count
        argv=cpu['workload'][0]['cmd']; guest=Path(cpu['workload'][0]['executable'])
        self.assertTrue(runner.config_audit(actual, old, True, argv, guest)['passed'])

    def test_reject_core_and_memory_drift(self):
        for where,key in [('system.l2','size'), ('system.cpu','issueWidth')]:
            old=self.config(); actual=copy.deepcopy(old)
            obj=actual['system']['l2'] if where=='system.l2' else actual['system']['cpu'][0]
            obj[key]+=1
            cpu=actual['system']['cpu'][0]
            with self.assertRaises(AssertionError):
                runner.config_audit(actual, old, False, cpu['workload'][0]['cmd'], Path(cpu['workload'][0]['executable']))

    def test_export_hash_checks_existing_gate(self):
        p=runner.OLD/'runs/classic_cache_vl16_gate'
        arrays,hashes=runner.input_arrays(p,320,1,1)
        self.assertEqual(arrays['O'].shape, (1,320,64))
        self.assertEqual(len(hashes),4)

    def test_reference_matches_existing_gate(self):
        # Intercept progress only, do not write into previous accepted evidence.
        p=runner.OLD/'runs/classic_cache_vl16_gate'
        arrays,_=runner.input_arrays(p,320,1,1)
        save=runner.save
        try:
            runner.save=lambda *args: None
            expected=runner.reference(arrays,128,None)
        finally: runner.save=save
        error=np.abs(arrays['O'].astype(np.float64)-expected.astype(np.float64))
        self.assertTrue(np.all(error <= 2e-6+.01*np.abs(expected.astype(np.float64))))


if __name__=='__main__': unittest.main()
