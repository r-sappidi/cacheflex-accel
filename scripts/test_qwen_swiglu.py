"""Discriminating checks for Qwen tails, analytical accounting and config audit."""
import json
from pathlib import Path
import tempfile
import unittest
from qwen_swiglu_model import describe,BASELINE
from run_qwen_swiglu import command,config_audit,CONTROL


class QwenSwiGLUTests(unittest.TestCase):
    def test_qwen_column_imbalance_and_tails(self):
        x=describe(1024,896,4864,BASELINE)
        self.assertAlmostEqual(x['flops_imbalance'],(2*(40/38)+(8/7))/3)
        down=[a['rows']*a['cols'] for a in x['ownership'] if a['n']==896]
        self.assertAlmostEqual(max(down)/(sum(down)/8),8/7)
        y=describe(1024,896,4864,dict(mp=8,kc=1024,g=0))
        self.assertEqual(y['flops_imbalance'],1)
        self.assertEqual(y['spm_panel_bytes'],256*1024)
        self.assertEqual(y['totals']['flops'],6*1024*896*4864)

    def test_copy_counts_against_direct_loop_enumeration(self):
        for t in (1,24,1024):
            for mp in (1,2,4,8):
                for kc in (128,256,512,1024):
                    for g in (0,1):
                        x=describe(t,896,4864,dict(mp=mp,kc=kc,g=g))
                        counts=[0]*8
                        for a in x['ownership']:
                            if not a['rows']:continue
                            for k0 in range(0,a['k'],kc):
                                counts[a['core']]+=4*min(kc,a['k']-k0)*(a['cols']//128)
                        self.assertEqual(counts,[a['copies'] for a in x['cores']])

    def test_kc_traffic_combination_and_empty_partitions(self):
        a=describe(24,896,4864,dict(mp=8,kc=128,g=0))
        b=describe(24,896,4864,dict(mp=8,kc=1024,g=1))
        self.assertEqual(a['totals']['copies'],b['totals']['copies'])
        self.assertEqual(sum(x['flops']>0 for x in b['cores']),3)
        self.assertLess(b['totals']['accumulator_bytes'],a['totals']['accumulator_bytes'])
        self.assertLess(b['totals']['a_pack_bytes'],a['totals']['a_pack_bytes'])

    def test_full_config_rejects_unrelated_hardware_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);guest=Path('/guest');inp=Path('/input')
            c=dict(BASELINE,arm='spm',t=24,d=896,f=4864)
            cmd,args=command(p,guest,inp,c,2)
            self.assertIn('--l1d_num_spm_ways=6',cmd)
            data=json.loads((CONTROL/'config.json').read_text().replace(str(CONTROL),str(p)))
            s=data['system'];s['readfile']=str(inp)
            for cpu in s['cpu']:
                if isinstance(cpu['workload'][0],dict):
                    cpu['workload'][0]['cmd']=args;cpu['workload'][0]['executable']=str(guest)
            for i in range(8):s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']=6
            (p/'config.json').write_text(json.dumps(data));config_audit(p,guest,inp,args,'spm')
            s['cpu'][3]['LQEntries']+=1
            (p/'config.json').write_text(json.dumps(data))
            with self.assertRaises(AssertionError):config_audit(p,guest,inp,args,'spm')


if __name__=='__main__':unittest.main()
