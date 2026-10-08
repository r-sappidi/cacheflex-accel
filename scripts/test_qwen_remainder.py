"""AI-12 independent numerical, layout, configuration and budget checks."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_qwen_remainder as run
from qwen_remainder_reference import OPS,generate,validate,tables

class Tests(unittest.TestCase):
    def test_oracle_export_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            for op in OPS:
                for t in (1,25):
                    p=Path(d)/f'{op}{t}.bin';generate(p,op,t,263)
                    ref=np.load(p.with_suffix('.npz'))['expected'];out=Path(d)/'out.bin';ref.tofile(out)
                    self.assertTrue(validate(out,p,op)['full_oracle'])
                    ref[0]=np.float16(float(ref[0])+1);ref.tofile(out)
                    with self.assertRaises(AssertionError):validate(out,p,op)
    def test_rope_identity_and_quarter_turn(self):
        table,freq=tables(25)
        self.assertTrue(np.array_equal(table[0,:32],np.ones(32)))
        self.assertTrue(np.array_equal(table[0,32:],np.zeros(32)))
        self.assertEqual(freq[0],1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rope.bin';generate(p,'rope',25,263)
            x=np.fromfile(p,dtype='<f2',offset=40,count=16*25*64).reshape(16,25,64)
            y=np.load(p.with_suffix('.npz'))['expected'].reshape(16,25,64)
            self.assertTrue(np.array_equal(x[:,0],y[:,0]))
            c,s=table[1,0].astype(np.float16),table[1,32].astype(np.float16)
            self.assertEqual(y[0,1,0],np.float16(np.float16(x[0,1,0]*c)-np.float16(x[0,1,32]*s)))
    def test_embedding_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'embedding.bin';generate(p,'embedding',25,263)
            n=np.fromfile(p,dtype='<u8',count=5);x=np.fromfile(p,dtype='<f2',offset=40,count=int(n[0])).reshape(263,896)
            ids=np.fromfile(p,dtype='<u4',offset=40+int(sum(n[:3]))*2)
            self.assertEqual((ids[0],ids[1],ids[-1]),(0,0,262))
            self.assertTrue(np.array_equal(np.load(p.with_suffix('.npz'))['expected'].reshape(25,896),x[ids]))
    def test_shape_partition(self):
        for op in OPS:
            for t in (1,25,1024,4096):
                for b in run.choices(op):
                    desc=run.describe(op,t,151936,b)
                    self.assertEqual(sum(desc['rows_per_core']),151936 if op=='logits' else t)
                    self.assertEqual(desc['spm_commands'],0)
    def test_config_and_reject_unrelated_delta(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);inp=Path('/input');c=dict(op='rope',arm='spm',t=25,v=263,b=8)
            cmd,args=run.command(p,inp,c)
            self.assertEqual(args,['qwen-remainder','2','25','263','8'])
            cfg=json.loads((run.CONTROL/'config.json').read_text().replace(str(run.CONTROL),str(p)));s=cfg['system']
            s['readfile']=str(inp);s['cpu'][0]['workload'][0]['cmd']=args;s['cpu'][0]['workload'][0]['executable']=str(run.BASE/'guest_v1/guest')
            for i in range(8):s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways']=6
            (p/'config.json').write_text(json.dumps(cfg));run.config_audit(p,run.BASE/'guest_v1/guest',inp,args,'spm')
            s['cpu'][0]['SQEntries']+=1;(p/'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(AssertionError):run.config_audit(p,run.BASE/'guest_v1/guest',inp,args,'spm')
    def test_future_budget_not_just_active(self):
        budget=run.read(run.JOURNAL/'budget.json')
        for remaining in (40,23,22,20,10,0):
            values={'budget.json':budget,'status.json':dict(remaining={'a':remaining},active=[],failed={}),
                    'resources.json':dict(updated=run.time.time(),pressure=False,cpu_pct_of_host=17,user_ram_pct=4)}
            with patch.object(run,'read',side_effect=lambda p:values[Path(p).name]),patch.object(run,'owned',return_value={}):
                slots,evidence=run.capacity()
            self.assertLessEqual(evidence['old_reserved_bytes']+slots*6*run.GIB,budget['ram_limit'] if slots else max(budget['ram_limit'],evidence['old_reserved_bytes']))
            self.assertLessEqual(slots+evidence['old_future_slots'],max(22,evidence['old_future_slots']))
            if remaining>=22:self.assertEqual(slots,0)
    def test_short_and_nonfinite_export_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            inp=Path(d)/'norm.bin';generate(inp,'rmsnorm',25,263);out=Path(d)/'out.bin'
            ref=np.load(inp.with_suffix('.npz'))['expected'];ref[:-1].tofile(out)
            with self.assertRaises(AssertionError):validate(out,inp,'rmsnorm')
            ref[0]=np.nan;ref.tofile(out)
            with self.assertRaises(AssertionError):validate(out,inp,'rmsnorm')
if __name__=='__main__':unittest.main()
