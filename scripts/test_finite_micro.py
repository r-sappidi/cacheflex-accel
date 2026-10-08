"""Static routing regressions and negative full-config audit tests."""
import copy,json,re,tempfile,unittest
from pathlib import Path
import run_finite_micro as r

class FiniteTests(unittest.TestCase):
    def test_return_before_both_request_lanes(self):
        text=(r.ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm').read_text()
        self.assertLess(text.index('in_port(returnBufferFromL0_in'),text.index('in_port(spmBufferFromL0_in'))
        self.assertLess(text.index('in_port(returnBufferFromL0_in'),text.index('in_port(messageBufferFromL0_in'))
        ret=text.split('in_port(returnBufferFromL0_in',1)[1].split('in_port(spmBufferFromL0_in',1)[0]
        for msg in ('INV_DATA','INV_ACK','PUTX'):self.assertIn(msg,ret)
        self.assertNotIn('SPM_LD',ret)
    def test_escape_transitions_pop_return(self):
        text=(r.ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm').read_text()
        bodies=re.findall(r'transition\([^\n]*Escape[^\n]*\)\s*\{([^}]+)\}',text)
        self.assertGreaterEqual(len(bodies),10)
        for body in bodies:
            self.assertIn('k_popL0ReturnQueue',body)
            self.assertNotIn('k_spmPopL0RequestQueue',body)
        action=text.split('action(u_writeDataFromEscape',1)[1].split('\n  action(',1)[0]
        self.assertIn('peek(returnBufferFromL0_in',action)
    def test_profile_command_overrides(self):
        for small in (False,True):
            c=r.case('test',small=small,guard=4096)
            cmd,args=r.command(c,Path('/tmp/config-test'))
            self.assertIn('--cacheflex-finite-resources',cmd)
            for i in range(8):
                self.assertIn(f'system.cpu[{i}].cacheFlexBoundedTracking=True',cmd)
                self.assertIn(f'system.cpu[{i}].LQEntries={8 if small else 32}',cmd)
                self.assertIn(f'system.ruby.l0_cntrl{i}.sequencer.max_outstanding_requests=4096',cmd)
    def test_unrelated_capacity_or_latency_rejected(self):
        source=r.OUT/'runs/default_cache'
        if not (source/'config.json').exists():self.skipTest('default configuration not yet generated')
        c=r.case('default_cache',finite=False);_,args=r.command(c,source)
        for field,value in [('size',1048576),('tagAccessLatency',9)]:
            data=json.loads((source/'config.json').read_text())
            data['system']['ruby']['l1_cntrl0']['cache'][field]=value
            with tempfile.TemporaryDirectory(prefix='finite-config-audit-') as tmp:
                p=Path(tmp)
                (p/'config.json').write_text(json.dumps(data).replace(str(source),str(p)))
                with self.assertRaises(AssertionError):r.audit(p,c,args)
    def test_instantiated_finite_resources(self):
        source=r.BASE/'finite_model_20260926/experiment_v2/runs/roundtrip/config.json'
        if not source.exists():self.skipTest('finite configuration not generated')
        data=json.loads(source.read_text())
        flat=dict(r.leaves(data))
        for k,v in flat.items():
            if not k.startswith('system.ruby.'):continue
            parent=k.rsplit('.',1)[0]
            if k.endswith('.buffer_size') and flat.get(parent+'.type')=='MessageBuffer':
                expected=12 if re.fullmatch(r'system.ruby.l0_cntrl\d+.bufferToL1',parent) else 4 if parent.endswith('.returnBufferFromL0') else 16
                self.assertEqual(v,expected,k)
            if k.endswith('.number_of_TBEs'):
                self.assertEqual(v,16 if '.l0_cntrl' in k else 32 if '.l1_cntrl' in k else 64,k)
            if k.endswith('Credits.capacity'):
                expected=32 if 'microLlcMissCredits' in k else 16 if 'microLlcWriteCredits' in k else 20 if re.search(r'l1_cntrl\d+.missCredits',k) else 8
                self.assertEqual(v,expected,k)
        for i in range(8):
            seq=data['system']['ruby'][f'l0_cntrl{i}']['sequencer']
            self.assertTrue(seq['cacheflex_finite_ingress'])
            self.assertEqual(seq['cacheflex_max_line_targets'],16)
            self.assertTrue(data['system']['cpu'][i]['cacheFlexBoundedTracking'])

if __name__=='__main__':unittest.main()
