"""Source-contract regressions; simulator gates remain mandatory."""
import unittest
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'gem5/src'

class TranslationContracts(unittest.TestCase):
    def test_override_serialization(self):
        # Load only the pure serializer; no simulator/NumPy imports required.
        source=ast.parse((ROOT/'scripts/run_micro_resources.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef)
                and n.name=='param_assignment')
        namespace={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<serializer>','exec'),namespace)
        for value in ('524288',1024,False,True):
            assignment=namespace['param_assignment']('size',value)
            actual=ast.literal_eval(ast.parse(assignment).body[0].value)
            self.assertEqual(actual,value)
            self.assertIs(type(actual),type(value))

    def test_independent_spm_consumes_queue(self):
        s=(SRC/'mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm').read_text()
        a=s.index('action(micro_forwardSPM,')
        b=s.index('action(f_sendFlushToL1,',a)
        self.assertNotIn('check_allocate(downstream',s[a:b])
        self.assertIn('microSpm_in.dequeue(clockEdge());',s)
        a=s.index('}, MicroSPM)')
        self.assertIn('micro_popSPM;',s[a:a+100])

    def test_credit_lifetimes(self):
        for level in ('L1','L2'):
            s=(SRC/f'mem/ruby/protocol/MESI_Three_Level_SPM-{level}cache.sm').read_text()
            self.assertIn('missCredits.allocate(address);',s)
            self.assertIn('writeCredits.allocate(address);',s)
            a=s.index('action(s_deallocateTBE,')
            self.assertIn('missCredits.release(address);',s[a:a+600])
            self.assertIn('writeCredits.release(address);',s[a:a+600])

    def test_pooled_llc(self):
        s=(ROOT/'scripts/micro_resource_se.py').read_text()
        self.assertIn('microLlcMissCredits = pool(32)',s)
        self.assertIn('llc.missCredits = ruby_system.microLlcMissCredits',s)
        self.assertNotIn('32 // options.num_l2caches',s)

    def test_bridge_uses_l1_write_buffer(self):
        s=(SRC/'mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm').read_text()
        a=s.index('{SPMCP_fetch, SPMWB_store})')
        self.assertIn('micro_bridgeWrite;',s[a:a+160])
        s=(SRC/'mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm').read_text()
        a=s.index('action(k_popL0RequestQueue,')
        self.assertIn('l0WriteCredits.release(in_msg.addr)',s[a:a+400])

    def test_sequencer_is_guard_not_model(self):
        s=(ROOT/'scripts/run_micro_resources.py').read_text()
        self.assertIn("cacheFlexBoundedTracking'] = False",s)
        self.assertIn("4096 if kind == 'large_guard' else 1024",s)
        self.assertIn("stats == numeric_stats(base / 'stats.txt')",s)

    def test_writeback_completion_independent_of_profile(self):
        s=(SRC/'cpu/o3/lsq_unit.cc').read_text()
        self.assertIn('if (pkt->SPMresp || pkt->req->isSPMWBStore())',s)
        self.assertNotIn('boundedTracking && pkt->req->isSPMWBStore()',s)
        s=(SRC/'mem/ruby/system/Sequencer.cc').read_text()
        self.assertIn('const Addr spm_slot = pkt->req->getSPMDstAddr();',s)

if __name__ == '__main__': unittest.main()
