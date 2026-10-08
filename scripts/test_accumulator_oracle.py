import unittest
from pathlib import Path
from profile_swiglu_accumulator_pcs import select, OLD


class OracleSelectionTests(unittest.TestCase):
    def test_exact_frozen_microkernel_loads(self):
        for arch,first,last in [('cache',0x4019f0,0x401a90),('spm',0x4014f4,0x401594)]:
            loads,stores=select(OLD/'builds/custom_g0'/arch)
            self.assertEqual((loads[0]['pc'],loads[-1]['pc']),(first,last))
            self.assertEqual(len({x['pc'] for x in loads+stores}),32)
            self.assertTrue(all(x['instruction'].startswith('ld1w') for x in loads))

    def test_source_retains_real_request_and_verification(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'gem5/src/mem/ruby/system/Sequencer.cc').read_text()
        body=text.split('Sequencer::tryEarlyRead(PacketPtr pkt)')[1].split('Sequencer::tryIdealStaging')[0]
        self.assertIn('m_RequestTable.count(line)',body)
        self.assertIn('m_ruby_system->functionalRead(&probe)',body)
        self.assertIn('background->senderState = nullptr',body)
        self.assertIn('issueRequest(background, RubyRequestType_LD)',body)
        self.assertNotIn('markRemoved',body)
        self.assertIn('Early accumulator read differed from coherent response',text)
        params=(root/'gem5/src/mem/ruby/system/Sequencer.py').read_text()
        self.assertIn('cacheflex_early_read_pcs = VectorParam.Addr([]',params)


if __name__=='__main__':unittest.main()
