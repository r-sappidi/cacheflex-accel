"""Static ABI/resource invariants; runtime gates remain mandatory."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def source(p):return (ROOT/p).read_text()
class Tests(unittest.TestCase):
    def test_default_off(self):
        s=source('gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm')
        self.assertIn('bool l3_reduction_pipeline := "False"',s)
        self.assertIn('bool l3_reduction_fp16 := "False"',s)
        self.assertIn('assert(l3ReductionActive <= l3_reduction_entries)',s)
    def test_real_port_and_alu_bound(self):
        s=source('gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm')
        self.assertIn('!l3ReductionIssue.isSet',s)
        self.assertIn('alu_start < l3ReductionAluNext',s)
        self.assertIn('alu_start + cyclesToTicks(l3_reduction_interval)',s)
        self.assertGreaterEqual(s.count('l3Occupy(256)'),2)
        self.assertLess(s.index('in_port(l3ReductionIssue_in'),s.index('in_port(l3Reduction_in'))
    def test_full_destination_hazards(self):
        s=source('gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm')
        self.assertIn('if (in_msg.L3Op == 7) { footprint := 256; }',s)
        s=source('gem5/src/mem/ruby/system/SequencerReduction.cc')
        self.assertIn('return 256; // A compact contribution',s)
        self.assertIn('i < 4; ++i) ++reductionPending',s)
    def test_real_compact_payload(self):
        s=source('gem5/src/mem/ruby/system/SequencerReduction.cc')
        self.assertIn('c.halfContribution ? 128 : 256',s)
        self.assertIn('reductionTransportBytes+=next->getSize()+16',s)
        s=source('gem5/src/mem/ruby/network/Network.cc')
        self.assertIn('return m_control_msg_size + 136;',s)
    def test_exact_widen_nonfused(self):
        s=source('gem5/src/mem/ruby/structures/CacheMemory.cc')
        self.assertIn('add=f16_to_f32(narrow)',s)
        self.assertIn('f32_add(f32_mul(old,float32_t{uint32_t(scale)}),add)',s)
    def test_typed_header_nonoverlap(self):
        for alpha in (0,0x3f400003,0xbf000001,0xffffffff):
            for bank in range(8):
                a=(1<<60)|(1<<59)|(1<<58)|(alpha<<24)|(bank<<20)|(3<<16)|(511<<6)
                self.assertEqual((a>>24)&0xffffffff,alpha)
                self.assertEqual((a>>20)&7,bank)
                self.assertEqual((a>>16)&7,3)
                self.assertEqual((a>>6)&1023,511)
if __name__=='__main__':unittest.main()
