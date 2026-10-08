"""Static invariants supplement full simulated arithmetic/coherence gates."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def read(p):return (ROOT/'gem5/src'/p).read_text()
class Tests(unittest.TestCase):
    def test_flags_do_not_overlap_scalar_or_slot(self):
        for scalar in (0,0x3f400003,0xffffffff):
            a=(1<<60)|(1<<59)|(1<<58)|(1<<57)|(1<<56)|(scalar<<24)|(7<<20)|(3<<16)|(511<<6)
            self.assertEqual((a>>24)&0xffffffff,scalar)
            self.assertEqual((a>>20)&7,7)
            self.assertEqual((a>>16)&7,3)
    def test_opt_in(self):
        s=read('mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm')
        self.assertIn('bool l3_reduction_initialize := "False"',s)
        self.assertIn('bool l3_reduction_finalize := "False"',s)
        self.assertIn('assert(l3_reduction_finalize && in_msg.Len == 128)',s)
    def test_transport_and_hazards(self):
        s=read('mem/ruby/system/Sequencer.cc')
        self.assertIn('pkt->getSize()!=sizeof(Addr) || (pkt->getAddr() & 63)',s)
        s=read('mem/request.hh')
        self.assertIn('shared && isSPMWBStore()',s)
        self.assertIn('dst_spm_addr |= finalize',s)
        s=read('mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm')
        self.assertEqual(s.count('out_msg.L3Finalize := in_msg.L3Finalize'),2)
        s=read('mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm')
        self.assertIn('out_msg.SrcAddr := in_msg.L3Finalize',s)
        self.assertIn('MessageSizeType:SPM_FinalizeControl',s)
        s=read('cpu/o3/lsq_unit.cc')
        self.assertGreaterEqual(s.count('wayCount = 2'),2)
        self.assertIn('WB RAW defer',s)
        self.assertIn('auto it = spm.storeQueue.begin()',s)
        self.assertIn('older->seqNum >= inst->seqNum',s)
        s=read('mem/ruby/system/SequencerReduction.cc')
        self.assertIn('cfReduction::Finalize',s)
        self.assertIn('return 128',s)
    def test_bounded_finalization_and_release(self):
        s=read('mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm')
        body=s.split('action(l3_finalizeFinish,')[1].split('action(l3_reduceFinish,')[0]
        self.assertIn('enqueue(responseL2Network_out, ResponseMsg, l3_spm_latency)',body)
        self.assertIn('out_msg.L3Op := 5',body)
        self.assertIn('l3ReductionRelease(address, 128)',body)
        self.assertIn('l3Occupy(128)',body)
        self.assertIn('l3ReductionCount := l3ReductionCount - 1',body)
        self.assertIn('TBEs.deallocate',body)
    def test_exact_softfloat_finalize(self):
        s=read('mem/ruby/structures/CacheMemory.cc')
        body=s.split('CacheMemory::l3FinalizeResult')[1].split('\n}',1)[0]
        self.assertIn('readSPMData',body)
        self.assertIn('f32_to_f16',body)
        self.assertIn('f32_mul',body)
        self.assertIn('softfloat_round_near_even',body)
        self.assertNotIn('functional',body)
if __name__=='__main__':unittest.main()
