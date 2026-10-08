"""Static/ABI guards supplement, never replace, directed simulator tests."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_port_wakeup_precedes_internal_work(self):
        s=(ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm').read_text()
        self.assertLess(s.index('in_port(l3ServicePort_in'),s.index('in_port(l3ReductionTimer_in'))
        self.assertLess(s.index('in_port(l3ServicePort_in'),s.index('in_port(l3Reduction_in'))
    def test_header_roundtrip(self):
        for bank in range(8):
            for scale in (0,0x3f400003,0x3f800000,0xbf000001,0x7f7fffff):
                addr=(1<<60)|(1<<59)|(scale<<24)|(bank<<20)|(3<<16)|(127<<6)
                self.assertEqual((addr>>24)&0xffffffff,scale)
                self.assertEqual((addr>>20)&7,bank)
                self.assertEqual((addr>>16)&7,3)
                self.assertEqual((addr>>6)&1023,127)
    def test_no_remote_read_modify_write(self):
        s=(ROOT/'gem5/src/mem/ruby/system/SequencerReduction.cc').read_text()
        body=s.split('void Sequencer::serviceL3Reduction()')[1].split('void Sequencer::completeL3Reduction')[0]
        self.assertNotIn('MemCmd::ReadReq',body)
        self.assertIn('Request::SPMST',body)
        self.assertIn('l3ReductionFlight.size() < reductionSinkCap',body)
    def test_scalar_not_aligned_away(self):
        s=(ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L0cache.sm').read_text()
        self.assertEqual(s.count('out_msg.SrcAddr := in_msg.SrcAddr;'),2)
    def test_delayed_install_and_bank_port(self):
        s=(ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm').read_text()
        begin=s.split('action(l3_reduceStart,')[1].split('action(l3_reduceFinish,')[0]
        self.assertNotIn('writeSPMData',begin)
        self.assertIn('Cycles arithmetic := l3_reduction_arithmetic',begin)
        self.assertIn('Cycles delay := l3_spm_latency',begin)
        self.assertIn('delay := delay + arithmetic',begin)
        self.assertIn('l3Occupy(256)',begin)
        finish=s.split('action(l3_reduceFinish,')[1].split('action(l3_access,')[0]
        self.assertIn('l3Occupy(256)',finish)
        self.assertIn('l3ReductionRelease',finish)
    def test_full_wire_payload(self):
        s=(ROOT/'gem5/src/mem/ruby/network/Network.cc').read_text()
        self.assertIn('return m_control_msg_size + 264;',s)
    def test_barrier_credit_until_install(self):
        s=(ROOT/'gem5/src/mem/ruby/system/SequencerReduction.cc').read_text()
        body=s.split('void Sequencer::completeL3Reduction')[1]
        self.assertIn('cfReduction::pending.find',body)
        self.assertIn('reductionPending.erase',body)
        self.assertIn('reductionCompleted',body)
if __name__=='__main__':unittest.main()
