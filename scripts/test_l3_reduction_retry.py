"""Source guards for the narrow retry-only intervention; runtime gates separate."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_default_off(self):
        s=(ROOT/'gem5/src/cpu/o3/BaseO3CPU.py').read_text()
        self.assertIn('cacheFlexIndependentReduction = Param.Bool(False',s)
        ports=s.split('_cached_ports =')[1].split('\n')[0]
        self.assertNotIn('reduction_port',ports)
    def test_no_coherent_flag_release(self):
        s=(ROOT/'gem5/src/cpu/o3/lsq.cc').read_text()
        body=s.split('LSQ::recvReductionRetry()')[1].split('\n}')[0]
        self.assertIn('reductionBlocked(false)',body)
        for forbidden in ('cacheBlocked(', 'spmBlocked(', 'bridgeBlocked(', 'recvReqRetry('):
            self.assertNotIn(forbidden,body)
    def test_only_command_uses_new_interface(self):
        s=(ROOT/'gem5/src/cpu/o3/lsq_unit.cc').read_text()
        body=s.split('LSQUnit::trySendPacket(')[1].split('LSQUnit::startStale')[0]
        self.assertIn('data_pkt->req->isSPMST() && (data_pkt->getAddr() & (Addr(1) << 60))',body)
        self.assertIn('reduction ? lsq->getReductionPort() : *dcachePort',body)
        reject=body.split('if (cache_got_blocked) {')[1].split('} else if (bridge)')[0]
        self.assertIn('reductionBlocked(true)',reject)
        self.assertNotIn('cacheBlocked(true)',reject)
    def test_same_issue_budget(self):
        s=(ROOT/'gem5/src/cpu/o3/lsq_unit.cc').read_text()
        body=s.split('LSQUnit::trySendPacket(')[1].split('LSQUnit::startStale')[0]
        self.assertEqual(body.count('spmPortsAvailable(isLoad)'),1)
        self.assertEqual(body.count('spmPortBusy(isLoad)'),1)
        self.assertNotIn('reductionPorts',body)
    def test_config_does_not_change_resources(self):
        s=(ROOT/'scripts/l3_reduction_retry_se.py').read_text()
        for forbidden in ('entries=', 'latency=', 'buffer_size=', 'StorePorts=', 'LoadPorts='):
            self.assertNotIn(forbidden,s)
        self.assertIn('old_connect(self,cpu)',s)
        self.assertIn('cpu.reduction_port=self.in_ports',s)
if __name__=='__main__':unittest.main()
