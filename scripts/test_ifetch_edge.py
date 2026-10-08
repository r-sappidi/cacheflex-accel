import copy
import json
from pathlib import Path
import unittest
from run_ifetch_edge import ROOT, prior, audit_config, command, control_stats


class IfetchEdgeTests(unittest.TestCase):
    def configs(self,edge):
        old=json.loads((prior('spm',512)/'config.json').read_text())
        new=copy.deepcopy(old)
        for i in range(8):
            seq=new['system']['ruby'][f'l0_cntrl{i}']['sequencer']
            seq['cacheflex_ifetch_same_tick_response']=edge
            seq['cacheflex_ideal_staging_cache']=None
            seq['cacheflex_ideal_staging_controller']=None
        return old,new
    def test_only_declared_change_accepted(self):
        a,b=self.configs(True);audit_config(a,b,True)
        b['system']['cpu'][0]['fetchWidth']+=1
        with self.assertRaises(AssertionError):audit_config(a,b,True)
    def test_all_core_flags_audited(self):
        a,b=self.configs(True);b['system']['ruby']['l0_cntrl7']['sequencer']['cacheflex_ifetch_same_tick_response']=False
        with self.assertRaises(AssertionError):audit_config(a,b,True)
    def test_oracle_forbidden(self):
        a,b=self.configs(False);b['system']['ruby']['l0_cntrl0']['sequencer']['cacheflex_ideal_staging_cache']='x'
        with self.assertRaises(AssertionError):audit_config(a,b,False)
    def test_matched_commands_same_ifetch_intervention(self):
        for backend in ('cache','spm','spm_fast'):
            cmd=command(backend,4096,True,Path('/tmp/ifetch-test-not-created'))
            flags=[x for x in cmd if 'cacheflex_ifetch_same_tick_response' in x]
            self.assertEqual(len(flags),8)
            self.assertTrue(all(x.endswith('=True') for x in flags))
            self.assertFalse(any('oracle' in x for x in cmd))
    def test_default_stats_extra_must_be_zero(self):
        control_stats([{'x':2}],[{'x':2,'a.ifetchEdgeDirect':0}])
        with self.assertRaises(AssertionError):control_stats([{'x':2}],[{'x':3,'a.ifetchEdgeDirect':0}])
        with self.assertRaises(AssertionError):control_stats([{'x':2}],[{'x':2,'a.ifetchEdgeDirect':1}])
    def test_queue_order_retry_and_ifetch_scope(self):
        hh=(ROOT/'gem5/src/mem/ruby/system/RubyPort.hh').read_text()
        body=hh.split('bool deliverCompletedIfetch(PacketPtr pkt)',1)[1].split('class MemResponsePort',1)[0]
        self.assertIn('assert(pkt->req->isInstFetch())',body)
        self.assertIn('size() == 0 && !waitingOnRetry',body)
        self.assertLess(body.index('waitingOnRetry = true'),body.index('schedSendTiming(pkt, curTick())'))
        cc=(ROOT/'gem5/src/mem/ruby/system/RubyPort.cc').read_text()
        self.assertEqual(cc.count('queue.deliverCompletedIfetch(pkt)'),1)
        self.assertIn('if (owner.ifetchSameTickResponse && pkt->req->isInstFetch())',cc)
        py=(ROOT/'gem5/src/mem/ruby/system/Sequencer.py').read_text()
        self.assertIn('cacheflex_ifetch_same_tick_response = Param.Bool(False,',py)


if __name__=='__main__':unittest.main()
