"""Structural regression checks for bounded silent-forward retry progress."""
from pathlib import Path
import re
import unittest
import json
from fa_headroom_profiles import leaves

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L1cache.sm'

class RetryTests(unittest.TestCase):
    def action(self,name):
        return P.read_text().split('action('+name+',',1)[1].split('\n  action(',1)[0]

    def test_nack_has_no_request_output_dependency(self):
        a=self.action('spm_nackSilentGetS')
        self.assertIn('enqueue(responseNetwork_out',a)
        self.assertNotIn('requestNetwork_out',a)

    def test_receive_is_sink_and_bounded_by_existing_tbe(self):
        a=self.action('spm_rememberSilentRetry')
        for s in ('assert(is_valid(tbe))','assert(!spmSilentRetryTimer.isSet(address))',
                  'tbe.DstSPMAddr == in_msg.DstSPMAddr'):
            self.assertIn(s,a)
        for s in ('enqueue(', 'TBEs.allocate', 'check_allocate'):
            self.assertNotIn(s,a)
        self.assertIn('assert(!spmSilentRetryTimer.isSet(address))',self.action('s_deallocateTBE'))

    def test_reissue_preserves_transaction(self):
        a=self.action('spm_reissueSilentGetS')
        for s in ('out_msg.SrcAddr := tbe.SrcAddr','out_msg.DstSPMAddr := tbe.DstSPMAddr',
                  'out_msg.Requestor := machineID','spmSilentRetryTimer.unset(address)'):
            self.assertIn(s,a)
        self.assertNotIn('TBEs.allocate',a)
        self.assertNotIn('Credits',a)

    def test_generated_receive_has_no_resource_check(self):
        p=ROOT/'gem5/build/ARM_MESI_Three_Level_SPM/mem/ruby/protocol/MESI_Three_Level_SPM/L1Cache_Transitions.cc'
        text=p.read_text()
        t=text.split('case HASH_FUN(L1Cache_State_IX_D, L1Cache_Event_SPM_SilentRetry):',1)[1].split('return TransitionResult_Valid;',1)[0]
        self.assertNotIn('areNSlotsAvailable',t)
        self.assertIn('o_popL2ResponseQueue',t)

    def test_original_deadlock_replay_configuration(self):
        base=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927'
        old=base/'experiment/runs/spread_full_set'
        new=base/'experiment_retry_v2/runs/spread_full_set'
        if not (new/'validated.json').exists():
            self.skipTest('Repaired deadlock replay not yet validated')
        def config(p):
            return dict(leaves(json.loads((p/'config.json').read_text().replace(str(p),'<OUT>'))))
        a,b=config(old),config(new)
        changes={k:(a.get(k),b.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
        self.assertEqual(len(changes),16)
        for k,v in changes.items():
            self.assertTrue((k.endswith('.spm_silent_retry') and v==(None,True)) or
                            (k.endswith('.spm_silent_retry_latency') and v==(None,8)),(k,v))
        result=json.loads((new/'validated.json').read_text())
        self.assertGreater(result['silent_retry_received'],0)
        self.assertEqual(result['silent_retry_received'],result['silent_retry_reissued'])
        self.assertEqual(result['arithmetic']['errors'],0)

if __name__=='__main__': unittest.main()
