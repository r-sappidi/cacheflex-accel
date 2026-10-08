"""Independent invariants/oracles, not snapshots of model implementation."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from qwen_placement_model import Candidate, account, layout, ranges, phase_dag, Scenario, estimate, calibrate
from run_qwen_placement_model import SOURCE, observations, launchable
from check_qwen_placement_plan import check


class PlacementTests(unittest.TestCase):
    def test_complementary_partition_is_disjoint_complete(self):
        c=Candidate(t=4096)
        seen=[]
        for core in range(8):
            owned=[q for a,b in ranges(c,core) for q in range(a,b)]
            self.assertEqual(len(owned),512)
            self.assertEqual(sum(owned),512*4095//2)
            seen+=owned
        self.assertEqual(sorted(seen),list(range(4096)))

    def test_head_copy_counts_closed_form(self):
        c=Candidate(placement='qo');a=account(c)
        self.assertEqual(a['staged_bytes'],14*4096*(128+256))
        self.assertEqual(a['layout']['spm_bytes'],192*1024)
        self.assertEqual(a['totals']['copies'],344064)
        self.assertEqual(a['totals']['q_rows_packed'],14*4096)
        self.assertEqual(a['layout']['claimed_ways_histogram'],{'1':512,'5':512})

    def test_remapping_preserves_bytes_and_removes_set_concentration(self):
        base=Candidate(placement='qo');a=layout(base);b=layout(replace(base,stride=512))
        self.assertEqual(a['spm_bytes'],b['spm_bytes'])
        self.assertEqual(b['claimed_ways_histogram'],{'3':1024})
        self.assertEqual(b['min_coherent_ways'],5)

    def test_static_kv_stages_once_for_either_lifetime(self):
        c=Candidate(ownership='kv',placement='kv',lifetime='kv_group')
        a=account(c);b=account(replace(c,lifetime='all_heads'))
        self.assertEqual(a['staged_bytes'],2*4096*64*2*2)
        self.assertEqual(a['staged_bytes'],b['staged_bytes'])
        self.assertEqual(a['layout']['spm_bytes'],128*1024)
        self.assertEqual(b['layout']['spm_bytes'],256*1024)
        self.assertEqual(a['totals']['pairs'],b['totals']['pairs'])

    def test_causal_pairs_independent_token_bruteforce(self):
        # Whole rectangular Bc tiles are computed, even on the diagonal.
        for causal in (True,False):
            c=Candidate(t=256,bc=128,causal=causal,placement='cache',hq=2,hkv=1)
            expected=sum(1 for q in range(c.t) for k in range(c.t)
                         if not causal or (k//c.bc)*c.bc<((q//c.br)+1)*c.br)*c.hq
            self.assertEqual(account(c)['totals']['pairs'],expected)
            self.assertEqual(account(c)['useful_pairs'],2*sum(q+1 if causal else 256 for q in range(256)))

    def test_kv_merge_counts_from_independent_participant_sets(self):
        c=Candidate(t=2048,bc=128,ownership='kv',placement='kv',lifetime='kv_group',hq=2,hkv=1)
        owners={k:core for core in range(8) for a,b in ranges(c,core) for k in range(a,b)}
        expected=0
        for q in range(0,c.t,c.br):
            participants={owners[k] for k in range(q+c.br)}
            expected+=(len(participants)-1)*c.br*c.hq
        m=account(c)
        self.assertEqual(m['totals']['merge_rows'],expected)
        self.assertEqual(m['merge_payload_bytes'],expected*320)
        self.assertEqual(m['totals']['pairs'],account(replace(c,ownership='query_head',placement='cache',lifetime='head'))['totals']['pairs'])

    def test_capacity_and_geometry_fail_closed(self):
        with self.assertRaises(ValueError):account(Candidate(t=16384,placement='qo'))
        with self.assertRaises(ValueError):account(Candidate(t=16384,ownership='kv',placement='kv',lifetime='all_heads'))
        with self.assertRaises(ValueError):account(Candidate(ownership='kv',placement='kv',lifetime='kv_group',bc=512))
        with self.assertRaises(ValueError):account(Candidate(placement='garbage'))

    def test_dag_respects_overlap_and_dependencies(self):
        nodes=[dict(id='a',deps=[],cycles=10),dict(id='b',deps=[],cycles=20),dict(id='c',deps=['a','b'],cycles=5)]
        self.assertEqual(phase_dag(nodes),25)
        nodes[1]['deps']=['a']
        self.assertEqual(phase_dag(nodes),35)
        with self.assertRaises(KeyError):phase_dag([dict(id='a',deps=['missing'],cycles=1)])

    def test_no_copy_free_hardware_or_guest_promotion(self):
        c=Candidate(ownership='kv',placement='kv',lifetime='kv_group')
        self.assertIsNone(launchable(c))
        self.assertIsNone(launchable(Candidate(placement='qv')))
        self.assertIsNone(launchable(Candidate(sync='dependency')))
        self.assertIsNotNone(launchable(Candidate(placement='v')))
        s=Scenario()
        self.assertGreater(estimate(c,s)['us'],0)
        self.assertLessEqual(estimate(replace(c,sync='dependency'),s)['us'],estimate(c,s)['us'])

    def test_ignored_r_does_not_duplicate_head_candidates(self):
        self.assertEqual(Candidate(r=128).key,Candidate(r=512).key)
        self.assertNotEqual(Candidate(ownership='query_legacy',r=128).key,
                            Candidate(ownership='query_legacy',r=512).key)

    def test_plan_refuses_automatic_or_overbudget_execution(self):
        p={'schema':'cacheflex-placement-plan-v1','auto_launch':True}
        with self.assertRaises(ValueError):check(p)
        p.update(auto_launch=False,resource_policy={'cpu_host_fraction':.21,'ram_host_fraction':.2,'max_new_simulators':2})
        with self.assertRaises(ValueError):check(p)

    def test_failed_fit_cannot_promote_a_candidate(self):
        c=Candidate(placement='qv')
        p=dict(schema='cacheflex-placement-plan-v1',auto_launch=False,model_validation_passed=False,
               resource_policy=dict(cpu_host_fraction=.2,ram_host_fraction=.2,max_new_simulators=2),
               jobs=[dict(case=c.__dict__,performance_selection_allowed=True)])
        with self.assertRaisesRegex(ValueError,'failed model'):check(p)

    @unittest.skipUnless(SOURCE.exists(),'repository accepted fixtures unavailable')
    def test_all_saved_acceptances_match_independent_accounting(self):
        rows,audit,sig=observations(SOURCE)
        self.assertEqual(len(audit),18)
        self.assertEqual(len(rows),12)
        self.assertTrue(all(x['counts_exact'] for x in audit))


if __name__=='__main__':unittest.main()
