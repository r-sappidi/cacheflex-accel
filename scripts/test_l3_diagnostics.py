"""Diagnostic isolation, finite admission, plan coverage and trace interpretation."""
import json
import unittest
from pathlib import Path
from l3_diagnostic_profiles import ROOT, DIRECT, PROFILES, command, allowed_config, can_launch, trace_window
from run_l3_diagnostics import make_plan, check_config, summarize_trace
from fa_headroom_profiles import leaves

class DiagnosticTests(unittest.TestCase):
    def test_every_plan_source_is_accepted_and_guests_exist(self):
        plan=make_plan()
        self.assertEqual(len(plan['gates']),19)
        self.assertEqual(len(plan['targets']),8)
        self.assertEqual(len({j['name'] for j in plan['gates']+plan['targets']}),27)
        for j in plan['gates']+plan['targets']:
            source=Path(j['source']);row=json.loads((source/'accepted.json').read_text())
            self.assertGreater(row['roi_us'],0)
            cmd=json.loads((source/'command.json').read_text())
            self.assertTrue(Path(cmd[cmd.index('--cmd')+1]).is_file())
            command(source,Path('/build'),Path('/out'),j['profile'],trace_window(source) if j['trace'] else None)

    def test_only_named_config_differences_permitted(self):
        old=dict(leaves(json.loads((DIRECT/'runs/t4096_direct/config.json').read_text())))
        for profile in PROFILES:
            new={**old,**allowed_config(profile)}
            check_config(old,new,profile)
            for key,value in [('system.ruby.l2_cntrl0.l3_reduction_entries',8),
                              ('system.ruby.l2_cntrl0.l3_spm_latency',1),
                              ('system.cpu0.issueWidth',99)]:
                with self.assertRaises(AssertionError):check_config(old,{**new,key:value},profile)

    def test_command_preserves_guest_network_and_finite_resources(self):
        source=DIRECT/'runs/t4096_direct'
        old=json.loads((source/'command.json').read_text())
        for profile in PROFILES:
            cmd=command(source,Path('/build'),Path('/out'),profile)
            for option in ('--cmd','--options'):
                self.assertEqual(cmd[cmd.index(option)+1],old[old.index(option)+1])
            self.assertEqual([x for x in cmd if 'buffer_size=' in x],[x for x in old if 'buffer_size=' in x])
            self.assertEqual([x for x in cmd if 'direct-' in x],[x for x in old if 'direct-' in x])

    def test_admission_refuses_resource_pressure_or_seventh_simulator(self):
        r=dict(cpu_idle_pct=70,available_gib=256,free_disk_gib=200,swap_used_gib=0,simulators=list(range(5)))
        self.assertTrue(can_launch(r))
        for key,value in [('cpu_idle_pct',49),('available_gib',127),('free_disk_gib',99),
                          ('swap_used_gib',.01),('simulators',list(range(6)))]:
            self.assertFalse(can_launch({**r,key:value}))
        self.assertTrue(can_launch({**r,'simulators':list(range(6))},simulator=False))

    def test_trace_handles_window_boundaries_without_inventing_matches(self):
        s=summarize_trace('CFRED phase=issue serial=3 age=12\nCFRED phase=complete serial=2 age=99\n'
                          'CFRED phase=sample source=4 flight=4 oldest_age=100\n')
        self.assertEqual(s['mean_issue_age_core_cycles'],12)
        self.assertEqual(s['mean_completion_age_core_cycles'],99)
        self.assertEqual(s['sampled_source_full'],1)
        self.assertIn('not stall-cycle',s['caveat'])

    def test_source_overrides_leave_ordinary_accesses_and_init_read_bypass(self):
        text=(ROOT/'gem5/src/mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm').read_text()
        self.assertIn('Cycles l3_reduction_read_delay := 0;',text)
        self.assertIn('Cycles l3_reduction_reply_delay := 0;',text)
        self.assertIn('in_msg.L3Op == 7 && l3_reduction_read_delay > intToCycles(0)',text)
        self.assertIn('delay := intToCycles(0); // Initialize',text)
        ordinary=text.split('action(l3_access,',1)[1].split('action(l3_release,',1)[0]
        self.assertNotIn('l3_reduction_read_delay',ordinary)
        self.assertNotIn('l3_reduction_reply_delay',ordinary)
        finish=text.split('action(l3_reduceFinish,',1)[1].split('action(l3_access,',1)[0]
        self.assertLess(finish.index('writeSPMData'),finish.index('enqueue('))
        self.assertIn('l3Occupy(256)',finish)

if __name__=='__main__':unittest.main()
