"""Evidence-backed structural checks on completed timing reports."""
import json
from pathlib import Path
import unittest

BASE=Path(__file__).resolve().parents[1]/'results/cacheflex_flash_attention/access_timing_20260922'


class AccessTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex=BASE/'experiment_v3'
        cls.report=json.loads((cls.ex/'breakdown.json').read_text())

    def test_complete_nonoverlapping_partition(self):
        r=self.report;rows=r['intervals']
        self.assertEqual(rows[0]['start'],r['issue'])
        self.assertEqual(rows[-1]['end'],r['install'])
        self.assertTrue(all(a['end']==b['start'] for a,b in zip(rows,rows[1:])))
        self.assertTrue(all(a['ticks']>=0 for a in rows))
        self.assertEqual(sum(a['ticks'] for a in rows),r['install']-r['issue'])
        self.assertAlmostEqual(sum(a['percent'] for a in rows),100)
        self.assertAlmostEqual(sum(a['percent'] for a in r['summary']),100)

    def test_observer_equivalence(self):
        for name in ['scan','trace']:
            gate=json.loads((self.ex/name/'validated.json').read_text())
            for key in ['exact_finite_nonhost_stats','exact_output',
                        'inherited_independent_arithmetic_pass','exact_config_except_outdir']:
                self.assertTrue(gate[key])

    def test_packet_identity_and_complete_flits(self):
        self.assertEqual(len(self.report['packets']),2)
        for packet in self.report['packets']:
            for stage in ['NI_SEND','NI_ARRIVE','NI_COMPLETE']:
                ids=[int(e['flit']) for e in packet['events'] if e['event']==stage]
                self.assertEqual(sorted(ids),list(range(packet['flits'])))

    def test_forwarding_gate_explanation(self):
        failures=self.report['forwarding_rejections']
        self.assertTrue(failures)
        self.assertEqual({e['resource'] for e in failures},
                         {'(*m_writeCredits_ptr)','(*m_downstreamWriteCredits_ptr)'})
        self.assertEqual(self.report['total_core_cycles'],711.25)

    def test_ingress_head_of_line_explanation(self):
        r=self.report['ingress_wait_explanation']
        self.assertTrue(r['all_wait_cycles_observed'])
        self.assertEqual(len(r['older_source_lines']),7)
        self.assertEqual(sum(r['blocked_core_cycles'].values()),497)
        self.assertEqual(r['blocked_core_cycles']['L1 write/bridge credits full'],39)


if __name__=='__main__':unittest.main()
