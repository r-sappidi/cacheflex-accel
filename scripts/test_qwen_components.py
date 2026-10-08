"""Check accounting boundaries with accepted, partial and conflicting evidence."""
import json
import tempfile
import unittest
from pathlib import Path
import collect_qwen_components as c


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sources = {k: Path(self.tmp.name)/k for k in c.SOURCES}
        self.rows = {k: [] for k in self.sources}
        for p in self.sources.values():
            p.mkdir()
        self.publish()

    def publish(self):
        for k, p in self.sources.items():
            (p/'status.json').write_text(json.dumps(dict(stage='running', completed=self.rows[k])))

    def row(self, kind, name, arm='cache', first=10, repeat=1, stage='screen_t1024', op='qkv'):
        path = self.sources[kind]/name
        path.mkdir()
        row = dict(name=name, path=str(path), first_us=first, repeat_us=repeat,
                   roi_us=first, case=dict(t=1024, arm=arm, stage=stage, op=op))
        (path/'accepted.json').write_text(json.dumps(row))
        self.rows[kind].append(row)
        self.publish()
        return row

    def layer(self, backend='cache'):
        return next(r for r in c.collect(self.sources)['layers']
                    if r['t']==1024 and r['backend']==backend)

    def test_partial_and_gate_results_do_not_fill_targets(self):
        partial = self.sources['swiglu']/'partial'
        partial.mkdir()
        (partial/'stats.txt').write_text('simSeconds 0.001\n')
        self.row('projection', 'gate', stage='tail_gates')
        layer = self.layer()
        self.assertTrue(all(v is None for v in layer['components'].values()))
        self.assertIsNone(layer['layer_projection_us'])
        self.assertIsNone(layer['model_projection_us'])

    def test_selection_uses_first_call_not_warm_call(self):
        self.row('swiglu', 'cold_winner', first=10, repeat=9)
        self.row('swiglu', 'warm_winner', first=20, repeat=1)
        value = self.layer()['components']['swiglu']
        self.assertEqual(value['us'], 10)
        self.assertEqual(value['metric'], 'first_us')
        self.assertEqual(len(value['sha256']), 64)

    def test_l3_reuses_spm_ffn_without_inventing_missing_costs(self):
        self.row('swiglu', 'spm', arm='spm', first=42)
        self.row('fa', 't1024_direct', first=3)
        for backend in ('private_spm', 'spm_l3_noc', 'spm_l3_direct'):
            layer = self.layer(backend)
            self.assertEqual(layer['components']['swiglu']['us'], 42)
            self.assertIn('rope', layer['missing'])
            self.assertIsNone(layer['model_projection_us'])
        self.assertIsNone(self.layer()['components']['swiglu'])
        self.assertEqual(self.layer('spm_l3_direct')['measured_component_subtotal_us'], 45)

    def test_changed_or_missing_acceptance_is_rejected(self):
        row = self.row('swiglu', 'accepted')
        marker = Path(row['path'])/'accepted.json'
        marker.write_text('{}')
        with self.assertRaises(AssertionError):
            c.collect(self.sources)
        marker.unlink()
        with self.assertRaises(FileNotFoundError):
            c.collect(self.sources)


if __name__ == '__main__':
    unittest.main()
