"""Regression tests for paired-profile intervention isolation."""
import json
from pathlib import Path
import tempfile
import unittest
import run_calibrated_sensitivities as s

class SensitivityTests(unittest.TestCase):
    def test_same_platform_except_declared_intervention(self):
        for profile in ('old','micro'):
            for variant in s.VARIANTS:
                cmd=s.make_command(profile,'spm',4096,variant,Path('/tmp/test-out'))
                self.assertEqual('--cacheflex-micro-timing' in cmd,profile=='micro')
                self.assertEqual(cmd[0],str(s.SIM))
                self.assertIn('--l1d_size=64kB',cmd)
                self.assertIn('--l1d_num_spm_ways=2',cmd)
                self.assertIn('--num-cpus=8',cmd)
                for k,v in s.parameters(variant).items():
                    self.assertEqual(cmd.count(f'{k}={v!r}'),1,(profile,variant,k))
                self.assertFalse(any('spm_separate_backing=True' in v for v in cmd))

    def test_old_and_new_use_same_intervention_values(self):
        for variant in s.VARIANTS:
            for arch in ('cache','spm'):
                # Explicit overrides must not accidentally inherit a profile's
                # old value because a duplicate -P occurs later on the command.
                for profile in ('old','micro'):
                    cmd=s.make_command(profile,arch,512,variant,Path('/tmp/test-out'))
                    for k in s.parameters(variant):
                        self.assertEqual(sum(v.startswith(k+'=') for v in cmd),1)

    def test_config_audit_rejects_unrelated_capacity(self):
        old=s.prior('micro','spm',512)
        with tempfile.TemporaryDirectory(prefix='cf-sensitivity-test-') as tmp:
            p=Path(tmp);d=json.loads((old/'config.json').read_text().replace(str(old),str(p)))
            d['system']['ruby']['l1_cntrl0']['cache']['size']*=2
            (p/'config.json').write_text(json.dumps(d))
            with self.assertRaises(AssertionError):s.audit(old,p,'control')

    def test_configuration_must_exercise_requested_knob(self):
        old=s.prior('micro','spm',512)
        with tempfile.TemporaryDirectory(prefix='cf-sensitivity-test-') as tmp:
            p=Path(tmp);(p/'config.json').write_text((old/'config.json').read_text().replace(str(old),str(p)))
            with self.assertRaises(AssertionError):s.audit(old,p,'demand_isolation')
            self.assertEqual(s.audit(old,p,'control'),[])

if __name__=='__main__':unittest.main()
