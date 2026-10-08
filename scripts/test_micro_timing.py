"""Timing-profile acceptance and negative configuration-audit tests."""
import json
from pathlib import Path
import tempfile
import unittest
from build_access_timing import ROOT,sha
from run_timing_fa import config_audit

OUT=ROOT/'results/cacheflex_flash_attention/timing_alignment_20260922'
OLD=ROOT/'results/cacheflex_flash_attention/demand_isolation_20260922/experiment'

def check_global_guard(s,cores=8):
    values=[s[f'system.ruby.l0_cntrl{i}.sequencer.m_globalAdmissionFullRejects'] for i in range(cores)]
    assert all(v==0 for v in values),values
    return values

class TimingTests(unittest.TestCase):
    def test_same_elf_probe_calibration(self):
        record=json.loads((OUT/'calibration.json').read_text())
        self.assertTrue(record['passed'])
        self.assertEqual(record['guest_sha256'],sha(OUT/'guest_v4/guest'))
        self.assertEqual(record['profile_sha256'],sha(OUT/'fa_v3/configs/deprecated/example/micro_timing_se.py'))
        for p,h in record['files'].items():self.assertEqual(sha(p),h,p)

    def test_full_configuration_audit(self):
        for arch in ('cache','spm'):
            config_audit(OLD/f'{arch}_t512_off',OUT/f'fa_v3/{arch}_t512')

    def changed_config(self,mutate):
        old=OLD/'spm_t512_off';original=OUT/'fa_v3/spm_t512'
        data=json.loads((original/'config.json').read_text())
        with tempfile.TemporaryDirectory(prefix='cacheflex-timing-test-') as tmp:
            new=Path(tmp)
            # Rewrite only output-directory paths so the normal audit applies.
            text=json.dumps(data).replace(str(original),str(new));data=json.loads(text)
            mutate(data)
            (new/'config.json').write_text(json.dumps(data))
            with self.assertRaises(AssertionError):config_audit(old,new)

    def test_capacity_change_rejected(self):
        self.changed_config(lambda d:d['system']['ruby']['l1_cntrl0']['cache'].update(size=131072))

    def test_unexpected_service_delay_rejected(self):
        self.changed_config(lambda d:d['system']['ruby']['l1_cntrl0'].update(l1_response_latency=12))

    def test_guard_requires_actual_counters(self):
        with self.assertRaises(KeyError):check_global_guard({})
        s={f'system.ruby.l0_cntrl{i}.sequencer.m_globalAdmissionFullRejects':0 for i in range(8)}
        self.assertEqual(check_global_guard(s),[0]*8)
        s['system.ruby.l0_cntrl7.sequencer.m_globalAdmissionFullRejects']=1
        with self.assertRaises(AssertionError):check_global_guard(s)

if __name__=='__main__':unittest.main()
