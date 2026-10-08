import configparser
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_clean_controls import command, config_audit, counter_delta, prior_path


class CleanControlsTest(unittest.TestCase):
    def test_command_only_changes_sim_output_and_recorder(self):
        original = ['oldsim', '--outdir=old', 'se.py', '--cmd', 'frozen-guest', '-P', 'canonical-argv']
        result = command(original, Path('newsim'), Path('newrun'), 0)
        self.assertEqual(result[2:-1], original[2:])
        self.assertEqual(result[:2], ['newsim', '--outdir=newrun'])
        self.assertEqual(result[-1], '--ruby-flight-recorder-entries=0')
        with self.assertRaises(AssertionError):
            command(original + ['--debug-flags=ProtocolTrace'], Path('sim'), Path('run'), 0)

    def test_config_audit_rejects_hardware_changes(self):
        prior = prior_path('spm', 512)
        cfg = configparser.ConfigParser(interpolation=None, strict=False)
        cfg.read(prior / 'config.ini')
        for section in cfg.sections():
            for key, value in list(cfg[section].items()):
                cfg[section][key] = value.replace(str(prior), '<RESULT>')
            if any(section == f'system.ruby.{kind}_cntrl{i}' for kind in ('l0', 'l1', 'l2', 'dir') for i in range(8)):
                cfg[section]['flight_recorder_entries'] = '0'
                cfg[section]['flight_recorder_messages'] = 'true'
        self.assertTrue(config_audit(prior, cfg, 0)['all_prior_fields_equal'])
        cfg['system.cpu0']['numrobentries'] = '256'
        with self.assertRaises(AssertionError):
            config_audit(prior, cfg, 0)

    def test_counter_comparison_ignores_only_path(self):
        self.assertFalse(counter_delta({'path': 'a', 'roi_us': 1}, {'path': 'b', 'roi_us': 1}))
        self.assertIn('roi_us', counter_delta({'roi_us': 1}, {'roi_us': 2}))


if __name__ == '__main__':
    unittest.main()
