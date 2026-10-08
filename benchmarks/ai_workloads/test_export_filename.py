import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from repair_screen_export import patch


class ExportFilenameTests(unittest.TestCase):
    def test_frozen_guests_change_only_hook_and_writer(self):
        root=Path(__file__).resolve().parents[2]
        builds=root/'results/ai_workloads/mapping_screen_20260917/builds'
        if not builds.exists():
            self.skipTest('Frozen regression guests unavailable')
        for family in ('projection','mlp_separate','mlp_combined'):
            for arch in ('cache','spm'):
                with self.subTest(family=family,arch=arch), tempfile.TemporaryDirectory() as tmp:
                    guest=Path(tmp)/arch
                    shutil.copy2(builds/family/arch,guest)
                    before=guest.read_bytes()
                    report=patch(guest)
                    after=guest.read_bytes()
                    self.assertEqual(len(before),len(after))
                    self.assertEqual(before,guest.with_suffix('.original').read_bytes())
                    self.assertNotEqual(report['original_sha256'],report['patched_sha256'])
                    changed=[i for i,(a,b) in enumerate(zip(before,after)) if a!=b]
                    self.assertTrue(all(any(lo<=i<hi for lo,hi in report['ranges']) for i in changed))
                    with self.assertRaises(AssertionError):patch(guest)


if __name__=='__main__':unittest.main()
