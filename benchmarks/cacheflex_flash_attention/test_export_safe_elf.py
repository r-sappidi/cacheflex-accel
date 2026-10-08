"""Regression evidence for export corruption and immutable-ROI patching."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from export_safe_elf import export_hash

B = Path(__file__).resolve().parents[2]/'results/cacheflex_flash_attention'
REPAIR = B/'export_repair_20260926'


class ExportTests(unittest.TestCase):
    def test_patches_change_only_audited_bytes(self):
        for audit in sorted((REPAIR/'builds').glob('*/guest.audit.json')):
            row = json.loads(audit.read_text())
            a = Path(row['original']).read_bytes()
            b = audit.with_name('guest').read_bytes()
            self.assertEqual(len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    self.assertTrue(any(lo <= i < hi for lo, hi in row['changed_ranges']))

    def test_old_cache_export_is_corrupt_but_guest_matches_validated_result(self):
        bad = B/'cache_tuning_20260925/experiment/runs/qpack_t4096_br16_bc128_g16'
        good = B/'kv_partition_20260925/experiment_v3/runs/target_t4096_n8_q_cache_br16_g8_l2512'
        log = (bad/'run.log').read_text()
        with self.assertRaisesRegex(AssertionError, 'guest/export hash mismatch'):
            export_hash(bad/'output.fp16', log)
        export_hash(good/'output.fp16', log)
        a = np.fromfile(bad/'output.fp16', dtype=np.uint16)
        b = np.fromfile(good/'output.fp16', dtype=np.uint16)
        where = np.flatnonzero(a != b)
        np.testing.assert_array_equal(where, np.arange(180*64+32, 181*64))
        self.assertTrue(np.all(a[where] == 0))

    def test_llama_fp32_rounding_matches_guest_hash(self):
        p = B/'llama_cpu_20260925/experiment_v2/runs/ported_sve_ready_t4096'
        log = (p/'run.log').read_text()
        with self.assertRaisesRegex(AssertionError, 'guest/export hash mismatch'):
            export_hash(p/'output.fp16', log)
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d)/'output.fp16'
            np.fromfile(p/'output.fp32', dtype=np.float32).astype(np.float16).tofile(dst)
            export_hash(dst, log)


if __name__ == '__main__':unittest.main()
