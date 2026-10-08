import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from build_q_resident import q_kernel


class QResidentTests(unittest.TestCase):
    def test_load_replacement(self):
        here=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'kernels.hpp'
            subprocess.run([sys.executable,str(here/'make_search_kernels.py'),str(target),'1','0','16','1'],check=True)
            source=q_kernel(target.read_text())
        kernel=source[source.index('static void cf_wide_qk_qspm'):]
        self.assertEqual(kernel.count('spm.ld1qd'),6)
        self.assertEqual(kernel.count('tbl z'),6)
        self.assertNotIn('ld1rqh',kernel)
        self.assertEqual(kernel.count('ld1h {z'),3)
        self.assertIn('"p2"',kernel)
        self.assertEqual(kernel.count('#0x1fc0'),6)
        self.assertNotIn('#-64',kernel)

    def test_slot_capacity(self):
        for rows in (16,32,64):
            slots=list(range(0,rows*64*2,64))
            self.assertLessEqual(len(slots),128)
            self.assertLess(max(slots),8192)
        self.assertEqual(16384*64*2//64,32768)

    def test_bounded_address_preserves_valid_q(self):
        for address in range(0,8192,16):
            self.assertEqual(address & 0x1fc0, address & ~63)
            self.assertLessEqual((address & 48)+15,63)
        self.assertEqual((-128)&0x1fc0,0x1f80)
        self.assertEqual((((-128)&0x1fc0)>>16)&7,0)
