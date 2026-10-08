"""Checks of analytical scheduling constraints, not hardware validation."""
import unittest
from diagnose_l3_pv_feasibility import schedule

class FiniteScheduleTests(unittest.TestCase):
    def test_port_and_compute_bounds(self):
        for g in (1,4,16):
            for lanes in (16,64,128,256):
                for share in (1,.5,.25):
                    s=schedule(g,lanes,4,share)
                    self.assertGreaterEqual(s['tile_cycles'],g*256*64/lanes)
                    self.assertGreaterEqual(s['tile_cycles'],s['bank_read_write_bytes']/64/share)
                    self.assertLessEqual(s['max_prefetch_occupancy'],4)
    def test_latency_and_buffer_sensitivity(self):
        slow=schedule(16,256,1,1)
        buffered=schedule(16,256,16,1)
        self.assertGreater(slow['tile_cycles'],buffered['tile_cycles'])
        self.assertGreaterEqual(buffered['tile_cycles'],schedule(16,256,16,1,latency=9)['tile_cycles'])
    def test_dependency_prevents_single_row_widening_gain(self):
        self.assertEqual(schedule(1,64,16,1)['tile_cycles'],schedule(1,256,16,1)['tile_cycles'])
        self.assertLess(schedule(16,256,16,1)['tile_cycles'],schedule(16,64,16,1)['tile_cycles'])
    def test_shared_bandwidth_and_explicit_storage(self):
        for q in (1,4,16):
            full=schedule(16,128,q,1)
            half=schedule(16,128,q,.5)
            self.assertGreater(half['tile_cycles'],full['tile_cycles'])
            self.assertEqual(full['internal_register_bytes'],8192+2048+4096+128*q)
        self.assertEqual(schedule(16,128,16,1)['bank_read_write_bytes'],8192+8192+32768)

if __name__=='__main__':unittest.main()
