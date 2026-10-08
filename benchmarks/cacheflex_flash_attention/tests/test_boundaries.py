import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analyze_boundaries import markers,analyze

class BoundariesTest(unittest.TestCase):
    def test_reject_ambiguous_pc(self):
        with patch('analyze_boundaries.subprocess.check_output',return_value=
                   '400 T cf_boundary_QK_1\n400 T cf_boundary_final_sync_2\n'):
            with self.assertRaises(AssertionError):markers('unused')

    def test_clipped_additive_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'stats.txt').write_text('finalTick 10000\nsimTicks 8000\nEnd Simulation Statistics\n')
            lines=[]
            for c in range(8):
                for tick,pc in ((1000,1),(3000,2),(8000+c,3),(9000,4)):
                    lines.append(f'{tick}: system.cpu{c}.commit: CFBOUND pc={hex(pc)} sn=1\n')
            (path/'boundaries.trace').write_text(''.join(lines))
            with patch('analyze_boundaries.markers',return_value={1:'startup_sync',2:'QK',3:'final_sync',4:'done'}):
                result=analyze(path,'unused')
            self.assertEqual(result['last_arriving_core'],7)
            self.assertEqual(result['terminal_arrival_spread_us'],7/1e6)
            for phases in result['per_core_us'].values():
                self.assertAlmostEqual(sum(phases.values()),.008)

if __name__=='__main__':unittest.main()
