import tempfile
import unittest
from pathlib import Path
import numpy as np
from screen_reference import core_header, gemm, generate, validate, mlp_source
from run_mapping_screen import candidates, expected_copies


class MappingScreenTests(unittest.TestCase):
    def test_team_adapter(self):
        source=core_header()
        self.assertIn('int np=team/s.mp',source)
        self.assertNotIn('int np=Threads/s.mp',source)
        self.assertIn('int team=8',source)
        self.assertNotIn('void* worker',source)

    def test_reference_and_reject_corruption(self):
        for shared in (0,1):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp);inp=path/'input.bin'
                expected=generate(inp,5,8,32,2,shared,4)
                expected.tofile(path/'output.bin')
                self.assertEqual(validate(path,inp,5,8,32,2,shared)['errors'],0)
                expected.flat[0]=100;expected.tofile(path/'output.bin')
                with self.assertRaises(AssertionError):validate(path,inp,5,8,32,2,shared)

    def test_exact_small_gemm(self):
        a=np.array([[1,2],[3,4]],dtype=np.float16)
        np.testing.assert_array_equal(gemm(a,np.eye(2,dtype=np.float16),1),a)

    def test_plan_and_copy_counts(self):
        plan=candidates();self.assertEqual(len(plan),80)
        self.assertEqual(len({str(sorted(c.items())) for c in plan}),80)
        self.assertEqual({c['family'] for c in plan},{'grouped_projection','shared_weight_batch','pointwise_1x1','swiglu'})
        for c in plan:
            self.assertEqual(c['team']%c['mp'],0)
            self.assertLessEqual(4*c['kc']*64,262144)
            self.assertGreater(expected_copies(c),0)

    def test_export_is_after_roi(self):
        source=mlp_source()
        self.assertIn('screen_export(s.y',source)
        self.assertNotIn('m5_write_file(s.y',source)


if __name__=='__main__':unittest.main()
