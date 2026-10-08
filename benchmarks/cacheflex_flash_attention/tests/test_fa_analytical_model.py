import unittest
from fa_analytical_model import Shape,model

class ModelTests(unittest.TestCase):
    def test_exact_t16k(self):
        m=model(Shape(16384))
        self.assertEqual(m['spmcp_count'],16777216)
        self.assertEqual(m['staged_bytes'],2**30)
        self.assertEqual(m['spm_load_instructions'],134217728)
        self.assertEqual(m['fp16_fmla_instructions'],2147483648)
        self.assertEqual(m['total_scratch_bytes_per_core'],151552)
        self.assertEqual(m['main_array_bytes'],40*2**20)
    def test_group_tradeoff(self):
        a,b=model(Shape(16384)),model(Shape(16384,group=64))
        self.assertEqual(a['spmcp_count'],4*b['spmcp_count'])
        self.assertEqual(a['fp16_fmla_instructions'],b['fp16_fmla_instructions'])
        self.assertEqual(b['scratch_bytes_per_core']['output_acc'],4*a['scratch_bytes_per_core']['output_acc'])
    def test_panel_size(self):
        a,b=model(Shape(16384)),model(Shape(16384,bc=128))
        self.assertEqual(a['spmcp_count'],b['spmcp_count'])
        self.assertEqual(a['claimed_spm_bytes_per_core'],4*b['claimed_spm_bytes_per_core'])
    def test_tail_group(self):
        m=model(Shape(640,group=16))
        self.assertEqual(m['groups_per_core'],3)
        self.assertEqual(m['spmcp_count'],8*3*640*2)
    def test_all_group(self):
        m=model(Shape(16384,group=0))
        self.assertEqual(m['groups_per_core'],1)
        self.assertEqual(m['staged_bytes'],16*2**20)
        self.assertFalse(m['aggregate_private_footprint_fits'])
    def test_invalid(self):
        for s in (Shape(17),Shape(1024,br=8),Shape(1024,d=128)):
            with self.assertRaises(ValueError):model(s)

if __name__=='__main__':unittest.main()
