import unittest
from model import describe,candidates

def case(t=16384,family='kv_spm',bc=128,r=128,br=16):
    return dict(t=t,family=family,bc=bc,r=r,br=br)

class ModelTests(unittest.TestCase):
    def test_persistent_capacity(self):
        m=describe(case());self.assertTrue(m['valid'])
        self.assertEqual(m['spm_payload_bytes_per_core'],256*1024)
        self.assertEqual(m['max_spm_ways_per_set'],4)
        self.assertEqual(m['coherent_capacity_bytes'],256*1024)
        self.assertFalse(describe(case(t=32768))['valid'])
    def test_dynamic_slots_not_full_way_reservation(self):
        m=describe(case(family='q_spm'))
        self.assertEqual(m['spm_payload_bytes_per_core'],16*1024)
        self.assertEqual(m['coherent_capacity_bytes'],496*1024)
        self.assertEqual(m['min_coherent_ways_per_set'],6)
    def test_copies(self):
        self.assertEqual(describe(case())['spmcp_count']*64,2*1024**2)
        self.assertEqual(describe(case(family='q_spm'))['spmcp_count']*64,256*1024**2)
        self.assertEqual(describe(case(family='q_cache'))['spmcp_count'],0)
    def test_driver_constraints(self):
        self.assertFalse(describe(case(bc=384))['valid'])
        self.assertFalse(describe(case(r=100))['valid'])
        self.assertFalse(describe(case(br=8))['valid'])
    def test_model_allows_coherent_overflow(self):
        m=describe(case(br=64,bc=1024,r=1024))
        self.assertTrue(m['valid']);self.assertGreater(m['l2_pressure_ratio'],1)
    def test_candidate_count(self):
        legal,bad=candidates(16384)
        self.assertEqual(len(legal),384);self.assertFalse(bad)

if __name__=='__main__':unittest.main()
