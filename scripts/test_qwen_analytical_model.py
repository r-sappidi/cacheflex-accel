import unittest
from qwen_analytical_model import case,describe,rank


class AnalyticalTests(unittest.TestCase):
    def test_full_causal_pairs_and_padding(self):
        c=case(2048,'l3');m=describe(c)
        self.assertEqual(m['pairs'],14*(2048*2048+2048*256)//2)
        self.assertEqual(m['useful_pairs'],14*2048*2049//2)
        self.assertEqual(m['commands'],129024)
        self.assertEqual(m['staged_bytes'],int(15.75*2**20))

    def test_r_changes_ownership_not_useful_math(self):
        a=describe(case(2048,'l3'));b=describe(case(2048,'l3',128))
        self.assertEqual(a['pairs'],b['pairs'])
        self.assertEqual(a['commands'],b['commands'])
        self.assertEqual(2*a['copies'],b['copies'])
        self.assertLess(b['max_core_pairs'],a['max_core_pairs'])
        self.assertEqual(a['l3_accumulator_bytes_per_bank'],2*b['l3_accumulator_bytes_per_bank'])

    def test_tile_tradeoff(self):
        a=describe(case(2048,'l3'));b=describe(case(2048,'l3',256,512))
        self.assertGreater(b['pairs'],a['pairs'])
        self.assertLess(b['commands'],a['commands'])
        self.assertEqual(b['private_v_spm_bytes'],2*a['private_v_spm_bytes'])

    def test_current_scheduler_screen_is_size_specific(self):
        a=rank(2048,'l3');b=rank(4096,'l3')
        self.assertTrue(any(x['case']['r']==512 and x['case']['bc']==256 for x in a['deferred']))
        self.assertTrue(any(x['case']['r']==512 and x['case']['bc']==256 for x in b['ranked']))


if __name__=='__main__':unittest.main()
