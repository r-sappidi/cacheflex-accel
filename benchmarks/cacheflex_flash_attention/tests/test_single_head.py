from pathlib import Path
import unittest
import numpy as np
from build_single_head import transform
from single_head_reference import reference
from fa_independent_reference import reference as multi_reference


class SingleHeadTests(unittest.TestCase):
    def test_reference_matches_first_head(self):
        np.testing.assert_array_equal(reference(128,128),multi_reference(128,True,32,128)[0])

    def test_partition_and_export(self):
        source=(Path(__file__).resolve().parents[1]/'spm_multicore.cpp').read_text()
        source+='\n// builder export count: (size_t)8*T*d\n'
        new=transform(source)
        self.assertNotIn('(size_t)Cores*T*d',new)
        self.assertNotIn('(size_t)Cores*t*dim',new)
        self.assertNotIn('(size_t)8*T*d',new)
        self.assertIn('n_ib=(T/Cores+Br-1)/Br',new)
        self.assertEqual(new.count('(size_t)c.tid*(T/Cores)*d'),2)
        self.assertIn('T%(Cores*Br)!=0',new)
