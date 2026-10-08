"""Independent invariants for GQA ownership and causal reference semantics."""
import unittest
from unittest.mock import patch
import numpy as np
import qwen_l3_reference as oracle
from run_qwen_l3_attention import case,work
from fa_independent_reference import inputs as old_inputs


class QwenGqaTests(unittest.TestCase):
    def test_pattern_compatibility(self):
        q,k,v=oracle.inputs(128,8,1);a,b,c=old_inputs(128,True)
        np.testing.assert_array_equal(q,a)
        np.testing.assert_array_equal(k[0],b)
        np.testing.assert_array_equal(v[0],c)

    def test_task_partition_and_causal_counts(self):
        for t in (256,1024,2048,4096):
            for bc in (128,256,512):
                if bc>t:continue
                c=case(t,'l3',bc);w=work(c)
                self.assertEqual(sum(w['tasks']),14*t//256)
                self.assertLessEqual(max(w['tasks'])-min(w['tasks']),1)
                # Independent enumeration by head and query row: each row in a
                # group executes blocks through that group's last visible key.
                blocks=sum(((row//256+1)*256+bc-1)//bc for row in range(t))*14
                self.assertEqual(sum(w['updates']),blocks)
                self.assertEqual(sum(w['copies']),blocks*bc*2//256)
                if t>=2048:
                    self.assertLess(max(w['updates'])/(sum(w['updates'])/8),1.15)

    def test_gqa_first_row_and_future_invariance(self):
        t=128;q,k,v=oracle.inputs(t,14,2)
        base=oracle.reference(t,128)
        np.testing.assert_array_equal(base[:,0],v[np.arange(14)//7,0])
        # Large changes exclusively to future keys/values cannot affect earlier
        # rows; modifications to KV head1 must not affect query heads0..6.
        kk=k.copy();vv=v.copy();kk[1,64:]*=-2;vv[1,64:]*=-4
        with patch.object(oracle,'inputs',return_value=(q,kk,vv)):
            changed=oracle.reference(t,128)
        np.testing.assert_array_equal(base[:,:64],changed[:,:64])
        np.testing.assert_array_equal(base[:7],changed[:7])
        self.assertTrue(np.any(base[7:,64:]!=changed[7:,64:]))


if __name__=='__main__':unittest.main()
