import unittest
import numpy as np
from reference import merge,reference

class ReferenceTests(unittest.TestCase):
    def test_partition_merge(self):
        rng=np.random.default_rng(7)
        for scale in (.02,1,80):
            s=rng.normal(size=(5,32))*scale;v=rng.normal(size=(32,64))
            ms=[];ls=[];os=[]
            for a,b in ((0,3),(3,15),(15,32)):
                m=s[:,a:b].max(axis=1);w=np.exp(s[:,a:b]-m[:,None])
                ms.append(m);ls.append(w.sum(axis=1));os.append(w@v[a:b])
            got=merge(np.array(ms,dtype=np.float32),np.array(ls,dtype=np.float32),np.array(os,dtype=np.float32))
            w=np.exp(s-s.max(axis=1)[:,None]);want=(w@v)/w.sum(axis=1)[:,None]
            np.testing.assert_allclose(got,want,rtol=.003,atol=.002)

    def test_slot_layout(self):
        for sets in (128,1024):
            slots=[((r%sets)<<6)|(((r//sets)*2+w)<<16) for r in range(3*sets) for w in range(2)]
            self.assertEqual(len(slots),len(set(slots)))
            self.assertLess(max(slots),6<<16)
            for j in range(0,3*sets,128):
                self.assertEqual(j//sets,(j+127)//sets)

    def test_one_partition(self):
        from single_head_reference import reference as old
        np.testing.assert_array_equal(reference(128,128,1),old(128,128))

if __name__=='__main__':unittest.main()
