"""Host permutation test independent of generated SVE source."""
import unittest
import numpy as np
def zip_vectors(a,b,half):
    h=len(a)//2;s=slice(half*h,(half+1)*h)
    return np.stack((a[s],b[s]),axis=1).reshape(-1)
def pack64(a):
    r=np.zeros((8,128),np.uint16);r[:,:64]=a
    t=[zip_vectors(r[i],r[i+1],h).view(np.uint32) for i in (0,2,4,6) for h in (0,1)]
    u=[zip_vectors(t[i],t[i+2],h).view(np.uint64) for i in (0,1,4,5) for h in (0,1)]
    d=[zip_vectors(u[i],u[i+4],h).view(np.uint16) for i in range(4) for h in (0,1)]
    return np.concatenate(d[:4])
class PackingTests(unittest.TestCase):
    def test_lane_identifiers(self):
        a=np.arange(512,dtype=np.uint16).reshape(8,64)
        np.testing.assert_array_equal(pack64(a),a.T.reshape(-1))
    def test_random_bits(self):
        rng=np.random.default_rng(137)
        for _ in range(20):
            a=rng.integers(0,65536,(8,64),dtype=np.uint16)
            np.testing.assert_array_equal(pack64(a),a.T.reshape(-1))
if __name__=='__main__':unittest.main()
