import random,unittest
from collections import Counter
from analyze_residency import TimeWeightedCounter
class Integral(unittest.TestCase):
    def test_exact(self):
        for seed in range(20):
            rng=random.Random(seed);plain=Counter();expected=Counter();area=Counter();fast=TimeWeightedCounter(area,0);t=0
            for _ in range(2000):
                delta=rng.randrange(5)
                for key,value in plain.items():expected[key]+=value*delta
                t+=delta;fast.advance(t);key=rng.randrange(30)
                change=-1 if plain[key] and rng.randrange(2) else 1
                plain[key]+=change;fast[key]+=change
            for key,value in plain.items():expected[key]+=value*7
            fast.advance(t+7);fast.flush()
            self.assertEqual(dict(plain),dict(fast));self.assertEqual(dict(expected),dict(area))
if __name__=='__main__':unittest.main()
