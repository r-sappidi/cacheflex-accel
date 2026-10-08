"""Accounting invariants; timing equivalence is enforced separately by runner."""
import copy
import unittest
from run_access_aggregate import validate_aggregate

class AggregateContracts(unittest.TestCase):
    def sample(self):
        group=dict(count=1,total_ticks=300,category_ticks={'CPU':100,'network':200},
                   log2_tick_histogram=[0]*8+[1]+[0]*55)
        return dict(begun=1,completed=1,inflight=0,live_messages=0,live_packets=0,
                    missing_install=0,overlap_ticks=0,llc_match_failures=0,
                    groups={k:copy.deepcopy(group) for k in ('all','core_0','path_llc_hit')})
    def test_valid(self):
        self.assertTrue(validate_aggregate(self.sample(),1)['additive_partition'])
    def test_missing_copy(self):
        with self.assertRaises(AssertionError):validate_aggregate(self.sample(),2)
    def test_gap(self):
        d=self.sample();d['groups']['all']['category_ticks']['CPU']=99
        with self.assertRaises(AssertionError):validate_aggregate(d,1)
    def test_overlap(self):
        d=self.sample();d['overlap_ticks']=1
        with self.assertRaises(AssertionError):validate_aggregate(d,1)
    def test_unfinished(self):
        for k in ('inflight','live_packets','live_messages','missing_install'):
            d=self.sample();d[k]=1
            with self.assertRaises(AssertionError):validate_aggregate(d,1)
    def test_path_sum(self):
        d=self.sample();d['groups'].pop('path_llc_hit')
        with self.assertRaises(AssertionError):validate_aggregate(d,1)
    def test_bad_histogram(self):
        d=self.sample();d['groups']['all']['log2_tick_histogram'][0]+=1
        with self.assertRaises(AssertionError):validate_aggregate(d,1)

if __name__=='__main__':unittest.main()
