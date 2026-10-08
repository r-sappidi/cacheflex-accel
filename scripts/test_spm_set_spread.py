"""Mapping invariants and trace qualification; dynamic simulator gates separate."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from analyze_spm_set_spread import analyze

ROOT=Path(__file__).resolve().parents[1]

class PlacementTests(unittest.TestCase):
    def test_bijective_all_supported_geometries(self):
        for sets in (128,256,512,1024):
            for stride in (0,sets//2):
                slots={(w,(s+w*stride)&(sets-1)) for w in range(8) for s in range(sets)}
                self.assertEqual(len(slots),8*sets)
    def test_same_capacity_even_distribution(self):
        old=Counter(s for w in range(2) for s in range(512))
        new=Counter((s+w*512)&1023 for w in range(2) for s in range(512))
        self.assertEqual(sum(old.values()),1024)
        self.assertEqual(sum(new.values()),1024)
        self.assertEqual(set(old.values()),{2})
        self.assertEqual(set(new.values()),{1})
    def test_all_slot_paths_use_mapping(self):
        src=(ROOT/'gem5/src/mem/ruby/structures/CacheMemory.cc').read_text()
        self.assertEqual(src.count('spmPhysicalSet(spm_set, spm_way)'),3)
        self.assertIn('m_spm_tag_sets[address] = cacheSet',src)
        self.assertIn('m_spm_tag_sets.erase(address)',src)
        self.assertNotIn('SPM wide read slot %#x changed set',src)
        self.assertIn('m_spm_set_stride = p.spm_set_stride',src)
    def test_trace_requires_retired_read_refill_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace'
            # Test fixture only; production traces are emitted by gem5.
            p.write_text('''1: system.ruby.l0_cntrl0.sequencer: CFRES ISSUE id=0x1 addr=0x200 va=0x400200 pc=0x401980 bytes=64 type=LD core=0
2: system.cpu0.commit: CFRES RETIRED va=0x400200 pc=0x401980 bytes=64 store=0 spm=0
3: system.ruby.l1_cntrl0.cache: CFRES REMOVE addr=0x200 set=8 way=0 permission=Read_Only
4: system.ruby.l1_cntrl0.cache: CFRES ALLOC addr=0x200 set=8 way=0 spm=0 entry=0x1
5: system.ruby.l0_cntrl0.sequencer: CFRES ISSUE id=0x2 addr=0x200 va=0x400200 pc=0x401980 bytes=64 type=LD core=0
6: system.cpu0.commit: CFRES RETIRED va=0x400200 pc=0x401980 bytes=64 store=0 spm=0
7: system.ruby.l1_cntrl0.cache: CFRES REMOVE addr=0x300 set=12 way=0 permission=Read_Only
8: system.ruby.l1_cntrl0.cache: CFRES ALLOC addr=0x300 set=12 way=0 spm=0 entry=0x2
''')
            x=analyze(p)
            self.assertEqual(x['qualified_removal_refill_reuse_by_set_pc'],{'set8:packed_K':1})
            self.assertEqual(sum(x['private_l2_removals_by_set'].values()),2)

if __name__=='__main__':unittest.main()
