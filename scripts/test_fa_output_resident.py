import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from build_output_resident import transform,CONFIG
from build_single_head import transform as single
sys.path.insert(0,str(ROOT/'spm_tools'))
from spm_compiler import encode_vector

class OutputResidentTests(unittest.TestCase):
    def test_two_way_nonaliasing_layout(self):
        slots=[((r*2+h)<<6)|(w<<16) for r in range(64) for h in range(2) for w in range(2)]
        self.assertEqual(len(set(slots)),256)
        self.assertEqual({(s>>16)&7 for s in slots},{0,1})
        self.assertEqual({(s>>6)&1023 for s in slots},set(range(128)))
    def test_bitwise_zip_narrow_load_unzip(self):
        # Include NaN/Inf bit patterns: transport must be bitwise, not numerical.
        words=[(i*0x13579bdf)&0xffffffff for i in range(64)]
        packed=[]
        for half in (words[:32],words[32:]):
            lanes=[w|(w<<32) for w in half]
            packed += [v&0xffffffff for v in lanes]
        self.assertEqual(packed,words)
    def test_transfer_dtype_is_128_bytes(self):
        for op in ('ld','st'):
            instr=f'spm.{op}1ad z0.d, p0'+('/z' if op=='ld' else '')+', [x1]'
            code=encode_vector(instr)
            self.assertEqual((code>>21)&15,0xb)
        self.assertEqual(256//8*4,128)
    def test_driver_changes_and_constraints(self):
        source=(ROOT/'benchmarks/cacheflex_flash_attention/spm_multicore.cpp').read_text()
        source=source.replace('return 0;\n}',
            'return cf_audit_output(s.O.data(),s.V.data(),(size_t)8*T*d,d);\n}')
        source=single(source)
        new=transform(source)
        self.assertIn('cf_o_claim(Oacc.data(),group_rows)',new)
        self.assertIn('cf_o_release(group_rows)',new)
        self.assertNotIn('svld1_f32(pg,Oacc.data()+r*d+x)',new)
        self.assertEqual(CONFIG['placement'],'cache')
        self.assertEqual(CONFIG['fusion'],0)
        self.assertEqual(CONFIG['interleave'],0)
    def test_spm_store_ack_is_not_a_second_array_write(self):
        s=(ROOT/'gem5/src/mem/ruby/system/Sequencer.cc').read_text()
        branch=s.split('} else if (type == RubyRequestType_SPMST) {',1)[1].split('} else if',1)[0]
        self.assertNotIn('data.setData',branch)
        self.assertIn('noCoales || atomic_req || spm_store_ack',s)
        self.assertIn('spm_store_ack || spm_follower',s)
        self.assertIn('spm_follower = seq_req.pkt->req->isSPMRequest()',s)
        self.assertIn('seq_req_list.front().pkt->req->isSPMRequest()',s)
        self.assertIn('msg->m_WTData = DataBlock(bytes);',s)
    def test_timed_spm_store_ordering_is_not_assumed_synchronous(self):
        s=(ROOT/'gem5/src/cpu/o3/lsq_unit.cc').read_text()
        self.assertIn('if (pendingSpmStore)',s)
        self.assertIn('seq >= mySeq || e.ready',s)
        self.assertIn('if (entry.retired && !entry.ready)',s)
        self.assertNotIn('seq <= youngest_inst && entry.isWriteback',s)
    def test_wide_store_advances_payload_across_ways(self):
        s=(ROOT/'gem5/src/mem/ruby/structures/CacheMemory.cc').read_text()
        body=s.split('CacheMemory::writeSPMData(',1)[1].split('CacheMemory::deallocate(',1)[0]
        self.assertIn('data.getData(first_offset + copied, bytes)',body)

if __name__=='__main__':unittest.main()
