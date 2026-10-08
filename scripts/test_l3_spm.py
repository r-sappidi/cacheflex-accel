"""ABI and generated-protocol regression checks; not runtime correctness proof."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'gem5/src/mem/ruby/protocol'
OLD=ROOT/'results/cacheflex_l3_spm/initial_20260928/preserved/gem5/src/mem/ruby/protocol'
def action(s,n):
    start=s.index('  action('+n+',');i=s.index('{',start);depth=1
    while depth:
        i+=1
        if s[i]=='{':depth+=1
        elif s[i]=='}':depth-=1
    return s[start:i+1]
class L3Tests(unittest.TestCase):
    def test_instruction_prefetch_stale_fill_cleanup(self):
        text=(PROTOCOL/'MESI_Three_Level_SPM-L0cache.sm').read_text()
        block=text.split('transition(PF_Inst_IS, Data_Stale, I) {')[1].split('}')[0]
        for op in ('u_writeInstToCache','s_deallocateTBE','ff_deallocateCacheBlock',
                   'o_popIncomingResponseQueue','kd_wakeUpDependents'):
            self.assertIn(op+';',block)
        self.assertNotIn('hx_ifetch_hit',block)
    def test_private_actions_unchanged(self):
        old=(OLD/'MESI_Three_Level_SPM-L1cache.sm').read_text()
        new=(PROTOCOL/'MESI_Three_Level_SPM-L1cache.sm').read_text()
        for name in ('spm_installFromTBE','spm_installFromCache','spm_ackToL0'):
            self.assertEqual(action(old,name),action(new,name))
    def test_namespace_and_slots(self):
        seen=set()
        for bank in range(8):
            for way in range(8):
                for row in range(512):
                    raw=(1<<59)|(bank<<20)|(way<<16)|(row<<6)
                    canonical=0x68000000|(raw&0x77ffff)
                    self.assertNotIn(canonical,seen);seen.add(canonical)
                    self.assertEqual((canonical>>20)&7,bank)
                    self.assertEqual((canonical>>16)&7,way)
                    self.assertEqual((canonical>>6)&1023,row)
                    self.assertNotEqual(canonical,0x60000000|(raw&0x7ffff))
        self.assertEqual(len(seen)*64,2*1024*1024)
    def test_no_extra_backing(self):
        text=(ROOT/'scripts/l3_spm_se.py').read_text()
        self.assertNotIn('spm_separate_backing=True',text)
        self.assertIn('llc.L2cache.num_spm_ways=',text)
        c=(ROOT/'gem5/src/mem/ruby/structures/CacheMemory.cc').read_text()
        self.assertIn('m_num_spm_ways > m_coherent_assoc - 2',c)
    def test_source_response_not_blocked_on_install_network(self):
        text=(PROTOCOL/'MESI_Three_Level_SPM-L1cache.sm').read_text()
        for name in ('spm_bufferResponseData','spm_bufferCacheData','l3_schedule'):
            self.assertNotIn('enqueue(',action(text,name))
        self.assertIn('transition(IX_D, L3SourceData, L3_SEND)',text)
        self.assertIn('transition(L3_WB_READ, L3WritebackData, L3_WB_SEND)',text)
    def test_payload_lengths(self):
        text=(ROOT/'gem5/src/mem/ruby/network/Network.cc').read_text()
        for n in (128,256):self.assertIn(f'return m_control_msg_size + {n};',text)
    def test_unclaimed_read_preserves_private_zero_semantics(self):
        text=(ROOT/'gem5/src/mem/ruby/structures/CacheMemory.cc').read_text()
        read=text.split('CacheMemory::readSPMData(')[1].split('CacheMemory::writeSPMData(')[0]
        self.assertIn('result.clear();',read)
        self.assertIn('entry != nullptr && entry->isScratchpad()',read)
        validate=text.split('CacheMemory::validateL3SPM(')[1].split('\n}\n',1)[0]
        self.assertIn('m_cache_num_sets',validate)
        self.assertNotIn('!e || !e->isScratchpad()',validate)
    def test_shared_claim_reuses_normal_eviction(self):
        text=(PROTOCOL/'MESI_Three_Level_SPM-L2cache.sm').read_text()
        start=text.index('if (in_msg.Type == CoherenceRequestType:L3_SPM)')
        end=text.index('} else if (in_msg.Type == CoherenceRequestType:FLUSH)',start)
        self.assertIn('trigger(Event:L2_Replacement, victim',text[start:end])
        self.assertIn('trigger(Event:L2_Replacement_clean, victim',text[start:end])
if __name__=='__main__':unittest.main()
