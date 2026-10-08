from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analyze_residency import analyze, events, phase

class ResidencyTests(unittest.TestCase):
    def test_eviction_reuse_is_not_elapsed_penalty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            shared=['Q','K','V','O','KT','Kpack']
            local=['qtile','spanel','ptile','pvpanel','qkap','pvap','oacc','m','l']
            lines=['FLASH_SHARED '+' '.join(f'{name}={hex((i+1)*0x100000)}' for i,name in enumerate(shared))]
            for core in range(8):
                lines.append(f'FLASH_ADDR tid={core} '+' '.join(f'{name}={hex(0x1000000+core*0x100000+i*0x10000)}' for i,name in enumerate(local)))
            (p/'run.log').write_text('\n'.join(lines))
            cache='system.ruby.l1_cntrl0.cache';seq='system.ruby.l0_cntrl0.sequencer'
            (p/'debug.trace').write_text('\n'.join([
                '1: pt: CFRES MAP va=0x100000 pa=0x1000 bytes=4096',
                f'50: {cache}: CFRES ALLOC addr=0x1000 set=0 way=0 spm=0',
                '100: CFRES BEGIN core=0 work=0',
                '110: system.cpu0.commit: CFRES HEAD pc=0x4000 executed=0',
                f'120: {cache}: CFRES PROBE incoming=0x2000 victim=0x1000 set=0 way=0',
                f'130: {cache}: CFRES REMOVE addr=0x1000 set=0 way=0 permission=Invalid',
                f'160: {seq}: CFRES ISSUE id=0xabc addr=0x1000 va=0x100000 pc=0x4000 bytes=64 type=LD core=0',
                f'200: {seq}: CFRES DONE id=0xabc addr=0x1000 va=0x100000 pc=0x4000 bytes=64 type=LD source=L2Cache external=1',
                '300: CFRES END core=0 work=0']))
            with patch('analyze_residency.symbols',return_value={0x4000:[('sve_mla_8x3VL','kernel.cpp:1')]}):
                result=analyze(p,p/'unused.debug',128)
            self.assertEqual(result['average_allocated_bytes'][cache+'|Q'],9.6)
            self.assertEqual(result['removals'][cache+'|Q|QK|True|True'],1)
            self.assertEqual(result['read_responses']['Q|L2Cache|True|summed_service_ticks'],40)
            self.assertEqual(result['roi_ticks'],200)
    def test_native_pv_phase(self):
        self.assertEqual(phase([('cache_pv_8x64_vl16','x:1')]),'PV')

if __name__=='__main__':unittest.main()
