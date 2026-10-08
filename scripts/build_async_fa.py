"""Isolated same-ELF JIT/whole-panel/chunked lookahead FA experiment."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from fa_export_patch import patch


def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def build(out):
    old=ROOT/'results/cacheflex_flash_attention/single_head_20260915/builds/spm_g4'
    out.mkdir(parents=True)
    for name in ('driver.cpp','kernels.hpp','pv_fullwidth.hpp','build_command.sh'):
        shutil.copy2(old/name,out/name)
    s=(out/'driver.cpp').read_text()
    s=replace_once(s,'static size_t search_bc;','static size_t search_bc;\nstatic int async_stage_mode=0; // 0=JIT,1=panel-ahead,2=chunk-ahead')
    s=replace_once(s,'return (CACHEFLEX_FLASH_STAGE_K?banks*(search_bc/128)*64:0)+bank*search_bc;',
        '// 128 physical sets/way: use ways 0/1 and 2/3, not aliased set 128+.\n    return bank*(2u<<10);')
    s=replace_once(s,'if(CACHEFLEX_FLASH_K_OVERLAP && !CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {',
        'if(async_stage_mode && CACHEFLEX_FLASH_K_OVERLAP && !CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {')
    s=replace_once(s,'} else if(CACHEFLEX_FLASH_K_OVERLAP) {',
        '} else if(CACHEFLEX_FLASH_K_OVERLAP && async_stage_mode) {')
    s=replace_once(s,'if(next<T) { int nbc=std::min(Bc,T-next);',
        'if(async_stage_mode==1 && next<T) { int nbc=std::min(Bc,T-next);')
    s=replace_once(s,'stage_kv(s.KT.data(),s.V.data(),T,d,j,bc,bb,CACHEFLEX_FLASH_MULTICAST);',
        'stage_kv(s.KT.data(),s.V.data(),T,d,j,bc,bb,CACHEFLEX_FLASH_MULTICAST,kset_base);')
    s=replace_once(s,'auto qk_phase = [&](size_t local_ib) {', '''auto stage_next_chunk = [&](size_t local_ib) {
        if(async_stage_mode!=2 || j+Bc>=T) return;
        CF_PHASE(stage_K_panel);
        const int next=j+Bc, nbc=std::min(Bc,T-next);
        const int first=(nbc*local_ib)/group_count;
        const int last=(nbc*(local_ib+1))/group_count;
        const size_t dst=search_vbase((panel+1)&1);
        for(int r=first;r<last;++r) for(size_t way=0;way<2;++way)
          copy_line(((dst+r)<<6)|(way<<16),s.V.data()+(size_t)(next+r)*d+way*32);
      };
      auto qk_phase = [&](size_t local_ib) {
        stage_next_chunk(local_ib);''')
    s=replace_once(s,'search_bc=Bc;', '''search_bc=Bc;
 async_stage_mode=argc>5?atoi(argv[5]):0;
 if(Bc!=128 || Br!=16 || d!=64 || async_stage_mode<0 || async_stage_mode>2) return 2;
 printf("ASYNC_STAGING mode=%d buffers=2 ways=4 total_L2_unchanged=1\\n",async_stage_mode);''')
    (out/'driver.cpp').write_text(s)
    script=(out/'build_command.sh').read_text().replace(str(old),str(out))
    assert script.count('-DCACHEFLEX_FLASH_K_OVERLAP=0')==2
    script=script.replace('-DCACHEFLEX_FLASH_K_OVERLAP=0','-DCACHEFLEX_FLASH_K_OVERLAP=1')
    (out/'build_command.sh').write_text(script)
    with (out/'build.log').open('w') as log:
        subprocess.run(['bash',str(out/'build_command.sh')],stdout=log,stderr=subprocess.STDOUT,check=True)
    guest=patch(out/'guest')
    (out/'geometry.json').write_text(json.dumps(dict(Br=16,Bc=128,G=4,d=64,
        buffer0_ways=[0,1],buffer1_ways=[2,3],sets_per_buffer=128,bytes_per_buffer=16384,
        modes=['JIT alternating buffers','whole-panel lookahead','chunked lookahead'],
        note='Same ELF for all modes. Four eligible ways consume existing 64KiB L2; no added SRAM.'),indent=2))
    return guest


if __name__=='__main__':print(build(Path(sys.argv[1]).resolve()))
