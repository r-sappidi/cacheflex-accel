import sys,shutil,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/cacheflex_flash_attention/v_only_ring_20261003';G=BASE/'guest_v1'
OLD=ROOT/'results/cacheflex_flash_attention/qwen_matrix_20260930/guest_v3'
sys.path.insert(0,str(ROOT/'benchmarks/cacheflex_flash_attention'))
from export_safe_elf import patch

def main():
 G.mkdir(parents=True)
 for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
 s=(OLD/'driver.cpp').read_text();a=s.index('static void* legacy_worker(');b=s.index('\n}',a)+2
 w=s[a:b].replace('legacy_worker','vring_worker')
 w=w.replace('    pthread_barrier_wait(&s.barrier);','    if(m5_sum(0x52494e47,0,c.id,0,0,0)!=0)std::abort();\n    pthread_barrier_wait(&s.barrier);',1)
 w=w.replace('        for(int j=kbegin;j<kend;j+=s.bc) {\n            const int bc=std::min(s.bc,kend-j);\n            if(s.spm && !s.kv) stage(vhead+j*64,bc,s.sets);', '''        for(int wave=0;wave<s.t;wave+=8*s.bc)for(int step=0;step<8;++step) {
            const int j=wave+((c.id-step+8)%8)*s.bc,bc=s.bc;
            if(step==0 || layout_mode==16) ring_stage_v(vhead+j*64,bc);
            asm volatile("dsb sy":::"memory");
            pthread_barrier_wait(&s.barrier);
            if(step && layout_mode==17 && c.id==0)
                if(m5_sum(0x52494e47,1,0,768,bc,2)!=0)std::abort();
            pthread_barrier_wait(&s.barrier);
            if(j>=kend)continue;''')
 w=w.replace('s.spm?vbase(0,s.sets):','s.spm?768UL:')
 w=w.replace('    if(s.spm)release(s.kv?local:s.bc,s.sets);','''    asm volatile("dsb sy":::"memory");
    for(int r=0;r<s.bc;++r)for(int way=0;way<2;++way){
        size_t slot=(size_t(way)<<16)+((768UL+r)<<6);
        asm volatile("SPMREL_8_IMM xzr, [%0, #0]\\n"::"r"(slot):"memory");
    }
    asm volatile("dsb sy":::"memory");''')
 w=w.replace('if(c.id==0){m5_work_end','if(c.id==0){m5_sum(0x52494e47,2,0,0,0,0);m5_work_end')
 helper=(ROOT/'benchmarks/qwen/fa/ring/guest_helpers.hpp').read_text();a=helper.index('static void ring_stage_v');b=helper.index('\n}',a)+2
 s=s.replace('int main(int argc',helper[a:b]+'\n'+w+'\nint main(int argc')
 s=s.replace('layout_mode>11','layout_mode>17').replace('if(layout_mode>=9&&','if(layout_mode>=9&&layout_mode<=11&&')
 s=s.replace('    auto worker=layout_mode>=9?', '    if(layout_mode>=12 && (layout_mode<16 || br!=16 || bc>256 || t%(8*bc) || (t/(br*g))%8 || !spm))return 2;\n    auto worker=layout_mode>=16?vring_worker:layout_mode>=9?')
 s=s.replace('export_layout_pad[512]','export_layout_pad[1024]');(G/'driver.cpp').write_text(s)
 (G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
 subprocess.run(['bash',str(G/'build.sh')],cwd=ROOT,check=True)
 patch(G/'guest',G/'guest.export-safe')
 (G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in G.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
