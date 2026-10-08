"""Isolated FA44 guest; existing source and ELF remain immutable."""
import json,hashlib,shutil,subprocess,sys,os,resource
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
OLD=R/'results/cacheflex_flash_attention/bottleneck_20261004/guest_v1'
G=R/'results/cacheflex_flash_attention/paced_staging_20261004/guest_v1'
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'));from export_safe_elf import patch

def transform(s):
 s=s.replace('static int stage_simple=0, packed_producer=0;', 'static int stage_simple=0, packed_producer=0;\nstatic int staging_schedule=0; // 0 original,1 paced,2 overlap,3 double-buffer burst')
 marker='static void release(size_t rows,size_t sets) {'
 helper='''// Copy a bounded subset to its final slots. Hardware RAW/WAR tracking is retained.
static void stage_chunk(const __fp16* src,size_t first,size_t count,size_t slot_offset,size_t sets) {
    PHASE(stage_chunk);
    for(size_t r=first;r<first+count;++r)for(size_t w=0;w<2;++w) {
        size_t dst=(vbase(slot_offset+r,sets)<<6)|(w<<16);
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\\n" ::"r"(dst),"r"(src+r*64+w*32):"memory");
    }
}
'''
 assert s.count(marker)==1;s=s.replace(marker,helper+marker)
 old='''            if(s.spm && !s.kv) stage(vhead+j*64,bc,s.sets);
            for(int bi=0;bi<nr;bi+=s.br) {'''
 new='''            const size_t slot_offset=(staging_schedule>=2)?((j/s.bc)%2)*s.bc:0;
            if(s.spm && !s.kv) {
                if(staging_schedule==0) stage(vhead+j*64,bc,s.sets);
                else if(staging_schedule==3) stage_chunk(vhead+j*64,0,bc,slot_offset,s.sets);
                // Paced: copy during first softmax block. Overlap: only startup
                // is paced; subsequent tiles were staged by the previous tile.
            }
            for(int bi=0;bi<nr;bi+=s.br) {'''
 assert s.count(old)==1;s=s.replace(old,new)
 old='''                    else svst1_f32(pf,dst,svmul_n_f32_x(pf,svld1_f32(pf,dst),a));
                }
                PHASE(pv);'''
 new='''                    else svst1_f32(pf,dst,svmul_n_f32_x(pf,svld1_f32(pf,dst),a));
                    if(s.spm && bi==0 && (staging_schedule==1 || (staging_schedule==2 && j==kbegin)))
                        stage_chunk(vhead+j*64,r*16,16,slot_offset,s.sets);
                }
                PHASE(pv);'''
 assert s.count(old)==1;s=s.replace(old,new)
 old='const auto vptr=s.spm?vbase(0,s.sets):reinterpret_cast<size_t>(vhead+j*64);'
 assert s.count(old)==1;s=s.replace(old,'const auto vptr=s.spm?vbase(slot_offset,s.sets):reinterpret_cast<size_t>(vhead+j*64);')
 old='''                }
            }
        }
        if(s.kv) {'''
 new='''                }
                // nr=256, Br=16: sixteen 16-row copy batches, one after each
                // PV block, complete the next V tile exactly once before use.
                if(s.spm && staging_schedule==2 && j+s.bc<kend)
                    stage_chunk(vhead+(j+s.bc)*64,bi,16,((j/s.bc+1)%2)*s.bc,s.sets);
            }
        }
        if(s.kv) {'''
 assert s.count(old)==1;s=s.replace(old,new)
 old='if(s.spm)release(s.kv?local:s.bc,s.sets);'
 assert s.count(old)==1;s=s.replace(old,'if(s.spm)release(staging_schedule>=2 && s.t>=512?2*s.bc:(s.kv?local:s.bc),s.sets);')
 s=s.replace('if(argc!=17)','if(argc!=18)')
 marker='    layout_mode=atoi(argv[14]);'
 s=s.replace(marker,'''    staging_schedule=atoi(argv[17]);
    if(staging_schedule<0||staging_schedule>3||(!spm&&staging_schedule)||
       accumulator_mode!=1||kernel_variant!=1||stage_simple||packed_producer)return 2;
    std::printf("STAGING_SCHEDULE mode=%d chunk_rows=16 slots=%d\\n",staging_schedule,staging_schedule>=2?2:1);
'''+marker)
 return s

def main():
 G.mkdir(parents=True,exist_ok=False)
 for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
 (G/'driver.cpp').write_text(transform((OLD/'driver.cpp').read_text()))
 (G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
 def limit():
  resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));budget=json.loads((R/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,[budget['cpu_set'][-1]])
 subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True,preexec_fn=limit)
 patch(G/'guest',G/'guest.export-safe')
 (G/'disassembly.txt').write_bytes(subprocess.check_output(['aarch64-linux-gnu-objdump','-d','-C',str(G/'guest.export-safe')]))
 files=[p for p in G.iterdir() if p.is_file()]+list(OLD.glob('*.hpp'))+[OLD/'driver.cpp',H/'build.py']
 (G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
if __name__=='__main__':main()
