"""Build an isolated single-ELF factorial; active guests are never edited."""
import hashlib,json,shutil,subprocess,sys,os,resource
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
BASE=R/'results/cacheflex_flash_attention/bottleneck_20261004';G=BASE/'guest_v1'
OLD=R/'results/cacheflex_flash_attention/qwen_matrix_20260930/guest_v3'
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'))
from export_safe_elf import patch
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 G.mkdir(parents=True,exist_ok=False)
 for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
 s=(OLD/'driver.cpp').read_text()
 s=s.replace('static int layout_mode=0;', 'static int layout_mode=0;\nstatic int stage_simple=0, packed_producer=0;')
 old='''    PHASE(stage);
    for(size_t r=0;r<rows;++r) for(size_t w=0;w<2;++w) {'''
 new='''    PHASE(stage);
    if(stage_simple && rows<=sets) {
        for(size_t r=0;r<rows;++r) {
            size_t a=r<<6, b=a|(1UL<<16);
            asm volatile("SPMCP_64_IMM %0, [%1, #0]\\n" ::"r"(a),"r"(src+r*64):"memory");
            asm volatile("SPMCP_64_IMM %0, [%1, #0]\\n" ::"r"(b),"r"(src+r*64+32):"memory");
        }
        return;
    }
    for(size_t r=0;r<rows;++r) for(size_t w=0;w<2;++w) {'''
 assert s.count(old)==1;s=s.replace(old,new)
 # Write rounded FP16 directly to the eight-row interleaved PV panel.
 old='sum=svadd_f32_x(pf,sum,z);svst1_f16(ph,pr+x,cf_dense_f32_to_f16(pf,z));'
 new='''sum=svadd_f32_x(pf,sum,z);
                        auto rounded=cf_dense_f32_to_f16(pf,z);
                        if(packed_producer) {
                            auto bits=svunpklo_u32(svreinterpret_u16_f16(rounded));
                            auto* dst=reinterpret_cast<uint16_t*>(pa.p+(r/8)*bc*8+x*8+(r%8));
                            svst1h_scatter_u32index_u32(pf,dst,svindex_u32(0,8),bits);
                        } else svst1_f16(ph,pr+x,rounded);'''
 assert s.count(old)==3;s=s.replace(old,new,1)
 old='cf_vector_pack_A(prob.p,s.bc,pa.p,br,r,bc,0,16,bc);'
 assert old in s;s=s.replace(old,'if(!packed_producer) '+old,1)
 s=s.replace('if(argc!=15)', 'if(argc!=17)')
 old='    layout_mode=atoi(argv[14]);'
 new='''    stage_simple=atoi(argv[15]);packed_producer=atoi(argv[16]);
    if((stage_simple!=0&&stage_simple!=1)||(packed_producer!=0&&packed_producer!=1))return 2;
    layout_mode=atoi(argv[14]);
    if(layout_mode>1||br!=16||bc!=256||g!=16||sets!=1024)return 2;
    std::printf("BOTTLENECK stage_simple=%d packed_producer=%d\\n",stage_simple,packed_producer);'''
 assert s.count(old)==1;s=s.replace(old,new)
 (G/'driver.cpp').write_text(s)
 (G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
 def limits():
  resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
 subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True,preexec_fn=limits)
 patch(G/'guest',G/'guest.export-safe')
 (G/'disassembly.txt').write_text(subprocess.check_output(['aarch64-linux-gnu-objdump','-d','-C',str(G/'guest.export-safe')],text=True))
 (G/'frozen.json').write_text(json.dumps({str(p):sha(p) for p in [*G.iterdir(),*OLD.glob('*.hpp'),OLD/'driver.cpp',H/'build.py'] if p.is_file()},indent=2)+'\n')
if __name__=='__main__':main()
