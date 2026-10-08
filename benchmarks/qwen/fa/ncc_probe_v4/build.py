import json,sys,subprocess,shutil,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[4];H=Path(__file__).resolve().parent
ROOT=R/'results/cacheflex_flash_attention/ncc_probe_20261005_v4';G=ROOT/'guest_v1';B=ROOT/'build_v1'
OLD=R/'results/cacheflex_flash_attention/bottleneck_20261004/guest_v1'
G.mkdir(parents=True);B.mkdir()
for p in OLD.glob('*.hpp'):shutil.copy2(p,G/p.name)
s=(OLD/'driver.cpp').read_text().replace('static int layout_mode=0;','static int layout_mode=0;\nstatic int ncc_mode=0;')
# Exact same PV operand loads, with arithmetic and recurrence removed.
pv=(OLD/'pv_direct.hpp').read_text();a=pv.index('static __attribute__((noinline,aligned(64))) void cf_pv16_spm_cpu_paired(');z=pv.index('static __attribute__',a+20);f=pv[a:z]
a=f.index('    asm volatile(');b=f.index('"mov x7, %x[alpha]');cl=f.index('    : : [A]')
body=f[a:b];body='\n'.join(x for x in body.splitlines() if not '"fmla ' in x and not re.search(r'"mov z(?:8|9|1[0-5])\.b',x))+'\n'
clobber=f[f.index('    : "cc"',cl):];helper='static __attribute__((noinline)) void pv_read_traffic(const __fp16* p,size_t v,size_t k) {\n'+body+'    : : [A] "r"(p),[B] "r"(v),[K] "r"(k)\n'+clobber
helper+='''
static void touch_read(const void* p,size_t bytes) {
 for(size_t x=0;x<bytes;x+=256){const char* q=static_cast<const char*>(p)+x;
 asm volatile("ptrue p0.b\\nld1b {z0.b}, p0/z, [%0]\\n"::"r"(q):"p0","z0","memory");}
}
static void touch_rw(void* p,size_t bytes) {
 for(size_t x=0;x<bytes;x+=256){char* q=static_cast<char*>(p)+x;
 asm volatile("ptrue p0.b\\nld1b {z0.b}, p0/z, [%0]\\nst1b {z0.b}, p0, [%0]\\n"::"r"(q):"p0","z0","memory");}
}
static void ncc_pack(int op,const void* src,void* dst,size_t bytes) {
 touch_read(src,bytes);asm volatile("dsb sy":::"memory");
 m5_sum(0x4e434350,op,reinterpret_cast<size_t>(src),reinterpret_cast<size_t>(dst),0,ncc_mode);
 touch_rw(dst,bytes);asm volatile("dsb sy":::"memory");
}
'''
s=s.replace('static void* legacy_worker',helper+'\nstatic void* legacy_worker',1)
old='''        for(int d=0;d<64;++d) for(int x=0;x<128;++x)
            s.kpack.p[kh*s.t*64+(j/128*64+d)*128+x]=s.k.p[kh*s.t*64+(j+x)*64+d];'''
assert s.count(old)==1
s=s.replace(old,'''        if(ncc_mode==1) ncc_pack(1,s.k.p+kh*s.t*64+j*64,s.kpack.p+kh*s.t*64+j*64,128*64*2);
        else {
'''+old+'\n        }')
old='if(!packed_producer) cf_vector_pack_A(prob.p,s.bc,pa.p,br,r,bc,0,16,bc);';assert s.count(old)==1
s=s.replace(old,'''if(ncc_mode==2) ncc_pack(2,prob.p,pa.p,16*256*2);
                    else '''+old)
marker='''                    if(kernel_variant) {
                        auto slot='''
new='''                    if(ncc_mode>=3 && ncc_mode<=8) {
                        // Drain makes functional proxy reads safe. Controls below
                        // retain this boundary to expose diagnostic synchronization cost.
                        if(ncc_mode>=4)pv_read_traffic(pa.p,0,bc);
                        asm volatile("dsb sy":::"memory");
                        m5_sum(0x4e434350,3,reinterpret_cast<size_t>(pa.p),reinterpret_cast<size_t>(pv.p),0,ncc_mode);
                        for(int a=0;a<16;++a){auto* dst=o+(bi+r+a)*64;
                            auto contribution=cf_dense_f16_to_f32(pf,svld1_f16(ph,pv.p+a*64));
                            auto scaled=svmul_n_f32_x(pf,svld1_f32(pf,dst),alpha.p[r+a]);
                            svst1_f32(pf,dst,svadd_f32_x(pf,scaled,contribution));
                        }
                        continue;
                    }
                    if(ncc_mode==9)asm volatile("dsb sy":::"memory");
                    if(kernel_variant) {
                        auto slot='''
assert s.count(marker)==1;s=s.replace(marker,new)
s=s.replace('if(argc!=17)','if(argc!=18)')
marker='    layout_mode=atoi(argv[14]);'
s=s.replace(marker,'''    ncc_mode=atoi(argv[17]);
    if(ncc_mode<0||ncc_mode>9||(!spm&&ncc_mode)||br!=16||bc!=256||g!=16||sets!=1024||accumulator_mode!=1||kernel_variant!=1||stage_simple||packed_producer)return 2;
    std::printf("NCC_CONFIG mode=%d\\n",ncc_mode);
'''+marker)
marker='''    if(c.id==0){m5_work_end(0,0);m5_dump_stats(0,0);std::printf("KV_ROI_END\\n");}'''
assert s.count(marker)==1;s=s.replace(marker,'''    if(ncc_mode)m5_sum(0x4e434350,9,0,0,0,ncc_mode);
'''+marker)
s=s.replace('static int ncc_mode=0;', 'static int ncc_mode=0;\nextern \"C\" uint64_t ncc_sum(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t) asm(\"m5_sum\");').replace('m5_sum(0x4e434350','ncc_sum(0x4e434350')
s=s.replace('0,ncc_mode);', '0,ncc_mode);asm volatile(\"dsb sy\":::\"memory\");')
s=s.replace('asm volatile(\"dsb sy\":::\"memory\");', 'asm volatile(\"dsb sy\\nisb\\n\":::\"memory\");')
(G/'driver.cpp').write_text(s);(G/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(G)))
subprocess.run(['bash',str(G/'build.sh')],cwd=R,check=True)
sys.path.insert(0,str(R/'benchmarks/cacheflex_flash_attention'));from export_safe_elf import patch
patch(G/'guest',G/'guest.export-safe')
(G/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in G.iterdir() if p.is_file()},indent=2))
old=R/'results/retained_protocol_restart_20261002/gem5/src/sim/pseudo_inst.cc'
s=old.read_text();s=s.replace('namespace gem5\n{','#include "hook.hh"\n\nnamespace gem5\n{',1)
# sum is existing serializing diagnostic operation; other operations untouched.
marker='m5sum(ThreadContext *tc, uint64_t a, uint64_t b, uint64_t c,\n                         uint64_t d, uint64_t e, uint64_t f)\n{'
assert marker in s;s=s.replace(marker,marker+'\n    if(a==0x4e434350)return nccProbe(tc,b,c,d,e,f);')
(B/'pseudo_inst.cc').write_text(s);shutil.copy2(H/'hook.hh',B/'hook.hh')
print('Guest built; isolated hook prepared')
