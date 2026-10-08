"""Build an isolated Qwen-shaped causal GQA guest from the accepted L3 kernel."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'benchmarks/cacheflex_flash_attention')]
from build_access_timing import sha, save
from export_safe_elf import patch

OLD = ROOT/'results/cacheflex_l3_spm/kernel_tuning_20260928/guest_v2'
BASE = ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928'


def source():
    s = (OLD/'driver.cpp').read_text()
    # Reserve read-only layout padding before the ELF is linked. The existing
    # post-ROI exporter still enforces unchanged mapped pages and kernel bytes.
    s += '\n__attribute__((used)) static const unsigned char export_layout_pad[512]={0};\n'
    def replace(a, b):
        nonlocal s
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    replace('int t,cores,br,bc,group,sets;bool kv,spm;',
            'int t,cores,br,bc,group,sets,hq,hkv;bool kv,spm,causal;')
    replace('int G,int S,bool KV,bool SPM):', 'int G,int S,bool KV,bool SPM,int HQ,int HKV,bool CAUSAL):')
    replace('sets(S),kv(KV),spm(SPM),', 'sets(S),hq(HQ),hkv(HKV),kv(KV),spm(SPM),causal(CAUSAL),')
    replace('q(T*64),k(T*64),v(T*64),out(T*64),kpack(T*64),',
            'q(HQ*T*64),k(HKV*T*64),v(HKV*T*64),out(HQ*T*64),kpack(HKV*T*64),')
    begin=s.index('    // Disjoint K panels packed')
    end=s.index('        PHASE(pack_q);',begin)
    s=s[:begin]+'''    // Pack each physical KV head once, cooperatively inside the ROI.
    for(int panel=c.id;panel<s.hkv*(s.t/128);panel+=s.cores) {
        int kh=panel/(s.t/128),j=(panel%(s.t/128))*128;
        for(int d=0;d<64;++d) for(int x=0;x<128;++x)
            s.kpack.p[kh*s.t*64+(j/128*64+d)*128+x]=s.k.p[kh*s.t*64+(j+x)*64+d];
    }
    pthread_barrier_wait(&s.barrier);
    const int groups=(s.t+rows-1)/rows;
    const int kbegin=0;
    size_t tasks_done=0,updates_done=0;
    // Rotate query groups between heads before round-robin assignment, so the
    // same core does not repeatedly receive the long causal-prefix groups.
    // Each worker exclusively owns one LLC bank's output slots and reuses them
    // only after normalization, drain and release of the previous task.
    for(int task=c.id;task<s.hq*groups;task+=s.cores) {
        const int head=task/groups,kh=head/(s.hq/s.hkv);
        const int qi=((task%groups+head)%groups)*rows,qend=s.t;
        const int nr=std::min(rows,qend-qi);
        const int kend=s.causal?std::min(s.t,((qi+nr+s.bc-1)/s.bc)*s.bc):s.t;
        const auto* qhead=s.q.p+head*s.t*64;
        const auto* vhead=s.v.p+kh*s.t*64;
        const auto* kpacked=s.kpack.p+kh*s.t*64;
        auto* outhead=s.out.p+head*s.t*64;
        ++tasks_done;
        updates_done+=nr*((kend+s.bc-1)/s.bc);
'''+s[end:]
    replace('cf_pack_q64(s.q.p+qi*64,qa.p,nr);','cf_pack_q64(qhead+qi*64,qa.p,nr);')
    s=s.replace('s.v.p+j*64','vhead+j*64')
    replace('s.kpack.p+(j+nb)/128*64*128','kpacked+(j+nb)/128*64*128')
    s=s.replace('s.out.p+(qi+r)*64','outhead+(qi+r)*64')
    replace('                PHASE(softmax);', '''                // Skip wholly future KV blocks above, and zero future columns
                // within the boundary block. This mask is applied before max/exp.
                if(s.causal && j+bc>qi+bi+1) {
                    for(int r=0;r<br;++r) {
                        auto* sr=scores.p+r*s.bc;
                        for(int x=0;x<bc;x+=64) {
                            auto index=svindex_u16(j+x,1);
                            auto visible=svcmple_n_u16(ph,index,qi+bi+r);
                            auto value=svld1_f16(ph,sr+x);
                            svst1_f16(ph,sr+x,svsel_f16(visible,value,svdup_n_f16(-INFINITY)));
                        }
                    }
                }
                PHASE(softmax);''')
    # exp approximation is not assumed to handle -infinity; use explicit zero
    # weights on masked lanes after exponentiating a finite harmless argument.
    replace('                        auto z=sve_exp_f32(pf,svmla_n_f32_x(pf,svdup_n_f32(-mn),cf_dense_f16_to_f32(pf,svld1_f16(ph,sr+x)),.125f));', '''                        auto score=cf_dense_f16_to_f32(pf,svld1_f16(ph,sr+x));
                        auto visible=svcmple_n_u32(pf,svindex_u32(j+x,1),qi+bi+r);
                        auto arg=svmla_n_f32_x(pf,svdup_n_f32(-mn),score,.125f);
                        if(s.causal)arg=svsel_f32(visible,arg,svdup_n_f32(0));
                        auto z=sve_exp_f32(pf,arg);
                        if(s.causal)z=svsel_f32(visible,z,svdup_n_f32(0));''')
    replace('struct Context{Shared* s;int id;};',
            'struct Context{Shared* s;int id;size_t tasks_done=0,updates_done=0;};')
    replace('    PHASE(final_wait);pthread_barrier_wait(&s.barrier);', '''    static_cast<Context*>(arg)->tasks_done=tasks_done;
    static_cast<Context*>(arg)->updates_done=updates_done;
    PHASE(final_wait);pthread_barrier_wait(&s.barrier);''')
    replace('    uint64_t h=1469598103934665603ULL;size_t bad=0;', '''    for(int i=0;i<n;++i)std::printf("GQA_WORK core=%d tasks=%zu row_updates=%zu\\n",i,c[i].tasks_done,c[i].updates_done);
    uint64_t h=1469598103934665603ULL;size_t bad=0;''')
    replace('if(argc!=11)', 'if(argc!=14)')
    replace('accumulator_mode kernel_variant\\n', 'accumulator_mode kernel_variant hq hkv causal\\n')
    replace('    if((n!=1&&n!=4&&n!=8)||t<=0||t%n||t/n%128||',
            '    if((n!=1&&n!=4&&n!=8)||t<=0||t>16384||t%128||')
    replace('    Shared s(t,n,br,bc,g,sets,kv,spm);', '''    int hq=atoi(argv[11]),hkv=atoi(argv[12]),causal=atoi(argv[13]);
    if(hq<1||hq>32||hkv<1||hq%hkv||(causal!=0&&causal!=1)||
       br*g>512||t%(br*g)||t%bc||!kernel_variant||
       (accumulator_mode!=1&&accumulator_mode!=5))return 2;
    Shared s(t,n,br,bc,g,sets,kv,spm,hq,hkv,causal);''')
    replace('    for(size_t i=0;i<(size_t)t*64;++i){s.q.p[i]=pattern(i,1);s.k.p[i]=pattern(i,2);s.v.p[i]=pattern(i,3);}', '''    for(size_t i=0;i<s.q.n;++i)s.q.p[i]=pattern(i,1);
    for(size_t i=0;i<s.k.n;++i){s.k.p[i]=pattern(i,2);s.v.p[i]=pattern(i,3);}
    std::printf("GQA_CONFIG hq=%d hkv=%d d=64 batch=1 causal=%d tasks=%d\\n",hq,hkv,causal,hq*(t/(br*g)));''')
    replace('i<(size_t)t*64;++i){uint16_t bits;', 'i<s.out.n;++i){uint16_t bits;')
    replace('bad,t*64);','bad,hq*t*64);')
    replace('m5_write_file(s.out.p,t*128,0,"output.fp16")!=(uint64_t)t*128',
            'm5_write_file(s.out.p,s.out.n*2,0,"output.fp16")!=s.out.n*2')
    return s


def main():
    out=BASE/os.environ.get('CACHEFLEX_QWEN_GUEST','guest_v4')
    out.mkdir(parents=True)
    for p in OLD.glob('*.hpp'):shutil.copy2(p,out/p.name)
    (out/'driver.cpp').write_text(source())
    command=(OLD/'build.sh').read_text().replace(str(OLD),str(out))
    (out/'build.sh').write_text(command)
    with (out/'build.log').open('w') as log:
        subprocess.run(['bash',str(out/'build.sh')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    save(out/'source_reference.json',dict(source=str(OLD),ledger=['FA-06','FA-07','FA-08','FA-09','L3-05'],
         new='14Q/2KV causal GQA, deterministic head/query-group tasks; unchanged PV assembly'))
    # These are the previously runtime-validated compute/epilogue sources.
    save(out/'kernel_audit.json',{n:dict(sha256=sha(out/n),exact_prior=sha(out/n)==sha(OLD/n))
         for n in ('pv_direct.hpp','pv_fullwidth.hpp','kernels.hpp','fast_qpack.hpp')})
    save(out/'frozen.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(out,sha(out/'guest.export-safe'),flush=True)


if __name__=='__main__':main()
