"""FA-51 guest v2 (geometry-general FA-50 flags): isolated single-ELF softmax/PV software optimizations (runtime flag word).

Derived from the accepted FA-39 guest (bottleneck_20261004/guest_v1, itself from
matrix guest_v3). Only the legacy worker changes; with sm_flags=0 its original
code runs verbatim. Flags (bitmask, argv[17]):
  1  softmax processes four rows interleaved (per-row operation order unchanged)
  2  PV epilogue without alpha multiply when every row alpha==1, or first KV block
  4  row max from an fmaxv QK epilogue (non-boundary blocks), skipping pass 1
  8  causal mask folded into pass 1 (boundary blocks); separate mask pass removed
  16 Q pre-scaled by 0.125 after packing; softmax scale becomes 1 (exact barring
     FP16 subnormals; validated, not assumed)
"""
import hashlib, json, os, re, resource, shutil, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[4]; H = Path(__file__).resolve().parent
SRC = R / 'results/cacheflex_flash_attention/bottleneck_20261004/guest_v1'
BASE = R / 'results/cacheflex_flash_attention/softmax_tune_20261005'; G = BASE / 'guest_v1'
sys.path.insert(0, str(R / 'benchmarks/cacheflex_flash_attention'))
from export_safe_elf import patch


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old)); return s.replace(old, new)


def function(text, start, stop):
    i = text.index(start); j = text.index(stop, i + len(start)); return text[i:j]


def qk_max_kernel(k):
    f = function(k, 'static void cf_wide_qk_cache(', 'static void cf_wide_qk_spm(')
    f = one(f, 'static void cf_wide_qk_cache(const __fp16* a,const __fp16* b,__fp16* c,int k,int ld,size_t base) {',
            'static void cf_wide_qk_cache_max(const __fp16* a,const __fp16* b,__fp16* c,int k,int ld,size_t base,__fp16* rmax) {')
    regs = list(range(8, 24))
    epilogue = '"mov x6, %x[Mp]\\n"\n' + ''.join(f'"fmaxv h24, p0, z{z}.h\\n"\n"str h24, [x6, #{2 * i}]\\n"\n' for i, z in enumerate(regs))
    anchor = ': : [Ap] "r"(a),[Bp] "r"(b),[Cp] "r"(c),[K] "r"((size_t)k),[ldc] "r"((size_t)ld),[base] "r"(base)'
    f = one(f, anchor, epilogue + anchor.replace('[base] "r"(base)', '[base] "r"(base),[Mp] "r"(rmax)'))
    f = one(f, ': "cc","memory","p0","p1","x5",', ': "cc","memory","p0","p1","x5","x6",')
    return f


def noalpha(pv, name):
    f = function(pv, f'static __attribute__((noinline,aligned(64))) void {name}(', 'static __attribute__((noinline,aligned(64)))')
    f = f.replace(f'void {name}(', f'void {name}_noalpha(', 1)
    loads = re.findall(r'"ld1rw \{z4\.s\}, p3/z, \[x7, #\d+\]\\n"\n', f)
    muls = re.findall(r'"fmul z5\.s, p3/m, z5\.s, z4\.s\\n"\n', f)
    assert len(loads) == len(muls) == 16, (name, len(loads), len(muls))
    for x in loads + muls: f = f.replace(x, '', 1)
    assert 'z4.s' not in f.split('asm volatile(')[1].split(': :')[0]
    return f


def transform(driver):
    head, rest = driver.split('static void* legacy_worker(void* arg) {', 1)
    legacy, tail = rest.split('static void* qo_worker(void* arg) {', 1)
    head = one(head, 'static int stage_simple=0, packed_producer=0;', 'static int stage_simple=0, packed_producer=0;\nstatic int sm_flags=0;')
    L = legacy
    L = one(L, '    Buffer<float> alpha(s.br);\n', '    Buffer<float> alpha(s.br);\n    Buffer<__fp16> rmbuf(128);\n')
    L = one(L, '        cf_pack_q64(qhead+qi*64,qa.p,nr);\n',
            '        cf_pack_q64(qhead+qi*64,qa.p,nr);\n'
            '        if(sm_flags&16) { const auto hp=svptrue_b16();\n'
            '            for(int i=0;i<nr*64;i+=(int)svcnth()) svst1_f16(hp,qa.p+i,svmul_n_f16_x(hp,svld1_f16(hp,qa.p+i),(__fp16)0.125f)); }\n')
    L = one(L, '                const int br=std::min(s.br,nr-bi);\n',
            '                const int br=std::min(s.br,nr-bi);\n                bool unit_alpha=false;\n'
            '                const bool boundary=s.causal && j+bc>qi+bi+1;\n')
    qk = ('                for(int r=0;r<br;r+=16) for(int nb=0;nb<bc;nb+=128)\n'
          '                    cf_wide_qk_cache(qa.p+(bi+r)*64,\n'
          '                        kpacked+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0);\n')
    L = one(L, qk, '                if((sm_flags&4) && !boundary) {\n'
                   '                for(int r=0;r<br;r+=16) for(int nb=0;nb<bc;nb+=128)\n'
                   '                    cf_wide_qk_cache_max(qa.p+(bi+r)*64,\n'
                   '                        kpacked+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0,rmbuf.p+(nb/128)*16);\n'
                   '                } else {\n' + qk + '                }\n')
    L = one(L, '                if(s.causal && j+bc>qi+bi+1) {\n', '                if(s.causal && j+bc>qi+bi+1 && !(sm_flags&8)) {\n')
    i = L.index('                PHASE(softmax);\n'); k = L.index('                PHASE(pv);\n', i)
    original = L[i + len('                PHASE(softmax);\n'):k]
    assert 'packed_producer' in original and original.count('for(int r=0;r<br;++r) {') == 1
    new = r'''                PHASE(softmax);
                if(sm_flags&(1|4|8|16)) {
                    const float qs=(sm_flags&16)?1.0f:.125f;
                    const bool fold=(sm_flags&8) && boundary;
                    const bool qkmax=(sm_flags&4) && !boundary;
                    const int step=(sm_flags&1)?4:1;
#define FA50_P1(u) { auto v=cf_dense_f16_to_f32(pf,svld1_f16(ph,scores.p+(r0+u)*s.bc+x)); \
                    if(fold){auto vis=svcmple_n_u32(pf,svindex_u32(j+x,1),qi+bi+r0+u);v=svsel_f32(vis,v,svdup_n_f32(-INFINITY));} \
                    mx##u=svmax_f32_x(pf,mx##u,(sm_flags&16)?v:svmul_n_f32_x(pf,v,.125f)); }
#define FA50_MN(u) float mn##u; \
                    if(qkmax){float hm=-INFINITY;for(int q=0;q<bc/128;++q)hm=std::max(hm,(float)rmbuf.p[q*16+r0+u]); \
                        mn##u=std::max(m[bi+r0+u],hm*qs);} \
                    else mn##u=std::max(m[bi+r0+u],svmaxv_f32(pf,mx##u)); \
                    float a##u=expf(m[bi+r0+u]-mn##u);auto sum##u=svdup_n_f32(0);
#define FA50_P2(u) { auto score=cf_dense_f16_to_f32(pf,svld1_f16(ph,scores.p+(r0+u)*s.bc+x)); \
                    auto visible=svcmple_n_u32(pf,svindex_u32(j+x,1),qi+bi+r0+u); \
                    auto arg=svmla_n_f32_x(pf,svdup_n_f32(-mn##u),score,qs); \
                    if(s.causal)arg=svsel_f32(visible,arg,svdup_n_f32(0)); \
                    auto z=sve_exp_f32(pf,arg); \
                    if(s.causal)z=svsel_f32(visible,z,svdup_n_f32(0)); \
                    sum##u=svadd_f32_x(pf,sum##u,z); \
                    svst1_f16(ph,prob.p+(r0+u)*s.bc+x,cf_dense_f32_to_f16(pf,z)); }
#define FA50_FIN(u) { m[bi+r0+u]=mn##u;l[bi+r0+u]=a##u*l[bi+r0+u]+svaddv_f32(pf,sum##u); \
                    if(accumulator_mode) alpha.p[r0+u]=a##u; \
                    else {auto* dst=o+(bi+r0+u)*64;svst1_f32(pf,dst,svmul_n_f32_x(pf,svld1_f32(pf,dst),a##u));} }
                    for(int r0=0;r0<br;r0+=step) {
                        if(step==4) {
                            auto mx0=svdup_n_f32(-1e30f),mx1=mx0,mx2=mx0,mx3=mx0;
                            if(!qkmax) for(int x=0;x<bc;x+=64){FA50_P1(0) FA50_P1(1) FA50_P1(2) FA50_P1(3)}
                            FA50_MN(0) FA50_MN(1) FA50_MN(2) FA50_MN(3)
                            for(int x=0;x<bc;x+=64){FA50_P2(0) FA50_P2(1) FA50_P2(2) FA50_P2(3)}
                            FA50_FIN(0) FA50_FIN(1) FA50_FIN(2) FA50_FIN(3)
                        } else {
                            auto mx0=svdup_n_f32(-1e30f);
                            if(!qkmax) for(int x=0;x<bc;x+=64){FA50_P1(0)}
                            FA50_MN(0)
                            for(int x=0;x<bc;x+=64){FA50_P2(0)}
                            FA50_FIN(0)
                        }
                    }
#undef FA50_P1
#undef FA50_MN
#undef FA50_P2
#undef FA50_FIN
                } else {''' + '\n' + original + '''                }
                if(sm_flags&2) {
                    unit_alpha=(j==kbegin);
                    if(!unit_alpha){unit_alpha=true;for(int q=0;q<br;++q) if(alpha.p[q]!=1.0f){unit_alpha=false;break;}}
                }
'''
    L = L[:i] + new + L[k:]
    for arm in ('spm', 'cache'):
        call = f'if(kernel_variant==1)cf_pv16_{arm}_cpu_paired(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);'
        L = one(L, call, f'if(kernel_variant==1&&unit_alpha)cf_pv16_{arm}_cpu_paired_noalpha(pa.p,vptr,bc,alpha.p+r,0,o+(bi+r)*64);\n'
                         f'                            else ' + call)
    tail = one(tail, 'if(argc!=17)', 'if(argc!=18)')
    tail = one(tail, '    if((stage_simple!=0&&stage_simple!=1)||(packed_producer!=0&&packed_producer!=1))return 2;\n',
               '    if((stage_simple!=0&&stage_simple!=1)||(packed_producer!=0&&packed_producer!=1))return 2;\n'
               '    sm_flags=atoi(argv[17]);\n    if(sm_flags<0||sm_flags>31||(sm_flags&&packed_producer))return 2;\n'
               '    std::printf("SOFTMAX_OPT flags=%d\\n",sm_flags);\n')
    tail = one(tail, '    if(layout_mode>1||br!=16||bc!=256||g!=16||sets!=1024)return 2;\n', '    if(layout_mode>1||sets!=1024||((sm_flags&4)&&br!=16)||(sm_flags&1)&&br%4)return 2;\n')
    return head + 'static void* legacy_worker(void* arg) {' + L + 'static void* qo_worker(void* arg) {' + tail


def main():
    G.mkdir(parents=True, exist_ok=False)
    for p in SRC.glob('*.hpp'): shutil.copy2(p, G / p.name)
    k = (G / 'kernels.hpp').read_text()
    k = one(k, 'static void cf_wide_qk_spm(', qk_max_kernel(k) + 'static void cf_wide_qk_spm(')
    (G / 'kernels.hpp').write_text(k)
    pv = (G / 'pv_direct.hpp').read_text()
    extra = noalpha(pv, 'cf_pv16_spm_cpu_paired') + noalpha(pv, 'cf_pv16_cache_cpu_paired')
    (G / 'pv_direct.hpp').write_text(pv + '\n' + extra)
    (G / 'driver.cpp').write_text(transform((SRC / 'driver.cpp').read_text()))
    (G / 'build.sh').write_text((SRC / 'build.sh').read_text().replace(str(SRC), str(G)))
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (3 * 2**30, 3 * 2**30)); os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    subprocess.run(['bash', str(G / 'build.sh')], cwd=R, check=True, preexec_fn=limits)
    patch(G / 'guest', G / 'guest.export-safe')
    (G / 'disassembly.txt').write_text(subprocess.check_output(['aarch64-linux-gnu-objdump', '-d', '-C', str(G / 'guest.export-safe')], text=True))
    frozen = {str(p): sha(p) for p in [*G.iterdir(), *SRC.glob('*.hpp'), SRC / 'driver.cpp', H / 'build.py'] if p.is_file()}
    (G / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('built', G / 'guest.export-safe')


if __name__ == '__main__':
    main()
