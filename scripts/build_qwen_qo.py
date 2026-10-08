"""FA-33 isolated Q+O guest; reuse frozen arithmetic and unchanged simulator."""
import hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
from export_safe_elf import patch
OLD=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/guest_v4'
BASE=ROOT/'results/cacheflex_flash_attention/qwen_qo_ring_20260930'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(text,name):
 a=text.index('static ',text.index(name)-100) if False else text.rfind('static ',0,text.index(name));b=text.index('\n}',text.index(name))+2;return text[a:b]
def kernels():
 q=function((OLD/'kernels.hpp').read_text(),'cf_wide_qk_cache(').replace('cf_wide_qk_cache','qo_qk')
 q=q.replace('"ptrue p1.d\\n"','"ptrue p1.d\\n"\n"mov x16, #8\\n"\n"whilelt p2.d, xzr, x16\\n"\n"index z31.b, #0, #1\\n"\n"and z31.b, z31.b, #15\\n"')
 for z,ptr in [(0,21),(1,23),(5,21),(6,23)]:
  old=f'"ld1rqh {{z{z}.h}}, p0/z, [x{ptr}]\\n"';assert q.count(old)==(2 if z in (0,1) else 1)
  q=q.replace(old,f'"and x16, x{ptr}, #0x7ffc0\\n"\n"and x17, x{ptr}, #48\\n"\n"dup z30.b, w17\\n"\n"add z30.b, z30.b, z31.b\\n"\n"spm.ld1qd z{z}.d, p2/z, [x16]\\n"\n"tbl z{z}.b, {{z{z}.b}}, z30.b\\n"')
 q=q.replace('"p0","p1",','"p0","p1","p2",').replace('"x16",','"x16","x17",')
 result=q+'\n'
 for arm in ('cache','spm'):
  f=function((OLD/'pv_direct.hpp').read_text(),f'cf_pv16_{arm}_cpu_paired(').replace(f'cf_pv16_{arm}_cpu_paired',f'qo_pv_{arm}')
  assert f.count('"ld1w {z5.s}, p3/z, [x9]\\n"')==16
  f=f.replace('"ptrue p3.s\\n"','"ptrue p3.s\\n"\n"ptrue p4.d\\n"')
  f=re.sub(r'"add x9, x10, #(\d+)\\n"',lambda m:'"add x9, x10, #'+str(int(m[1])//4)+'\\n"',f)
  f=f.replace('"ld1w {z5.s}, p3/z, [x9]\\n"','"spm.ld1qd z5.d, p4/z, [x9]\\n"').replace('"st1w {z5.s}, p3, [x9]\\n"','"spm.st1h z5.h, p0, [x9]\\n"').replace('"p3",','"p3","p4",')
  result+=f+'\n'
 return '#pragma once\n'+result
HELPERS=r'''
static int layout_mode=0;
static inline void qo_store(size_t row,svfloat32_t v) {
    size_t a=row*64;
    asm volatile("ptrue p0.h\nspm.st1h %0.h, p0, [%1]\n"::"w"(v),"r"(a):"p0","memory");
}
static inline svfloat32_t qo_load(size_t row) {
    svfloat32_t v;size_t a=row*64;
    asm volatile("ptrue p0.d\nspm.ld1qd %0.d, p0/z, [%1]\n":"=w"(v):"r"(a):"p0","memory");return v;
}
static void qo_stage(const __fp16* src,int rows) {
    for(size_t b=0;b<size_t(rows)*128;b+=64){size_t dst=(4UL<<16)+b;
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(dst),"r"(reinterpret_cast<const char*>(src)+b):"memory");}
}
static void qo_vstage(const __fp16* src,int rows) {
    for(int r=0;r<rows;++r)for(int w=0;w<2;++w){size_t dst=((512UL+r)<<6)|(size_t(w)<<16);
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(dst),"r"(src+r*64+w*32):"memory");}
}
static void qo_claim_o(float* zero,int rows) {
    std::fill(zero,zero+rows*64,0.f);
    for(int r=0;r<rows;++r)for(int w=0;w<4;++w){size_t dst=(size_t(w)<<16)+r*64;
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(dst),"r"(zero+r*64+w*16):"memory");}
    asm volatile("dsb sy":::"memory");
}
static void qo_release(int rows,int bc,bool resident,bool vstage) {
    asm volatile("dsb sy":::"memory");
    if(resident){
      for(int i=0;i<rows*2;++i){size_t a=(4UL<<16)+i*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
      for(int r=0;r<rows;++r)for(int w=0;w<4;++w){size_t a=(size_t(w)<<16)+r*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
    }
    if(vstage)for(int r=0;r<bc;++r)for(int w=0;w<2;++w){size_t a=(size_t(w)<<16)+(512+r)*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
    asm volatile("dsb sy":::"memory");
}
'''
def source():
 old=(OLD/'driver.cpp').read_text();s=old.replace('#include "pv_direct.hpp"','#include "pv_direct.hpp"\n#include "qo_kernels.hpp"\n'+HELPERS)
 s=s.replace('static void* worker(void* arg)','static void* legacy_worker(void* arg)')
 start=old.index('                // Skip wholly future');end=old.index('                PHASE(pv);',start)
 softmax=old[start:end].replace('qi+bi','gqi').replace('if(accumulator_mode) alpha.p[r]=a;\n                    else svst1_f32(pf,dst,svmul_n_f32_x(pf,svld1_f32(pf,dst),a));','alpha.p[r]=a;')
 worker=r'''
static void* qo_worker(void* arg) {
 auto& ctx=*static_cast<Context*>(arg);auto& s=*ctx.s;int id=ctx.id;
 int rows=s.t/8,half=rows/2;const bool resident=layout_mode>=4,vstage=layout_mode==3||layout_mode==5;
 Buffer<__fp16> qa(rows*64),scores(s.br*s.bc),prob(s.br*s.bc),pa(16*s.bc);
 Buffer<float> alpha(s.br);float* o=s.partial.p+id*rows*64;float* m=s.maximum.p+id*rows;float* l=s.denominator.p+id*rows;
 const auto pf=svptrue_b32();const auto ph=svwhilelt_b16_u64(0,64);
 pthread_barrier_wait(&s.barrier);
 if(id==0){std::printf("KV_ROI_BEGIN\n");m5_reset_stats(0,0);m5_work_begin(0,0);}
 pthread_barrier_wait(&s.barrier);
 for(int panel=id;panel<s.hkv*(s.t/128);panel+=8){int kh=panel/(s.t/128),j=panel%(s.t/128)*128;
   for(int d=0;d<64;++d)for(int x=0;x<128;++x)s.kpack.p[kh*s.t*64+(j/128*64+d)*128+x]=s.k.p[kh*s.t*64+(j+x)*64+d];}
 pthread_barrier_wait(&s.barrier);
 size_t updates=0,copies=0;
 for(int head=0;head<s.hq;++head){
  const int kh=head/(s.hq/s.hkv);const auto* qhead=s.q.p+head*s.t*64;const auto* vhead=s.v.p+kh*s.t*64;const auto* kpacked=s.kpack.p+kh*s.t*64;auto* outhead=s.out.p+head*s.t*64;
  auto global_row=[&](int local){return local<half?id*half+local:(15-id)*half+local-half;};
  cf_pack_q64(qhead+id*half*64,qa.p,half);cf_pack_q64(qhead+(15-id)*half*64,qa.p+half*64,half);
  std::fill(m,m+rows,-1e30f);std::fill(l,l+rows,0.f);
  if(resident){qo_stage(qa.p,rows);copies+=6*rows;qo_claim_o(o,rows);}
  else std::fill(o,o+rows*64,0.f);
  int end=s.causal?std::min(s.t,((global_row(rows-1)+1+s.bc-1)/s.bc)*s.bc):s.t;
  for(int j=0;j<end;j+=s.bc){int bc=s.bc;
    if(vstage){qo_vstage(vhead+j*64,bc);copies+=2*bc;}
    for(int bi=0;bi<rows;bi+=s.br){int br=s.br,gqi=global_row(bi);
      if(s.causal&&j>=gqi+br)continue;
      updates+=br;
      for(int r=0;r<br;r+=16)for(int nb=0;nb<bc;nb+=128){
        const auto* qptr=resident?reinterpret_cast<const __fp16*>((4UL<<16)+(bi+r)*128):qa.p+(bi+r)*64;
        if(resident)qo_qk(qptr,kpacked+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0);
        else cf_wide_qk_cache(qptr,kpacked+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0);
      }
SOFTMAX
      for(int r=0;r<br;r+=16){cf_vector_pack_A(prob.p,s.bc,pa.p,br,r,bc,0,16,bc);
        size_t vp=vstage?512UL:reinterpret_cast<size_t>(vhead+j*64);
        float* dst=resident?reinterpret_cast<float*>(size_t(bi+r)*64):o+(bi+r)*64;
        if(resident){if(vstage)qo_pv_spm(pa.p,vp,bc,alpha.p+r,0,dst);else qo_pv_cache(pa.p,vp,bc,alpha.p+r,0,dst);}
        else {if(vstage)cf_pv16_spm_cpu_paired(pa.p,vp,bc,alpha.p+r,0,dst);else cf_pv16_cache_cpu_paired(pa.p,vp,bc,alpha.p+r,0,dst);}
      }
    }
  }
  for(int r=0;r<rows;++r){auto value=resident?qo_load(r):svld1_f32(pf,o+r*64);
    svst1_f16(ph,outhead+global_row(r)*64,cf_dense_f32_to_f16(pf,svmul_n_f32_x(pf,value,1.f/l[r])));}
  qo_release(rows,s.bc,resident,vstage);
  pthread_barrier_wait(&s.barrier);
 }
 ctx.tasks_done=s.hq;ctx.updates_done=updates;
 pthread_barrier_wait(&s.barrier);
 if(id==0){m5_work_end(0,0);m5_dump_stats(0,0);std::printf("KV_ROI_END\n");}
 std::printf("QO_WORK core=%d copies=%zu\n",id,copies);
 return nullptr;
}
'''.replace('SOFTMAX',softmax)
 idx=s.index('int main(');s=s[:idx]+worker+'\n'+s[idx:];s=s.replace('if(argc!=14)','if(argc!=15)')
 s=s.replace('    Shared s(t,n,br,bc,g,sets,kv,spm,hq,hkv,causal);','''    layout_mode=atoi(argv[14]);
    if(n!=8||layout_mode<0||layout_mode>5||accumulator_mode!=1||kernel_variant!=1||br!=16||bc>512||
       (layout_mode>=2&&(t>4096||t%256||br*g!=t/8)))return 2;
    Shared s(t,n,br,bc,g,sets,kv,spm,hq,hkv,causal);
    auto worker=layout_mode>=2?qo_worker:legacy_worker;
    std::printf("QO_CONFIG mode=%d rows_per_core=%d\\n",layout_mode,t/8);''')
 return s

def main():
 out=BASE/os.environ.get('QO_GUEST','guest_v2');out.mkdir(parents=True)
 for p in OLD.glob('*.hpp'):shutil.copy2(p,out/p.name)
 (out/'qo_kernels.hpp').write_text(kernels());(out/'driver.cpp').write_text(source())
 (out/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(out)))
 with (out/'build.log').open('w') as f:subprocess.run(['bash',str(out/'build.sh')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 patch(out/'guest',out/'guest.export-safe')
 (out/'frozen.json').write_text(json.dumps({str(p):sha(p) for p in out.iterdir() if p.is_file()},indent=2));print(out,sha(out/'guest.export-safe'))
if __name__=='__main__':main()
