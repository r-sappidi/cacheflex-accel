"""FA-35 guest: placement decomposition, Br/Bc/R, and static KV software merge."""
import os,shutil,subprocess,json,hashlib
from pathlib import Path
from build_qwen_qo import source as prior_source, kernels, OLD, function
from export_safe_elf import patch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/qwen_matrix_20260930'


def source():
    s=prior_source()
    old=function(s,'qo_release(')
    s=s.replace(old,r'''static void qo_release(int rows,int bc,bool q,bool o,bool v) {
      asm volatile("dsb sy":::"memory");
      if(q)for(int i=0;i<rows*2;++i){size_t a=(4UL<<16)+i*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
      if(o)for(int r=0;r<rows;++r)for(int w=0;w<4;++w){size_t a=(size_t(w)<<16)+r*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
      if(v)for(int r=0;r<bc;++r)for(int w=0;w<2;++w){size_t a=(size_t(w)<<16)+(512+r)*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
      asm volatile("dsb sy":::"memory");
    }''')
    s=s.replace('int rows=s.t/8,half=rows/2;const bool resident=layout_mode>=4,vstage=layout_mode==3||layout_mode==5;',
        'int rows=s.br*s.group,half=rows/2,slices=s.t/8/rows;const bool qresident=layout_mode==4||layout_mode==5||layout_mode==6,oresident=layout_mode==4||layout_mode==5||layout_mode==7,vstage=layout_mode==3||layout_mode==5||layout_mode==6||layout_mode==7;')
    s=s.replace('for(int head=0;head<s.hq;++head){','for(int head=0;head<s.hq;++head)for(int slice=0;slice<slices;++slice){')
    s=s.replace('local<half?id*half+local:(15-id)*half+local-half','local<half?id*(s.t/16)+slice*half+local:(15-id)*(s.t/16)+slice*half+local-half')
    s=s.replace('qhead+id*half*64','qhead+global_row(0)*64').replace('qhead+(15-id)*half*64','qhead+global_row(half)*64')
    s=s.replace('if(resident){qo_stage(qa.p,rows);copies+=6*rows;qo_claim_o(o,rows);}',
                'if(qresident){qo_stage(qa.p,rows);copies+=2*rows;} if(oresident){qo_claim_o(o,rows);copies+=4*rows;}')
    s=s.replace('qptr=resident?','qptr=qresident?').replace('if(resident)qo_qk','if(qresident)qo_qk')
    s=s.replace('dst=resident?','dst=oresident?').replace('if(resident){if(vstage)','if(oresident){if(vstage)')
    s=s.replace('value=resident?','value=oresident?').replace('qo_release(rows,s.bc,resident,vstage);','qo_release(rows,s.bc,qresident,oresident,vstage);')
    s=s.replace('ctx.tasks_done=s.hq;','ctx.tasks_done=s.hq*slices;')
    start=s.index('    if(n!=8||layout_mode<0||layout_mode>5')
    end=s.index('    Shared s(',start)
    s=s[:start]+'''    if(n!=8||layout_mode<0||layout_mode>11||kernel_variant!=1||
       (br!=16&&br!=32&&br!=64)||(bc!=128&&bc!=256&&bc!=512)||t>4096||
       (layout_mode==8?accumulator_mode!=5:accumulator_mode!=1))return 2;
    if(layout_mode>=2&&layout_mode<=7&&(t%256||br*g>t/8||(br*g)%(2*br)||t/8%(br*g)))return 2;
    if(layout_mode>=9&&(t%2048||bc>t/16||(t/16)%bc))return 2;
'''+s[end:]
    s=s.replace('auto worker=layout_mode>=2?qo_worker:legacy_worker;',
                'auto worker=layout_mode>=9?kv_worker:((layout_mode>=2&&layout_mode<=7)?qo_worker:legacy_worker);')
    # Frozen mathematical softmax body, with no changed FP16 FMA sequence.
    original=(OLD/'driver.cpp').read_text()
    soft=original[original.index('                // Skip wholly future'):original.index('                PHASE(pv);')]
    kv=KV_WORKER.replace('SOFTMAX_BODY',soft)
    s=s.replace('int main(int argc,char** argv)',kv+'\nint main(int argc,char** argv)')
    assert 'resident?' not in s.replace('qresident?','').replace('oresident?','')
    return s


KV_WORKER=r'''
static void* kv_worker(void* arg) {
 auto& ctx=*static_cast<Context*>(arg);auto& s=*ctx.s;int id=ctx.id;
 int rows=s.br*s.group,local=s.t/8,half=s.t/16;
 bool resident=layout_mode>=10,both=layout_mode==11;
 Buffer<__fp16> qa(rows*64),scores(s.br*s.bc),prob(s.br*s.bc),pa(16*s.bc);
 Buffer<float> alpha(s.br);float* o=s.partial.p+id*rows*64;
 float* m=s.maximum.p+id*rows;float* l=s.denominator.p+id*rows;
 const auto pf=svptrue_b32();const auto ph=svwhilelt_b16_u64(0,64);
 auto keyrow=[&](int n){return n<half?id*half+n:(15-id)*half+n-half;};
 size_t tasks=0,updates=0,copies=0;
 pthread_barrier_wait(&s.barrier);
 if(id==0){std::printf("KV_ROI_BEGIN\n");m5_reset_stats(0,0);m5_work_begin(0,0);}
 pthread_barrier_wait(&s.barrier);
 for(int panel=id;panel<s.hkv*(s.t/128);panel+=8){int kh=panel/(s.t/128),j=panel%(s.t/128)*128;
   for(int d=0;d<64;++d)for(int x=0;x<128;++x)s.kpack.p[kh*s.t*64+(j/128*64+d)*128+x]=s.k.p[kh*s.t*64+(j+x)*64+d];}
 pthread_barrier_wait(&s.barrier);
 auto install=[&](int kh,int bank){
   for(int n=0;n<local;n+=128){int j=keyrow(n);
     for(int d=0;d<64;++d)for(int w=0;w<4;++w){
       size_t dst=(size_t(w)<<16)+((bank*(local/128)+n/128)*64+d)*64;
       auto* from=s.kpack.p+kh*s.t*64+(j/128*64+d)*128+w*32;
       asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(dst),"r"(from):"memory");++copies;}}
   for(int n=0;n<local;++n)for(int w=0;w<2;++w){
     size_t dst=(size_t(4+w)<<16)+(bank*local+n)*64;
     auto* from=s.v.p+kh*s.t*64+keyrow(n)*64+w*32;
     asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(dst),"r"(from):"memory");++copies;}
   asm volatile("dsb sy":::"memory");
 };
 auto relinquish=[&](int count){
   asm volatile("dsb sy":::"memory");
   for(int w=0;w<4;++w)for(int n=0;n<count*(local/128)*64;++n){size_t a=(size_t(w)<<16)+n*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
   for(int w=4;w<6;++w)for(int n=0;n<count*local;++n){size_t a=(size_t(w)<<16)+n*64;asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");}
   asm volatile("dsb sy":::"memory");
 };
 if(both)for(int kh=0;kh<s.hkv;++kh)install(kh,kh);
 for(int head=0;head<s.hq;++head){int kh=head/(s.hq/s.hkv),bank=both?kh:0;
   auto* qhead=s.q.p+head*s.t*64;auto* vhead=s.v.p+kh*s.t*64;
   auto* kpacked=s.kpack.p+kh*s.t*64;auto* outhead=s.out.p+head*s.t*64;
   if(resident&&!both&&head%(s.hq/s.hkv)==0)install(kh,0);
   for(int qi=0;qi<s.t;qi+=rows){++tasks;
     cf_pack_q64(qhead+qi*64,qa.p,rows);
     std::fill(o,o+rows*64,0.f);std::fill(m,m+rows,-1e30f);std::fill(l,l+rows,0.f);
     for(int n=0;n<local;n+=s.bc){int j=keyrow(n),bc=s.bc;
       if(s.causal&&j>=qi+rows)continue;
       for(int bi=0;bi<rows;bi+=s.br){int br=s.br;
         if(s.causal&&j>=qi+bi+br)continue;
         updates+=br;
         for(int r=0;r<br;r+=16)for(int nb=0;nb<bc;nb+=128){
           if(resident)cf_wide_qk_spm(qa.p+(bi+r)*64,nullptr,scores.p+r*s.bc+nb,64,s.bc,(bank*(local/128)+(n+nb)/128)*64);
           else cf_wide_qk_cache(qa.p+(bi+r)*64,kpacked+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0);}
SOFTMAX_BODY
         for(int r=0;r<br;r+=16){
           cf_vector_pack_A(prob.p,s.bc,pa.p,br,r,bc,0,16,bc);
           if(resident)cf_pv16_spm_cpu_static(pa.p,(4UL<<10)+bank*local+n,bc,alpha.p+r,0,o+(bi+r)*64);
           else cf_pv16_cache_cpu_paired(pa.p,reinterpret_cast<size_t>(vhead+j*64),bc,alpha.p+r,0,o+(bi+r)*64);
         }
       }
     }
     pthread_barrier_wait(&s.barrier);
     for(int r=id;r<rows;r+=8){float mx=-1e30f;
       for(int p=0;p<8;++p)if(s.denominator.p[p*rows+r]>0)mx=std::max(mx,s.maximum.p[p*rows+r]);
       float den=0;auto value=svdup_n_f32(0);
       for(int p=0;p<8;++p){float lp=s.denominator.p[p*rows+r];if(lp==0)continue;
         float a=expf(s.maximum.p[p*rows+r]-mx);den+=a*lp;
         value=svadd_f32_x(pf,value,svmul_n_f32_x(pf,svld1_f32(pf,s.partial.p+(p*rows+r)*64),a));}
       svst1_f16(ph,outhead+(qi+r)*64,cf_dense_f32_to_f16(pf,svmul_n_f32_x(pf,value,1.f/den)));
     }
     pthread_barrier_wait(&s.barrier);
   }
   if(resident&&!both&&(head+1)%(s.hq/s.hkv)==0)relinquish(1);
 }
 if(both)relinquish(s.hkv);
 ctx.tasks_done=tasks;ctx.updates_done=updates;
 pthread_barrier_wait(&s.barrier);
 if(id==0){m5_work_end(0,0);m5_dump_stats(0,0);std::printf("KV_ROI_END\n");}
 std::printf("QO_WORK core=%d copies=%zu\n",id,copies);return nullptr;
}
'''


def main():
    out=BASE/os.environ.get('MATRIX_GUEST','guest_v1');out.mkdir(parents=True,exist_ok=False)
    for p in OLD.glob('*.hpp'):shutil.copy2(p,out/p.name)
    pv=(out/'pv_direct.hpp').read_text()
    bounded=function(pv,'cf_pv16_spm_cpu_paired(').replace('cf_pv16_spm_cpu_paired','cf_pv16_spm_cpu_static')
    bounded=bounded.replace('"add x22, x22, #1\\n"','"add x22, x22, #1\\n"\n"bfi x22, xzr, #10, #2\\n"')
    assert 'bfi x22' in bounded
    (out/'pv_direct.hpp').write_text(pv+'\n'+bounded+'\n')
    (out/'qo_kernels.hpp').write_text(kernels());(out/'driver.cpp').write_text(source())
    (out/'build.sh').write_text((OLD/'build.sh').read_text().replace(str(OLD),str(out)))
    with (out/'build.log').open('w') as f:subprocess.run(['bash',str(out/'build.sh')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    patch(out/'guest',out/'guest.export-safe')
    (out/'frozen.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))
    print(out)


if __name__=='__main__':main()
