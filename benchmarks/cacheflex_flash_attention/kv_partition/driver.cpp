// Single-head, noncausal FA. Existing VL16 FP16 GEMMs; FP32 online state.
// Modes: query partition + streaming V, or KV partition + persistent private V.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <arm_sve.h>
#include <gem5/m5ops.h>
#include "correct_fa_conversions.hpp"
#include "kernels.hpp"
#include "pv_fullwidth.hpp"
#include "math_helpers.hpp"
#include "driver_vector_pack.hpp"

#define PHASE(n) asm volatile(".global kv_phase_" #n "_%=\nkv_phase_" #n "_%=:\n" ::: "memory")
template<class T> struct Buffer {
    T* p; size_t n;
    explicit Buffer(size_t count):n(count) {
        p=static_cast<T*>(std::aligned_alloc(64,(n*sizeof(T)+63)&~size_t(63)));
        if(!p) std::abort();
        std::memset(p,0,n*sizeof(T));
    }
    ~Buffer(){std::free(p);}
    Buffer(const Buffer&)=delete;
};
static __fp16 pattern(size_t i,uint64_t salt) {
    uint64_t x=i+salt*0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;i=(x^(x>>31))>>29;
    const uint16_t b[]={0x2c00,0xac00,0x3000,0xb000,0x2800,0xa800,0x3400,0xb400};
    uint16_t bits=b[(i*5+salt*3)&7];__fp16 value;std::memcpy(&value,&bits,2);return value;
}
// Hardware encoding has ten set bits even when fewer physical sets exist.
static size_t vbase(size_t row,size_t sets) {return (row%sets)+((row/sets)*2<<10);}
static void stage(const __fp16* src,size_t rows,size_t sets) {
    PHASE(stage);
    for(size_t r=0;r<rows;++r) for(size_t w=0;w<2;++w) {
        size_t dst=(vbase(r,sets)<<6)|(w<<16);
        const __fp16* from=src+r*64+w*32;
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\n" ::"r"(dst),"r"(from):"memory");
    }
    // No whole-panel fence: existing per-slot RAW tracking orders consumers.
}
static void release(size_t rows,size_t sets) {
    PHASE(release);
    asm volatile("dsb sy" ::: "memory");
    for(size_t r=0;r<rows;++r) for(size_t w=0;w<2;++w) {
        size_t slot=(vbase(r,sets)<<6)|(w<<16);
        asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n" ::"r"(slot):"memory");
    }
    asm volatile("dsb sy" ::: "memory");
}
struct Shared {
    int t,cores,br,bc,group,sets;bool kv,spm;
    Buffer<__fp16> q,k,v,out,kpack;
    // Structure-of-arrays, core slabs rounded to cacheline multiples.
    Buffer<float> partial,maximum,denominator;
    pthread_barrier_t barrier;
    Shared(int T,int C,int R,int B,int G,int S,bool KV,bool SPM):
      t(T),cores(C),br(R),bc(B),group(G),sets(S),kv(KV),spm(SPM),
      q(T*64),k(T*64),v(T*64),out(T*64),kpack(T*64),
      partial(C*R*G*64),maximum(C*R*G),denominator(C*R*G) {
        pthread_barrier_init(&barrier,nullptr,C);
    }
    ~Shared(){pthread_barrier_destroy(&barrier);}
};
struct Context{Shared* s;int id;};
static void* worker(void* arg) {
    auto c=*static_cast<Context*>(arg);auto& s=*c.s;
    const int rows=s.br*s.group,local=s.t/s.cores;
    Buffer<__fp16> qa(rows*64),scores(s.br*s.bc),prob(s.br*s.bc),pa(16*s.bc),pv(16*64);
    float* o=s.partial.p+c.id*rows*64;
    float* m=s.maximum.p+c.id*rows;float* l=s.denominator.p+c.id*rows;
    const auto pf=svptrue_b32();const auto ph=svwhilelt_b16_u64(0,64);
    pthread_barrier_wait(&s.barrier);
    if(c.id==0){std::printf("KV_ROI_BEGIN\n");m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&s.barrier);
    PHASE(pack_k);
    // Disjoint K panels packed by their eventual KV owner (all modes).
    for(int j=c.id*local;j<(c.id+1)*local;j+=128)
      for(int d=0;d<64;++d) for(int x=0;x<128;++x)
        s.kpack.p[(j/128*64+d)*128+x]=s.k.p[(j+x)*64+d];
    if(s.spm && s.kv) stage(s.v.p+c.id*local*64,local,s.sets);
    pthread_barrier_wait(&s.barrier);
    const int qbegin=s.kv?0:c.id*local,qend=s.kv?s.t:(c.id+1)*local;
    const int kbegin=s.kv?c.id*local:0,kend=s.kv?(c.id+1)*local:s.t;
    for(int qi=qbegin;qi<qend;qi+=rows) {
        const int nr=std::min(rows,qend-qi);
        PHASE(pack_q);
        cf_vector_pack_A(s.q.p,64,qa.p,s.t,qi,64,0,nr,64);
        std::fill(m,m+nr,-1e30f);std::fill(l,l+nr,0.f);std::fill(o,o+nr*64,0.f);
        for(int j=kbegin;j<kend;j+=s.bc) {
            const int bc=std::min(s.bc,kend-j);
            if(s.spm && !s.kv) stage(s.v.p+j*64,bc,s.sets);
            for(int bi=0;bi<nr;bi+=s.br) {
                const int br=std::min(s.br,nr-bi);
                PHASE(qk);
                for(int r=0;r<br;r+=16) for(int nb=0;nb<bc;nb+=128)
                    cf_wide_qk_cache(qa.p+(bi+r)*64,
                        s.kpack.p+(j+nb)/128*64*128,scores.p+r*s.bc+nb,64,s.bc,0);
                PHASE(softmax);
                for(int r=0;r<br;++r) {
                    auto* sr=scores.p+r*s.bc;auto* pr=prob.p+r*s.bc;
                    auto mx=svdup_n_f32(-1e30f);
                    for(int x=0;x<bc;x+=64)
                        mx=svmax_f32_x(pf,mx,svmul_n_f32_x(pf,cf_dense_f16_to_f32(pf,svld1_f16(ph,sr+x)),.125f));
                    float mn=std::max(m[bi+r],svmaxv_f32(pf,mx));
                    float a=expf(m[bi+r]-mn);auto sum=svdup_n_f32(0);
                    for(int x=0;x<bc;x+=64) {
                        auto z=sve_exp_f32(pf,svmla_n_f32_x(pf,svdup_n_f32(-mn),cf_dense_f16_to_f32(pf,svld1_f16(ph,sr+x)),.125f));
                        sum=svadd_f32_x(pf,sum,z);svst1_f16(ph,pr+x,cf_dense_f32_to_f16(pf,z));
                    }
                    m[bi+r]=mn;l[bi+r]=a*l[bi+r]+svaddv_f32(pf,sum);
                    auto* dst=o+(bi+r)*64;
                    svst1_f32(pf,dst,svmul_n_f32_x(pf,svld1_f32(pf,dst),a));
                }
                PHASE(pv);
                for(int r=0;r<br;r+=16) {
                    cf_vector_pack_A(prob.p,s.bc,pa.p,br,r,bc,0,16,bc);
                    if(s.spm) cf_pv16_spm(pa.p,vbase(s.kv?j-kbegin:0,s.sets),pv.p,bc,64);
                    else cf_pv16_cache(pa.p,reinterpret_cast<size_t>(s.v.p+j*64),pv.p,bc,64);
                    for(int a=0;a<16;++a) {
                        auto* dst=o+(bi+r+a)*64;
                        svst1_f32(pf,dst,svadd_f32_x(pf,svld1_f32(pf,dst),cf_dense_f16_to_f32(pf,svld1_f16(ph,pv.p+a*64))));
                    }
                }
            }
        }
        if(s.kv) {
            PHASE(reduction_wait);pthread_barrier_wait(&s.barrier);
            PHASE(reduction);
            // Distinct output rows per reducer. Stable merge of (m,l,unnormalized O).
            for(int r=c.id;r<nr;r+=s.cores) {
                float mx=-1e30f;
                for(int p=0;p<s.cores;++p)mx=std::max(mx,s.maximum.p[p*rows+r]);
                float den=0;auto val=svdup_n_f32(0);
                for(int p=0;p<s.cores;++p) {
                    float a=expf(s.maximum.p[p*rows+r]-mx);
                    den+=a*s.denominator.p[p*rows+r];
                    val=svadd_f32_x(pf,val,svmul_n_f32_x(pf,svld1_f32(pf,s.partial.p+(p*rows+r)*64),a));
                }
                svst1_f16(ph,s.out.p+(qi+r)*64,cf_dense_f32_to_f16(pf,svmul_n_f32_x(pf,val,1.f/den)));
            }
            PHASE(reduction_reuse_wait);pthread_barrier_wait(&s.barrier);
        } else {
            PHASE(normalize);
            for(int r=0;r<nr;++r)
                svst1_f16(ph,s.out.p+(qi+r)*64,cf_dense_f32_to_f16(pf,svmul_n_f32_x(pf,svld1_f32(pf,o+r*64),1.f/l[r])));
        }
    }
    if(s.spm)release(s.kv?local:s.bc,s.sets);
    PHASE(final_wait);pthread_barrier_wait(&s.barrier);
    if(c.id==0){m5_work_end(0,0);m5_dump_stats(0,0);std::printf("KV_ROI_END\n");}
    return nullptr;
}
int main(int argc,char** argv) {
    if(argc!=9){std::fprintf(stderr,"T cores Br Bc group sets kv_partition spm\n");return 2;}
    int t=atoi(argv[1]),n=atoi(argv[2]),br=atoi(argv[3]),bc=atoi(argv[4]);
    int g=atoi(argv[5]),sets=atoi(argv[6]),kv=atoi(argv[7]),spm=atoi(argv[8]);
    if((n!=1&&n!=4&&n!=8)||t<=0||t%n||t/n%128||br<=0||br%16||bc<=0||bc%128||
       g<=0||sets<128||sets>1024||sets%bc||svcnth()!=128||
       (kv!=0&&kv!=1)||(spm!=0&&spm!=1)||(spm&&kv&&t/n>3*sets))return 2;
    setvbuf(stdout,nullptr,_IONBF,0);
    Shared s(t,n,br,bc,g,sets,kv,spm);
    for(size_t i=0;i<(size_t)t*64;++i){s.q.p[i]=pattern(i,1);s.k.p[i]=pattern(i,2);s.v.p[i]=pattern(i,3);}
    std::printf("KV_CONFIG T=%d cores=%d br=%d bc=%d group=%d sets=%d kv=%d spm=%d V_bytes_per_core=%d\n",t,n,br,bc,g,sets,kv,spm,t/n*128);
    Context c[8];pthread_t threads[8];
    for(int i=0;i<n;++i){c[i]={&s,i};if(i&&pthread_create(&threads[i],nullptr,worker,&c[i]))return 3;}
    worker(&c[0]);for(int i=1;i<n;++i)pthread_join(threads[i],nullptr);
    uint64_t h=1469598103934665603ULL;size_t bad=0;
    for(size_t i=0;i<(size_t)t*64;++i){uint16_t bits;std::memcpy(&bits,s.out.p+i,2);h=(h^bits)*1099511628211ULL;bad+=!std::isfinite(float(s.out.p[i]));}
    std::printf("KV_RESULT hash=%016llx nonfinite=%zu elements=%d\n",(unsigned long long)h,bad,t*64);
    if(m5_write_file(s.out.p,t*128,0,"output.fp16")!=(uint64_t)t*128)return 4;
    return bad?5:0;
}
