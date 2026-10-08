#include <cmath>
#define sqrtf sqrt
#define main cacheflex_micro_spm_flash_single_main
#include "../../../cacheflex_micro/kernels/attention_kernels_spm/11_flash_attn_kv_spm_vl4.cpp"
#undef main
#undef sqrtf

#ifdef CACHEFLEX_FLASH_VL16
#ifdef CACHEFLEX_FLASH_SEARCH_KERNELS
#include CACHEFLEX_FLASH_SEARCH_KERNELS
#else
#include "../../../cacheflex_micro/experiments/end2end/src/spm_qk_kernel_vl16.hpp"
#include "../../../cacheflex_micro/experiments/end2end/src/pv_kernel_8x64_vl16.hpp"
#endif
#define NT_VL4 size_t(128)
#define sve_spm_mla_8x96_vl4(a,c,k,ld,set) spm_kernel_1vl_vl16(a,c,k,ld,k,set)
#define sve_cache_pv_8x64 cache_pv_8x64_vl16
#ifdef CACHEFLEX_FLASH_SEARCH
#define sve_spm_pv_8x64_vl4 spm_pv_8x64_vl16
#endif
#else
#include "../../../cacheflex_micro/experiments/flash_attention/src/pv_kernel_8x64.hpp"
#endif

#include <pthread.h>
#ifndef CACHEFLEX_FLASH_DRIVER_ABLATION
#define CACHEFLEX_FLASH_DRIVER_ABLATION 0
#endif
#if CACHEFLEX_FLASH_DRIVER_ABLATION
#include "driver_vector_pack.hpp"
#endif
#ifndef CACHEFLEX_FLASH_MATCHED
#define CACHEFLEX_FLASH_MATCHED 0
#endif
#ifndef CACHEFLEX_FLASH_BOUNDARIES
#define CACHEFLEX_FLASH_BOUNDARIES 0
#endif
#if CACHEFLEX_FLASH_MATCHED
#if CACHEFLEX_FLASH_BOUNDARIES
#define CF_PHASE(name) __asm__ __volatile__(".global cf_boundary_" #name "_%=\ncf_boundary_" #name "_%=:\n" ::: "memory")
#else
#define CF_PHASE(name) __asm__ __volatile__("" ::: "memory")
#endif
#else
#define CF_PHASE(name) do {} while(0)
#endif
#ifndef CACHEFLEX_FLASH_WIDE_ROWS
#define CACHEFLEX_FLASH_WIDE_ROWS 0
#endif
#ifndef CACHEFLEX_FLASH_FUSE_RESCALE
#define CACHEFLEX_FLASH_FUSE_RESCALE 0
#endif

#ifndef CACHEFLEX_FLASH_MULTICAST
#define CACHEFLEX_FLASH_MULTICAST 0
#endif
#ifndef CACHEFLEX_FLASH_PRECLAIM
#define CACHEFLEX_FLASH_PRECLAIM 0
#endif
#ifndef CACHEFLEX_FLASH_PRETOUCH
#define CACHEFLEX_FLASH_PRETOUCH 0
#endif
#ifndef CACHEFLEX_FLASH_IB_GROUP
#define CACHEFLEX_FLASH_IB_GROUP 0
#endif
#ifndef CACHEFLEX_FLASH_STAGE_K
#define CACHEFLEX_FLASH_STAGE_K 1
#endif
#ifndef CACHEFLEX_FLASH_STAGE_V
#define CACHEFLEX_FLASH_STAGE_V 1
#endif
#ifndef CACHEFLEX_FLASH_K_OVERLAP
#define CACHEFLEX_FLASH_K_OVERLAP 0
#endif
#ifndef CACHEFLEX_FLASH_MULTICAST_PIN_ALL
#define CACHEFLEX_FLASH_MULTICAST_PIN_ALL 0
#endif
#ifndef CACHEFLEX_FLASH_MULTICAST_READY
#define CACHEFLEX_FLASH_MULTICAST_READY 0
#endif
#ifndef CACHEFLEX_FLASH_BATCH3
#define CACHEFLEX_FLASH_BATCH3 0
#endif
#ifndef CACHEFLEX_FLASH_INTERLEAVE_IB
#define CACHEFLEX_FLASH_INTERLEAVE_IB 0
#endif
#ifndef CACHEFLEX_FLASH_PIN_KV
#define CACHEFLEX_FLASH_PIN_KV 0
#endif
#ifndef CACHEFLEX_FLASH_DMA
#define CACHEFLEX_FLASH_DMA 0
#endif

extern "C" void cacheflex_qk_8x96(const __fp16*,const __fp16*,__fp16*,int,int);

namespace {
constexpr int Cores=8;
constexpr uint64_t AllCoreMask=uint64_t{0xff}<<24;
#ifdef CACHEFLEX_FLASH_SEARCH
// Read-only after main initializes it, before starting workers.
static size_t search_bc;
static const __fp16 *matched_kcache;
static size_t search_vbase(size_t bank) {
    const size_t banks=CACHEFLEX_FLASH_K_OVERLAP?2:1;
    return (CACHEFLEX_FLASH_STAGE_K?banks*(search_bc/128)*64:0)+bank*search_bc;
}
#endif

static inline void copy_line(size_t slot,const void *src,uint64_t mask=0)
{
    uint64_t dst=slot|mask;
    __asm__ __volatile__("SPMCP_64_IMM %x[d], [%x[s], #0]\n"
                         ::[d]"r"(dst),[s]"r"(src):"memory");
}
static inline void copy_three_lines(size_t slot,const void *src,uint64_t mask=0)
{
    uint64_t dst=slot|mask;
    __asm__ __volatile__("SPMCPB3_64 %x[d], [%x[s]]\n"
                         ::[d]"r"(dst),[s]"r"(src):"memory");
}
static inline void dma_2d(size_t dst_set,size_t dst_way,size_t rows,
                          size_t lines,size_t src_stride,const void *src,
                          uint64_t mask=0)
{
    uint64_t desc=(dst_set&0x3ff)|((dst_way&7)<<10)|
        (((rows-1)&0x3ff)<<13)|(((lines-1)&7)<<23)|
        ((src_stride&0xffff)<<26)|(((mask>>24)&0xff)<<42);
    __asm__ __volatile__("SPMDMA_2D_64 %x[d], [%x[s]]\n"
                         ::[d]"r"(desc),[s]"r"(src):"memory");
}
static inline void drain_slot(size_t slot)
{
    uint64_t v;
    __asm__ __volatile__("SPMLDR_8_IMM %x[v], [%x[s], #0]\n"
                         :[v]"=r"(v):[s]"r"(slot):"memory");
}
static inline void claim_slot(size_t slot)
{
    uint64_t dst=slot|AllCoreMask|(uint64_t{1}<<23);
    __asm__ __volatile__("SPMCP_64_IMM %x[d], [xzr, #0]\n"
                         ::[d]"r"(dst):"memory");
}

static void claim_layout(size_t ksets,size_t vbase,size_t vsets)
{
    if(CACHEFLEX_FLASH_STAGE_K) for(size_t set=0;set<ksets;++set)
        for(size_t way=0;way<3;++way) claim_slot((way<<16)|(set<<6));
    if(CACHEFLEX_FLASH_STAGE_V) for(size_t set=0;set<vsets;++set)
        for(size_t way=0;way<2;++way) claim_slot((way<<16)|((vbase+set)<<6));
    if(CACHEFLEX_FLASH_STAGE_K)
        drain_slot((2u<<16)|((ksets-1)<<6));
    if(CACHEFLEX_FLASH_STAGE_V)
        drain_slot((1u<<16)|((vbase+vsets-1)<<6));
}

static void claim_pinned_kv(size_t ksets,size_t vsets)
{
    if(CACHEFLEX_FLASH_STAGE_K) for(size_t set=0;set<ksets;++set)
        for(size_t way=0;way<3;++way) claim_slot((way<<16)|(set<<6));
    if(CACHEFLEX_FLASH_STAGE_V) for(size_t set=0;set<vsets;++set)
        for(size_t way=4;way<6;++way) claim_slot((way<<16)|(set<<6));
    if(CACHEFLEX_FLASH_STAGE_K)
        drain_slot((2u<<16)|((ksets-1)<<6));
    if(CACHEFLEX_FLASH_STAGE_V)
        drain_slot((5u<<16)|((vsets-1)<<6));
}

static void stage_kv(const __fp16 *KT,const __fp16 *V,int T,int d,
                     int j,int bc,size_t bblocks,bool multicast,
                     size_t kset_base=0,bool drain=true)
{
    uint64_t mask=multicast?AllCoreMask:0;
#ifdef CACHEFLEX_FLASH_SEARCH
    const size_t bank=kset_base/((search_bc/128)*64);
    if(CACHEFLEX_FLASH_STAGE_K)
      for(size_t nb=0;nb<bblocks;++nb) for(int k=0;k<d;++k)
        for(size_t way=0;way<4;++way)
          copy_line(((kset_base+nb*d+k)<<6)|(way<<16),
                    CACHEFLEX_FLASH_MATCHED ?
                    matched_kcache+(((size_t)j/128+nb)*d+k)*128+way*32 :
                    KT+(size_t)k*T+j+nb*128+way*32);
    if(CACHEFLEX_FLASH_STAGE_V)
      for(int r=0;r<bc;++r) for(size_t way=0;way<2;++way)
        copy_line(((search_vbase(bank)+r)<<6)|(way<<16),
                  V+(size_t)(j+r)*d+way*32);
    // Dedicated address-aware RAW/WAR tracking orders each reused slot.
    return;
#endif
    if(CACHEFLEX_FLASH_STAGE_K) {
      for(size_t nb=0;nb<bblocks;++nb) {
        if(CACHEFLEX_FLASH_DMA) {
          dma_2d(kset_base+nb*(size_t)d,0,d,3,(size_t)T*2,
                 KT+(size_t)j+nb*NT_VL4,mask);
        } else for(int k=0;k<d;++k) {
            size_t set=kset_base+nb*(size_t)d+k;
            const __fp16 *src=KT+(size_t)k*T+j+nb*NT_VL4;
            if(CACHEFLEX_FLASH_BATCH3)
                copy_three_lines((set<<6)|(0u<<16),src,mask);
            else {
                copy_line((set<<6)|(0u<<16),src+0,mask);
                copy_line((set<<6)|(1u<<16),src+32,mask);
                copy_line((set<<6)|(2u<<16),src+64,mask);
#ifdef CACHEFLEX_FLASH_VL16
                copy_line((set<<6)|(3u<<16),src+96,mask);
#endif
            }
        }
      }
    }
    size_t vbase=2*bblocks*(size_t)d;
    if(CACHEFLEX_FLASH_STAGE_V && CACHEFLEX_FLASH_DMA) {
        size_t vset=CACHEFLEX_FLASH_PIN_KV?(size_t)j:vbase;
        size_t vlo=CACHEFLEX_FLASH_PIN_KV?4u:0u;
        dma_2d(vset,vlo,bc,2,(size_t)d*2,V+(size_t)j*d,mask);
    } else if(CACHEFLEX_FLASH_STAGE_V) for(int r=0;r<bc;++r) {
        const __fp16 *src=V+(size_t)(j+r)*d;
        size_t vset=CACHEFLEX_FLASH_PIN_KV?(size_t)(j+r):vbase+r;
        size_t vlo=CACHEFLEX_FLASH_PIN_KV?4u:0u;
        copy_line((vset<<6)|(vlo<<16),src,mask);
        copy_line((vset<<6)|((vlo+1)<<16),src+32,mask);
    }
    if(drain && CACHEFLEX_FLASH_STAGE_K)
        drain_slot(CACHEFLEX_FLASH_DMA ? (kset_base<<6) :
                   ((2u<<16)|((kset_base+bblocks*(size_t)d-1)<<6)));
    if(drain && CACHEFLEX_FLASH_STAGE_V) {
        size_t vlo=CACHEFLEX_FLASH_PIN_KV?4u:0u;
        size_t vset=CACHEFLEX_FLASH_PIN_KV?(size_t)j:vbase;
        drain_slot(CACHEFLEX_FLASH_DMA ? ((vlo<<16)|(vset<<6)) :
                   (((vlo+1)<<16)|((vset+bc-1)<<6)));
    }
}

static __fp16 pattern(size_t i,unsigned salt)
{
#if CACHEFLEX_FLASH_RICH_INPUT
    uint64_t x=i+uint64_t(salt)*0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    i=(x^(x>>31))>>29;
#endif
    static const uint16_t bits[]={0x2c00,0xac00,0x3000,0xb000,0x2800,0xa800,0x3400,0xb400};
    uint16_t b=bits[(i*5+salt*3)&7]; __fp16 v; std::memcpy(&v,&b,2); return v;
}

struct Shared {
    int T,d,Br,Bc; float scale;
    AlignedBuffer<__fp16> Q,K,V,O,KT,Kcache;
    pthread_barrier_t bar;
    Shared(int t,int dim,int br,int bc):T(t),d(dim),Br(br),Bc(bc),
      scale(1.f/std::sqrt((float)dim)),Q((size_t)Cores*t*dim),K((size_t)t*dim),
      V((size_t)t*dim),O((size_t)Cores*t*dim),KT((size_t)dim*t),
      Kcache(((size_t)t+NT_VL4-1)/NT_VL4*dim*NT_VL4)
      {pthread_barrier_init(&bar,nullptr,Cores);}
    ~Shared(){pthread_barrier_destroy(&bar);}
};
struct Ctx{Shared*s;int tid;};

static void *worker(void *p)
{
    auto &c=*static_cast<Ctx*>(p); Shared&s=*c.s;
    int T=s.T,d=s.d,Br=s.Br,Bc=s.Bc;
    size_t kbank_stride=((size_t)Bc+NT_VL4-1)/NT_VL4*(size_t)d;
    size_t n_ib=(T+Br-1)/Br, abmax=((size_t)Br+MT_VAL-1)/MT_VAL;
    size_t ib_group=CACHEFLEX_FLASH_IB_GROUP?std::min((size_t)CACHEFLEX_FLASH_IB_GROUP,n_ib):n_ib;
    size_t group_rows=std::min((size_t)T,ib_group*(size_t)Br);
    AlignedBuffer<__fp16> qap(ib_group*abmax*MT_VAL*d),
      S((CACHEFLEX_FLASH_INTERLEAVE_IB?group_rows:(size_t)Br)*(size_t)Bc),
      P((CACHEFLEX_FLASH_INTERLEAVE_IB?group_rows:(size_t)Br)*(size_t)Bc),
      PV(8*d),pap(8*Bc);
    AlignedBuffer<float> Oacc(group_rows*d),m(group_rows),l(group_rows);
    AlignedBuffer<float> rescale(CACHEFLEX_FLASH_FUSE_RESCALE?group_rows:0);
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 16
    AlignedBuffer<__fp16> qcopy((size_t)Br*d);
    if(CACHEFLEX_FLASH_PRETOUCH) std::memset(qcopy.data(),0,(size_t)Br*d*2);
#endif
    if(CACHEFLEX_FLASH_PRETOUCH){
      size_t scratch_rows=CACHEFLEX_FLASH_INTERLEAVE_IB?group_rows:(size_t)Br;
      std::memset(qap.data(),0,ib_group*abmax*MT_VAL*d*2);std::memset(S.data(),0,scratch_rows*(size_t)Bc*2);
      std::memset(P.data(),0,scratch_rows*(size_t)Bc*2);std::memset(PV.data(),0,8*d*2);
      std::memset(pap.data(),0,8*Bc*2);std::memset(Oacc.data(),0,group_rows*d*4);
      std::memset(m.data(),0,group_rows*4);std::memset(l.data(),0,group_rows*4);
    }
    const __fp16 *Q=s.Q.data()+(size_t)c.tid*T*d;
    __fp16 *O=s.O.data()+(size_t)c.tid*T*d;

    std::printf("FLASH_ADDR tid=%d qap=%p S=%p P=%p PV=%p pap=%p Oacc=%p m=%p l=%p\n",
        c.tid,qap.data(),S.data(),P.data(),PV.data(),pap.data(),Oacc.data(),m.data(),l.data());

    std::printf("FLASH_THREAD_READY tid=%d\n",c.tid);
    if(CACHEFLEX_FLASH_MULTICAST && CACHEFLEX_FLASH_PRECLAIM && c.tid==0) {
        size_t maxbb=((size_t)Bc+NT_VL4-1)/NT_VL4;
        claim_layout(maxbb*d,maxbb*d,Bc);
    }
    pthread_barrier_wait(&s.bar);
    if(c.tid==0){std::printf("FLASH_ROI begin rpns=%llu\n",(unsigned long long)m5_rpns());m5_reset_stats(0,0);m5_work_begin(0,0);CF_PHASE(prepare_K);transpose_K(s.K.data(),s.KT.data(),T,d);
      if(!CACHEFLEX_FLASH_STAGE_K || CACHEFLEX_FLASH_MATCHED){
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 4
        size_t nts=((size_t)T+NT_VL4-1)/NT_VL4;
        for(size_t nb=0;nb<nts;++nb) {
          auto* dst=s.Kcache.data()+nb*d*NT_VL4;
          std::memset(dst,0,d*NT_VL4*2);
          size_t nc=std::min(NT_VL4,(size_t)T-nb*NT_VL4);
          for(int k=0;k<d;++k) for(size_t x=0;x<nc;x+=svcnth()) {
            auto pg=svwhilelt_b16_u64(x,nc);
            svst1_f16(pg,dst+k*NT_VL4+x,svld1_f16(pg,s.KT.data()+(size_t)k*T+nb*NT_VL4+x));
          }
        }
#else
        size_t nts=((size_t)T+NT_VL4-1)/NT_VL4;for(size_t nb=0;nb<nts;++nb)for(int k=0;k<d;++k)for(size_t x=0;x<NT_VL4;++x){size_t col=nb*NT_VL4+x;s.Kcache.data()[(nb*d+k)*NT_VL4+x]=col<(size_t)T?s.KT.data()[(size_t)k*T+col]:(__fp16)0;}
#endif
      }}
    CF_PHASE(startup_sync);
    pthread_barrier_wait(&s.bar);
    if(CACHEFLEX_FLASH_MULTICAST && !CACHEFLEX_FLASH_PRECLAIM && c.tid==0) {
        size_t maxbb=((size_t)Bc+NT_VL4-1)/NT_VL4;
        if(CACHEFLEX_FLASH_PIN_KV)
          claim_pinned_kv(((size_t)T+Bc-1)/Bc*maxbb*d,T);
        else claim_layout(maxbb*d,maxbb*d,Bc);
    }
    if(!(CACHEFLEX_FLASH_DRIVER_ABLATION & 8)) pthread_barrier_wait(&s.bar);

    if(CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {
      if(c.tid==0) for(int j=0;j<T;j+=Bc) {
        int bc=std::min(Bc,T-j); size_t bb=((size_t)bc+NT_VL4-1)/NT_VL4;
        size_t panel=(size_t)j/Bc;
        stage_kv(s.KT.data(),s.V.data(),T,d,j,bc,bb,true,
                 panel*kbank_stride,true);
      }
      pthread_barrier_wait(&s.bar);
    }

    for(size_t group_begin=0;group_begin<n_ib;group_begin+=ib_group){
     CF_PHASE(query_pack_init);
     size_t group_count=std::min(ib_group,n_ib-group_begin);
     size_t active_rows=std::min(group_count*(size_t)Br,(size_t)T-group_begin*(size_t)Br);
     for(size_t local_ib=0;local_ib<group_count;++local_ib){int i=(group_begin+local_ib)*Br,br=std::min(Br,T-i);
       const __fp16* qp=Q;size_t qm=T,qi=i;
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 16
       std::memcpy(qcopy.data(),Q+(size_t)i*d,(size_t)br*d*2);
       qp=qcopy.data();qm=br;qi=0;
#endif
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 2
       cf_vector_pack_A(qp,d,qap.data()+local_ib*abmax*MT_VAL*d,qm,qi,d,0,br,d);
#else
       pack_A_fp16_8row(qp,d,qap.data()+local_ib*abmax*MT_VAL*d,qm,qi,d,0,br,d);
#endif
     }
     std::fill_n(Oacc.data(),active_rows*d,0.f);std::fill_n(m.data(),active_rows,-1e30f);std::fill_n(l.data(),active_rows,0.f);
     if(CACHEFLEX_FLASH_K_OVERLAP && !CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {
       CF_PHASE(stage_K_initial);
       int bc0=std::min(Bc,T); size_t bb0=((size_t)bc0+NT_VL4-1)/NT_VL4;
       if(CACHEFLEX_FLASH_MULTICAST) {
         if(CACHEFLEX_FLASH_MULTICAST_READY)
           stage_kv(s.KT.data(),s.V.data(),T,d,0,bc0,bb0,false,0,true);
         else if(c.tid==0)
           stage_kv(s.KT.data(),s.V.data(),T,d,0,bc0,bb0,true,0,true);
         pthread_barrier_wait(&s.bar);
       } else stage_kv(s.KT.data(),s.V.data(),T,d,0,bc0,bb0,false,0,false);
     }
     for(int j=0;j<T;j+=Bc){
      CF_PHASE(stage_K_panel);
      int bc=std::min(Bc,T-j);size_t bb=((size_t)bc+NT_VL4-1)/NT_VL4,bcp=bb*NT_VL4;
      size_t panel=(size_t)j/Bc;
      size_t kset_base=CACHEFLEX_FLASH_MULTICAST_PIN_ALL?
          panel*kbank_stride:(panel&1)*kbank_stride;
#ifdef CACHEFLEX_FLASH_SEARCH
      if(!CACHEFLEX_FLASH_K_OVERLAP) kset_base=0;
#endif
      if(CACHEFLEX_FLASH_MULTICAST_PIN_ALL) {
        // All panels were installed before the compute loop.
      } else if(CACHEFLEX_FLASH_K_OVERLAP) {
        int next=j+Bc;
        if(next<T) { int nbc=std::min(Bc,T-next); size_t nbb=((size_t)nbc+NT_VL4-1)/NT_VL4;
          size_t next_base=((panel+1)&1)*kbank_stride;
          if(!CACHEFLEX_FLASH_MULTICAST || c.tid==0)
            stage_kv(s.KT.data(),s.V.data(),T,d,next,nbc,nbb,
                     CACHEFLEX_FLASH_MULTICAST,next_base,false); }
      } else {
        if(!CACHEFLEX_FLASH_MULTICAST||c.tid==0)
          stage_kv(s.KT.data(),s.V.data(),T,d,j,bc,bb,CACHEFLEX_FLASH_MULTICAST);
#ifndef CACHEFLEX_FLASH_SEARCH
        pthread_barrier_wait(&s.bar);
#endif
      }
      auto qk_phase = [&](size_t local_ib) {
        CF_PHASE(QK);
        size_t ib=group_begin+local_ib, local_i=local_ib*(size_t)Br;
        int i=ib*Br,br=std::min(Br,T-i);size_t abs=((size_t)br+MT_VAL-1)/MT_VAL;
        const __fp16*q=qap.data()+local_ib*abmax*MT_VAL*d;
        size_t scratch_i=CACHEFLEX_FLASH_INTERLEAVE_IB?local_i:0;
        __fp16 *sbuf=S.data()+scratch_i*bcp;
        for(size_t mb=0;mb<abs;){
          size_t step=1;
#if CACHEFLEX_FLASH_WIDE_ROWS
          if(abs-mb>=CACHEFLEX_FLASH_WIDE_ROWS/8) step=CACHEFLEX_FLASH_WIDE_ROWS/8;
#endif
          for(size_t nb=0;nb<bb;++nb){
#if CACHEFLEX_FLASH_WIDE_ROWS
          if(step==CACHEFLEX_FLASH_WIDE_ROWS/8) {
            if(CACHEFLEX_FLASH_STAGE_K) cf_wide_qk_spm(q+mb*8*d,nullptr,sbuf+mb*8*bcp+nb*128,d,bcp,kset_base+nb*d);
            else cf_wide_qk_cache(q+mb*8*d,s.Kcache.data()+(((size_t)j/128+nb)*d*128),sbuf+mb*8*bcp+nb*128,d,bcp,0);
            continue;
          }
#endif
          if(CACHEFLEX_FLASH_STAGE_K) sve_spm_mla_8x96_vl4(q+mb*MT_VAL*d,sbuf+mb*MT_VAL*bcp+nb*NT_VL4,d,bcp,kset_base+nb*(size_t)d);
          else {
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 32
            cache_qk_8x1VL_ldc(q+mb*MT_VAL*d,s.Kcache.data()+(((size_t)j/NT_VL4+nb)*d*NT_VL4),sbuf+mb*MT_VAL*bcp+nb*NT_VL4,d,bcp);
#else
            cacheflex_qk_8x96(q+mb*MT_VAL*d,s.Kcache.data()+(((size_t)j/NT_VL4+nb)*d*NT_VL4),sbuf+mb*MT_VAL*bcp+nb*NT_VL4,d,bcp);
#endif
          }
          }
          mb+=step;
        }
        size_t nc=std::min(NT_VL4,(size_t)T-((size_t)j/NT_VL4+bb-1)*NT_VL4);
        if(nc<NT_VL4)for(int r=0;r<br;++r)for(size_t x=nc;x<NT_VL4;++x)
          sbuf[(size_t)r*bcp+(bb-1)*NT_VL4+x]=(__fp16)-65504.f;
      };
      auto softmax_phase = [&](size_t local_ib) {
        CF_PHASE(softmax_rescale);
        size_t ib=group_begin+local_ib, local_i=local_ib*(size_t)Br;
        int i=ib*Br,br=std::min(Br,T-i);
        size_t scratch_i=CACHEFLEX_FLASH_INTERLEAVE_IB?local_i:0;
        __fp16 *sbuf=S.data()+scratch_i*bcp, *pbuf=P.data()+scratch_i*bcp;
        if(CACHEFLEX_FLASH_INTERLEAVE_IB) {
          for(int r=0;r<br;r+=2) {
            int r1=std::min(r+1,br-1); auto *sr0=sbuf+(size_t)r*bcp,*sr1=sbuf+(size_t)r1*bcp;
            auto *pr0=pbuf+(size_t)r*bcp,*pr1=pbuf+(size_t)r1*bcp; size_t vl=svcntw();
            svfloat32_t vm0=svdup_n_f32(-1e30f),vm1=svdup_n_f32(-1e30f);
            for(size_t x=0;x<(size_t)bc;x+=vl) { auto pg=svwhilelt_b32_u64(x,bc);auto ph=svwhilelt_b16_u64(x,std::min(x+vl,(size_t)bc));
              vm0=svmax_f32_m(pg,vm0,svmul_n_f32_x(pg,svcvt_f32_f16_z(pg,svld1_f16(ph,sr0+x)),s.scale));
              vm1=svmax_f32_m(pg,vm1,svmul_n_f32_x(pg,svcvt_f32_f16_z(pg,svld1_f16(ph,sr1+x)),s.scale)); }
            float mij0=svmaxv_f32(svptrue_b32(),vm0),mij1=svmaxv_f32(svptrue_b32(),vm1);
            float mn0=std::max(m.data()[local_i+r],mij0),mn1=std::max(m.data()[local_i+r1],mij1);
            float a0=expf(m.data()[local_i+r]-mn0),a1=expf(m.data()[local_i+r1]-mn1);
            svfloat32_t sum0=svdup_n_f32(0),sum1=svdup_n_f32(0);
            for(size_t x=0;x<(size_t)bc;x+=vl) { auto pg=svwhilelt_b32_u64(x,bc);auto ph=svwhilelt_b16_u64(x,std::min(x+vl,(size_t)bc));
              auto z0=sve_exp_f32(pg,svmla_n_f32_x(pg,svdup_n_f32(-mn0),svcvt_f32_f16_z(pg,svld1_f16(ph,sr0+x)),s.scale));
              auto z1=sve_exp_f32(pg,svmla_n_f32_x(pg,svdup_n_f32(-mn1),svcvt_f32_f16_z(pg,svld1_f16(ph,sr1+x)),s.scale));
              sum0=svadd_f32_m(pg,sum0,z0);sum1=svadd_f32_m(pg,sum1,z1);
              svst1_f16(ph,pr0+x,svcvt_f16_f32_z(pg,z0));svst1_f16(ph,pr1+x,svcvt_f16_f32_z(pg,z1)); }
            m.data()[local_i+r]=mn0;l.data()[local_i+r]=a0*l.data()[local_i+r]+svaddv_f32(svptrue_b32(),sum0);
            scale_f32_sve(Oacc.data()+(local_i+r)*d,a0,d);
            if(r1!=r) {m.data()[local_i+r1]=mn1;l.data()[local_i+r1]=a1*l.data()[local_i+r1]+svaddv_f32(svptrue_b32(),sum1);
              scale_f32_sve(Oacc.data()+(local_i+r1)*d,a1,d);}
          }
          return;
        }
        for(int r=0;r<br;++r){
          auto*sr=sbuf+(size_t)r*bcp;auto*pr=pbuf+(size_t)r*bcp;size_t vl=svcntw();
          svfloat32_t vm=svdup_n_f32(-1e30f);
          for(size_t x=0;x<(size_t)bc;x+=vl){auto pg=svwhilelt_b32_u64(x,bc);auto ph=svwhilelt_b16_u64(x,std::min(x+vl,(size_t)bc));
            vm=svmax_f32_m(pg,vm,svmul_n_f32_x(pg,svcvt_f32_f16_z(pg,svld1_f16(ph,sr+x)),s.scale));}
          float mij=svmaxv_f32(svptrue_b32(),vm),mn=std::max(m.data()[local_i+r],mij),a=expf(m.data()[local_i+r]-mn);svfloat32_t sum=svdup_n_f32(0);
          for(size_t x=0;x<(size_t)bc;x+=vl){auto pg=svwhilelt_b32_u64(x,bc);auto ph=svwhilelt_b16_u64(x,std::min(x+vl,(size_t)bc));
            auto z=sve_exp_f32(pg,svmla_n_f32_x(pg,svdup_n_f32(-mn),svcvt_f32_f16_z(pg,svld1_f16(ph,sr+x)),s.scale));sum=svadd_f32_m(pg,sum,z);svst1_f16(ph,pr+x,svcvt_f16_f32_z(pg,z));}
          m.data()[local_i+r]=mn;l.data()[local_i+r]=a*l.data()[local_i+r]+svaddv_f32(svptrue_b32(),sum);
          if(CACHEFLEX_FLASH_FUSE_RESCALE) rescale.data()[local_i+r]=a;
          else scale_f32_sve(Oacc.data()+(local_i+r)*d,a,d);
        }
      };
      auto pv_phase = [&](size_t local_ib) {
        CF_PHASE(PV_pack_accumulate);
        size_t ib=group_begin+local_ib, local_i=local_ib*(size_t)Br;
        int i=ib*Br,br=std::min(Br,T-i);size_t abs=((size_t)br+MT_VAL-1)/MT_VAL;
        size_t scratch_i=CACHEFLEX_FLASH_INTERLEAVE_IB?local_i:0;
        __fp16 *pbuf=P.data()+scratch_i*bcp;
        // Adding 4*1024 to the logical set number becomes way 4 after the
        // PV kernel's <<6 address formation; its existing OR selects way 5.
        size_t vb=CACHEFLEX_FLASH_PIN_KV?((4u<<10)+(size_t)j):2*bb*(size_t)d;
#ifdef CACHEFLEX_FLASH_SEARCH
        vb=search_vbase(CACHEFLEX_FLASH_K_OVERLAP?(panel&1):0);
#endif
        for(size_t mb=0;mb<abs;++mb){size_t row=mb*MT_VAL,rows=std::min((size_t)MT_VAL,(size_t)br-row);
#if CACHEFLEX_FLASH_DRIVER_ABLATION & 1
          cf_vector_pack_A(pbuf,bcp,pap.data(),br,row,bc,0,rows,bc);
#else
          pack_A_fp16_8row(pbuf,bcp,pap.data(),br,row,bc,0,rows,bc);
#endif
          if(CACHEFLEX_FLASH_STAGE_V) sve_spm_pv_8x64_vl4(pap.data(),PV.data(),bc,d,vb);
          else sve_cache_pv_8x64(pap.data(),s.V.data()+(size_t)j*d,PV.data(),bc,d);
          for(size_t r=0;r<rows;++r) {
            if(CACHEFLEX_FLASH_FUSE_RESCALE) {
              float* dst=Oacc.data()+(local_i+row+r)*d;
              const __fp16* src=PV.data()+r*d;
              for(size_t x=0;x<(size_t)d;x+=svcntw()) {
                auto pg=svwhilelt_b32_u64(x,d);
                auto ph=svwhilelt_b16_u64(x,std::min(x+svcntw(),(size_t)d));
                auto scaled=svmul_n_f32_x(pg,svld1_f32(pg,dst+x),rescale.data()[local_i+row+r]);
                // Preserve the original separately rounded multiply then add.
                __asm__ __volatile__("" : "+w"(scaled));
                svst1_f32(pg,dst+x,svadd_f32_x(pg,scaled,svcvt_f32_f16_z(pg,svld1_f16(ph,src+x))));
              }
            } else acc_f16_to_f32_sve(PV.data()+r*d,Oacc.data()+(local_i+row+r)*d,d);
          }}
      };
      if(CACHEFLEX_FLASH_INTERLEAVE_IB) {
        for(size_t local_ib=0;local_ib<group_count;++local_ib) qk_phase(local_ib);
        for(size_t local_ib=0;local_ib<group_count;++local_ib) softmax_phase(local_ib);
        for(size_t local_ib=0;local_ib<group_count;++local_ib) pv_phase(local_ib);
      } else {
        for(size_t local_ib=0;local_ib<group_count;++local_ib) {
          qk_phase(local_ib); softmax_phase(local_ib); pv_phase(local_ib);
        }
      }
      if(CACHEFLEX_FLASH_K_OVERLAP && CACHEFLEX_FLASH_MULTICAST &&
         !CACHEFLEX_FLASH_MULTICAST_PIN_ALL &&
         !CACHEFLEX_FLASH_MULTICAST_READY) {
        int next=j+Bc;
        if(c.tid==0 && next<T) {
          int nbc=std::min(Bc,T-next); size_t nbb=((size_t)nbc+NT_VL4-1)/NT_VL4;
          size_t next_base=((panel+1)&1)*kbank_stride;
          drain_slot((2u<<16)|((next_base+nbb*(size_t)d-1)<<6));
        }
        pthread_barrier_wait(&s.bar);
      }
#ifndef CACHEFLEX_FLASH_SEARCH
      else if(!CACHEFLEX_FLASH_K_OVERLAP) pthread_barrier_wait(&s.bar);
#endif
     }
     CF_PHASE(final_normalize);
     for(size_t r=0;r<active_rows;++r){float inv=1.f/l.data()[r];size_t vl=svcntw();for(size_t x=0;x<(size_t)d;x+=vl){auto pg=svwhilelt_b32_u64(x,d);auto ph=svwhilelt_b16_u64(x,std::min(x+vl,(size_t)d));svst1_f16(ph,O+(group_begin*(size_t)Br+r)*d+x,svcvt_f16_f32_z(pg,svmul_n_f32_x(pg,svld1_f32(pg,Oacc.data()+r*d+x),inv)));}}
    }
    CF_PHASE(final_sync);
    pthread_barrier_wait(&s.bar);CF_PHASE(done);if(c.tid==0){m5_work_end(0,0);m5_dump_stats(0,0);std::printf("FLASH_ROI end rpns=%llu\n",(unsigned long long)m5_rpns());}return nullptr;
}
}

int main(int argc,char**argv)
{
#ifdef CACHEFLEX_FLASH_VL16
 if(svcnth()!=128 || CACHEFLEX_FLASH_MULTICAST || CACHEFLEX_FLASH_DMA || CACHEFLEX_FLASH_BATCH3 || CACHEFLEX_FLASH_PRECLAIM) return 2;
#ifndef CACHEFLEX_FLASH_SEARCH
 if(CACHEFLEX_FLASH_STAGE_V) return 2;
#endif
#endif
 setbuf(stdout,nullptr);int T=argc>1?atoi(argv[1]):256,d=argc>2?atoi(argv[2]):64,Br=argc>3?atoi(argv[3]):64,Bc=argc>4?atoi(argv[4]):96;
 Bc=((Bc+(int)NT_VL4-1)/(int)NT_VL4)*(int)NT_VL4;
#ifdef CACHEFLEX_FLASH_SEARCH
 search_bc=Bc;
 if(CACHEFLEX_FLASH_FUSE_RESCALE && CACHEFLEX_FLASH_INTERLEAVE_IB) return 2;
 if(T<=0 || T%128 || Br<=0 || Br%8 || Bc<=0) return 2;
 const size_t banks=CACHEFLEX_FLASH_K_OVERLAP?2:1;
 const size_t sets=banks*((CACHEFLEX_FLASH_STAGE_K?(Bc/128)*64:0)+(CACHEFLEX_FLASH_STAGE_V?Bc:0));
 if(sets>1024) return 2;
#endif
 if(d!=64 || (CACHEFLEX_FLASH_MULTICAST_PIN_ALL && (T+Bc-1)/Bc>4))return 2;Shared s(T,d,Br,Bc);
#ifdef CACHEFLEX_FLASH_SEARCH
 matched_kcache=s.Kcache.data();
#endif
 for(size_t i=0;i<(size_t)Cores*T*d;++i)s.Q.data()[i]=pattern(i,1);for(size_t i=0;i<(size_t)T*d;++i){s.K.data()[i]=pattern(i,2);s.V.data()[i]=pattern(i,3);}std::memset(s.O.data(),0,(size_t)Cores*T*d*2);
 printf("FLASH_SHARED Q=%p K=%p V=%p O=%p KT=%p\n",s.Q.data(),s.K.data(),s.V.data(),s.O.data(),s.KT.data());
 pthread_t th[Cores];Ctx c[Cores];c[0]={&s,0};for(int i=1;i<Cores;++i){c[i]={&s,i};pthread_create(&th[i],nullptr,worker,&c[i]);}worker(&c[0]);for(int i=1;i<Cores;++i)pthread_join(th[i],nullptr);
 double sum=0;uint64_t hash=1469598103934665603ULL;size_t nonfinite=0;
 for(size_t i=0;i<(size_t)Cores*T*d;++i){uint16_t bits;std::memcpy(&bits,&s.O.data()[i],sizeof(bits));sum+=(float)s.O.data()[i];hash=(hash^bits)*1099511628211ULL;nonfinite+=((bits&0x7c00)==0x7c00);}
 printf("FLASH_RESULT variant=%s cores=8 T=%d Hq=8 Hkv=1 d=%d Br=%d Bc=%d checksum=%.10g hash=%016llx nonfinite=%zu\n",CACHEFLEX_FLASH_MULTICAST?"spm_multicast":"spm_private",T,d,Br,Bc,sum,(unsigned long long)hash,nonfinite);return 0;
}
