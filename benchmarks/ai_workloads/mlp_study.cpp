#ifdef IDENTICAL_CACHE_BASELINE
#define main unused_paper_main
#include "v1_24x1vl.cpp"
#undef main
#else
#define VL_16
#include "kernels_tile_spm.hpp"
#endif
#include <pthread.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <gem5/m5ops.h>
#ifndef MLP_ROWS
#define MLP_ROWS 24
#endif
#ifndef MLP_COMBINED
#define MLP_COMBINED 0
#endif
#ifndef MLP_ACC32
#define MLP_ACC32 0
#endif
constexpr int TileM=MLP_ROWS;
#include "mlp_helpers.hpp"
#if MLP_ACC32
static_assert(TileM==8,"FP32 study uses eight-row tile");
#include "mlp_fp32_kernel.hpp"
#endif

namespace {
constexpr int MaxThreads=8;
int Threads=8, KC=256, NT=128;
struct Shared {
 int t,mpad,d,f,mp,warm,repeats,prepack;
 __fp16 *pg,*pu,*pd;
 __fp16 *wgu=nullptr,*pgu=nullptr,*gu=nullptr;
 float *acc=nullptr;
 size_t total_errors=0;
 __fp16 *x,*wg,*wu,*wd,*gate,*up,*y,*ref;
 pthread_barrier_t barrier;
};
struct Worker {Shared* s; int id;};
__fp16* alloc(size_t n) {
 auto* p=static_cast<__fp16*>(std::aligned_alloc(64,((n*2+63)/64)*64));
 if(!p) std::abort(); std::memset(p,0,n*2); return p;
}
#ifndef IDENTICAL_CACHE_BASELINE
void stage_weights(const __fp16*b,int n,int kc) {
 if(NT==128){pack_B_tile_to_spm_4way(b,n,kc,KC,0);return;}
 // Narrowed 24/16/8 x one-vector sensitivity path; one 64B way at VL4.
 if(NT!=32)std::abort();
 for(int r=0;r<kc;++r) {
  uint64_t slot=(uint64_t)r<<6;
  const __fp16*src=b+(size_t)r*n;
  asm volatile("SPMCP_64_IMM %x[slot], [%x[src], #0]\n" : : [slot]"r"(slot),[src]"r"(src) : "memory");
 }
}
#endif
void gemm(Worker&w,const __fp16*a,const __fp16*b,__fp16*c,int k,int n,
          __fp16*ap,__fp16*bp,int lda=0) {
 auto&s=*w.s;int np=Threads/s.mp,m=w.id/np,nn=w.id%np;
 int mb0=(s.mpad/TileM)*m/s.mp,mb1=(s.mpad/TileM)*(m+1)/s.mp;
 if(mb0==mb1) return;
 for(int k0=0;k0<k;k0+=KC) {
  int kc=std::min(KC,k-k0);
  packA24Neon(a,lda?lda:k,ap,s.mpad,mb0*TileM,(mb1-mb0)*TileM,k0,kc);
  for(int nb=(n/NT)*nn/np;nb<(n/NT)*(nn+1)/np;++nb) {
#ifdef IDENTICAL_CACHE_BASELINE
   const __fp16* panel=bp;
   if(s.prepack) panel=b+(size_t)nb*k*NT+(size_t)k0*NT;
   else for(int r=0;r<kc;++r)std::copy_n(b+(size_t)(k0+r)*n+nb*NT,NT,bp+(size_t)r*NT);
#if MLP_ACC32
   for(int mb=mb0;mb<mb1;++mb)
    mlp_fp32_tile(ap+(size_t)(mb-mb0)*kc*TileM,panel,s.acc+(size_t)mb*TileM*n+nb*NT,kc,n,k0==0);
#else
   cache_gemm_fused_24x1VL(ap,panel,c+(size_t)mb0*TileM*n+nb*NT,mb1-mb0,kc,n,k0==0);
#endif
#else
   stage_weights(b+(size_t)k0*n+nb*NT,n,kc);
#if MLP_ACC32
   for(int mb=mb0;mb<mb1;++mb)
    mlp_fp32_tile(ap+(size_t)(mb-mb0)*kc*TileM,nullptr,s.acc+(size_t)mb*TileM*n+nb*NT,kc,n,k0==0);
#else
   spm_gemm_fused_tile(ap,c+(size_t)mb0*TileM*n+nb*NT,mb1-mb0,kc,n,k0==0,0);
#endif
#endif
  }
 }
#if MLP_ACC32
 for(int row=mb0*TileM;row<mb1*TileM;++row)
  for(int col=(n/NT)*nn/np*NT;col<(n/NT)*(nn+1)/np*NT;col+=4)
   vst1_f16(c+(size_t)row*n+col,vcvt_f16_f32(vld1q_f32(s.acc+(size_t)row*n+col)));
#endif
}
void pack_weights(Worker&w,const __fp16*b,__fp16*p,int k,int n) {
 for(int nb=w.id;nb<n/NT;nb+=Threads)
  for(int r=0;r<k;++r)std::copy_n(b+(size_t)r*n+nb*NT,NT,p+((size_t)nb*k+r)*NT);
}
size_t check(Shared&s,int iteration) {
 size_t errors=0,nonfinite=0;double maxerr=0,checksum=0;uint64_t hash=1469598103934665603ULL;
 for(size_t i=0;i<(size_t)s.t*s.d;++i) {
  float actual=s.y[i],ref=s.ref[i];double e=std::abs(actual-ref);maxerr=std::max(maxerr,e);
  nonfinite+=!std::isfinite(actual);errors+=!std::isfinite(actual)||e>0.000002+0.01*std::abs(ref);
  checksum+=actual;uint16_t bits;std::memcpy(&bits,s.y+i,2);hash=(hash^bits)*1099511628211ULL;
 }
 printf("MLP_RESULT iteration=%d hash=%016llx checksum=%.12g max_abs=%.9g errors=%zu nonfinite=%zu %s\n",iteration,(unsigned long long)hash,checksum,maxerr,errors,nonfinite,errors?"FAIL":"PASS");
 return errors;
}
void* worker(void*p) {
 auto&w=*static_cast<Worker*>(p);auto&s=*w.s;
 auto*ap=alloc((size_t)s.mpad*KC);auto*bp=alloc(KC*NT);
 for(int iteration=0;iteration<s.warm+s.repeats;++iteration) {
  pthread_barrier_wait(&s.barrier);
  if(w.id==0&&iteration>=s.warm) {
   printf("MLP_ROI begin epoch=%d\n",iteration-s.warm);
   m5_reset_stats(0,0);m5_work_begin(0,0);
  }
  pthread_barrier_wait(&s.barrier);
#if MLP_COMBINED
  if(iteration==0) {
   for(int r=w.id;r<s.d;r+=Threads) {
    std::copy_n(s.wg+(size_t)r*s.f,s.f,s.wgu+(size_t)r*2*s.f);
    std::copy_n(s.wu+(size_t)r*s.f,s.f,s.wgu+(size_t)r*2*s.f+s.f);
   }
  }
  pthread_barrier_wait(&s.barrier);
#endif
#ifdef IDENTICAL_CACHE_BASELINE
  if(s.prepack&&iteration==0) {
#if MLP_COMBINED
   pack_weights(w,s.wgu,s.pgu,s.d,2*s.f);
#else
   pack_weights(w,s.wg,s.pg,s.d,s.f);pack_weights(w,s.wu,s.pu,s.d,s.f);
#endif
   pack_weights(w,s.wd,s.pd,s.f,s.d);
  }
#endif
  pthread_barrier_wait(&s.barrier);
  const __fp16 *wg=s.wg,*wu=s.wu,*wd=s.wd;
#ifdef IDENTICAL_CACHE_BASELINE
  if(s.prepack){wg=s.pg;wu=s.pu;wd=s.pd;}
#endif
#if MLP_COMBINED
  const __fp16* wgu=s.wgu;
#ifdef IDENTICAL_CACHE_BASELINE
  if(s.prepack)wgu=s.pgu;
#endif
  gemm(w,s.x,wgu,s.gu,s.d,2*s.f,ap,bp);
#else
  gemm(w,s.x,wg,s.gate,s.d,s.f,ap,bp);
  gemm(w,s.x,wu,s.up,s.d,s.f,ap,bp);
#endif
  pthread_barrier_wait(&s.barrier);
#if MLP_COMBINED
  for(int r=s.mpad*w.id/Threads;r<s.mpad*(w.id+1)/Threads;++r)
   mlp_swiglu(s.gu+(size_t)r*2*s.f,s.gu+(size_t)r*2*s.f+s.f,s.f);
#else
  size_t begin=(size_t)s.mpad*s.f*w.id/Threads,end=(size_t)s.mpad*s.f*(w.id+1)/Threads;
  mlp_swiglu(s.gate+begin,s.up+begin,end-begin);
#endif
  pthread_barrier_wait(&s.barrier);
#if MLP_COMBINED
  gemm(w,s.gu,wd,s.y,s.f,s.d,ap,bp,2*s.f);
#else
  gemm(w,s.gate,wd,s.y,s.f,s.d,ap,bp);
#endif
  pthread_barrier_wait(&s.barrier);
  if(w.id==0) {
   if(iteration>=s.warm){m5_work_end(0,0);m5_dump_stats(0,0);puts("MLP_ROI end");}
   s.total_errors+=check(s,iteration);
  }
  pthread_barrier_wait(&s.barrier);
 }
 std::free(ap);std::free(bp);return nullptr;
}
}
int main(int argc,char**argv) {
 setbuf(stdout,nullptr);
 if(argc!=10)return 2;
 Shared s{};s.t=atoi(argv[1]);s.d=atoi(argv[2]);s.f=atoi(argv[3]);s.mp=atoi(argv[4]);KC=atoi(argv[5]);
 s.warm=atoi(argv[6]);s.repeats=atoi(argv[7]);s.prepack=atoi(argv[8]);Threads=atoi(argv[9]);
 NT=svcnth();s.mpad=(s.t+TileM-1)/TileM*TileM;
 if(s.t<=0||s.d<=0||s.f<=0||s.d%NT||s.f%NT||s.mp<1||Threads<1||Threads>MaxThreads||Threads%s.mp||KC<1||KC>1024||s.warm<0||s.repeats<1)return 2;
 printf("MLP_CONFIG T=%d D=%d F=%d MP=%d KC=%d warm=%d repeats=%d prepack=%d threads=%d NT=%d\n",s.t,s.d,s.f,s.mp,KC,s.warm,s.repeats,s.prepack,Threads,NT);
 s.x=alloc((size_t)s.mpad*s.d);s.wg=alloc((size_t)s.d*s.f);s.wu=alloc((size_t)s.d*s.f);s.wd=alloc((size_t)s.f*s.d);
 s.gate=alloc((size_t)s.mpad*s.f);s.up=alloc((size_t)s.mpad*s.f);s.y=alloc((size_t)s.mpad*s.d);s.ref=alloc((size_t)s.t*s.d);
 uint64_t offset=0;
 for(auto item:std::vector<std::pair<__fp16*,size_t>>{{s.x,(size_t)s.t*s.d},{s.wg,(size_t)s.d*s.f},{s.wu,(size_t)s.d*s.f},{s.wd,(size_t)s.f*s.d},{s.ref,(size_t)s.t*s.d}}) {
  size_t bytes=item.second*2;if(m5_read_file(item.first,bytes,offset)!=bytes)return 3;offset+=bytes;
 }
 s.pg=alloc((size_t)s.d*s.f);s.pu=alloc((size_t)s.d*s.f);s.pd=alloc((size_t)s.f*s.d);
#if MLP_ACC32
 s.acc=static_cast<float*>(std::aligned_alloc(64,(size_t)s.mpad*std::max(s.d,2*s.f)*sizeof(float)));
 if(!s.acc)return 5;
 std::memset(s.acc,0,(size_t)s.mpad*std::max(s.d,2*s.f)*sizeof(float));
#endif
#if MLP_COMBINED
 s.wgu=alloc((size_t)s.d*2*s.f);s.pgu=alloc((size_t)s.d*2*s.f);s.gu=alloc((size_t)s.mpad*2*s.f);
#endif
 pthread_barrier_init(&s.barrier,nullptr,Threads);
 pthread_t th[Threads];Worker w[Threads];
 for(int i=1;i<Threads;++i){w[i]={&s,i};if(pthread_create(&th[i],nullptr,worker,&w[i]))return 4;}
 w[0]={&s,0};worker(&w[0]);for(int i=1;i<Threads;++i)pthread_join(th[i],nullptr);
 if(m5_write_file(s.y,(size_t)s.t*s.d*2,0,"output.bin")!=(size_t)s.t*s.d*2)return 6;
 return s.total_errors?1:0;
}
