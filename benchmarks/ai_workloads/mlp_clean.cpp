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
constexpr int TileM=24;
#include "mlp_helpers.hpp"

namespace {
constexpr int Threads=8, KC=256, NT=128;
struct Shared {
 int t,mpad,d,f,mp;
 __fp16 *x,*wg,*wu,*wd,*gate,*up,*y,*ref;
 pthread_barrier_t barrier;
};
struct Worker {Shared* s; int id;};
__fp16* alloc(size_t n) {
 auto* p=static_cast<__fp16*>(std::aligned_alloc(64,((n*2+63)/64)*64));
 if(!p) std::abort(); std::memset(p,0,n*2); return p;
}
void gemm(Worker&w,const __fp16*a,const __fp16*b,__fp16*c,int k,int n,
          __fp16*ap,__fp16*bp) {
 auto&s=*w.s;int np=Threads/s.mp,m=w.id/np,nn=w.id%np;
 int mb0=(s.mpad/24)*m/s.mp,mb1=(s.mpad/24)*(m+1)/s.mp;
 if(mb0==mb1) return;
 for(int k0=0;k0<k;k0+=KC) {
  int kc=std::min(KC,k-k0);
  packA24Neon(a,k,ap,s.mpad,mb0*24,(mb1-mb0)*24,k0,kc);
  for(int nb=(n/NT)*nn/np;nb<(n/NT)*(nn+1)/np;++nb) {
#ifdef IDENTICAL_CACHE_BASELINE
   for(int r=0;r<kc;++r)std::copy_n(b+(size_t)(k0+r)*n+nb*NT,NT,bp+(size_t)r*NT);
   cache_gemm_fused_24x1VL(ap,bp,c+(size_t)mb0*24*n+nb*NT,mb1-mb0,kc,n,k0==0);
#else
   pack_B_tile_to_spm_4way(b+(size_t)k0*n+nb*NT,n,kc,KC,0);
   spm_gemm_fused_tile(ap,c+(size_t)mb0*24*n+nb*NT,mb1-mb0,kc,n,k0==0,0);
#endif
  }
 }
}
void* worker(void*p) {
 auto&w=*static_cast<Worker*>(p);auto&s=*w.s;
 auto*ap=alloc((size_t)s.mpad*KC);auto*bp=alloc(KC*NT);
 pthread_barrier_wait(&s.barrier);
 if(w.id==0){puts("MLP_ROI begin");m5_reset_stats(0,0);m5_work_begin(0,0);}
 pthread_barrier_wait(&s.barrier);
 gemm(w,s.x,s.wg,s.gate,s.d,s.f,ap,bp);
 gemm(w,s.x,s.wu,s.up,s.d,s.f,ap,bp);
 pthread_barrier_wait(&s.barrier);
 size_t begin=(size_t)s.mpad*s.f*w.id/Threads,end=(size_t)s.mpad*s.f*(w.id+1)/Threads;
 mlp_swiglu(s.gate+begin,s.up+begin,end-begin);
 pthread_barrier_wait(&s.barrier);
 gemm(w,s.gate,s.wd,s.y,s.f,s.d,ap,bp);
 pthread_barrier_wait(&s.barrier);
 if(w.id==0){m5_work_end(0,0);m5_dump_stats(0,0);puts("MLP_ROI end");}
 std::free(ap);std::free(bp);return nullptr;
}
}
int main(int argc,char**argv) {
 setbuf(stdout,nullptr);
 if(argc!=5||svcnth()!=128)return 2;
 Shared s{};s.t=atoi(argv[1]);s.d=atoi(argv[2]);s.f=atoi(argv[3]);s.mp=atoi(argv[4]);s.mpad=(s.t+23)/24*24;
 if(s.t<=0||s.d<=0||s.f<=0||s.d%128||s.f%128||(s.mp!=2&&s.mp!=4))return 2;
 s.x=alloc((size_t)s.mpad*s.d);s.wg=alloc((size_t)s.d*s.f);s.wu=alloc((size_t)s.d*s.f);s.wd=alloc((size_t)s.f*s.d);
 s.gate=alloc((size_t)s.mpad*s.f);s.up=alloc((size_t)s.mpad*s.f);s.y=alloc((size_t)s.mpad*s.d);s.ref=alloc((size_t)s.t*s.d);
 uint64_t offset=0;
 for(auto item:std::vector<std::pair<__fp16*,size_t>>{{s.x,(size_t)s.t*s.d},{s.wg,(size_t)s.d*s.f},{s.wu,(size_t)s.d*s.f},{s.wd,(size_t)s.f*s.d},{s.ref,(size_t)s.t*s.d}}) {
  size_t bytes=item.second*2;if(m5_read_file(item.first,bytes,offset)!=bytes)return 3;offset+=bytes;
 }
 pthread_barrier_init(&s.barrier,nullptr,Threads);
 pthread_t th[Threads];Worker w[Threads];
 for(int i=1;i<Threads;++i){w[i]={&s,i};if(pthread_create(&th[i],nullptr,worker,&w[i]))return 4;}
 w[0]={&s,0};worker(&w[0]);for(int i=1;i<Threads;++i)pthread_join(th[i],nullptr);
 size_t errors=0,nonfinite=0;double maxerr=0,checksum=0;uint64_t hash=1469598103934665603ULL;
 for(size_t i=0;i<(size_t)s.t*s.d;++i) {
  float actual=s.y[i],ref=s.ref[i];double e=std::abs(actual-ref);maxerr=std::max(maxerr,e);
  nonfinite+=!std::isfinite(actual);errors+=!std::isfinite(actual)||e>0.000002+0.01*std::abs(ref);
  checksum+=actual;uint16_t bits;std::memcpy(&bits,s.y+i,2);hash=(hash^bits)*1099511628211ULL;
 }
 printf("MLP_RESULT T=%d D=%d F=%d MP=%d hash=%016llx checksum=%.12g max_abs=%.9g errors=%zu nonfinite=%zu %s\n",s.t,s.d,s.f,s.mp,(unsigned long long)hash,checksum,maxerr,errors,nonfinite,errors?"FAIL":"PASS");
 return errors?1:0;
}
