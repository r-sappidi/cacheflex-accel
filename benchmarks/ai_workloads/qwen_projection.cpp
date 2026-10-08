// Qwen projections: shared validated GEMM, timed bias and FA layout adapters.
#include "projection_core.hpp"
namespace {
struct Projection {
 Shared core{}; int mode,n,t,repeats; // mode0 QKV; mode1 output projection
 __fp16 *input,*weights,*packed,*bias,*row,*out,*reference;
 size_t errors=0;
};
struct Job {Projection*p;int id;};
size_t check_projection(Projection&p,int iter) {
 size_t errors=0,nonfinite=0;double maxerr=0;uint64_t h=1469598103934665603ULL;
 for(size_t i=0;i<(size_t)p.t*p.n;++i){float x=p.out[i],y=p.reference[i];double e=std::abs(x-y);maxerr=std::max(maxerr,e);
  nonfinite+=!std::isfinite(x);errors+=!std::isfinite(x)||e>2e-6+.01*std::abs(y);
  uint16_t b;memcpy(&b,p.out+i,2);h=(h^b)*1099511628211ULL;}
 printf("PROJ_RESULT iteration=%d hash=%016llx errors=%zu nonfinite=%zu max_abs=%.9g %s\n",iter,(unsigned long long)h,errors,nonfinite,maxerr,errors?"FAIL":"PASS");return errors;
}
void* projection_worker(void*ptr){auto&j=*static_cast<Job*>(ptr);auto&p=*j.p;auto&s=p.core;Worker w{&s,j.id};
 auto*ap=alloc((size_t)s.mpad*KC);auto*bp=alloc((size_t)KC*NT);
 for(int it=0;it<p.repeats;++it){
  pthread_barrier_wait(&s.barrier);
  if(j.id==0){printf("PROJ_ROI begin epoch=%d\n",it);m5_reset_stats(0,0);m5_work_begin(0,0);}
  pthread_barrier_wait(&s.barrier);
#ifdef IDENTICAL_CACHE_BASELINE
  if(it==0)pack_weights(w,p.weights,p.packed,896,p.n);
#endif
  // Output projection accepts exactly the head-major FA output layout.
  if(p.mode==1)for(int r=j.id;r<p.t;r+=Threads)for(int h=0;h<14;++h)
   std::copy_n(p.input+((size_t)h*p.t+r)*64,64,s.x+(size_t)r*896+h*64);
  pthread_barrier_wait(&s.barrier);
#ifdef IDENTICAL_CACHE_BASELINE
  const auto*b=p.packed;
#else
  const auto*b=p.weights;
#endif
  gemm(w,s.x,b,p.row,896,p.n,ap,bp);
  pthread_barrier_wait(&s.barrier);
  if(p.mode==0){
   // All18 heads serialized as Q14,K2,V2, each [T,64]. Read unrounded
   // FP32 accumulator so bias is added before the single output rounding.
   for(int h=j.id;h<18;h+=Threads)for(int r=0;r<p.t;++r)for(int d=0;d<64;d+=4){
    int col=h*64+d;auto v=vaddq_f32(vld1q_f32(s.acc+(size_t)r*p.n+col),vcvt_f32_f16(vld1_f16(p.bias+col)));
    vst1_f16(p.out+((size_t)h*p.t+r)*64+d,vcvt_f16_f32(v));
   }
  }
  pthread_barrier_wait(&s.barrier);
  if(j.id==0){m5_work_end(0,0);m5_dump_stats(0,0);puts("PROJ_ROI end");p.errors+=check_projection(p,it);}
  pthread_barrier_wait(&s.barrier);
 }
 free(ap);free(bp);return nullptr;
}
}
int main(int argc,char**argv){setbuf(stdout,nullptr);if(argc!=6)return 2;Projection p{};
 p.mode=atoi(argv[1]);p.t=atoi(argv[2]);p.core.mp=atoi(argv[3]);KC=atoi(argv[4]);p.repeats=atoi(argv[5]);
 NT=svcnth();Threads=8;p.n=p.mode==0?1152:896;
 if((p.mode!=0&&p.mode!=1)||p.t<1||p.t>32768||(p.core.mp!=1&&p.core.mp!=2&&p.core.mp!=4&&p.core.mp!=8)||KC<128||KC>1024||KC%128||NT!=128||p.repeats<1)return 2;
 auto&s=p.core;s.t=p.t;s.mpad=(p.t+7)/8*8;s.prepack=1;
 p.input=alloc((size_t)s.mpad*896);s.x=p.mode==0?p.input:alloc((size_t)s.mpad*896);
 p.weights=alloc((size_t)896*p.n);p.packed=alloc((size_t)896*p.n);p.bias=alloc(p.n);
 p.row=alloc((size_t)s.mpad*p.n);p.out=p.mode==0?alloc((size_t)p.t*p.n):p.row;p.reference=alloc((size_t)p.t*p.n);
 s.acc=static_cast<float*>(std::aligned_alloc(64,(size_t)s.mpad*p.n*4));if(!s.acc)return 3;memset(s.acc,0,(size_t)s.mpad*p.n*4);
 uint64_t off=0;for(auto a:std::vector<std::pair<__fp16*,size_t>>{{p.input,(size_t)p.t*896},{p.weights,(size_t)896*p.n},{p.bias,(size_t)p.n},{p.reference,(size_t)p.t*p.n}}){size_t bytes=a.second*2;if(m5_read_file(a.first,bytes,off)!=bytes)return 4;off+=bytes;}
 printf("PROJ_CONFIG mode=%d T=%d N=%d MP=%d KC=%d repeats=%d\n",p.mode,p.t,p.n,s.mp,KC,p.repeats);
 pthread_barrier_init(&s.barrier,nullptr,8);pthread_t th[8];Job j[8];
 for(int i=1;i<8;++i){j[i]={&p,i};if(pthread_create(th+i,nullptr,projection_worker,j+i))return 5;}
 j[0]={&p,0};projection_worker(j);for(int i=1;i<8;++i)pthread_join(th[i],nullptr);
 alignas(64)uint16_t scratch[32];auto*src=reinterpret_cast<volatile uint16_t*>(p.out);
 for(size_t i=0;i<(size_t)p.t*p.n;i+=32){for(int z=0;z<32;++z)scratch[z]=src[i+z];asm volatile("dsb sy":::"memory");if(m5_write_file(scratch,64,i*2,"output.bin")!=64)return 6;}
 return p.errors?1:0;
}
