// Projection-family screen. The generated header reuses the validated GEMM
// primitives, with an explicit team size instead of a fixed eight-core team.
#include "screen_core.hpp"

namespace {
struct Screen {
 int m,k,n,groups,shared,team,mp,repeats,mpad;
 __fp16 *x,*weights,*packed,*y,*reference;
 pthread_barrier_t barrier;
 size_t errors=0;
};
struct Task { Screen* s; int id; };

size_t validate(Screen&s,int iteration) {
 size_t errors=0,nonfinite=0; double maxerr=0;
 uint64_t hash=1469598103934665603ULL;
 for(int g=0;g<s.groups;++g)for(int r=0;r<s.m;++r)for(int c=0;c<s.n;++c) {
  auto value=s.y[((size_t)g*s.mpad+r)*s.n+c];
  float ref=s.reference[((size_t)g*s.m+r)*s.n+c];
  double e=std::abs(float(value)-ref);maxerr=std::max(maxerr,e);
  nonfinite+=!std::isfinite(float(value));
  errors+=!std::isfinite(float(value))||e>0.000002+0.01*std::abs(ref);
  uint16_t bits;memcpy(&bits,&value,2);hash=(hash^bits)*1099511628211ULL;
 }
 printf("SCREEN_RESULT iteration=%d hash=%016llx errors=%zu nonfinite=%zu max_abs=%.9g %s\n",
  iteration,(unsigned long long)hash,errors,nonfinite,maxerr,errors?"FAIL":"PASS");
 return errors;
}

void* screen_worker(void*ptr) {
 auto&t=*static_cast<Task*>(ptr);auto&s=*t.s;
 Shared local{};local.mpad=s.mpad;local.mp=s.mp;local.prepack=1;
 Worker w{&local,t.id%s.team};
 auto*ap=alloc((size_t)s.mpad*KC);auto*bp=alloc(KC*NT);
 for(int iter=0;iter<s.repeats;++iter) {
  pthread_barrier_wait(&s.barrier);
  if(t.id==0){puts("SCREEN_ROI begin");m5_reset_stats(0,0);m5_work_begin(0,0);}
  pthread_barrier_wait(&s.barrier);
#ifdef IDENTICAL_CACHE_BASELINE
  if(iter==0) {
   Worker packer{&local,t.id};
   for(int g=0;g<(s.shared?1:s.groups);++g)
    pack_weights(packer,s.weights+(size_t)g*s.k*s.n,s.packed+(size_t)g*s.k*s.n,s.k,s.n);
  }
#endif
  pthread_barrier_wait(&s.barrier);
  for(int g=t.id/s.team;g<s.groups;g+=Threads/s.team) {
   int wg=s.shared?0:g;
#ifdef IDENTICAL_CACHE_BASELINE
   const auto*b=s.packed+(size_t)wg*s.k*s.n;
#else
   const auto*b=s.weights+(size_t)wg*s.k*s.n;
#endif
   gemm(w,s.x+(size_t)g*s.mpad*s.k,b,s.y+(size_t)g*s.mpad*s.n,s.k,s.n,ap,bp,0,s.team);
  }
  pthread_barrier_wait(&s.barrier);
  if(t.id==0){m5_work_end(0,0);m5_dump_stats(0,0);puts("SCREEN_ROI end");s.errors+=validate(s,iter);}
  pthread_barrier_wait(&s.barrier);
 }
 std::free(ap);std::free(bp);return nullptr;
}
}

int main(int argc,char**argv) {
 setbuf(stdout,nullptr);if(argc!=11)return 2;
 Screen s{};s.m=atoi(argv[1]);s.k=atoi(argv[2]);s.n=atoi(argv[3]);s.groups=atoi(argv[4]);
 s.shared=atoi(argv[5]);s.team=atoi(argv[6]);s.mp=atoi(argv[7]);KC=atoi(argv[8]);s.repeats=atoi(argv[9]);
 Threads=atoi(argv[10]);NT=svcnth();s.mpad=(s.m+TileM-1)/TileM*TileM;
 if(Threads!=8||NT!=128||s.m<1||s.k<1||s.n<1||s.groups<1||s.team<1||8%s.team||s.mp<1||s.team%s.mp||
    s.k%128||s.n%128||KC<1||KC>512||s.repeats<1||(s.shared!=0&&s.shared!=1))return 2;
 s.x=alloc((size_t)s.groups*s.mpad*s.k);s.y=alloc((size_t)s.groups*s.mpad*s.n);
 s.weights=alloc((size_t)(s.shared?1:s.groups)*s.k*s.n);
 s.packed=alloc((size_t)(s.shared?1:s.groups)*s.k*s.n);
 s.reference=alloc((size_t)s.groups*s.m*s.n);
 uint64_t off=0;
 for(int g=0;g<s.groups;++g){size_t bytes=(size_t)s.m*s.k*2;
  if(m5_read_file(s.x+(size_t)g*s.mpad*s.k,bytes,off)!=bytes)return 3;off+=bytes;}
 for(auto item:std::vector<std::pair<__fp16*,size_t>>{{s.weights,(size_t)(s.shared?1:s.groups)*s.k*s.n},{s.reference,(size_t)s.groups*s.m*s.n}}){
  size_t bytes=item.second*2;if(m5_read_file(item.first,bytes,off)!=bytes)return 3;off+=bytes;}
 pthread_barrier_init(&s.barrier,nullptr,Threads);pthread_t threads[8];Task tasks[8];
 for(int i=1;i<8;++i){tasks[i]={&s,i};if(pthread_create(&threads[i],nullptr,screen_worker,&tasks[i]))return 4;}
 tasks[0]={&s,0};screen_worker(&tasks[0]);for(int i=1;i<8;++i)pthread_join(threads[i],nullptr);
 // Export via timing-loaded, drained 64B stack chunks, not Ruby bulk reads.
 uint64_t offset=0;alignas(64) uint16_t scratch[32];
 for(int g=0;g<s.groups;++g)for(size_t j=0;j<(size_t)s.m*s.n;j+=32){
  auto*src=reinterpret_cast<volatile uint16_t*>(s.y+(size_t)g*s.mpad*s.n+j);
  for(int z=0;z<32;++z)scratch[z]=src[z];asm volatile("dsb sy":::"memory");
  if(m5_write_file(scratch,64,offset,"output.bin")!=64)return 5;offset+=64;
 }
 return s.errors?1:0;
}
