// AI-12: Qwen-shaped remainder. Same ELF on cache/private-SPM hardware.
// Streaming operands use coherent loads; no artificial staging or extra SRAM.
#include <arm_sve.h>
#include <gem5/m5ops.h>
#include <pthread.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int D=896;
struct State {int op,t,v,b; uint64_t n[5]; __fp16 *x,*w,*z,*out,*ref; uint32_t* ids; pthread_barrier_t barrier; int errors=0;};
struct Job {State*s;int id;};
void* alloc(size_t bytes) {auto*p=std::aligned_alloc(64,(bytes+63)/64*64);if(!p)std::exit(3);memset(p,0,bytes);return p;}
// Widen each halfword into the low half of its FP32 lane before FCVT.
svfloat32_t load(svbool_t pg,const __fp16*p) {return svcvt_f32_f16_x(pg,svreinterpret_f16_u32(svld1uh_u32(pg,(const uint16_t*)p)));}
void store(svbool_t pg,__fp16*p,svfloat32_t x) {svst1h_u32(pg,(uint16_t*)p,svreinterpret_u32_f16(svcvt_f16_f32_x(pg,x)));}
svfloat32_t roundhalf(svbool_t pg,svfloat32_t x) {return svcvt_f32_f16_x(pg,svcvt_f16_f32_x(pg,x));}
void norm(State&s,int r) {
 auto pg=svptrue_b32();auto a=svdup_f32(0);
 for(int k=0;k<D;k+=svcntw()){auto x=load(pg,s.x+(size_t)r*D+k);a=svadd_f32_x(pg,a,svmul_f32_x(pg,x,x));}
 float scale=1.f/sqrtf(svaddv_f32(pg,a)/D+1e-6f);
 for(int k=0;k<D;k+=svcntw()){
  auto x=roundhalf(pg,svmul_n_f32_x(pg,load(pg,s.x+(size_t)r*D+k),scale));
  store(pg,s.out+(size_t)r*D+k,svmul_f32_x(pg,x,load(pg,s.w+k)));
 }
}
template<int B> void logits(State&s,int row,int end) {
 auto pg=svptrue_b32();auto a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
 for(int k=0;k<D;k+=svcntw()){
  auto x=load(pg,s.x+k);
  a0=svadd_f32_x(pg,a0,svmul_f32_x(pg,x,load(pg,s.w+(size_t)row*D+k)));
  if constexpr(B>=2)if(row+1<end)a1=svadd_f32_x(pg,a1,svmul_f32_x(pg,x,load(pg,s.w+(size_t)(row+1)*D+k)));
  if constexpr(B>=4){
   if(row+2<end)a2=svadd_f32_x(pg,a2,svmul_f32_x(pg,x,load(pg,s.w+(size_t)(row+2)*D+k)));
   if(row+3<end)a3=svadd_f32_x(pg,a3,svmul_f32_x(pg,x,load(pg,s.w+(size_t)(row+3)*D+k)));
  }
 }
 s.out[row]=svaddv_f32(pg,a0);
 if constexpr(B>=2)if(row+1<end)s.out[row+1]=svaddv_f32(pg,a1);
 if constexpr(B>=4){if(row+2<end)s.out[row+2]=svaddv_f32(pg,a2);if(row+3<end)s.out[row+3]=svaddv_f32(pg,a3);}
}
void execute(State&s,int id) {
 int rows=s.op==6?s.v:s.t,lo=rows*id/8,hi=rows*(id+1)/8;
 for(int rb=lo;rb<hi;rb+=s.b){
  if(s.op==6){if(s.b==1)logits<1>(s,rb,hi);else if(s.b==2)logits<2>(s,rb,hi);else logits<4>(s,rb,hi);continue;}
  for(int r=rb;r<std::min(hi,rb+s.b);++r){
   if(s.op==0||s.op==5){norm(s,r);continue;}
   if(s.op==1||s.op==4){
    for(int k=0;k<D;k+=svcnth()){
     auto pg=svptrue_b16();auto x=svld1_f16(pg,s.x+(s.op==4?(size_t)s.ids[r]*D:(size_t)r*D)+k);
     if(s.op==1)x=svadd_f16_x(pg,x,svld1_f16(pg,s.w+(size_t)r*D+k));
     svst1_f16(pg,s.out+(size_t)r*D+k,x);
    }continue;
   }
   if(s.op==3){ // Tables are shared by all layers; generation is timed separately.
    for(int k=0;k<32;++k){float frequency;memcpy(&frequency,s.ids+k,4);float a=(float)r*frequency;s.out[(size_t)r*64+k]=cosf(a);s.out[(size_t)r*64+32+k]=sinf(a);}continue;
   }
   // Q14 + K2, head-major; V is deliberately untouched. Split-half rotary.
   auto pg=svwhilelt_b32((uint64_t)0,(uint64_t)32);
   auto c=load(pg,s.z+(size_t)r*64),sn=load(pg,s.z+(size_t)r*64+32);
   for(int h=0;h<16;++h){size_t off=((size_t)h*s.t+r)*64;
    auto a=load(pg,s.x+off),b=load(pg,s.x+off+32);
    auto ac=roundhalf(pg,svmul_f32_x(pg,a,c)),bs=roundhalf(pg,svmul_f32_x(pg,b,sn));
    auto bc=roundhalf(pg,svmul_f32_x(pg,b,c)),as=roundhalf(pg,svmul_f32_x(pg,a,sn));
    store(pg,s.out+off,svsub_f32_x(pg,ac,bs));store(pg,s.out+off+32,svadd_f32_x(pg,bc,as));
   }
  }
 }
}
void* worker(void*arg){auto&j=*(Job*)arg;auto&s=*j.s;
 for(int it=0;it<2;++it){pthread_barrier_wait(&s.barrier);
  if(j.id==0){printf("REMAINDER_ROI begin epoch=%d\n",it);m5_reset_stats(0,0);m5_work_begin(0,0);}
  pthread_barrier_wait(&s.barrier);execute(s,j.id);pthread_barrier_wait(&s.barrier);
  if(j.id==0){m5_work_end(0,0);m5_dump_stats(0,0);uint64_t hash=1469598103934665603ULL;size_t nonfinite=0;
   for(size_t i=0;i<s.n[4];++i){uint16_t u;memcpy(&u,s.out+i,2);hash=(hash^u)*1099511628211ULL;nonfinite+=!std::isfinite((float)s.out[i]);}
   printf("REMAINDER_RESULT iteration=%d hash=%016llx nonfinite=%zu\n",it,(unsigned long long)hash,nonfinite);s.errors+=nonfinite;
  }pthread_barrier_wait(&s.barrier);
 }return nullptr;
}
}
int main(int argc,char**argv){setbuf(stdout,nullptr);if(argc!=5)return 2;State s{};s.op=atoi(argv[1]);s.t=atoi(argv[2]);s.v=atoi(argv[3]);s.b=atoi(argv[4]);
 if(s.op<0||s.op>6||s.t<1||s.t>32768||s.v<1||s.v>151936||s.b<1||svcntw()!=64||(s.op==6&&s.b!=1&&s.b!=2&&s.b!=4))return 2;
 if(m5_read_file(s.n,sizeof(s.n),0)!=sizeof(s.n))return 4;
 uint64_t expected=s.op==6?s.v:s.op==2?(uint64_t)s.t*1024:s.op==3?(uint64_t)s.t*64:(uint64_t)s.t*896;
 if(s.n[4]!=expected)return 4;
 s.x=(__fp16*)alloc(s.n[0]*2);s.w=(__fp16*)alloc(s.n[1]*2);s.z=(__fp16*)alloc(s.n[2]*2);s.ids=(uint32_t*)alloc(s.n[3]*4);s.out=(__fp16*)alloc(s.n[4]*2);
 void* arrays[]={s.x,s.w,s.z,s.ids};uint64_t off=sizeof(s.n);
 for(int i=0;i<4;++i){size_t bytes=s.n[i]*(i==3?4:2);if(bytes&&m5_read_file(arrays[i],bytes,off)!=bytes)return 4;off+=bytes;}
 if(s.op==4)for(int i=0;i<s.t;++i)if(s.ids[i]>=(uint32_t)s.v)return 4;
 pthread_barrier_init(&s.barrier,nullptr,8);pthread_t threads[8];Job jobs[8];
 for(int i=1;i<8;++i){jobs[i]={&s,i};if(pthread_create(threads+i,nullptr,worker,jobs+i))return 5;}
 jobs[0]={&s,0};worker(jobs);for(int i=1;i<8;++i)pthread_join(threads[i],nullptr);
 alignas(64)uint16_t scratch[32];auto*src=(volatile uint16_t*)s.out;
 for(size_t i=0;i<s.n[4];i+=32){size_t count=std::min((uint64_t)32,s.n[4]-i);for(size_t z=0;z<count;++z)scratch[z]=src[i+z];asm volatile("dsb sy":::"memory");if(m5_write_file(scratch,count*2,i*2,"output.bin")!=count*2)return 6;}
 return s.errors?1:0;
}
