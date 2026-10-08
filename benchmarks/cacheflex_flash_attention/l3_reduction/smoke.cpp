#include "reduction.hpp"
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <gem5/m5ops.h>
constexpr int C=8,R=16,N=8;
static pthread_barrier_t barrier;
static std::atomic<unsigned> errors{0};
alignas(64) static float destination[C][64];
// Low mantissa bits intentionally nonzero: catches scalar-header truncation.
static float scale(){uint32_t bits=0x3f400003;float x;std::memcpy(&x,&bits,4);return x;}
static float step(float x,float a,float b){volatile float product=x*a;return product+b;}
static float expected(int row,int lane) {
    float x=row+lane/128.f;
    for(int n=0;n<N;++n)x=step(x,scale(),n/4.f+row/16.f);
    return x;
}
static void verify(uint64_t slot,int row) {
    alignas(64) float v[64];svst1_f32(svptrue_b32(),v,shared_load(slot));
    for(int i=0;i<64;++i)if(v[i]!=expected(row,i))++errors;
}
static void* worker(void* p) {
    unsigned id=(uintptr_t)p;alignas(64) float v[64];
    for(int r=0;r<R;++r){for(int i=0;i<64;++i)v[i]=r+i/128.f;
        shared_store(shared_slot(id,r),svld1_f32(svptrue_b32(),v));}
    shared_drain();pthread_barrier_wait(&barrier);
    if(!id){m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&barrier);
    for(int n=0;n<N;++n)for(int r=0;r<R;++r)
        shared_reduce(shared_slot(id,r),scale(),svdup_n_f32(n/4.f+r/16.f));
    // No explicit fence: final-value loads must wait for posted updates.
    for(int r=0;r<R;++r)verify(shared_slot(id,r),r);
    shared_drain();pthread_barrier_wait(&barrier);
    for(int r=0;r<R;++r)verify(shared_slot((id+1)%C,r),r);
    shared_reduce(shared_slot(id,0),scale(),svdup_n_f32(.125f));
    for(int w=0;w<4;++w)shared_writeback(shared_slot(id,0,w),destination[id]+16*w);
    for(int i=0;i<64;++i)if(destination[id][i]!=step(expected(0,i),scale(),.125f))++errors;
    shared_reduce(shared_slot(id,1),1.f,svdup_n_f32(.25f));
    shared_store(shared_slot(id,1),svdup_n_f32(77.f));
    svst1_f32(svptrue_b32(),v,shared_load(shared_slot(id,1)));
    for(float x:v)if(x!=77.f)++errors;
    shared_drain();pthread_barrier_wait(&barrier);
    for(int r=0;r<R;++r)for(int w=0;w<4;++w)shared_release(shared_slot(id,r,w));
    shared_drain();
    shared_store(shared_slot(id,0),svdup_n_f32(0.f));
    svst1_f32(svptrue_b32(),v,shared_load(shared_slot(id,0)));
    for(float x:v)if(x!=0.f)++errors;
    shared_drain();pthread_barrier_wait(&barrier);
    if(!id){m5_work_end(0,0);m5_dump_stats(0,0);}
    return nullptr;
}
int main(){
    setvbuf(stdout,nullptr,_IONBF,0);if(svcntw()!=64)return 2;
    pthread_barrier_init(&barrier,nullptr,C);pthread_t threads[C];
    for(unsigned i=1;i<C;++i)if(pthread_create(threads+i,nullptr,worker,(void*)(uintptr_t)i))return 3;
    worker(nullptr);for(int i=1;i<C;++i)pthread_join(threads[i],nullptr);
    printf("L3_REDUCTION_SMOKE %s errors=%u commands=%d\n",errors.load()?"FAIL":"PASS",errors.load(),C*(R*N+2));
    return errors.load()?1:0;
}
