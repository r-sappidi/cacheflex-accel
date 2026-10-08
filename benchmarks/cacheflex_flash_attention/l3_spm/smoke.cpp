#include "shared_spm.hpp"
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <gem5/m5ops.h>
static constexpr int C=8,R=16;
alignas(64) static float source[C][64],dest[C][64];
static pthread_barrier_t barrier;
static std::atomic<unsigned> errors{0};
static int rounds=4;
static void check(uint64_t a,float base,bool masked=false) {
    alignas(64) float out[64];svst1_f32(svptrue_b32(),out,shared_load(a));
    for(int i=0;i<64;++i)if(out[i]!=(masked&&i<32?999.f:base))++errors;
}
static void* worker(void* raw) {
    const unsigned id=(uintptr_t)raw,other=(id+1)%C;
    alignas(64) float out[64];
    // Pressure the ordinary hierarchy before lazy claims; source owner remains
    // dirty in the private hierarchy for the subsequent cross-core copy test.
    auto* pressure=(volatile uint64_t*)std::aligned_alloc(64,512*1024);
    if(!pressure)std::abort();
    for(unsigned i=0;i<512*1024/8;++i)pressure[i]=i+id;
    pthread_barrier_wait(&barrier);
    if(id==0){m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&barrier);
    check(shared_slot(id,R),0.f); // Unclaimed reads match private SPM semantics.
    for(int n=0;n<rounds;++n) {
        for(int i=0;i<64;++i){source[id][i]=float(id*100+i+n);dest[id][i]=-1.f;}
        for(unsigned r=0;r<R;++r)
            shared_store(shared_slot(id,r),svdup_n_f32(float(id*100+r+n)));
        // Immediate RAW and predicated WAW, no explicit intermediate fence.
        check(shared_slot(id,0),float(id*100+n));
        shared_store(shared_slot(id,0),svdup_n_f32(999),svwhilelt_b32_u64(0,32));
        check(shared_slot(id,0),float(id*100+n),true);
        shared_drain();pthread_barrier_wait(&barrier);
        for(unsigned r=0;r<R;++r)check(shared_slot(other,r),float(other*100+r+n),r==0);
        // Source is another core's dirty coherent line, destination is shared.
        for(unsigned w=0;w<4;++w)shared_copy(shared_slot(id,R,w),source[other]+16*w);
        svst1_f32(svptrue_b32(),out,shared_load(shared_slot(id,R)));
        for(int i=0;i<64;++i)if(out[i]!=float(other*100+i+n))++errors;
        for(unsigned w=0;w<4;++w)shared_writeback(shared_slot(id,R,w),dest[other]+16*w);
        // Coherent read must observe the pending bridge even without a DSB.
        for(int i=0;i<64;++i)if(dest[other][i]!=float(other*100+i+n))++errors;
        shared_drain();pthread_barrier_wait(&barrier);
        for(int i=0;i<64;++i)if(dest[id][i]!=float(id*100+i+n))++errors;
        for(unsigned r=0;r<=R;++r)for(unsigned w=0;w<4;++w)
            shared_release(shared_slot(id,r,w));
        shared_drain();pthread_barrier_wait(&barrier);
    }
    // Check ordinary dirty data survived any recalls/evictions used by claims.
    for(unsigned i=0;i<512*1024/8;++i)if(pressure[i]!=i+id)++errors;
    pthread_barrier_wait(&barrier);
    if(id==0){m5_work_end(0,0);m5_dump_stats(0,0);}
    std::free((void*)pressure);return nullptr;
}
int main(int argc,char**argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(svcntw()!=64)return 2;
    if(argc>1)rounds=atoi(argv[1]);
    pthread_barrier_init(&barrier,nullptr,C);
    pthread_t ts[C];for(unsigned i=1;i<C;++i)if(pthread_create(ts+i,nullptr,worker,(void*)(uintptr_t)i))return 3;
    worker(nullptr);for(int i=1;i<C;++i)pthread_join(ts[i],nullptr);
    printf("L3_SPM_SMOKE %s errors=%u rounds=%d\n",errors.load()?"FAIL":"PASS",errors.load(),rounds);
    return errors.load()?1:0;
}
