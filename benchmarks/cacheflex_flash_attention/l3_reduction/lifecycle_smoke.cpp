#include "reduction.hpp"
#include "lifecycle.hpp"
#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <gem5/m5ops.h>
constexpr unsigned C=8,R=8,N=4,E=3;
static unsigned mode=3;
static pthread_barrier_t barrier;
static std::atomic<unsigned> errors{0};
alignas(128) static float16_t output[C][R][64];
static float scale(){uint32_t b=0x3f400003;float v;std::memcpy(&v,&b,4);return v;}
static float value(unsigned e,unsigned n,unsigned r,unsigned i){return float(float16_t(e/8.f+n/4.f+r/2.f+i/32.f));}
static float expected(unsigned e,unsigned r,unsigned i){
    float x=0;for(unsigned n=0;n<N;++n){volatile float p=x*(n?scale():0.f);x=p+value(e,n,r,i);}return x;
}
static float reciprocal(unsigned r){return 1.f/(3.f+r);}
static void verify(unsigned id,unsigned e){
    volatile float16_t* p=&output[id][0][0];
    for(unsigned r=0;r<R;++r)for(unsigned i=0;i<64;++i){
        volatile float result=expected(e,r,i)*reciprocal(r);
        if(p[r*64+i]!=float16_t(result))++errors;
    }
}
static void* worker(void* arg){
    unsigned id=(uintptr_t)arg;shared_output_bank=id;
    alignas(64) float16_t half[64];alignas(64) float read[64];
    pthread_barrier_wait(&barrier);
    if(!id){m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&barrier);
    for(unsigned e=0;e<E;++e){
        // Make the output dirty and read-shared; writeback must invalidate both.
        for(unsigned r=0;r<R;++r)for(unsigned i=0;i<64;++i)output[id][r][i]=-7;
        pthread_barrier_wait(&barrier);
        volatile float16_t prior=output[(id+1)%C][0][0];if(prior!=-7)++errors;
        pthread_barrier_wait(&barrier);
        if(!(mode&1))output_claim(nullptr,R);
        for(unsigned n=0;n<N;++n)for(unsigned r=0;r<R;++r){
            for(unsigned i=0;i<64;++i)half[i]=value(e,n,r,i);
            auto v=svld1_f16(svwhilelt_b16_u64(0,64),half);
            if((mode&1)&&n==0)output_reduce_initialize(r,0.f,v);
            else output_reduce_half(r,n?scale():0.f,v);
        }
        if(mode&2){
            for(unsigned r=0;r<R;++r)output_finalize(r,reciprocal(r),output[id][r]);
            // No explicit fence before dependent regular loads: shadow records
            // must protect each full64B coherent destination until installation.
            verify(id,e);
            // The second source slot also belongs to finalization's footprint.
            svst1_f32(svptrue_b32(),read,shared_load(shared_slot(id,0,1)));
            for(float x:read)if(x!=0.f)++errors;
        }else{
            for(unsigned r=0;r<R;++r){
                svst1_f32(svptrue_b32(),read,output_load(r));
                for(unsigned i=0;i<64;++i){
                    if(read[i]!=expected(e,r,i))++errors;
                    volatile float result=read[i]*reciprocal(r);output[id][r][i]=float16_t(result);
                }
            }
            output_release(R);
        }
        shared_drain();pthread_barrier_wait(&barrier);
        verify((id+1)%C,e); // Coherent observer after publish.
        pthread_barrier_wait(&barrier);
        // Next epoch reuses exactly the same slots, including lazy first claim.
    }
    if(!id){m5_work_end(0,0);m5_dump_stats(0,0);}
    return nullptr;
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IONBF,0);if(svcntw()!=64)return 2;
    if(argc>1)mode=atoi(argv[1]);if(mode<1||mode>3)return 3;
    pthread_barrier_init(&barrier,nullptr,C);pthread_t threads[C];
    for(unsigned i=1;i<C;++i)if(pthread_create(threads+i,nullptr,worker,(void*)(uintptr_t)i))return 4;
    worker(nullptr);for(unsigned i=1;i<C;++i)pthread_join(threads[i],nullptr);
    printf("L3_LIFECYCLE_SMOKE %s errors=%u commands=%u initializations=%u finalizations=%u\n",
        errors.load()?"FAIL":"PASS",errors.load(),C*R*N*E,(mode&1)?C*R*E:0,(mode&2)?2*C*R*E:0);
    return errors.load()?1:0;
}
