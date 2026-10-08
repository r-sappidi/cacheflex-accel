#include "shared_spm.hpp"
#include <pthread.h>
#include <gem5/m5ops.h>
#include <cstdio>
#include <cstdlib>
static constexpr unsigned R=16,N=256;
alignas(256) static uint64_t backing[8][R][32];
static pthread_barrier_t barrier;
static unsigned cores=8,mode=2;
static unsigned errors[8]={};
static void* worker(void* raw) {
    unsigned id=(uintptr_t)raw;
    for(unsigned r=0;r<R;++r) {
        uint64_t next=mode==0?(uint64_t)backing[id][(r+1)%R]:
            (mode==1?uint64_t((r+1)%R)<<6:shared_slot(id,(r+1)%R));
        for(auto&v:backing[id][r])v=next;
        if(mode==1)for(unsigned w=0;w<4;++w)shared_copy((uint64_t(w)<<16)|(r<<6),backing[id][r]+8*w);
        if(mode==2) {
            auto p=svptrue_b64();auto value=svdup_n_u64(next);auto a=shared_slot(id,r);
            asm volatile("spm.st1d %0.d, %1, [%2]\n"::"w"(value),"Upl"(p),"r"(a):"memory");
        }
    }
    shared_drain();pthread_barrier_wait(&barrier);
    if(id==0){m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&barrier);
    uint64_t a=mode==0?(uint64_t)backing[id][0]:(mode==1?0:shared_slot(id,0));
    const uint64_t start=a;
    for(unsigned i=0;i<N;++i) {
        auto p=svptrue_b64();svuint64_t v;
        if(mode==0)v=svld1_u64(p,(const uint64_t*)a);
        else asm volatile("spm.ld1d %0.d, %1/z, [%2]\n":"=w"(v):"Upl"(p),"r"(a):"memory");
        a=svlastb_u64(p,v);
    }
    errors[id]=a!=start;
    pthread_barrier_wait(&barrier);
    if(id==0){m5_work_end(0,0);m5_dump_stats(0,0);}
    return nullptr;
}
int main(int argc,char**argv) {
    if(argc!=3||svcntd()!=32)return 2;
    mode=atoi(argv[1]);cores=atoi(argv[2]);if(mode>2||(cores!=1&&cores!=8))return 2;
    pthread_barrier_init(&barrier,nullptr,cores);pthread_t ts[8];
    for(unsigned i=1;i<cores;++i)pthread_create(ts+i,nullptr,worker,(void*)(uintptr_t)i);
    worker(nullptr);for(unsigned i=1;i<cores;++i)pthread_join(ts[i],nullptr);
    unsigned e=0;for(auto x:errors)e+=x;
    printf("L3_LATENCY %s mode=%u workers=%u dependent_loads=%u errors=%u\n",e?"FAIL":"PASS",mode,cores,N*cores,e);
    return e?1:0;
}
