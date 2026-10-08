#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <gem5/m5ops.h>
#include CACHEFLEX_FLASH_SEARCH_KERNELS
#ifndef RESIDENT_SPM
#define RESIDENT_SPM 0
#endif
static pthread_barrier_t barrier;
static unsigned errors[8];
static void* worker(void* arg) {
    const size_t tid=(size_t)arg;
    __fp16 *a,*b,*c;
    if(posix_memalign((void**)&a,64,64*64*2) ||
       posix_memalign((void**)&b,64,64*128*2) ||
       posix_memalign((void**)&c,64,64*128*2)) std::abort();
    for(int i=0;i<64*64;++i)a[i]=(__fp16)0.0625f;
    for(int k=0;k<64;++k)for(int j=0;j<128;++j)b[k*128+j]=(__fp16)(0.03125f*(1+j%3));
    for(int i=0;i<64*128;++i)c[i]=0;
    if(RESIDENT_SPM)for(size_t k=0;k<64;++k)for(size_t w=0;w<4;++w){
        uint64_t dst=(k<<6)|(w<<16);const __fp16* src=b+k*128+w*32;
        asm volatile("SPMCP_64_IMM %x[d], [%x[s], #0]"::[d]"r"(dst),[s]"r"(src):"memory");
    }
    auto kernel=[&](){for(int m=0;m<64;m+=8){
        if(RESIDENT_SPM)spm_kernel_1vl_vl16(a+m*64,c+m*128,64,128,64,0);
        else cache_qk_8x1VL_ldc(a+m*64,b,c+m*128,64,128);
    }};
    kernel(); // Warm cache / complete all SPM copies before measured region.
    pthread_barrier_wait(&barrier);
    if(!tid){std::puts("RESIDENT_ROI begin");m5_reset_stats(0,0);m5_work_begin(0,0);}
    pthread_barrier_wait(&barrier);
    for(int r=0;r<128;++r)kernel();
    pthread_barrier_wait(&barrier);
    if(!tid){m5_work_end(0,0);m5_dump_stats(0,0);std::puts("RESIDENT_ROI end");}
    for(int m=0;m<64;++m)for(int j=0;j<128;++j)
        errors[tid]+=(float)c[m*128+j]!=0.125f*(1+j%3);
    free(a);free(b);free(c);return nullptr;
}
int main(){
    if(svcnth()!=128)return 2;
    pthread_barrier_init(&barrier,nullptr,8);pthread_t threads[8];
    for(size_t i=1;i<8;++i)pthread_create(&threads[i],nullptr,worker,(void*)i);
    worker(nullptr);for(int i=1;i<8;++i)pthread_join(threads[i],nullptr);
    unsigned total=0;for(auto e:errors)total+=e;
    std::printf("RESIDENT_RESULT errors=%u spm=%d\n",total,RESIDENT_SPM);
    return total?1:0;
}
