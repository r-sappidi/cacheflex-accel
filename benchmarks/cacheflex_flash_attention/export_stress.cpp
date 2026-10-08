#include <cstdio>
#include <cstdlib>
#include <cstdint>
extern "C" unsigned long cf_export_hook(void*,unsigned long,unsigned long,const char*);
int main(int argc,char**argv) {
    size_t n=argc>1?strtoul(argv[1],nullptr,0):1048576;
    auto *p=static_cast<uint64_t*>(aligned_alloc(64,n));
    if(!p || n%64) return 2;
    for(size_t i=0;i<n/8;++i) p[i]=i^0x123456789abcdef0ULL;
    volatile char name[64];
    const char *literal="stress.bin";
    for(unsigned i=0;i<11;++i) name[i]=literal[i];
    asm volatile("dsb sy":::"memory");
    printf("EXPORT_STRESS bytes=%zu filename=%p payload=%p\n",n,(void*)name,p);
    fflush(stdout);
    auto done=cf_export_hook(p,n,0,(const char*)name);
    printf("EXPORT_DONE %lu\n",done);
    free(p);
    return done==n?0:3;
}
