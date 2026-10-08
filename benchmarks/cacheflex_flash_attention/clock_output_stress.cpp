// Diagnostic only: exercise output-only syscall buffers after cache eviction.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <sys/syscall.h>
#include <unistd.h>
int main() {
    constexpr size_t n=8*1024*1024;
    auto *p=static_cast<volatile uint64_t*>(aligned_alloc(64,n));
    auto *t=static_cast<timespec*>(aligned_alloc(64,64));
    if(!p || !t) return 2;
    setbuf(stdout,nullptr);
    t->tv_sec=123;t->tv_nsec=456;
    printf("CLOCK_OUTPUT ptr=%p\n",static_cast<void*>(t));
    asm volatile("dsb sy":::"memory");
    for(size_t i=0;i<n/8;++i) p[i]=i;
    asm volatile("dsb sy":::"memory");
    long rc=syscall(SYS_clock_gettime,CLOCK_MONOTONIC,t);
    printf("CLOCK_DONE rc=%ld sec=%ld nsec=%ld\n",rc,t->tv_sec,t->tv_nsec);
    return rc!=0;
}
