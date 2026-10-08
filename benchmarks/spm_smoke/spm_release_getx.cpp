// Stress an SPM snapshot release racing another core's coherent store.
// Shared data accesses use explicit assembly; barriers delimit each iteration.
#include "common_spm.hpp"
#include <pthread.h>
#include <atomic>

alignas(64) static double lines[64][8];
static pthread_barrier_t barrier;
static std::atomic<int> errors{0};

static void store(double *p, double x) {
    asm volatile("str %d1, [%0]" :: "r"(p), "w"(x) : "memory");
}
static double load(const double *p) {
    double x;
    asm volatile("ldr %d0, [%1]" : "=w"(x) : "r"(p) : "memory");
    return x;
}
static void *writer(void *) {
    for (int iter=0; iter<64; ++iter) {
        pthread_barrier_wait(&barrier);
        for (int i=0; i<(iter%16)*8; ++i) asm volatile("nop");
        for (int line=0; line<64; ++line) store(&lines[line][0], iter+1001);
        asm volatile("dsb sy" ::: "memory");
        pthread_barrier_wait(&barrier);
    }
    return nullptr;
}
int main() {
    pthread_barrier_init(&barrier, nullptr, 2);
    pthread_t thread;
    pthread_create(&thread, nullptr, writer, nullptr);
    ROI_BEGIN();
    for (int iter=0; iter<64; ++iter) {
        for (int line=0; line<64; ++line)
            for (int lane=0; lane<8; ++lane) store(&lines[line][lane], iter+1);
        asm volatile("dsb sy" ::: "memory");
        pthread_barrier_wait(&barrier);
        for (int line=0; line<64; ++line) {
            uint64_t slot = spm_addr(0, line);
            asm volatile("SPMCP_64_IMM %0, [%1, #0]" :: "r"(slot), "r"(lines[line]) : "memory");
            alignas(64) double snapshot[8];
            asm volatile("dsb sy\nptrue p0.d\nspm.ld1d z0.d, p0/z, [%0]\nst1d z0.d, p0, [%1]"
                :: "r"(slot), "r"(snapshot) : "p0", "z0", "memory");
            if (snapshot[0] != iter+1 && snapshot[0] != iter+1001) ++errors;
            for (int lane=1; lane<8; ++lane)
                if (snapshot[lane] != iter+1) ++errors;
        }
        pthread_barrier_wait(&barrier);
        for (int line=0; line<64; ++line)
            if (load(&lines[line][0]) != iter+1001) ++errors;
    }
    ROI_END();
    pthread_join(thread, nullptr);
    std::printf("SPM_RELEASE_GETX %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors.load());
    return errors != 0;
}
