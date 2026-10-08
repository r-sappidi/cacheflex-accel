#include "common_spm.hpp"

#ifdef ROUNDTRIP_NO_WB_BARRIER
#define WB_DRAIN ""
#else
#define WB_DRAIN "dsb sy\n"
#endif

// Single-line VL4 test: cache->SPM, direct load/store, SPM->cache.
// Coherent source/destination remain distinct; no concurrent source writers.
int main()
{
    alignas(64) volatile std::uint64_t source[8], modified[8], output[8], dest[8];
    int errors = 0;
    for (int iteration=0; iteration<64; ++iteration) {
        for (int i=0; i<8; ++i) {
            source[i] = iteration*83+i+1;
            modified[i] = iteration*107+i+999;
            output[i] = 0;
            dest[i] = 0;
        }
        std::uint64_t slot = spm_addr(0, iteration%32);
        asm volatile("dsb sy\nSPMCP_64_IMM %0, [%1, #0]\ndsb sy\n"
                     "ptrue p0.d\nspm.ld1d z0.d, p0/z, [%0]\n"
                     "st1d z0.d, p0, [%2]\ndsb sy\n"
                     :: "r"(slot), "r"(source), "r"(output)
                     : "p0", "z0", "memory");
        for (int i=0; i<8; ++i) errors += output[i] != source[i];
        asm volatile("ptrue p0.d\nld1d z0.d, p0/z, [%1]\n"
                     "spm.st1d z0.d, p0, [%0]\ndsb sy\n"
                     "SPMWB_64_IMM %0, [%2, #0]\n" WB_DRAIN
                     :: "r"(slot), "r"(modified), "r"(dest)
                     : "p0", "z0", "memory");
        for (int i=0; i<8; ++i) errors += dest[i] != modified[i];
    }
    std::printf("SPM_ROUNDTRIP %s errors=%d\n", errors ? "FAIL" : "PASS", errors);
    return errors != 0;
}
