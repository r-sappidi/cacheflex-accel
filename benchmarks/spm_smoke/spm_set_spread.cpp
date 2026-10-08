#include "common_spm.hpp"

// VL16: a 256-byte vector spans four ways whose physical set rotations differ.
// Start at way 1 so even the first lane exercises nonidentity placement.
int main()
{
    alignas(64) volatile std::uint64_t source[32], modified[32], output[32], dest[32];
    const unsigned sets[] = {0, 511, 512, 1023};
    int errors = 0;
    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        const unsigned set = sets[iteration % 4];
        for (unsigned i = 0; i < 32; ++i) {
            source[i] = 17 + iteration * 997 + i;
            modified[i] = 1999 + iteration * 877 + i;
            dest[i] = output[i] = 0;
        }
        asm volatile("dsb sy" ::: "memory");
        for (unsigned w = 0; w < 4; ++w) {
            auto slot = spm_addr(w + 1, set);
            asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"
                         :: "r"(slot), "r"(source + 8*w) : "memory");
        }
        auto first = spm_addr(1, set);
        asm volatile("ptrue p0.d\nspm.ld1d z0.d, p0/z, [%0]\n"
                     "st1d z0.d, p0, [%1]\ndsb sy\n"
                     :: "r"(first), "r"(output) : "p0", "z0", "memory");
        for (unsigned i = 0; i < 32; ++i) errors += output[i] != source[i];
        asm volatile("ptrue p0.d\nld1d z0.d, p0/z, [%1]\n"
                     "spm.st1d z0.d, p0, [%0]\ndsb sy\n"
                     :: "r"(first), "r"(modified) : "p0", "z0", "memory");
        for (unsigned w = 0; w < 4; ++w) {
            auto slot = spm_addr(w + 1, set);
            asm volatile("SPMWB_64_IMM %0, [%1, #0]\n"
                         :: "r"(slot), "r"(dest + 8*w) : "memory");
        }
        // Deliberately no post-WB fence: coherent destination RAW must hold.
        for (unsigned i = 0; i < 32; ++i) errors += dest[i] != modified[i];
        asm volatile("dsb sy" ::: "memory");
        for (unsigned w = 0; w < 4; ++w) {
            auto slot = spm_addr(w + 1, set);
            asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n" :: "r"(slot) : "memory");
        }
        asm volatile("dsb sy" ::: "memory");
    }
    std::printf("SPM_SET_SPREAD %s errors=%d\n", errors ? "FAIL" : "PASS", errors);
    return errors != 0;
}
