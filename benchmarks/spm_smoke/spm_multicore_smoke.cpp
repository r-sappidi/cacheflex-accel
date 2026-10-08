#include "common_spm.hpp"

#include <algorithm>
#include <cstdint>

static inline void spmcp64(std::uint64_t dst_spm, const void *src)
{
    asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"
                 :
                 : "r"(dst_spm), "r"(src)
                 : "memory");
}

static inline void consume_spm64(std::uint64_t src_spm, volatile double *sink)
{
    asm volatile(
        "ptrue p0.d\n"
        "spm.ld1d z0.d, p0/z, [%0]\n"
        "faddv d0, p0, z0.d\n"
        "fmov %d1, d0\n"
        : "+r"(src_spm), "=w"(*sink)
        :
        : "p0", "z0", "memory");
}

int main(int argc, char **argv)
{
    const int core_id = argc > 1 ? std::atoi(argv[1]) : 0;
    const int cores = argc > 2 ? std::atoi(argv[2]) : 1;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 100;
    const int sets_per_core = argc > 4 ? std::atoi(argv[4]) : 32;
    const int base_set = argc > 5 ? std::atoi(argv[5]) : 128;

    const int total_sets = std::max(1, cores * sets_per_core);
    AlignedBuffer<double> src(total_sets * 2 * 8);
    for (std::size_t i = 0; i < src.size(); ++i) {
        src.data()[i] = double((i + 1) * (core_id + 1));
    }

    volatile double sink0 = 0.0;
    volatile double sink1 = 0.0;
    int errors = 0;
    ROI_BEGIN();
    for (int iter = 0; iter < iters; ++iter) {
        for (int local_set = 0; local_set < sets_per_core; ++local_set) {
            const unsigned set = unsigned(base_set + core_id * sets_per_core + local_set);
            const std::size_t src_line =
                std::size_t(core_id * sets_per_core + local_set) * 2;
            const double *line0 = &src.data()[src_line * 8];
            const double *line1 = &src.data()[(src_line + 1) * 8];
            const std::uint64_t way0 = spm_addr(0, set);
            const std::uint64_t way1 = spm_addr(1, set);

            asm volatile("dsb sy" ::: "memory");
            spmcp64(way0, line0);
            spmcp64(way1, line1);
            asm volatile("dsb sy" ::: "memory");

            consume_spm64(way0, &sink0);
            consume_spm64(way1, &sink1);

            double expected0 = 0.0;
            double expected1 = 0.0;
            for (int lane = 0; lane < 8; ++lane) {
                expected0 += line0[lane];
                expected1 += line1[lane];
            }
            if (sink0 != expected0 || sink1 != expected1) {
                ++errors;
            }
        }
    }
    ROI_END();

    std::printf(
        "spm_multicore_smoke %s core=%d cores=%d iters=%d "
        "sets_per_core=%d base_set=%d sink0=%f sink1=%f errors=%d\n",
        errors == 0 ? "PASS" : "FAIL", core_id, cores, iters,
        sets_per_core, base_set, double(sink0), double(sink1), errors);
    return errors == 0 ? 0 : 1;
}
