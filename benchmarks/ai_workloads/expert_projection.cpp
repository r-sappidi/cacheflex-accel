#include <arm_sve.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef CACHEFLEX_SPM
#include "kernels_spm.hpp"
#else
#include "common.hpp"
#include "kernels_sve.hpp"
#endif

int main(int argc, char **argv) {
    const int tokens = argc > 1 ? std::atoi(argv[1]) : 32;
    const int dim = argc > 2 ? std::atoi(argv[2]) : 512;
    const int experts = argc > 3 ? std::atoi(argv[3]) : 4;
    const int kc = argc > 4 ? std::atoi(argv[4]) : 256;
    const int mc = argc > 5 ? std::atoi(argv[5]) : 128;
    const size_t matrix = static_cast<size_t>(dim) * dim;

    AlignedBuffer<__fp16> input(static_cast<size_t>(tokens) * dim);
    AlignedBuffer<__fp16> output(static_cast<size_t>(tokens) * dim);
    AlignedBuffer<__fp16> weights(matrix * experts);
#ifdef CACHEFLEX_INT8
    AlignedBuffer<int8_t> quantized(matrix * experts);
    for (size_t i = 0; i < matrix * experts; ++i)
        quantized.data()[i] = static_cast<int8_t>((i * 17 + 5) % 31 - 15);
#endif
    for (size_t i = 0; i < static_cast<size_t>(tokens) * dim; ++i)
        input.data()[i] = static_cast<__fp16>((static_cast<int>(i % 13) - 6) / 16.0f);

#ifndef CACHEFLEX_INT8
    for (size_t i = 0; i < matrix * experts; ++i)
        weights.data()[i] = static_cast<__fp16>((static_cast<int>((i * 17 + 5) % 31) - 15) / 64.0f);
#endif

#ifndef CACHEFLEX_SPM
    const size_t packed_size = prepack_B_knm_size(dim, dim, kc);
    AlignedBuffer<__fp16> packed(packed_size);
#endif

    ROI_BEGIN();
    const auto start = Clock::now();
    for (int e = 0; e < experts; ++e) {
        __fp16 *w = weights.data() + static_cast<size_t>(e) * matrix;
#ifdef CACHEFLEX_INT8
        const int8_t *q = quantized.data() + static_cast<size_t>(e) * matrix;
        const float scale = 1.0f / 64.0f;
        for (size_t i = 0; i < matrix; ++i)
            w[i] = static_cast<__fp16>(static_cast<float>(q[i]) * scale);
#endif
#ifdef CACHEFLEX_SPM
        gemm_spm(input.data(), w, output.data(), tokens, dim, dim, dim, mc, kc);
#else
        prepack_B_knm(w, packed.data(), dim, dim, kc);
        gemm_knm_prepacked(input.data(), packed.data(), output.data(),
                           tokens, dim, dim, mc, kc);
#endif
    }
    const double elapsed_us = us_since(start);
    ROI_END();

    double checksum = 0.0;
    bool finite = true;
    for (size_t i = 0; i < static_cast<size_t>(tokens) * dim; ++i) {
        const float value = static_cast<float>(output.data()[i]);
        checksum += value;
        finite &= std::isfinite(value);
    }
    const double flops = 2.0 * tokens * dim * dim * experts;
    std::printf("EXPERT_PROJECTION mode=%s precision=%s tokens=%d dim=%d experts=%d "
                "time_us=%.1f gflops=%.2f checksum=%.9f %s\n",
#ifdef CACHEFLEX_SPM
                "spm",
#else
                "cache",
#endif
#ifdef CACHEFLEX_INT8
                "int8-weight",
#else
                "fp16",
#endif
                tokens, dim, experts, elapsed_us, flops / (elapsed_us * 1e3),
                checksum, finite ? "PASS" : "FAIL");
    return finite ? 0 : 1;
}
