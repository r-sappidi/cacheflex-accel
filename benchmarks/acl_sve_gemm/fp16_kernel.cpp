#include <arm_sve.h>
#include <cstddef>

// Use the paper driver's packing routines as well as its fused compute
// kernels.  Keeping these wrappers in the same translation unit ensures that
// the multicore harness executes the same VL=4 pack-A and pack-B code as
// v1_fused/v3_fused_vl4.
#include "../../../cacheflex_micro/kernels/llama_bench/kernels_sve.hpp"

// Keep the multicore harness on the exact VL=4 FP16 compute kernels used by
// cacheflex_micro's paper sweep.  The wrapper gives the shared driver a stable
// external interface while the header's implementations remain static.
#define VL_4
#include "../../../cacheflex_micro/experiments/new_gemm/src/kernels_fused.hpp"

namespace arm_gemm {

void
pack_fp16_a_8row(const __fp16 *a, std::size_t lda, __fp16 *packed,
                 std::size_t m, std::size_t m0, std::size_t k,
                 std::size_t k0, std::size_t mc, std::size_t kc)
{
    pack_A_fp16_8row(a, lda, packed, m, m0, k, k0, mc, kc);
}

void
pack_fp16_b_3vl(__fp16 *packed, const __fp16 *b, std::size_t width,
                std::size_t stride_bytes, std::size_t rows)
{
    sve_pack_B_3VL(reinterpret_cast<uint16_t *>(packed),
                   reinterpret_cast<const uint16_t *>(b),
                   width, stride_bytes, rows);
}

void
sve_interleaved_fp16_mla_8x3VL(const __fp16 *a, const __fp16 *b,
                               __fp16 *c, int blocks, int k, int ldc,
                               bool first_k)
{
    cache_gemm_fused_8x3VL(a, b, c, blocks, k, ldc, first_k ? 1 : 0);
}

#ifdef CACHEFLEX_SPM_GEMM
void
sve_interleaved_fp16_mla_8x3VL_spm(const __fp16 *a, __fp16 *c,
                                   int blocks, int k, int ldc,
                                   bool first_k, std::size_t base_set)
{
    spm_gemm_fused_8x3VL(a, c, blocks, k, ldc, first_k ? 1 : 0, base_set);
}
#endif

} // namespace arm_gemm
