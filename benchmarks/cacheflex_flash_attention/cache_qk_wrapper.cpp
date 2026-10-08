#include <cmath>
#define sqrtf sqrt
#define main cacheflex_qk_source_main
#ifdef CACHEFLEX_FLASH_SEARCH_KERNELS
#include <arm_sve.h>
#include <cstddef>
#include CACHEFLEX_FLASH_SEARCH_KERNELS
#else
#include "../../../cacheflex_micro/experiments/flash_attention/src/cache_flash_attn.cpp"
#endif
#undef main
#undef sqrtf

extern "C" void cacheflex_qk_8x96(const __fp16 *a, const __fp16 *b,
                                   __fp16 *c, int k, int ldc)
{
#ifdef CACHEFLEX_FLASH_SEARCH_KERNELS
    cache_qk_8x1VL_ldc(a, b, c, k, ldc);
#else
    sve_mla_8x3VL_ldc(a, b, c, k, ldc);
#endif
}
