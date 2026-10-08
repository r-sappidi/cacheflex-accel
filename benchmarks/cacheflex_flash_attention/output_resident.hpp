#pragma once
#include <arm_sve.h>
#include <cstddef>
#include <cstdint>

// VL16/d64: each FP32 row occupies two sets across ways 0 and 1.
// dtype=0xb transfers 32 uint32 memory elements to/from 32 uint64
// register lanes: exactly 128 bytes, not a predicated 256-byte request.
// ZIP/UZP preserve FP32 bits; these are NOT numeric conversions.
static inline size_t cf_o_slot(size_t row, size_t half) {
    return ((row & 63) * 2 + half) << 6;
}
static inline svfloat32_t cf_o_load(const float* cache, size_t row) {
#if CACHEFLEX_OUTPUT_SPM
    svuint64_t lo, hi;
    auto p=svptrue_b64();
    size_t a=cf_o_slot(row,0), b=cf_o_slot(row,1);
    asm volatile("spm.ld1ad %0.d, %2/z, [%3]\n"
                 "spm.ld1ad %1.d, %2/z, [%4]\n"
                 : "=&w"(lo), "=&w"(hi) : "Upl"(p), "r"(a), "r"(b) : "memory");
    return svreinterpret_f32_u32(svuzp1_u32(svreinterpret_u32_u64(lo),
                                           svreinterpret_u32_u64(hi)));
#else
    return svld1_f32(svptrue_b32(),cache+row*64);
#endif
}
static inline void cf_o_store(float* cache, size_t row, svfloat32_t value) {
#if CACHEFLEX_OUTPUT_SPM
    auto bits=svreinterpret_u32_f32(value);
    auto lo=svreinterpret_u64_u32(svzip1_u32(bits,bits));
    auto hi=svreinterpret_u64_u32(svzip2_u32(bits,bits));
    auto p=svptrue_b64();
    size_t a=cf_o_slot(row,0), b=cf_o_slot(row,1);
    asm volatile("spm.st1ad %0.d, %2, [%3]\n"
                 "spm.st1ad %1.d, %2, [%4]\n"
                 :: "w"(lo), "w"(hi), "Upl"(p), "r"(a), "r"(b) : "memory");
#else
    svst1_f32(svptrue_b32(),cache+row*64,value);
#endif
}
static inline void cf_o_scale(float* cache,size_t row,float alpha) {
    cf_o_store(cache,row,svmul_n_f32_x(svptrue_b32(),cf_o_load(cache,row),alpha));
}
static inline void cf_o_add(float* cache,size_t row,const __fp16* pv) {
    auto p=svptrue_b32();
    auto h=svld1_f16(svwhilelt_b16_u64(0,64),pv);
    auto x=svcvt_f32_f16_z(p,h); // project's dense-lane conversion wrapper
    cf_o_store(cache,row,svadd_f32_x(p,cf_o_load(cache,row),x));
}
static inline void cf_o_claim(float* cache,size_t rows) {
#if CACHEFLEX_OUTPUT_SPM
    // Existing scalar copy is necessary to claim a slot before SPMST.
    // Source is the existing, pre-zeroed Oacc allocation, not new backing SRAM.
    for(size_t r=0;r<rows;++r) for(size_t half=0;half<2;++half)
        for(size_t way=0;way<2;++way) {
            size_t dst=cf_o_slot(r,half)|(way<<16);
            const char* src=reinterpret_cast<const char*>(cache+r*64)+half*128+way*64;
            asm volatile("SPMCP_64_IMM %0, [%1, #0]\n" ::"r"(dst),"r"(src):"memory");
        }
    asm volatile("dsb sy" ::: "memory");
#endif
}
static inline void cf_o_release(size_t rows) {
#if CACHEFLEX_OUTPUT_SPM
    asm volatile("dsb sy" ::: "memory");
    for(size_t r=0;r<rows;++r) for(size_t half=0;half<2;++half)
        for(size_t way=0;way<2;++way) {
            size_t slot=cf_o_slot(r,half)|(way<<16);
            asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n" ::"r"(slot):"memory");
        }
    asm volatile("dsb sy" ::: "memory");
#endif
}
