#pragma once
// Unchanged packing algorithm from micro end2end/cache_flash_attn.cpp;
// renamed and specialized to its eight-row tile. Bit-preserving permutation.
static void cf_vector_pack_A(
    const __fp16* A, size_t lda, __fp16* Apanel,
    size_t M, size_t m0, size_t K, size_t k0, size_t mc, size_t kc)
{
    const size_t VLh = svcnth();
    size_t rows = std::min(mc, M - m0);
    kc = std::min(kc, K - k0);
    size_t ablocks = (rows + 8 - 1) / 8;
    __fp16* pad_row = static_cast<__fp16*>(__builtin_alloca(kc * sizeof(__fp16)));
    std::memset(pad_row, 0, kc * sizeof(__fp16));
    for (size_t mb = 0; mb < ablocks; ++mb) {
        size_t row_base = m0 + mb * 8;
        __fp16* dst = Apanel + mb * (kc * 8);
        const __fp16* rp[8];
        for (size_t r = 0; r < 8; ++r) {
            size_t gr = row_base + r;
            rp[r] = (gr < M) ? (A + gr * lda + k0) : pad_row;
        }
        size_t ki = 0;
        for (; ki + VLh <= kc; ki += VLh) {
            svbool_t pg = svptrue_b16();
            svfloat16_t r0=svld1_f16(pg,rp[0]+ki), r1=svld1_f16(pg,rp[1]+ki);
            svfloat16_t r2=svld1_f16(pg,rp[2]+ki), r3=svld1_f16(pg,rp[3]+ki);
            svfloat16_t r4=svld1_f16(pg,rp[4]+ki), r5=svld1_f16(pg,rp[5]+ki);
            svfloat16_t r6=svld1_f16(pg,rp[6]+ki), r7=svld1_f16(pg,rp[7]+ki);
            svfloat16_t t01lo=svzip1_f16(r0,r1), t01hi=svzip2_f16(r0,r1);
            svfloat16_t t23lo=svzip1_f16(r2,r3), t23hi=svzip2_f16(r2,r3);
            svfloat16_t t45lo=svzip1_f16(r4,r5), t45hi=svzip2_f16(r4,r5);
            svfloat16_t t67lo=svzip1_f16(r6,r7), t67hi=svzip2_f16(r6,r7);
            svfloat32_t s01lo=svreinterpret_f32_f16(t01lo), s01hi=svreinterpret_f32_f16(t01hi);
            svfloat32_t s23lo=svreinterpret_f32_f16(t23lo), s23hi=svreinterpret_f32_f16(t23hi);
            svfloat32_t s45lo=svreinterpret_f32_f16(t45lo), s45hi=svreinterpret_f32_f16(t45hi);
            svfloat32_t s67lo=svreinterpret_f32_f16(t67lo), s67hi=svreinterpret_f32_f16(t67hi);
            svfloat32_t u0=svzip1_f32(s01lo,s23lo), u1=svzip2_f32(s01lo,s23lo);
            svfloat32_t u2=svzip1_f32(s01hi,s23hi), u3=svzip2_f32(s01hi,s23hi);
            svfloat32_t u4=svzip1_f32(s45lo,s67lo), u5=svzip2_f32(s45lo,s67lo);
            svfloat32_t u6=svzip1_f32(s45hi,s67hi), u7=svzip2_f32(s45hi,s67hi);
            svfloat64_t d0=svzip1_f64(svreinterpret_f64_f32(u0),svreinterpret_f64_f32(u4));
            svfloat64_t d1=svzip2_f64(svreinterpret_f64_f32(u0),svreinterpret_f64_f32(u4));
            svfloat64_t d2=svzip1_f64(svreinterpret_f64_f32(u1),svreinterpret_f64_f32(u5));
            svfloat64_t d3=svzip2_f64(svreinterpret_f64_f32(u1),svreinterpret_f64_f32(u5));
            svfloat64_t d4=svzip1_f64(svreinterpret_f64_f32(u2),svreinterpret_f64_f32(u6));
            svfloat64_t d5=svzip2_f64(svreinterpret_f64_f32(u2),svreinterpret_f64_f32(u6));
            svfloat64_t d6=svzip1_f64(svreinterpret_f64_f32(u3),svreinterpret_f64_f32(u7));
            svfloat64_t d7=svzip2_f64(svreinterpret_f64_f32(u3),svreinterpret_f64_f32(u7));
            svst1_f16(pg,dst+ki*8+0*VLh,svreinterpret_f16_f64(d0));
            svst1_f16(pg,dst+ki*8+1*VLh,svreinterpret_f16_f64(d1));
            svst1_f16(pg,dst+ki*8+2*VLh,svreinterpret_f16_f64(d2));
            svst1_f16(pg,dst+ki*8+3*VLh,svreinterpret_f16_f64(d3));
            svst1_f16(pg,dst+ki*8+4*VLh,svreinterpret_f16_f64(d4));
            svst1_f16(pg,dst+ki*8+5*VLh,svreinterpret_f16_f64(d5));
            svst1_f16(pg,dst+ki*8+6*VLh,svreinterpret_f16_f64(d6));
            svst1_f16(pg,dst+ki*8+7*VLh,svreinterpret_f16_f64(d7));
        }
        for (; ki < kc; ++ki)
            for (size_t r = 0; r < 8; ++r)
                dst[ki*8 + r] = rp[r][ki];
    }
}
