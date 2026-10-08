#pragma once
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gem5/m5ops.h>

// SVE widening FCVT reads alternate halfword lanes. Duplicate the dense
// lower half before conversion; compact alternate lanes after narrowing.
static inline svfloat32_t cf_dense_f16_to_f32(svbool_t pg,svfloat16_t h) {
    return svcvt_f32_f16_z(pg,svzip1_f16(h,h));
}
static inline svfloat16_t cf_dense_f32_to_f16(svbool_t pg,svfloat32_t f) {
    auto h=svcvt_f16_f32_z(pg,f);
    return svuzp1_f16(h,h);
}
#define svcvt_f32_f16_z(pg,h) cf_dense_f16_to_f32(pg,h)
#define svcvt_f16_f32_z(pg,f) cf_dense_f32_to_f16(pg,f)

static inline int cf_audit_output(const __fp16* output,const __fp16* value,
                                 size_t count,size_t d) {
    size_t errors=0;
#if !CACHEFLEX_FLASH_RICH_INPUT
    // Every token has identical Q/K/V for the historical periodic input.
    // Correct attention must therefore return V at every query/head.
    for(size_t i=0;i<count;++i) {
        uint16_t actual,expected;
        std::memcpy(&actual,output+i,2);std::memcpy(&expected,value+i%d,2);
        errors+=actual!=expected;
    }
    std::printf("FA_ANALYTIC errors=%zu\n",errors);
#endif
    if(m5_write_file(const_cast<__fp16*>(output),count*2,0,"output.fp16")!=count*2) return 3;
    return errors?4:0;
}
