#pragma once
#include <arm_sve.h>
#include <cstdint>
// Experimental SE ABI: 8 LLC banks, up to 8 eligible ways/bank. Rows use
// one set, with consecutive 64B chunks in consecutive ways (existing SPM ISA).
static inline uint64_t shared_slot(unsigned bank,unsigned row,unsigned way=0) {
    return (uint64_t(1)<<59)|(uint64_t(bank)<<20)|(uint64_t(way)<<16)|(uint64_t(row)<<6);
}
static inline void shared_drain(){asm volatile("dsb sy":::"memory");}
static inline svfloat32_t shared_load(uint64_t a) {
    svfloat32_t v;auto p=svptrue_b32();
    asm volatile("spm.ld1w %0.s, %1/z, [%2]\n":"=w"(v):"Upl"(p),"r"(a):"memory");return v;
}
static inline void shared_store(uint64_t a,svfloat32_t v,svbool_t p=svptrue_b32()) {
    asm volatile("spm.st1w %0.s, %1, [%2]\n"::"w"(v),"Upl"(p),"r"(a):"memory");
}
static inline void shared_copy(uint64_t a,const void* src) {
    asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(a),"r"(src):"memory");
}
static inline void shared_writeback(uint64_t a,void* dst) {
    asm volatile("SPMWB_64_IMM %0, [%1, #0]\n"::"r"(a),"r"(dst):"memory");
}
static inline void shared_release(uint64_t a) {
    asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");
}

// Adapter for the existing checked FA output-residency driver. Each worker
// owns a disjoint shared-bank region. Arithmetic remains on the CPU.
static thread_local unsigned shared_output_bank=0;
static inline svfloat32_t output_load(unsigned r){return shared_load(shared_slot(shared_output_bank,r));}
static inline void output_store(unsigned r,svfloat32_t x){shared_store(shared_slot(shared_output_bank,r),x);}
static inline void output_claim(float*,unsigned rows) {
    for(unsigned r=0;r<rows;++r)output_store(r,svdup_n_f32(0));
    // Same-core RAW/WAW tracking, not a whole-panel wait, orders consumption.
}
static inline void output_release(unsigned rows) {
    shared_drain();
    for(unsigned r=0;r<rows;++r)for(unsigned w=0;w<4;++w)
        shared_release(shared_slot(shared_output_bank,r,w));
    shared_drain();
}
