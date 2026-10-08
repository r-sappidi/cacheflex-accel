#pragma once
#include <arm_sve.h>
#include <cstdint>
#include <cstring>
// Experimental command ABI: existing SPM store encoding, opt-in command space.
// Bits 18:0 = slot, 51:20 = FP32 scale, bit 60 = ordered scale/add command.
// Both operands genuinely pass through CPU registers/store datapath.
static inline uint64_t output_slot(unsigned row) {
    return (uint64_t(2)<<16) | (uint64_t(row)<<6);
}
static inline svfloat32_t output_load(unsigned row) {
    svfloat32_t x;auto p=svptrue_b32();auto a=output_slot(row);
    asm volatile("spm.ld1w %0.s, %1/z, [%2]\n":"=w"(x):"Upl"(p),"r"(a):"memory");
    return x;
}
static inline void output_store(unsigned row,svfloat32_t x) {
    auto p=svptrue_b32();auto a=output_slot(row);
    asm volatile("spm.st1w %0.s, %1, [%2]\n"::"w"(x),"Upl"(p),"r"(a):"memory");
}
static inline void output_reduce(unsigned row,float scale,svfloat32_t x) {
    uint32_t bits;std::memcpy(&bits,&scale,4);
    auto a=output_slot(row)|(uint64_t(bits)<<20)|(uint64_t(1)<<60);
    auto p=svptrue_b32();
    asm volatile("spm.st1w %0.s, %1, [%2]\n"::"w"(x),"Upl"(p),"r"(a):"memory");
}
static inline void output_claim(float* zero,unsigned rows) {
    for(unsigned r=0;r<rows;++r)for(unsigned w=0;w<4;++w) {
        auto a=output_slot(r)+(uint64_t(w)<<16);const float* src=zero+r*64+w*16;
        asm volatile("SPMCP_64_IMM %0, [%1, #0]\n"::"r"(a),"r"(src):"memory");
    }
    asm volatile("dsb sy":::"memory");
}
static inline void output_release(unsigned rows) {
    asm volatile("dsb sy":::"memory");
    for(unsigned r=0;r<rows;++r)for(unsigned w=0;w<4;++w) {
        auto a=output_slot(r)+(uint64_t(w)<<16);
        asm volatile("SPMREL_8_IMM xzr, [%0, #0]\n"::"r"(a):"memory");
    }
    asm volatile("dsb sy":::"memory");
}
