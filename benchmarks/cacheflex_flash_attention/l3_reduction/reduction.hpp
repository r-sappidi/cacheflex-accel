#pragma once
#include "shared_spm.hpp"
#include <cstring>
// bit60=command, bit59=shared, bits55:24=FP32 alpha, bits22:20=bank.
// Slot bits remain unchanged. The scalar header does not alias bank identity.
static inline void shared_reduce(uint64_t slot,float scale,svfloat32_t value) {
    uint32_t bits;std::memcpy(&bits,&scale,4);
    const auto a=slot|(uint64_t(bits)<<24)|(uint64_t(1)<<60);
    auto p=svptrue_b32();
    asm volatile("spm.st1w %0.s, %1, [%2]\n"::"w"(value),"Upl"(p),"r"(a):"memory");
}
static inline void output_reduce(unsigned row,float scale,svfloat32_t value) {
    shared_reduce(shared_slot(shared_output_bank,row),scale,value);
}
