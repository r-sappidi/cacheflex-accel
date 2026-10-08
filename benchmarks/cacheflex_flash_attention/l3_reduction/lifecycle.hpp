#pragma once
#include "half_contribution.hpp"
static inline void output_reduce_initialize(unsigned row,float scale,svfloat16_t value) {
    uint32_t bits;std::memcpy(&bits,&scale,4);
    auto addr=shared_slot(shared_output_bank,row)|(uint64_t(bits)<<24)|
        (uint64_t(1)<<60)|(uint64_t(1)<<58)|(uint64_t(1)<<56);
    auto p=svptrue_b16();
    asm volatile("spm.st1h %0.h, %1, [%2]\n"::"w"(value),"Upl"(p),"r"(addr):"memory");
}
static inline void output_finalize(unsigned row,float reciprocal,float16_t* destination) {
    uint32_t bits;std::memcpy(&bits,&reciprocal,4);
    for(unsigned half=0;half<2;++half) {
        auto slot=shared_slot(shared_output_bank,row,half*2)|
                  (uint64_t(bits)<<24)|(uint64_t(1)<<57);
        // Destination translation, coherent ordering and completion use the
        // existing bridge. Never encode a guest virtual pointer as physical.
        shared_writeback(slot,destination+half*32);
    }
}
