#pragma once
// Typed command: bit58 says the low 128B contain 64 dense FP16 values.
// The existing VL16 core-to-source capture is still a full 256B vector store;
// upper lanes are ignored by this explicit ABI, NOT generic lossy compression.
// Source-to-bank transmission is 128B. All four FP32 destination slots remain
// locked until the bank widens exactly and installs separately rounded mul/add.
static inline void output_reduce_half(unsigned row,float scale,svfloat16_t value) {
    uint32_t bits;std::memcpy(&bits,&scale,4);
    auto addr=shared_slot(shared_output_bank,row)|(uint64_t(bits)<<24)|
              (uint64_t(1)<<60)|(uint64_t(1)<<58);
    auto p=svptrue_b16();
    asm volatile("spm.st1h %0.h, %1, [%2]\n"::"w"(value),"Upl"(p),"r"(addr):"memory");
}
