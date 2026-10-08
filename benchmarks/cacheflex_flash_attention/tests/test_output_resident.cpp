#include <arm_sve.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "../output_resident.hpp"

int main() {
    if(svcntw()!=64) return 2;
    alignas(64) float bootstrap[128] = {};
    alignas(64) uint32_t input[64], actual[64];
    cf_o_claim(bootstrap,2);
    size_t errors=0;
    for(size_t iteration=0;iteration<64;++iteration) {
        for(size_t i=0;i<64;++i) input[i]=(i+iteration*131)*0x13579bdfU;
        auto v=svreinterpret_f32_u32(svld1_u32(svptrue_b32(),input));
        cf_o_store(bootstrap,iteration%2,v);
        asm volatile("dsb sy" ::: "memory");
        auto got=svreinterpret_u32_f32(cf_o_load(bootstrap,iteration%2));
        svst1_u32(svptrue_b32(),actual,got);
        for(size_t i=0;i<64;++i) {
            if(input[i]!=actual[i] && errors<8)
                std::printf("MISMATCH iteration=%zu lane=%zu expected=%08x actual=%08x\n",
                            iteration,i,input[i],actual[i]);
            errors+=input[i]!=actual[i];
        }
    }
    cf_o_release(2);
    std::printf("OUTPUT_RESIDENT_ROUNDTRIP errors=%zu elements=4096\n",errors);
    return errors?1:0;
}
