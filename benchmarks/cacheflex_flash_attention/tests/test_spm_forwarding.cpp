#include <arm_sve.h>
#include <cstdio>
#include <cstdint>
#include "../output_resident.hpp"

// No barrier between the stores and loads: exercise pending SQ forwarding.
int main() {
    if (svcntw()!=64) return 2;
    alignas(64) float bootstrap[128]={};
    alignas(64) uint32_t a[64], b[64], out[64];
    cf_o_claim(bootstrap,2);
    size_t errors=0, full_errors=0, partial_errors=0;
    for (unsigned r=0;r<128;++r) {
        for (unsigned i=0;i<64;++i) {
            a[i]=(i+r*131)*0x13579bdfU;
            b[i]=~a[i];
        }
        auto av=svreinterpret_f32_u32(svld1_u32(svptrue_b32(),a));
        auto bv=svreinterpret_f32_u32(svld1_u32(svptrue_b32(),b));
        cf_o_store(bootstrap,0,av);
        cf_o_store(bootstrap,1,bv); // different slot must not alias
        cf_o_store(bootstrap,0,bv); // youngest writer must win
        auto got=svreinterpret_u32_f32(cf_o_load(bootstrap,0));
        svst1_u32(svptrue_b32(),out,got);
        for(unsigned i=0;i<64;++i) if(out[i]!=b[i]) {
            if(full_errors<8) std::printf("FULL r=%u i=%u expected=%08x actual=%08x\n",r,i,b[i],out[i]);
            ++full_errors;
        }

        // An 8B scalar store covers only part of a subsequent 128B load.
        // This must wait/merge in SRAM, never forward incomplete SQ payload.
        size_t addr=cf_o_slot(0,0);
        uint64_t payload=uint64_t(a[0])|(uint64_t(a[1])<<32);
        asm volatile("SPMSTR_8_IMM %0, [%1, #0]\n"
                     ::"r"(payload),"r"(addr):"memory");
        got=svreinterpret_u32_f32(cf_o_load(bootstrap,0));
        svst1_u32(svptrue_b32(),out,got);
        for(unsigned i=0;i<64;++i) if(out[i]!=(i<2?a[i]:b[i])) {
            if(partial_errors<8) std::printf("PART r=%u i=%u expected=%08x actual=%08x\n",r,i,i<2?a[i]:b[i],out[i]);
            ++partial_errors;
        }
    }
    cf_o_release(2);
    errors=full_errors+partial_errors;
    std::printf("SPM_FORWARD_TEST errors=%zu elements=16384 full=%zu partial=%zu\n",errors,full_errors,partial_errors);
    return errors?1:0;
}
