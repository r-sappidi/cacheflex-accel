#include <arm_sve.h>
#include <cstdio>
#include <cstdint>
#include "../output_resident.hpp"

static inline svfloat32_t delayed(svfloat32_t x) {
    asm volatile(
        ".rept 24\n"
        "fadd %0.s, %0.s, %0.s\n"
        ".endr\n" : "+w"(x));
    return x;
}

int main() {
    if(svcntw()!=64) return 2;
    alignas(64) float backing[256]={}, actual[64];
    cf_o_claim(backing,4);
    size_t errors=0, checked=0;
    for(unsigned r=0;r<64;++r) {
        cf_o_store(backing,0,svdup_f32(-1.f));
        cf_o_store(backing,1,svdup_f32(3.f));
        asm volatile("dsb sy" ::: "memory");
        auto value=delayed(svdup_f32(float(r+1)));
        cf_o_store(backing,0,value);
        // Older store data is delayed; this load's slot is disjoint.
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,1));
        for(float x:actual) {errors+=x!=3.f;++checked;}
        // Same-slot load MUST NOT bypass the pending store's data.
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,0));
        for(float x:actual) {errors+=x!=float(r+1)*16777216.f;++checked;}

        // A late base operand cannot be read until its own producer is ready.
        auto late=delayed(svdup_f32(float(r&1)+1.f));
        unsigned row=unsigned(svlastb_f32(svptrue_b32(),late)/16777216.f)-1;
        cf_o_store(backing,row,svdup_f32(9.f));
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,row));
        for(float x:actual) {errors+=x!=9.f;++checked;}

        // Exercise both outcomes of a data-dependent branch; speculative
        // stores and their early metadata must be discarded on squash.
        if ((r*17)%7<3) cf_o_store(backing,2,svdup_f32(5.f));
        else cf_o_store(backing,2,svdup_f32(7.f));
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,2));
        for(float x:actual) {errors+=x!=((r*17)%7<3?5.f:7.f);++checked;}

        // Predicated store: full-range early metadata must be conservative.
        size_t addr=cf_o_slot(3,0);
        auto p=svwhilelt_b64_u64(0,8);
        auto raw=svdup_u64(0x41200000); // 10.f in each narrowed FP32 element
        auto all=svptrue_b64();
        cf_o_store(backing,3,svdup_f32(11.f));
        asm volatile("spm.st1ad %0.d, %1, [%2]\n"
                     ::"w"(raw),"Upl"(p),"r"(addr):"memory");
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,3));
        for(unsigned i=0;i<64;++i) {errors+=actual[i]!=(i<8?10.f:11.f);++checked;}
        // Immediate offset is scaled by the 128-byte memory vector length.
        addr=cf_o_slot(2,0)-128;
        auto immediateValue=svreinterpret_u32_f32(
            delayed(svdup_f32(10.f/16777216.f)));
        raw=svreinterpret_u64_u32(svzip1_u32(immediateValue,immediateValue));
        asm volatile("spm.st1ad %0.d, %1, [%2, #8, mul vl]\n"
                     ::"w"(raw),"Upl"(all),"r"(addr):"memory");
        svst1_f32(svptrue_b32(),actual,cf_o_load(backing,2));
        for(unsigned i=0;i<32;++i) {errors+=actual[i]!=10.f;++checked;}
    }
    cf_o_release(4);
    std::printf("SPM_EARLY_ADDRESS errors=%zu elements=%zu\n",errors,checked);
    return errors?1:0;
}
