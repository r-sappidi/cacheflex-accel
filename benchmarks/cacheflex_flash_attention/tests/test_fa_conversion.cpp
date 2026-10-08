// Independent diagnostic for the conversion idiom used in FA softmax/helpers.
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
int main() {
    if(svcnth()!=128) return 2;
    alignas(64) __fp16 input[128];alignas(64) float output[64];
    const uint16_t bits[8]={0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800};
    for(int i=0;i<128;++i)std::memcpy(input+i,bits+i%8,2);
    auto pg=svptrue_b32();auto ph=svwhilelt_b16_u64(0,64);
    svst1_f32(pg,output,svcvt_f32_f16_z(pg,svld1_f16(ph,input)));
    unsigned errors=0;
    for(int i=0;i<64;++i)errors+=output[i]!=float(i%8+1);
    std::printf("FA_CONVERSION errors=%u first=%g,%g,%g,%g lane32=%g\n",errors,
        double(output[0]),double(output[1]),double(output[2]),double(output[3]),double(output[32]));
    return 0;
}
