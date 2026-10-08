#include <arm_sve.h>
#include <cstdio>

// Isolate the FA driver's dense-array conversion, including tail predicates.
int main() {
    __fp16 input[256], old_half[256], fixed_half[256];
    float expected[128], old_float[128], fixed_float[128];
    unsigned failures=0, old_errors=0;
    const unsigned lanes=svcntw();
    for (unsigned n=1;n<=lanes;++n) {
        for(unsigned i=0;i<lanes;++i) {
            input[i]=(__fp16)(i+1); expected[i]=float(i+1);
        }
        auto ph=svwhilelt_b16_u64(0,n);
        auto ps=svwhilelt_b32_u64(0,n);
        auto h=svld1_f16(ph,input);
        svst1_f32(ps,old_float,svcvt_f32_f16_z(ps,h));
        svst1_f32(ps,fixed_float,svcvt_f32_f16_z(ps,svzip1_f16(h,h)));
        auto f=svld1_f32(ps,expected);
        auto narrowed=svcvt_f16_f32_z(ps,f);
        svst1_f16(ph,old_half,narrowed);
        svst1_f16(ph,fixed_half,svuzp1_f16(narrowed,narrowed));
        for(unsigned i=0;i<n;++i) {
            old_errors+=(old_float[i]!=expected[i])+(old_half[i]!=input[i]);
            failures+=(fixed_float[i]!=expected[i])+(fixed_half[i]!=input[i]);
        }
        if(n==lanes) {
            printf("widen original:");
            for(unsigned i=0;i<8 && i<n;++i) printf(" %g",old_float[i]);
            printf("\nwiden fixed:");
            for(unsigned i=0;i<8 && i<n;++i) printf(" %g",fixed_float[i]);
            printf("\n");
        }
    }
    printf("LANE_CHECK lanes=%u original_errors=%u fixed_errors=%u\n",lanes,old_errors,failures);
    return failures || !old_errors;
}
