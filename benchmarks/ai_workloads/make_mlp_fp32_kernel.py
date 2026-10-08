"""Matched 8-row SVE FP32-accumulation kernel, half inputs and staged B.
This is a precision-study contender, not a claim of optimal FP32 assembly.
"""
from pathlib import Path
import sys
out=Path(sys.argv[1])
s='''// Generated matched arithmetic; sixteen FP32 accumulator vectors.
static void mlp_fp32_tile(const __fp16* a,const __fp16* b,float* c,int k,int ldc,bool first) {
 auto pg=svptrue_b32(); const int half=svcntw();
'''
for r in range(8):
    for h in range(2):
        s+=f' auto c{r}_{h}=first?svdup_n_f32(0):svld1_f32(pg,c+{r}*ldc+{h}*half);\n'
s+=''' for(int ki=0;ki<k;++ki) {
  svfloat16_t hb;
#ifdef IDENTICAL_CACHE_BASELINE
  hb=svld1_f16(svptrue_b16(),b+(size_t)ki*svcnth());
#else
  uint64_t slot=(uint64_t)ki<<6;
  asm volatile("ptrue p1.d\\nspm.ld1qd %0.d, p1/z, [%1]\\n" : "=w"(hb) : "r"(slot) : "p1","memory");
#endif
  auto b0=svcvt_f32_f16_x(pg,svzip1_f16(hb,hb));
  auto b1=svcvt_f32_f16_x(pg,svzip2_f16(hb,hb));
'''
for r in range(8):
    s+=f'  float a{r}=a[ki*8+{r}];\n'
    for h in range(2):s+=f'  c{r}_{h}=svmla_n_f32_x(pg,c{r}_{h},b{h},a{r});\n'
s+=' }\n'
for r in range(8):
    for h in range(2):s+=f' svst1_f32(pg,c+{r}*ldc+{h}*half,c{r}_{h});\n'
s+='}\n'
out.write_text(s)
