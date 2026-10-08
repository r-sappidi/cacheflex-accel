// Independent ordered FP32 dot products. Half-input products are exactly
// representable in FP32; -ffp-contract=off preserves rounding after each add.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
extern "C" void ordered_gemm(const float* a,const float* b,float* out,int m,int k,int n) {
 #pragma omp parallel for schedule(static) num_threads(4)
 for(int row=0;row<m;++row) {
  for(int col=0;col<n;col+=256) {
   const int end=std::min(n,col+256);
   for(int j=col;j<end;++j)out[(long)row*n+j]=0;
   for(int l=0;l<k;++l) {
    float x=a[(long)row*k+l];
    #pragma omp simd
    for(int j=col;j<end;++j)out[(long)row*n+j]+=x*b[(long)l*n+j];
   }
  }
 }
}

// Scalar emulation of the published SVE helper's FP32 polynomial and Arm
// reciprocal-estimate refinement, independent of the guest implementation.
// Inputs used here make 1+exp(-gate) positive and normal.
extern "C" void custom_activation(const float* g,const float* u,float* out,int n) {
 for(int i=0;i<n;++i) {
  float x=std::max(-88.f,std::min(88.f,-g[i]));
  float nf=std::nearbyint(x*1.44269504088896f);
  float r=std::fma(-nf,6.93147182e-1f,x);
  r=std::fma(-nf,-1.90465430e-9f,r);
  float p=1.f/120.f;
  p=std::fma(r,p,1.f/24.f);p=std::fma(r,p,1.f/6.f);
  p=std::fma(r,p,.5f);p=std::fma(r,p,1.f);p=std::fma(r,p,1.f);
  float d=std::ldexp(p,(int)nf)+1.f;
  int exponent;
  double m=std::frexp((double)d,&exponent);
  double estimate=std::floor(256.0/( (std::floor(m*512.0)+.5)/512.0 )+.5)/256.0;
  float inv=std::ldexp((float)estimate,-exponent);
  inv=std::fma(-d,inv,2.f)*inv;
  out[i]=(g[i]*inv)*u[i];
 }
}

// Scalar arithmetic oracle for ACL's SVE FP16 SWISH: FP32 exp polynomial,
// then FP16 add, reciprocal estimate/refinement and multiply. It deliberately
// does not replace the separate full-operator float64 mathematical error audit.
// Source: pinned ACL src/core/NEON/SVEMath.inl and activation/generic/sve/fp16_impl.h.
static float from_bits(uint32_t bits) {
 float value;std::memcpy(&value,&bits,sizeof(value));return value;
}
static float half(double value) {
 if(!std::isfinite(value)||value==0)return (float)value;
 int exponent;std::frexp(std::abs(value),&exponent);
 double spacing=std::ldexp(1.0,std::max(exponent-11,-24));
 double result=std::nearbyint(value/spacing)*spacing;
 if(std::abs(result)>=65536)return std::copysign(INFINITY,value);
 return (float)result;
}
extern "C" void acl_activation(const float* g,float* out,int count) {
 const float c1=from_bits(0x3f7ffff6),c2=from_bits(0x3efffedb);
 const float c3=from_bits(0x3e2aaf33),c4=from_bits(0x3d2b9f17);
 const float c5=from_bits(0x3c072010),shift=from_bits(0x4b00007f);
 for(int i=0;i<count;++i) {
  float x=-g[i];
  float z=std::fma(x,from_bits(0x3fb8aa3b),shift);
  float n=z-shift;
  float r=std::fma(n,from_bits(0xbf317200),x);
  r=std::fma(n,from_bits(0xb5bfbe8e),r);
  float r2=r*r,p1=c1*r;
  float p23=std::fma(c3,r,c2),p45=std::fma(c5,r,c4);
  float p2345=std::fma(p45,r2,p23);
  float p12345=std::fma(p2345,r2,p1);
  float scale=std::ldexp(1.f,(int)n);
  float e=half(std::fma(p12345,scale,scale));
  float d=half(1.0+e);
  int exponent;double m=std::frexp((double)d,&exponent);
  double estimate=std::floor(256.0/((std::floor(m*512.0)+.5)/512.0)+.5)/256.0;
  float inv=half(std::ldexp(estimate,-exponent));
  float step=half(2.0-(double)d*inv);
  inv=half((double)step*inv);
  out[i]=half((double)g[i]*inv);
 }
}
