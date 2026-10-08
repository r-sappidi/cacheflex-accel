#include "reduction.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
// Destination reuse, queue saturation, immediate RAW, DSB visibility, ordinary
// overwrite ordering, and SPMWB without an intervening explicit fence.
int main() {
    alignas(64) float zero[8*64]={}, expected[8][64]={}, out[64], backing[64]={};
    const auto p=svptrue_b32();
    if(svcntw()!=64)return 2;
    output_claim(zero,8);
    for(int round=0;round<64;++round)for(int row=0;row<8;++row) {
        float scale=(round%3)?1.f:.5f;
        float v=float((row+round)%7-3)*.125f;
        output_reduce(row,scale,svdup_n_f32(v));
        for(int i=0;i<64;++i)expected[row][i]=expected[row][i]*scale+v;
    }
    unsigned errors=0;
    for(int row=0;row<8;++row) {
        svst1_f32(p,out,output_load(row)); // immediate RAW, no explicit fence
        for(int i=0;i<64;++i)errors+=out[i]!=expected[row][i];
    }
    output_reduce(0,.5f,svdup_n_f32(2.f));
    output_store(0,svdup_n_f32(7.f)); // must follow the pending reduction
    output_reduce(0,.5f,svdup_n_f32(2.f)); // must follow the ordinary store
    svst1_f32(p,out,output_load(0));
    for(float x:out)errors+=x!=5.5f;
    output_reduce(1,0.f,svdup_n_f32(3.f));
    asm volatile("dsb sy":::"memory"); // early ACK must not finish a barrier
    svst1_f32(p,out,output_load(1));
    for(float x:out)errors+=x!=3.f;
    output_reduce(2,0.f,svdup_n_f32(9.f));
    for(unsigned w=0;w<4;++w) {
        auto slot=output_slot(2)+(uint64_t(w)<<16);float* dst=backing+16*w;
        asm volatile("SPMWB_64_IMM %0, [%1, #0]\n"::"r"(slot),"r"(dst):"memory");
    }
    asm volatile("dsb sy":::"memory");
    for(float x:backing)errors+=x!=9.f;
    // Drain before release is architectural, not an unbounded driver delay.
    output_reduce(3,1.f,svdup_n_f32(1.f));
    output_release(8);
    output_claim(zero,8);
    output_reduce(0,0.f,svdup_n_f32(11.f));
    svst1_f32(p,out,output_load(0));
    for(float x:out)errors+=x!=11.f;
    output_release(8);
    std::printf("REDUCTION_SMOKE %s errors=%u\n",errors?"FAIL":"PASS",errors);
    return errors?1:0;
}
