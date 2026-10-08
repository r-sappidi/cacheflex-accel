#include "common_spm.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

// Identical guest on Classic and Ruby. Random cyclic pointer chains suppress
// sequential prefetch; warmup and allocation are excluded from each epoch.
// Width is the number of bytes actually requested, not architectural VL.
static __attribute__((noinline)) std::uint64_t chase(
    std::uint64_t p, unsigned steps, unsigned width, bool spm)
{
    const unsigned lanes=width/8;
    if (spm) {
        asm volatile("whilelo p0.d, xzr, %x[n]\n ptrue p1.d, vl1\n"
            "1:\n spm.ld1d z0.d, p0/z, [%x[p]]\n"
            "lastb %x[p], p1, z0.d\n subs %w[c], %w[c], #1\n b.ne 1b\n"
            : [p]"+r"(p),[c]"+r"(steps):[n]"r"(std::uint64_t(lanes))
            :"p0","p1","z0","cc","memory");
    } else {
        asm volatile("whilelo p0.d, xzr, %x[n]\n ptrue p1.d, vl1\n"
            "1: ld1d z0.d, p0/z, [%x[p]]\n"
            "lastb %x[p], p1, z0.d\n subs %w[c], %w[c], #1\n b.ne 1b\n"
            : [p]"+r"(p),[c]"+r"(steps):[n]"r"(std::uint64_t(lanes))
            :"p0","p1","z0","cc","memory");
    }
    return p;
}

#define STEP(I, OP) OP " z" #I ".d, p0/z, [%x[p" #I "]]\n lastb %x[p" #I "], p1, z" #I ".d\n"
#define EIGHT(OP) STEP(0,OP) STEP(1,OP) STEP(2,OP) STEP(3,OP) STEP(4,OP) STEP(5,OP) STEP(6,OP) STEP(7,OP)
#define STREAM(OP) asm volatile("whilelo p0.d, xzr, %x[n]\n ptrue p1.d, vl1\n1:\n" EIGHT(OP) \
    "subs %w[c], %w[c], #1\n b.ne 1b\n" \
    :[p0]"+r"(p[0]),[p1]"+r"(p[1]),[p2]"+r"(p[2]),[p3]"+r"(p[3]), \
     [p4]"+r"(p[4]),[p5]"+r"(p[5]),[p6]"+r"(p[6]),[p7]"+r"(p[7]),[c]"+r"(steps) \
    :[n]"r"(std::uint64_t(width/8)):"p0","p1","z0","z1","z2","z3","z4","z5","z6","z7","cc","memory")
static __attribute__((noinline)) void stream(std::uint64_t* p,unsigned steps,unsigned width,bool spm)
{
    if(spm){STREAM("spm.ld1d");}else{STREAM("ld1d");}
}

int main(int argc,char**argv)
{
    if(argc!=5) return 2;
    bool spm=std::strncmp(argv[1],"spm",3)==0;
    bool parallel=std::strstr(argv[1],"stream")!=nullptr;
    unsigned kib=std::strtoul(argv[2],nullptr,0);
    unsigned width=std::strtoul(argv[3],nullptr,0);
    unsigned rounds=std::strtoul(argv[4],nullptr,0);
    if((width!=64 && width!=128 && width!=256)||!kib||!rounds) return 2;
    unsigned nodes=spm?32:kib*1024/width;
    // MICRO's SPM packet covers the full architectural VL even with a narrow
    // predicate. Initialize all four ways so masked-off ways cannot trigger
    // its whole-request zero fallback; do not mistake this for width scaling.
    const unsigned storage_width=spm?256:width;
    AlignedBuffer<std::uint64_t> data(nodes*storage_width/8);
    std::vector<unsigned> order(nodes);
    for(unsigned i=0;i<nodes;++i) order[i]=i;
    std::uint64_t rng=20260922;
    for(unsigned i=nodes-1;i;i--){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;std::swap(order[i],order[rng%(i+1)]);}
    auto addr=[&](unsigned i){return spm?spm_addr(0,i):reinterpret_cast<std::uint64_t>(data.data()+i*width/8);};
    for(unsigned i=0;i<nodes;i++)for(unsigned j=0;j<storage_width/8;j++)
        data.data()[order[i]*storage_width/8+j]=addr(order[(i+1)%nodes]);
    asm volatile("dsb sy":::"memory");
    if(spm)for(unsigned i=0;i<nodes;i++)for(unsigned w=0;w<storage_width/64;w++){
        auto slot=spm_addr(w,i);auto src=data.data()+i*storage_width/8+w*8;
        asm volatile("SPMCP_64_IMM %0, [%1, #0]"::"r"(slot),"r"(src):"memory");
    }
    asm volatile("dsb sy":::"memory");
    auto start=addr(order[0]);auto p=chase(start,4*nodes,width,spm);
    if(p!=start) return 3;
    std::uint64_t pointers[8],expected[8];
    for(unsigned i=0;i<8;i++)pointers[i]=expected[i]=addr(order[i*nodes/8]);
    // Three repeated warm epochs; exact same instruction sequence/data.
    for(unsigned epoch=0;epoch<3;epoch++){
        m5_reset_stats(0,0);
        if(parallel)stream(pointers,rounds*nodes,width,spm);
        else p=chase(p,rounds*nodes,width,spm);
        m5_dump_stats(0,0);
        if(p!=start) return 4;
        for(unsigned i=0;i<8;i++)if(pointers[i]!=expected[i])return 5;
    }
    std::printf("TIMING_PROBE PASS mode=%s KiB=%u bytes=%u loads=%u epochs=3\n",
        argv[1],kib,width,rounds*nodes*(parallel?8:1));
    return 0;
}
