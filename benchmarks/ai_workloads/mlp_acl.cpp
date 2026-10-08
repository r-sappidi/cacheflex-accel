// Thin ACL operator composition: no custom compute, packing, or scheduler.
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include <gem5/m5ops.h>
#include <arm_sve.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace arm_compute;

static void init(Tensor &a, size_t cols, size_t rows) {
    a.allocator()->init(TensorInfo(TensorShape(cols, rows), 1, DataType::F16));
}
static void read_tensor(Tensor &a, size_t cols, size_t rows, uint64_t &offset) {
    for (size_t r=0; r<rows; ++r) {
        auto *p=a.ptr_to_element(Coordinates(0,r));
        if (m5_read_file(p, cols*2, offset)!=cols*2) std::exit(3);
        offset+=cols*2;
    }
}
static void dump_tensor(Tensor &a, size_t cols, size_t rows, const char *name) {
    for(size_t r=0;r<rows;++r) {
        if(m5_write_file(a.ptr_to_element(Coordinates(0,r)),cols*2,r*cols*2,name)!=cols*2)
            std::exit(5);
    }
}
int main(int argc, char **argv) {
    setbuf(stdout,nullptr);
    if (argc!=5 || svcnth()!=128) return 2;
    const int t=atoi(argv[1]), d=atoi(argv[2]), f=atoi(argv[3]);
    const bool warm=std::strcmp(argv[4],"warm")==0;
    if(t<=0 || d<=0 || f<=0 || (!warm && std::strcmp(argv[4],"first"))) return 2;
    // Never force an experimental local ACL kernel-filter extension.
    if(std::getenv("ACL_GEMM_KERNEL_FILTER")) return 2;
    NEScheduler::get().set_num_threads(8);
    Tensor x,wg,wu,wd,gate,up,act,hidden,y;
    init(x,d,t); init(wg,f,d); init(wu,f,d); init(wd,d,f);
    init(gate,f,t); init(up,f,t); init(act,f,t); init(hidden,f,t); init(y,d,t);
    for(auto *w : {&wg,&wu,&wd}) w->info()->set_are_values_constant(true);
    NEGEMM gate_gemm,up_gemm,down_gemm;
    // Constant-weight preparation belongs to the first run; FP16 SVE1 path.
    GEMMInfo info(false,false,true,0,false,false,GEMMLowpOutputStageInfo(),false,true);
    gate_gemm.configure(&x,&wg,nullptr,&gate,1.f,0.f,info);
    up_gemm.configure(&x,&wu,nullptr,&up,1.f,0.f,info);
    NEActivationLayer silu;
    silu.configure(&gate,&act,ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH,1.f));
    NEPixelWiseMultiplication multiply;
    multiply.configure(&act,&up,&hidden,1.f,ConvertPolicy::SATURATE,RoundingPolicy::TO_ZERO);
    down_gemm.configure(&hidden,&wd,nullptr,&y,1.f,0.f,info);
    for(auto *a : {&x,&wg,&wu,&wd,&gate,&up,&act,&hidden,&y}) {
        a->allocator()->allocate(); std::memset(a->buffer(),0,a->info()->total_size());
    }
    uint64_t offset=0;
    read_tensor(x,d,t,offset); read_tensor(wg,f,d,offset);
    read_tensor(wu,f,d,offset); read_tensor(wd,d,f,offset);
    std::vector<__fp16> ref(size_t(t)*d);
    if(m5_read_file(ref.data(),ref.size()*2,offset)!=ref.size()*2) return 3;
    auto run=[&] {gate_gemm.run(); up_gemm.run(); silu.run(); multiply.run(); down_gemm.run();};
    printf("ACL_ISA sve=%d sve2=%d fp16=%d vl_bits=%zu threads=%u\n",
           CPUInfo::get().has_sve(),CPUInfo::get().has_sve2(),CPUInfo::get().has_fp16(),
           svcntb()*8,NEScheduler::get().num_threads());
    // A full unmeasured invocation warms ACL workspaces, weights and caches.
    // This is explicitly NOT a cold-cache prepacked measurement.
    if(warm) run();
    printf("MLP_ROI begin backend=ACL mode=%s T=%d D=%d F=%d\n",argv[4],t,d,f);
    m5_reset_stats(0,0); m5_work_begin(0,0);
    run();
    m5_work_end(0,0); m5_dump_stats(0,0);
    puts("MLP_ROI end");
    // Post-ROI diagnostics permit independent per-operator rounding checks.
    dump_tensor(gate,f,t,"gate.bin"); dump_tensor(up,f,t,"up.bin");
    dump_tensor(act,f,t,"silu.bin"); dump_tensor(hidden,f,t,"hidden.bin");
    dump_tensor(y,d,t,"output.bin");
    size_t errors=0,nonfinite=0; double maxerr=0,sumerr2=0,sumref2=0,checksum=0;
    uint64_t hash=1469598103934665603ULL;
    for(int r=0;r<t;++r) {
        const auto *row=reinterpret_cast<const __fp16*>(y.ptr_to_element(Coordinates(0,r)));
        for(int c=0;c<d;++c) {
            float actual=row[c], expected=ref[size_t(r)*d+c];
            double e=std::abs(actual-expected); maxerr=std::max(maxerr,e);
            sumerr2+=e*e; sumref2+=double(expected)*expected;
            nonfinite+=!std::isfinite(actual);
            errors+=!std::isfinite(actual)||e>0.000002+0.01*std::abs(expected);
            checksum+=actual; uint16_t bits; std::memcpy(&bits,row+c,2);
            hash=(hash^bits)*1099511628211ULL;
        }
    }
    printf("MLP_RESULT backend=ACL mode=%s T=%d D=%d F=%d hash=%016llx checksum=%.12g legacy_max_abs=%.9g legacy_relative_l2=%.9g legacy_errors=%zu nonfinite=%zu CHECK_HOST\n",
           argv[4],t,d,f,(unsigned long long)hash,checksum,maxerr,
           std::sqrt(sumerr2/std::max(sumref2,1e-30)),errors,nonfinite);
    return nonfinite?1:0;
}
