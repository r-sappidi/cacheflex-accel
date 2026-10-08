#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/CPP/CPPTypes.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/Tensor.h"

#include <gem5/m5ops.h>

#include <cstdio>
#include <cstring>

using namespace arm_compute;

int
main()
{
    constexpr size_t M = 784;
    constexpr size_t N = 1024;
    constexpr size_t K = 256;
    constexpr unsigned Threads = 8;

    Tensor a;
    Tensor b;
    Tensor c;
    a.allocator()->init(TensorInfo(TensorShape(K, M), 1, DataType::F16));
    b.allocator()->init(TensorInfo(TensorShape(N, K), 1, DataType::F16));
    c.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F16));

    b.info()->set_are_values_constant(false);
    NEGEMM gemm;
    // ACL gates its SVE1 FP16-accumulate implementations on fast_math. This
    // matches the paper kernel's FP16 inputs, accumulation, and output.
    GEMMInfo info(false, false, false, 0, false, false,
                  GEMMLowpOutputStageInfo(), false, true);
    gemm.configure(&a, &b, nullptr, &c, 1.0f, 0.0f, info);

    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();
    std::memset(a.buffer(), 0, a.info()->total_size());
    std::memset(b.buffer(), 0, b.info()->total_size());
    std::memset(c.buffer(), 0, c.info()->total_size());

    NEScheduler::get().set_num_threads(Threads);
    std::fprintf(stderr, "ACL_ISA sve=%d sve2=%d fp16=%d\n",
                 CPUInfo::get().has_sve(), CPUInfo::get().has_sve2(),
                 CPUInfo::get().has_fp16());
    std::fprintf(stderr,
                 "ACL_ROI begin threads=%u M=%zu N=%zu K=%zu type=F16\n",
                 Threads, M, N, K);
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);
    gemm.run();
    m5_work_end(0, 0);
    std::fprintf(stderr, "ACL_ROI end\n");

    const auto *out = reinterpret_cast<const unsigned short *>(c.buffer());
    std::printf("ACL_W7_F16 done first=0x%04x\n", out[0]);
    return 0;
}
