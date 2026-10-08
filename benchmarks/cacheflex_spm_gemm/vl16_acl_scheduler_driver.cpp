#ifdef ACL_SPM_VARIANT
#define VL_16
#include "kernels_tile_spm.hpp"
#else
#define main cacheflex_reference_v1_24x1vl_main
#include "v1_24x1vl.cpp"
#undef main
#endif

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include <gem5/m5ops.h>

using namespace arm_compute;

namespace {
constexpr int Threads = 8;
constexpr int TileM = 24;

struct GemmState
{
    int m, n, k, kc, iterations, tileN, mTiles, nTiles;
    __fp16 *a;
    __fp16 *b;
    __fp16 *c;
    __fp16 *aPacked[Threads];
    __fp16 *bPacked[Threads];
};

class GemmWindowKernel final : public ICPPKernel
{
  public:
    explicit GemmWindowKernel(GemmState &state) : s(state)
    {
        // Match arm_gemm's 2D scheduling path. DimX represents M tiles and
        // DimY represents N tiles; ACL's split_2d() selects the core grid.
        Window w;
        w.set(Window::DimX, Window::Dimension(0, s.mTiles, 1));
        w.set(Window::DimY, Window::Dimension(0, s.nTiles, 1));
        ICPPKernel::configure(w);
    }

    const char *name() const override
    {
#ifdef ACL_SPM_VARIANT
        return "cacheflex_spm_vl16_acl_window";
#else
        return "cache_vl16_acl_window";
#endif
    }

    void run_nd(const Window &window, const ThreadInfo &info,
                const Window &locator) override
    {
        const int worker = info.thread_id;
        const int mtBegin = window.x().start();
        const int mtEnd = window.x().end();
        const int ntBegin = window.y().start();
        const int ntEnd = window.y().end();
        const int mBegin = mtBegin * TileM;
        const int localRows =
            std::max(0, std::min(s.m, mtEnd * TileM) - mBegin);
        const int localBlocks = mtEnd - mtBegin;

        std::fprintf(stderr,
            "ACL_MAP worker=%d/%d m_tiles=[%d,%d) n_tiles=[%d,%d) "
            "locator=(%d/%d,%d/%d)\n",
            worker, info.num_threads, mtBegin, mtEnd, ntBegin, ntEnd,
            locator.x().start(), locator.x().end(), locator.y().start(),
            locator.y().end());

        for (int iteration = 0; iteration < s.iterations; ++iteration) {
            for (int k0 = 0; k0 < s.k; k0 += s.kc) {
                const int kc = std::min(s.kc, s.k - k0);
#ifdef ACL_SPM_VARIANT
                pack_A_tile(s.a, s.k, s.aPacked[worker], s.m, mBegin,
                            s.k, k0, localRows, kc);
#else
                pack_A_24row(s.a, s.k, s.aPacked[worker], s.m, mBegin,
                             s.k, k0, localRows, kc);
#endif
                for (int nt = ntBegin; nt < ntEnd; ++nt) {
                    const int nBegin = nt * s.tileN;
                    const int nc = std::min(s.tileN, s.n - nBegin);
#ifdef ACL_SPM_VARIANT
                    const __fp16 *panel = s.b +
                        static_cast<size_t>(k0) * s.n + nBegin;
                    if (nc == s.tileN) {
                        pack_B_tile_to_spm_4way(panel, s.n, kc, s.kc, 0);
                    } else {
                        pack_b(worker, k0, nBegin, nc, kc);
                        pack_B_tile_to_spm_4way(
                            s.bPacked[worker], s.tileN, kc, s.kc, 0);
                    }
                    spm_gemm_fused_tile(
                        s.aPacked[worker],
                        s.c + static_cast<size_t>(mBegin) * s.n + nBegin,
                        localBlocks, kc, s.n, k0 == 0, 0);
#else
                    pack_b(worker, k0, nBegin, nc, kc);
                    cache_gemm_fused_24x1VL(
                        s.aPacked[worker], s.bPacked[worker],
                        s.c + static_cast<size_t>(mBegin) * s.n + nBegin,
                        localBlocks, kc, s.n, k0 == 0);
#endif
                }
            }
        }
    }

  private:
    void pack_b(int worker, int k0, int nBegin, int nc, int kc)
    {
        std::fill(s.bPacked[worker],
                  s.bPacked[worker] + static_cast<size_t>(kc) * s.tileN,
                  static_cast<__fp16>(0));
        for (int kk = 0; kk < kc; ++kk)
            std::copy_n(s.b + static_cast<size_t>(k0 + kk) * s.n + nBegin,
                        nc, s.bPacked[worker] +
                            static_cast<size_t>(kk) * s.tileN);
    }

    GemmState &s;
};
} // namespace

int main(int argc, char **argv)
{
    GemmState s{};
    s.m = argc > 1 ? std::atoi(argv[1]) : 784;
    s.k = argc > 2 ? std::atoi(argv[2]) : 256;
    s.n = argc > 3 ? std::atoi(argv[3]) : 1024;
    s.iterations = argc > 4 ? std::atoi(argv[4]) : 1;
    s.kc = argc > 5 ? std::atoi(argv[5]) : 256;
    s.tileN = static_cast<int>(svcnth());
    s.mTiles = (s.m + TileM - 1) / TileM;
    s.nTiles = (s.n + s.tileN - 1) / s.tileN;
    if (s.m <= 0 || s.n <= 0 || s.k <= 0 || s.iterations <= 0 ||
        s.kc <= 0 || s.kc > 1024 || svcntb() != 256) {
        std::fprintf(stderr, "invalid W7/VL16 configuration\n");
        return 2;
    }

    const size_t aElems = static_cast<size_t>(s.m) * s.k;
    const size_t bElems = static_cast<size_t>(s.k) * s.n;
    const size_t paddedM = static_cast<size_t>(s.mTiles) * TileM;
    s.a = new (std::nothrow) __fp16[aElems];
    const size_t bAlloc = (bElems * sizeof(__fp16) + 63) & ~size_t(63);
    s.b = static_cast<__fp16 *>(std::aligned_alloc(64, bAlloc));
    s.c = new (std::nothrow) __fp16[paddedM * s.n]();
    for (int i = 0; i < Threads; ++i) {
        s.aPacked[i] = new (std::nothrow) __fp16[
            static_cast<size_t>(s.mTiles) * TileM * s.kc];
        s.bPacked[i] = new (std::nothrow) __fp16[
            static_cast<size_t>(s.kc) * s.tileN];
    }
    if (!s.a || !s.b || !s.c) {
        std::fprintf(stderr, "allocation failed\n");
        return 2;
    }
    for (int i = 0; i < Threads; ++i) {
        if (!s.aPacked[i] || !s.bPacked[i]) {
            std::fprintf(stderr, "worker allocation failed\n");
            return 2;
        }
    }

    const uint64_t aBytes = aElems * sizeof(__fp16);
    const uint64_t bBytes = bElems * sizeof(__fp16);
    if (m5_read_file(s.a, aBytes, 0) != aBytes ||
        m5_read_file(s.b, bBytes, aBytes) != bBytes) {
        std::fprintf(stderr, "m5_read_file failed\n");
        return 2;
    }

    NEScheduler::get().set_num_threads(Threads);
    GemmWindowKernel kernel(s);
    std::fprintf(stderr, "ACL_COMMON_ROI begin kernel=%s\n", kernel.name());
    m5_reset_stats(0, 0);
    m5_work_begin(0, 0);
    NEScheduler::get().schedule(
        &kernel, IScheduler::Hints(IScheduler::split_dimensions_all));
    m5_work_end(0, 0);
    m5_dump_stats(0, 0);
    std::fprintf(stderr, "ACL_COMMON_ROI end\n");

    double checksum = 0.0;
    int errors = 0;
    for (int row = 0; row < s.m; ++row) {
        for (int col = 0; col < s.n; ++col) {
            const float value = static_cast<float>(
                s.c[static_cast<size_t>(row) * s.n + col]);
            checksum += value;
            errors += std::fabs(value) > 1.0e-4f;
        }
    }
    std::printf("%s %s checksum=%.9f errors=%d\n", kernel.name(),
                errors == 0 ? "PASS" : "FAIL", checksum, errors);
    return errors == 0 ? 0 : 1;
}
