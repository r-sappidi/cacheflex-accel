#ifdef IDENTICAL_CACHE_BASELINE
// Import the paper's authoritative cache-side 24x1VL implementation into this
// translation unit. Renaming its standalone entry point lets this multicore
// driver call the exact pack_A_24row/cache_gemm_fused_24x1VL routines.
#define main cacheflex_reference_v1_24x1vl_main
#include "v1_24x1vl.cpp"
#undef main
#else
#define VL_16
#include "kernels_tile_spm.hpp"
#ifdef CACHEFLEX_MULTICAST
#include "multicast_spm_kernel.hpp"
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <pthread.h>
#include <arm_neon.h>

#include <gem5/m5ops.h>

namespace {

constexpr int NumThreads = 8;
#ifndef CACHEFLEX_M_PARTITIONS
#define CACHEFLEX_M_PARTITIONS 2
#endif
constexpr int MPartitions = CACHEFLEX_M_PARTITIONS;
constexpr int NPartitions = NumThreads / MPartitions;
static_assert(MPartitions > 0 && NumThreads % MPartitions == 0);
constexpr int TileM = 24;

template <int Parties>
class alignas(64) SpinBarrier
{
  public:
    void wait(bool &localSense)
    {
        localSense = !localSense;
        if (arrivals.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            arrivals.store(Parties, std::memory_order_relaxed);
            sense.store(localSense, std::memory_order_release);
            return;
        }
        while (sense.load(std::memory_order_acquire) != localSense)
            asm volatile("yield" ::: "memory");
    }

  private:
    std::atomic<int> arrivals{Parties};
    std::atomic<bool> sense{false};
};

struct Context
{
    enum class Operator { Gemm, ConvSilu, DiffusionGeluResidual, LoraResidual };
    int m;
    int n;
    int k;
    int iterations;
    int kc;
    int tileN;
    int mTiles;
    int nTiles;
    __fp16 *a;
    __fp16 *b;
    __fp16 *c;
    __fp16 *expected;
    __fp16 *bias;
    __fp16 *residual;
    __fp16 *loraDown;
    __fp16 *loraUp;
    float *loraTmp;
    int loraRank{16};
    Operator op{Operator::Gemm};
    __fp16 *aPacked[NumThreads];
    __fp16 *bPad[NumThreads];
    struct PhaseCounters {
        uint64_t packA{0}, stageB{0}, kernel{0};
        uint64_t loopOther{0}, finalWait{0}, total{0};
    } phase[NumThreads];
    SpinBarrier<NumThreads> barrier;
    SpinBarrier<NPartitions> loraBarrier[MPartitions];
#ifdef CACHEFLEX_MULTICAST
    SpinBarrier<MPartitions> multicastBarrier[NPartitions];
#endif
};

inline float activate(float x, Context::Operator op)
{
    if (op == Context::Operator::ConvSilu)
        return x / (1.0f + std::exp(-x));
    if (op == Context::Operator::DiffusionGeluResidual) {
        constexpr float SqrtTwoOverPi = 0.7978845608028654f;
        return 0.5f * x *
            (1.0f + std::tanh(SqrtTwoOverPi * (x + 0.044715f*x*x*x)));
    }
    return x;
}

struct ThreadArg
{
    Context *ctx;
    int id;
    bool barrierSense{false};
};

// Pack one 24-row block as three independent 8xK transposes. Processing eight
// K values at once turns 192 scalar loads/stores into 24 vector loads/stores
// while preserving the kernel's [k][rows 0..23] layout exactly.
void packA24Neon(const __fp16 *a, int lda, __fp16 *packed, int m,
                 int mBegin, int localRows, int kBegin, int kc)
{
    alignas(16) uint16_t zeros[8]{};
    const int blocks = (localRows + TileM - 1) / TileM;
    for (int mb = 0; mb < blocks; ++mb) {
        const int rowBase = mBegin + mb * TileM;
        __fp16 *dst = packed + static_cast<size_t>(mb) * kc * TileM;
        for (int group = 0; group < 3; ++group) {
            const uint16_t *rows[8];
            for (int r = 0; r < 8; ++r) {
                const int row = rowBase + group * 8 + r;
                rows[r] = row < m
                    ? reinterpret_cast<const uint16_t *>(
                          a + static_cast<size_t>(row) * lda + kBegin)
                    : zeros;
            }

            int k = 0;
            for (; k + 8 <= kc; k += 8) {
                const uint16x8_t r0 = vld1q_u16(rows[0] + k);
                const uint16x8_t r1 = vld1q_u16(rows[1] + k);
                const uint16x8_t r2 = vld1q_u16(rows[2] + k);
                const uint16x8_t r3 = vld1q_u16(rows[3] + k);
                const uint16x8_t r4 = vld1q_u16(rows[4] + k);
                const uint16x8_t r5 = vld1q_u16(rows[5] + k);
                const uint16x8_t r6 = vld1q_u16(rows[6] + k);
                const uint16x8_t r7 = vld1q_u16(rows[7] + k);

                const uint16x8x2_t h01 = vtrnq_u16(r0, r1);
                const uint16x8x2_t h23 = vtrnq_u16(r2, r3);
                const uint16x8x2_t h45 = vtrnq_u16(r4, r5);
                const uint16x8x2_t h67 = vtrnq_u16(r6, r7);
                const uint32x4x2_t w0 = vtrnq_u32(
                    vreinterpretq_u32_u16(h01.val[0]),
                    vreinterpretq_u32_u16(h23.val[0]));
                const uint32x4x2_t w1 = vtrnq_u32(
                    vreinterpretq_u32_u16(h01.val[1]),
                    vreinterpretq_u32_u16(h23.val[1]));
                const uint32x4x2_t w2 = vtrnq_u32(
                    vreinterpretq_u32_u16(h45.val[0]),
                    vreinterpretq_u32_u16(h67.val[0]));
                const uint32x4x2_t w3 = vtrnq_u32(
                    vreinterpretq_u32_u16(h45.val[1]),
                    vreinterpretq_u32_u16(h67.val[1]));
                const uint64x2_t w00 = vreinterpretq_u64_u32(w0.val[0]);
                const uint64x2_t w01 = vreinterpretq_u64_u32(w0.val[1]);
                const uint64x2_t w10 = vreinterpretq_u64_u32(w1.val[0]);
                const uint64x2_t w11 = vreinterpretq_u64_u32(w1.val[1]);
                const uint64x2_t w20 = vreinterpretq_u64_u32(w2.val[0]);
                const uint64x2_t w21 = vreinterpretq_u64_u32(w2.val[1]);
                const uint64x2_t w30 = vreinterpretq_u64_u32(w3.val[0]);
                const uint64x2_t w31 = vreinterpretq_u64_u32(w3.val[1]);
                const uint16x8_t cols[8] = {
                    vreinterpretq_u16_u64(vtrn1q_u64(w00, w20)),
                    vreinterpretq_u16_u64(vtrn1q_u64(w10, w30)),
                    vreinterpretq_u16_u64(vtrn1q_u64(w01, w21)),
                    vreinterpretq_u16_u64(vtrn1q_u64(w11, w31)),
                    vreinterpretq_u16_u64(vtrn2q_u64(w00, w20)),
                    vreinterpretq_u16_u64(vtrn2q_u64(w10, w30)),
                    vreinterpretq_u16_u64(vtrn2q_u64(w01, w21)),
                    vreinterpretq_u16_u64(vtrn2q_u64(w11, w31))};
                for (int col = 0; col < 8; ++col)
                    vst1q_u16(reinterpret_cast<uint16_t *>(
                        dst + static_cast<size_t>(k + col) * TileM +
                        group * 8), cols[col]);
            }
            for (; k < kc; ++k)
                for (int r = 0; r < 8; ++r)
                    dst[static_cast<size_t>(k) * TileM + group * 8 + r] =
                        reinterpret_cast<const __fp16 *>(rows[r])[k];
        }
    }
}

void *worker(void *opaque)
{
    auto *arg = static_cast<ThreadArg *>(opaque);
    Context &ctx = *arg->ctx;
    const int mPart = arg->id / NPartitions;
    const int nPart = arg->id % NPartitions;

    // Partition on micro-kernel boundaries so no two threads can update the
    // same 24x128 C tile.  For W7 this gives 16 and 17 M tiles, while every N
    // partition receives exactly two 128-column panels.
    const int mtBegin = ctx.mTiles * mPart / MPartitions;
    const int mtEnd = ctx.mTiles * (mPart + 1) / MPartitions;
    const int ntBegin = ctx.nTiles * nPart / NPartitions;
    const int ntEnd = ctx.nTiles * (nPart + 1) / NPartitions;
    const int mBegin = mtBegin * TileM;
    const int localRows = std::max(0, std::min(ctx.m, mtEnd * TileM) - mBegin);
    const int localBlocks = mtEnd - mtBegin;

    std::fprintf(stderr,
        "GEMM_MAP core=%d m_part=%d n_part=%d m_tiles=[%d,%d) "
        "n_tiles=[%d,%d) rows=[%d,%d) cols=[%d,%d)\n",
        arg->id, mPart, nPart, mtBegin, mtEnd, ntBegin, ntEnd,
        mBegin, mBegin + localRows, ntBegin * ctx.tileN,
        std::min(ctx.n, ntEnd * ctx.tileN));

    ctx.barrier.wait(arg->barrierSense);
    if (arg->id == 0) {
        std::fprintf(stderr, "GEMM_SYNC all_threads_ready\n");
        m5_reset_stats(0, 0);
        m5_work_begin(0, 0);
    }
    ctx.barrier.wait(arg->barrierSense);
    if (arg->id == 0)
        std::fprintf(stderr, "GEMM_ROI begin\n");

#ifdef CACHEFLEX_PHASE_PROFILE
    const uint64_t workStart = m5_rpns();
#endif

    for (int iteration = 0; iteration < ctx.iterations; ++iteration) {
        for (int k0 = 0; k0 < ctx.k; k0 += ctx.kc) {
            const int kc = std::min(ctx.kc, ctx.k - k0);

#ifdef CACHEFLEX_PHASE_PROFILE
            uint64_t phaseStart = m5_rpns();
#endif
            packA24Neon(ctx.a, ctx.k, ctx.aPacked[arg->id], ctx.m,
                        mBegin, localRows, k0, kc);
#ifdef CACHEFLEX_PHASE_PROFILE
            ctx.phase[arg->id].packA += m5_rpns() - phaseStart;
#endif

#if defined(CACHEFLEX_MULTICAST) && defined(CACHEFLEX_MULTICAST_WAY_CLAIM)
            if (iteration == 0 && k0 == 0) {
                uint8_t claimMask = 0;
                for (int mp = 0; mp < MPartitions; ++mp)
                    claimMask |= static_cast<uint8_t>(
                        1u << (mp * NPartitions + nPart));
                if (mPart == 0)
                    claim_B_spm_4way_multicast(ctx.kc, claimMask);
                ctx.multicastBarrier[nPart].wait(arg->barrierSense);
            }
#endif

            // There is deliberately no inner barrier.  Each thread copies
            // only its N partition into its private SPM and consumes it for
            // its unique (M partition, N partition) output rectangle.
            for (int nt = ntBegin; nt < ntEnd; ++nt) {
                const int nBegin = nt * ctx.tileN;
                const int nc = std::min(ctx.tileN, ctx.n - nBegin);
                const __fp16 *bPanel = ctx.b +
                    static_cast<size_t>(k0) * ctx.n + nBegin;

#ifdef IDENTICAL_CACHE_BASELINE
#ifdef CACHEFLEX_PHASE_PROFILE
                phaseStart = m5_rpns();
#endif
                // Match the reference cache driver: B is packed into a
                // contiguous Kx1VL panel and consumed through coherent loads.
                std::fill(ctx.bPad[arg->id],
                          ctx.bPad[arg->id] +
                              static_cast<size_t>(kc) * ctx.tileN,
                          static_cast<__fp16>(0));
                for (int kk = 0; kk < kc; ++kk)
                    std::copy_n(
                        ctx.b + static_cast<size_t>(k0 + kk) * ctx.n + nBegin,
                        nc,
                        ctx.bPad[arg->id] +
                            static_cast<size_t>(kk) * ctx.tileN);
#ifdef CACHEFLEX_PHASE_PROFILE
                const uint64_t kernelStart = m5_rpns();
                ctx.phase[arg->id].stageB += kernelStart - phaseStart;
#endif
                if (localBlocks != 0) {
                    cache_gemm_fused_24x1VL(
                        ctx.aPacked[arg->id], ctx.bPad[arg->id],
                        ctx.c + static_cast<size_t>(mBegin) * ctx.n + nBegin,
                        localBlocks, kc, ctx.n, k0 == 0);
                }
#ifdef CACHEFLEX_PHASE_PROFILE
                ctx.phase[arg->id].kernel += m5_rpns() - kernelStart;
#endif
#else
#ifdef CACHEFLEX_PHASE_PROFILE
                phaseStart = m5_rpns();
#endif
#ifdef CACHEFLEX_MULTICAST
                uint8_t multicastMask = 0;
                for (int mp = 0; mp < MPartitions; ++mp)
                    multicastMask |= static_cast<uint8_t>(
                        1u << (mp * NPartitions + nPart));
                const bool bootstrapPanel = nt == ntBegin;
#ifdef CACHEFLEX_MULTICAST_BOOTSTRAP
                if (bootstrapPanel) {
                    // Establish every private slot through the validated
                    // coherent single-destination path. Later multicast
                    // panels overwrite existing SPM entries and therefore
                    // cannot race dynamic cache-to-SPM way conversion.
                    if (nc == ctx.tileN) {
                        pack_B_tile_to_spm_4way(
                            bPanel, ctx.n, kc, ctx.kc, 0);
                    } else {
                        std::fill(ctx.bPad[arg->id],
                                  ctx.bPad[arg->id] +
                                      static_cast<size_t>(kc) * ctx.tileN,
                                  static_cast<__fp16>(0));
                        for (int kk = 0; kk < kc; ++kk)
                            std::copy_n(
                                ctx.b + static_cast<size_t>(k0 + kk) * ctx.n +
                                    nBegin,
                                nc,
                                ctx.bPad[arg->id] +
                                    static_cast<size_t>(kk) * ctx.tileN);
                        pack_B_tile_to_spm_4way(
                            ctx.bPad[arg->id], ctx.tileN, kc, ctx.kc, 0);
                    }
                } else
#endif
                if (mPart == 0) {
                    const __fp16 *multicastSource = bPanel;
                    size_t multicastStride = ctx.n;
                    if (nc != ctx.tileN) {
                        std::fill(ctx.bPad[arg->id],
                                  ctx.bPad[arg->id] +
                                      static_cast<size_t>(kc) * ctx.tileN,
                                  static_cast<__fp16>(0));
                        for (int kk = 0; kk < kc; ++kk)
                            std::copy_n(
                                ctx.b + static_cast<size_t>(k0 + kk) * ctx.n +
                                    nBegin,
                                nc,
                                ctx.bPad[arg->id] +
                                    static_cast<size_t>(kk) * ctx.tileN);
                        multicastSource = ctx.bPad[arg->id];
                        multicastStride = ctx.tileN;
                    }
                    pack_B_tile_to_spm_4way_multicast(
                        multicastSource, multicastStride, kc, multicastMask);
                }
                ctx.multicastBarrier[nPart].wait(arg->barrierSense);
#else
                if (nc == ctx.tileN) {
                    pack_B_tile_to_spm_4way(
                        bPanel, ctx.n, kc, ctx.kc, 0);
                } else {
                    std::fill(ctx.bPad[arg->id],
                              ctx.bPad[arg->id] +
                                  static_cast<size_t>(kc) * ctx.tileN,
                              static_cast<__fp16>(0));
                    for (int kk = 0; kk < kc; ++kk)
                        std::copy_n(
                            ctx.b + static_cast<size_t>(k0 + kk) * ctx.n +
                                nBegin,
                            nc,
                            ctx.bPad[arg->id] +
                                static_cast<size_t>(kk) * ctx.tileN);
                    pack_B_tile_to_spm_4way(
                        ctx.bPad[arg->id], ctx.tileN, kc, ctx.kc, 0);
                }
#endif

#ifdef CACHEFLEX_PHASE_PROFILE
                const uint64_t kernelStart = m5_rpns();
                ctx.phase[arg->id].stageB += kernelStart - phaseStart;
#endif
                if (localBlocks != 0) {
                    spm_gemm_fused_tile(
                        ctx.aPacked[arg->id],
                        ctx.c + static_cast<size_t>(mBegin) * ctx.n + nBegin,
                        localBlocks, kc, ctx.n, k0 == 0, 0);
                }
#ifdef CACHEFLEX_MULTICAST
                // No producer may overwrite a shared private-SPM panel until
                // every M partition in this N group has consumed it.
                ctx.multicastBarrier[nPart].wait(arg->barrierSense);
#endif
#ifdef CACHEFLEX_PHASE_PROFILE
                ctx.phase[arg->id].kernel += m5_rpns() - kernelStart;
#endif
#endif
            }
        }

        // Model-level fused epilogue. Each worker touches only its existing
        // output rectangle, so this adds no synchronization or ownership
        // difference between the coherent and CacheFlex implementations.
        if (ctx.op != Context::Operator::Gemm) {
            if (ctx.op == Context::Operator::LoraResidual) {
                // Threads in an M partition share X*A. One producer computes
                // it, then both N partitions consume disjoint columns of the
                // rank-16 up projection.
                if (nPart == 0) {
                    for (int row = mBegin; row < mBegin + localRows; ++row) {
                        for (int r = 0; r < ctx.loraRank; ++r) {
                            float sum = 0.0f;
                            for (int kk = 0; kk < ctx.k; ++kk)
                                sum += static_cast<float>(ctx.a[
                                           static_cast<size_t>(row) * ctx.k + kk]) *
                                       static_cast<float>(ctx.loraDown[
                                           static_cast<size_t>(kk) * ctx.loraRank + r]);
                            ctx.loraTmp[static_cast<size_t>(row) * ctx.loraRank + r] = sum;
                        }
                    }
                }
                ctx.loraBarrier[mPart].wait(arg->barrierSense);
                for (int row = mBegin; row < mBegin + localRows; ++row) {
                    for (int nt = ntBegin; nt < ntEnd; ++nt) {
                        const int nBegin = nt * ctx.tileN;
                        const int nEnd = std::min(ctx.n, nBegin + ctx.tileN);
                        for (int col = nBegin; col < nEnd; ++col) {
                            float update = 0.0f;
                            for (int r = 0; r < ctx.loraRank; ++r)
                                update += ctx.loraTmp[
                                              static_cast<size_t>(row) * ctx.loraRank + r] *
                                          static_cast<float>(ctx.loraUp[
                                              static_cast<size_t>(r) * ctx.n + col]);
                            const size_t index = static_cast<size_t>(row) * ctx.n + col;
                            ctx.c[index] = static_cast<__fp16>(
                                static_cast<float>(ctx.c[index]) + update);
                        }
                    }
                }
                ctx.loraBarrier[mPart].wait(arg->barrierSense);
                continue;
            }
            for (int row = mBegin; row < mBegin + localRows; ++row) {
                for (int nt = ntBegin; nt < ntEnd; ++nt) {
                    const int nBegin = nt * ctx.tileN;
                    const int nEnd = std::min(ctx.n, nBegin + ctx.tileN);
                    for (int col = nBegin; col < nEnd; ++col) {
                        const size_t index = static_cast<size_t>(row) * ctx.n + col;
                        float value = static_cast<float>(ctx.c[index]) +
                                      static_cast<float>(ctx.bias[col]);
                        value = activate(value, ctx.op);
                        if (ctx.op == Context::Operator::DiffusionGeluResidual)
                            value += static_cast<float>(ctx.residual[index]);
                        ctx.c[index] = static_cast<__fp16>(value);
                    }
                }
            }
        }
    }

    // End the ROI only after all eight disjoint rectangles are complete.
#ifdef CACHEFLEX_PHASE_PROFILE
    const uint64_t workEnd = m5_rpns();
    auto &phase = ctx.phase[arg->id];
    const uint64_t classified = phase.packA + phase.stageB + phase.kernel;
    phase.loopOther = workEnd - workStart - classified;
#endif
    ctx.barrier.wait(arg->barrierSense);
#ifdef CACHEFLEX_PHASE_PROFILE
    const uint64_t syncEnd = m5_rpns();
    phase.finalWait = syncEnd - workEnd;
    phase.total = syncEnd - workStart;
#endif
    if (arg->id == 0) {
        m5_work_end(0, 0);
        m5_dump_stats(0, 0);
        std::fprintf(stderr, "GEMM_ROI end\n");
    }
    ctx.barrier.wait(arg->barrierSense);
#if defined(CACHEFLEX_MULTICAST) && defined(CACHEFLEX_MULTICAST_WAY_CLAIM)
    // Rejoin the coherent-only software stack with all dynamically claimed
    // ways returned. This cleanup is deliberately outside the measured ROI.
    release_B_spm_4way(ctx.kc);
    ctx.barrier.wait(arg->barrierSense);
#endif
    return nullptr;
}

} // namespace

int main(int argc, char **argv)
{
#ifdef CACHEFLEX_VALIDATE_PACK
    alignas(16) __fp16 testA[24 * 9]{};
    alignas(16) __fp16 testPacked[24 * 9]{};
    auto *testABits = reinterpret_cast<uint16_t *>(testA);
    const auto *testPackedBits =
        reinterpret_cast<const uint16_t *>(testPacked);
    for (int row = 0; row < 24; ++row)
        for (int k = 0; k < 9; ++k)
            testABits[row * 9 + k] = static_cast<uint16_t>(row * 100 + k);
    packA24Neon(testA, 9, testPacked, 24, 0, 24, 0, 9);
    for (int k = 0; k < 9; ++k)
        for (int row = 0; row < 24; ++row)
            if (testPackedBits[k * 24 + row] !=
                static_cast<uint16_t>(row * 100 + k)) {
                std::fprintf(stderr, "PACK_A_NEON FAIL k=%d row=%d\n", k,
                             row);
                return 1;
            }
    std::fprintf(stderr, "PACK_A_NEON PASS\n");
    return 0;
#endif
    const int m = argc > 1 ? std::atoi(argv[1]) : 784;
    const int k = argc > 2 ? std::atoi(argv[2]) : 256;
    const int n = argc > 3 ? std::atoi(argv[3]) : 1024;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 1;
    const int kc = argc > 5 ? std::atoi(argv[5]) : 256;
    const bool checkReference = argc > 6 && std::atoi(argv[6]) != 0;
    const char *operatorName = argc > 7 ? argv[7] : "gemm";

    const int tileN = static_cast<int>(svcnth());
    if (m <= 0 || n <= 0 || k <= 0 || iterations <= 0 ||
        kc <= 0 || kc > 1024 || svcntb() != 256) {
        std::fprintf(stderr,
            "require positive dimensions/iterations, KC<=1024, and VL=16 "
            "(2048-bit SVE); got sve_bits=%d\n",
            static_cast<int>(svcntb() * 8));
        return 2;
    }

    Context ctx{};
    ctx.m = m;
    ctx.n = n;
    ctx.k = k;
    ctx.iterations = iterations;
    ctx.kc = kc;
    ctx.tileN = tileN;
    ctx.mTiles = (m + TileM - 1) / TileM;
    ctx.nTiles = (n + tileN - 1) / tileN;
    if (std::strcmp(operatorName, "gemm") == 0)
        ctx.op = Context::Operator::Gemm;
    else if (std::strcmp(operatorName, "conv_silu") == 0)
        ctx.op = Context::Operator::ConvSilu;
    else if (std::strcmp(operatorName, "diffusion_gelu_residual") == 0)
        ctx.op = Context::Operator::DiffusionGeluResidual;
    else if (std::strcmp(operatorName, "lora_residual") == 0)
        ctx.op = Context::Operator::LoraResidual;
    else {
        std::fprintf(stderr, "unknown operator: %s\n", operatorName);
        return 2;
    }
    if (ctx.mTiles < MPartitions || ctx.nTiles < NPartitions) {
        std::fprintf(stderr, "matrix has too few tiles for a 2x4 partition\n");
        return 2;
    }

    const size_t aElems = static_cast<size_t>(m) * k;
    const size_t bElems = static_cast<size_t>(k) * n;
    const size_t paddedM = static_cast<size_t>(ctx.mTiles) * TileM;
    ctx.a = new (std::nothrow) __fp16[aElems];
    const size_t bBytes = (bElems * sizeof(__fp16) + 63) & ~size_t(63);
    ctx.b = static_cast<__fp16 *>(std::aligned_alloc(64, bBytes));
    ctx.c = new (std::nothrow) __fp16[paddedM * n]();
    ctx.expected = checkReference
        ? new (std::nothrow) __fp16[static_cast<size_t>(m) * n] : nullptr;
    ctx.bias = ctx.op == Context::Operator::Gemm
        ? nullptr : new (std::nothrow) __fp16[n];
    ctx.residual = ctx.op == Context::Operator::DiffusionGeluResidual
        ? new (std::nothrow) __fp16[static_cast<size_t>(m) * n] : nullptr;
    ctx.loraDown = ctx.op == Context::Operator::LoraResidual
        ? new (std::nothrow) __fp16[static_cast<size_t>(k) * ctx.loraRank] : nullptr;
    ctx.loraUp = ctx.op == Context::Operator::LoraResidual
        ? new (std::nothrow) __fp16[static_cast<size_t>(ctx.loraRank) * n] : nullptr;
    ctx.loraTmp = ctx.op == Context::Operator::LoraResidual
        ? new (std::nothrow) float[static_cast<size_t>(m) * ctx.loraRank] : nullptr;
    if (!ctx.a || !ctx.b || !ctx.c ||
        (checkReference && !ctx.expected) ||
        (ctx.op != Context::Operator::Gemm && !ctx.bias) ||
        (ctx.op == Context::Operator::DiffusionGeluResidual && !ctx.residual) ||
        (ctx.op == Context::Operator::LoraResidual &&
         (!ctx.loraDown || !ctx.loraUp || !ctx.loraTmp))) {
        std::fprintf(stderr, "matrix allocation failed\n");
        return 2;
    }

    for (int id = 0; id < NumThreads; ++id) {
        const int mPart = id / NPartitions;
        const int mtBegin = ctx.mTiles * mPart / MPartitions;
        const int mtEnd = ctx.mTiles * (mPart + 1) / MPartitions;
        ctx.aPacked[id] = new (std::nothrow) __fp16[
            static_cast<size_t>(mtEnd - mtBegin) * TileM * kc];
        ctx.bPad[id] = new (std::nothrow) __fp16[
            static_cast<size_t>(kc) * tileN];
        if (!ctx.aPacked[id] || !ctx.bPad[id]) {
            std::fprintf(stderr, "per-core packing allocation failed\n");
            return 2;
        }
    }

    // Match the existing controlled gem5 runs: initialize zero FP16 inputs
    // functionally before the ROI, without adding initialization traffic.
    const uint64_t aBytes = aElems * sizeof(__fp16);
    const uint64_t rawBBytes = bElems * sizeof(__fp16);
    if (m5_read_file(ctx.a, aBytes, 0) != aBytes ||
        m5_read_file(ctx.b, rawBBytes, aBytes) != rawBBytes) {
        std::fprintf(stderr, "m5_read_file input initialization failed\n");
        return 2;
    }
    const uint64_t referenceBytes =
        static_cast<uint64_t>(m) * n * sizeof(__fp16);
    if (checkReference &&
        m5_read_file(ctx.expected, referenceBytes, aBytes + rawBBytes) !=
            referenceBytes) {
        std::fprintf(stderr, "m5_read_file reference initialization failed\n");
        return 2;
    }
    if (ctx.bias) {
        for (int col = 0; col < n; ++col)
            ctx.bias[col] = static_cast<__fp16>((col % 17 - 8) / 64.0f);
    }
    if (ctx.residual) {
        for (size_t i = 0; i < static_cast<size_t>(m) * n; ++i)
            ctx.residual[i] = static_cast<__fp16>((static_cast<int>(i % 13) - 6) / 128.0f);
    }
    if (ctx.loraDown) {
        for (size_t i = 0; i < static_cast<size_t>(k) * ctx.loraRank; ++i)
            ctx.loraDown[i] = static_cast<__fp16>((static_cast<int>(i % 11) - 5) / 256.0f);
        for (size_t i = 0; i < static_cast<size_t>(ctx.loraRank) * n; ++i)
            ctx.loraUp[i] = static_cast<__fp16>((static_cast<int>(i % 7) - 3) / 128.0f);
    }
    // Convert the host-computed GEMM reference into the complete fused
    // operator reference before entering the measured region.
    if (checkReference && ctx.op == Context::Operator::LoraResidual) {
        for (int row = 0; row < m; ++row) {
            for (int r = 0; r < ctx.loraRank; ++r) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; ++kk)
                    sum += static_cast<float>(ctx.a[static_cast<size_t>(row) * k + kk]) *
                           static_cast<float>(ctx.loraDown[
                               static_cast<size_t>(kk) * ctx.loraRank + r]);
                ctx.loraTmp[static_cast<size_t>(row) * ctx.loraRank + r] = sum;
            }
            for (int col = 0; col < n; ++col) {
                float update = 0.0f;
                for (int r = 0; r < ctx.loraRank; ++r)
                    update += ctx.loraTmp[static_cast<size_t>(row) * ctx.loraRank + r] *
                              static_cast<float>(ctx.loraUp[
                                  static_cast<size_t>(r) * n + col]);
                const size_t index = static_cast<size_t>(row) * n + col;
                ctx.expected[index] = static_cast<__fp16>(
                    static_cast<float>(ctx.expected[index]) + update);
            }
        }
    } else if (checkReference && ctx.op != Context::Operator::Gemm) {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                const size_t index = static_cast<size_t>(row) * n + col;
                float value = static_cast<float>(ctx.expected[index]) +
                              static_cast<float>(ctx.bias[col]);
                value = activate(value, ctx.op);
                if (ctx.residual)
                    value += static_cast<float>(ctx.residual[index]);
                ctx.expected[index] = static_cast<__fp16>(value);
            }
        }
    }

#ifdef IDENTICAL_CACHE_BASELINE
    const char *implementation = "cache_vl16_2m4n";
#else
    const char *implementation = "cacheflex_spm_vl16_2m4n";
#endif
    std::printf("%s cores=8 M=%d K=%d N=%d "
                "iterations=%d KC=%d tile_M=%d tile_N=%d operator=%s\n",
                implementation, m, k, n, iterations, kc, TileM, tileN,
                operatorName);

    pthread_t threads[NumThreads];
    ThreadArg args[NumThreads];
    args[0] = {&ctx, 0, false};
    for (int id = 1; id < NumThreads; ++id) {
        args[id] = {&ctx, id, false};
        if (pthread_create(&threads[id], nullptr, worker, &args[id]) != 0) {
            std::perror("pthread_create");
            return 2;
        }
    }
    worker(&args[0]);
    for (int id = 1; id < NumThreads; ++id)
        pthread_join(threads[id], nullptr);

#ifdef CACHEFLEX_PHASE_PROFILE
    uint64_t sums[5]{};
    uint64_t totalThreadTime = 0;
    for (int id = 0; id < NumThreads; ++id) {
        const auto &p = ctx.phase[id];
        sums[0] += p.packA; sums[1] += p.stageB; sums[2] += p.kernel;
        sums[3] += p.loopOther; sums[4] += p.finalWait;
        totalThreadTime += p.total;
        std::printf("PHASE_CORE core=%d total_ns=%llu pack_a_ns=%llu "
                    "stage_b_ns=%llu kernel_ns=%llu loop_other_ns=%llu "
                    "final_wait_ns=%llu\n", id,
                    (unsigned long long)p.total,
                    (unsigned long long)p.packA,
                    (unsigned long long)p.stageB,
                    (unsigned long long)p.kernel,
                    (unsigned long long)p.loopOther,
                    (unsigned long long)p.finalWait);
    }
    const char *phaseNames[5] = {
        "pack_a", "stage_b", "kernel", "loop_other", "final_wait"};
    for (int i = 0; i < 5; ++i)
        std::printf("PHASE_SUM name=%s thread_ns=%llu percent=%.4f\n",
                    phaseNames[i], (unsigned long long)sums[i],
                    totalThreadTime ? 100.0 * sums[i] / totalThreadTime : 0.0);
    std::printf("PHASE_SUM name=total thread_ns=%llu percent=100.0000\n",
                (unsigned long long)totalThreadTime);
#endif

    double checksum = 0.0;
    int errors = 0;
    float maxAbsError = 0.0f;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const float value = static_cast<float>(
                ctx.c[static_cast<size_t>(row) * n + col]);
            checksum += value;
            const float expected = checkReference
                ? static_cast<float>(ctx.expected[
                      static_cast<size_t>(row) * n + col])
                : 0.0f;
            const float absError = std::fabs(value - expected);
            maxAbsError = std::max(maxAbsError, absError);
            if (absError > 1.0e-3f)
                ++errors;
        }
    }
    std::printf("%s %s input=%s checksum=%.9f max_abs_error=%.9g errors=%d",
                implementation, errors == 0 ? "PASS" : "FAIL",
                checkReference ? "pattern-reference" : "zero", checksum,
                maxAbsError,
                errors);
#ifndef IDENTICAL_CACHE_BASELINE
    unsigned long long expectedClaimSpmcp = 0;
#ifdef CACHEFLEX_MULTICAST
#ifdef CACHEFLEX_MULTICAST_BOOTSTRAP
    const unsigned long long copyPanels =
        static_cast<unsigned long long>(MPartitions * NPartitions) +
        static_cast<unsigned long long>(ctx.nTiles - NPartitions);
#else
    const unsigned long long copyPanels = ctx.nTiles;
#endif
    const unsigned long long expectedSpmcp =
        static_cast<unsigned long long>(iterations) * copyPanels * k * 4;
#ifdef CACHEFLEX_MULTICAST_WAY_CLAIM
    expectedClaimSpmcp =
        static_cast<unsigned long long>(NPartitions) * ctx.kc * 4;
#endif
#else
    const unsigned long long expectedSpmcp =
        static_cast<unsigned long long>(iterations) *
        MPartitions *
        ctx.nTiles * k * 4;
#endif
    std::printf(" expected_spmcp=%llu claim_spmcp=%llu multicast=%d",
                expectedSpmcp + expectedClaimSpmcp, expectedClaimSpmcp,
#ifdef CACHEFLEX_MULTICAST
                1
#else
                0
#endif
    );
#endif
    std::printf("\n");

    delete[] ctx.a;
    std::free(ctx.b);
    delete[] ctx.c;
    delete[] ctx.expected;
    delete[] ctx.bias;
    delete[] ctx.residual;
    delete[] ctx.loraDown;
    delete[] ctx.loraUp;
    delete[] ctx.loraTmp;
    for (int id = 0; id < NumThreads; ++id) {
        delete[] ctx.aPacked[id];
        delete[] ctx.bPad[id];
    }
    return errors == 0 ? 0 : 1;
}
