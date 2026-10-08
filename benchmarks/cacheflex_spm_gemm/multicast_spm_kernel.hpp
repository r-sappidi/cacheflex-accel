#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#ifndef CACHEFLEX_MULTICAST_WINDOW
#define CACHEFLEX_MULTICAST_WINDOW 256
#endif

// Reserve four ways in every selected private L1 without reading B. Bit 23
// distinguishes the claim-only operation; bits 31:24 retain the core mask.
static inline void
claim_B_spm_4way_multicast(size_t kc, uint8_t coreMask)
{
    const uint64_t claimMask = (static_cast<uint64_t>(coreMask) << 24) |
                               (uint64_t{1} << 23);
    for (size_t way = 0; way < 4; ++way) {
        const size_t wayBits = way << 16;
        size_t claimRows = kc;
        __asm__ __volatile__(
            "mov x10, #0\n"
            "mov x15, %x[way]\n"
            "mov x17, %x[claim]\n"
            "1:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "orr x16, x16, x17\n"
            // The source operand is architecturally present but Ruby does
            // not access it for a claim-only SPMCP.
            "SPMCP_64_IMM x16, [xzr, #0]\n"
            "add x10, x10, #1\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 1b\n"
            : [rows] "+r"(claimRows)
            : [way] "r"(wayBits), [claim] "r"(claimMask)
            : "cc", "memory", "x10", "x15", "x16", "x17");

        // Each load waits only for the claim to the same logical slot. This
        // preserves the micro repository's address-aware shadow semantics.
        size_t drainRows = kc;
        __asm__ __volatile__(
            "mov x10, #0\n"
            "mov x15, %x[way]\n"
            "2:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "SPMLDR_8_IMM x14, [x16, #0]\n"
            "add x10, x10, #1\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 2b\n"
            : [rows] "+r"(drainRows)
            : [way] "r"(wayBits)
            : "cc", "memory", "x10", "x14", "x15", "x16");
    }
}

static inline void
release_B_spm_4way(size_t kc)
{
    for (size_t way = 0; way < 4; ++way) {
        const size_t wayBits = way << 16;
        size_t rows = kc;
        __asm__ __volatile__(
            "mov x10, #0\n"
            "mov x15, %x[way]\n"
            "1:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "SPMREL_8_IMM xzr, [x16, #0]\n"
            "add x10, x10, #1\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 1b\n"
            : [rows] "+r"(rows)
            : [way] "r"(wayBits)
            : "cc", "memory", "x10", "x15", "x16");
    }
}

// Multicast SPMCP encodes an eight-core destination mask in destination bits
// 31:24. The local slot remains in bits 18:0, preserving ordinary SPMLD
// addressing and the single-copy ABI.
static inline void
pack_B_tile_to_spm_4way_multicast(const __fp16 *bRow0, size_t nStride,
                                  size_t kc, uint8_t coreMask,
                                  size_t spmBaseSet = 0,
                                  size_t windowRows = CACHEFLEX_MULTICAST_WINDOW)
{
    const size_t strideBytes = nStride * sizeof(__fp16);
    for (size_t begin = 0; begin < kc; begin += windowRows) {
        size_t rows = std::min(windowRows, kc - begin);
        const __fp16 *src = bRow0 + begin * nStride;
        const size_t base = spmBaseSet + begin;
        const uint64_t encodedMask = static_cast<uint64_t>(coreMask) << 24;
        size_t stageRows = rows;
        __asm__ __volatile__(
            "mov x10, %x[base]\n"
            "mov x12, %x[src]\n"
            "mov x15, %x[mask]\n"
            "1:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "SPMCP_64_IMM x16, [x12, #0]\n"
            "orr x14, x16, #0x10000\n"
            "SPMCP_64_IMM x14, [x12, #64]\n"
            "orr x14, x16, #0x20000\n"
            "SPMCP_64_IMM x14, [x12, #128]\n"
            "orr x14, x16, #0x30000\n"
            "SPMCP_64_IMM x14, [x12, #192]\n"
            "add x10, x10, #1\n"
            "add x12, x12, %x[stride]\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 1b\n"
            : [rows] "+r"(stageRows)
            : [src] "r"(src), [stride] "r"(strideBytes),
              [base] "r"(base), [mask] "r"(encodedMask)
            : "cc", "memory", "x10", "x12", "x14", "x15", "x16");

        // Drain each row through the issuing core's address-aware shadow
        // table. Multicast SPMCP responds only after every remote install ack,
        // so reaching the following group barrier makes all consumers safe.
        const size_t commitSlot = ((base + rows - 1) << 6) | (3u << 16);
        __asm__ __volatile__(
            "mov x16, %x[slot]\n"
            "SPMLDR_8_IMM x14, [x16, #0]\n"
            :: [slot] "r"(commitSlot)
            : "memory", "x14", "x16");
    }
}

// Convert the four logical ways one at a time. Draining each way before
// claiming the next makes dynamic way conversion visible to Ruby and avoids
// four simultaneous claims racing in the same physical set.
static inline void
bootstrap_B_tile_to_spm_4way(const __fp16 *bRow0, size_t nStride, size_t kc)
{
    const size_t strideBytes = nStride * sizeof(__fp16);
    for (size_t way = 0; way < 4; ++way) {
        const __fp16 *src = bRow0 + way * 32;
        const size_t wayBits = way << 16;
        size_t stageRows = kc;
        __asm__ __volatile__(
            "mov x10, #0\n"
            "mov x12, %x[src]\n"
            "mov x15, %x[way]\n"
            "1:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "SPMCP_64_IMM x16, [x12, #0]\n"
            "add x10, x10, #1\n"
            "add x12, x12, %x[stride]\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 1b\n"
            : [rows] "+r"(stageRows)
            : [src] "r"(src), [stride] "r"(strideBytes),
              [way] "r"(wayBits)
            : "cc", "memory", "x10", "x12", "x15", "x16");

        size_t drainRows = kc;
        __asm__ __volatile__(
            "mov x10, #0\n"
            "mov x15, %x[way]\n"
            "2:\n"
            "lsl x16, x10, #6\n"
            "orr x16, x16, x15\n"
            "SPMLDR_8_IMM x14, [x16, #0]\n"
            "add x10, x10, #1\n"
            "subs %x[rows], %x[rows], #1\n"
            "bne 2b\n"
            : [rows] "+r"(drainRows)
            : [way] "r"(wayBits)
            : "cc", "memory", "x10", "x14", "x15", "x16");
    }
}
