/*
 * Copyright (c) 2019-2021, 2023-2026 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__aarch64__)

#include <cstddef>

namespace arm_gemm {

// Compute packed C = A * B in fixed-size tiles. Each tile has 8 rows and
// 3*SVE_VL float columns (48 columns when SVE vectors are 512 bits wide).
//
// Packing makes the data consumed at each K step contiguous:
//   Apanel: 8 values, one from each output row
//   Bpanel: 3 vectors covering the output columns
// Cpanel is written as eight rows of three vectors. The caller handles edge
// padding, so this kernel can use every SVE lane without per-lane bounds tests.
void sve_interleaved_fp32_mla_8x3VL(
    const float *Apanel,
    const float *Bpanel,
    float *Cpanel,
    int ablocks,
    int bblocks,
    int K)
{

    // Keep read-only loop values together so the assembly can use fixed
    // offsets, while its general-purpose registers remain free for pointers.
    struct KernelArgs {
        size_t K = {};
        const float *Bpanel = {};
        size_t bblocks = {};
    } ka;

    // The first K step is always executed; x20 counts the remaining steps.
    ka.K = (K/1) - 1;
    ka.Bpanel = Bpanel;
    ka.bblocks = bblocks;

    __asm__ __volatile__(
      // All lanes are valid because panels are padded to whole tiles.
      "ptrue p0.b\n"
      "cbz %x[ablocks], 6f\n"
      "1:"  // Loop over groups of eight output rows.
      "ldr x23, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x21, %x[Apanel]\n"
      "2:"  // Loop over groups of 3*SVE_VL output columns.
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x21\n"
      // z8-z31 are the 8x3 accumulator grid: three vectors per row. Keeping
      // the whole output tile in registers avoids partial C loads/stores.
      "mov z8.b, #0\n"
      "mov z9.b, #0\n"
      "mov z10.b, #0\n"
      // Start loading the three B vectors for the first K position early,
      // interleaved with accumulator setup to expose memory-level parallelism.
      "ld1w { z4.s }, p0/Z, [x22]\n"
      "mov z11.b, #0\n"
      "mov z12.b, #0\n"
      "ld1w { z5.s }, p0/Z, [x22, #1, MUL VL]\n"
      "cmp x20, #0x2\n"
      "mov z13.b, #0\n"
      "mov z14.b, #0\n"
      "mov z15.b, #0\n"
      "mov z16.b, #0\n"
      // ld1rqw broadcasts four packed A scalars as a repeating vector. z0 and
      // z1 therefore hold the eight A values needed for all output rows.
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "mov z17.b, #0\n"
      "mov z18.b, #0\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "mov z19.b, #0\n"
      "mov z20.b, #0\n"
      "ld1w { z6.s }, p0/Z, [x22, #2, MUL VL]\n"
      "mov z21.b, #0\n"
      "mov z22.b, #0\n"
      "mov z23.b, #0\n"
      "mov z24.b, #0\n"
      "mov z25.b, #0\n"
      "mov z26.b, #0\n"
      "mov z27.b, #0\n"
      "mov z28.b, #0\n"
      "mov z29.b, #0\n"
      "mov z30.b, #0\n"
      "mov z31.b, #0\n"
      "blt 4f\n"
      // Process two K positions per iteration. FMLA multiplies one B vector by
      // one selected A scalar and accumulates a full output-row vector. Loads
      // for the next half are mixed with arithmetic to hide their latency.
      "3:"  // Two-way-unrolled K loop.
      "fmla z8.s, z4.s, z0.s[0]\n"
      "fmla z11.s, z4.s, z0.s[1]\n"
      "ld1rqw { z2.s }, p0/Z, [%x[Apanel], #32]\n"
      "fmla z14.s, z4.s, z0.s[2]\n"
      "fmla z17.s, z4.s, z0.s[3]\n"
      "ld1rqw { z3.s }, p0/Z, [%x[Apanel], #48]\n"
      "fmla z20.s, z4.s, z1.s[0]\n"
      "fmla z23.s, z4.s, z1.s[1]\n"
      "sub x20, x20, #0x2\n"
      "fmla z26.s, z4.s, z1.s[2]\n"
      "fmla z29.s, z4.s, z1.s[3]\n"
      "ld1w { z4.s }, p0/Z, [x22, #3, MUL VL]\n"
      "fmla z9.s, z5.s, z0.s[0]\n"
      "fmla z12.s, z5.s, z0.s[1]\n"
      "cmp x20, #0x2\n"
      "fmla z15.s, z5.s, z0.s[2]\n"
      "fmla z18.s, z5.s, z0.s[3]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "fmla z21.s, z5.s, z1.s[0]\n"
      "fmla z24.s, z5.s, z1.s[1]\n"
      "fmla z27.s, z5.s, z1.s[2]\n"
      "fmla z30.s, z5.s, z1.s[3]\n"
      "ld1w { z5.s }, p0/Z, [x22, #4, MUL VL]\n"
      "fmla z10.s, z6.s, z0.s[0]\n"
      "fmla z13.s, z6.s, z0.s[1]\n"
      "fmla z16.s, z6.s, z0.s[2]\n"
      "fmla z19.s, z6.s, z0.s[3]\n"
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "fmla z22.s, z6.s, z1.s[0]\n"
      "fmla z25.s, z6.s, z1.s[1]\n"
      "fmla z28.s, z6.s, z1.s[2]\n"
      "fmla z31.s, z6.s, z1.s[3]\n"
      "ld1w { z6.s }, p0/Z, [x22, #5, MUL VL]\n"
      "addvl x22, x22, #6\n"
      "fmla z8.s, z4.s, z2.s[0]\n"
      "fmla z11.s, z4.s, z2.s[1]\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "fmla z14.s, z4.s, z2.s[2]\n"
      "fmla z17.s, z4.s, z2.s[3]\n"
      "fmla z20.s, z4.s, z3.s[0]\n"
      "fmla z23.s, z4.s, z3.s[1]\n"
      "fmla z26.s, z4.s, z3.s[2]\n"
      "fmla z29.s, z4.s, z3.s[3]\n"
      "ld1w { z4.s }, p0/Z, [x22]\n"
      "fmla z9.s, z5.s, z2.s[0]\n"
      "fmla z12.s, z5.s, z2.s[1]\n"
      "fmla z15.s, z5.s, z2.s[2]\n"
      "fmla z18.s, z5.s, z2.s[3]\n"
      "fmla z21.s, z5.s, z3.s[0]\n"
      "fmla z24.s, z5.s, z3.s[1]\n"
      "fmla z27.s, z5.s, z3.s[2]\n"
      "fmla z30.s, z5.s, z3.s[3]\n"
      "ld1w { z5.s }, p0/Z, [x22, #1, MUL VL]\n"
      "fmla z10.s, z6.s, z2.s[0]\n"
      "fmla z13.s, z6.s, z2.s[1]\n"
      "fmla z16.s, z6.s, z2.s[2]\n"
      "fmla z19.s, z6.s, z2.s[3]\n"
      "fmla z22.s, z6.s, z3.s[0]\n"
      "fmla z25.s, z6.s, z3.s[1]\n"
      "fmla z28.s, z6.s, z3.s[2]\n"
      "fmla z31.s, z6.s, z3.s[3]\n"
      "ld1w { z6.s }, p0/Z, [x22, #2, MUL VL]\n"
      "bge 3b\n"
      // Every call has at least one K position. Handle it here, then handle a
      // possible second remainder when the two-way loop could not consume it.
      "4:"  // K tail.
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "addvl x22, x22, #3\n"
      "fmla z8.s, z4.s, z0.s[0]\n"
      "fmla z11.s, z4.s, z0.s[1]\n"
      "fmla z14.s, z4.s, z0.s[2]\n"
      "fmla z17.s, z4.s, z0.s[3]\n"
      "fmla z20.s, z4.s, z1.s[0]\n"
      "fmla z23.s, z4.s, z1.s[1]\n"
      "fmla z26.s, z4.s, z1.s[2]\n"
      "fmla z29.s, z4.s, z1.s[3]\n"
      "fmla z9.s, z5.s, z0.s[0]\n"
      "fmla z12.s, z5.s, z0.s[1]\n"
      "fmla z15.s, z5.s, z0.s[2]\n"
      "fmla z18.s, z5.s, z0.s[3]\n"
      "fmla z21.s, z5.s, z1.s[0]\n"
      "fmla z24.s, z5.s, z1.s[1]\n"
      "fmla z27.s, z5.s, z1.s[2]\n"
      "fmla z30.s, z5.s, z1.s[3]\n"
      "fmla z10.s, z6.s, z0.s[0]\n"
      "fmla z13.s, z6.s, z0.s[1]\n"
      "fmla z16.s, z6.s, z0.s[2]\n"
      "fmla z19.s, z6.s, z0.s[3]\n"
      "fmla z22.s, z6.s, z1.s[0]\n"
      "fmla z25.s, z6.s, z1.s[1]\n"
      "fmla z28.s, z6.s, z1.s[2]\n"
      "fmla z31.s, z6.s, z1.s[3]\n"
      "cbz x20, 5f\n"
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ld1w { z7.s }, p0/Z, [x22]\n"
      "ld1w { z4.s }, p0/Z, [x22, #1, MUL VL]\n"
      "ld1w { z5.s }, p0/Z, [x22, #2, MUL VL]\n"
      "addvl x22, x22, #3\n"
      "fmla z8.s, z7.s, z0.s[0]\n"
      "fmla z11.s, z7.s, z0.s[1]\n"
      "fmla z14.s, z7.s, z0.s[2]\n"
      "fmla z17.s, z7.s, z0.s[3]\n"
      "fmla z20.s, z7.s, z1.s[0]\n"
      "fmla z23.s, z7.s, z1.s[1]\n"
      "fmla z26.s, z7.s, z1.s[2]\n"
      "fmla z29.s, z7.s, z1.s[3]\n"
      "fmla z9.s, z4.s, z0.s[0]\n"
      "fmla z12.s, z4.s, z0.s[1]\n"
      "fmla z15.s, z4.s, z0.s[2]\n"
      "fmla z18.s, z4.s, z0.s[3]\n"
      "fmla z21.s, z4.s, z1.s[0]\n"
      "fmla z24.s, z4.s, z1.s[1]\n"
      "fmla z27.s, z4.s, z1.s[2]\n"
      "fmla z30.s, z4.s, z1.s[3]\n"
      "fmla z10.s, z5.s, z0.s[0]\n"
      "fmla z13.s, z5.s, z0.s[1]\n"
      "fmla z16.s, z5.s, z0.s[2]\n"
      "fmla z19.s, z5.s, z0.s[3]\n"
      "fmla z22.s, z5.s, z1.s[0]\n"
      "fmla z25.s, z5.s, z1.s[1]\n"
      "fmla z28.s, z5.s, z1.s[2]\n"
      "fmla z31.s, z5.s, z1.s[3]\n"
      // Store the accumulator grid in row-major order. Vector-length-scaled
      // addressing keeps the same binary valid for any hardware SVE width.
      "5:"  // K reduction complete.
      "st1w { z8.s }, p0, [%x[Cpanel]]\n"
      "subs x23, x23, #0x1\n"
      "st1w { z9.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z10.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z11.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z12.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z13.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z14.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z15.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "st1w { z18.s }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "st1w { z19.s }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "st1w { z20.s }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1w { z21.s }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1w { z22.s }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1w { z23.s }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1w { z24.s }, p0, [%x[Cpanel]]\n"
      "st1w { z25.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z26.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z27.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z28.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z29.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z30.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z31.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bgt 1b\n"
      "6:"  // Exit.
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "p0", "x20", "x21", "x22", "x23", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#ifdef CACHEFLEX_SPM_GEMM
// The SPM benchmark uses a separate entry point so the checked-in ACL
// baseline remains byte-for-byte representative of the upstream kernel.
// B is laid out as three SPM ways, with set k holding the three vectors for
// reduction step k.  This kernel is intentionally specialized for 512-bit
// SVE, the width at which each FP32 vector is exactly one cache/SPM line.
void sve_interleaved_fp32_mla_8x3VL_spm(
    const float *Apanel, float *Cpanel, int K, bool accumulate, int mblocks,
    std::size_t a_block_stride, std::size_t c_block_stride)
{
    // Keep encoded slots away from SE physical pages allocated upward from
    // address zero. Ruby masks the architectural set/way fields when it maps
    // this private window into the shared cache data array.
    const unsigned long way0 = 0x60000000UL;
    const unsigned long way1 = 0x60000000UL | (1UL << 16);
    const unsigned long way2 = 0x60000000UL | (2UL << 16);

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "mov x22, %[way0]\n"
      "mov x24, %[way1]\n"
      "mov x25, %[way2]\n"
      "mov x26, %x[Apanel]\n"
      "mov x27, %x[Cpanel]\n"
      "mov x28, %x[mblocks]\n"
      "mov x17, %x[a_stride]\n"
      "mov x18, %x[c_stride]\n"
      "lsl x17, x17, #2\n"
      "lsl x18, x18, #2\n"
      "7:\n"
      "mov x22, %[way0]\n"
      "mov x24, %[way1]\n"
      "mov x25, %[way2]\n"
      "mov x20, %x[K]\n"
      "mov x21, x26\n"
      "cbnz %w[accumulate], 8f\n"
      "mov z8.b, #0\n"  "mov z9.b, #0\n"  "mov z10.b, #0\n"
      "mov z11.b, #0\n" "mov z12.b, #0\n" "mov z13.b, #0\n"
      "mov z14.b, #0\n" "mov z15.b, #0\n" "mov z16.b, #0\n"
      "mov z17.b, #0\n" "mov z18.b, #0\n" "mov z19.b, #0\n"
      "mov z20.b, #0\n" "mov z21.b, #0\n" "mov z22.b, #0\n"
      "mov z23.b, #0\n" "mov z24.b, #0\n" "mov z25.b, #0\n"
      "mov z26.b, #0\n" "mov z27.b, #0\n" "mov z28.b, #0\n"
      "mov z29.b, #0\n" "mov z30.b, #0\n" "mov z31.b, #0\n"
      "b 9f\n"
      "8:\n"
      "ld1w { z8.s }, p0/z, [x27]\n"
      "ld1w { z9.s }, p0/z, [x27, #1, mul vl]\n"
      "ld1w { z10.s }, p0/z, [x27, #2, mul vl]\n"
      "ld1w { z11.s }, p0/z, [x27, #3, mul vl]\n"
      "ld1w { z12.s }, p0/z, [x27, #4, mul vl]\n"
      "ld1w { z13.s }, p0/z, [x27, #5, mul vl]\n"
      "ld1w { z14.s }, p0/z, [x27, #6, mul vl]\n"
      "ld1w { z15.s }, p0/z, [x27, #7, mul vl]\n"
      "addvl x23, x27, #16\n"
      "ld1w { z16.s }, p0/z, [x23, #-8, mul vl]\n"
      "ld1w { z17.s }, p0/z, [x23, #-7, mul vl]\n"
      "ld1w { z18.s }, p0/z, [x23, #-6, mul vl]\n"
      "ld1w { z19.s }, p0/z, [x23, #-5, mul vl]\n"
      "ld1w { z20.s }, p0/z, [x23, #-4, mul vl]\n"
      "ld1w { z21.s }, p0/z, [x23, #-3, mul vl]\n"
      "ld1w { z22.s }, p0/z, [x23, #-2, mul vl]\n"
      "ld1w { z23.s }, p0/z, [x23, #-1, mul vl]\n"
      "ld1w { z24.s }, p0/z, [x23]\n"
      "ld1w { z25.s }, p0/z, [x23, #1, mul vl]\n"
      "ld1w { z26.s }, p0/z, [x23, #2, mul vl]\n"
      "ld1w { z27.s }, p0/z, [x23, #3, mul vl]\n"
      "ld1w { z28.s }, p0/z, [x23, #4, mul vl]\n"
      "ld1w { z29.s }, p0/z, [x23, #5, mul vl]\n"
      "ld1w { z30.s }, p0/z, [x23, #6, mul vl]\n"
      "ld1w { z31.s }, p0/z, [x23, #7, mul vl]\n"
      "9:\n"
      "10:\n"
      "spm.ld1w z4.s, p0/z, [x22]\n"
      "spm.ld1w z5.s, p0/z, [x24]\n"
      "spm.ld1w z6.s, p0/z, [x25]\n"
      "ld1rqw { z0.s }, p0/z, [x21]\n"
      "ld1rqw { z1.s }, p0/z, [x21, #16]\n"
      "fmla z8.s, z4.s, z0.s[0]\n"  "fmla z11.s, z4.s, z0.s[1]\n"
      "fmla z14.s, z4.s, z0.s[2]\n" "fmla z17.s, z4.s, z0.s[3]\n"
      "fmla z20.s, z4.s, z1.s[0]\n" "fmla z23.s, z4.s, z1.s[1]\n"
      "fmla z26.s, z4.s, z1.s[2]\n" "fmla z29.s, z4.s, z1.s[3]\n"
      "fmla z9.s, z5.s, z0.s[0]\n"  "fmla z12.s, z5.s, z0.s[1]\n"
      "fmla z15.s, z5.s, z0.s[2]\n" "fmla z18.s, z5.s, z0.s[3]\n"
      "fmla z21.s, z5.s, z1.s[0]\n" "fmla z24.s, z5.s, z1.s[1]\n"
      "fmla z27.s, z5.s, z1.s[2]\n" "fmla z30.s, z5.s, z1.s[3]\n"
      "fmla z10.s, z6.s, z0.s[0]\n" "fmla z13.s, z6.s, z0.s[1]\n"
      "fmla z16.s, z6.s, z0.s[2]\n" "fmla z19.s, z6.s, z0.s[3]\n"
      "fmla z22.s, z6.s, z1.s[0]\n" "fmla z25.s, z6.s, z1.s[1]\n"
      "fmla z28.s, z6.s, z1.s[2]\n" "fmla z31.s, z6.s, z1.s[3]\n"
      "add x21, x21, #32\n"
      "add x22, x22, #64\n"
      "add x24, x24, #64\n"
      "add x25, x25, #64\n"
      // A taken prediction on the final loop branch may issue one younger
      // SPM load.  Wrap that speculative address to valid set zero rather
      // than exposing nonexistent set 256.
      "bic x22, x22, #0x4000\n"
      "bic x24, x24, #0x4000\n"
      "bic x25, x25, #0x4000\n"
      "subs x20, x20, #1\n"
      "bne 10b\n"
      "st1w { z8.s }, p0, [x27]\n"
      "st1w { z9.s }, p0, [x27, #1, mul vl]\n"
      "st1w { z10.s }, p0, [x27, #2, mul vl]\n"
      "st1w { z11.s }, p0, [x27, #3, mul vl]\n"
      "st1w { z12.s }, p0, [x27, #4, mul vl]\n"
      "st1w { z13.s }, p0, [x27, #5, mul vl]\n"
      "st1w { z14.s }, p0, [x27, #6, mul vl]\n"
      "st1w { z15.s }, p0, [x27, #7, mul vl]\n"
      "addvl x23, x27, #16\n"
      "st1w { z16.s }, p0, [x23, #-8, mul vl]\n"
      "st1w { z17.s }, p0, [x23, #-7, mul vl]\n"
      "st1w { z18.s }, p0, [x23, #-6, mul vl]\n"
      "st1w { z19.s }, p0, [x23, #-5, mul vl]\n"
      "st1w { z20.s }, p0, [x23, #-4, mul vl]\n"
      "st1w { z21.s }, p0, [x23, #-3, mul vl]\n"
      "st1w { z22.s }, p0, [x23, #-2, mul vl]\n"
      "st1w { z23.s }, p0, [x23, #-1, mul vl]\n"
      "st1w { z24.s }, p0, [x23]\n"
      "st1w { z25.s }, p0, [x23, #1, mul vl]\n"
      "st1w { z26.s }, p0, [x23, #2, mul vl]\n"
      "st1w { z27.s }, p0, [x23, #3, mul vl]\n"
      "st1w { z28.s }, p0, [x23, #4, mul vl]\n"
      "st1w { z29.s }, p0, [x23, #5, mul vl]\n"
      "st1w { z30.s }, p0, [x23, #6, mul vl]\n"
      "st1w { z31.s }, p0, [x23, #7, mul vl]\n"
      "add x26, x26, x17\n"
      "add x27, x27, x18\n"
      "subs x28, x28, #1\n"
      "bne 7b\n"
      :
      : [Apanel] "r" (Apanel), [Cpanel] "r" (Cpanel), [K] "r" (K),
        [accumulate] "r" (accumulate), [way0] "r" (way0),
        [way1] "r" (way1), [way2] "r" (way2), [mblocks] "r" (mblocks),
        [a_stride] "r" (a_block_stride), [c_stride] "r" (c_block_stride)
      : "cc", "memory", "p0", "x17", "x18", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
        "z0", "z1", "z4", "z5", "z6", "z8", "z9", "z10", "z11",
        "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
        "z28", "z29", "z30", "z31");
}
#endif

} // namespace arm_gemm

#endif // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__aarch64__)
