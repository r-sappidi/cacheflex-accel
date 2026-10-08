#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ACL_ROOT:?set ACL_ROOT to an Arm Compute Library checkout}"
CROSS="$ROOT/tools/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
M5OP="$ROOT/gem5/util/m5/build/arm64/abi/arm64/m5op.o"

"$CROSS" -O3 -std=c++14 -march=armv8.2-a+sve+fp16+dotprod -static \
    -DARM_COMPUTE_CPP_SCHEDULER=1 -DARM_COMPUTE_ENABLE_NEON \
    -DARM_COMPUTE_ENABLE_SVE -DARM_COMPUTE_ENABLE_FP16 \
    -I"$ACL_ROOT" -I"$ACL_ROOT/include" -I"$ROOT/gem5/include" \
    "$ROOT/benchmarks/acl_full_stack_gemm/w7_fp16.cpp" \
    "$M5OP" "$ACL_ROOT/build/libarm_compute-static.a" \
    -pthread -ldl -lrt \
    -o "$ROOT/benchmarks/acl_full_stack_gemm/w7_fp16_acl"
