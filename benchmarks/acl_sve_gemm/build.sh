#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT/setup_spm_env.sh" >/dev/null

OUTDIR="$SCRIPT_DIR/bin_gem5"
mkdir -p "$OUTDIR"

"$CROSS_CXX" \
    -std=c++17 -O3 -static -pthread -march=armv8.2-a+sve+fp16 \
    -DARM_COMPUTE_ENABLE_SVE \
    -I"$M5_INCLUDE" \
    "$SCRIPT_DIR/gemm_driver.cpp" \
    "$SCRIPT_DIR/fp16_kernel.cpp" \
    "$M5OP_OBJ" \
    -o "$OUTDIR/acl_sve_gemm"

echo "$OUTDIR/acl_sve_gemm"
