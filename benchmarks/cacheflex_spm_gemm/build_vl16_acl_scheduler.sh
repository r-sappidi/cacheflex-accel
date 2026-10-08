#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MICRO_ROOT="$(cd "$ROOT/../cacheflex_micro" && pwd)"
: "${ACL_ROOT:=/data2/rsappidi/cacheflex_mc_eval_cleanup_20260817/third_party/ComputeLibrary}"
ACL_LIB="$ACL_ROOT/build/build_sve/libarm_compute-static.a"
source "$ROOT/setup_spm_env.sh" >/dev/null
mkdir -p "$SCRIPT_DIR/bin_gem5"

common=(
    -std=c++17 -O3 -march=armv8.2-a+sve+fp16+dotprod
    -mbranch-protection=none -DARM_COMPUTE_CPP_SCHEDULER=1
    -DARM_COMPUTE_ENABLE_NEON -DARM_COMPUTE_ENABLE_SVE
    -DARM_COMPUTE_ENABLE_FP16 -DGEM5 -DGEM5_SE
    -I"$M5_INCLUDE" -I"$ACL_ROOT" -I"$ACL_ROOT/include"
    -I"$MICRO_ROOT/experiments/vl_length/tile_shape"
)

"$CROSS_CXX" "${common[@]}" -I"$MICRO_ROOT/kernels/llama_bench" \
    -DIDENTICAL_CACHE_BASELINE -static -pthread \
    -o "$SCRIPT_DIR/bin_gem5/cache_vl16_acl_scheduler" \
    "$SCRIPT_DIR/vl16_acl_scheduler_driver.cpp" "$M5OP_OBJ" "$ACL_LIB" \
    -ldl -lrt -lm

asm="$SCRIPT_DIR/bin_gem5/spm_vl16_acl_scheduler.s"
enc="$SCRIPT_DIR/bin_gem5/spm_vl16_acl_scheduler_enc.s"
obj="$SCRIPT_DIR/bin_gem5/spm_vl16_acl_scheduler.o"
"$CROSS_CXX" "${common[@]}" -I"$MICRO_ROOT/kernels/llama_bench_spm" \
    -DACL_SPM_VARIANT -S -o "$asm" \
    "$SCRIPT_DIR/vl16_acl_scheduler_driver.cpp"
python3 "$SPM_COMPILER" "$asm" "$enc"
"$CROSS_CXX" -c -o "$obj" "$enc"
"$CROSS_CXX" -static -pthread \
    -o "$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_acl_scheduler" \
    "$obj" "$M5OP_OBJ" "$ACL_LIB" -ldl -lrt -lm

echo "$SCRIPT_DIR/bin_gem5/cache_vl16_acl_scheduler"
echo "$SCRIPT_DIR/bin_gem5/cacheflex_spm_vl16_acl_scheduler"
