#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
source "$root/setup_spm_env.sh" >/dev/null
group="${1:-4}"
name="$here/bin/spm_scalar_overlap_group${group}_diagnostic"
"$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 \
 -mbranch-protection=none -DGEM5 -DGEM5_SE -DVL_4 \
 -DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_STAGE_V=0 \
 -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_IB_GROUP="$group" \
 -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/kernels/llama_bench_spm" \
 -I"$root/../cacheflex_micro/kernels/attention_kernels_spm" \
 -S "$here/spm_multicore.cpp" -o "$name.s"
python3 "$SPM_COMPILER" "$name.s" "$name.enc.s"
"$CROSS_CXX" -std=c++17 -O3 -march=armv8.2-a+sve+fp16 \
 -mbranch-protection=none -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" \
 -c "$here/cache_qk_wrapper.cpp" -o "$name.wrapper.o"
"$CROSS_CXX" -static -pthread "$name.enc.s" "$name.wrapper.o" "$M5OP_OBJ" -lm -o "$name"
