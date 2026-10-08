#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute output directory}"
mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none
 -DGEM5 -DGEM5_SE -DVL_16 -DCACHEFLEX_FLASH_VL16=1
 -DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1
 -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/kernels/llama_bench_spm"
 -I"$root/../cacheflex_micro/kernels/attention_kernels_spm")
"$CROSS_CXX" "${flags[@]}" -static -pthread "$here/cache_multicore.cpp" "$M5OP_OBJ" -lm -o "$out/cache"
"$CROSS_CXX" "${flags[@]}" -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_K_OVERLAP=1 \
 -S "$here/spm_multicore.cpp" -o "$out/spm.s"
python3 "$SPM_COMPILER" "$out/spm.s" "$out/spm_enc.s"
"$CROSS_CXX" -static -pthread "$out/spm_enc.s" "$M5OP_OBJ" -lm -o "$out/spm"
sha256sum "$out/cache" "$out/spm" > "$out/binaries.sha256"
