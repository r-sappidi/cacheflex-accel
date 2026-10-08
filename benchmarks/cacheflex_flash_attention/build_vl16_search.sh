#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute directory}"; placement="${2:?cache|k|v|kv}"
group="${3:?0=all or positive group}"; overlap="${4:?0|1}"
unroll="${5:-1}"; order="${6:-0}"
rows="${7:-0}"; pipeline="${8:-0}"; fusion="${9:-0}"; rich="${10:-0}"
mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
k=0; v=0
case "$placement" in cache) ;; k) k=1;; v) v=1;; kv) k=1;v=1;; *) exit 2;; esac
python3 "$here/make_search_kernels.py" "$out/kernels.hpp" "$unroll" "$order" "$rows" "$pipeline"
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none
 -DGEM5 -DGEM5_SE -DVL_16 -DCACHEFLEX_FLASH_VL16=1 -DCACHEFLEX_FLASH_SEARCH=1
 -DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1
 -DCACHEFLEX_FLASH_STAGE_K="$k" -DCACHEFLEX_FLASH_STAGE_V="$v"
 -DCACHEFLEX_FLASH_IB_GROUP="$group" -DCACHEFLEX_FLASH_K_OVERLAP="$overlap"
 -DCACHEFLEX_FLASH_WIDE_ROWS="$rows" -DCACHEFLEX_FLASH_FUSE_RESCALE="$fusion"
 -DCACHEFLEX_FLASH_RICH_INPUT="$rich"
 "-DCACHEFLEX_FLASH_SEARCH_KERNELS=\"$out/kernels.hpp\""
 -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/kernels/llama_bench_spm"
 -I"$root/../cacheflex_micro/kernels/attention_kernels_spm")
"$CROSS_CXX" "${flags[@]}" -S "$here/spm_multicore.cpp" -o "$out/kernel.s"
python3 "$SPM_COMPILER" "$out/kernel.s" "$out/kernel_enc.s"
"$CROSS_CXX" "${flags[@]}" -c "$here/cache_qk_wrapper.cpp" -o "$out/qk.o"
"$CROSS_CXX" -static -pthread "$out/kernel_enc.s" "$out/qk.o" "$M5OP_OBJ" -lm -o "$out/guest"
sha256sum "$out/guest" "$out/kernels.hpp" > "$out/hashes.txt"
