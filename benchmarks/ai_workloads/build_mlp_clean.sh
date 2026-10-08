#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)";root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute build directory}";mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
python3 "$here/mlp_reference.py" helpers "$out/mlp_helpers.hpp"
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE
 -I"$out" -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/experiments/vl_length/tile_shape")
"$CROSS_CXX" "${flags[@]}" -I"$root/../cacheflex_micro/kernels/llama_bench" -DIDENTICAL_CACHE_BASELINE -DCACHEFLEX_CACHE_VL16_PIPELINED -static -pthread "$here/mlp_clean.cpp" "$M5OP_OBJ" -lm -o "$out/cache"
"$CROSS_CXX" "${flags[@]}" -I"$root/../cacheflex_micro/kernels/llama_bench_spm" -DCACHEFLEX_VL16_PIPELINED -S "$here/mlp_clean.cpp" -o "$out/spm.s"
python3 "$SPM_COMPILER" "$out/spm.s" "$out/spm_enc.s"
"$CROSS_CXX" -static -pthread "$out/spm_enc.s" "$M5OP_OBJ" -lm -o "$out/spm"
sha256sum "$out/cache" "$out/spm" "$out/mlp_helpers.hpp" > "$out/hashes.txt"
