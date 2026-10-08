#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)";root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute build directory}";mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
cp "$here/mlp_study.cpp" "$out/mlp_study.cpp"
python3 "$here/mlp_reference.py" helpers "$out/mlp_helpers.hpp"
rows="${MLP_ROWS:-24}"
if [[ "$rows" != 24 ]]; then python3 "$here/make_mlp_row_kernels.py" "$out" "$rows"; fi
printf '%s\n' "$rows" > "$out/rows.txt"
combined="${MLP_COMBINED:-0}"
acc32="${MLP_ACC32:-0}"
printf '%s\n' "$acc32" > "$out/acc32.txt"
if [[ "$acc32" == 1 ]]; then python3 "$here/make_mlp_fp32_kernel.py" "$out/mlp_fp32_kernel.hpp"; fi
printf '%s\n' "$combined" > "$out/combined.txt"
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none -DGEM5 -DGEM5_SE -DMLP_ROWS="$rows" -DMLP_COMBINED="$combined" -DMLP_ACC32="$acc32"
 -I"$out" -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/experiments/vl_length/tile_shape")
"$CROSS_CXX" "${flags[@]}" -I"$root/../cacheflex_micro/kernels/llama_bench" -DIDENTICAL_CACHE_BASELINE -DCACHEFLEX_CACHE_VL16_PIPELINED -static -pthread "$here/mlp_study.cpp" "$M5OP_OBJ" -lm -o "$out/cache"
"$CROSS_CXX" "${flags[@]}" -I"$root/../cacheflex_micro/kernels/llama_bench_spm" -DCACHEFLEX_VL16_PIPELINED -S "$here/mlp_study.cpp" -o "$out/spm.s"
python3 "$SPM_COMPILER" "$out/spm.s" "$out/spm_enc.s"
"$CROSS_CXX" -static -pthread "$out/spm_enc.s" "$M5OP_OBJ" -lm -o "$out/spm"
sha256sum "$out/cache" "$out/spm" "$out/mlp_helpers.hpp" > "$out/hashes.txt"
