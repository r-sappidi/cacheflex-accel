#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute directory}"; mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
python3 "$here/make_search_kernels.py" "$out/kernels.hpp" 1 0 0 0
for spm in 0 1; do
 "$CROSS_CXX" -O3 -g -std=c++17 -march=armv8.2-a+sve+fp16 -DGEM5 -DGEM5_SE -I"$M5_INCLUDE" \
  -DRESIDENT_SPM="$spm" "-DCACHEFLEX_FLASH_SEARCH_KERNELS=\"$out/kernels.hpp\"" \
  -S "$here/resident_qk.cpp" -o "$out/$spm.s"
 python3 "$SPM_COMPILER" "$out/$spm.s" "$out/$spm.enc.s"
 "$CROSS_CXX" -static -pthread "$out/$spm.enc.s" "$M5OP_OBJ" -lm -o "$out/$spm"
done
