#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new directory}"; placement="${2:?cache or spm}"; group="${3:-0}"; ablation="${4:-0}"
mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
k=0; if [[ "$placement" == spm ]]; then k=1; else [[ "$placement" == cache ]]; fi
python3 "$here/make_search_kernels.py" "$out/kernels.hpp" 1 0 0 0
flags=(-std=c++17 -O3 -g -march=armv8.2-a+sve+fp16 -mbranch-protection=none
 -DGEM5 -DGEM5_SE -DVL_16 -DCACHEFLEX_FLASH_VL16=1 -DCACHEFLEX_FLASH_SEARCH=1
 -DCACHEFLEX_FLASH_MATCHED=1 -DCACHEFLEX_FLASH_PRETOUCH=1 -DCACHEFLEX_FLASH_DRIVER_ABLATION="$ablation"
 -DCACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1 -DCACHEFLEX_FLASH_STAGE_K="$k"
 -DCACHEFLEX_FLASH_STAGE_V=0 -DCACHEFLEX_FLASH_IB_GROUP="$group" -DCACHEFLEX_FLASH_K_OVERLAP=1
 "-DCACHEFLEX_FLASH_SEARCH_KERNELS=\"$out/kernels.hpp\""
 -I"$M5_INCLUDE" -I"$root/../cacheflex_micro/kernels/llama_bench_spm"
 -I"$root/../cacheflex_micro/kernels/attention_kernels_spm")
"$CROSS_CXX" "${flags[@]}" -c "$here/cache_qk_wrapper.cpp" -o "$out/qk.o"
for markers in 0 1; do
 "$CROSS_CXX" "${flags[@]}" -DCACHEFLEX_FLASH_BOUNDARIES="$markers" -S "$here/spm_multicore.cpp" -o "$out/$markers.s"
 python3 "$SPM_COMPILER" "$out/$markers.s" "$out/$markers.enc.s"
 "$CROSS_CXX" -static -pthread "$out/$markers.enc.s" "$out/qk.o" "$M5OP_OBJ" -lm -o "$out/$markers.debug"
 aarch64-linux-gnu-objcopy --dump-section .text="$out/$markers.text" "$out/$markers.debug" "$out/$markers.copy"
done
# Both observer-off and observer-on simulations use this SAME labeled guest.
# Empty versus labeled inline asm can influence GCC's optimization, even
# though labels encode no instructions; do not claim those builds match.
cp "$out/1.debug" "$out/guest.debug"
aarch64-linux-gnu-objcopy --strip-debug "$out/guest.debug" "$out/guest"
cp "$here/spm_multicore.cpp" "$out/spm_multicore.cpp"
