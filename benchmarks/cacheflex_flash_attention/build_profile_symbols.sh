#!/usr/bin/env bash
# Debug-only side ELFs; simulation must execute the original production ELFs.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new output directory required}"
mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
flags=(-std=c++17 -O3 -g -march=armv8.2-a+sve+fp16 -mbranch-protection=none
       -DGEM5 -DGEM5_SE -DCACHEFLEX_FLASH_PRETOUCH=1 -I"$M5_INCLUDE")
"$CROSS_CXX" "${flags[@]}" -static -pthread "$here/cache_multicore.cpp" "$M5OP_OBJ" -lm -o "$out/cache.debug"
for group in 0 4; do
 name="$out/spm_group$group.debug"
 "$CROSS_CXX" "${flags[@]}" -DVL_4 -DCACHEFLEX_FLASH_STAGE_V=0 \
  -DCACHEFLEX_FLASH_K_OVERLAP=1 -DCACHEFLEX_FLASH_IB_GROUP="$group" \
  -I"$root/../cacheflex_micro/kernels/llama_bench_spm" \
  -I"$root/../cacheflex_micro/kernels/attention_kernels_spm" \
  -S "$here/spm_multicore.cpp" -o "$name.s"
 python3 "$SPM_COMPILER" "$name.s" "$name.enc.s"
 "$CROSS_CXX" "${flags[@]}" -c "$here/cache_qk_wrapper.cpp" -o "$name.wrapper.o"
 "$CROSS_CXX" -static -pthread "$name.enc.s" "$name.wrapper.o" "$M5OP_OBJ" -lm -o "$name"
done
for pair in 'cache cache_pretouch_flash_multicore' 'spm_group0 spm_konly_overlap_flash_multicore' 'spm_group4 spm_scalar_overlap_group4_diagnostic'; do
 read -r side original <<< "$pair"
 aarch64-linux-gnu-objcopy --dump-section .text="$out/$side.production.text" "$here/bin/$original" /dev/null
 aarch64-linux-gnu-objcopy --dump-section .text="$out/$side.debug.text" "$out/$side.debug" /dev/null
 cmp "$out/$side.production.text" "$out/$side.debug.text"
 sha256sum "$out/$side.production.text" "$out/$side.debug.text"
done
