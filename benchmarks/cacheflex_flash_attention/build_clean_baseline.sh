#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute output directory required}"
mkdir "$out"
source "$root/setup_spm_env.sh" >/dev/null
flags=(-std=c++17 -O3 -march=armv8.2-a+sve+fp16 -mbranch-protection=none
 -DGEM5 -DGEM5_SE -DCACHEFLEX_FLASH_PRETOUCH=1
 -DCACHEFLEX_FLASH_SKIP_UNUSED_TIMERS=1 -I"$M5_INCLUDE" -static -pthread)
"$CROSS_CXX" "${flags[@]}" "$here/cache_multicore.cpp" "$M5OP_OBJ" -lm -o "$out/cache_clean"
"$CROSS_CXX" "${flags[@]}" -g "$here/cache_multicore.cpp" "$M5OP_OBJ" -lm -o "$out/cache_clean.debug"
for elf in cache_clean cache_clean.debug; do
 aarch64-linux-gnu-objcopy --dump-section .text="$out/$elf.text" "$out/$elf" /dev/null
done
cmp "$out/cache_clean.text" "$out/cache_clean.debug.text"
sha256sum "$out/cache_clean" "$out/cache_clean.debug" > "$out/binaries.sha256"
