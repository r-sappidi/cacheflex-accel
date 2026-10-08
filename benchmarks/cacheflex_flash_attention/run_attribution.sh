#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute output directory required}"
[[ "$out" == /* ]]
mkdir "$out"
# Freeze the simulator for all runs; never race a subsequent rebuild.
cp "$root/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt" "$out/gem5.opt"
sha256sum "$out/gem5.opt" "$here/bin/cache_pretouch_flash_multicore" \
 "$here/bin/spm_konly_overlap_flash_multicore" \
 "$here/bin/spm_scalar_overlap_group4_diagnostic" > "$out/binaries.sha256"
run() {
 local name="$1" variant="$2" tokens="$3" enabled="$4" bin="$5"
 env -i PATH="$PATH" HOME="$HOME" CACHEFLEX_FLASH_PROFILE="$enabled" \
  CACHEFLEX_GEM5_BIN="$out/gem5.opt" CACHEFLEX_FLASH_BIN="$bin" \
  CACHEFLEX_FLASH_OUT="$out/$name" \
  bash "$here/run_8core.sh" "$variant" "$tokens" 64 96 > "$out/$name.host.log" 2>&1
}
for tokens in 128 1024; do
 pids=()
 for enabled in 0 1; do
  run "cache_t${tokens}_p${enabled}" cache_pretouch "$tokens" "$enabled" "$here/bin/cache_pretouch_flash_multicore" & pids+=("$!")
  run "spm_t${tokens}_p${enabled}" spm_konly_overlap "$tokens" "$enabled" "$here/bin/spm_konly_overlap_flash_multicore" & pids+=("$!")
 done
 if [[ "$tokens" == 1024 ]]; then
  run spm_group4_t1024_p1 spm_konly_overlap 1024 1 "$here/bin/spm_scalar_overlap_group4_diagnostic" & pids+=("$!")
 fi
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ "$failed" == 0 ]]
done
