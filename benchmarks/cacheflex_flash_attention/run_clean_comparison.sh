#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute output directory}"
clean="${2:?clean baseline ELF}"
mkdir "$out"
mkdir "$out/guests"
cp "$root/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt" "$out/gem5.opt"
cp "$clean" "$out/guests/cache"
cp "$here/bin/cache_pretouch_flash_multicore" "$out/guests/cache_legacy"
cp "$here/bin/spm_konly_overlap_flash_multicore" "$out/guests/spm"
cp "$here/bin/spm_scalar_overlap_group4_diagnostic" "$out/guests/spm_group4"
sha256sum "$out/gem5.opt" "$out"/guests/* > "$out/binaries.sha256"
run() {
 local name="$1" variant="$2" tokens="$3" profile="$4" lru="$5" guest="$6"
 env -i PATH="$PATH" HOME="$HOME" CACHEFLEX_FLASH_PROFILE="$profile" \
 CACHEFLEX_FLASH_MATCHED_LRU="$lru" CACHEFLEX_GEM5_BIN="$out/gem5.opt" \
 CACHEFLEX_FLASH_BIN="$out/guests/$guest" CACHEFLEX_FLASH_OUT="$out/$name" \
 bash "$here/run_8core.sh" "$variant" "$tokens" 64 96 > "$out/$name.host.log" 2>&1
}
for t in 128 1024; do
 pids=()
 for profile in 0 1; do
  run "cache_t${t}_p${profile}" cache_pretouch "$t" "$profile" 1 cache & pids+=("$!")
  run "spm_t${t}_p${profile}" spm_konly_overlap "$t" "$profile" 1 spm & pids+=("$!")
 done
 if [[ "$t" == 1024 ]]; then
  run spm_group4_t1024_p1 spm_konly_overlap 1024 1 1 spm_group4 & pids+=("$!")
  run legacy_lru_t1024 cache_pretouch 1024 1 1 cache_legacy & pids+=("$!")
  run legacy_tree_t1024 cache_pretouch 1024 1 0 cache_legacy & pids+=("$!")
  run clean_tree_t1024 cache_pretouch 1024 1 0 cache & pids+=("$!")
 fi
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ "$failed" == 0 ]]
done
