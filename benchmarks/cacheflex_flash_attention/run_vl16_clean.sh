#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${1:?new absolute result directory}"
build="${2:?VL16 build directory}"
mkdir "$out"
cp "$root/gem5/build/ARM_MESI_Three_Level_SPM/gem5.opt" "$out/gem5.opt"
mkdir "$out/guests"
cp "$build/cache" "$build/spm" "$out/guests/"
sha256sum "$out/gem5.opt" "$out"/guests/* > "$out/binaries.sha256"
for t in 128 1024; do
 pids=()
 for variant in cache spm; do
  ways=0; mode=cache_pretouch
  if [[ "$variant" == spm ]]; then ways=4; mode=spm_konly_overlap; fi
  for profile in 0 1; do
   env -i PATH="$PATH" HOME="$HOME" CACHEFLEX_FLASH_VL=16 \
    CACHEFLEX_FLASH_MATCHED_LRU=1 CACHEFLEX_FLASH_SPM_WAYS="$ways" \
    CACHEFLEX_FLASH_PROFILE="$profile" CACHEFLEX_FLASH_BIN="$out/guests/$variant" \
    CACHEFLEX_GEM5_BIN="$out/gem5.opt" CACHEFLEX_FLASH_OUT="$out/${variant}_t${t}_p${profile}" \
    bash "$here/run_8core.sh" "$mode" "$t" 64 128 \
    > "$out/${variant}_t${t}_p${profile}.host.log" 2>&1 & pids+=("$!")
  done
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 [[ "$failed" == 0 ]]
 python3 "$here/validate_vl16_clean.py" "$out" "$t" > "$out/validation_t${t}.json"
done
